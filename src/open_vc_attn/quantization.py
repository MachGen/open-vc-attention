"""Attention-only FP8 quantizers. BSD-3-Clause; MachGen contributors."""


import torch
from triton import jit, next_power_of_2
from triton import language as tl

_FP8_E4M3_MAX = 448.0


@jit
def _quant_cast(vals, QMAX: tl.constexpr, IS_INT8: tl.constexpr):
    """Apply scale-then-cast tail shared by per-head and per-block kernels.
    For int8: round-half-away-from-zero matches SageAttention's CUDA reference.
    """
    if IS_INT8:
        vals = vals + 0.5 * tl.where(vals >= 0, 1.0, -1.0)
        return tl.clamp(vals, -128.0, 127.0)
    return tl.clamp(vals, -QMAX, QMAX)


@jit
def _per_head_amax_kernel(
    X,
    AMAX,
    S,
    H,
    D,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute per-head |x|max via atomic reduction.

    Grid (B, num_blocks_S, H): each CTA reduces over a (BLOCK_L, D) tile and
    atomicMax-es its local amax into AMAX[B, H]. Full SM occupancy on B200.
    """
    b = tl.program_id(0)
    blk = tl.program_id(1)
    h = tl.program_id(2)

    base = b * S * H * D + h * D
    row_stride = H * D

    s_off = blk * BLOCK_L + tl.arange(0, BLOCK_L)
    s_mask = s_off < S
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    ptrs = X + base + s_off[:, None] * row_stride + d_off[None, :]
    mask = s_mask[:, None] & d_mask[None, :]

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    local_amax = tl.max(tl.abs(vals))
    tl.atomic_max(AMAX + b * H + h, local_amax)


@jit
def _per_head_scale_cast_kernel(
    X,
    OUT,
    AMAX,
    S,
    H,
    D,
    IS_INT8: tl.constexpr,
    QMAX: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scale + cast pass using the global per-head amax from kernel 1.

    Same grid layout as _per_head_amax_kernel; each CTA reads its tile, scales
    by QMAX/amax[b,h], and stores cast output.
    """
    b = tl.program_id(0)
    blk = tl.program_id(1)
    h = tl.program_id(2)

    base = b * S * H * D + h * D
    row_stride = H * D

    s_off = blk * BLOCK_L + tl.arange(0, BLOCK_L)
    s_mask = s_off < S
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    ptrs = X + base + s_off[:, None] * row_stride + d_off[None, :]
    out_ptrs = OUT + base + s_off[:, None] * row_stride + d_off[None, :]
    mask = s_mask[:, None] & d_mask[None, :]

    amax = tl.load(AMAX + b * H + h)
    scale = QMAX / tl.maximum(amax, 1e-12)
    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32) * scale
    vals = _quant_cast(vals, QMAX, IS_INT8)
    tl.store(out_ptrs, vals.to(OUT.dtype.element_ty), mask=mask)


def _per_head_quant_triton(x, qmax, out_dtype, is_int8):
    """Per-head quantization: scale-per-(batch, head), int8 or fp8 output.

    Two-pass design with full SM occupancy:
      pass 1 — many CTAs compute (BLOCK_L, D) tile amax, atomicMax into AMAX[B,H]
      pass 2 — same grid; each CTA reads its tile + AMAX[B,H], scales + casts.
    Replaces the old single-kernel (B*H,) grid which under-utilized the GPU
    (e.g. only 20 CTAs at B=2 H=10 → ~12× slower than per-block).
    """
    if x.dim() == 3:
        x = x[None, ...]
    x = x.contiguous()
    b, s, h, d = x.shape

    BLOCK_L = 128
    BLOCK_D = next_power_of_2(d)
    num_blocks = (s + BLOCK_L - 1) // BLOCK_L

    out = torch.empty_like(x, dtype=out_dtype)
    # amax buffer zeroed so atomic_max accumulates correctly.
    amax = torch.zeros((b, h), dtype=torch.float32, device=x.device)
    descale = torch.empty((b, h), dtype=torch.float32, device=x.device)

    grid = (b, num_blocks, h)
    _per_head_amax_kernel[grid](
        x,
        amax,
        s,
        h,
        d,
        BLOCK_L=BLOCK_L,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    _per_head_scale_cast_kernel[grid](
        x,
        out,
        amax,
        s,
        h,
        d,
        IS_INT8=is_int8,
        QMAX=qmax,
        BLOCK_L=BLOCK_L,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    # descale = amax / QMAX, computed on the global amax tensor (tiny torch op).
    torch.div(amax.clamp_(min=1e-12), qmax, out=descale)
    return out, descale


def fp8_quantize_per_head_triton(x):
    """FP8 e4m3 per-head quant. (B,S,H,D) → (x_fp8, descale[B,H] fp32)."""
    return _per_head_quant_triton(x, _FP8_E4M3_MAX, torch.float8_e4m3fn, is_int8=False)


@jit
def _per_block_quant_kernel(
    X,
    OUT,
    SCALES,
    S,
    H,
    D,
    NUM_BLOCKS,
    IS_INT8: tl.constexpr,
    QMAX: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    blk = tl.program_id(1)
    h = tl.program_id(2)

    base = b * S * H * D + h * D
    row_stride = H * D

    s_off = blk * BLOCK_L + tl.arange(0, BLOCK_L)
    s_mask = s_off < S
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    ptrs = X + base + s_off[:, None] * row_stride + d_off[None, :]
    out_ptrs = OUT + base + s_off[:, None] * row_stride + d_off[None, :]
    mask = s_mask[:, None] & d_mask[None, :]

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(vals)), 1e-12)
    scale = QMAX / amax
    tl.store(SCALES + b * H * NUM_BLOCKS + h * NUM_BLOCKS + blk, amax / QMAX)

    vals = _quant_cast(vals * scale, QMAX, IS_INT8)
    tl.store(out_ptrs, vals.to(OUT.dtype.element_ty), mask=mask)


def _per_block_quant_triton(x, block_l, qmax, out_dtype, is_int8):
    if x.dim() == 3:
        x = x[None, ...]
    x = x.contiguous()
    b, s, h, d = x.shape
    num_blocks = (s + block_l - 1) // block_l
    out = torch.empty_like(x, dtype=out_dtype)
    descale = torch.empty((b, h, num_blocks), dtype=torch.float32, device=x.device)
    _per_block_quant_kernel[(b, num_blocks, h)](
        x,
        out,
        descale,
        s,
        h,
        d,
        num_blocks,
        IS_INT8=is_int8,
        QMAX=qmax,
        BLOCK_L=block_l,
        BLOCK_D=next_power_of_2(d),
        num_warps=4,
    )
    return out, descale


def fp8_quantize_per_block_triton(x, block_l=128):
    """FP8 e4m3 per-block quant. descale shape (B, H, num_blocks) fp32."""
    return _per_block_quant_triton(
        x, block_l, _FP8_E4M3_MAX, torch.float8_e4m3fn, is_int8=False
    )
