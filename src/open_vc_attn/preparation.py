"""Default preparation fusion for eligible dense, single-sequence B200 attention.

The quantization groups and FP32 descales match quantization.py. V still uses
an ordered global amax pass; its second pass writes the final packed layout.
"""

import torch
import triton
import triton.language as tl

from .quantization import _per_head_amax_kernel


@triton.jit
def _quantize_qk(
    Q,
    K,
    Q8,
    K8,
    QS,
    KS,
    AMAX,
    S,
    H,
    BLOCKS: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BLOCK_L: tl.constexpr = 128,
    D: tl.constexpr = 128,
):
    block, head, which = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    src = Q if which == 0 else K
    dst = Q8 if which == 0 else K8
    scales = QS if which == 0 else KS
    scale_stride = Q_BLOCKS if which == 0 else BLOCKS
    # V amax is consumed only by the next stream-ordered kernel.
    # Exactly one Q CTA initializes each head, eliminating a separate fill.
    if which == 0 and block == 0:
        tl.store(AMAX + head, 0.0)
    rows = block * BLOCK_L + tl.arange(0, BLOCK_L)
    cols = tl.arange(0, D)
    offsets = rows.to(tl.int64)[:, None] * H * D + head * D + cols[None, :]
    values = tl.load(src + offsets, rows[:, None] < S, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values)), 1e-12)
    tl.store(scales + head * scale_stride + block, amax / 448.0)
    values = tl.clamp(values * (448.0 / amax), -448.0, 448.0)
    tl.store(dst + offsets, values.to(dst.dtype.element_ty), rows[:, None] < S)
    if Q_BLOCKS > BLOCKS:
        if which == 0 and block == 0:
            tl.store(QS + head * Q_BLOCKS + BLOCKS, 1.0)


@triton.jit
def _cast_pack_v(
    V,
    PACKED,
    AMAX,
    VS,
    S,
    H,
    PADDED: tl.constexpr,
    ROWS: tl.constexpr = 64,
    D: tl.constexpr = 128,
):
    block, head = tl.program_id(0), tl.program_id(1)
    rows = block * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    amax = tl.maximum(tl.load(AMAX + head), 1e-12)
    values = tl.load(
        V + rows.to(tl.int64)[:, None] * H * D + head * D + cols[None, :],
        rows[:, None] < S,
        other=0.0,
    ).to(tl.float32)
    values = tl.clamp(values * (448.0 / amax), -448.0, 448.0)
    tl.store(
        PACKED + head.to(tl.int64) * D * PADDED + cols[None, :] * PADDED + rows[:, None],
        values.to(PACKED.dtype.element_ty),
        rows[:, None] < PADDED,
    )
    if block == 0:
        tl.store(VS + head, amax / 448.0)


def prepare_fp8_fused(q, k, v, *, cu_seqlens=None):
    """Prepare contiguous [S,H,128] or [1,S,H,128] self-attention for B200.

    Requires S >= 32768 and the packed dense ExpCast path. Pass the pipeline's
    existing int32 CUDA [0,S] metadata to avoid allocating/copying it here.
    Metadata values are the caller's contract, as in raw_forward. The general
    prepare_fp8 API remains suitable for reference, causal and LSE calls.
    """
    from .api import PreparedFP8, _packed_dsl_available, validate_qkv

    validate_qkv(q, k, v)
    if (q.ndim == 4 and q.shape[0] != 1) or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Fused preparation requires single-sequence [S,H,128] self-attention")
    if q.shape[-3] < 32768 or any(not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("Fused preparation requires contiguous Q/K/V and S >= 32768")
    if torch.cuda.get_device_capability(q.device) != (10, 0):
        raise ValueError("Fused preparation is currently validated for B200 only")
    if not _packed_dsl_available():
        raise ValueError("Fused preparation requires nvidia-cutlass-dsl 4.6.2")
    shape = tuple(q.shape)
    q, k, v = [t.reshape(-1, t.shape[-2], t.shape[-1]) for t in (q, k, v)]
    s, h, d = q.shape
    if cu_seqlens is not None and (
        cu_seqlens.shape != (2,)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != q.device
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be contiguous int32 [0,S] on the QKV device")
    with torch.cuda.device(q.device):
        if cu_seqlens is None:
            cu_seqlens = torch.arange(2, device=q.device, dtype=torch.int32) * s
        blocks = triton.cdiv(s, 128)
        q_blocks = triton.cdiv(blocks, 2) * 2
        padded = blocks * 128
        q8, k8 = [torch.empty_like(q, dtype=torch.float8_e4m3fn) for _ in range(2)]
        qs = torch.empty((1, h, q_blocks), dtype=torch.float32, device=q.device)
        ks = torch.empty((1, h, blocks), dtype=torch.float32, device=q.device)
        packed = torch.empty((h, d, padded), dtype=torch.float8_e4m3fn, device=q.device)
        amax = torch.empty((1, h), dtype=torch.float32, device=q.device)
        vs = torch.empty((1, h), dtype=torch.float32, device=q.device)
        _quantize_qk[(blocks, h, 2)](
            q,
            k,
            q8,
            k8,
            qs,
            ks,
            amax,
            s,
            h,
            blocks,
            q_blocks,
            num_warps=4,
        )
        _per_head_amax_kernel[(1, blocks, h)](
            v,
            amax,
            s,
            h,
            d,
            BLOCK_L=128,
            BLOCK_D=128,
            num_warps=4,
        )
        _cast_pack_v[(triton.cdiv(padded, 64), h)](
            v,
            packed,
            amax,
            vs,
            s,
            h,
            padded,
            num_warps=4,
        )
        result = packed[..., :s].movedim(-1, 0)
        layout = dict(
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=s, max_seqlen_k=s
        )
        return PreparedFP8(q8, k8, result, qs, ks, vs, layout, shape, v_prepacked=True)
