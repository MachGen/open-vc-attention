"""V repair tokens on the packed ExpCast path.

Ordinary FP8 rounds V once per element. For the tokens whose V rounds worst
(largest sum of squared residuals, per head), this appends a repair entry: a
copy of the token's FP8 key and its V residual, also in FP8. The repairs form a
K/V prefix that both the default and the mid-window scans visit after the
original tokens; they add to the
output but are removed from the softmax denominator. This approximates the
missing V contribution; it is not a lossless correction.

Residuals use V's per-head scale, so repair and original contributions share
units. Each repair uses its original key and per-column K descale. Floating
point evaluation order can differ from the original score. Remaining differences: Q/K
quantization, ExpCast probabilities, FP8 rounding of the residual itself, and
the repair's ExpCast code being formed against the final row maximum.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ._dispatch import DEFAULT_MID_WINDOW_BLOCKS, DEFAULT_VERSION
from .api import _layout, _packed_dsl_available, prepare_fp8_fused, raw_forward, validate_qkv
from .quantization import (
    _FP8_E4M3_MAX,
    _per_block_quant_kernel,
    _per_head_amax_kernel,
    _quant_cast,
    next_power_of_2,
)


@dataclass(frozen=True)
class PreparedVRepair:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_descale: torch.Tensor
    k_descale: torch.Tensor
    v_descale: torch.Tensor
    repair_k_descale: torch.Tensor | None
    layout: dict
    output_shape: tuple
    repair_tokens: int
    selected_tokens: int = 0
    v_prepacked: bool = False


@triton.jit
def _v_quant_score_kernel(
    X,
    OUT,
    AMAX,
    SCORE,
    length,
    heads,
    QMAX: tl.constexpr,
    BLOCK_L: tl.constexpr,
    D: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    rows = block * BLOCK_L + tl.arange(0, BLOCK_L)
    cols = tl.arange(0, D)
    mask = rows[:, None] < length
    offsets = rows.to(tl.int64)[:, None] * (heads * D) + head * D + cols[None, :]
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    descale = tl.maximum(tl.load(AMAX + head), 1e-12) / QMAX
    codes = _quant_cast(x * (QMAX / tl.maximum(tl.load(AMAX + head), 1e-12)), QMAX, False)
    stored = codes.to(OUT.dtype.element_ty)
    tl.store(OUT + offsets, stored, mask=mask)
    residual = x - stored.to(tl.float32) * descale
    tl.store(SCORE + head * length + rows, tl.sum(residual * residual, axis=1), mask=rows < length)


@triton.jit
def _repair_rows_kernel(
    K,
    V,
    VSRC,
    KDESCALE,
    VDESCALE,
    INDEX,
    REPAIR_KD,
    selected,
    rows,
    length,
    heads,
    k_blocks,
    QMAX: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    real = row < selected
    source = tl.load(INDEX + head * selected + tl.where(real, row, 0)).to(tl.int64)
    cols = tl.arange(0, D)
    target = row.to(tl.int64) * (heads * D) + head * D + cols
    origin = (rows + source) * (heads * D) + head * D + cols
    tl.store(K + target, tl.load(K + origin))
    tl.store(
        REPAIR_KD + head * rows + row,
        tl.load(KDESCALE + head * k_blocks + source // 128),
    )
    descale = tl.load(VDESCALE + head)
    original = tl.load(VSRC + source * (heads * D) + head * D + cols).to(tl.float32)
    rounded = tl.load(V + origin).to(tl.float32) * descale
    residual = tl.where(real, original - rounded, 0.0)
    codes = tl.clamp(residual / descale, -QMAX, QMAX)
    tl.store(V + target, codes.to(V.dtype.element_ty))


@torch.no_grad()
def prepare_v_repair(q, k, v, *, budget):
    """Prepare BF16/FP16 [S,H,128] or [1,S,H,128] with a V repair budget.

    budget is the fraction of tokens repaired per head (0.02 = worst 2%). Q/K use
    per-128-token-block scales and V a per-head scale. B200, S >= 32768 and
    DSL 4.6.2 are required. Zero selected tokens use fused preparation without
    repair overhead. Positive budgets use residual scoring and top-k selection.
    """
    validate_qkv(q, k, v)
    if q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError("V repair expects equal self-attention shapes ending in D=128")
    if not 0.0 <= budget < 1.0:
        raise ValueError("budget must be in [0, 1)")
    if q.shape[-3] < 32768 or (q.ndim == 4 and q.shape[0] != 1):
        raise ValueError("V repair requires one sequence with S >= 32768")
    if torch.cuda.get_device_capability(q.device) != (10, 0) or not _packed_dsl_available():
        raise ValueError("V repair currently requires B200 and nvidia-cutlass-dsl 4.6.2")
    with torch.cuda.device(q.device):
        q, k, v, layout, shape = _layout(q, k, v)
        if q.ndim != 3:
            raise ValueError("V repair supports one sequence")
        s, h, d = v.shape
        selected = round(budget * s)
        rows = math.ceil(selected / 128) * 128
        if selected == 0:
            plain = prepare_fp8_fused(q, k, v)
            return PreparedVRepair(
                plain.q,
                plain.k,
                plain.v,
                plain.q_descale,
                plain.k_descale,
                plain.v_descale,
                None,
                plain.layout,
                shape,
                0,
                0,
                plain.v_prepacked,
            )
        total = rows + s
        q_blocks, k_blocks = math.ceil(s / 128), math.ceil(s / 128)
        device = q.device

        q8 = torch.empty((1, s, h, d), device=device, dtype=torch.float8_e4m3fn)
        q_descale = torch.empty((1, h, q_blocks), device=device, dtype=torch.float32)
        k_all = torch.empty((total, h, d), device=device, dtype=torch.float8_e4m3fn)
        k_descale = torch.empty((1, h, k_blocks), device=device, dtype=torch.float32)
        v_all = torch.empty((total, h, d), device=device, dtype=torch.float8_e4m3fn)
        for x, out, scales in ((q, q8, q_descale), (k, k_all[rows:], k_descale)):
            _per_block_quant_kernel[(1, q_blocks, h)](
                x,
                out,
                scales,
                s,
                h,
                d,
                q_blocks,
                IS_INT8=False,
                QMAX=_FP8_E4M3_MAX,
                BLOCK_L=128,
                BLOCK_D=next_power_of_2(d),
                num_warps=4,
            )

        amax = torch.zeros((1, h), device=device, dtype=torch.float32)
        _per_head_amax_kernel[(1, q_blocks, h)](
            v, amax, s, h, d, BLOCK_L=128, BLOCK_D=next_power_of_2(d), num_warps=4
        )
        score = torch.empty((h, s), device=device, dtype=torch.float32)
        _v_quant_score_kernel[(q_blocks, h)](
            v, v_all[rows:], amax, score, s, h, QMAX=_FP8_E4M3_MAX, BLOCK_L=128, D=d, num_warps=4
        )
        v_descale = amax.clamp(min=1e-12) / _FP8_E4M3_MAX

        layout = dict(layout)
        index = torch.topk(score, selected, dim=1, sorted=False).indices.to(torch.int32)
        repair_kd = torch.empty((1, h, rows), device=device, dtype=torch.float32)
        _repair_rows_kernel[(rows, h)](
            k_all,
            v_all,
            v,
            k_descale,
            v_descale,
            index.contiguous(),
            repair_kd,
            selected,
            rows,
            s,
            h,
            k_blocks,
            QMAX=_FP8_E4M3_MAX,
            D=d,
            num_warps=1,
        )
        ones = torch.ones((1, h, rows // 128), device=device, dtype=torch.float32)
        k_descale_all = torch.cat((ones, k_descale), dim=2)
        layout["cu_seqlens_k"] = torch.arange(2, device=device, dtype=torch.int32) * total
        layout["max_seqlen_k"] = total
        # Repair rows only run on the packed-V path, so pack once here rather than in
        # every attention call.
        from ._kernels.blackwell.flash_attn.cute.v_layout import pack_v

        return PreparedVRepair(
            q8[0],
            k_all,
            pack_v(v_all),
            q_descale,
            k_descale_all,
            v_descale,
            repair_kd,
            layout,
            shape,
            rows,
            selected,
            v_prepacked=True,
        )


def attention_v_repair(
    prepared, *, mid_window_blocks=DEFAULT_MID_WINDOW_BLOCKS, softmax_scale=None
):
    """Dense Open-VC ExpCast attention plus optional V repair tokens.

    mid_window_blocks=4 is the default dense scan; None selects the original
    scan. Causal, LSE and reference-backend calls are not supported here.
    """
    out, _ = raw_forward(
        prepared.q,
        prepared.k,
        prepared.v,
        version=DEFAULT_VERSION,
        q_descale=prepared.q_descale,
        k_descale=prepared.k_descale,
        v_descale=prepared.v_descale,
        **prepared.layout,
        expcast=True,
        softmax_scale=softmax_scale,
        mid_window_blocks=mid_window_blocks,
        return_lse=False,
        repair_k_descale=prepared.repair_k_descale,
        v_prepacked=prepared.v_prepacked,
    )
    return out.reshape(prepared.output_shape)
