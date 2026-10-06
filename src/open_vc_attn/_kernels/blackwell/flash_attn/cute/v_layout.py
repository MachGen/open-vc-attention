"""Prepare sequence-contiguous FP8 V for the dense fused-denominator kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_v(
    source,
    target,
    length: tl.constexpr,
    channels: tl.constexpr,
    padded: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    batch = tl.program_id(2)
    value = tl.load(
        source + batch * length * channels + row[:, None] * channels + col[None, :],
        (row[:, None] < length) & (col[None, :] < channels),
        0,
    )
    tl.store(
        target + batch * channels * padded + col[None, :] * padded + row[:, None],
        value,
        (row[:, None] < padded) & (col[None, :] < channels),
    )


def pack_v(value: torch.Tensor) -> torch.Tensor:
    length, heads, dim = value.shape[-3:]
    padded = triton.cdiv(length, 128) * 128
    batch = value.shape[0] if value.ndim == 4 else 1
    buffer = torch.empty((batch, heads, dim, padded), device=value.device, dtype=value.dtype)
    transpose_v[(triton.cdiv(padded, 64), triton.cdiv(heads * dim, 128), batch)](
        value.contiguous().view(torch.uint8),
        buffer.view(torch.uint8),
        length,
        heads * dim,
        padded,
        64,
        128,
        num_warps=4,
    )
    result = buffer[..., :length].movedim(-1, -3)
    return result if value.ndim == 4 else result.squeeze(0)
