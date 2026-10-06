"""Value-guided K/V permutation and block demeaning for SM100 FP8 attention."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _assign_values(
    V,
    C,
    Labels,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    head = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    channels = tl.arange(0, D)
    clusters = tl.arange(0, BK)
    values = tl.load(
        V + (head * N + rows[:, None]) * D + channels[None, :], rows[:, None] < N, 0
    )
    centers = tl.load(
        C + (head * K + clusters[None, :]) * D + channels[:, None],
        clusters[None, :] < K,
        0,
    )
    distances = -2 * tl.dot(values, centers)
    distances += tl.sum(centers.to(tl.float32) * centers.to(tl.float32), axis=0)[
        None, :
    ]
    distances = tl.where(clusters[None, :] < K, distances, float("inf"))
    labels = tl.argmin(distances, axis=1)
    tl.store(Labels + head * N + rows, labels, rows < N)


@dataclass(frozen=True)
class PreparedVSmooth:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    means: torch.Tensor
    scale: torch.Tensor
    permutation: torch.Tensor
    centroids: torch.Tensor | None

    def forward_kwargs(self):
        return dict(v_smooth=True, v_smooth_means=self.means, v_smooth_scale=self.scale)


def prefer_head_major_v_smooth(
    q: torch.Tensor, *, expcast: bool = False, mid_window_blocks: int | None = 4
) -> bool:
    from vc_attn._kernels.v2.flash_attn.cute.tuning import (
        CUTLASS_DSL_VERSION,
        PROFILED_MID_WINDOW_BLOCKS,
        V_SMOOTH_HEAD_MAJOR_MIN_SEQLEN,
        V_SMOOTH_HEAD_MAJOR_NUM_HEADS,
    )

    # Restrict layout selection to the measured Blackwell / Wan configuration.
    return (
        mid_window_blocks == PROFILED_MID_WINDOW_BLOCKS
        and CUTLASS_DSL_VERSION == "4.6.0"
        and q.is_cuda
        and q.ndim == 3
        and q.shape[0] >= V_SMOOTH_HEAD_MAJOR_MIN_SEQLEN
        and q.shape[1:] == (V_SMOOTH_HEAD_MAJOR_NUM_HEADS, 128)
        and torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))
    )


@torch.no_grad()
def prepare_v_smooth(
    q,
    k,
    v,
    *,
    clusters=64,
    iterations=4,
    permutation=None,
    centroids=None,
    expand_means=False,
    head_major_kv=False,
):
    """Prepare dense self-attention; callers explicitly own/reuse grouping state.

    Accepts [B,S,H,128] or single-sequence [S,H,128] BF16 inputs. Returned
    means are BF16 in residual-code units; scales are FP32 per batch/head/channel.
    expand_means preconverts means to FP32 for the measured DSL 4.6 varlen path.
    head_major_kv preserves contiguous tokens within each K/V head.
    K/V use the same per-head permutation. Q and output row order are unchanged.
    Group counts and Lloyd iterations are tunable, not specified by the paper.
    """
    if (
        q.ndim not in (3, 4)
        or q.shape != k.shape
        or q.shape != v.shape
        or q.shape[-1] != 128
    ):
        raise ValueError(
            "V-Smooth expects equal self-attention Q/K/V shapes ending in D=128"
        )
    if any(
        x.dtype != torch.bfloat16 or x.device != q.device or not x.is_cuda
        for x in (q, k, v)
    ):
        raise ValueError(
            "V-Smooth preprocessing requires BF16 Q/K/V on one CUDA device"
        )
    squeeze = q.ndim == 3
    qb, kb, vb = (x.unsqueeze(0) if squeeze else x for x in (q, k, v))
    batch, length, heads, dim = qb.shape
    if length == 0 or not 1 <= clusters <= min(128, length) or iterations < 1:
        raise ValueError(
            "Require nonempty tokens, 1 <= clusters <= min(128,S), iterations >= 1"
        )
    vh = vb.permute(0, 2, 1, 3).contiguous().reshape(batch * heads, length, dim)
    centers = None
    if permutation is None:
        if centroids is None:
            indices = torch.linspace(0, length - 1, clusters, device=q.device).long()
            centers = vh[:, indices].contiguous()
        else:
            if (
                centroids.shape != (batch, heads, clusters, dim)
                or centroids.dtype != torch.bfloat16
                or centroids.device != q.device
            ):
                raise ValueError(
                    "centroids must be BF16 [B,H,clusters,128] on the input device"
                )
            centers = centroids.reshape(batch * heads, clusters, dim).contiguous()
        labels = torch.empty(
            (batch * heads, length), device=q.device, dtype=torch.int32
        )
        values = vh.float()
        ones = torch.ones_like(labels, dtype=torch.float32)
        for _ in range(iterations):
            _assign_values[(batch * heads, triton.cdiv(length, 128))](
                vh,
                centers,
                labels,
                length,
                clusters,
                dim,
                128,
                triton.next_power_of_2(clusters),
            )
            sums = torch.zeros(
                (batch * heads, clusters, dim), device=q.device, dtype=torch.float32
            )
            counts = torch.zeros(
                (batch * heads, clusters), device=q.device, dtype=torch.float32
            )
            ids = labels.long()
            sums.scatter_add_(1, ids[..., None].expand(-1, -1, dim), values)
            counts.scatter_add_(1, ids, ones)
            centers = torch.where(
                counts[..., None] > 0,
                sums / counts[..., None].clamp_min(1),
                centers.float(),
            ).to(torch.bfloat16)
        _assign_values[(batch * heads, triton.cdiv(length, 128))](
            vh,
            centers,
            labels,
            length,
            clusters,
            dim,
            128,
            triton.next_power_of_2(clusters),
        )
        permutation = torch.argsort(labels, dim=-1, stable=True).reshape(
            batch, heads, length
        )
        centers = centers.reshape(batch, heads, clusters, dim)
    else:
        if centroids is not None:
            raise ValueError("Pass a permutation or centroids, not both")
        if (
            permutation.shape != (batch, heads, length)
            or permutation.dtype != torch.int64
            or permutation.device != q.device
        ):
            raise ValueError("permutation must be int64 [B,H,S] on the input device")
        expected = torch.arange(length, device=q.device).expand(batch, heads, length)
        if not torch.equal(permutation.sort(dim=-1).values, expected):
            raise ValueError(
                "Each head permutation must contain every token exactly once"
            )
    indices = permutation[..., None].expand(-1, -1, -1, dim)
    kp = torch.gather(kb.permute(0, 2, 1, 3), 2, indices)
    vp = torch.gather(vb.permute(0, 2, 1, 3), 2, indices)
    blocks = triton.cdiv(length, 128)
    padded = torch.nn.functional.pad(vp, (0, 0, 0, blocks * 128 - length))
    grouped = padded.reshape(batch, heads, blocks, 128, dim).float()
    counts = torch.full((blocks,), 128, device=q.device, dtype=torch.float32)
    counts[-1] = length - (blocks - 1) * 128
    means = grouped.sum(dim=3) / counts[None, None, :, None]
    residual = grouped - means.unsqueeze(3)
    if length % 128:
        residual[:, :, -1, length % 128 :] = 0
    amax = residual.abs().amax(dim=(2, 3))
    scale = torch.where(amax > 0, amax / 448.0, 1.0)
    codes = (
        (residual / scale[:, :, None, None, :]).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    codes = codes.reshape(batch, heads, blocks * 128, dim)[:, :, :length].permute(
        0, 2, 1, 3
    )
    if not head_major_kv:
        codes = codes.contiguous()
    means = (
        (means / scale[:, :, None, :])
        .permute(0, 2, 1, 3)
        .to(torch.bfloat16)
        .contiguous()
    )
    if expand_means:
        # Preserve rounding while removing conversions from correction tiles.
        means = means.float()
    qp = qb.to(torch.float8_e4m3fn)
    if head_major_kv:
        kp = kp.to(torch.float8_e4m3fn).permute(0, 2, 1, 3)
    else:
        kp = kp.permute(0, 2, 1, 3).to(torch.float8_e4m3fn).contiguous()
    if squeeze:
        qp, kp, codes = (x.squeeze(0) for x in (qp, kp, codes))
    return PreparedVSmooth(
        qp, kp, codes, means, scale.contiguous(), permutation, centers
    )
