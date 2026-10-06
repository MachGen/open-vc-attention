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


@triton.jit
def _accumulate_centers(
    V,
    Labels,
    Sums,
    Counts,
    N,
    K: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STEPS: tl.constexpr,
):
    """Per-cluster sums and counts of one head's rows via one-hot tensor-core products."""
    head = tl.program_id(0)
    channels = tl.arange(0, D)
    clusters = tl.arange(0, BK)
    sums = tl.zeros((BK, D), dtype=tl.float32)
    counts = tl.zeros((BK,), dtype=tl.float32)
    for step in range(STEPS):
        rows = (tl.program_id(1) * STEPS + step) * BN + tl.arange(0, BN)
        valid = rows < N
        labels = tl.load(Labels + head * N + rows, valid, -1)
        onehot = (labels[:, None] == clusters[None, :]).to(tl.bfloat16)
        values = tl.load(V + (head * N + rows[:, None]) * D + channels[None, :], valid[:, None], 0)
        sums += tl.dot(tl.trans(onehot), values)
        counts += tl.sum(onehot.to(tl.float32), axis=0)
    keep = clusters < K
    tl.atomic_add(
        Sums + (head * K + clusters[:, None]) * D + channels[None, :], sums, keep[:, None]
    )
    tl.atomic_add(Counts + head * K + clusters, counts, keep)


@triton.jit
def _block_stats(
    V, Perm, Means, Amax, S, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr
):
    """Per (head, block): gather V rows in group order, block mean and channel residual max."""
    head = tl.program_id(0)
    block = tl.program_id(1)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    valid = rows < S
    channels = tl.arange(0, D)
    tokens = tl.load(Perm + head * S + rows, valid, 0)
    values = tl.load(
        V + (tokens[:, None] * H + head) * D + channels[None, :], valid[:, None], 0.0
    ).to(tl.float32)
    count = tl.minimum(S - block * BLOCK, BLOCK).to(tl.float32)
    mean = tl.sum(values, axis=0) / count
    residual = tl.where(valid[:, None], values - mean[None, :], 0.0)
    tl.store(Means + (head * tl.cdiv(S, BLOCK) + block) * D + channels, mean)
    tl.atomic_max(Amax + head * D + channels, tl.max(tl.abs(residual), axis=0))


@triton.jit
def _block_write(
    K,
    V,
    Perm,
    Means,
    Amax,
    KOut,
    Codes,
    MeansOut,
    Scale,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per (head, block): write permuted FP8 K, FP8 V residual codes and scaled means."""
    head = tl.program_id(0)
    block = tl.program_id(1)
    blocks = tl.cdiv(S, BLOCK)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    valid = rows < S
    channels = tl.arange(0, D)
    amax = tl.load(Amax + head * D + channels)
    scale = tl.where(amax > 0, amax / 448.0, 1.0)
    mean = tl.load(Means + (head * blocks + block) * D + channels)
    tokens = tl.load(Perm + head * S + rows, valid, 0)
    source = (tokens[:, None] * H + head) * D + channels[None, :]
    target = (rows[:, None] * H + head) * D + channels[None, :]
    values = tl.load(V + source, valid[:, None], 0.0).to(tl.float32)
    codes = tl.minimum(tl.maximum((values - mean[None, :]) / scale[None, :], -448.0), 448.0)
    tl.store(Codes + target, codes.to(tl.float8e4nv), valid[:, None])
    keys = tl.load(K + source, valid[:, None], 0.0)
    tl.store(KOut + target, keys.to(tl.float8e4nv), valid[:, None])
    scaled = (mean / scale).to(tl.bfloat16).to(tl.float32)
    tl.store(MeansOut + (block * H + head) * D + channels, scaled)
    if block == 0:
        tl.store(Scale + head * D + channels, scale)


@torch.no_grad()
def apply_v_smooth(q, k, v, permutation):
    """Fused V-Smooth preparation for one sequence with an existing grouping.

    q, k, v: contiguous BF16 [S,H,128]; permutation: int64 [1,H,S] from an earlier
    ``prepare_v_smooth`` call. Gathers K/V into group order, demeans V per 128-token
    block and quantizes in two passes, matching ``prepare_v_smooth`` with
    ``expand_means=True``. Returns a ``PreparedVSmooth`` (centroids are not recomputed).
    """
    length, heads, dim = v.shape
    if dim != 128 or q.shape != v.shape or k.shape != v.shape:
        raise ValueError("Fused V-Smooth expects matching [S,H,128] self-attention inputs")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    blocks = triton.cdiv(length, 128)
    perm = permutation.reshape(heads, length).contiguous()
    means = torch.empty((heads, blocks, dim), device=v.device, dtype=torch.float32)
    amax = torch.zeros((heads, dim), device=v.device, dtype=torch.float32)
    _block_stats[(heads, blocks)](v, perm, means, amax, length, heads, dim, 128)
    k_out = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    codes = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    means_out = torch.empty((1, blocks, heads, dim), device=v.device, dtype=torch.float32)
    scale = torch.empty((1, heads, dim), device=v.device, dtype=torch.float32)
    _block_write[(heads, blocks)](
        k, v, perm, means, amax, k_out, codes, means_out, scale, length, heads, dim, 128
    )
    return PreparedVSmooth(
        q.to(torch.float8_e4m3fn), k_out, codes, means_out, scale, permutation, None
    )


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
    from open_vc_attn._kernels.blackwell.flash_attn.cute.fp8_tuning import CUTLASS_DSL_VERSION

    # Restrict head-major layout selection to the tuned Blackwell configuration.
    return (
        mid_window_blocks == 4
        and CUTLASS_DSL_VERSION == "4.6.2"
        and q.is_cuda
        and q.ndim == 3
        and q.shape[0] >= 49152
        and q.shape[1:] == (40, 128)
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
    check_permutation=True,
):
    """Prepare dense self-attention; callers explicitly own/reuse grouping state.

    Accepts [B,S,H,128] or single-sequence [S,H,128] BF16 inputs. Returned
    means are BF16 in residual-code units; scales are FP32 per batch/head/channel.
    expand_means preconverts means to FP32 for the DSL 4.6 varlen path.
    head_major_kv preserves contiguous tokens within each K/V head.
    check_permutation=False skips the host-synchronizing permutation check, for reusing a
    permutation returned by an earlier call (for example inside CUDA graph capture).
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
    centers = None
    if permutation is None:
        vh = vb.permute(0, 2, 1, 3).contiguous().reshape(batch * heads, length, dim)
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
        steps = 8
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
            _accumulate_centers[(batch * heads, triton.cdiv(length, 128 * steps))](
                vh,
                labels,
                sums,
                counts,
                length,
                clusters,
                dim,
                128,
                triton.next_power_of_2(clusters),
                steps,
            )
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
        if check_permutation and not torch.equal(permutation.sort(dim=-1).values, expected):
            raise ValueError(
                "Each head permutation must contain every token exactly once"
            )
    if expand_means and not head_major_kv and batch == 1:
        # Fused permute/demean/quantize (two Triton passes) for the FP32-means layout.
        fused = apply_v_smooth(qb[0], kb[0], vb[0], permutation)
        result = PreparedVSmooth(
            fused.q, fused.k, fused.v, fused.means, fused.scale, permutation, centers
        )
        if not squeeze:
            result = PreparedVSmooth(
                result.q[None], result.k[None], result.v[None], result.means, result.scale,
                permutation, centers,
            )
        return result
    indices = permutation[..., None].expand(-1, -1, -1, dim)
    kp = torch.gather(kb.permute(0, 2, 1, 3), 2, indices)
    vp = torch.gather(vb.permute(0, 2, 1, 3), 2, indices)
    blocks = triton.cdiv(length, 128)
    padded = torch.nn.functional.pad(vp, (0, 0, 0, blocks * 128 - length))
    grouped = padded.reshape(batch, heads, blocks, 128, dim).float()
    counts = torch.full((blocks,), 128, device=q.device, dtype=torch.float32)
    # Device-side fill keeps preparation capturable in a CUDA graph.
    counts[-1:].fill_(float(length - (blocks - 1) * 128))
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
