"""Measured SM100 kernel choices; configurations outside the profiled set keep the baseline."""

from importlib.metadata import version
from typing import NamedTuple

CUTLASS_DSL_VERSION = version("nvidia-cutlass-dsl")

# Configurations the low-bit paths were profiled on (docs/quantization/expcast.md).
# Anything outside them keeps the general implementation.
PROFILED_DSL_VERSIONS = ("4.4.1", "4.6.0")
PROFILED_SKIP_SOFTMAX_ERROR = 7230.0
PROFILED_MID_WINDOW_BLOCKS = 4

# Packing V into K-major layout only pays off for large dense calls.
PACK_V_MIN_QUERY_HEAD_ROWS = 1 << 20
PACK_V_MIN_SEQLEN = 32768

# V-Smooth prefetches block means once there are enough keys to fill the mean pipeline.
V_SMOOTH_PREFETCH_MIN_SEQLEN_K = 4096
# Head-major K/V was measured only on the Wan 2.2 720p self-attention shape.
V_SMOOTH_HEAD_MAJOR_MIN_SEQLEN = 49152
V_SMOOTH_HEAD_MAJOR_NUM_HEADS = 40


def _profiled(eligible: bool, skip_softmax_error: float, dsl_version: str) -> bool:
    return (
        eligible
        and dsl_version in PROFILED_DSL_VERSIONS
        and skip_softmax_error in (0.0, PROFILED_SKIP_SOFTMAX_ERROR)
    )


class FP8Tuning(NamedTuple):
    reallocate_registers: bool
    same_stage_release: bool
    quantized_p_sum: bool
    split_expcast_encoding: bool
    tensor_core_denominator: bool
    rebalance_denominator_registers: bool


def select_fp8_tuning(
    *,
    expcast: bool,
    skip_softmax_error: float,
    eligible: bool,
    dsl_version: str = CUTLASS_DSL_VERSION,
) -> FP8Tuning:
    profiled = _profiled(eligible, skip_softmax_error, dsl_version)
    skip = skip_softmax_error > 0.0
    registers = profiled and dsl_version == "4.4.1" and not expcast and not skip
    release = profiled and expcast
    split_encoding = profiled and expcast and not skip
    # With skip enabled, decoded-P normalization is an additional approximation.
    return FP8Tuning(
        registers,
        release,
        expcast or (profiled and skip),
        split_encoding,
        split_encoding,
        split_encoding and dsl_version == "4.6.0",
    )


class NVFP4ExpCastTuning(NamedTuple):
    half_codes: bool
    tile_encoder: bool
    same_stage_release: bool
    reallocate_registers: bool


def select_nvfp4_expcast_tuning(
    *,
    expcast: bool,
    skip_softmax_error: float,
    eligible: bool,
    dsl_version: str = CUTLASS_DSL_VERSION,
) -> NVFP4ExpCastTuning:
    enabled = expcast and _profiled(eligible, skip_softmax_error, dsl_version)
    return NVFP4ExpCastTuning(
        enabled,
        enabled and skip_softmax_error == PROFILED_SKIP_SOFTMAX_ERROR,
        enabled,
        enabled and dsl_version == "4.6.0",
    )


class RegisterSplit(NamedTuple):
    softmax: int
    correction: int
    other: int


def select_register_split(
    *,
    head_dim_padded: int,
    paged_kv_non_tma: bool,
    block_sparse: bool,
    low_bit_p: bool,
    sparse_descale_rank: int,
    rebalance_denominator: bool,
    fp8_reallocate: bool,
    nvfp4_reallocate: bool,
    nvfp4_fused_skip: bool,
    v_smooth_tuned: bool,
    v_smooth_fp32_means: bool,
    inline_rescale: bool,
) -> RegisterSplit:
    """Per-warpgroup register budgets for the SM100 forward kernel.

    Each value was swept on B200. The base split depends on head dim, KV
    loading and sparsity; later rules are narrower measured configurations and
    take precedence over earlier ones.
    """
    other = 80 if paged_kv_non_tma else 48
    if head_dim_padded < 96:
        split = RegisterSplit(184 if paged_kv_non_tma else 200, 64, other)
    elif block_sparse:
        # Int8 sparse hd128 sweep (softmax/correction/other, 480p/720p delta):
        # 192/80/48 +85%, 192/72/48 +17%, 176/80/48 +2.7%, 160/96/48 +0.3% (chosen).
        # The sparse correction warp does extra work per iteration and wants 96.
        # Zero-spill configs still differed >=6x, so sweep and time jointly.
        split = RegisterSplit(184 if paged_kv_non_tma else 160, 96, other)
    else:
        # Swept optimum for fp8 and int8 hd128; int8 regresses 3-4% with more
        # correction registers because its softmax is issue-bound.
        split = RegisterSplit(184 if paged_kv_non_tma else 192, 80, other)
    rules = (
        (low_bit_p, RegisterSplit(192, 88, 40)),
        # Sparse per-block K scaling puts softmax on the critical path; per-head
        # scaling shifts it toward correction.
        (low_bit_p and sparse_descale_rank == 3, RegisterSplit(208, 56, 40)),
        (low_bit_p and sparse_descale_rank == 2, RegisterSplit(184, 96, 48)),
        (rebalance_denominator, RegisterSplit(168, 136, 40)),
        (fp8_reallocate, RegisterSplit(184, 104, 40)),
        (nvfp4_reallocate, RegisterSplit(176, 120, 40)),
        (nvfp4_fused_skip, RegisterSplit(176, 120, 40)),
        (v_smooth_tuned, RegisterSplit(184, 104, 40)),
        (v_smooth_fp32_means, RegisterSplit(176, 120, 40)),
        # Inline rescaling leaves the correction warps almost idle.
        (inline_rescale, RegisterSplit(232, 24, 40)),
    )
    for enabled, override in rules:
        if enabled:
            split = override
    return split
