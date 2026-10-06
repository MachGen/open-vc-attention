"""Measured SM100 low-bit choices; unknown compiler versions keep the baseline."""

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
    profiled = (
        eligible
        and dsl_version in PROFILED_DSL_VERSIONS
        and skip_softmax_error in (0.0, PROFILED_SKIP_SOFTMAX_ERROR)
    )
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
    enabled = (
        expcast
        and eligible
        and dsl_version in PROFILED_DSL_VERSIONS
        and skip_softmax_error in (0.0, PROFILED_SKIP_SOFTMAX_ERROR)
    )
    return NVFP4ExpCastTuning(
        enabled,
        enabled and skip_softmax_error == PROFILED_SKIP_SOFTMAX_ERROR,
        enabled,
        enabled and dsl_version == "4.6.0",
    )
