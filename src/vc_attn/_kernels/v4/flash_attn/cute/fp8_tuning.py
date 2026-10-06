"""Measured SM100 low-bit choices; unknown compiler versions keep the baseline."""

from importlib.metadata import version
from typing import NamedTuple

CUTLASS_DSL_VERSION = version("nvidia-cutlass-dsl")


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
    fused_skip_softmax: bool = False,
) -> FP8Tuning:
    profiled = (
        eligible
        and dsl_version in ("4.4.1", "4.6.0")
        and (skip_softmax_error in (0.0, 7230.0) or fused_skip_softmax)
    )
    skip = skip_softmax_error > 0.0
    registers = profiled and dsl_version == "4.4.1" and not expcast and not skip
    release = profiled and expcast
    split_encoding = profiled and expcast and (not skip or fused_skip_softmax)
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
        and dsl_version in ("4.4.1", "4.6.0")
        and skip_softmax_error in (0.0, 7230.0)
    )
    return NVFP4ExpCastTuning(
        enabled,
        enabled and skip_softmax_error == 7230.0,
        enabled,
        enabled and dsl_version == "4.6.0",
    )
