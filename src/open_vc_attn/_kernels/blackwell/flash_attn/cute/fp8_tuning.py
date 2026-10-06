"""Tuned SM100 low-bit choices; unknown compiler versions keep the defaults."""

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
    eligible: bool,
    dsl_version: str = CUTLASS_DSL_VERSION,
) -> FP8Tuning:
    tuned = eligible and dsl_version in ("4.4.1", "4.6.2")
    split_encoding = tuned and expcast
    return FP8Tuning(
        tuned and dsl_version == "4.4.1" and not expcast,
        split_encoding,
        expcast,
        split_encoding,
        split_encoding,
        split_encoding and dsl_version == "4.6.2",
    )


class NVFP4ExpCastTuning(NamedTuple):
    half_codes: bool
    same_stage_release: bool
    reallocate_registers: bool


def select_nvfp4_expcast_tuning(
    *,
    expcast: bool,
    eligible: bool,
    dsl_version: str = CUTLASS_DSL_VERSION,
) -> NVFP4ExpCastTuning:
    enabled = expcast and eligible and dsl_version in ("4.4.1", "4.6.2")
    return NVFP4ExpCastTuning(enabled, enabled, enabled and dsl_version == "4.6.2")
