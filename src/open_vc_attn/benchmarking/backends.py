"""The public benchmark compares BF16, the VC-Attention baseline and Open-VC."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    version: str
    mode: str
    description: str


BACKENDS = {
    "bf16": Backend("reference", "bf16", "Upstream FlashAttention-4 BF16 (flash-attn-4 4.0.0b33)"),
    "vc": Backend(
        "open-vc",
        "vsmooth",
        "VC-Attention: ExpCast + V-Smooth, original scan, no Open-VC optimizations",
    ),
    "open-vc": Backend("open-vc", "expcast", "Open-VC Attn: ExpCast with Open-VC optimizations"),
}
DEFAULT_BACKENDS = tuple(BACKENDS)
DEFAULT_BASELINE = "bf16"
CANDIDATE = "open-vc"


def get_backend(name):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {tuple(BACKENDS)}")
    return BACKENDS[name]


__all__ = ["BACKENDS", "CANDIDATE", "DEFAULT_BACKENDS", "DEFAULT_BASELINE", "get_backend"]
