"""Lazy implementation selection; importing this module does not load CUDA."""

from functools import lru_cache
from importlib import import_module

DEFAULT_VERSION = "open-vc"
DEFAULT_MID_WINDOW_BLOCKS = 4
INTERFACES = {
    "open-vc": "open_vc_attn._kernels.blackwell.flash_attn.cute.interface",
    "reference": "flash_attn.cute.interface",
}
VERSIONS = tuple(INTERFACES)


@lru_cache(None)
def interface(version=DEFAULT_VERSION):
    if version not in INTERFACES:
        raise ValueError(f"Unknown implementation {version!r}; choose from {VERSIONS}")
    return import_module(INTERFACES[version])
