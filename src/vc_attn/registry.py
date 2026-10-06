"""Explicit names keep baseline identity separate from dtype and feature flags."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Backend:
    version: str
    mode: str
    description: str


BACKENDS = {
    "bf16_ref": Backend("baseline", "bf16", "Pinned BF16 source baseline"),
    "fp8_ref": Backend("baseline", "fp8", "Pinned ordinary FP8 source baseline, with descales"),
    "fp8_control": Backend("scaled", "fp8", "Scaled snapshot with descales and ExpCast disabled"),
    "vc_v1": Backend("v1", "expcast", "First extracted VC revision"),
    "vc_v2": Backend("v2", "expcast", "Second extracted VC revision"),
    "vc_v3": Backend("v3", "expcast", "Third extracted VC revision, general scaled ExpCast"),
    "vc_v4": Backend("v4", "expcast", "Latest VC snapshot with original-scan ExpCast dispatch"),
    "vc_v4_mid4": Backend(
        "v4", "expcast_mid4", "Default VC: mid-window 4, eligible B200 fusedpipe, no D"
    ),
    "fp8_v4": Backend("v4", "fp8", "Latest ordinary FP8 with SM103 scheduling improvements"),
    "vc_scaled": Backend("scaled", "expcast", "Scaled ExpCast with eligible fused dispatch"),
    "vc_scaled_mid4": Backend(
        "scaled", "expcast_mid4", "Scaled ExpCast with explicit mid-window 4"
    ),
    "vc_vsmooth": Backend("scaled", "vsmooth", "Experimental V-Smooth plus ExpCast"),
    "vc_nvfp4": Backend("scaled", "nvfp4", "Experimental NVFP4 Q/K plus ExpCast"),
    "upstream_bf16": Backend("upstream", "bf16", "Optional upstream flash-attn-4 BF16"),
    "torch_sdpa": Backend("torch", "bf16", "PyTorch-selected SDPA backend; not a pinned kernel"),
    "native_v6": Backend("native", "fp8", "Original B300/SM103 native CUDA v6"),
}

DEFAULT_VERSION = "v4"
DEFAULT_MID_WINDOW_BLOCKS = 4
DEFAULT_BACKENDS = ("bf16_ref", "fp8_ref", "vc_v4_mid4")
VERSIONS = ("baseline", "v1", "v2", "v3", "scaled", "v4")


def source_manifest():
    return json.loads(Path(__file__).with_name("source_manifest.json").read_text())


def get_backend(name):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[name]


def main():
    print(
        json.dumps(
            {
                "backends": {k: asdict(v) for k, v in BACKENDS.items()},
                "defaults": {
                    "version": DEFAULT_VERSION,
                    "mid_window_blocks": DEFAULT_MID_WINDOW_BLOCKS,
                    "benchmark_backends": DEFAULT_BACKENDS,
                    "sass_d_enabled": False,
                },
                "source_revisions": {
                    k: v["source_revision"] for k, v in source_manifest()["versions"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
