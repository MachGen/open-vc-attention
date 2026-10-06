"""Print package metadata without importing Torch or initializing CUDA."""

import json
from dataclasses import asdict

from .. import __version__
from .._dispatch import DEFAULT_MID_WINDOW_BLOCKS, DEFAULT_VERSION
from ..benchmarking.backends import BACKENDS


def main():
    print(
        json.dumps(
            {
                "package": "open-vc-attn",
                "version": __version__,
                "backends": {name: asdict(spec) for name, spec in BACKENDS.items()},
                "defaults": {
                    "version": DEFAULT_VERSION,
                    "mid_window_blocks": DEFAULT_MID_WINDOW_BLOCKS,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
