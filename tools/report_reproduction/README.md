# Report reproduction tools

Use the [complete reproduction guide](../../docs/reports/reproduction.md) for
environment setup, commands, controls, timing boundaries and failure handling.

| File | Role |
|---|---|
| [build.py](build.py) | CPU-only AOT controls, source/ABI/native provenance and opt-in D derivative |
| [benchmark.py](benchmark.py) | B200 kernel-only replay, byte checks and balanced raw sampling |
| [summarize.py](summarize.py) | Recompute paired statistics from complete results |
| [common.py](common.py) | Source staging, identities and data-only measurement contracts |

Build artifacts stay under the caller's fresh build directory. Nothing is
installed into the public kernel package. D requires `--include-d` at both steps.
The tools reuse the documented cache-key serialization in
[idle_benchmark/key_format.py](../idle_benchmark/key_format.py).

[Repository tools](../README.md)
