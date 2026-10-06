# Repository tools

Run these scripts from the repository root. They support source maintenance,
optional native builds, framework registration and pipeline measurement.

| Script | Purpose | Typical command |
|---|---|---|
| [audit_release.py](audit_release.py) | Checks forbidden artifacts/imports, private-path/credential patterns, attribution and frozen hashes | `python tools/audit_release.py` |
| [audit_upstream.py](audit_upstream.py) | Compares extracted sources and quantizers with an upstream revision | `python tools/audit_upstream.py /path/to/upstream --revision origin/feat/VC-attn` |
| [idle_benchmark/](idle_benchmark/README.md) | CPU compilation and bounded B300 sampling between workloads | See the directory guide |
| [report_reproduction/](report_reproduction/README.md) | Matched B200 report controls and kernel-only replay; D opt-in | See the reproduction guide |
| [report_pdf/](report_pdf/README.md) | Editable English/Chinese PDF sources | See the build guide |
| [build_native.py](build_native.py) | Compiles original B300 / SM103 native v6 with CUDA 13 | `python tools/build_native.py --arch sm_103a --output build/libvc_native_v6.so` |
| [install_sglang.py](install_sglang.py) | Preimage-checks, previews, applies or reverses the pinned registration patch | `python tools/install_sglang.py /path/to/sglang` |
| [benchmark_pipeline.py](benchmark_pipeline.py) | Times warm synchronous SGLang generation, excludes model loading and compares matching configuration fingerprints | `python tools/benchmark_pipeline.py --help` |

The operator benchmark is the installed `vc-attn-bench` command, implemented in
[benchmark.py](../src/vc_attn/benchmark.py). It has different timer boundaries
from `benchmark_pipeline.py`. See [the benchmark protocol](../docs/benchmarking.md)
and [SGLang integration](../docs/sglang.md).

The audit is a source-boundary check, not a comprehensive security scanner.
The pipeline runner is provided for integration; complete MiniMax generation
and model-quality validation have not been recorded for this release.

[Repository](../README.md)

Installed package commands also include `vc-attn-check` (metadata and optional GPU
smoke), `vc-attn-report` (CPU-only result tables) and `vc-attn-info` (backend/source
identities). See [Installation](../docs/installation.md) for a complete first run.
