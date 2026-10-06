# Packaging and usability review

This review covers the public API and wrappers, benchmark methodology, dependency
installation, distribution contents, documentation, licenses and source boundaries.
Frozen kernel bodies retain their original hashes; this is not a new formal audit
of every kernel algorithm or a full model-quality evaluation.

## Follow-up review: general-purpose onboarding

The 2026-09-30 pass checked the README against the public wrapper, backend
registry, kernel dispatch, benchmark CLI, native plan, framework adapter and
test coverage. It also rechecked packaging boundaries and directory guides.

| Finding | Change |
|---|---|
| The homepage led with one model's production shape before explaining the library | Lead with the attention API, supported configurations and a minimal PyTorch call; link model-specific examples from the integration section |
| API features could be mistaken for benchmark CLI features | State that the CLI measures single-sequence, noncausal self-attention at D=128, even though the API accepts additional configurations |
| The current-source A/B was described as isolating ExpCast | Explain that the flag can also change fused dispatch, packing, normalization and scheduling |
| Performance details obscured getting started | Keep all eight measured shapes and B300 native-v6 numbers in a compact summary; link full protocols and baseline latencies |

All homepage latencies and ratios were recomputed from the published raw
samples. The 11 CPU contracts, source audit, 40 directory guides, local Markdown
links and README example syntax passed. This pass changes documentation only;
the GPU results below are the existing recorded runs, not new GPU measurements.

## Changes from the review

| Finding | Change | Validation |
|---|---|---|
| Reproduction constraints included extras, which pip rejects | Plain names in the constraints file; Cu13 extras stay in package requirements | Clean public-source installation and `pip check`; CPU CI runs the same constraints parser |
| Installation assumed an existing CUDA/Torch environment | Complete venv + official CUDA 13 Torch + package installation sequence | Fresh Linux Python 3.12 virtual environments, independent of platform site-packages |
| First-call success and supported contracts were hard to find | Metadata check, opt-in GPU smoke, expected output, support matrix and troubleshooting | Installed console scripts, tiny FP32 oracle and convenience/prepared-output equivalence |
| Benchmark output required manually reading JSON | Automatic result table and CPU-only `vc-attn-report` | Baseline switching, raw-sample recomputation and invalid/incomplete-report rejection |
| Native options could fail late in a large workload | Preflight rejects missing library, unsupported timing scope or GPU | Invalid combinations fail before Torch import where possible; B300 target checked before allocating inputs |
| Compiler selection could ignore explicit `CUDA_HOME` | Explicit toolkit selection takes precedence over PATH | Original SM103 source compiled with CUDA 13; no kernel body changes |
| Native cleanup did not explicitly select the owning device | Plan destruction now enters its tensor's CUDA device context | Source/lifecycle review; three native B300 operator tests passed; a nonzero-device lifecycle-specific test was not run |
| Reports lacked wrapper identity and machine configuration | UTC timestamps, runtime/manifest hashes and driver/clock/power snapshots | Recorded by the packaged benchmark; no private host names or paths |
| GPU availability was consumed by lazy compilation and long uninterrupted runs | Optional CPU AOT preparation and a bounded idle-window sampler | GPUs hidden during compilation; four CuTe libraries rebuilt byte-for-byte; eight Triton specializations; complete guarded B300 rounds |

## Executed checks

- Eleven CPU contracts pass, including failure cases for incomplete results and invalid CLI arguments.
- Wheel console scripts install and run. The source distribution includes installation, troubleshooting, design and directory guides.
- Root README selection, local documentation links and all 40 source-directory guides were checked.
- Source auditing retains frozen hashes, attribution and the attention-only dependency boundary. No model weights, private captures, prebuilt CUDA DSO or B200-native compatibility code is shipped.
- The clean B200 install passes metadata checks, `vc-attn-check --smoke`, the quickstart example and eight GPU tests. Three B300-native tests are skipped on B200.
- The original native B300 v6 compiles with 128 registers and zero stack/spill. Its three B300 tail/descale correctness tests pass; fresh matched B300 timings are now recorded with the shared-window qualification.

See [Performance](performance.md) for accepted timing samples and measurement
boundaries. The fresh B300 AOT run collected 12 complete paired rounds on each
of four shapes, rejecting interference and pausing for production activity. It
is shared-window evidence, not an exclusive GPU qualification. The public helper
rebuilds identical libraries and dispatch/ABI metadata. Its small FP32 oracle and
S=129 smoke passed; a separate production-shape smoke reached its guarded
ten-minute limit during preparation and is not counted as a complete run.
Full SGLang/MiniMax generation and model-quality checks are also outstanding;
operator tests do not replace them.

## Reproduce the review

Use [Installation](installation.md), then the commands in
[CONTRIBUTING](../CONTRIBUTING.md). Follow [the benchmark protocol](benchmarking.md)
on an allocated GPU. Keep failures and interrupted runs separate from accepted
performance evidence. The source audit is a targeted boundary/provenance check,
not a general-purpose security scanner.

[Repository](../README.md) · [Documentation](README.md)
