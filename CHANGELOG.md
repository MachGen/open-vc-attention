# Changelog

## Unreleased

- Add `prepare_fp8_fused` for contiguous, single-sequence B200 inputs with
  D=128 and S>=32768; preserve current block/head quantization scales.
- Combine Q/K launch and Q-scale padding; fuse V cast, packed-layout output
  and descale. Reuse caller metadata for three preparation kernels.
- Add validated prepacked-V consumption on the existing dense v4 path.
  Keep default preparation, attention arithmetic and disabled D unchanged.

- Added public CPU AOT builders and a B200 kernel-only report runner with matched
  pre-fusedpipe/mid4, fusedpipe and BF16 controls. Experimental D requires explicit
  build and replay opt-in; API defaults are unchanged.
- Expanded both technical reports to 15 pages with equivalent long-sequence
  coverage, public reproduction instructions and environment/provenance records.
  Included editable PDF sources; historical measurements are unchanged.
- Unified the public API and default benchmark on `v4` with mid-window 4
  (`vc_v4_mid4`), using source-level fusedpipe on eligible B200 calls.
- Disabled experimental D binary patching in public API and benchmark calls
  while its remaining issues are investigated. Only explicit helper experiments
  and offline report replay can opt in; frozen kernel code and all patch guards
  are unchanged.
- Updated report candidate selection for mid4, recorded scan/D configuration
  and adapter provenance, reconciled current-default documentation, and added
  default-policy regression coverage. GPU CI now includes the v4 test suite.

### Earlier development changes

- Default dense `attention()` and `attention_fp8()` calls to `mid_window_blocks=4`,
  selecting fusedpipe on eligible B200 configurations. Explicit `None` keeps the
  original scan; causal and BF16 calls keep their respective traversals. The D
  binary guard and low-level snapshot defaults were unchanged at that revision;
  the public D path is now disabled as described above.
- Published [English and Chinese fusedpipe/D technical reports](docs/reports/README.md)
  and sanitized long-sequence measurements. The English report includes the newer
  synthetic-input sweep; the Chinese report covers the earlier captured-input cohorts.

- Added the `v4` snapshot from VC-attn `8aa761eac`, including unscaled original-scan ExpCast dispatch and both SM103 ordinary-FP8 scheduling improvements. The default API, diffusion adapter and benchmark now select `v4` / `vc_v4`; historical snapshots and backend identities remain available. See [upstream synchronization](docs/upstream-sync.md) for source coverage and validation.

- Added a technical whitepaper covering VC and native FlashAttn V6, separate BF16 performance tables, numerical policies, reproducibility and integration limits.

- Reorganized the README around the general attention API, supported configurations, installation and custom workloads. Retained B200/B300 performance summaries and moved model-specific usage to the integration guides.
- Clarified the benchmark CLI's single-sequence/noncausal scope and that the ExpCast setting can change fused kernel dispatch, not just probability conversion.

- Recorded fresh matched B300 BF16/ordinary-FP8/native-v6/VC performance on four MiniMax shapes, with complete raw paired samples and an explicit shared-GPU protocol.

- Added CPU-only CuTe/Triton compilation and an optional B300 idle-window benchmark helper. Complete paired rounds are retained across pauses; interfered rounds are discarded.

## 0.1.2

- Fixed pip reproduction constraints containing unsupported extras; documented a clean CUDA 13 installation path.
- Added `vc-attn-check` for metadata validation and an opt-in GPU smoke, plus `vc-attn-report` for CPU-only tables with explicit baseline selection.
- Added first-run, support-matrix, troubleshooting and package-design guides.
- Improved benchmark preflight errors, runtime provenance and device-configuration metadata; reports print a result table after completion.
- Honored explicit `CUDA_HOME` for native builds and selected the owning CUDA device during native-plan cleanup.
- Added fresh full-protocol B200 results from the clean public environment. Three native B300 operator tests passed; guarded B300 performance attempts were interrupted and excluded.
- Reviewed distribution contents, entry points, documentation links, source hashes and attribution; see [review findings](docs/review.md) for validation scope and remaining B300 qualification.

## 0.1.1

- Focused the homepage and default benchmark on scaled ExpCast VC with fixed BF16/FP8 baselines.
- Added B200 full-protocol measurements and provenance-checked historical B300 numbers, with explicit denominator identities.
- Kept original native v6 for B300 / SM103 only; removed the B200 compatibility implementation from current source and distributions.
- Moved historical VC comparisons and optional experiments to the documentation appendix.
- Renamed the GitHub directory guide so GitHub displays the root project README.

## 0.1.0

- Initial standalone extraction of attention kernels, inference API and explicit FP8 preparation.
- Randomized paired benchmarks with accuracy, isolation and source metadata.
- Pinned SGLang diffusion registration, packed-sequence adapter and pipeline runner.
- CPU contracts, Blackwell GPU tests, licenses, attribution and source auditing.

See [Performance](docs/performance.md) for measured coverage and integration limits.
