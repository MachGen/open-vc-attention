# Documentation guide

| Guide | What it explains |
|---|---|
| [Installation and first run](installation.md) | Clean environment, a working first call and expected outputs |
| [API and integration](integration.md) | Supported tensors, FP8 preparation, direct PyTorch use and advanced helpers |
| [Troubleshooting](troubleshooting.md) | Dependency, device, compilation and measurement failures |
| [Benchmark protocol](benchmarking.md) | Custom shapes, timing scopes, paired sampling, input captures and GPU isolation |
| [Performance](performance.md) | B200 and B300 results, exact baselines, evidence and validation limits |
| [How it works](design.md) | Data flow, fused dispatch, package layers and framework responsibilities |
| [Fusedpipe and D reports](reports/README.md) | English and Chinese B200 scheduling reports, with a long-sequence supplement |
| [Report reproduction](reports/reproduction.md) | Matched controls, kernel-only runner, environments and validation boundaries |
| [Technical whitepaper](whitepaper.md) | VC and FlashAttn V6 algorithms, separate BF16 comparisons, numerical validation and reproducibility |
| [SGLang diffusion example](sglang.md) | Optional framework registration, packed sequences and a MiniMax-H3 pipeline example |
| [Project overview](overview.md) | Longer engineering article with motivation, measured gains and historical context |
| [Recorded measurements](benchmarks/README.md) | Raw samples and provenance behind published tables |
| [Packaging and usability review](review.md) | Findings, fixes, executed checks and remaining validation |
| [VC-attn synchronization](upstream-sync.md) | Upstream commit coverage, preserved snapshots and regression validation |
| [Appendix](appendix/README.md) | Historical VC snapshots, optional experiments and initial validation |

Operator timings, numerical tests and full model generation have different
measurement boundaries. Start with the [root quickstart](../README.md) for a first call.

[Repository](../README.md)
