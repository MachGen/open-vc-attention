# Fusedpipe and D scheduling reports

These reports explain the B200 / SM100 source-level fusedpipe schedule and the
guarded D SASS patch, including dependencies, register liveness, compatibility
checks and the limits of the measured gains.

| Artifact | Scope |
|---|---|
| [English technical report](VC-Attention-Fusedpipe-D-Technical-Report.pdf) | 15 pages; scheduling, both measurement cohorts, public reproduction and environment records |
| [中文技术报告](VC-Attention-Fusedpipe-D-Tech-Report-zh.pdf) | 15 页；调度机制、两组实验、长序列结果、公开复现步骤、环境与完整指令字 |
| [Reproduction guide](reproduction.md) | Runnable matched controls, timing boundary and failure handling |
| [Historical environment](historical-environment.json) | Recovered original environment with explicit unknown fields and provenance hashes |
| [Reproduction validation](reproduction-validation.json) | New tool verification kept separate from historical timings |
| [Long-sequence measurements](VC-Attention-Long-Sequence-Results.json) | Four accepted cases, generation seeds, all 192 paired timing samples and artifact hashes |
| [PDF manifest](manifest.json) | Hashes and sizes of the reviewed public PDFs |

Editable [PDF builders](../../tools/report_pdf/README.md) and sanitized
[captured-cohort table data](captured-summary.json) are included. The Chinese and
English editions now cover the same long-sequence cohort and reproduction path.

The reports analyze implementation revision `cbb2da8` and remain historical
records, including their D experiments. The current convenience API and default
benchmark use `mid_window_blocks=4`, enabling fusedpipe on eligible B200 calls.
**D is disabled in public API and benchmark calls because it has unresolved
issues.** The reports' D columns do not describe the release default. Explicit
`mid_window_blocks=None` restores the original scan. See the
[API guide](../integration.md#scan-order) for causal and low-level behavior.

The long-sequence measurements use synthetic BF16 Q/K/V, all three descales,
dense noncausal attention, D=128 and no skipping. They time one attention kernel,
excluding quantization and V packing. BF16 was measured in the same run.
Across the four shapes, fusedpipe improves speed by 1.39-2.20%; the additional D
effect ranges from -0.12% to +0.22% and is not a consistent long-sequence gain.

The earlier captured-input cohorts are separate experiments. Their private model
inputs and internal experiment files are not distributed. Neither kernel timing
nor matching FP8 scheduling outputs establishes end-to-end model speed or quality.

To benchmark the default scan with the existing explicit backend identity:

```bash
vc-attn-bench --backends bf16_ref fp8_ref vc_v4_mid4 --baseline bf16_ref \
  --shapes 32768x7x128 --output results/mid4.json
vc-attn-report results/mid4.json
```

This CLI command includes internal V packing in its attention scope and therefore
does not reproduce the reports' kernel-only timing boundary. Historical benchmark
names, including the original-scan `vc_v4`, retain their existing configurations.

[Documentation](../README.md) · [Repository](../../README.md)
