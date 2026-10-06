# Recorded benchmark samples

These files contain timing samples and measurement metadata, not activation
tensors or model weights. The main B200 and B300 records have different origins;
read [Performance](../performance.md) before comparing them.

| File | Scope and coverage |
|---|---|
| [b200-public-install.json](b200-public-install.json) | Archived 0.1.2 wheel, clean public dependencies, full 12-round protocol; S=73426/188214, H=7/56 |
| [b200-main.json](b200-main.json) | Archived 0.1.1 full 12-round standalone comparison on the same four shapes |
| [b200-historical-expcast.json](b200-historical-expcast.json) | Archived 2026-09-23 captured-input ExpCast experiment; eight rounds, explicit mid-window 4, source of the historical 1.95x headline |
| [b200-historical-combined.json](b200-historical-combined.json) | Separate archived V-Smooth plus ExpCast optimization; six rounds, preprocessing excluded |
| [b300-idle-windows.json](b300-idle-windows.json) | Fresh 0.1.2 CPU-AOT B300 comparison, four shapes, 12 paired rounds each, shared-GPU idle-window protocol; includes native v6 |
| [b300-recorded.json](b300-recorded.json) | Sanitized historical source report from 2026-09-26; S=73426, H=56; default scan plus an appendix experiment, raw samples and 31 matched source hashes |
| [b200-events.json](b200-events.json) | Archived short comparison including historical VC snapshots; the same four B200 shapes |
| [b200-graph.json](b200-graph.json) | Archived quantization-plus-attention CUDA Graph validation; S=4096/188214, H=7 |
| [b200-vsmooth-events.json](b200-vsmooth-events.json) | Archived eager V-Smooth including preparation; S=4096, H=7 |
| [b200-optin-graph.json](b200-optin-graph.json) | Archived V-Smooth/NVFP4 attention-only graph validation; S=4096, H=7 |

For standalone CLI results, read `status`, `baseline`, `scope`, `timing`,
`versions` and `settings` first. Each `shapes` entry contains raw samples,
randomized orders, accuracy and speedups. `ratio_of_medians` and
`median_paired_ratio` are different summaries. Compare within a matched run.

The historical B300 export deliberately has a different schema. Read
`record_kind`, `standalone_wheel_rerun`, `baseline_definitions`, `isolation`
and `provenance`; its `default_scan` supplies the historical row in the performance writeup.
`appendix_mid_window_4` is a separate experiment in that historical record,
not its default configuration. Current public API defaults are documented separately.
Its FP8 control is not the standalone B200 run's `fp8_ref`.

The archived short runs are described in [initial validation](../appendix/initial-validation.md).
Use [the protocol](../benchmarking.md) for new measurements. Save local results
in the ignored root `results/` directory; publish only reviewed, sanitized,
complete records with an explicit measurement boundary.

[Repository](../../README.md) · [Parent directory](../README.md)
