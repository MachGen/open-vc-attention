# B200 results

`comparison.json` holds the raw paired samples, medians, speedups and accuracy of BF16, VC and Open-VC for [performance](../../../docs/performance.md). `repair.json` holds Open-VC's V residual repair budgets on the same input. Both were assembled by [`tools/reports/records.py`](../../../tools/reports/records.py) from `open-vc-attn-bench` runs with CUDA graph timing, made from a clean checkout. Each record keeps the runs' source provenance: per-file SHA-256 of every package source (kernels included), a combined package hash and the git commit. `python tools/reports/records.py check` validates the records against what the report build reads.

`numerical-boundary-validation.json` is a numerical diagnosis of FP8 error on a BF16-sensitive layer (step 48 / layer 49 of the same model), with an FP64 reference and Q/K-only and V-only quantization controls; see [performance](../../../docs/performance.md#sensitive-layers). It contains no timing results.
