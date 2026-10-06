# Appendices

Historical implementations and optional experiments live outside the main
quickstart. The recommended API and default benchmark select scaled ExpCast VC.

- [Version behavior](versions.md): exact source revisions, historical v1/v2/v3, the scaled dispatch gate and baseline identities.
- [Initial validation](initial-validation.md): short regression and graph samples from the first extraction.
- [B300 scan-order experiment](b300-scan-order.md): explicit mid-window 4, its 1.81x BF16 comparison and why the denominator matters.
- [Historical B200 development](b200-development.md): the archived 1.95x result, separate V-Smooth optimization and their original samples.
- [Frozen source directories](../../src/vc_attn/_kernels/README.md): code and per-directory guides for each retained snapshot.

Historical VC backends remain available for regression work, but are opt-in:

```bash
vc-attn-bench --shapes 188214x7x128 \
  --backends bf16_ref fp8_ref vc_v1 vc_v2 vc_v3 vc_scaled \
  --baseline fp8_ref --output results/history.json
```

`vc_scaled_mid4`, `vc_nvfp4` and `vc_vsmooth` are additional experiments.
Their preparation, precision and scan-order settings must be reported separately.
See [benchmark protocol](../benchmarking.md) for supported timing combinations.

[Repository](../../README.md) · [Parent directory](../README.md)
