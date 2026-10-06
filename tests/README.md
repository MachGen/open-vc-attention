# Test suites

| Location | What it checks | Requirements |
|---|---|---|
| [cpu/](cpu/README.md) | Backend/shape contracts, statistics, packed boundaries, CLI/report failures, diagnostics, lazy imports and source provenance | Installed package; no CUDA dependencies needed for this suite |
| [test_gpu.py](test_gpu.py) | FP32-reference comparisons, repeatability, batch/packed isolation, graph replay, large scaled path and optional native tails/scales | Compatible Torch/CUDA stack and an allocated supported Blackwell GPU |
| [test_v4_gpu.py](test_v4_gpu.py) | Default B200 mid4 packing and benchmark/API parity without D; original-scan packing, LSE fallback, scaled-snapshot parity and SM103 ordinary-FP8 regressions | Blackwell, DSL 4.6.0; set `VC_ATTN_TEST_LARGE=1` for large cases |

Run from the repository root after installing development dependencies:

```bash
python -m pytest tests/cpu
python -m pytest tests/test_gpu.py tests/test_v4_gpu.py
VC_ATTN_TEST_LARGE=1 python -m pytest tests/test_gpu.py tests/test_v4_gpu.py -m large
```

The large case is opt-in. Native cases are skipped unless
`VC_ATTN_NATIVE_LIBRARY` names a locally built SM103 DSO on a B300; see
[native build instructions](../src/vc_attn/native/README.md).

Numerical tolerances describe operator checks, not video-quality acceptance.
Performance measurement belongs to [the benchmark CLI](../docs/benchmarking.md),
and executed coverage is in [Performance](../docs/performance.md).

[Repository](../README.md)
