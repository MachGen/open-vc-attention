# Benchmark protocol

`vc-attn-bench` runs every requested backend on the same GPU and input in one
process. The default backends are `bf16_ref`, `fp8_ref` and `vc_v4_mid4`.
The VC candidate matches the public API's dense mid-window-4 scan, with
source-level fusedpipe on eligible B200 calls and D disabled. Historical
`vc_v4` and `vc_scaled` retain their original-scan configurations.
The default baseline is `fp8_ref`; name it explicitly in shared reports.
The baseline must appear in the backend list. Missing dependencies and failed
backends stop the run rather than silently reducing the comparison set.

## Choose a workload

Shapes are written as `SxHxD`: sequence length, head count and head dimension.
The CLI currently measures single-sequence, noncausal self-attention with equal
Q/K/V shapes and D=128. The API also accepts other configurations, but these
commands do not benchmark causal attention, cross-attention or batches.

```bash
vc-attn-bench --shapes 4096x8x128 32768x32x128 \
  --baseline fp8_ref --output results/attention.json
vc-attn-report results/attention.json
```

For model-specific shapes, `--preset minimax` selects S=32768/73426/188214,
H=7/56, D=128. A preset overrides `--shapes`; choose one input method.
Synthetic inputs are the default. Use a local capture as described below when
the model's activation distribution matters.

## Timer boundaries

| Scope | Includes | Excludes |
|---|---|---|
| `attention` | Attention call, output allocation, automatic V packing | FP8 preparation, compilation, warmup, model, communication |
| `quantize-attention` | New quantization/preparation on every call plus attention | Compilation, warmup, model, communication |

Both scopes are operator measurements. Neither is model/request E2E latency.
The native v6 plans support `attention` only. Experimental NVFP4 uses per-16
Q/K scaling and direct E4M3 V conversion; V-Smooth has its own residual scales.
Their input preparation differs from the ordinary FP8 experiment and is labeled.

CUDA events time repeated eager calls by default. `--timing graph` captures one
call per backend after compilation and warmup, then times graph replays. CPU
enqueue overhead and gaps differ between these methods; compare like with like.
Outputs and graph objects remain alive through measurement.

Frozen V-Smooth preparation contains a host-to-device scalar write, and NVFP4
copies a host lookup table. Neither preparation can be captured by the tested
PyTorch version. For `vc_vsmooth` and `vc_nvfp4`, use eager events when timing
preparation, or capture only `--scope attention`. The harness
rejects this unsupported combination before allocation; it does not silently
exclude grouping cost or change the frozen snapshot.

The default is 3 seconds of initial warmup per backend, 12 randomized paired
rounds, 6 warm calls and 10 timed calls per round. For a longer
10-second warmup protocol, pass `--warm-seconds 10`. No clock or power settings
are changed. Every round stores its order and raw samples.

`ratio_of_medians` is baseline median / candidate median. `median_paired_ratio`
is the median of same-round ratios. Both are saved under distinct names;
latency reduction is `1 - candidate_median / baseline_median`. Do not mix runs
or average different shapes without specifying the weighting.

## Inputs and correctness

Default inputs are seeded synthetic BF16 tensors, not a private MiniMax capture.
`--preset minimax` selects S=32768/73426/188214, H=7/56, D=128. These are shapes,
not a bundled model or proof of model-distribution representativeness.

For your own inputs, save `{"q": q.cpu(), "k": k.cpu(), "v": v.cpu()}` as a
tensor-only `.pt` file and pass `--input file.pt --shapes SxHx128`. Loading uses
`weights_only=True`; exact shape matching is required. Captures and filesystem
paths are excluded from the result JSON. Only a content hash is retained.

Every backend is checked for finite output and full-output relative L2, RMSE
and maximum absolute error versus `bf16_ref`. This is a comparison, not an
automatic quality pass. GPU tests additionally use a small FP32 dense oracle,
check tail tokens/zero inputs/batch boundaries and repeatability. Approximate
operator tolerances are not a video-quality or deployment acceptance threshold.

## Isolation and reproducibility

Reserve an exclusive GPU, then set `CUDA_VISIBLE_DEVICES` normally. The tool
checks for foreign compute PIDs before and after each timed variant. It does
not stop any process or alter a scheduler. PID checks are not a replacement for
an allocation and cannot rule out every transient or non-compute workload.
`--allow-shared-gpu` explicitly disables the check and marks the run non-isolated.

Only `status=complete` reports with the intended isolation and accuracy are
candidates for publication. Interrupted/failed runs retain `status=invalid` and
must not enter performance tables. JSON includes source revisions, installed
versions, CUDA runtime, GPU type, settings, raw samples, and ownership-check count.
New runs also record the configured scan window and disabled D policy for the
ordinary packaged backends, UTC timestamps, runtime/manifest hashes and start/end
driver, clock, power-limit and P-state snapshots. Snapshots describe the sampled
state and do not prove fixed clocks throughout the run.

For permitted testing on a shared B300, the optional
[CPU compilation and idle-window tools](../tools/idle_benchmark/README.md) compile
before allocating GPU work and retry complete paired rounds after interference.
They record `isolation_checked=false`; use the explicit shared-window label when
reporting those results. This protocol uses shorter warmup and timed batches and
must not be presented as the default exclusive protocol.

## Native and upstream references

Original native v6 is **B300 / SM103 only**. Build with CUDA 13 on a system
with an SM103-capable toolkit, then run on an allocated B300:

```bash
python -m pip install -e '.[upstream]'
python tools/build_native.py --arch sm_103a --output build/libvc_native_v6.so
VC_ATTN_NATIVE_LIBRARY="$PWD/build/libvc_native_v6.so" \
  python -m pytest tests/test_gpu.py -k native
vc-attn-bench --shapes 73426x7x128 188214x7x128 \
  --backends upstream_bf16 bf16_ref fp8_ref native_v6 vc_scaled \
  --baseline native_v6 --native-library build/libvc_native_v6.so \
  --output results/native-comparison.json
```

Native v6 uses SM103-only `tcgen05.ld.red`; the wrapper rejects other
architectures. It has a distinct numerical policy and is not `fp8_ref`.
The result records the loaded library SHA-256. The [B300 performance table](performance.md) contains a matched standalone
comparison collected with the separately described CPU-AOT idle-window protocol.

To compare the ExpCast setting within the current source, use `fp8_v4`
and `vc_v4`. This differs from the pinned ordinary-FP8 reference. Enabling
ExpCast can also select the fused packing/normalization/scheduling path, so this
comparison measures the full configuration change, not ExpCast arithmetic alone:

```bash
vc-attn-bench --shapes 73426x56x128 \
  --backends upstream_bf16 fp8_v4 vc_v4 --baseline fp8_v4 \
  --output results/current-source.json
```

Use `--input` with representative tensors to reproduce an input distribution;
the default synthetic input is not a replacement for the historical capture.
See [published performance](performance.md) for exact denominator identities,
and [the appendix](appendix/README.md) for optional historical comparisons.
Do not use profiler replay latency as normal benchmark latency. Record NCU
metrics separately when attributing a gain to a specific kernel mechanism.

## Read a result without a GPU

The benchmark prints the result table automatically. Recreate it from raw
samples, selecting a different measured baseline if needed:

```bash
vc-attn-report results/minimax.json
vc-attn-report results/minimax.json --baseline bf16_ref
vc-attn-report results/native-comparison.json --baseline native_v6
```

This CPU-only command accepts completed standalone benchmark JSON. It recomputes
medians from samples, requires complete paired sample counts, and rejects missing
backends or invalid runs. It labels shared-GPU reports explicitly. It does not
merge different runs or convert the imported historical B300 schema. Candidate
selection prefers `vc_v4_mid4`, then historical `vc_v4` / `vc_scaled`, then the
last measured backend. Use `--candidate` to select another measured route.
