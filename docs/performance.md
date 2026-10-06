# Performance and validation

The tables below are archived measurements of `scaled` at `4245ca87a`. The
default API now selects `v4` at `8aa761eac`; historical backends remain available.
See [upstream synchronization](upstream-sync.md) for the new snapshot's validation.

The recommended configuration is scaled ExpCast VC with the default scan,
per-128-token-block Q/K scales, per-head V scales, no LSE, and no softmax/PV
skipping. All measurements below use MHA, D=128 and one varlen sequence.

## B200: standalone wheel, full benchmark protocol

Run with the 0.1.2 standalone wheel in a fresh public-dependency virtual environment on one scheduler-reserved NVIDIA B200.
Torch 2.11.0+cu130, Triton 3.6.0, CuTe DSL 4.6.0, quack 0.6.1,
cuda-python 13.0.3. No clock or power configuration changes.
Synthetic BF16 input seed is `20260928 + S + H`; H=7 and H=56 are independent inputs.

Three seconds of initial warmup per backend, 12 randomized paired rounds,
6 warm calls and 10 timed calls per backend per round. Eager CUDA-event timing
includes output allocation/device work and internal V packing; Q/K/V
quantization, compilation, warmup, model and communication are excluded.
Reported latency is the median of round means. Ratios use medians from the same
run, not the median paired ratio; both statistics are stored in JSON.

| S | H | BF16 ms | Ordinary FP8 ms | VC ms | VC vs BF16 | VC vs FP8 |
|---:|---:|---:|---:|---:|---:|---:|
| 73426 | 7 | 15.147 | 11.922 | **10.843** | **1.397x** | **1.100x** |
| 188214 | 7 | 99.214 | 79.375 | **54.197** | **1.831x** | **1.465x** |
| 73426 | 56 | 121.581 | 96.813 | **69.217** | **1.757x** | **1.399x** |
| 188214 | 56 | 795.860 | 633.943 | **432.280** | **1.841x** | **1.467x** |

Data: [complete raw samples and environment](benchmarks/b200-public-install.json).
Every row completed all rounds and passed finite-output checks. The benchmark
checked compute-process ownership around each variant, with an external
foreign-PID guard polling every 0.5 seconds; no interference was detected.

Relative L2 versus the pinned BF16 reference ranges from **5.29–5.43%**
for ordinary FP8 and **5.52–5.68%** for VC on these synthetic inputs.
The raw JSON includes per-shape RMSE and maximum absolute error.

Reproduce after reserving a GPU:

```bash
vc-attn-bench --shapes 73426x7x128 188214x7x128 73426x56x128 188214x56x128 \
  --backends bf16_ref fp8_ref vc_scaled --baseline fp8_ref \
  --rounds 12 --repeats 10 --warm-calls 6 --warm-seconds 3 \
  --scope attention --timing events --output results/b200-main.json
```

## B300: standalone wheel, CPU AOT and monitored idle windows

Measured on **2026-09-29** with the clean 0.1.2 public-dependency installation,
NVIDIA B300 SXM6 AC, Torch 2.11.0+cu130, Triton 3.6.0, CuTe DSL 4.6.0,
quack 0.6.1 and cuda-python 13.0.3. The fixed BF16/ordinary-FP8 source is the
same as B200; original SM103 native v6 is a third denominator.

Four CuTe attention libraries and eight Triton specializations were compiled on
CPU before GPU execution. The native v6 DSO was also built beforehand. Kernel
bodies and configurations are unchanged; the CuTe functions use the AOT
fake-tensor/TVM-FFI ABI. The published CPU helper rebuilt all four attention
libraries byte-for-byte. This measures the AOT path, not first-use JIT behavior.

The sampler kept one physical GPU and seeded synthetic inputs, using 12 complete
randomized paired rounds per shape, two warm calls and three timed calls per
backend, without a separate multi-second warmup. CUDA events include VC's
internal V packing. Quantization and native plan creation/output-buffer setup
are outside timing. No model or communication is included.

| S | H | BF16 ms | Ordinary FP8 ms | Native v6 ms | VC ms | VC vs BF16 | VC vs FP8 | VC vs v6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 188214 | 7 | 86.059 | 57.950 | 55.325 | 51.356 | 1.676x | 1.128x | 1.077x |
| 188214 | 56 | 692.213 | 468.977 | 443.996 | 410.089 | 1.688x | 1.144x | 1.083x |
| 73426 | 7 | 11.848 | 9.205 | 7.434 | 10.110 | 1.172x | 0.910x | 0.735x |
| 73426 | 56 | 105.654 | 71.118 | 67.517 | 64.734 | 1.632x | 1.099x | 1.043x |

All 48 accepted rounds passed the selected-GPU idle-window guard. The run
discarded **12** interfered windows/rounds, kept their rejection reasons,
and retried without accepting partial pairs or filtering by latency. It waited
for stable idle, polled foreign activity/process-set changes every 0.1 seconds,
and checked delayed NVML samples four seconds after each window. All four
backends produced finite outputs; the raw record includes full-output errors
against BF16 and the initial small FP32-oracle check. Relative L2 is **5.29–5.43%**
for ordinary FP8 and native v6, and **5.52–5.68%** for VC on these synthetic inputs.

This is **shared-GPU sampling with resident model allocations**, recorded as
`isolation_checked=false`. NVML has finite sampling resolution; this does not
prove exclusive isolation. Brief warmup particularly affects short kernels.
The B200 run uses an exclusive allocation, a different warmup/repeat count and
normal JIT dispatch: do not interpret cross-card absolute latencies as a
controlled B200-to-B300 hardware speedup.

At S=73426/H=7 the large-path size gate fails. The reported VC latency is slower
than ordinary FP8 and native v6 in this run. Preserve this result when assessing
backend selection; an optimization is not a win for every shape. The targeted
longer-warmup follow-up is recorded separately below.

Data: [complete paired samples, AOT hashes and guard metadata](benchmarks/b300-idle-windows.json).
Reproduce with [CPU compilation and idle-window sampling](../tools/idle_benchmark/README.md).
Use `vc-attn-report` with `--baseline bf16_ref`, `fp8_ref` or `native_v6` to
recompute each denominator from the same raw samples.

### Short-shape warmup and batch-length follow-up

A separate S=73426/H=7 retry used six warm calls and ten timed calls per
backend, with the same AOT libraries, GPU, seed and idle-window guard.
The 12-minute guard budget expired before a complete 12-round comparison,
so the retry is marked `partial` and contributes no published latency or
speedup. The main table remains the original complete two-warm/three-timed
run. A longer-batch confirmation still requires a suitable testing window.

## Historical B300 measurement with source provenance

This historical B300 result was measured on **2026-09-26**, using captured
MiniMax-H3 Q/K/V and the original scaled source. It is **not a rerun of the
standalone wheel**. Earlier interrupted standalone attempts remain excluded;
the fresh complete AOT run above has its own protocol and record.

The recorded revision is `4245ca87a02a476e89b793a5541a1f0576684b01`.
All **31 shared CuTe source-file hashes** recorded in that report match the
original pre-namespace-rewrite hashes of this repository's scaled snapshot.
That establishes the kernel-source identity; it does not validate the new
wrapper or change the original measurement boundary. The public JSON retains
the raw samples, software versions, source-report hashes and matched file hashes.
Private capture tensors, host identities and filesystem paths are omitted.

NVIDIA B300 SXM6 AC, Torch 2.11.0+cu130, CuTe DSL 4.6.0, quack 0.6.1.
Attention-only CUDA events; internal V packing included, quantization excluded.
Five seconds initial warmup, 12 accepted paired rounds, 6 warm calls and 10 timed
calls per round. This was a shared GPU with resident model allocations: idle
windows were screened for overlapping work using NVML process utilization,
including delayed utilization samples. Four overlapping rounds were discarded.
This is weaker isolation than the B200 scheduler-exclusive allocation.

| S | H | Upstream BF16 ms | FP8 control ms | VC ms | VC vs BF16 | VC vs FP8 control |
|---:|---:|---:|---:|---:|---:|---:|
| 73426 | 56 | 106.172 | 70.183 | **61.393** | **1.729x** | **1.143x** |

This historical measurement uses `mid_window_blocks=None`. VC relative L2 versus upstream
BF16 is **2.845%**, FP8 control is **2.846%**; both have finite outputs.
No additional B300 shapes are claimed from this record.
Data: [sanitized samples and provenance](benchmarks/b300-recorded.json).
The explicit mid-window 4 experiment is documented in the
[scan-order appendix](appendix/b300-scan-order.md).

To measure the same backend identities on a reserved B300, use
`--backends upstream_bf16 fp8_control vc_scaled --baseline fp8_control`.
Install `.[upstream]` and use your own representative `--input` capture to match
the input distribution. The private historical capture is not distributed, so
synthetic-input reruns should not be expected to reproduce its exact latency
or error. The full [protocol](benchmarking.md) defines supported input files.

## Exact baseline identities

| Name | Source | Meaning |
|---|---|---|
| `bf16_ref` | `2cae9072801704491b37f14037b4baa32c3958dc` | Fixed BF16 source reference used in both standalone GPU runs |
| `fp8_ref` | Same fixed revision | Ordinary FP8 with descales; both standalone GPU runs |
| `upstream_bf16` | `flash-attn-4==4.0.0b21` | Historical B300 BF16 denominator; exact interface hash recorded |
| `fp8_control` | `4245ca87a02a476e89b793a5541a1f0576684b01` | Same scaled source as VC, external descales, ExpCast off; historical B300 FP8 denominator |
| `vc_scaled` | Same scaled revision | ExpCast on, eligible fused dispatch; recommended implementation |
| `native_v6` | Original CUDA/inline-PTX source, hashes in manifest | Separate B300 / SM103-only FP8 baseline, measured in the fresh B300 table |

The standalone B200 and B300 runs use the same pinned BF16/FP8 source identities.
The historical B300 report uses different denominators and captured inputs.
Warmup, compilation and isolation also differ across records; do not infer
relative hardware performance from their absolute latencies.

The scaled dispatch admits external descales into the fused path. V packing,
Tensor Core accumulation of the normalizer, inline rescaling and segmented
probability publication work together. At S=73426/H=7 the size gate fails and
the established path runs. The other listed shapes pass the size gate on both GPUs.
The data do not isolate exp2 instruction cost. Implementation details and
historical comparisons are in the [appendix](appendix/versions.md).

## Packaged validation and limits

- CPU contracts: 11 passed.
- B200 0.1.2 wheel: **8 passed, 3 skipped**, including large fused dispatch, FP32-reference comparisons, tails, repeatability, batch/packed isolation and graph replay after input changes. Three native tests require B300 and a local SM103 DSO.
- The wheel ran with an import guard rejecting platform namespaces; none loaded. Frozen-source hashes and attention-only packaging were audited.
- Original native v6 compiled with CUDA 13: 128 registers, zero stack/spill. Its three B300 native tests passed (S=1/129/1025, H=7); the fresh B300 table includes its matched timings.
- The SGLang registration patch and adapter operator math were checked. Full SGLang initialization, MiniMax generation, distributed communication and model-quality validation have not been run for this release.

Operator errors and finite outputs are not model-quality acceptance. Validate
matched model outputs and report end-to-end generation latency separately from
these attention measurements. Earlier short regression and experimental-path
runs are retained in [initial validation](appendix/initial-validation.md).

[Repository](../README.md) · [Documentation](README.md)
