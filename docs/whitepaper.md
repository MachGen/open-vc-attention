# MachGen Attention on NVIDIA Blackwell

**VC Attention and FlashAttn V6**

Technical Whitepaper | Revision 1.0 | 30 September 2026
MachGen contributors | Software evaluated: vc-attention 0.1.2

This is an archived evaluation of the `scaled` snapshot. Current calls use
`v4` with mid-window 4 and eligible B200 fusedpipe; the experimental D patch is
disabled. See [the API guide](integration.md#scan-order) for current behavior.
The measurements and historical dispatch descriptions below are unchanged.

## Abstract

MachGen Attention provides two implementations of low-bit, forward attention
for NVIDIA Blackwell GPUs. VC Attention combines FP8 activations and scaled
ExpCast with a fused execution path written in CuTe DSL. FlashAttn V6 is
MachGen's native CUDA and inline-PTX implementation specialized for B300. Both
retain dense attention coverage in the configurations evaluated here.

The project recommends **VC Attention on B200** and **FlashAttn V6 on B300**.
For the long-sequence shape `[S,H,D]=[188214,7,128]`, VC takes 54.197 ms on B200
versus 99.214 ms for the pinned BF16 reference, a **1.831x speedup**. FlashAttn V6
takes 55.325 ms on B300 versus 86.059 ms for that GPU's BF16 reference, a
**1.556x speedup**. VC results on B300 are also reported. These recommendations
identify the project's starting configurations; measured rankings remain
shape-dependent.

This whitepaper explains the numerical data flow, implementation differences,
performance evidence and integration contract. Measurements cover the attention
operator on synthetic inputs. B200 uses an exclusive GPU allocation; B300 uses
CPU-precompiled kernels in monitored shared-GPU idle windows. Input quantization
and model execution are outside these timings. The results establish neither
end-to-end model speedup nor model-quality acceptance. Every performance table
uses **BF16 as its denominator**. [B200 record](benchmarks/b200-public-install.json),
[B300 record](benchmarks/b300-idle-windows.json).

## 1 Scope and implementation families

Dense attention is a significant part of long-sequence inference because each
query interacts with every key. For one self-attention sequence, arithmetic
work grows as O(S squared H D), where S is sequence length, H is head count
and D is head dimension. Tiling avoids materializing the full score matrix,
but probability formation, normalization, data movement and synchronization
still have to keep pace with matrix multiplication. This is the context for
the IO-aware tiled approach introduced by
[FlashAttention](https://arxiv.org/abs/2205.14135).

The repository focuses on the attention operator, input preparation and
measurement. It contains no model weights, request scheduler or application
services. MachGen's VC implementation builds on the
[VC-Attention paper](https://arxiv.org/abs/2609.15810) and retained upstream
FlashAttention CuTe code. It is an independent implementation, with additional
kernel engineering. FlashAttn V6 denotes MachGen's own native implementation;
upstream FlashAttention is a separate project.

| Property | VC Attention | FlashAttn V6 |
|---|---|---|
| Implementation | CuTe DSL with Triton preparation | Native CUDA and inline PTX |
| Hardware | B200 / SM100 and B300 / SM103 | B300 / SM103 only |
| Project recommendation | Default starting point for B200 | Default starting point for B300 |
| Probability path | Scaled ExpCast encoding | Approximate exp2 followed by E4M3 conversion |
| Main entry point | `attention` or `attention_fp8` | `NativePlan` with prepared FP8 tensors |
| Evaluated workload | Single-sequence, noncausal MHA, D=128 | Same operator workload |

The VC convenience API accepts BF16/FP16 tensors shaped `[S,H,128]` or
`[B,S,H,128]`, with equal Q/K/V head counts. It also accepts causal and
cross-attention configurations, but those configurations are outside the
performance tables in this paper. The native plan accepts a single
self-attention sequence and uses the standard 1/sqrt(128) softmax scale.
Both are inference interfaces. The public wrappers do not provide backward,
GQA/MQA, arbitrary masks, dropout or a paged KV-cache decoding backend.
See [the API contract](integration.md) and [the native interface](../src/vc_attn/native/README.md).

## 2 Numerical data flow

### 2.1 Attention and activation preparation

For each head, the target operation is scaled dot-product attention:

$$
O = \mathrm{softmax}(\sigma QK^T)V, \qquad \sigma = D^{-1/2}.
$$

The low-bit implementations approximate this operator. Q and K are quantized
per 128-token block per head; V is quantized per head over the sequence.
For each such group X, the preparation uses a positive FP32 descale d:

$$
d_X = \frac{\max(\max |X|, 10^{-12})}{448}, \qquad X_8 = \mathrm{E4M3}(X/d_X).
$$

The quantizer clamps scaled values to the E4M3 finite range before conversion.
Conceptually, X is reconstructed as d_X times X_8. Q/K descales have shape
`[B,H,ceil(S/128)]`; V descales have shape `[B,H]`. K uses its own sequence
length when Q and K lengths differ. Separate sequences must be quantized
independently so a scale group cannot cross a sequence boundary.
V's per-head descale is restored when the final output is normalized.
[Preparation source](../src/vc_attn/quantization.py).

The two inference paths share a preparation step:

```text
BF16 / FP16 Q, K, V
        |
        v
FP8 preparation + FP32 descales
        |
        +--> VC dispatch --> optional V packing --> scaled ExpCast attention
        |
        +--> FlashAttn V6 plan --> native FP8 attention on B300
        |
        v
BF16 output
```

Preparation produces new buffers. Reusing them is valid only while the
activations are unchanged. VC's `attention` includes preparation;
`prepare_fp8` followed by `attention_fp8` exposes the boundary. Native V6 uses
the same prepared tensors through a plan with reusable output storage.

### 2.2 Applying Q and K scales during the tile scan

Fix a query block with descale d_q and a key block j with descale d_kj.
Let R_j be the raw FP8 matrix-product result. Define a as the query descale
combined with the softmax scale and the conversion to base-2 exponentials:

$$
R_j = Q_8 K_{8,j}^{T}, \qquad a = \sigma \log_2(e) d_q.
$$

The logits in base-2 units are a times d_kj times R_j. The running row maximum
m is stored after applying K's descale but before applying a. With no rescale
deadband, the online update is:

$$
m_{new} = \max(m_{old}, d_{k,j}\max R_j), \qquad \alpha = 2^{a(m_{old}-m_{new})}.
$$

For each score, the centered base-2 exponent is:

$$
x_j = a(d_{k,j}R_j-m_{new}).
$$

This placement matters when K scales differ between blocks. Scaling only the
raw scores but comparing unscaled tile maxima gives the wrong running maximum.
Multiplying m_new by d_kj again applies the key scale twice. In the implementation,
`update_row_max` scales the tile maximum once, while probability encoding folds
d_kj into the score multiplier and keeps the row-max bias in the shared scale
space. [Softmax source](../src/vc_attn/_kernels/scaled/flash_attn/cute/softmax.py).

For an ideal online softmax, the output numerator and denominator are rescaled
by alpha when the maximum changes. New tile contributions are then added,
and final division normalizes the output. The implementations below use this
structure with different probability approximations and normalization policies.
Their numerical results need not be identical.

## 3 VC Attention

### 3.1 Scaled ExpCast

ExpCast uses an affine approximation in the positive E4M3 encoding space.
For the centered exponent x, the conceptual probability code is:

$$
c = \mathrm{clip}(\mathrm{round}(8x + 119.65), 0, 120).
$$

The byte c is interpreted as an E4M3 value, rather than converted numerically
from the integer c. The peak code is approximately 120, representing a value
of 256. This supplies a scaled approximation to the exponential. The normalization
must use the corresponding probability representation; the common scale is
removed by the final division.

This equation describes the numerical idea, not the literal instruction
sequence. The scaled implementation combines score scaling and the row-max bias
with FMA operations, clamps codes, and packs bytes using rounding and permutation
helpers. The ExpCast path uses a max offset of 8 and a zero rescale deadband.
It avoids a separate exponential followed by FP8 conversion for every score;
online rescaling can still use an exponential.
[Encoding source](../src/vc_attn/_kernels/scaled/flash_attn/cute/utils.py),
[softmax source](../src/vc_attn/_kernels/scaled/flash_attn/cute/softmax.py).

### 3.2 The fused path for large sequences

Scaled ExpCast enables the large fused path to consume externally quantized
Q/K/V with descales. The gain is a combination of changes in that path:

| Component | Implementation purpose |
|---|---|
| V packing | Supplies the K-major layout consumed by the fused Tensor Core path |
| Tensor Core denominator | Accumulates the normalizer with the probability/value work, using added constant columns |
| Inline output rescaling | Updates accumulated output when the running maximum changes |
| Segmented probability delivery | Makes portions of P available to the consumer without waiting for an entire probability tile |
| Warp and register scheduling | Coordinates score conversion, correction and output work within the CTA |

The constant-column denominator implements the identity P times a vector of
ones equals the row sum of P. Accumulating this with the PV work moves that
reduction into the Tensor Core data path. The output and normalizer still have
to be rescaled consistently as the scan advances.
[Kernel implementation](../src/vc_attn/_kernels/scaled/flash_attn/cute/flash_fwd_sm100.py).

For the default scaled FP8 call, necessary eligibility conditions include
one noncausal sequence, D=128, Q and K lengths at least 32768, S_q times H at
least 1048576, no LSE request, and CuTe DSL 4.6.0 on SM100 or SM103. The dispatcher
also checks layout, tiling and feature compatibility. `[73426,7,128]` fails the
size gate; the other shapes reported here pass it. Supported calls outside the
fused path use the general kernel path. [Dispatch source](../src/vc_attn/_kernels/scaled/flash_attn/cute/interface.py).

### 3.3 What the measured gain establishes

The BF16 comparison measures the combined effect of lower-precision arithmetic,
probability representation and execution scheduling. It is not a measurement
of ExpCast in isolation. Even a same-source comparison with ExpCast disabled
can change packing and fused dispatch, as well as the probability computation.
Attributing a fixed percentage to one instruction would require additional
controlled variants and profiling evidence.

The default evaluated path visits every QK and PV tile. V-Smooth, optional
sparsity, skip-softmax and skip-PV are outside this evaluation. Historical
experiments remain in the repository appendix and do not contribute to the
headline numbers in this paper.

## 4 FlashAttn V6 on B300

### 4.1 Native execution and memory movement

FlashAttn V6 is a separate MachGen CUDA implementation, built for `sm_103a`.
It uses Tensor Memory (TMEM) for matrix results, Tensor Memory Accelerator
(TMA) transfers for Q/K/V tiles, and explicit barriers for producer/consumer
coordination. A 512-thread CTA handles two 128-row query stages while scanning
128-token key/value tiles. The code assigns different warp groups to loading,
matrix multiplication, softmax and output correction.

The softmax stage loads score fragments and their maxima with
`tcgen05.ld.red...f32.max`. Combining the TMEM load with a reduction avoids a
separate software maximum tree for those loaded fragments. Tail tokens still
require explicit masking. The native code uses architecture-specific operations
and is shipped only for B300; the package does not include a B200 compatibility
port. NVIDIA documents the instruction in the
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-ld).
[Native source](../src/vc_attn/native/fa_fp8_v6.cu),
[TMEM helpers](../src/vc_attn/native/tmem_ops.cuh).

### 4.2 A different probability and normalization policy

V6 does not use VC's affine ExpCast encoding. Its softmax loop applies the Q/K
descales, evaluates `ex2.approx.ftz.f32`, and converts the resulting probabilities
with `cvt.rn.satfinite.e4m3x2.f32` for PV. Its exponent has an offset of 4,
corresponding to a common factor of 16. The row sum is accumulated from the
floating-point probabilities before their FP8 conversion, whereas PV consumes
the rounded FP8 values.

V6 also retains the old maximum when the proposed shift is within a four-unit
base-2 deadband. Beyond that threshold, it rescales the running numerator and
denominator. These policies differ from VC's zero-deadband ExpCast path and
help explain why accuracy must be measured separately for the two implementations.
The whitepaper makes no claim that either implementation is numerically exact.

### 4.3 Plan lifetime and compiler evidence

`NativePlan` owns tensor maps, output buffers and the loaded native library.
It consumes prepared FP8 tensors, runs on their CUDA device and returns a view
of reusable BF16 output. A later call overwrites that output; callers must clone
results they need to retain and rebuild the plan for new input tensors.
The recorded native build reported 128 registers with zero stack/spill, and
three B300 tail/descale operator tests passed. These are observations of the
recorded build, not guarantees for arbitrary compiler options.
[Native plan](../src/vc_attn/native_plan.py), [validation record](performance.md).

## 5 Evaluation protocol

### 5.1 Inputs and baseline

The evaluated shapes use S in `{73426,188214}`, H in `{7,56}` and D=128, with
one noncausal sequence and equal Q/K/V heads. The longest H=7 shape matches a
MiniMax production shape, but the tensors in these runs are seeded synthetic
BF16 inputs. The seed is `20260928 + S + H`. H=7 and H=56 therefore use different
inputs; all implementations within a shape and run use the same inputs.

Every speedup in this paper is calculated against `bf16_ref`, the repository's
fixed BF16 implementation at source revision
`2cae9072801704491b37f14037b4baa32c3958dc`. This is a pinned source reference,
not an unspecified current upstream installation. VC uses scaled source
revision `4245ca87a02a476e89b793a5541a1f0576684b01`. The native implementation
is identified by the file hashes in the
[source manifest](../src/vc_attn/source_manifest.json).

The raw runs also contain an ordinary-FP8 control. Their stored default report
baseline may be `fp8_ref`; this paper recomputes BF16 ratios from the raw paired
samples. It does not relabel the existing FP8 ratios. FlashAttn V6 is a candidate
implementation, not the denominator for VC.

### 5.2 Timing boundaries and environment

| Item | B200 | B300 |
|---|---|---|
| Measurement date in UTC | 28 September 2026 | 29 September 2026 |
| Hardware | NVIDIA B200 | NVIDIA B300 SXM6 AC |
| Compilation | Normal CuTe JIT before timing | CuTe/Triton CPU AOT before GPU work; native DSO prebuilt |
| GPU access | Exclusive scheduler allocation | Monitored idle windows on a shared GPU |
| Initial warmup | 3 seconds per backend | No separate multi-second warmup |
| Accepted rounds per shape | 12 | 12 |
| Warm calls per backend per round | 6 | 2 |
| Timed calls per backend per round | 10 | 3 |
| Timer | Eager CUDA events | Eager CUDA events |

Both runs use vc-attention 0.1.2, PyTorch 2.11.0+cu130, Triton 3.6.0,
CuTe DSL 4.6.0, quack 0.6.1 and cuda-python 13.0.3. No clock or power setting
was changed by the benchmark. The result files retain environment and source
metadata; sampled device state does not prove constant clocks throughout a run.

The timer includes each attention call as implemented, including VC's internal
V packing and output allocation. FP8 input preparation, compilation, warmup,
model execution and communication are excluded. V6 plan construction and
reusable-output setup are also excluded. This is an operator comparison of the
provided invocation paths, not a comparison with identical allocation behavior.

### 5.3 Paired samples and acceptance

Within each round, backend order is randomized. One sample is the elapsed time
of that backend's repeated calls divided by the repeat count. Reported latency
is the median of 12 such samples, and speedup is the ratio of the BF16 median
to the candidate median. This differs from the median of per-round ratios.

The B300 guard accepted 48 complete paired rounds across four shapes and
discarded 12 interfered windows/rounds. It watched activity and process-set
changes on the selected GPU, including delayed NVML utilization samples. No
partial pair or latency-based outlier filter contributes to the tables.
Resident model allocations remained present; the run records
`isolation_checked=false`. NVML sampling cannot establish exclusive isolation.

CPU preparation exported four CuTe attention libraries and eight Triton
specializations. The public helper reproduced the four attention libraries
byte-for-byte, with matching dispatch/ABI metadata. Its small oracle/smoke
passed, but a separate large-shape smoke reached its time limit. A longer-warmup
retry for `[73426,7,128]` also remained incomplete. Neither partial attempt
contributes performance samples. [AOT guide](../tools/idle_benchmark/README.md),
[measurement limits](performance.md).

## 6 Performance against BF16

### 6.1 VC Attention on B200

| Shape `[S,H,D]` | BF16 ms | VC ms | Speedup vs BF16 |
|---|---:|---:|---:|
| 188214,7,128 | 99.214 | 54.197 | 1.831x |
| 188214,56,128 | 795.860 | 432.280 | 1.841x |
| 73426,7,128 | 15.147 | 10.843 | 1.397x |
| 73426,56,128 | 121.581 | 69.217 | 1.757x |

### 6.2 VC Attention on B300

| Shape `[S,H,D]` | BF16 ms | VC ms | Speedup vs BF16 |
|---|---:|---:|---:|
| 188214,7,128 | 86.059 | 51.356 | 1.676x |
| 188214,56,128 | 692.213 | 410.089 | 1.688x |
| 73426,7,128 | 11.848 | 10.110 | 1.172x |
| 73426,56,128 | 105.654 | 64.734 | 1.632x |

### 6.3 FlashAttn V6 on B300

FlashAttn V6 is shown as a separate implementation with its own BF16 comparison.

| Shape `[S,H,D]` | BF16 ms | FlashAttn V6 ms | Speedup vs BF16 |
|---|---:|---:|---:|
| 188214,7,128 | 86.059 | 55.325 | 1.556x |
| 188214,56,128 | 692.213 | 443.996 | 1.559x |
| 73426,7,128 | 11.848 | 7.434 | 1.594x |
| 73426,56,128 | 105.654 | 67.517 | 1.565x |

At `[188214,7,128]`, the recommended B200 VC configuration reduces attention
time by 45.4% relative to BF16. The recommended B300 V6 configuration reduces
it by 35.7%. These percentages are `1 - candidate_median / BF16_median`;
latency reduction and speedup are different quantities.

The B300 data show that the ranking between VC and V6 varies by shape. The
project's B300 recommendation does not imply that V6 is faster on every row.
The exclusive B200 and shared-window B300 protocols also differ in compilation,
warmup and repeat count, so the tables do not establish a controlled hardware
speedup between B200 and B300. Raw records remain the authority for the values:
[B200](benchmarks/b200-public-install.json), [B300](benchmarks/b300-idle-windows.json).

## 7 Numerical validation and evidence limits

Relative L2 error is measured against the pinned BF16 output across the complete
output tensor:

$$
\epsilon_{rel} = \frac{\|O_{candidate}-O_{BF16}\|_2}{\max(\|O_{BF16}\|_2,10^{-12})}.
$$

The norm is equivalent to the Frobenius norm after flattening all output
positions, heads and channels. Both implementations produced finite outputs.

| Shape `[S,H,D]` | B200 VC relative L2 | B300 VC relative L2 | B300 V6 relative L2 |
|---|---:|---:|---:|
| 188214,7,128 | 5.665% | 5.665% | 5.422% |
| 188214,56,128 | 5.676% | 5.676% | 5.432% |
| 73426,7,128 | 5.515% | 5.515% | 5.287% |
| 73426,56,128 | 5.655% | 5.655% | 5.421% |

VC's errors fall between 5.52% and 5.68% on both GPUs; V6's B300 errors fall
between 5.29% and 5.43%. These values include activation quantization and the
respective attention implementation. They do not isolate ExpCast error. The
raw records also retain RMSE and maximum absolute error.

The recorded B200 package validation passed eight GPU tests, including small
FP32-reference comparisons, repeatability, tail tokens, batch/packed isolation,
graph replay after input changes and the opt-in large fused path. Three native
tests are skipped on B200 and passed separately on B300 for S=1, 129 and 1025.
CPU tests and source audits check packaging and contracts; they cannot establish
GPU numerical correctness.

Synthetic Gaussian inputs provide reproducible operator comparisons, but do
not characterize model-specific outliers or quality after repeated attention
layers. Full SGLang/MiniMax generation, distributed communication and model
quality have not been validated for this release. Exclusive B300 confirmation
and complete model-level evaluation remain separate work.

## 8 Integration and reproduction

### 8.1 Calling VC and FlashAttn V6

Use the [installation guide](installation.md) to create the tested environment.
For VC, start with a direct local attention call:

```python
import torch
from vc_attn import attention

q, k, v = [torch.randn(4096, 8, 128, device="cuda", dtype=torch.bfloat16)
           for _ in range(3)]
with torch.inference_mode():
    output = attention(q, k, v)
```

For V6, build the native library with CUDA 13 and use a B300 at runtime:

```bash
python tools/build_native.py --arch sm_103a --output build/libvc_native_v6.so
```

```python
from contextlib import closing
from vc_attn import prepare_fp8
from vc_attn.native_plan import NativePlan

prepared = prepare_fp8(q, k, v)
with closing(NativePlan(prepared, "build/libvc_native_v6.so")) as plan:
    output = plan().clone()
```

The top-level `attention()` call selects VC on either supported GPU; it does
not automatically route B300 to V6. Rebuild preparation for changed activations.
The native wrapper's output lifetime and single-sequence contract remain part
of the integration API.

### 8.2 Recomputing and collecting operator measurements

The published tables can be regenerated without a GPU:

```bash
vc-attn-report docs/benchmarks/b200-public-install.json --baseline bf16_ref
vc-attn-report docs/benchmarks/b300-idle-windows.json --baseline bf16_ref
vc-attn-report docs/benchmarks/b300-idle-windows.json \
  --baseline bf16_ref --candidate native_v6
```

For an exclusive B200 run, keep the original comparison set and change only
the reporting denominator to BF16:

```bash
vc-attn-bench --shapes 73426x7x128 188214x7x128 73426x56x128 188214x56x128 \
  --backends bf16_ref fp8_ref vc_scaled --baseline bf16_ref \
  --rounds 12 --repeats 10 --warm-calls 6 --warm-seconds 3 \
  --scope attention --timing events --output results/b200-bf16.json
```

To follow the B300 CPU-AOT protocol, use the exact commands in the
[idle-window guide](../tools/idle_benchmark/README.md), including preparation
before GPU work and complete-round acceptance. An exclusive B300 run through
the general CLI is useful validation, but has a different execution protocol
and must be reported separately. Use `--scope quantize-attention` when evaluating
VC's full preparation cost; native plans currently support attention-only timing.

### 8.3 Integrating into an inference pipeline

Replace the model's local attention call after Q/K normalization, positional
transforms and any input sequence-parallel exchange. Preserve the model's scale,
mask semantics, sequence boundaries and output exchange. The opt-in SGLang
diffusion adapter provides one example with packed-sequence routing and a
MiniMax-H3 configuration. It currently calls the VC API; V6 is exposed through
its native plan, not through that adapter. [Integration guide](integration.md),
[SGLang example](sglang.md).

For end-to-end evaluation, hold checkpoint, seeds, precision, caching, offload,
sequence-parallel configuration and output settings fixed. Measure warm request
latency separately from model loading, and inspect matched outputs. If attention
accounts for fraction f of baseline runtime and its speedup is g, the idealized
end-to-end speedup is `1 / ((1-f) + f/g)` before new preparation costs. Neither f
nor an end-to-end speedup is measured by the operator tables in this paper.

## 9 Reproducibility and attribution

This document summarizes existing completed measurements; it introduces no new
GPU run. The evaluated package is 0.1.2. The repository snapshot used to prepare
this whitepaper is `53b2d3c2cf3c492e98b28a5c16976484f1549511`. Kernel identity is
pinned independently by the source manifest and by hashes in each result file.
The exact records used for all numerical tables are listed below.

| Record | SHA-256 |
|---|---|
| `b200-public-install.json` | `5f93a7a34e5cb433b4ab4f13be0f9a38fbc0248d329088020dd36df9708bde79` |
| `b300-idle-windows.json` | `d43dc6d78d40f46c000853c2477d7488836f0a1d82e09b42e342c971c8c584f2` |

Frozen kernels retain original and namespace-adjusted hashes. BSD-3-Clause
headers, author attribution and third-party notices are preserved. SGLang patch
context retains Apache-2.0 attribution. See [LICENSE](../LICENSE),
[NOTICE](../NOTICE), [AUTHORS](../AUTHORS) and [CITATION.cff](../CITATION.cff).
Historical VC variants and optional experiments are documented in the appendix;
this whitepaper focuses on the main VC implementation and native FlashAttn V6.

## References

1. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
2. Li et al. *VC-Attention: Value Smoothing and Softmax Casting for Low-bit Attention*. 2026. [arXiv:2609.15810](https://arxiv.org/abs/2609.15810). This whitepaper describes MachGen's independent implementation and measurements.
3. NVIDIA. *Parallel Thread Execution ISA*, fifth-generation Tensor Core instructions and Tensor Memory loads. [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-ld).
4. MachGen contributors. *VC Attention source, benchmark protocol and raw measurements*. [Repository](https://github.com/MachGen/vc-attention), [protocol](benchmarking.md), [source manifest](../src/vc_attn/source_manifest.json).

[Repository and quickstart](../README.md) | [Documentation](README.md)
