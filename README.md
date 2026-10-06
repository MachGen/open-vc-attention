# VC Attention

Low-bit attention inference for **NVIDIA B200 and B300**, with a PyTorch API,
reproducible benchmarks and optional framework adapters.

VC Attention computes scaled dot-product attention using FP8 Q/K/V and
scaled ExpCast for softmax probabilities. The default call accepts BF16/FP16
tensors, prepares the FP8 inputs and returns BF16 output. It evaluates every
attention tile by default, with no softmax or PV skipping.

The library is optimized for long, dense sequences. It can be used directly in
PyTorch or integrated at a model's attention layer; it has no model or serving
framework dependency. Performance depends on shape and dispatch, and low-bit
results are approximate. See [measured performance](#performance) and
[how it works](docs/design.md).

Read the [technical whitepaper](docs/whitepaper.md) for the VC and FlashAttn V6
algorithms, BF16 comparisons, numerical validation and reproduction methods.

## Supported configurations

| Property | Public API |
|---|---|
| Platform | Linux, Python 3.10+, CUDA 13-compatible driver |
| GPUs | B200 (SM100), B300 (SM103) |
| Inputs | BF16 or FP16 Q/K/V on the same CUDA device |
| Layout | `[S,H,128]` or `[B,S,H,128]`; equal Q/K/V head counts (MHA) |
| Operation | Forward inference; self-attention or cross-attention; dense or causal |
| Default output | BF16, with the same shape as Q |

`B` is batch size, `S` is sequence length and `H` is the number of heads.
Head dimension is currently **128**. The public API does not support backward,
GQA/MQA, arbitrary masks, dropout or paged KV-cache decoding. The framework
adapter has a narrower noncausal contract. See the [API guide](docs/integration.md)
for causal alignment, optional LSE and packed-sequence handling.

## Quick start

The tested dependency stack is PyTorch 2.11.0 with CUDA 13, CuTe DSL 4.6.0 and
quack 0.6.1. Install from this repository:

```bash
git clone https://github.com/MachGen/vc-attention.git
cd vc-attention
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install . -c requirements-tested.txt
python -m pip check
vc-attn-check
# On an allocated idle GPU; the first call compiles the kernels:
vc-attn-check --smoke
```

`requirements-tested.txt` pins the tested dependencies. A CUDA Toolkit compiler
is needed only for the optional native baseline. See [installation](docs/installation.md)
for development setup and [troubleshooting](docs/troubleshooting.md) for common failures.

```python
import torch
from vc_attn import attention

# Layout is sequence, heads, head dimension.
q, k, v = [torch.randn(4096, 8, 128, device="cuda", dtype=torch.bfloat16)
           for _ in range(3)]
with torch.inference_mode():
    out = attention(q, k, v)
print(out.shape, out.dtype)  # torch.Size([4096, 8, 128]) torch.bfloat16
```

The default call uses scaled ExpCast and includes input quantization.
It selects the `v4` snapshot from VC-attn `8aa761eac`; the previous `scaled`
snapshot remains selectable for reproducing historical reports.
Dense calls default to `mid_window_blocks=4`. Eligible B200 configurations use
the source-level fusedpipe schedule. **The experimental D SASS patch is disabled
in the public API and benchmark** while its remaining issues are investigated.
Pass `mid_window_blocks=None` to restore the original scan. Causal calls retain
their masked traversal.
See the [English and Chinese scheduling reports](docs/reports/README.md) for
the implementation, measurements and limits, and the
[report reproduction guide](docs/reports/reproduction.md) for matched controls
and a kernel-only runner. Its experimental D route requires explicit opt-in.

To control the preparation boundary explicitly:

```python
from vc_attn import prepare_fp8, attention_fp8

prepared = prepare_fp8(q, k, v)
out = attention_fp8(prepared)
```

Rebuild `prepared` when the input activations change. For custom softmax scales,
cross-attention and other options, see [API and integration](docs/integration.md).

For dense single-sequence B200 calls with `S >= 32768`, an opt-in
[`prepare_fp8_fused`](docs/integration.md#opt-in-fused-preparation-on-b200)
path combines preparation work and writes packed V directly while preserving
the existing quantization scales. The default API keeps its general preparation path.

## Benchmark your workload

The benchmark accepts `SxHxD` shapes and compares the pinned BF16 reference,
ordinary FP8 reference and VC on identical inputs. It currently measures
**single-sequence, noncausal self-attention** at D=128.

The default backends are `bf16_ref`, `fp8_ref` and `vc_v4_mid4`. The VC candidate
matches `attention()`'s dense scan and uses eligible B200 fusedpipe without D.
Select `vc_v4` explicitly to measure the historical original scan.

```bash
# Small correctness/performance smoke.
vc-attn-bench --preset smoke --output results/smoke.json
vc-attn-report results/smoke.json

# Attention only: input quantization is outside the timer.
vc-attn-bench --shapes 4096x8x128 32768x32x128 \
  --baseline fp8_ref --output results/attention.json

# Include input preparation and time CUDA Graph replay.
vc-attn-bench --shapes 32768x32x128 --scope quantize-attention \
  --timing graph --output results/with-quantization.json
```

Reserve the GPU before benchmarking. Results retain raw paired samples,
numerical errors, source revisions and environment metadata. Use your own
Q/K/V tensors with `--input` to evaluate a representative input distribution.
See the [benchmark guide](docs/benchmarking.md) for input files, baseline
selection, warmup, model-specific presets and shared-GPU sampling.

## Performance

We recommend **VC Attention for B200** and **FlashAttn V6 for B300**.
Both are MachGen implementations; FlashAttn V6 is specialized for B300 / SM103.
The tables below use the same **BF16 reference** as the baseline on each GPU.

Measured with the standalone 0.1.2 package on synthetic BF16 inputs: one
noncausal sequence, D=128, equal query/KV head counts. Latency is the median
of 12 paired rounds. **Speedup = BF16 median / implementation median.**

These archived measurements use the `scaled` snapshot (`4245ca87a`), not the
new default `v4`. See [upstream synchronization](docs/upstream-sync.md) for the
new revision's scope and validation; no new speedup is inferred from these tables.

### VC Attention — B200 and B300

| GPU | Shape `[S,H,D]` | BF16 ms | VC ms | Speedup vs BF16 |
|---|---|---:|---:|---:|
| B200 | 73426,7,128 | 15.147 | 10.843 | 1.397x |
| B200 | 188214,7,128 | 99.214 | 54.197 | 1.831x |
| B200 | 73426,56,128 | 121.581 | 69.217 | 1.757x |
| B200 | 188214,56,128 | 795.860 | 432.280 | 1.841x |
| B300 | 73426,7,128 | 11.848 | 10.110 | 1.172x |
| B300 | 188214,7,128 | 86.059 | 51.356 | 1.676x |
| B300 | 73426,56,128 | 105.654 | 64.734 | 1.632x |
| B300 | 188214,56,128 | 692.213 | 410.089 | 1.688x |

### FlashAttn V6 — B300

Our native CUDA/inline-PTX implementation for B300, shown separately with BF16
as its baseline. See the [FlashAttn V6 guide](src/vc_attn/native/README.md) for
building and calling it; the `attention()` API above selects VC.

| Shape `[S,H,D]` | BF16 ms | FlashAttn V6 ms | Speedup vs BF16 |
|---|---:|---:|---:|
| 73426,7,128 | 11.848 | 7.434 | 1.594x |
| 188214,7,128 | 86.059 | 55.325 | 1.556x |
| 73426,56,128 | 105.654 | 67.517 | 1.565x |
| 188214,56,128 | 692.213 | 443.996 | 1.559x |

- **Timing:** attention-only CUDA events, including VC's internal V packing;
  input quantization, compilation, model execution and communication are excluded.
  FlashAttn V6 plan creation and output-buffer setup are also outside the timer.
- **Baseline:** `bf16_ref`, pinned source revision `2cae9072`, for both implementations.
- **Collection:** B200 used an exclusive GPU; B300 used CPU-precompiled kernels
  and monitored shared-GPU idle windows. Warmup and repeat counts differ, so
  these rows do not establish a controlled speedup between the two GPUs.
- **Limits:** performance depends on shape. S=73426/H=7 misses the archived
  `scaled` snapshot's large fused path; current B200 eligibility differs.
  Relative L2 error versus BF16 is 5.52–5.68% for VC and 5.29–5.43% for
  FlashAttn V6 on these inputs; operator measurements do not establish model
  quality or end-to-end speedup.

See [performance and validation](docs/performance.md) for all raw samples,
exact protocols and numerical errors.

## Integrate with a model

Call `attention` after the model's Q/K normalization, positional transforms
and any input sequence-parallel exchange. Pass the result to the existing output
projection or output exchange. Keep the model's softmax scale and mask semantics.

The [integration guide](docs/integration.md) covers direct PyTorch integration
and tensor preparation. An optional [SGLang diffusion adapter](docs/sglang.md)
provides a concrete framework example, including packed sequences and a
MiniMax-H3 pipeline configuration. Its operator tests passed; full model
generation, distributed execution and model-quality validation remain unverified
for this release.

## Documentation and contributing

Each source directory has a guide describing its role and entry points.

| Directory | Contents |
|---|---|
| [src/](src/README.md) | Inference API, kernels, native baseline and adapters |
| [docs/](docs/README.md) | Installation, API, design, performance and benchmark guides |
| [examples/](examples/README.md) | First attention call and optional pipeline configuration |
| [integrations/](integrations/README.md) | External-framework registration artifacts |
| [tests/](tests/README.md) | CPU contracts and Blackwell GPU validation |
| [tools/](tools/README.md) | Source auditing, native builds and measurement helpers |
| [LICENSES/](LICENSES/README.md) | Third-party license texts and scope |
| [.github/](.github/DIRECTORY.md) | CI and contribution templates |

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests and benchmark
reporting requirements, and [the review record](docs/review.md) for validation
coverage. Report problems through [GitHub Issues](https://github.com/MachGen/vc-attention/issues).

## Attribution and license

This is an independent implementation of ideas from
[VC-Attention: Value Smoothing and Softmax Casting for Low-bit Attention](https://arxiv.org/abs/2609.15810),
with additional kernel scheduling work. It is not the paper authors' official release.
CuTe kernels derive from [FlashAttention](https://github.com/Dao-AILab/flash-attention),
with VC/low-bit extensions by MachGen contributors.

The project uses BSD-3-Clause with retained third-party notices; SGLang patch
context remains Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE),
[AUTHORS](AUTHORS) and [CITATION.cff](CITATION.cff). Model weights and datasets
are not distributed here.
