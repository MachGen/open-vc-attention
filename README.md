# Open-VC Attention

FP8 forward attention for NVIDIA Blackwell (B200 / B300), built on the FlashAttention-4 CuTe DSL kernel. Open-VC implements ExpCast from [VC-Attention](https://arxiv.org/abs/2609.15810), which writes FP8 softmax probabilities directly from scores without an exponential. It adds its own optimizations around ExpCast:

- per-block Q/K and per-head V scales folded into score conversion;
- a mid-window key traversal that settles the running maximum early;
- a warp-specialized pipeline with a packed V layout;
- fused three-kernel input preparation on B200;
- optional V residual repair.

<p align="center"><img src="docs/technical-report/figures/pipeline.png" alt="One iteration of the Open-VC main loop" width="880"></p>

<p align="center"><em>One iteration of the main loop for a 128-row query tile: tensor-core work (blue), CUDA-core softmax work (orange) and data movement (gray).</em></p>

See the [technical report](docs/technical-report/report.en.md) for the design and its numerics.

## Results

B200, BF16 Q/K/V captured from a MiniMax-H3 video denoising step, `S = 73,397`, `D = 128`. Speedups are relative to upstream FlashAttention-4 BF16 (`flash-attn-4` 4.0.0b33) in the same run. VC is VC-Attention's method (ExpCast + V-Smooth) on the same kernel family, without Open-VC's optimizations; its k-means grouping, which runs only on early denoising steps, is amortized in [performance](docs/performance.md#v-smooth-grouping).

<p align="center"><img src="docs/technical-report/figures/throughput.png" alt="Attention throughput of BF16, VC and Open-VC on B200" width="880"></p>

<p align="center"><em>Attention throughput on B200 (PFLOP/s, FLOP model 4S²HD). Labels give Open-VC's speedup over BF16; the solid line is the ceiling for a kernel that sends every exponential through MUFU.</em></p>

| Heads | Scope | BF16 (ms) | VC (ms) | Open-VC (ms) | VC speedup | Open-VC speedup |
|---:|---|---:|---:|---:|---:|---:|
| 7 | attention kernel | 14.178 | 8.561 | 6.983 | 1.66× | **2.03×** |
| 14 | attention kernel | 28.619 | 17.196 | 14.159 | 1.66× | **2.02×** |
| 56 | attention kernel | 114.355 | 70.160 | 57.256 | 1.63× | **2.00×** |
| 7 | complete call | 14.149 | 8.751 | 7.112 | 1.62× | **1.99×** |
| 14 | complete call | 28.660 | 17.542 | 14.333 | 1.63× | **2.00×** |
| 56 | complete call | 114.464 | 71.667 | 58.115 | 1.60× | **1.97×** |

Relative L2 error against BF16 is 2.98–3.19% for Open-VC and 2.86–3.09% for VC. See [performance](docs/performance.md) for methodology, accuracy and V residual repair, and [benchmarking](docs/benchmarking.md) to reproduce.

## Install

Requires Linux, Python 3.10+, CUDA 13 PyTorch, and a B200 (SM100) or B300 (SM103) GPU.

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -c requirements-tested.txt .
open-vc-attn-check --smoke   # compiles and runs a small correctness check on the GPU
```

## Use

```python
import torch
from open_vc_attn import attention

q, k, v = [torch.randn(32768, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
out = attention(q, k, v)  # FP8 quantization + ExpCast attention, BF16 output
```

The API covers forward inference with multi-head attention, head dimension 128, and FP16/BF16 inputs shaped `[S,H,128]` or `[B,S,H,128]`. Long, non-causal sequences (`S >= 32768`) use the packed fast path. Other supported calls use a general path with the same semantics.

V residual repair re-adds the quantization residual of the worst-rounded value tokens. It currently runs on B200:

```python
from open_vc_attn import prepare_v_repair, attention_v_repair

out = attention_v_repair(prepare_v_repair(q, k, v, budget=0.005))
```

The VC-Attention baseline used in the comparison is available as `open_vc_attn.baselines.vc_attention(q, k, v)`.

## Documentation

- [Installation](docs/installation.md), [API](docs/api.md), [troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md), [technical report](docs/technical-report/)
- [Performance](docs/performance.md), [benchmarking](docs/benchmarking.md)
- [SGLang integration](docs/integrations/sglang.md), [contributing](CONTRIBUTING.md)

## License and attribution

BSD-3-Clause. Open-VC derives from [FlashAttention](https://github.com/Dao-AILab/flash-attention) and implements ExpCast from VC-Attention. Open-VC is an implementation inspired by the published VC-Attention paper. See [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff).
