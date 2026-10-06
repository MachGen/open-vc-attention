# Performance

This page compares three implementations of the same attention operator on B200:

| Name | Implementation |
|---|---|
| `bf16` | Upstream FlashAttention-4 BF16 forward kernel (`flash-attn-4==4.0.0b33`, unmodified) |
| `vc` | VC-Attention's method on the same kernel family: ExpCast + V-Smooth, original key scan, none of Open-VC's optimizations |
| `open-vc` | Open-VC defaults: ExpCast, mid-window traversal, packed V, fused preparation |

`vc` is our implementation of the published VC-Attention method. V-Smooth (k-means value grouping, K/V permutation, per-block V demeaning restored in the kernel) runs on the kernel's tuned path with fused GPU preparation. Raw records with every sample and execution order: [`comparison.json`](../benchmarks/results/b200/comparison.json) and [`repair.json`](../benchmarks/results/b200/repair.json).

## Inputs

BF16 Q/K/V of shape `[73397, 56, 128]` captured from a MiniMax-H3 video denoising step (step 24, transformer block 20; 821 text tokens and 72 latent frames of 1,008 tokens), after QK normalization and RoPE. The H7 and H14 rows use the first 7 and 14 heads of that capture; H7 is the per-GPU shape of 8-way Ulysses parallelism on this model. The tensors are private; the records include their SHA-256.

## Attention kernel

Inputs, including packed V, are prepared before timing (Open-VC: fused, pre-packed FP8; VC: grouped, permuted and quantized). Each replay launches one attention kernel.

| Heads | BF16 (ms) | VC (ms) | Open-VC (ms) | VC speedup | Open-VC speedup |
|---:|---:|---:|---:|---:|---:|
| 7 | 14.178 | 8.561 | 6.983 | 1.66× | **2.03×** |
| 14 | 28.619 | 17.196 | 14.159 | 1.66× | **2.02×** |
| 56 | 114.355 | 70.160 | 57.256 | 1.63× | **2.00×** |

## Complete call

Each replay runs the full call from BF16 inputs, including sequence metadata and all FP8 preparation. VC's call includes V-Smooth's preparation with the value grouping reused.

| Heads | BF16 (ms) | VC (ms) | Open-VC (ms) | VC speedup | Open-VC speedup |
|---:|---:|---:|---:|---:|---:|
| 7 | 14.149 | 8.751 | 7.112 | 1.62× | **1.99×** |
| 14 | 28.660 | 17.542 | 14.333 | 1.63× | **2.00×** |
| 56 | 114.464 | 71.667 | 58.115 | 1.60× | **1.97×** |

## V-Smooth grouping

VC-Attention runs its k-means grouping only on the first quarter of denoising steps and reuses it afterwards. The benchmark times VC complete calls with a fresh grouping and with the reused grouping (CUDA events, interleaved, median of 5); the difference is the grouping cost, amortized over a quarter of the steps:

| Heads | Grouping increment (ms) | Share of VC attention time | Averaged over steps (first 25% group) | VC complete call incl. amortized grouping |
|---:|---:|---:|---:|---:|
| 7 | 1.01 | 12% | 3.0% | 1.57× |
| 14 | 1.84 | 11% | 2.7% | 1.59× |
| 56 | 4.71 | 7% | 1.7% | 1.57× |

## Accuracy

Error against the BF16 output:

| Heads | VC rel. L2 | Open-VC rel. L2 | Open-VC RMSE | Open-VC max abs. |
|---:|---:|---:|---:|---:|
| 7 | 3.095% | 3.194% | 0.1332 | 5.50 |
| 14 | 2.862% | 2.989% | 0.1376 | 10.25 |
| 56 | 2.870% | 2.980% | 0.1393 | 12.12 |

FP8 attention is approximate; validate model outputs on your own workload before deployment.

### Sensitive layers

FP8 error depends strongly on the layer. Most layers of the captured MiniMax-H3 model behave like the rows above, but the final layer, which the model keeps in BF16, does not. At step 48 / layer 49 (capture SHA-256 `d1969b1ac33ac13138034ca54a6fd9b07a32e90f6c285be8e6acf4b0762096d8`), Open-VC's full `[73426, 56, 128]` output has **57.20%** relative L2 error against BF16 (RMSE 22.13 against a BF16 output RMS of 38.69, maximum absolute error 714). Plain FP8 without ExpCast gives 57.19%, and fused and unfused preparation give identical bytes, so neither ExpCast nor preparation causes it.

An FP64 reference on 64 evenly spaced queries over all 56 heads and all 73,426 keys isolates the source:

| Sampled-query computation | Relative L2 vs FP64 |
|:---|---:|
| BF16 kernel | 0.139% |
| Quantize Q/K only, exact softmax | 56.22% |
| Quantize V only, exact softmax | 2.19% |
| Quantize Q/K/V, exact softmax | 56.27% |
| Open-VC | 56.28% |

Against exact attention on the same dequantized inputs, Open-VC's error is 0.71%. The error therefore comes from quantizing Q and K on this layer, not from the kernel. Keep layers the model runs in BF16 in BF16. Record: [`numerical-boundary-validation.json`](../benchmarks/results/b200/numerical-boundary-validation.json).

## V residual repair

Open-VC with repair on the H56 activations:

| Budget | Rel. L2 vs BF16 | Attention (ms) | Complete call (ms) |
|---:|---:|---:|---:|
| 0% | 2.980% | 57.20 | 58.14 |
| 0.5% | 2.924% | 57.73 | 59.05 |
| 1% | 2.919% | 58.23 | 59.57 |
| 2% | 2.912% | 59.44 | 60.81 |
| 4% | 2.905% | 61.42 | 62.76 |
| 8% | 2.894% | 65.52 | 66.99 |

A 0.5% budget lowers relative L2 error by 1.90% for 1.55% more time in the complete call.

## Method

- Milliseconds per CUDA Graph replay: CUDA events around ten replays, after six warm replays.
- 12 rounds with the three implementations in randomized order; speedup is the BF16 median divided by the candidate median within the same run.
- The benchmark checks before and after every sample that no other process uses the GPU.
- PyTorch 2.11.0+cu130, CuTe DSL 4.6.2, Triton 3.6.0, quack-kernels 0.6.4, flash-attn-4 4.0.0b33, cuda-python 13.0.3.

## Reproducing

```bash
open-vc-attn-bench --shapes 73397x56x128 --input capture.pt --timing graph --scope attention --output results/attention.json
open-vc-attn-bench --shapes 73397x56x128 --input capture.pt --timing graph --scope quantize-attention --output results/call.json
open-vc-attn-report results/attention.json
```

`--input` takes a tensor-only `{"q", "k", "v"}` file matching `--shapes`. Without it the benchmark uses random inputs, which exercise the same code but not the same data distribution.
