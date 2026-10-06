# MachGen Attention: Faster Video Attention on Blackwell

For a structured treatment of both implementations, use the
[technical whitepaper](whitepaper.md), including separate VC and FlashAttn V6
results against BF16, algorithm details and reproduction methods.

Longer videos and higher resolutions increase the work done by attention.
Making its matrix multiplications faster is only part of the problem:
probability computation, normalization, data movement and synchronization
must keep up as well.

MachGen's VC Attention package targets these costs together. On an NVIDIA B200
at the MiniMax production shape **`[S,H,D]=[188214,7,128]`**, the current standalone
package measures **54.20 ms**, compared with **99.21 ms for the pinned BF16
reference** and **79.38 ms for ordinary FP8**. That is **1.83x over BF16** and
**1.46x over ordinary FP8**, or about **45.4%** and **31.7%** less attention time,
respectively.

These measurements use synthetic tensors with the production shape. They time
the attention call, including internal V packing, but exclude input quantization,
compilation, communication and the rest of the model. They are not end-to-end
video-generation speedups. [The complete benchmark record](performance.md)
includes raw samples, numerical errors and exact baseline identities.

## Beyond faster matrix multiplication

The implementation builds on ideas from
[VC-Attention](https://arxiv.org/abs/2609.15810) and the CuTe kernels in
[FlashAttention](https://github.com/Dao-AILab/flash-attention). It is an
independent implementation with MachGen kernel engineering, rather than the
VC-Attention paper authors' official release.

The recommended configuration uses **scaled ExpCast**: FP8 Q/K/V, explicit
quantization descales, and directly encoded FP8 attention probabilities.
All QK and PV tiles are evaluated. The default does not use V-Smooth,
skip-softmax or skip-PV.

For eligible large single-sequence calls, several changes shorten the execution
path together:

- **Scaled ExpCast** incorporates per-block Q/K descales into probability encoding, allowing scaled inputs to use the fused path.
- **Automatic V packing** supplies the K-major layout used by that path. Its cost remains inside the reported attention time.
- **Tensor Core normalization** accumulates the denominator alongside PV instead of leaving the entire reduction on CUDA cores.
- **Inline output rescaling and segmented P delivery** reduce intermediate transfers and synchronization.
- **Warp reuse and phase-specific register allocation** coordinate softmax, output correction and the epilogue.

The performance gain therefore measures a complete execution path, not just
replacing an exponential instruction. Shape matters: the small `[73426,7,128]`
case does not pass the fused path's size gate and sees a smaller improvement.
[The design guide](design.md) describes the data flow; source and dispatch
details are recorded with the package.

## Current B200 results

The following results use the 0.1.2 standalone wheel, public dependencies and a
scheduler-reserved B200. Each row has 12 randomized paired rounds with 10 timed
calls per backend per round. All rows use D=128, one sequence and equal Q/K/V
head counts. H=7 and H=56 use independently generated inputs.

| S | H | Pinned BF16 | Ordinary FP8 | Scaled ExpCast | vs BF16 | vs FP8 |
|---:|---:|---:|---:|---:|---:|---:|
| 188214 | 7 | 99.214 ms | 79.375 ms | **54.197 ms** | **1.831x** | **1.465x** |
| 188214 | 56 | 795.860 ms | 633.943 ms | **432.280 ms** | **1.841x** | **1.467x** |
| 73426 | 7 | 15.147 ms | 11.922 ms | **10.843 ms** | **1.397x** | **1.100x** |
| 73426 | 56 | 121.581 ms | 96.813 ms | **69.217 ms** | **1.757x** | **1.399x** |

The BF16 and ordinary FP8 references come from the same pinned source revision,
`2cae9072801704491b37f14037b4baa32c3958dc`. Ordinary FP8 and scaled ExpCast use the
same per-128-token-block Q/K and per-head V quantization. These references are
distinct from an arbitrary current upstream FlashAttention installation and
from the separate native v6 implementation.

On these synthetic inputs, scaled ExpCast has **5.52–5.68% relative L2 error**
against BF16; ordinary FP8 has **5.29–5.43%**. All outputs are finite. Those
operator-level measurements are useful for regression testing, but model outputs
and video quality still need separate validation.

## What the earlier 1.95x result means

An earlier B200 development experiment used actual MiniMax-H3 Q/K/V captured
from **243 frames at 1344x768**, shape **`[73426,56,128]`**. It measured
**114.240 ms for upstream BF16 FlashAttention-4 and 58.516 ms for standalone
ExpCast**: **1.952x**, or **48.78%** less attention time.

That result used an explicit mid-window-4 scan, with V-Smooth and skipping off.
It is a historical captured-input experiment, not the current default package
result above. In another experiment, Tensor Core mean restoration and weight
delivery reduced the combined V-Smooth+ExpCast path from **83.875 to 70.571 ms**,
a **15.86%** reduction. The two gains belong to different paths and runs.

[The development appendix](appendix/b200-development.md) preserves both sets of
raw samples, source hashes, accuracy metrics and evidence limits. Neither result
establishes a ranking against separately published competitor benchmarks.

## B300 and inference integration

A fresh B300 standalone-package run now covers both sequence lengths and head
counts. At `[188214,7,128]`, VC measures **51.36 ms**, versus **86.06 ms BF16**,
**57.95 ms ordinary FP8**, and **55.32 ms original native v6**: respectively
**1.68x**, **1.13x**, and **1.08x**. These denominators come from the same run.

The kernels were compiled on CPU and measured in monitored idle windows between
production workloads. Each shape has 12 complete paired rounds; interfered
rounds were discarded. This shared-GPU, short-warmup AOT protocol is different
from the exclusive B200 protocol. It is not a controlled cross-card hardware
comparison. The smaller `[73426,7,128]` shape does not enter the large fused path
and does not show a VC win over FP8/native v6 in this run. [Full results and
validation limits](performance.md) include every shape and raw samples.

The API supports both a convenient BF16/FP16-input call and an explicit
prequantized interface:

```python
from vc_attn import attention

# CUDA BF16/FP16 tensors with shape [S, H, 128].
out = attention(q, k, v)
```

See the [installation guide](installation.md) for the tested environment and
first run. The opt-in [SGLang diffusion adapter](sglang.md) places attention
between the framework's sequence-parallel exchanges. SGLang continues to own
model execution, communication and scheduling. Adapter operator tests have
passed; full MiniMax generation, end-to-end latency and video-quality evaluation
have not yet been recorded for this release.

## Next steps

The next validation work is exclusive B300 confirmation and complete generation
tests with representative prompts and seeds. Quantization, reordering and
communication need to be included when assessing deployment value.

Further research includes training-free NVFP4 Q/K computation, block scaling,
outlier handling, and precision policies that vary across layers and denoising
steps. Optional V-Smooth and sparsity experiments are documented separately from
the recommended dense path. Their value must be assessed with their preparation
costs and output quality included.

[Repository and quickstart](../README.md) · [Benchmark protocol](benchmarking.md)

## References

- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451), Ted Zadouri et al., 2026.
- [VC-Attention: Value Smoothing and Softmax Casting for Low-bit Attention](https://arxiv.org/abs/2609.15810), Xingyang Li et al., 2026.
- [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087), Jiayi Yuan et al., first submitted 2025. Related work for optional skipping experiments; skipping is disabled in the measurements above.
- [VC-Attention: Faster Low-Bit Attention Without Retraining](https://www.nunchux.ai/blog/attention-is-the-video-bottleneck), Nunchux AI Team, 2026. Independent results and deployment schedule; not a controlled head-to-head comparison.
