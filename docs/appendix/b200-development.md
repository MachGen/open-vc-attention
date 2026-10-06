# Historical B200 development measurements

These are archived kernel-development experiments from 2026-09-23, not reruns
of the standalone package. They explain the earlier **1.95x** headline and the
separate **15.9%** combined-path improvement. The current package measurements
and pinned references are in [Performance](../performance.md).

Both experiments use captured MiniMax-H3 Q/K/V, `[S,H,D]=[73426,56,128]`, from
243 frames at 1344x768. They use B200, Torch 2.11.0+cu130, CuTe DSL 4.6.0,
quack 0.6.1 and upstream `flash-attn-4==4.0.0b21` for the BF16 denominator.
The upstream interface SHA-256 is
`45bd04e1ded38264905a5907d309588b2455c5ae9cc1931164a5f7126985ef1c`.

Both use `mid_window_blocks=4`, with skip-softmax and skip-PV disabled.
CUDA-event timing excludes input quantization and V-Smooth preprocessing,
and includes in-call V packing. This differs from today's default scan and
synthetic-input standalone protocol. Private activation tensors are not shipped.

## Standalone ExpCast: the 1.95x result

| Upstream BF16 | ExpCast before | ExpCast after | After vs upstream BF16 |
|---:|---:|---:|---:|
| 114.240 ms | 61.780 ms | **58.516 ms** | **1.952x** |

Eight paired rounds, six warm calls and six timed calls per round. The raw
before-source revision is `7d52e40dcf404c7d7ff176b92d11d943dea7f282`;
candidate identity is preserved through source-file hashes. The optimization
reduces ExpCast latency by **5.283%** (61.780 to 58.516 ms). The **48.778%**
reduction instead compares the final ExpCast kernel with upstream BF16.
These are different comparisons.

The measured path has **no V-Smooth and no skipped tiles**. Its scheduling work
rescales output in softmax warps before releasing PV, removes per-block scale
transfers/notifications, adjusts phase-specific registers and delivers P in
segments. It is not an isolated ExpCast-on/off ablation.

The archived accuracy check reports bitwise-identical output before/after,
2.8995% relative L2 and 4.25 maximum absolute error versus upstream BF16,
with finite outputs. These metrics do not establish video quality.

Data: [raw samples, numerical error and source hashes](../benchmarks/b200-historical-expcast.json).

## V-Smooth plus ExpCast: a separate experiment

| Upstream BF16 | Combined before | Combined after | After vs upstream BF16 |
|---:|---:|---:|---:|
| 111.537 ms | 83.875 ms | **70.571 ms** | **1.580x** |

Six paired rounds, ten warm calls and ten timed calls per round. Before-source
revision: `7fcd99433c1bf67d1728b6db198f999f702f6f00`. Tensor Core mean restoration
and vectorized weight delivery reduce combined-path latency by **15.861%**.
This percentage is not additive with the standalone ExpCast result above.

Combined output changes by 0.0118% relative L2 versus the prior implementation.
Against upstream BF16 it has 2.8593% relative L2 and 3.6875 maximum absolute
error, with finite outputs. Standalone ExpCast in this same run stays at about
60.3 ms; the combined-path change does not explain a gain in that control.

Data: [raw samples, numerical error and source hashes](../benchmarks/b200-historical-combined.json).

## Evidence limits and external comparisons

The archived JSON contains the full sample counts, source hashes and telemetry,
but lacks a per-round GPU-ownership/delayed-utilization audit. It is retained as
historical evidence, not promoted to the current standalone acceptance standard.
Its schema is intentionally distinct from `vc-attn-bench` output.

Nunchux's [public report](https://www.nunchux.ai/blog/attention-is-the-video-bottleneck)
uses the same model and video specifications, but its VC B200 result is weighted
across a deployed denoising schedule with V-Smooth enabled for the first quarter
of steps. Its proprietary extension is another implementation. Independently
reported ratios do not establish a head-to-head ranking: captures, schedules,
numerical policies and baseline runs must be matched first.

[Repository](../../README.md) · [Appendices](README.md)
