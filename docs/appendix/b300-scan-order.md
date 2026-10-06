# B300 scan-order experiment

The historical B300 source report also measured an explicit
`mid_window_blocks=4` configuration at `[S,H,D]=[73426,56,128]`.
This is a separate setting from the default scan shown on the project homepage.

| Setting | Upstream BF16 ms | FP8 control ms | Scaled VC ms | VC vs BF16 | VC vs FP8 control |
|---|---:|---:|---:|---:|---:|
| Default scan | 106.172 | 70.183 | 61.393 | 1.729x | 1.143x |
| Mid-window 4 | 106.218 | 69.909 | 58.595 | 1.813x | 1.193x |

The approximately 1.8x result uses **upstream BF16** as its denominator.
It is not a 1.8x improvement over ordinary FP8 or native v6. Both FP8 columns
use the same candidate source with external descales and ExpCast disabled.

The additional scaled fast path admits external descales into the fused
implementation. Its benefit combines V packing, Tensor Core normalizer
accumulation, inline output rescaling and segmented P publication;
these measurements do not isolate the cost of exp2. See [version behavior](versions.md).

These are historical source measurements, not standalone-package reruns.
See [performance provenance](../performance.md) and the
[raw samples](../benchmarks/b300-recorded.json), including discarded-round
counts and numerical errors. Mid-window 4 relative L2 versus BF16 is 2.874%.

[Repository](../../README.md) · [Parent directory](README.md)
