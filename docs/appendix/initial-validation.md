# Initial validation archive

These are archived short runs from the initial standalone extraction. Current
performance tables and packaged GPU coverage are in [Performance](../performance.md).
The original B300 native v6 compiled with CUDA 13 (128 registers, zero stack/spill);
no accepted B300 native timing is included.

## Short B200 performance validation

NVIDIA B200 / SM100, Torch 2.11.0+cu130, Triton 3.6.0, CuTe DSL 4.6.0,
quack 0.6.1, cuda-python 13.0.3. One scheduler-reserved GPU, unchanged clocks
and power configuration. An external monitor checked foreign CUDA PIDs every
0.5 seconds; the benchmark also checked before and after each variant. All
accepted runs completed without interference or import-guard errors.

These are short **harness validation samples**, not a cross-machine performance
claim: 0.3 seconds initial warmup per backend, 3 randomized paired rounds,
2 warm calls and 2 timed calls per round. Synthetic BF16 input seed is
`20260928 + S + H`; H=7 and H=56 inputs are independent. Attention-only CUDA
events include internal V packing and exclude Q/K/V quantization. Latencies
below are medians of round means; ratios are ratios of those medians. The raw
JSON also contains paired-ratio statistics, order and numerical errors.

### Frozen VC versions and ordinary FP8

Data: [all samples and environment](../benchmarks/b200-events.json).

| S | H | BF16 ms | Ordinary FP8 ms | VC v3 ms | Scaled VC ms | Scaled vs FP8 | Scaled vs v3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 73426 | 7 | 13.294 | 11.176 | 9.862 | 9.717 | 1.150x | 1.015x |
| 188214 | 7 | 97.357 | 77.280 | 69.565 | 54.097 | 1.429x | 1.286x |
| 73426 | 56 | 119.435 | 95.119 | 86.846 | 67.845 | 1.402x | 1.280x |
| 188214 | 56 | 793.058 | 631.582 | 560.208 | 430.923 | 1.466x | 1.300x |

V1/V2/V3 have the same outputs in the covered small tests and closely grouped
latencies here. The scaled path shows its additional gain on the three shapes
meeting its large-call size gate. At S=73426/H=7 it follows the existing path;
small timing differences are not evidence of an optimization.

Relative L2 versus the pinned BF16 reference is approximately 5.3–5.4% for
ordinary FP8 and 5.5–5.7% for these VC synthetic inputs. Finite output and these
operator errors do not prove video quality. Use your own representative inputs
and matched model outputs before adopting any approximate attention path.

## Graph and experimental paths

The core BF16/FP8/VC paths, including explicit mid-window 4, passed
quantization-plus-attention CUDA Graph measurements at S=4096 and S=188214,
H=7, D=128. See [graph samples](../benchmarks/b200-graph.json).
Experimental V-Smooth passed eager preparation-plus-attention timing, and
V-Smooth/NVFP4 both passed attention-only graph timing at S=4096/H=7:
[V-Smooth eager](../benchmarks/b200-vsmooth-events.json),
[experimental graph](../benchmarks/b200-optin-graph.json).
These smoke runs use 3 paired rounds, 3 timed calls, 2 warm calls and 0.2 seconds
initial warmup. They verify the harness paths and finite outputs; they do not
establish a model-quality benefit or stable small-kernel throughput.

The frozen V-Smooth/NVFP4 preparation paths contain host-to-device copies and
failed full capture in the tested environment. That combination is now rejected
explicitly. Their preparation cost is not silently removed from an inclusive
measurement. Earlier failed trials are excluded from the accepted data above.

## Integration boundary

The SGLang registration patch was checked, applied and reverted against its
pinned public source preimages. The framework-neutral adapter's actual GPU
math and packed/padded sequence behavior were tested. Full SGLang initialization,
MiniMax checkpoint loading, distributed communication, video generation and
output-quality evaluation have **not** been run for this release. No SGLang or
MiniMax end-to-end speedup is claimed. The pinned adapter and pipeline runner
are an integration starting point with an explicit validation gap.

GPU CI is manual and requires a maintainer-provided, reserved Blackwell runner.
No GPU results are inferred from CPU CI. To publish performance claims, repeat
the [full default benchmark protocol](../benchmarking.md) and record model-quality
results separately from operator error and latency.
