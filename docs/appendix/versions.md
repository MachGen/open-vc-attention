# Version behavior

| Version | Pinned source revision | Meaning |
|---|---|---|
| `baseline` | `2cae9072801704491b37f14037b4baa32c3958dc` | Ordinary BF16/FP8 reference |
| `v1` | `99d398d6fe32d3f13067c2511872723dafc53033` | Tuning constants, scratch cleanup, clearer errors |
| `v2` | `e9afd9cf3ca26c3e377d7b316e486dce0db73367` | Shared launch/register rules; softcap and batch-rank fixes |
| `v3` | `f7c124c3845ccb34ebf91d638c2f4d69676faa86` | Public forward API, dispatch/packaging, batch descale fix |
| `scaled` | `4245ca87a02a476e89b793a5541a1f0576684b01` | Fused ExpCast path supports external descales |
| `v4` (default) | `8aa761eac845d734dc5dbe48a6196c1fe1b0a7cf` | Original-scan unscaled ExpCast and ordinary SM103 FP8 scheduling |

Each namespace is a separate snapshot. Kernel bodies are retained; imports are
renamed and package initializers are replaced to avoid unrelated imports and
global compiler monkey-patching. The manifest records hashes before and after
namespace rewriting. The new convenience API is shared across snapshots.

## What changes in `v4`

The original scan (`mid_window_blocks=None`) can now use the packed-V / inline-rescale
ExpCast path without external descales. Scaled inputs already supported this path.
On eligible SM103 ordinary-FP8 calls, score scaling is interleaved with exp2 and
the correction wait moves after the row-sum computation. These two scheduling
changes stay off on SM100 and for ExpCast, V-Smooth, skipping and mid-window scans.

`attention()` and the diffusion adapter default to `v4` with mid-window 4.
The default benchmark uses `vc_v4_mid4`, matching that scan. Eligible B200 calls
use source-level fusedpipe; the experimental D patch is disabled.
`vc_v4` retains the original scan, while `fp8_v4` disables ExpCast.
`vc_scaled` and `fp8_control` still select the previous `scaled` snapshot.
The quantization defaults are unchanged.
See the [upstream audit](../upstream-sync.md) for the commit mapping and tests.

## What changes in `scaled`

With `expcast=True`, older versions rejected external Q/K/V descales from the
V-packing/inline-rescale fast path. The scaled revision admits that case on
SM100/SM103 with DSL 4.6.0, and admits `mid_window_blocks=None`. It also handles
per-block K scaling correctly in the fused encoder and row maximum.

The admitted path repacks V to K-major, accumulates the normalizer using the PV
Tensor Core multiply, rescales output in softmax warps, and publishes P in two
64-column segments. These mechanisms work together. ExpCast was already on in
the old version; the additional gain is not evidence for the cost of exp2 alone.

The large-call gate includes `S*H >= 1048576`, `S >= 32768`, D=128, one varlen
sequence, contiguous V, dense noncausal attention, no LSE and no skipping.
For example, `73426*7` fails the size gate, while `188214*7` passes. A passing
size gate alone is not proof that every other dispatch condition is satisfied.

## Baselines

`fp8_ref` uses the same Q/K per-128-token block and V per-head descales as VC.
`bf16_ref` is the corresponding fixed BF16 implementation. `fp8_control` instead
uses the scaled source with ExpCast disabled, to isolate that switch in one revision. `upstream_bf16` is
separately installed upstream code; its installed version and interface hash
are recorded. Do not rename any of these to native v6.

Native v6 is a distinct CUDA kernel with its own online-softmax numerical
policy. It requires SM103-specific `tcgen05.ld.red` instructions. The original source
and host guard are unchanged: build and run native v6 on B300 only. No prebuilt
binaries ship; the CuTe implementations also support B200.

Always compare matched runs. A speedup over BF16 and a speedup over ordinary
FP8 have different denominators. Mid-window 4 is the current convenience-API
default; historical original-scan backend names and recorded results retain
their meaning.
