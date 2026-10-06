# API and quick integration

```python
from vc_attn import attention, prepare_fp8, attention_fp8

# In a model's local dense attention call, after any Q/K normalization or RoPE:
out = attention(q, k, v, version="v4", mode="expcast", softmax_scale=scale)

# If the caller explicitly owns activation preparation:
p = prepare_fp8(q, k, v)
out = attention_fp8(p, version="v4", expcast=True, softmax_scale=scale)
```

Preparation creates new Q/K/V buffers. Reprepare when activations change.
Q/K descales have shape `[B,H,ceil(S/128)]`, V descales `[B,H]`, all FP32.
For one sequence, prepared activations are `[S,H,128]`, with B=1 scale tensors.
The `PreparedFP8` object keeps tensors and layout metadata alive.

FP16/BF16 input convenience calls validate device, dimensions and inference-only
use. Inputs may be strided; preparation makes them contiguous. Low-bit output is
BF16. `mode="bf16"` preserves the input dtype and bypasses quantization. Return
only output by default, or `(out,lse)` with `return_lse=True`. Requesting LSE can
disable the fused scaled path. Low-level unsupported combinations raise errors.

The public wrapper supports MHA at D=128; fixed-batch inputs and self/cross
attention are accepted. Packed multi-sequence inputs require segment-aware
preparation: never pass concatenated sequences as one unmasked sequence.

## Supported contract

| Property | Public convenience API |
|---|---|
| Devices | Linux CUDA, B200 / SM100 and B300 / SM103 |
| Layout | `[S,H,128]` for one sequence; `[B,S,H,128]` for independent batch items |
| Input / output | Matching BF16 or FP16 Q/K/V; FP8/ExpCast modes return BF16 |
| Heads | Equal Q/K/V head counts (MHA); GQA/MQA not supported by this wrapper |
| Sequence lengths | Self-attention or different Q/K lengths; K and V must match |
| Mask | Dense or `causal=True`; no arbitrary mask, dropout or local-window argument |
| Gradients | Forward inference only; requires-grad inputs are rejected |
| LSE | Optional `return_lse=True`; requesting it may disable the fused fast path |
| Packed sequences | Prepare each sequence independently through the adapter; never merge unrelated sequences |
| Decode / serving state | No paged KV cache or autoregressive decode backend |

For unequal Q/K lengths, causal masking follows FlashAttention's bottom-right
alignment. Confirm this matches the calling model.

Causal, fixed-batch and LSE paths do not promise the fast-path speedups shown
for dense single-sequence calls. The SGLang diffusion adapter has the narrower
noncausal contract described in [its guide](sglang.md).

A minimal integration replaces only the model's local dense attention call,
after Q/K normalization/RoPE and input exchange:

```python
with torch.inference_mode():
    out = attention(q, k, v, softmax_scale=q.shape[-1] ** -0.5)
```

Keep the model's tensor layout and output exchange unchanged. Start with BF16
reference comparisons on representative inputs before judging model outputs.

## Scan order

`attention()` and `attention_fp8()` default to `mid_window_blocks=4` for dense
low-bit calls. Eligible B200 / SM100 FP8 ExpCast calls therefore select the
source-level fusedpipe schedule. The experimental D binary patch is disabled
in public API and benchmark calls because it has unresolved issues.
This does not force unsupported configurations onto the fused path. Causal calls
ignore the scan-window option and retain masked traversal; `mode="bf16"` keeps
its native traversal. `raw_forward()` retains each snapshot's low-level defaults,
so pass `mid_window_blocks=4` explicitly when using that advanced interface.

For dense, noncausal attention, `mid_window_blocks` selects the first K/V block
relative to the query block's diagonal. It then scans left and visits the
remaining tail. This changes traversal order, not the attention mask or which
blocks are evaluated. The value counts 128-token K/V blocks, not tokens.

```python
out = attention(q, k, v)  # Dense default: mid_window_blocks=4.
out = attention(q, k, v, mid_window_blocks=None)  # Original scan.
out = attention(q, k, v, mid_window_blocks=8)
# Or reuse FP8 inputs prepared for the same activations:
out = attention_fp8(prepared, mid_window_blocks=8)
```

Use a nonnegative integer, such as `0`, `2`, `4` (the default), `8` or `16`.
Explicit `None` keeps the original scan. The value is a compile-time option: the first call with
a new window can compile a new kernel, and subsequent calls reuse its cache.
Do not use a per-call tensor for this argument. Stay within the kernel's signed
32-bit indexing range; window-plus-query-block indices must not overflow.

Eligible B200 scaled FP8 calls retain the fused descale/output pipeline across
window values, without D. D remains available only to experimental tooling via
the private `maybe_patch_compiled(..., enable=True)` helper, with all original
code and metadata guards retained. It is not a supported inference setting.
Different windows can produce different floating-point rounding and
latency, so choose using representative inputs. Causal, local, block-sparse and
split-KV configurations are outside this scan-order contract.

## Advanced kernel access

`raw_forward(q,k,v,version=...,**kwargs)` selects an exact kernel snapshot and
returns `(out,lse)`. This is an advanced interface; option compatibility follows
that snapshot. It exposes its varlen, mask, descale, NVFP4 and V-Smooth arguments.
The convenience API intentionally does not silently enable skipping or sparsity.

```python
from vc_attn.api import prepare_v_smooth, quantize_nvfp4
from vc_attn import raw_forward

# V-Smooth, BF16 q/k/v [S,H,128], cu = CUDA int32 [0,S].
p = prepare_v_smooth(q, k, v, version="v4")
out, _ = raw_forward(p.q, p.k, p.v, version="v4", expcast=True,
    cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=q.shape[0],
    max_seqlen_k=k.shape[0], mid_window_blocks=4, **p.forward_kwargs())

# NVFP4 fixed batch [B,S,H,128]; no external Q/K/V descales on this path.
qn, sq = quantize_nvfp4(q_batch)
kn, sk = quantize_nvfp4(k_batch)
out, _ = raw_forward(qn, kn, v_batch.to(torch.float8_e4m3fn),
    version="v4", mSFQ=sq, mSFK=sk, expcast=True, mid_window_blocks=4)
```

V-Smooth changes grouping/preparation and NVFP4 changes Q/K precision. Validate
them separately from the ordinary FP8/ExpCast comparison. See the benchmark's
experimental opt-in backend names.

## Framework boundaries

An attention adapter belongs after Q/K normalization, RoPE, and any input
all-to-all; its result goes to output all-to-all/projection. The adapter does
not own weights, caches, model loading, request scheduling or communications.
Use [the SGLang adapter](sglang.md) for a concrete integration.


## Opt-in fused preparation on B200

`prepare_fp8_fused` accepts contiguous BF16/FP16 `[S,H,128]` self-attention
inputs, one sequence, `S >= 32768`, on B200. Use v4 dense noncausal ExpCast
without LSE. Other configurations retain the existing `prepare_fp8` path;
passing a prepacked input to an unsupported dispatch raises an error.

```python
from vc_attn import prepare_fp8_fused, attention_fp8

# Reuse the pipeline's int32 CUDA tensor [0, S]. It must describe this input.
prepared = prepare_fp8_fused(q, k, v, cu_seqlens=cu_seqlens)
out = attention_fp8(prepared)
```

Pass existing `cu_seqlens` for CUDA Graph capture and to avoid metadata copies.
The caller owns its values, as with `raw_forward`; shape/device/dtype are checked
without a device-to-host synchronization. If omitted, metadata is built once per
preparation call. Rebuild prepared activations whenever Q/K/V changes.

Q and K retain per-128-token/per-head scales. V retains a current per-sequence,
per-head maximum: an initialized amax reduction precedes its cast/pack pass.
With existing metadata there are three preparation kernels: joint Q/K
quantization plus V amax initialization, V amax reduction, and V cast/pack/descale.
The Q/K launch finishes before the V reduction, preserving the global
initialization barrier. Q descale padding is written inside quantization.
`PreparedFP8.v` has a sequence-contiguous physical layout
and `v_prepacked=True`; `attention_fp8` forwards that contract to v4 and avoids
packing it again. The attention kernel and the default unfused API are unchanged.
