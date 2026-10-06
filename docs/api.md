# API

```python
from open_vc_attn import attention, prepare_fp8, attention_fp8, raw_forward
out = attention(q, k, v)
prepared = prepare_fp8(q, k, v)
out = attention_fp8(prepared)
out, lse = attention(q, k, v, return_lse=True)
bf16 = attention(q, k, v, version="reference", mode="bf16")
```

`attention` includes input quantization and attention. `prepare_fp8` freezes the current activations, and `attention_fp8` reuses the prepared tensors, including any automatic V packing. Re-prepare after activations change. Capturing `attention` in a CUDA graph captures quantization too; replay reads the current contents of stable input buffers.

Inputs must be matching FP16/BF16 tensors on one supported GPU, in `[S,H,128]` or `[B,S,H,128]` layout. K and V shapes must match, and Q must have the same batch and head count; cross-attention lengths may differ. Gradients, empty dimensions, GQA and head dimensions other than 128 are rejected. Output has Q's shape. ExpCast output is BF16; the BF16 reference keeps FP16 inputs in FP16.

| Option | Default | Meaning |
|---|---|---|
| `version` | `"open-vc"` | Implementation; `"reference"` selects the FlashAttention-4 BF16 reference |
| `mode` | `"expcast"` | `"bf16"` runs the BF16 reference and requires `version="reference"` |
| `mid_window_blocks` | `4` | Dense key traversal; `None` restores the original scan |
| `causal` | `False` | Uses masked traversal, overriding the mid window |
| `softmax_scale` | `None` | Kernel default `1/sqrt(128)`; explicit scale passes through |
| `return_lse` | `False` | Return `(output, LSE)`; may change dispatch |
| `preparation` | `"auto"` | `attention` only: fuse eligible B200 preparation; `"unfused"` opts out, `"fused"` requires eligibility |

`raw_forward` is an advanced interface that returns `(output, optional LSE)`. Callers own quantization scales, variable-length metadata and layout, and it does not apply the convenience API's defaults. Its keyword contract follows the selected kernel interface and is not part of the stable API. The package is forward-only, with no autograd support.

For packed sequences, use the diffusion adapter with explicit host segment boundaries. Each sequence is quantized separately, and trailing padding must be declared; it is never inferred from tensor length.

## Fused preparation

`attention(..., preparation="auto")` fuses preparation when all of these hold:

- contiguous self-attention shaped `[S,H,128]` or `[1,S,H,128]`, with `S >= 32768`;
- B200 with CuTe DSL 4.6.2;
- the default ExpCast mode;
- no causal mask and no LSE output.

Both the mid-window and original scans are supported. Other valid calls keep general preparation. `preparation="fused"` raises for ineligible calls; `"unfused"` opts out.

`prepare_fp8` remains the general, unpacked preparation, because its output may be used with the causal or LSE paths. To reuse fused preparation explicitly:

```python
from open_vc_attn import prepare_fp8_fused, attention_fp8

prepared = prepare_fp8_fused(q, k, v)
out = attention_fp8(prepared)
```

Fused preparation keeps per-block Q/K and per-head V quantization. It runs three kernels:

1. Q/K quantization with V-amax initialization.
2. V-amax reduction.
3. V cast, pack and descale.

Without caller-supplied metadata, the call also builds metadata on the device; this remains graph-capturable.

Pass `cu_seqlens=cu` to reuse a contiguous int32 CUDA tensor of shape `[2]` holding `[0,S]`. `PreparedFP8.v_prepacked=True` marks the padded, sequence-contiguous V layout; do not replace its V tensor with a contiguous copy. `attention_fp8` validates that layout, does not pack V twice, and rejects causal or LSE use of prepacked inputs.

## V residual repair

```python
from open_vc_attn import prepare_v_repair, attention_v_repair

prepared = prepare_v_repair(q, k, v, budget=0.005)
out = attention_v_repair(prepared)
out_original_scan = attention_v_repair(prepared, mid_window_blocks=None)
```

Supported inputs: B200 with CuTe DSL 4.6.2, and matching FP16/BF16 self-attention `[S,H,128]` or `[1,S,H,128]` with `S >= 32768`. Output is BF16 in Q's shape. Causal attention, LSE, cross-attention, multiple sequences and B300 are not supported.

`budget` is a fraction in `[0,1)`. Each head selects the `round(budget*S)` tokens with the largest squared V quantization residual. `selected_tokens` records that count, and `repair_tokens` records it padded to a multiple of 128; for `S=73397` and `budget=0.005` these are 367 and 384. A budget that rounds to zero tokens uses fused preparation with no repair work.

Selected residuals are stored as extra FP8 value rows, each paired with a duplicate of the original key; the duplicate keeps that key's descale. The kernel visits these rows after all original keys and restores the original softmax denominator before normalization. Repair rows therefore add to the output numerator without renormalizing the distribution. Repair corrects V quantization error only, not Q/K or probability error.

Rebuild the prepared object when activations change. `attention_v_repair` accepts `mid_window_blocks` and `softmax_scale`. See [the example](../examples/v_repair.py) and section 4.9 of the [technical report](technical-report/pdf/Open-VC-Attn-Technical-Report-en.pdf).

## VC-Attention baseline

```python
from open_vc_attn.baselines import attention_vc, prepare_vc, vc_attention

out = vc_attention(q, k, v)            # group, prepare and attend
grouped = prepare_vc(q, k, v)          # k-means grouping and preparation
out = attention_vc(prepare_vc(q, k, v, permutation=grouped.permutation), output_shape=q.shape)
```

This is the comparison baseline: VC-Attention's ExpCast with V-Smooth and the original key scan, without Open-VC's optimizations. V-Smooth groups value tokens with online k-means, permutes K and V by group, demeans V per 128-token block and restores the means in the kernel. Reusing a grouping (`permutation=`) runs a fused GPU preparation; pass `check_permutation=False` to skip validating it, for example inside CUDA graph capture. The baseline accepts one BF16 self-attention sequence, `[S,H,128]` or `[1,S,H,128]`, on SM100/SM103.
