# Benchmarking

`open-vc-attn-bench` times three backends on the same inputs and writes a JSON record with raw paired samples, accuracy against BF16, environment and GPU details.

| Backend | What runs |
|---|---|
| `bf16` | Upstream FlashAttention-4 BF16 forward kernel (`flash-attn-4==4.0.0b33`) |
| `vc` | VC-Attention: ExpCast + V-Smooth, original key scan, no Open-VC optimizations |
| `open-vc` | Open-VC defaults |

```bash
open-vc-attn-bench --preset smoke --output results/smoke.json
open-vc-attn-bench --preset video --timing graph --scope attention --output results/attention.json
open-vc-attn-bench --shapes 73397x56x128 --timing graph --scope quantize-attention \
  --input capture.pt --output results/operator.json
open-vc-attn-report results/attention.json
```

## Scopes

- `attention` times one attention kernel on inputs prepared beforehand. `open-vc` uses fused, pre-packed preparation when eligible (B200, `S >= 32768`); `vc` uses its grouped, permuted and quantized inputs.
- `quantize-attention` times the complete call from BF16 inputs. For `open-vc` this is `attention(q, k, v)`, with fused preparation on eligible B200 shapes. For `vc` it includes V-Smooth's per-call preparation with the value grouping reused, as VC-Attention does after its first denoising steps.

`--timing events` uses CUDA events around eager calls; `--timing graph` replays a captured CUDA graph. Compilation and warmup are excluded from samples.

## Options

- `--backends` and `--baseline` choose what to compare (default: all three, baseline `bf16`).
- `--preparation auto|unfused|fused` selects Open-VC's preparation route in `quantize-attention` scope.
- `--repair-budget 0.005` enables V residual repair for `open-vc` (B200, `S >= 32768`). Budget and selected token counts are recorded in the backend metadata.
- `--input capture.pt` loads a tensor-only `{"q", "k", "v"}` mapping whose shape matches `--shapes`.

The runner refuses to start if other processes are using the GPU. `--allow-shared-gpu` disables that check and marks the result as not isolated. Reserve a GPU for any published number.

VC-Attention runs its k-means grouping only on the first quarter of denoising steps, and timed `vc` calls reuse the grouping. The runner therefore also times `vc` complete calls with a fresh grouping and with the reused grouping, interleaved and with CUDA events (`vc_complete_call_events_ms`). Their difference is the grouping increment that the report amortizes over steps.

In `attention` scope every backend launches exactly one attention kernel per timed call: inputs, including packed V, are prepared beforehand on every path.

`benchmarks/configs/blackwell.json` lists the reference shapes and settings.
