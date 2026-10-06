# SGLang diffusion / MiniMax-H3

The adapter targets SGLang source revision
`0318a8d0af86ba14a05ca093aa43fafe446da23e`. The pinned public interfaces are:

- [AttentionImpl and packed-varlen contract](https://github.com/sgl-project/sglang/blob/0318a8d0af86ba14a05ca093aa43fafe446da23e/python/sglang/multimodal_gen/runtime/layers/attention/backends/attention_backend.py).
- [MiniMax-H3 attention caller](https://github.com/sgl-project/sglang/blob/0318a8d0af86ba14a05ca093aa43fafe446da23e/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py).
- [Python generator](https://github.com/sgl-project/sglang/blob/0318a8d0af86ba14a05ca093aa43fafe446da23e/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py).

## Installation

Install SGLang's diffusion dependencies using its pinned checkout's installation
instructions, then install VC Attention in the same environment. Do not resolve
dependency conflicts by silently replacing the measured DSL/quack pair.

```bash
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout 0318a8d0af86ba14a05ca093aa43fafe446da23e
# Follow that checkout's diffusion installation guide.
python -m pip install -e /path/to/vc-attention
cd /path/to/vc-attention
python tools/install_sglang.py /path/to/sglang          # inspect and check
python tools/install_sglang.py /path/to/sglang --apply
```

Registration adds one enum entry and one explicit CUDA dispatch branch; the
implementation stays in this package. File preimages are SHA-256 checked before
any edit. `--revert` requires matching patched files. There is no import-time
monkey-patching or process-wide SDPA replacement.

## Routing and layouts

Select `--attention-backend vc_attn`. Worker processes inherit `VC_ATTN_VERSION`
(`v4` by default, `v1`, `v2`, `v3`, `scaled`, or `baseline`) and `VC_ATTN_MODE` (`bf16`, `fp8`,
`expcast`). These options are validated. `baseline` + `expcast` is invalid.

The adapter quantizes incoming BF16/FP16 tensors on every call. It supports
dense noncausal MHA, D=128, and the packed-varlen interface used by MiniMax-H3.
`cu_seqlens_host` is required: the adapter never copies device sequence metadata
back to the CPU during a graph. Host and device metadata must agree, as in the
SGLang caller contract. Each real sequence is quantized independently so a
128-token quantization block cannot cross a sequence boundary. Explicit trailing
padding is zero-filled and excluded from attention.

Multi-sequence calls loop over real segments. This preserves the semantics but
does not promise optimal ragged-batch throughput. The framework continues to own
Ulysses/all-to-all. Ring attention, KV-cache decoding, dropout and causal
generation are not supported by this adapter and are not silently substituted.

The example fixes the Qwen3-VL text encoder to FA and the VAE to its default
Torch SDPA backend, both supported by the pinned model code. Only the DiT
backend changes between runs. For other models, use their supported component
overrides; this repository does not introduce or copy any model code.

## End-to-end measurement

`tools/benchmark_pipeline.py` loads one generator, warms it, then times complete
synchronous `generate` calls using a wall clock. Model loading is separately
recorded and excluded. Generation, decoding and output I/O remain included as
configured. The tool keeps seeds and settings fixed; prompts and model paths
are not copied into its result JSON. It does not download or bundle weights
on its own, but SGLang may resolve the supplied model ID normally.

First edit `examples/minimax_sglang.json` for a model/checkpoint you have access
to and its supported duration/task settings. The example leaves the model's
duration defaults intact. Use the official recipe for multi-GPU sizing and
MiniMax licensing. Reserve enough GPUs and memory before starting.

```bash
python tools/benchmark_pipeline.py --config examples/minimax_sglang.json \
  --backend vc_attn --vc-version baseline --vc-mode fp8 \
  --output results/pipeline-fp8.json

python tools/benchmark_pipeline.py --config examples/minimax_sglang.json \
  --backend vc_attn --vc-version v4 --vc-mode expcast \
  --baseline-json results/pipeline-fp8.json --output results/pipeline-v4.json

# Independent framework FA baseline; do not label it ordinary FP8.
python tools/benchmark_pipeline.py --config examples/minimax_sglang.json \
  --backend fa --output results/pipeline-fa.json
```

Ratios between these separate processes are explicitly labeled **unpaired**.
The tool rejects baseline JSON with different non-attention settings or SGLang
version or reported GPU/Torch/CUDA environment. Also hold GPU allocation, clocks, checkpoint contents, warmup, caching,
offload, resolution, duration, seed and sampler settings fixed. Config hashes
cannot prove that mutable checkpoint contents are identical; pin model revisions.
Run alternating A/B process order and retain all trials when publishing results.

Inspect generated outputs for quality with matched seeds. Report both warm
generation latency and any kernel/denoise measurements separately. The included
adapter and harness do not by themselves establish a MiniMax E2E speedup; see
[validation status](performance.md) for what has actually run.

Public references: [SGLang attention backends](https://github.com/sgl-project/sglang/blob/0318a8d0af86ba14a05ca093aa43fafe446da23e/docs/docs/sglang-diffusion/attention_backends.mdx),
[MiniMax-H3 recipe](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx).
