# Runtime attention adapters

| File | Responsibility |
|---|---|
| [packed.py](packed.py) | Validates host sequence boundaries and identifies explicit trailing padding without a device-to-host metadata copy |
| [diffusion.py](diffusion.py) | Framework-neutral dense/packed adapter; quantizes each real sequence independently |
| [sglang.py](sglang.py) | SGLang backend/implementation classes using the shared adapter |
| [__init__.py](__init__.py) | Package marker |

The adapter selects its snapshot/mode through `VC_ATTN_VERSION` and
`VC_ATTN_MODE`. It supports dense noncausal MHA at D=128. Packed calls require
matching host/device boundaries; padding is zero-filled. Ring attention,
KV-cache decoding and causal generation are outside this adapter's contract.

SGLang must first be registered using the
[external patch](../../../integrations/sglang/README.md). It retains ownership
of Ulysses/all-to-all and model execution. See the
[integration guide](../../../docs/sglang.md) and
[validation coverage](../../../docs/performance.md).

[Repository](../../../README.md) · [Parent directory](../README.md)
