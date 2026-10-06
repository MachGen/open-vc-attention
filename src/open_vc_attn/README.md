# Open-VC runtime package

`api.py` is the public inference API, and `quantization.py` and `preparation.py` prepare FP8 inputs (`preparation.py` is the fused B200 route). `v_repair.py` implements V residual repair, and `baselines.py` provides the VC-Attention baseline. `_dispatch.py` lazily selects the Open-VC kernels (`_kernels/blackwell/`) or the FlashAttention-4 BF16 reference (the upstream `flash-attn-4` package). `benchmarking/` and `cli/` provide the comparison benchmark and commands, and `integrations/` contains optional adapters.

The stable entry points are exported from `__init__.py`; internal kernel modules are not a public compatibility surface. See [API](../../docs/api.md) and [architecture](../../docs/architecture.md).
