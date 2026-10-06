# Ordinary BF16 / FP8 baseline

The fixed reference used by `bf16_ref` and `fp8_ref`. Ordinary FP8 uses Q/K per-block128 and V per-head descales. This snapshot does not implement the public ExpCast mode.

Pinned source revision: `2cae9072801704491b37f14037b4baa32c3958dc`.

- [flash_attn/](flash_attn/README.md) contains this snapshot's namespaced attention implementation.
- [nvfp4.py](nvfp4.py) contains the retained NVFP4 conversion helper.

Use the shared wrapper with your Q/K/V tensors:

```python
from vc_attn import attention
out = attention(q, k, v, version="baseline", mode="fp8")
```

The convenience API is forward-only, MHA, D=128. A helper's presence does not
make all low-level combinations supported. Source hashes are recorded in
[the manifest](../../source_manifest.json); preserve frozen source files.
See [version behavior](../../../../docs/appendix/versions.md) for baseline identity and dispatch limits.

[Repository](../../../../README.md) · [Parent directory](../README.md)
