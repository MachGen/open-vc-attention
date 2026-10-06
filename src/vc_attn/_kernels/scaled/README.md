# Scaled ExpCast snapshot

This snapshot admits external Q/K/V descales to the eligible V-packing/inline-rescale path. It includes per-block K-scale handling and allows the default `mid_window_blocks=None`. `vc_scaled` and `vc_scaled_mid4` select different scan settings of this same source.

Pinned source revision: `4245ca87a02a476e89b793a5541a1f0576684b01`.

- [flash_attn/](flash_attn/README.md) contains this snapshot's namespaced attention implementation.
- [nvfp4.py](nvfp4.py) contains the retained NVFP4 conversion helper.
- [v_smooth.py](v_smooth.py) prepares experimental value grouping and residual quantization.

Use the shared wrapper with your Q/K/V tensors:

```python
from vc_attn import attention
out = attention(q, k, v, version="scaled", mode="expcast")
```

The convenience API is forward-only, MHA, D=128. A helper's presence does not
make all low-level combinations supported. Source hashes are recorded in
[the manifest](../../source_manifest.json); preserve frozen source files.
See [version behavior](../../../../docs/appendix/versions.md) for baseline identity and dispatch limits.

[Repository](../../../../README.md) · [Parent directory](../README.md)
