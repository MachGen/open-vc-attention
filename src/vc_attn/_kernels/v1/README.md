# Historical VC v1

The first historical VC snapshot. It extracts tuning constants, removes scratch material and improves diagnostics. Select `vc_v1` in benchmarks for its ExpCast path.

Pinned source revision: `99d398d6fe32d3f13067c2511872723dafc53033`.

- [flash_attn/](flash_attn/README.md) contains this snapshot's namespaced attention implementation.
- [nvfp4.py](nvfp4.py) contains the retained NVFP4 conversion helper.
- [v_smooth.py](v_smooth.py) prepares experimental value grouping and residual quantization.

Use the shared wrapper with your Q/K/V tensors:

```python
from vc_attn import attention
out = attention(q, k, v, version="v1", mode="expcast")
```

The convenience API is forward-only, MHA, D=128. A helper's presence does not
make all low-level combinations supported. Source hashes are recorded in
[the manifest](../../source_manifest.json); preserve frozen source files.
See [version behavior](../../../../docs/appendix/versions.md) for baseline identity and dispatch limits.

[Repository](../../../../README.md) · [Parent directory](../README.md)
