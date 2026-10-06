# Historical VC v2

This snapshot shares launch arguments, register rules and wrappers, with softcap and batch-rank fixes. Select `vc_v2` to compare it against v1 or later revisions.

Pinned source revision: `e9afd9cf3ca26c3e377d7b316e486dce0db73367`.

- [flash_attn/](flash_attn/README.md) contains this snapshot's namespaced attention implementation.
- [nvfp4.py](nvfp4.py) contains the retained NVFP4 conversion helper.
- [v_smooth.py](v_smooth.py) prepares experimental value grouping and residual quantization.

Use the shared wrapper with your Q/K/V tensors:

```python
from vc_attn import attention
out = attention(q, k, v, version="v2", mode="expcast")
```

The convenience API is forward-only, MHA, D=128. A helper's presence does not
make all low-level combinations supported. Source hashes are recorded in
[the manifest](../../source_manifest.json); preserve frozen source files.
See [version behavior](../../../../docs/appendix/versions.md) for baseline identity and dispatch limits.

[Repository](../../../../README.md) · [Parent directory](../README.md)
