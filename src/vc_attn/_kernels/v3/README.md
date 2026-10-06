# Historical VC v3

This snapshot introduces the public forward API and dispatch/packaging refactor, including the batch-descale fix. Its ExpCast path is `vc_v3`; it predates the separately stored scaled fusion change.

Pinned source revision: `f7c124c3845ccb34ebf91d638c2f4d69676faa86`.

- [flash_attn/](flash_attn/README.md) contains this snapshot's namespaced attention implementation.
- [nvfp4.py](nvfp4.py) contains the retained NVFP4 conversion helper.
- [v_smooth.py](v_smooth.py) prepares experimental value grouping and residual quantization.

Use the shared wrapper with your Q/K/V tensors:

```python
from vc_attn import attention
out = attention(q, k, v, version="v3", mode="expcast")
```

The convenience API is forward-only, MHA, D=128. A helper's presence does not
make all low-level combinations supported. Source hashes are recorded in
[the manifest](../../source_manifest.json); preserve frozen source files.
See [version behavior](../../../../docs/appendix/versions.md) for baseline identity and dispatch limits.

[Repository](../../../../README.md) · [Parent directory](../README.md)
