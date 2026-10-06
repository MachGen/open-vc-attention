# V4: synchronized VC-attn snapshot

This default snapshot retains scaled ExpCast and extends the original-scan
packed-V / inline-rescale path to eligible unscaled FP8 inputs. It also includes
SM103 ordinary-FP8 interleaved score scaling / exp2 and deferred correction
waiting. `vc_v4`, `vc_v4_mid4` and `fp8_v4` select configurations of this source.

Pinned source revision: `8aa761eac845d734dc5dbe48a6196c1fe1b0a7cf`.
See the [upstream audit](../../../../docs/upstream-sync.md) for commit coverage.

- [flash_attn/](flash_attn/README.md) contains this snapshot's namespaced attention implementation.
- [nvfp4.py](nvfp4.py) contains the retained NVFP4 conversion helper.
- [v_smooth.py](v_smooth.py) prepares experimental value grouping and residual quantization.

Use the shared wrapper with your Q/K/V tensors:

```python
from vc_attn import attention
out = attention(q, k, v, version="v4", mode="expcast")
```

The convenience API is forward-only, MHA, D=128. A helper's presence does not
make all low-level combinations supported. Source hashes are recorded in
[the manifest](../../source_manifest.json); preserve frozen source files.
See [version behavior](../../../../docs/appendix/versions.md) for baseline identity and dispatch limits.

[Repository](../../../../README.md) · [Parent directory](../README.md)
