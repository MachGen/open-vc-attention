# Pinned SGLang registration

This integration targets SGLang revision
`0318a8d0af86ba14a05ca093aa43fafe446da23e`.

- [register.patch](register.patch) adds the `VC_ATTN` enum and CUDA dispatch branch.
- [manifest.json](manifest.json) records the pinned revision and SHA-256 hashes
  of the two files before and after applying the patch.

Run from the repository root against a checkout of that revision:

```bash
python tools/install_sglang.py /path/to/sglang          # verify and preview
python tools/install_sglang.py /path/to/sglang --apply
python tools/install_sglang.py /path/to/sglang --revert
```

The installer rejects mismatched files. The runtime implementation is
[vc_attn.integrations.sglang](../../src/vc_attn/integrations/sglang.py), and the
[guide](../../docs/sglang.md) explains backend selection and model-component routing.
Patch context is covered by [Apache-2.0](../../LICENSES/Apache-2.0.txt).
Full MiniMax generation remains outside the recorded validation coverage.

[Repository](../../README.md) · [Parent directory](../README.md)
