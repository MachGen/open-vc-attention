# Pinned SGLang registration

Target revision: `0318a8d0af86ba14a05ca093aa43fafe446da23e`.
`register.patch` adds the `OPEN_VC_ATTN` enum and CUDA dispatch branch.
`manifest.json` records SHA-256 preimages and patched files.

```bash
python integrations/sglang/install.py /path/to/sglang
python integrations/sglang/install.py /path/to/sglang --apply
python integrations/sglang/install.py /path/to/sglang --revert
```

The installer previews by default and rejects mismatched preimages. See [guide](../../docs/integrations/sglang.md) for component routing. Patch context is covered by [Apache-2.0](../../LICENSES/Apache-2.0.txt). Validate full model generation on your own workload.
