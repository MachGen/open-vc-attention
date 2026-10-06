# v1: namespaced FlashAttention package

This is the FlashAttention package namespace inside the `v1` snapshot.
Its implementation is under [cute/](cute/README.md). The local
[__init__.py](__init__.py) is intentionally minimal; it does not re-export or
initialize unrelated original package components.

The shared [API dispatcher](../../../api.py) loads this snapshot's
`cute.interface` when `version="v1"` is selected. Import `vc_attn` for
normal application use. This directory is not a separately installed
`flash_attn` distribution and does not replace the optional upstream baseline.

Read [the snapshot overview](../README.md) for the pinned revision and purpose.

[Repository](../../../../../README.md) · [Parent directory](../README.md)
