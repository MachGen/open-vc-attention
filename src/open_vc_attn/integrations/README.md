# Runtime adapters

`packed.py` validates host sequence boundaries. `diffusion.py` implements dense/packed sequence handling. `sglang.py` binds that adapter to the pinned SGLang attention interface. Registration and model-component routing are documented in `docs/integrations/sglang.md` at the repository root.
