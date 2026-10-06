# Examples

- `basic_attention.py` runs the whole operator.
- `prepared_fp8.py` separates preparation from attention.
- `compare_backends.py` compares BF16, the VC baseline and Open-VC on one input.
- `v_repair.py` shows V residual repair on a long B200 input.

GPU scripts require a Blackwell GPU. Random-input errors are diagnostics, not a model-quality test. SGLang setup is in `docs/integrations/sglang.md`.
