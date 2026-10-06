# Changelog

## 0.3.0

First public release of Open-VC Attention.

- FP8 forward attention for Blackwell SM100/SM103 with ExpCast, scales folded into score conversion, and a softmax denominator built from decoded probabilities.
- Mid-window key traversal, warp-specialized pipeline with packed V, and fused three-kernel B200 input preparation.
- V residual repair (`prepare_v_repair`, `attention_v_repair`).
- VC-Attention baseline (`open_vc_attn.baselines`) and a benchmark comparing BF16, VC and Open-VC.
- The BF16 reference is upstream FlashAttention-4 (`flash-attn-4==4.0.0b33`), installed as a dependency; the tested stack is CuTe DSL 4.6.2 with quack-kernels 0.6.4.
- Optional SGLang diffusion adapter.
