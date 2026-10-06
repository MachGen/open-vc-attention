# Troubleshooting

- **No CUDA / unsupported GPU:** run `open-vc-attn-check`, inspect the PyTorch CUDA build and driver, and use SM100/SM103. CPU metadata commands are not GPU support.
- **First call is slow:** compilation is excluded from benchmarks. Keep a writable cache and warm the actual shape/options before CUDA graph capture.
- **Dependency mismatch:** use the pinned constraints; CuTe schedule gates depend on DSL 4.6.2 and quack 0.6.4.
- **`flash_attn.cute` fails to import:** the BF16 reference comes from the `flash-attn-4` package. An installed FlashAttention-2 (`flash-attn`) package owns `flash_attn/__init__.py` and can shadow it; use an environment without FlashAttention-2.
- **Other CUDA processes detected:** reserve a GPU. Shared diagnostic mode is explicitly non-isolated and cannot establish performance.
- **Gradients / GQA / another head dimension:** these are outside the convenience API contract. Do not expect an automatic fallback.
- **LSE or causal path is slower:** these options change kernel dispatch. Compare the same requested outputs and mask on both implementations.
- **Packed input mismatch:** provide exact host boundaries and declare trailing padding; do not merge sequences to avoid metadata validation.

Include package version, source revision, `open-vc-attn-check` output, GPU/driver, shape, dtype, options, full error and a minimal synthetic reproduction when reporting an issue. Do not include model weights or private input data.
