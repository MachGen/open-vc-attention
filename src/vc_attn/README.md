# vc_attn package

The public entry point is `from vc_attn import attention`. Top-level wrappers
load the CUDA implementation lazily; importing `vc_attn` alone does not initialize CUDA.

| File or directory | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Public lazy wrappers and package version |
| [api.py](api.py) | Inference contracts, layouts, FP8 preparation and versioned forward calls |
| [preparation.py](preparation.py) | Opt-in B200 joint Q/K quantization and fused V cast/pack/descale, preserving existing scales |
| [quantization.py](quantization.py) | Frozen Triton Q/K block and V head quantizers |
| [registry.py](registry.py) | Explicit backend identities and `vc-attn-info` |
| [benchmark.py](benchmark.py) | Paired benchmark orchestration, numerical checks and GPU ownership checks |
| [measurement.py](measurement.py) | CPU-only shape parsing and timing/speedup summaries |
| [source_manifest.json](source_manifest.json) | Source revisions and hashes of extracted/transformed kernel and support files |
| [_sass_d.py](_sass_d.py) / [_sass_runtime.py](_sass_runtime.py) | Exact-match B200 native scheduling patch and per-function CuTe loader adapter |
| [_kernels/](_kernels/README.md) | Six isolated CuTe kernel snapshots |
| [check.py](check.py) / [report.py](report.py) | Environment diagnostics, opt-in GPU smoke and CPU-only benchmark tables |
| [native/](native/README.md) | Original native v6 for B300 / SM103 only |
| [native_plan.py](native_plan.py) | ctypes plan lifecycle and buffer ownership for a locally compiled native DSO |
| [integrations/](integrations/README.md) | Dense/packed diffusion and SGLang adapters |

The normal call flow is `attention` → optional `prepare_fp8` → `attention_fp8`
or `raw_forward` → the selected snapshot. The API owns tensor preparation;
frameworks own model execution and communication. See
[API contracts](../../docs/integration.md) and [benchmark scopes](../../docs/benchmarking.md).

The public API and default benchmark use the mid-window-4 scan, with source-level
fusedpipe on eligible B200 calls. **D is disabled in both entry points because
it has unresolved issues.** The kernel's existing adapter call returns the
original compiled function before inspecting compiler metadata or exporting
any object. Frozen kernel source and hashes remain unchanged.

Experimental tooling can explicitly call the private
`maybe_patch_compiled(compiled, mid_window_blocks=4, enable=True)` helper after
compilation. This is not a supported inference setting. It interleaves score loads and two max operations in each
hot loop. The adapter requires CuTe DSL 4.6.0 and verifies the complete native
code and metadata. Four verified scan-offset immediate fields may vary, and
each must equal the requested window. The complete remaining native code and
all executable metadata must match. Unmatched specializations keep their
original callable, including windows that cause the compiler to change the
instruction layout.
The source-level descale/output pipeline also accepts these windows, independent
of whether D matches. `None` retains the original scan path. Changing the window
selects a separately cached compiled kernel; it is not a runtime scalar argument.
The scheduling changes preserve numerical operations, barriers and register allocation.
No cubin, private input or global compiler hook is shipped. The original DSL
disk cache stays unchanged. AOT objects exported from the patched callable
retain D. External AOT loaders and tools that export directly from `cute.compile`
bypass this adapter and keep their original artifacts.

[Repository](../../README.md) · [Parent directory](../README.md)
