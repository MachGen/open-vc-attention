# scaled: CuTe DSL attention implementation

This directory holds the Python/CuTe DSL implementation and helpers for the
`scaled` snapshot. Start at [interface.py](interface.py) for dispatch and
[flash_fwd_sm100.py](flash_fwd_sm100.py) for the Blackwell kernel. The main data
flow is Q/K dot products → online softmax/probability conversion → PV accumulation
→ normalized output. The selected snapshot determines the low-bit details.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Minimal package initializer. |
| [ampere_helpers.py](ampere_helpers.py) | Retained earlier-architecture shared-memory and MMA helpers. |
| [barrier.py](barrier.py) | Low-level memory-order and barrier operations. |
| [blackwell_helpers.py](blackwell_helpers.py) | Blackwell MMA/tcgen05 instruction helpers. |
| [block_info.py](block_info.py) | Query/key block indexing and bounds. |
| [block_sparse_utils.py](block_sparse_utils.py) | Kernel-side block-sparse load and execution utilities. |
| [block_sparsity.py](block_sparsity.py) | Block-sparsity metadata, validation and helper structures. |
| [copy_utils.py](copy_utils.py) | Copies and layout conversions across attention memory stages. |
| [cute_dsl_ptxas.py](cute_dsl_ptxas.py) | Optional system-ptxas integration selected by an explicit environment setting. |
| [cute_dsl_utils.py](cute_dsl_utils.py) | CuTe compilation, device capacity and tensor-layout support. |
| [fast_math.py](fast_math.py) | Low-level arithmetic helper operations. |
| [flash_bwd.py](flash_bwd.py) | Retained backward kernel/support code imported by this snapshot; outside the public inference API. |
| [flash_bwd_postprocess.py](flash_bwd_postprocess.py) | Retained backward kernel/support code imported by this snapshot; outside the public inference API. |
| [flash_bwd_preprocess.py](flash_bwd_preprocess.py) | Retained backward kernel/support code imported by this snapshot; outside the public inference API. |
| [flash_bwd_sm100.py](flash_bwd_sm100.py) | Retained backward kernel/support code imported by this snapshot; outside the public inference API. |
| [flash_bwd_sm90.py](flash_bwd_sm90.py) | Retained backward kernel/support code imported by this snapshot; outside the public inference API. |
| [flash_fwd.py](flash_fwd.py) | Shared forward infrastructure and retained earlier-architecture implementations. |
| [flash_fwd_combine.py](flash_fwd_combine.py) | Combines partial forward outputs when the snapshot selects split work. |
| [flash_fwd_sm100.py](flash_fwd_sm100.py) | Blackwell forward kernel: staged QK/PV work, softmax and output handling. |
| [fp8_tuning.py](fp8_tuning.py) | Compiler/shape-dependent low-bit tuning choices. |
| [interface.py](interface.py) | Argument handling, compile specialization, dispatch and launch entry points. |
| [mask.py](mask.py) | Attention masks and masked tile handling. |
| [mma_sm100_desc.py](mma_sm100_desc.py) | SM100 MMA operand and instruction descriptor encodings. |
| [named_barrier.py](named_barrier.py) | Named barrier identifiers used by the kernel stages. |
| [pack_gqa.py](pack_gqa.py) | Retained grouped-query packing utilities. |
| [paged_kv.py](paged_kv.py) | Retained paged-KV addressing utilities. |
| [pipeline.py](pipeline.py) | Pipeline states and TMA/MMA producer-consumer synchronization. |
| [seqlen_info.py](seqlen_info.py) | Fixed-length and variable-length sequence metadata. |
| [softmax.py](softmax.py) | Online-softmax state, rescaling and score/probability helpers. |
| [tile_scheduler.py](tile_scheduler.py) | Maps attention work tiles onto GPU execution. |
| [utils.py](utils.py) | Shared layout, tensor-conversion and score-modifier utilities. |
| [v_layout.py](v_layout.py) | FP8 V packing/transposition helpers for eligible forward paths. |


[modified_utils/](modified_utils/README.md) preserves a minimal package namespace.
Retained backward, older-architecture, grouped-query and paged/sparse helpers
are part of the extracted module dependency set. Their presence does not extend
the supported public forward-inference contract.

[The snapshot overview](../../README.md) gives the exact revision.
Read [the version guide](../../../../../../docs/appendix/versions.md) before comparing kernels;
[the source audit](../../../../../../tools/audit_release.py) verifies frozen source hashes.

[Repository](../../../../../../README.md) · [Parent directory](../README.md)
