# Architecture

`api.py` validates public inputs and separates preparation from attention. `_dispatch.py` lazily loads either `_kernels/blackwell/` (Open-VC) or the upstream `flash-attn-4` package (the FlashAttention-4 BF16 reference). `baselines.py` provides the VC-Attention baseline, and `benchmarking/` compares all three.

Open-VC executes one integrated computation:

1. Quantize Q/K per 128-token block and head, and V per sequence and head; keep FP32 descales.
2. Form QK scores with descales folded into score conversion, keeping the running maximum in the scaled domain.
3. Encode probabilities directly as E4M3 codes with ExpCast; the softmax denominator sums the same decoded codes that feed the PV product. The mid-window traversal visits keys near the query's position first, so the running maximum settles early. It skips no blocks.
4. Select the execution schedule for architecture, shape, layout and requested outputs. Eligible shapes pack V and overlap score production, normalization and PV accumulation.
5. Rescale and normalize the accumulated output, restoring Q's shape.

On eligible B200 `attention()` calls, step 1 runs as three fused kernels:

1. Joint Q/K quantization and V-amax initialization.
2. V-amax reduction.
3. V cast, pack and descale.

The packed V tensor is consumed directly by step 4. `prepare_fp8` remains the general preparation, and `preparation="unfused"` opts out of fusion.

`v_repair.py` implements V residual repair:

1. Score tokens by their V quantization residual and select the worst per head.
2. Prepend each selected token's residual as an extra FP8 value row, paired with a duplicate of its key.
3. Visit those rows last, restoring the original softmax denominator before normalization.

This currently runs on long, single-sequence B200 self-attention.

The packed FP8 path requires head dimension 128, query and key lengths >= 32768, compatible descales and layout, no LSE, and CuTe DSL 4.6.2. SM103 additionally requires query length × heads >= 1,048,576. Shorter, causal and LSE calls use the general path. Exact gates are in `_kernels/blackwell/flash_attn/cute/interface.py`.

Both kernel trees contain only the Blackwell (SM100/SM103) forward pass. Shared helpers such as `block_sparse_utils.py`, `paged_kv.py`, `pack_gqa.py` and `mma_sm100_desc.py` are dependencies of that forward kernel, and the advanced `raw_forward` interface exposes more options than the convenience API supports. Kernel files keep their upstream copyright headers.
