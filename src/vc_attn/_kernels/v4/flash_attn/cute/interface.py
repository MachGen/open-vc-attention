# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

# Supported features:
# - BF16 & FP16 & FP8 (E4M3, E5M2) dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128.
# - (hdim_qk, hdim_v) = (192, 128) for Blackwell (i.e. DeepSeek shape)
# - varlen
# - sliding window
# - bwd pass for Ampere (will also run on Hopper/Blackwell, but will be slow)

# Features not supported yet:
# - split (i.e. FlashDecoding)
# - tuned block sizes
# - paged KV
# - append KV to existing KV cache
# - bwd pass optimized for Hopper/Blackwell

import os
import math
from functools import lru_cache
from typing import Optional, Tuple, Callable, Union

import torch


import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from vc_attn._kernels.v4.nvfp4 import e4m3_scale_view
from vc_attn._kernels.v4.flash_attn.cute.fp8_tuning import CUTLASS_DSL_VERSION


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from vc_attn._kernels.v4.flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from vc_attn._kernels.v4.flash_attn.cute import utils
from vc_attn._kernels.v4.flash_attn.cute.cute_dsl_utils import (
    to_cute_tensor,
    to_cute_aux_tensor,
    get_aux_tensor_metadata,
)
from vc_attn._kernels.v4.flash_attn.cute.flash_fwd import FlashAttentionForwardSm90
from vc_attn._kernels.v4.flash_attn.cute.flash_fwd_sm100 import (
    FlashAttentionForwardSm100,
    DescaleTensors,
    SvdCorrectionTensors,
)
from vc_attn._kernels.v4.flash_attn.cute.flash_bwd_preprocess import (
    FlashAttentionBackwardPreprocess,
)
from vc_attn._kernels.v4.flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from vc_attn._kernels.v4.flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from vc_attn._kernels.v4.flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from vc_attn._kernels.v4.flash_attn.cute.flash_bwd_postprocess import (
    FlashAttentionBackwardPostprocess,
)
from vc_attn._kernels.v4.flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine

from vc_attn._kernels.v4.flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
)


def to_cute_nvf4_qk_tensor(t: torch.Tensor) -> cute.Tensor:
    logical_shape = (*t.shape[:-1], t.shape[-1] * 2)
    logical_storage = torch.empty(logical_shape, device=t.device, dtype=torch.uint8)
    tensor = from_dlpack(logical_storage, assumed_align=16, enable_tvm_ffi=False)
    tensor.element_type = cutlass.Float4E2M1FN
    leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim).mark_compact_shape_dynamic(
        mode=leading_dim,
        stride_order=t.dim_order(),
        divisibility=32,
    )


@lru_cache(maxsize=None)
def _get_device_capability():
    """Cached device capability check."""
    return torch.cuda.get_device_capability()[0]


@lru_cache(maxsize=None)
def _get_device_capability_minor():
    """Cached device capability minor (e.g. 3 for sm_103 / Blackwell Ultra)."""
    return torch.cuda.get_device_capability()[1]


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    assert t.is_cuda, f"{name} must be on CUDA"


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
    torch.int8: cutlass.Int8,
}

if hasattr(torch, "float4_e2m1fn_x2"):
    torch2cute_dtype_map[torch.float4_e2m1fn_x2] = cutlass.Float4E2M1FN


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    output_amax: Optional[torch.Tensor] = None,
    output_amax_chunk_seqlen: int = 0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    skip_softmax_error: Union[
        float, torch.Tensor
    ] = 0.0,  # scalar OR (num_heads,) f32 per-head ε; indexed by kernel head_idx (kv-head when pack_gqa)
    skip_pv_gemm: bool = False,  # if True, MMA WG drops P·V WGMMA for tiles all softmax warps voted skip
    skip_counter: Optional[
        torch.Tensor
    ] = None,  # (num_heads, 3) int32: per-head [evaluated, softmax_skipped, pv_skipped]; indexed by the kernel's head_idx (kv-head when pack_gqa). None = disabled (zero overhead)
    skip_counter_sample_q_stride: int = 1,  # Count only every Nth query tile in skip_counter; 1 = all query tiles.
    skip_counter_sample_q_offset: int = 0,  # Query-tile offset for skip_counter sampling.
    skip_softmax_disabled_head_mask: int = 0,  # compile-time bitmask: heads with bit=1 bypass skip predicate while scalar skip remains enabled for the rest
    head_index_map: Optional[
        torch.Tensor
    ] = None,  # Optional int32 CUDA head list. Kernel launches len(list) logical heads and maps each to a real q/kv/o head.
    mid_window_blocks: Optional[
        int
    ] = None,  # SM100 dense path: scan diag±window first, then sweep tail. None = original right-to-left scan.
    svd_raw_q: Optional[
        torch.Tensor
    ] = None,  # SVD-CuTe MVP: raw [B,S,H,Draw] Q for selected exact-score correction.
    svd_raw_k: Optional[torch.Tensor] = None,  # SVD-CuTe MVP: raw [B,S,H,D(raw)] K.
    svd_raw_v: Optional[torch.Tensor] = None,  # Reserved for mixed-PV correction.
    svd_q_mean: Optional[
        torch.Tensor
    ] = None,  # Row bias for proxy background logits; selected exact QK is not biased.
    svd_q_low: Optional[torch.Tensor] = None,  # Optional proxy Q [B,S,H,RK] for direct raw-V delta.
    svd_k_coord: Optional[
        torch.Tensor
    ] = None,  # Optional proxy K coords [B,S,H,RK] for direct raw-V delta.
    svd_v_mean: Optional[torch.Tensor] = None,
    svd_v_basis: Optional[torch.Tensor] = None,
    svd_v_coord: Optional[torch.Tensor] = None,
    svd_topk: int = 0,
    svd_k_block: int = 64,
    mSFQ: Optional[torch.Tensor] = None,
    mSFK: Optional[torch.Tensor] = None,
    expcast: bool = False,
    v_smooth: bool = False,
    v_smooth_means: Optional[torch.Tensor] = None,
    v_smooth_scale: Optional[torch.Tensor] = None,
    fused_skip_softmax: bool = False,
    v_prepacked: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
        output_amax: Optional pre-allocated fp32 buffer for fused global abs-max over O. When set,
            the SM100 epilogue atomically accumulates per-chunk maxima into this 1-D tensor.
            Must be zero-initialized by the caller (or by this function immediately before launch).
        output_amax_chunk_seqlen: When > 0, bucket rows by ``row // chunk_seqlen`` into
            ``output_amax[chunk_id]``. When 0, all rows update ``output_amax[0]``.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
        v_smooth: Restore prepared block means in the online output accumulator.
            Pass the means and per-channel scales from prepare_v_smooth.
        expcast: Encode probabilities directly as E4M3 (VC-Attention Eq. 7).
            Inference-only SM100-family path with FP8 V; defaults to False.
        v_prepacked: Consume the exact padded sequence-contiguous layout from
            prepare_fp8_fused. Unsupported dispatches raise instead of unpacking.
        fused_skip_softmax: Opt in to the fused packed-V FP8 skip-softmax path.
            Requires B200, DSL 4.6.0, an eligible single-sequence varlen call,
            scalar skip_softmax_error in (0, 1], and skip_pv_gemm=False.
            Preserves dense ExpCast rounding on retained blocks and executes
            all PV/denominator MMA operations. Unsupported configurations raise
            ValueError; the default no-skip behavior is unchanged.

    The profiled SM100 FP8 skip path (epsilon 7230, mid window 4, no skip-PV)
    on DSL 4.4.1/4.6.0 normalizes decoded E4M3 probabilities even without
    ExpCast. This adds quantization and local FP16-sum rounding to its denominator.
    Profiled NVFP4 QK / E4M3 PV ExpCast uses packed FP16 code rounding and
    early scale-stage release; unprofiled configurations retain the generic encoder.
    """
    if v_prepacked:
        # Keep the logical [S,H,D] view: sequence is the contiguous dimension.
        # Reject malformed views before any normalization can silently unpack V.
        if v.ndim != 3:
            raise ValueError("v_prepacked requires a single-sequence [S,H,128] view")
        length, heads, dim = v.shape
        padded = (length + 127) // 128 * 128
        if (dim != 128 or v.dtype != torch.float8_e4m3fn
                or tuple(v.stride()) != (1, dim * padded, padded)
                or v.storage_offset() != 0
                or v.untyped_storage().nbytes() < heads * dim * padded):
            raise ValueError("v_prepacked must use the padded sequence-contiguous FP8 layout")
        q, k = [maybe_contiguous(t) for t in (q, k)]
    else:
        q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    mSFQ, mSFK = [maybe_contiguous(t) for t in (mSFQ, mSFK)]
    q_descale, k_descale, v_descale = [
        maybe_contiguous(t) for t in (q_descale, k_descale, v_descale)
    ]
    (
        svd_raw_q,
        svd_raw_k,
        svd_raw_v,
        svd_q_mean,
        svd_q_low,
        svd_k_coord,
        svd_v_mean,
        svd_v_basis,
        svd_v_coord,
    ) = [
        maybe_contiguous(t)
        for t in (
            svd_raw_q,
            svd_raw_k,
            svd_raw_v,
            svd_q_mean,
            svd_q_low,
            svd_k_coord,
            svd_v_mean,
            svd_v_basis,
            svd_v_coord,
        )
    ]
    is_nvf4_qk = mSFQ is not None
    num_head, head_dim_physical = q.shape[-2:]
    head_dim = head_dim_physical * 2 if is_nvf4_qk else head_dim_physical
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim_physical)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim_physical)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim_physical)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int8,
        torch.float4_e2m1fn_x2 if hasattr(torch, "float4_e2m1fn_x2") else None,
    ], "inputs must be float16, bfloat16, fp8 e4m3fn, fp8 e5m2, int8, or fp4"
    if is_nvf4_qk:
        assert hasattr(torch, "float4_e2m1fn_x2") and q.dtype == torch.float4_e2m1fn_x2, (
            "mSFQ/mSFK require packed NVFP4 q/k tensors"
        )
        assert k.dtype == torch.float4_e2m1fn_x2, "k must be packed NVFP4 when mSFQ is provided"
        assert mSFK is not None, "mSFK must be provided with mSFQ"
        assert v.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ), "v must be fp16, bf16, or fp8 for QK NVFP4"
    is_int8 = q.dtype == torch.int8
    if is_int8:
        assert k.dtype == torch.int8, "k must be int8 when q is int8"
        assert v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
            "v must be fp8 when q/k are int8"
        )
    elif not is_nvf4_qk:
        assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    assert all(
        t is None or t.is_cuda
        for t in (
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
            svd_raw_q,
            svd_raw_k,
            svd_raw_v,
            svd_q_mean,
            svd_q_low,
            svd_k_coord,
            svd_v_mean,
            svd_v_basis,
            svd_v_coord,
            mSFQ,
            mSFK,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if expcast and (not math.isfinite(softmax_scale) or softmax_scale <= 0.0):
        raise ValueError("ExpCast requires a positive finite softmax_scale")
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    is_quantized = is_fp8 or is_int8 or is_nvf4_qk
    if is_quantized and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise NotImplementedError(
            "FA4 CuTe quantized (FP8/INT8) backward is not supported yet (forward-only)."
        )
    if is_nvf4_qk:
        assert cu_seqlens_q is None and cu_seqlens_k is None, (
            "QK NVFP4 currently supports fixed-length tensors only"
        )
        assert seqused_q is None and seqused_k is None, (
            "QK NVFP4 does not support seqused_q/seqused_k yet"
        )
        assert page_table is None, "QK NVFP4 does not support paged KV yet"
        assert not pack_gqa, "QK NVFP4 does not support PackGQA yet"
        q_sf_seqlen = ((q.shape[1] + 127) // 128) * 128
        k_sf_seqlen = ((k.shape[1] + 127) // 128) * 128
        for t, name, expected_shape in (
            (mSFQ, "mSFQ", (q.shape[0], q_sf_seqlen, q.shape[2], head_dim // 16)),
            (mSFK, "mSFK", (k.shape[0], k_sf_seqlen, k.shape[2], head_dim // 16)),
        ):
            assert t.dtype in (torch.float8_e4m3fn, torch.int8), f"{name} must be E4M3-backed"
            assert t.shape == expected_shape, (
                f"{name} shape {tuple(t.shape)} != expected {expected_shape}"
            )
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are not used with QK NVFP4 scale tensors"
        )
    out_torch_dtype = torch.bfloat16 if is_quantized else q.dtype
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(
            out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device
        )

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if output_amax is not None:
        assert output_amax.dtype == torch.float32, "output_amax must be float32"
        assert output_amax.dim() == 1, "output_amax must be 1-D"
        assert output_amax.is_contiguous(), "output_amax must be contiguous"
        assert output_amax.device == device, "output_amax device mismatch"
        output_amax.zero_()

    if is_quantized and not is_nvf4_qk:
        # Descales are per-tensor-per-head (2D: (B, H_kv)) or per-block
        # (3D: (B, H_?, num_blocks)). V is always 2D.
        # Per-block Q uses the Q-head dim (so its middle dim is num_head);
        # per-block K uses the KV-head dim.
        for t, name, head_dim_expected in (
            (q_descale, "q_descale", num_head),
            (k_descale, "k_descale", num_head_kv),
        ):
            if t is not None:
                assert t.dtype == torch.float32, f"{name} must be float32"
                assert t.device == device, f"{name} device mismatch"
                if t.dim() == 2:
                    assert t.shape == (batch_size, num_head_kv), (
                        f"{name} shape {tuple(t.shape)} != expected {(batch_size, num_head_kv)}"
                    )
                elif t.dim() == 3:
                    assert t.shape[0] == batch_size and t.shape[1] == head_dim_expected, (
                        f"{name} 3D shape {tuple(t.shape)} must start with "
                        f"({batch_size}, {head_dim_expected}, num_blocks)"
                    )
                else:
                    raise AssertionError(f"{name} must be 2D or 3D, got {t.dim()}D")
        if v_descale is not None:
            _validate_tensor(
                v_descale, "v_descale", (batch_size, num_head_kv), torch.float32, device
            )
    elif not is_quantized:
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are only supported for quantized (FP8/INT8) inputs"
        )

    dtype = torch2cute_dtype_map[q.dtype]
    q_ptr_shape = (*q.shape[:-1], head_dim) if is_nvf4_qk else ()
    k_ptr_shape = (*k.shape[:-1], head_dim) if is_nvf4_qk else ()
    compute_capability = (
        _get_device_capability() if _compute_capability is None else _compute_capability
    )

    assert compute_capability in [9, 10, 11], (
        "Unsupported compute capability. Supported: 9.x, 10.x, 11.x"
    )
    if is_quantized:
        assert compute_capability == 10, (
            "FP8/INT8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
        )
    if v_smooth:
        if not (
            compute_capability == 10
            and _get_device_capability_minor() in (0, 3)
            and q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
            and head_dim == head_dim_v == 128
            and num_head == num_head_kv
            and m_block_size == n_block_size == 128
            and num_splits == 1
            and not causal
            and window_size_left is None
            and window_size_right is None
            and score_mod is None
            and mask_mod is None
            and softcap is None
            and page_table is None
            and block_sparse_tensors is None
            and learnable_sink is None
            and head_index_map is None
            and seqused_q is None
            and seqused_k is None
            and svd_raw_q is None
            and q_descale is None
            and k_descale is None
            and v_descale is None
            and aux_tensors is None
            and not skip_pv_gemm
            and not isinstance(skip_softmax_error, torch.Tensor)
            and skip_softmax_error == 0
            and (cu_seqlens_q is None or batch_size == 1)
            and (cu_seqlens_k is None or batch_size == 1)
        ):
            raise ValueError(
                "V-Smooth requires SM100/SM103 dense D128 E4M3 attention without skip, masks, GQA or external descales"
            )
        for tensor, shape, dtypes, name in (
            (
                v_smooth_means,
                (batch_size, math.ceil(seqlen_k / 128), num_head, 128),
                (torch.bfloat16, torch.float32),
                "v_smooth_means",
            ),
            (v_smooth_scale, (batch_size, num_head, 128), (torch.float32,), "v_smooth_scale"),
        ):
            if (
                tensor is None
                or tensor.shape != shape
                or tensor.dtype not in dtypes
                or tensor.device != q.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16 != 0
            ):
                raise ValueError(
                    f"{name} must be contiguous, 16-byte aligned {dtypes} on {q.device}, shape {shape}"
                )
        aux_tensors = [v_smooth_means, v_smooth_scale]
    elif v_smooth_means is not None or v_smooth_scale is not None:
        raise ValueError("V-Smooth metadata requires v_smooth=True")
    if expcast:
        if compute_capability != 10 or not is_quantized or v.dtype != torch.float8_e4m3fn:
            raise ValueError("ExpCast requires SM100-family quantized Q/K and E4M3 V")
        if svd_raw_q is not None:
            raise ValueError("ExpCast does not support SVD score/output correction")
    if skip_counter is not None:
        assert compute_capability == 10, (
            "skip_counter diagnostic is only wired into the SM100 kernel."
        )
        # Indexed in the kernel by head_idx. For non-pack_gqa that's the q-head;
        # under pack_gqa it's the kv-head. Caller sizes leading dim accordingly.
        assert (
            skip_counter.dim() == 2
            and skip_counter.shape[-1] == 3
            and skip_counter.dtype == torch.int32
        ), (
            f"skip_counter must be (num_heads, 3) int32, got shape={tuple(skip_counter.shape)} dtype={skip_counter.dtype}"
        )
        assert skip_counter.is_contiguous(), "skip_counter must be contiguous"
    skip_counter_sample_q_stride = int(skip_counter_sample_q_stride)
    skip_counter_sample_q_offset = int(skip_counter_sample_q_offset)
    assert skip_counter_sample_q_stride >= 1, "skip_counter_sample_q_stride must be >= 1"
    assert skip_counter_sample_q_offset >= 0, "skip_counter_sample_q_offset must be >= 0"
    if skip_counter_sample_q_stride == 1:
        skip_counter_sample_q_offset = 0
    else:
        skip_counter_sample_q_offset %= skip_counter_sample_q_stride
    if mid_window_blocks is not None:
        assert compute_capability in [10, 11], (
            "mid_window_blocks is only wired into the SM100 kernel."
        )
    skip_eps_is_tensor = isinstance(skip_softmax_error, torch.Tensor)
    if skip_eps_is_tensor:
        assert compute_capability == 10, (
            "Per-head skip_softmax_error is only wired into the SM100 kernel."
        )
        assert skip_softmax_error.dim() == 1 and skip_softmax_error.dtype == torch.float32, (
            f"per-head skip_softmax_error must be 1-D float32 of shape (num_heads,), "
            f"got shape={tuple(skip_softmax_error.shape)} dtype={skip_softmax_error.dtype}"
        )
        assert skip_softmax_error.is_contiguous() and skip_softmax_error.device.type == "cuda"
    assert (
        isinstance(skip_softmax_disabled_head_mask, int) and skip_softmax_disabled_head_mask >= 0
    ), "skip_softmax_disabled_head_mask must be a non-negative Python int"
    head_index_count = 0
    if head_index_map is not None:
        assert compute_capability == 10, "head_index_map is only wired into the SM100 kernel."
        assert not pack_gqa, "head_index_map MVP only supports non-pack-GQA."
        assert cu_seqlens_q is None and cu_seqlens_k is None, (
            "head_index_map MVP only supports fixed-length batch tensors."
        )
        assert head_index_map.dim() == 1 and head_index_map.dtype == torch.int32, (
            f"head_index_map must be 1-D int32, got shape={tuple(head_index_map.shape)} dtype={head_index_map.dtype}"
        )
        assert head_index_map.is_contiguous() and head_index_map.device.type == "cuda"
        head_index_count = int(head_index_map.numel())
        assert head_index_count > 0, "head_index_map must not be empty"
        assert (
            int(head_index_map.min().item()) >= 0 and int(head_index_map.max().item()) < num_head
        ), f"head_index_map values must be in [0, {num_head})"

    has_svd_correction = svd_raw_q is not None
    svd_raw_head_dim = 0
    if has_svd_correction:
        assert compute_capability == 10, "SVD-CuTe correction is only wired into the SM100 kernel."
        assert svd_raw_k is not None and svd_raw_v is not None and svd_q_mean is not None, (
            "SVD-CuTe correction requires raw Q/K/V tensors and q_mean."
        )
        assert cu_seqlens_q is None and cu_seqlens_k is None, (
            "SVD-CuTe correction MVP supports fixed-length tensors only."
        )
        assert page_table is None and seqused_q is None and seqused_k is None, (
            "SVD-CuTe correction MVP does not support paged/used-length tensors."
        )
        assert not pack_gqa, "SVD-CuTe correction MVP supports non-pack-GQA only."
        assert int(svd_topk) > 0, "svd_topk must be positive when SVD-CuTe correction is enabled."
        assert int(svd_k_block) > 0, "svd_k_block must be positive."
        _validate_tensor(
            svd_raw_q,
            "svd_raw_q",
            (batch_size, seqlen_q, num_head, svd_raw_q.shape[-1]),
            svd_raw_q.dtype,
            device,
        )
        assert svd_raw_q.shape[:3] == (batch_size, seqlen_q, num_head), (
            "svd_raw_q must be [B, S_q, H, D_raw]"
        )
        assert svd_raw_k.shape[:3] == (batch_size, seqlen_k, num_head_kv), (
            "svd_raw_k must be [B, S_k, H_kv, D_raw]"
        )
        assert svd_raw_v.shape[:3] == (batch_size, seqlen_k, num_head_kv), (
            "svd_raw_v must be [B, S_k, H_kv, D_raw]"
        )
        _validate_tensor(
            svd_q_mean, "svd_q_mean", (batch_size, seqlen_q, num_head), torch.float32, device
        )
        assert svd_raw_q.shape[-1] == svd_raw_k.shape[-1], "raw Q/K head dims must match"
        assert svd_raw_v.shape[-1] == svd_raw_q.shape[-1], (
            "raw V head dim must match raw Q/K for the mixed-PV MVP"
        )
        assert svd_raw_q.dtype in (torch.float16, torch.bfloat16, torch.float32), (
            "raw SVD Q/K/V must be fp16/bf16/fp32"
        )
        assert svd_raw_k.dtype == svd_raw_q.dtype and svd_raw_v.dtype == svd_raw_q.dtype, (
            "raw SVD Q/K/V dtypes must match"
        )
        svd_raw_head_dim = int(svd_raw_q.shape[-1])
        if svd_v_mean is not None:
            assert svd_v_basis is not None and svd_v_coord is not None, (
                "SVD mixed-PV tensors must be provided together."
            )
            assert svd_q_low is not None and svd_k_coord is not None, (
                "SVD direct raw-V delta requires q_low and k_coord."
            )
            _validate_tensor(
                svd_q_low,
                "svd_q_low",
                (batch_size, seqlen_q, num_head, head_dim),
                svd_q_low.dtype,
                device,
            )
            _validate_tensor(
                svd_k_coord,
                "svd_k_coord",
                (batch_size, seqlen_k, num_head_kv, head_dim),
                svd_k_coord.dtype,
                device,
            )
            _validate_tensor(
                svd_v_mean, "svd_v_mean", (num_head_kv, svd_raw_head_dim), torch.float32, device
            )
            _validate_tensor(
                svd_v_basis,
                "svd_v_basis",
                (num_head_kv, svd_raw_head_dim, svd_v_basis.shape[-1]),
                torch.float32,
                device,
            )
            _validate_tensor(
                svd_v_coord,
                "svd_v_coord",
                (batch_size, seqlen_k, num_head_kv, svd_v_basis.shape[-1]),
                svd_v_coord.dtype,
                device,
            )
            assert svd_q_low.dtype == q.dtype and svd_k_coord.dtype == k.dtype, (
                "SVD proxy tensors must match FA4 Q/K dtype."
            )
            assert svd_v_coord.dtype in (torch.float16, torch.bfloat16, torch.float32), (
                "svd_v_coord must be fp16/bf16/fp32."
            )

    use_block_sparsity = block_sparse_tensors is not None

    if mask_mod is None:
        if causal:
            window_size_right = 0
        local = window_size_left is not None or window_size_right is not None
        if window_size_left is not None or window_size_right is not None:
            if window_size_left is None and window_size_right == 0:
                causal, local = True, False
                window_size_right = None
            else:
                causal, local = False, True
    else:
        causal, local = False, False

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim.
        if head_dim == head_dim_v == 128 and not causal and not local and not use_block_sparsity:
            n_block_size = 192

    if compute_capability in [10, 11]:
        if pack_gqa and (128 % qhead_per_kvhead != 0):
            pack_gqa = False
        # TODO: fix GQA + SplitKV + non-varlen
        if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
            pack_gqa = False

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if compute_capability == 10:
        q_stage = 2 if seqlen_q_packgqa > m_block_size else 1
    else:
        q_stage = 1

    if v_smooth and q_stage != 2:
        raise ValueError("V-Smooth requires the two-query-stage kernel")
    if expcast and is_nvf4_qk and q_stage == 1:
        raise ValueError("ExpCast NVFP4 requires the two-stage query kernel")

    if num_splits < 1:
        m_block_size_effective = q_stage * m_block_size
        seqlen_k_loaded = (
            max_seqlen_k
            if not local
            else max(0, min(max_seqlen_k, window_size_right + window_size_left + 1 + m_block_size))
        )
        num_n_blocks = (seqlen_k_loaded + n_block_size - 1) // n_block_size
        num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_splits = num_splits_heuristic(
            total_mblocks,
            torch.cuda.get_device_properties(device).multi_processor_count,
            num_n_blocks,
            128,
        )

    is_split_kv = num_splits > 1
    if head_index_map is not None:
        assert not is_split_kv, "head_index_map MVP only supports non-SplitKV."
    if is_split_kv:
        out_partial = torch.empty(
            num_splits,
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=torch.float32,
            device=device,
        )
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    if mask_mod is not None:
        if is_varlen:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[1] != 1:
            pack_gqa = False
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if block_sparse_tensors is not None:
        if seqlen_q is None:
            raise ValueError(
                "Block sparsity requires fixed-length sequences (seqlen_q must be known)."
            )
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
            q_subtile_factor,
        ) = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            q_stage=q_stage,
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None

    # Repacking pays off only for large dense calls. Ragged K starts can violate
    # the K-major TMA alignment, so multi-sequence varlen keeps the original V.
    fp8_varlen_expcast_packing = (
        expcast
        and _get_device_capability_minor() in (0, 3)
        and CUTLASS_DSL_VERSION == "4.6.0"
        and q.dtype == k.dtype == torch.float8_e4m3fn
        and cu_seqlens_q is not None
        and lse is None
    )
    if fused_skip_softmax:
        if not (
            fp8_varlen_expcast_packing
            and _get_device_capability_minor() == 0
            and not skip_eps_is_tensor
            and 0.0 < float(skip_softmax_error) <= 1.0
            and not skip_pv_gemm
        ):
            raise ValueError(
                "fused_skip_softmax requires B200, DSL 4.6.0, varlen FP8 ExpCast, "
                "scalar epsilon in (0,1], and skip_pv_gemm=False"
            )
    # Long single-sequence B200 FP8 calls can amortize packing even at low H.
    # Keep both length floors and preserve the historical work threshold for
    # other devices/modes, including the separately tuned fused-skip path.
    sm100_fp8_dense = (
        fp8_varlen_expcast_packing
        and compute_capability == 10
        and _get_device_capability_minor() == 0
        and batch_size == 1
        and not skip_eps_is_tensor
        and skip_softmax_error == 0.0
    )
    packing_size_eligible = (
        max_seqlen_q >= 32768
        and max_seqlen_k >= 32768
        and (sm100_fp8_dense or total_q * num_head >= 1048576)
    )
    packed_v = (
        expcast
        and packing_size_eligible
        and all(
            (
                not v_smooth,
                compute_capability == 10,
                (
                    _get_device_capability_minor() == 0
                    or (
                        _get_device_capability_minor() == 3
                        and not is_nvf4_qk
                        and cu_seqlens_q is not None
                        and CUTLASS_DSL_VERSION == "4.6.0"
                    )
                ),
                CUTLASS_DSL_VERSION in ("4.4.1", "4.6.0"),
                v.dtype == torch.float8_e4m3fn
                and (q.dtype == k.dtype == torch.float8_e4m3fn or is_nvf4_qk),
                head_dim == head_dim_v == 128,
                m_block_size == n_block_size == 128,
                q_stage == 2,
                qhead_per_kvhead == 1,
                not use_block_sparsity,
                not causal,
                not local,
                not is_split_kv,
                page_table is None,
                cu_seqlens_k is None or batch_size == 1,
                seqused_q is None,
                seqused_k is None,
                v_prepacked or v.is_contiguous(),
                score_mod is None,
                mask_mod is None,
                svd_topk == 0,
                learnable_sink is None,
                head_index_map is None,
                (
                    (q_descale is None and k_descale is None and v_descale is None)
                    or fp8_varlen_expcast_packing
                ),
                not skip_pv_gemm,
                not skip_eps_is_tensor,
                not skip_softmax_disabled_head_mask,
                mid_window_blocks == 4
                or (fp8_varlen_expcast_packing and mid_window_blocks is None)
                or (
                    mid_window_blocks is not None
                    and mid_window_blocks >= 0
                    and sm100_fp8_dense
                ),
            )
        )
        and (
            not skip_eps_is_tensor
            and (
                skip_softmax_error == 0.0
                or fused_skip_softmax
                or (is_nvf4_qk and skip_softmax_error == 7230.0)
            )
        )
    )
    if fused_skip_softmax and not packed_v:
        raise ValueError("fused_skip_softmax requires the eligible dense packed-V path")
    if v_prepacked and not packed_v:
        raise ValueError("v_prepacked requires the eligible dense packed-V path")
    if packed_v and not v_prepacked:
        from vc_attn._kernels.v4.flash_attn.cute.v_layout import pack_v

        v = pack_v(v)

    inline_rescale = (
        packed_v
        and not is_nvf4_qk
        and cu_seqlens_q is not None
        and batch_size == 1
        and CUTLASS_DSL_VERSION == "4.6.0"
    )
    v_dtype = torch2cute_dtype_map[v.dtype]
    compile_key = (
        dtype,
        v_dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        output_amax is None,
        output_amax_chunk_seqlen if output_amax is not None else 0,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        (q_descale.dim() if q_descale is not None else None),
        (k_descale.dim() if k_descale is not None else None),
        v_descale is not None,
        m_block_size,
        n_block_size,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        compute_capability,
        page_size not in [None, 128],  # paged KV non-TMA
        q_subtile_factor,
        # baked in at compile time: scalar ε (sentinel 1.0 when per-head, real value otherwise),
        # plus a flag for the per-head compiled variant.
        1.0 if skip_eps_is_tensor else float(skip_softmax_error),
        skip_eps_is_tensor,
        fused_skip_softmax,
        skip_pv_gemm,  # baked in at compile time (MMA-WG skip branch on/off)
        skip_counter is not None,  # diag counter present / absent → separate compiled variant
        skip_counter_sample_q_stride if skip_counter is not None else 1,
        skip_counter_sample_q_offset if skip_counter is not None else 0,
        mid_window_blocks,  # baked in at compile time; None disables mid-out scan
        skip_softmax_disabled_head_mask,  # baked in at compile time; no tensor load
        head_index_count,  # baked in: launch only this many logical heads when nonzero
        has_svd_correction,
        int(svd_topk) if has_svd_correction else 0,
        int(svd_k_block) if has_svd_correction else 0,
        svd_raw_head_dim,
        is_nvf4_qk,
        expcast,
        v_smooth,
        packed_v,
        inline_rescale,
    )
    v_smooth_prefetch = v_smooth and cu_seqlens_q is not None and seqlen_k >= 4096
    v_smooth_head_major = (
        v_smooth_prefetch
        and seqlen_k >= 49152
        and q.shape[-2] == 40
        and k.stride(0) == 128
        and v.stride(0) == 128
    )
    if v_smooth:
        compile_key += (v_smooth_means.dtype, v_smooth_prefetch, v_smooth_head_major)
    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        skip_counter_tensor = (
            to_cute_tensor(skip_counter, assumed_align=4) if skip_counter is not None else None
        )
        skip_eps_tensor = (
            to_cute_tensor(skip_softmax_error, assumed_align=4) if skip_eps_is_tensor else None
        )
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        if is_nvf4_qk:
            from cutlass.cute.runtime import make_ptr

            q_tensor = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
            k_tensor = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
        else:
            q_tensor = to_cute_tensor(q)
            k_tensor = to_cute_tensor(k)
        v_tensor = to_cute_tensor(v, leading_dim=v.ndim - 3 if packed_v else -1)
        o_tensor = to_cute_tensor(out if not is_split_kv else out_partial)
        mSFQ_tensor = (
            to_cute_tensor(e4m3_scale_view(mSFQ), leading_dim=3, assumed_align=16)
            if mSFQ is not None
            else None
        )
        mSFK_tensor = (
            to_cute_tensor(e4m3_scale_view(mSFK), leading_dim=3, assumed_align=16)
            if mSFK is not None
            else None
        )
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        if output_amax is not None:
            output_amax_tensor = to_cute_tensor(output_amax.view(torch.int32), assumed_align=4)
        else:
            output_amax_tensor = None

        q_descale_tensor = (
            to_cute_tensor(q_descale, assumed_align=4, leading_dim=-1)
            if q_descale is not None
            else None
        )
        k_descale_tensor = (
            to_cute_tensor(k_descale, assumed_align=4, leading_dim=-1)
            if k_descale is not None
            else None
        )
        v_descale_tensor = (
            to_cute_tensor(v_descale, assumed_align=4, leading_dim=-1)
            if v_descale is not None
            else None
        )
        descale_tensors_tensor = (
            DescaleTensors(
                q_descale=q_descale_tensor,
                k_descale=k_descale_tensor,
                v_descale=v_descale_tensor,
            )
            if q_descale_tensor is not None
            or k_descale_tensor is not None
            or v_descale_tensor is not None
            else None
        )

        svd_tensors_tensor = None
        if has_svd_correction:
            svd_tensors_tensor = SvdCorrectionTensors(
                raw_q=to_cute_tensor(svd_raw_q, assumed_align=16, leading_dim=-1),
                raw_k=to_cute_tensor(svd_raw_k, assumed_align=16, leading_dim=-1),
                raw_v=to_cute_tensor(svd_raw_v, assumed_align=16, leading_dim=-1),
                q_mean=to_cute_tensor(svd_q_mean, assumed_align=4, leading_dim=-1),
                q_low=to_cute_tensor(svd_q_low, assumed_align=16, leading_dim=-1)
                if svd_q_low is not None
                else None,
                k_coord=to_cute_tensor(svd_k_coord, assumed_align=16, leading_dim=-1)
                if svd_k_coord is not None
                else None,
                v_mean=to_cute_tensor(svd_v_mean, assumed_align=16, leading_dim=-1)
                if svd_v_mean is not None
                else None,
                v_basis=to_cute_tensor(svd_v_basis, assumed_align=16, leading_dim=-1)
                if svd_v_basis is not None
                else None,
                v_coord=to_cute_tensor(svd_v_coord, assumed_align=16, leading_dim=-1)
                if svd_v_coord is not None
                else None,
            )

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [
                to_cute_tensor(buf) if v_smooth else to_cute_aux_tensor(buf) for buf in aux_tensors
            ]

        if compute_capability == 9:
            assert page_table is None, "paged KV not supported on SM 9.0"
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            # fa_fwd = FlashAttentionForwardSm80(
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=m_block_size,
                tile_n=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
            )
        elif compute_capability in [10, 11]:
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                is_sm103=compute_capability == 10 and _get_device_capability_minor() == 3,
                use_sm100_schedule=v_smooth or packed_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                q_stage=q_stage,
                is_persistent=not causal
                and not local
                and cu_seqlens_q is None
                and seqused_q is None
                and not is_split_kv,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                paged_kv_non_tma=page_size not in [None, 128],
                is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                q_subtile_factor=q_subtile_factor,
                skip_softmax_error=1.0 if skip_eps_is_tensor else float(skip_softmax_error),
                skip_softmax_error_per_head=skip_eps_is_tensor,
                fused_skip_softmax=fused_skip_softmax,
                skip_pv_gemm=skip_pv_gemm,
                expcast=expcast,
                v_smooth=v_smooth,
                v_smooth_prefetch=v_smooth_prefetch,
                v_smooth_head_major=v_smooth_head_major,
                inline_rescale=inline_rescale,
                skip_counter_sample_q_stride=skip_counter_sample_q_stride
                if skip_counter is not None
                else 1,
                skip_counter_sample_q_offset=skip_counter_sample_q_offset
                if skip_counter is not None
                else 0,
                skip_softmax_disabled_head_mask=skip_softmax_disabled_head_mask,
                head_index_count=head_index_count,
                use_block_sparsity=use_block_sparsity,
                mid_window_blocks=mid_window_blocks,
                svd_topk=int(svd_topk) if has_svd_correction else 0,
                svd_k_block=int(svd_k_block) if has_svd_correction else 64,
                svd_raw_head_dim=svd_raw_head_dim,
                svd_v_rank=int(svd_v_basis.shape[-1])
                if has_svd_correction and svd_v_basis is not None
                else 0,
                svd_num_k_blocks=math.ceil(seqlen_k / int(svd_k_block))
                if has_svd_correction
                else 0,
                svd_delta_to_output=has_svd_correction
                and svd_v_mean is not None
                and head_dim_v == svd_raw_head_dim,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x, 11.x"
            )
        # TODO: check @can_implement
        compile_args = [
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
        ]
        if compute_capability in [10, 11]:
            compile_args.append(descale_tensors_tensor)
        compile_args.extend([sparse_tensors, cute_aux_tensors])
        if compute_capability in [10, 11]:
            compile_args.append(skip_counter_tensor)
            compile_args.append(skip_eps_tensor)
            compile_args.append(
                to_cute_tensor(head_index_map, assumed_align=4)
                if head_index_map is not None
                else None
            )
            compile_args.append(svd_tensors_tensor)
            compile_args.append(output_amax_tensor)
            compile_args.append(int(output_amax_chunk_seqlen) if output_amax is not None else 0)
            compile_args.append(mSFQ_tensor)
            compile_args.append(mSFK_tensor)
            if is_nvf4_qk:
                compile_args.append(tuple(cutlass.Int32(0) for _ in q_ptr_shape))
                compile_args.append(tuple(cutlass.Int32(0) for _ in k_ptr_shape))
        compiled = cute.compile(
            *compile_args,
            options="--enable-tvm-ffi",
        )
        # D interleaves score loads and max leaves in the measured B200 binary.
        # Only verified scan-offset immediate fields may differ; the adapter
        # verifies the remaining complete native code and exact metadata.
        # Other compiler/specialization layouts keep their original code.
        if (
            compute_capability == 10
            and _get_device_capability_minor() == 0
            and CUTLASS_DSL_VERSION == "4.6.0"
            and is_fp8
            and packed_v
            and inline_rescale
            and expcast
            and mid_window_blocks is not None
            and mid_window_blocks >= 0
            and not skip_eps_is_tensor
            and skip_softmax_error == 0.0
            and not skip_pv_gemm
            and all(t is not None for t in (q_descale, k_descale, v_descale))
        ):
            from vc_attn._sass_runtime import maybe_patch_compiled

            compiled = maybe_patch_compiled(compiled, mid_window_blocks=mid_window_blocks)
        _flash_attn_fwd.compile_cache[compile_key] = compiled

    v_call = v.detach()
    if is_nvf4_qk:
        from cutlass.cute.runtime import make_ptr as _make_ptr

        q_call = _make_ptr(
            cutlass.Float4E2M1FN,
            q.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        k_call = _make_ptr(
            cutlass.Float4E2M1FN,
            k.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        q_call, k_call = q.detach(), k.detach()
    if is_fp8:
        q_call = q_call.view(torch.uint8)
        k_call = k_call.view(torch.uint8)
        v_call = v_call.view(torch.uint8)
    elif is_nvf4_qk and v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        v_call = v_call.view(torch.uint8)
    elif is_int8:
        # INT8 Q/K stay as int8 (CuTe expects int8, not uint8).
        # V is FP8 and needs uint8 view.
        v_call = v_call.view(torch.uint8)
    mSFQ_call = mSFQ.detach().view(torch.uint8) if is_nvf4_qk else None
    mSFK_call = mSFK.detach().view(torch.uint8) if is_nvf4_qk else None
    descale_tensors = (
        DescaleTensors(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
        if q_descale is not None or k_descale is not None or v_descale is not None
        else None
    )
    svd_tensors = (
        SvdCorrectionTensors(
            raw_q=svd_raw_q,
            raw_k=svd_raw_k,
            raw_v=svd_raw_v,
            q_mean=svd_q_mean,
            q_low=svd_q_low,
            k_coord=svd_k_coord,
            v_mean=svd_v_mean,
            v_basis=svd_v_basis,
            v_coord=svd_v_coord,
        )
        if has_svd_correction
        else None
    )

    call_args = [
        q_call,
        k_call,
        v_call,
        out.detach() if not is_split_kv else out_partial,
        lse_partial if is_split_kv else lse,
        softmax_scale,
        current_stream,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        window_size_left,
        window_size_right,
        learnable_sink,
    ]
    if compute_capability in [10, 11]:
        call_args.append(descale_tensors)
    call_args.extend(
        [
            normalized_block_sparse_tensors[:4]
            if normalized_block_sparse_tensors is not None
            else None,
            aux_tensors,
        ]
    )
    if compute_capability in [10, 11]:
        call_args.append(skip_counter)
        call_args.append(skip_softmax_error if skip_eps_is_tensor else None)
        call_args.append(head_index_map)
        call_args.append(svd_tensors)
        call_args.append(output_amax.view(torch.int32) if output_amax is not None else None)
        call_args.append(int(output_amax_chunk_seqlen) if output_amax is not None else 0)
        call_args.append(mSFQ_call)
        call_args.append(mSFK_call)
        if is_nvf4_qk:
            call_args.append(q_ptr_shape)
            call_args.append(k_ptr_shape)
    _flash_attn_fwd.compile_cache[compile_key](*call_args)
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_capability = _get_device_capability()
    assert compute_capability in [9, 10, 11], (
        "Unsupported compute capability. Supported: 9.x, 10.x, 11.x"
    )

    if compute_capability == 9:
        m_block_size = 80 if not causal else 64
        n_block_size = 128
        num_stages_Q = 2
        num_stages_dO = 2
        num_stages_PdS = 2
        SdP_swapAB = True
        dKV_swapAB = False
        dQ_swapAB = not causal
        AtomLayoutMSdP = 1
        AtomLayoutNdKV = 2
        AtomLayoutMdQ = 1
        cluster_size = 1
        assert window_size_left is None and window_size_right is None, (
            "local not supported yet on 9.x"
        )
        is_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is not None
        )
        assert not is_varlen, "varlen backward is not yet supported on sm90"
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        # TODO: support cluster size 2
        cluster_size = 1
    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = k.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    if causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if local:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True

    use_block_sparsity = block_sparse_tensors is not None

    # SM90 block-sparse backward: tile_m=64 is the GCD between a m_block_size that fits,
    # the base block_m of 128 from forward, and block-sparse size for subtiling.
    if compute_capability == 9 and use_block_sparsity:
        m_block_size = 64
        # dQ_swapAB tuning: use False when m_block_size=64 (same as causal case)
        dQ_swapAB = False

    # NB: this could be derived from the block_sparse_tensors but for now we hardcode it to 2
    subtile_factor = 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    assert all(
        t is None or t.is_cuda for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    # pack_gqa backward not yet supported in bwd
    pack_gqa = False
    if compute_capability not in [10, 11]:
        assert deterministic is False, "bwd deterministic only supported for sm100/sm110 for now"

    if score_mod is not None:
        assert score_mod_bwd is not None, "score_mod_bwd is required when score_mod is provided"
        assert softcap == 0.0, (
            "softcap and score_mod are mutually exclusive (different log2 scaling)"
        )
        assert cu_seqlens_q is None and cu_seqlens_k is None, (
            "varlen + score_mod not supported in bwd yet"
        )

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        dq_accum = torch.empty(
            batch_size,
            num_head,
            seqlen_q_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = torch.empty(
            num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    dKV_postprocess = qhead_per_kvhead > 1
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            num_n_blocks = seqlen_k_rounded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                seqlen_k_rounded = seqlen_k_rounded + n_block_size
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * n_block_size - 1) // n_block_size * n_block_size
            )
            num_n_blocks = total_k_rounded_padded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                total_k_rounded_padded = total_k_rounded_padded + n_block_size
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if deterministic:
        dQ_semaphore = torch.zeros(
            batch_size,
            num_head,
            seqlen_q_rounded // m_block_size,
            1,
            dtype=torch.int32,
            device="cuda",
        )
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(
            batch_size,
            num_head_kv,
            seqlen_k_rounded // n_block_size,
            2,
            dtype=torch.int32,
            device="cuda",
        )
        dV_semaphore = torch.zeros(
            batch_size,
            num_head_kv,
            seqlen_k_rounded // n_block_size,
            2,
            dtype=torch.int32,
            device="cuda",
        )
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1), lse * log2_e, and zero out dq_accum.
    compile_key_pre = (
        compute_capability,
        dtype,
        head_dim_v,
        m_block_size,
        num_threads,
        cu_seqlens_q is None,
        seqused_q is None,
    )
    if compile_key_pre not in _flash_attn_bwd.compile_cache_pre:
        o_tensor, do_tensor = [to_cute_tensor(t) for t in (out, dout)]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        cu_seqlens_q_tensor, seqused_q_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, seqused_q)
        ]
        arch = compute_capability * 10
        fa_bwd_pre = FlashAttentionBackwardPreprocess(
            dtype,
            head_dim_v,
            arch,
            m_block_size,
            num_threads=num_threads,
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            o_tensor,
            do_tensor,
            dpsum_tensor,
            lse_tensor,
            lse_log2_tensor,
            dq_accum_tensor,
            cu_seqlens_q_tensor,
            seqused_q_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_bwd.compile_cache_pre[compile_key_pre](
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        cu_seqlens_q,
        seqused_q,
        current_stream,
    )

    # NB num_threads application for 3 kernels
    # There are pre, main, post processing kernels, currenlty num_threads is only actually
    # used for the pre proc, and then we hard code to 384 for the main and post proc, and we do
    # before cache key gen
    num_threads = 384

    # Backward kernel: compute dk, dv, dq_accum.
    score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [
            to_cute_tensor(buf, assumed_align=None, fully_dynamic=True) for buf in aux_tensors
        ]

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            subtile_factor=subtile_factor,
        )

    if compute_capability == 9:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
        )
    else:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            deterministic,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
        )
    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [to_cute_tensor(t) for t in (dk_accum, dv_accum)]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(
                t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order()
            )
            if t is not None
            else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        fa_bwd_sm80 = FlashAttentionBackwardSm80(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            m_block_size,
            n_block_size,
            num_stages_Q,
            num_stages_dO,
            num_threads,
            pack_gqa,
            causal,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs=V_in_regs,
        )
        if compute_capability == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_stages_PdS,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                num_threads,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                subtile_factor=subtile_factor,
            )
        else:
            fa_bwd_obj = FlashAttentionBackwardSm100(
                head_dim,
                head_dim_v,
                is_causal=causal,
                is_local=local,
                qhead_per_kvhead=qhead_per_kvhead,
                # tile_m=m_block_size,
                # tile_n=n_block_size,
                cluster_size=cluster_size,
                # cluster_size=1,
                deterministic=deterministic,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                subtile_factor=subtile_factor,
            )

        # Block sparse tensors for backward use Q-direction indexing (transposed from forward).
        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            None,  # softcap - not yet supported in backward
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            cute_aux_tensors,
            sparse_tensors_compile,
            options="--enable-tvm-ffi",
        )
    _flash_attn_bwd.compile_cache[compile_key](
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk if not dKV_postprocess else dk_accum,
        dv if not dKV_postprocess else dv_accum,
        softmax_scale,
        current_stream,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        None,  # softcap - not yet supported in backward
        window_size_left,
        window_size_right,
        dQ_semaphore,
        dK_semaphore,
        dV_semaphore,
        aux_tensors,
        normalized_block_sparse_tensors[:4]
        if normalized_block_sparse_tensors is not None
        else None,
    )

    num_threads = 256 if compute_capability == 9 else 128
    arch = compute_capability * 10
    # Postprocess kernel: convert dq_accum from float32 to dq in bf16/fp16
    compile_key_post = (
        compute_capability,
        dtype,
        head_dim,
        m_block_size,
        num_threads,
        AtomLayoutMdQ,
        dQ_swapAB,
        cu_seqlens_q is None,
        seqused_q is None,
    )
    if compile_key_post not in _flash_attn_bwd.compile_cache_post:
        dq_accum_tensor = to_cute_tensor(dq_accum)
        dq_tensor = to_cute_tensor(dq)
        cu_seqlens_q_tensor, seqused_q_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, seqused_q)
        ]
        fa_bwd_post = FlashAttentionBackwardPostprocess(
            dtype, head_dim, arch, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
            fa_bwd_post,
            dq_accum_tensor,
            dq_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            seqused_q_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_bwd.compile_cache_post[compile_key_post](
        dq_accum,
        dq,
        softmax_scale,
        cu_seqlens_q,
        seqused_q,
        current_stream,
    )

    if dKV_postprocess:
        # Postprocess kernel: convert dk_accum & dv_accum from float32 to bf16/fp16
        compile_key_post = (
            compute_capability,
            dtype,
            head_dim,
            n_block_size,
            num_threads,
            AtomLayoutNdKV,
            dKV_swapAB,
            cu_seqlens_k is None,
            seqused_k is None,
        )
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            dk_accum_tensor = to_cute_tensor(dk_accum)
            dk_tensor = to_cute_tensor(dk)
            cu_seqlens_k_tensor, seqused_k_tensor = [
                to_cute_tensor(t, assumed_align=4) if t is not None else None
                for t in (cu_seqlens_k, seqused_k)
            ]
            arch = compute_capability * 10
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim, arch, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                dk_accum_tensor,
                dk_tensor,
                softmax_scale,
                cu_seqlens_k_tensor,
                seqused_k_tensor,
                current_stream,
                options="--enable-tvm-ffi",
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            dk_accum,
            dk,
            softmax_scale,
            cu_seqlens_k,
            seqused_k,
            current_stream,
        )
        compile_key_post = (
            compute_capability,
            dtype,
            head_dim_v,
            n_block_size,
            num_threads,
            AtomLayoutNdKV,
            dKV_swapAB,
            cu_seqlens_k is None,
            seqused_k is None,
        )
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            dv_accum_tensor = to_cute_tensor(dv_accum)
            dv_tensor = to_cute_tensor(dv)
            cu_seqlens_k_tensor, seqused_k_tensor = [
                to_cute_tensor(t, assumed_align=4) if t is not None else None
                for t in (cu_seqlens_k, seqused_k)
            ]
            arch = compute_capability * 10
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim_v, arch, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                dv_accum_tensor,
                dv_tensor,
                cutlass.Float32(1.0),
                cu_seqlens_k_tensor,
                seqused_k_tensor,
                current_stream,
                options="--enable-tvm-ffi",
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            dv_accum,
            dv,
            1.0,
            cu_seqlens_k,
            seqused_k,
            current_stream,
        )

    return dq, dk, dv


_flash_attn_bwd.compile_cache_pre = {}
_flash_attn_bwd.compile_cache = {}
_flash_attn_bwd.compile_cache_post = {}


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        mask_mod: Optional[Callable] = None,
        full_block_cnt: Optional[torch.Tensor] = None,
        full_block_idx: Optional[torch.Tensor] = None,
        mask_block_cnt: Optional[torch.Tensor] = None,
        mask_block_idx: Optional[torch.Tensor] = None,
        block_size: Optional[Tuple[int, int]] = None,
        mSFQ: Optional[torch.Tensor] = None,
        mSFK: Optional[torch.Tensor] = None,
    ):
        # Only create block sparse tensors if at least one block sparse parameter is provided
        block_sparse_tensors = None
        if any(
            t is not None for t in [full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx]
        ):
            block_sparse_tensors = BlockSparseTensorsTorch(
                full_block_cnt=full_block_cnt,
                full_block_idx=full_block_idx,
                mask_block_cnt=mask_block_cnt,
                mask_block_idx=mask_block_idx,
                block_size=block_size,
            )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            mSFQ=mSFQ,
            mSFK=mSFK,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, *((None,) * 22)  # Extra Nones is fine


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        output_amax: Optional[torch.Tensor] = None,
        output_amax_chunk_seqlen: int = 0,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            output_amax=output_amax,
            output_amax_chunk_seqlen=output_amax_chunk_seqlen,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        assert ctx.softcap == 0.0
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            deterministic=ctx.deterministic,
        )

        return dq, dk, dv, *((None,) * 19)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[torch.Tensor] = None,
    full_block_idx: Optional[torch.Tensor] = None,
    mask_block_cnt: Optional[torch.Tensor] = None,
    mask_block_idx: Optional[torch.Tensor] = None,
    block_size: Optional[Tuple[int, int]] = None,
    mSFQ: Optional[torch.Tensor] = None,
    mSFK: Optional[torch.Tensor] = None,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        mask_mod,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt,
        mask_block_idx,
        block_size,
        mSFQ,
        mSFK,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    output_amax: Optional[torch.Tensor] = None,
    output_amax_chunk_seqlen: int = 0,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        aux_tensors,
        output_amax,
        output_amax_chunk_seqlen,
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"
    assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    assert out_partial.stride(-1) == 1, "out_partial must be contiguous in the last dimension"
    assert lse_partial.stride(-2) == 1, "lse_partial must be contiguous in the seqlen dimension"
    assert lse_partial.shape == out_partial.shape[:-1]

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    # Validate output tensor shapes and types
    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == torch.float32, "lse must be fp32"

    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"

    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    m_block_size = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if m_block_size == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]

    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
    )

    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        out_partial_tensor = to_cute_tensor(out_partial, leading_dim=4 if not is_varlen else 3)
        lse_partial_tensor = to_cute_tensor(
            lse_partial, assumed_align=4, leading_dim=lse_partial.ndim - 2
        )
        out_tensor = to_cute_tensor(out, leading_dim=3 if not is_varlen else 2)
        lse_tensor = (
            to_cute_tensor(lse, assumed_align=4, leading_dim=lse.ndim - 2)
            if lse is not None
            else None
        )

        optional_tensors = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (cu_seqlens, seqused, num_splits_dynamic_ptr, semaphore_to_reset)
        ]
        cu_seqlens_tensor, seqused_tensor, num_splits_dynamic_tensor, semaphore_tensor = (
            optional_tensors
        )
        fa_combine = FlashAttentionForwardCombine(
            dtype=dtype,
            dtype_partial=dtype_partial,
            head_dim=head_dim,
            m_block_size=m_block_size,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
        )

        # Check if implementation is supported
        if not fa_combine.can_implement(
            dtype,
            dtype_partial,
            head_dim,
            m_block_size,
            k_block_size,
            log_max_splits,
            num_threads=256,
        ):
            raise RuntimeError(
                "FlashAttention combine kernel cannot be implemented with given parameters"
            )

        _flash_attn_fwd_combine.compile_cache[compile_key] = cute.compile(
            fa_combine,
            out_partial_tensor,
            lse_partial_tensor,
            out_tensor,
            lse_tensor,
            cu_seqlens_tensor,
            seqused_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_fwd_combine.compile_cache[compile_key](
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        num_splits_dynamic_ptr,
        semaphore_to_reset,
        current_stream,
    )


_flash_attn_fwd_combine.compile_cache = {}


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, total_q, num_heads), (
            "lse_partial shape mismatch for varlen"
        )
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, batch_size, seqlen, num_heads), (
            "lse_partial shape mismatch"
        )

    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype

    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )

    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device).transpose(
                0, 1
            )
        else:
            lse = torch.empty(
                batch_size, num_heads, seqlen, dtype=torch.float32, device=device
            ).transpose(1, 2)
    else:
        lse = None

    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
    )
    return out, lse
