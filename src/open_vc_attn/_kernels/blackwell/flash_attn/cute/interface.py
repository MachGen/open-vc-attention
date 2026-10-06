# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Forward-only Blackwell source subset for Open-VC Attn.
# Public input and feature contracts are documented in docs/api.md.
# Forward-only Blackwell (SM100/SM103) interface.

import os
import math
from functools import lru_cache
from typing import Optional, Tuple, Callable, Union

import torch


import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from open_vc_attn._kernels.blackwell.nvfp4 import e4m3_scale_view
from open_vc_attn._kernels.blackwell.flash_attn.cute.fp8_tuning import CUTLASS_DSL_VERSION


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from open_vc_attn._kernels.blackwell.flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from open_vc_attn._kernels.blackwell.flash_attn.cute import utils
from open_vc_attn._kernels.blackwell.flash_attn.cute.cute_dsl_utils import (
    to_cute_tensor,
    to_cute_aux_tensor,
    get_aux_tensor_metadata,
)
from open_vc_attn._kernels.blackwell.flash_attn.cute.flash_fwd_sm100 import (
    FlashAttentionForwardSm100,
    DescaleTensors,
    SvdCorrectionTensors,
)
from open_vc_attn._kernels.blackwell.flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine

from open_vc_attn._kernels.blackwell.flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
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
    repair_k_descale: Optional[torch.Tensor] = None,
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

    Tuned NVFP4 QK / E4M3 PV ExpCast uses packed FP16 code rounding and early
    scale-stage release; other configurations use the generic encoder.
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
            "Quantized attention is forward-only; detach inputs before calling."
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

    if compute_capability != 10:
        raise ValueError("This release supports Blackwell SM100/SM103 forward inference only")
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
            and (cu_seqlens_q is None or batch_size == 1)
            and (cu_seqlens_k is None or batch_size == 1)
        ):
            raise ValueError(
                "V-Smooth requires SM100/SM103 dense D128 E4M3 attention without masks, GQA or external descales"
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
    if repair_k_descale is not None:
        # K/V rows [0, R) are repair tokens: duplicated keys with FP8 V residuals.
        # repair_k_descale holds each repair key's original K descale.
        if v_smooth or aux_tensors is not None or not expcast:
            raise ValueError("Repair tokens require dense ExpCast without V-Smooth or aux tensors")
        repair_rows = repair_k_descale.shape[-1] if repair_k_descale.dim() == 3 else 0
        if (
            repair_k_descale.dim() != 3
            or repair_k_descale.shape[:2] != (batch_size, num_head_kv)
            or repair_rows == 0
            or repair_rows % 128 != 0
            or repair_rows >= seqlen_k
            or repair_k_descale.dtype != torch.float32
            or repair_k_descale.device != q.device
            or not repair_k_descale.is_contiguous()
        ):
            raise ValueError(
                "repair_k_descale must be contiguous float32 [batch, kv_heads, R] on the "
                f"input device with 0 < R < {seqlen_k} and R a multiple of 128"
            )
        aux_tensors = [repair_k_descale]
    if expcast:
        if compute_capability != 10 or not is_quantized or v.dtype != torch.float8_e4m3fn:
            raise ValueError("ExpCast requires SM100-family quantized Q/K and E4M3 V")
        if svd_raw_q is not None:
            raise ValueError("ExpCast does not support SVD score/output correction")
    if mid_window_blocks is not None:
        assert compute_capability == 10, (
            "mid_window_blocks is only wired into the SM100 kernel."
        )
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

    if compute_capability == 10:
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
        and CUTLASS_DSL_VERSION == "4.6.2"
        and q.dtype == k.dtype == torch.float8_e4m3fn
        and cu_seqlens_q is not None
        and lse is None
    )
    # Long single-sequence B200 FP8 calls can amortize packing even at low H.
    # Keep both length floors and the work threshold for other devices/modes.
    sm100_fp8_dense = (
        fp8_varlen_expcast_packing
        and compute_capability == 10
        and _get_device_capability_minor() == 0
        and batch_size == 1
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
                        and CUTLASS_DSL_VERSION == "4.6.2"
                    )
                ),
                CUTLASS_DSL_VERSION in ("4.4.1", "4.6.2"),
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
                mid_window_blocks == 4
                or (fp8_varlen_expcast_packing and mid_window_blocks is None)
                or (
                    mid_window_blocks is not None
                    and mid_window_blocks >= 0
                    and sm100_fp8_dense
                ),
            )
        )
    )
    if v_prepacked and not packed_v:
        raise ValueError("v_prepacked requires the eligible dense packed-V path")
    if packed_v and not v_prepacked:
        from open_vc_attn._kernels.blackwell.flash_attn.cute.v_layout import pack_v

        v = pack_v(v)

    inline_rescale = (
        packed_v
        and not is_nvf4_qk
        and cu_seqlens_q is not None
        and batch_size == 1
        and CUTLASS_DSL_VERSION == "4.6.2"
    )
    if repair_k_descale is not None and not inline_rescale:
        raise ValueError("Repair tokens require the packed single-sequence ExpCast path")
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
        mid_window_blocks,  # baked in at compile time; None disables mid-out scan
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
    if repair_k_descale is not None:
        compile_key += ("repair_prefix",)
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
                to_cute_tensor(buf)
                if v_smooth or repair_k_descale is not None
                else to_cute_aux_tensor(buf)
                for buf in aux_tensors
            ]

        if compute_capability == 10:
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
                repair_prefix=repair_k_descale is not None,
                expcast=expcast,
                v_smooth=v_smooth,
                v_smooth_prefetch=v_smooth_prefetch,
                v_smooth_head_major=v_smooth_head_major,
                inline_rescale=inline_rescale,
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
                f"Unsupported compute capability: {compute_capability}. Supported: Blackwell SM100/SM103"
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
        if compute_capability == 10:
            compile_args.append(descale_tensors_tensor)
        compile_args.extend([sparse_tensors, cute_aux_tensors])
        if compute_capability == 10:
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
    if compute_capability == 10:
        call_args.append(descale_tensors)
    call_args.extend(
        [
            normalized_block_sparse_tensors[:4]
            if normalized_block_sparse_tensors is not None
            else None,
            aux_tensors,
        ]
    )
    if compute_capability == 10:
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
