"""
Block-sparse runtime utilities for CUTE DSL kernels.

This module contains runtime execution functions for block-sparse attention kernels.
These utilities are used by CUTE DSL kernels to produce and consume block-sparse loads.
"""

from typing import Callable, Optional
from functools import partial
import math
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

# Import data structures from block_sparsity
from open_vc_attn._kernels.blackwell.flash_attn.cute.block_sparsity import BlockSparseTensors


@cute.jit
def sparse_tensor_m_block(
    m_block,
    qhead_per_kvhead: cutlass.Constexpr[int],
    q_subtile_factor: cutlass.Constexpr[int],
):
    """Map packed m_block indices to block-sparse tensor indices."""
    block = m_block
    if const_expr(qhead_per_kvhead != 1):
        block = block // qhead_per_kvhead
    if const_expr(q_subtile_factor != 1):
        block = block // q_subtile_factor
    return block


@cute.jit
def load_block_list_sm100(
    block_indices: cute.Tensor,
    block_count,
    load_q_with_first: cutlass.Constexpr,
    m_block,
    q_stage: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
):
    """SM100 version of load_block_list (no intra_wg_overlap, no extra_tx_count)."""
    if block_count > 0:
        # First iteration: load Q alongside K if requested
        n_block_first = block_indices[block_count - 1]

        if const_expr(load_q_with_first):
            # SM100 loads Q0 and optionally Q1
            load_Q(block=q_stage * m_block + 0, stage=0)
            if const_expr(q_stage == 2):
                load_Q(block=q_stage * m_block + 1, stage=1)

        # SM100 doesn't use producer_acquire for pipeline_kv in load path
        # The pipeline barriers are handled inside load_KV
        load_K(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()
        load_V(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()

        # Remaining blocks
        for offset in cutlass.range(1, block_count):
            n_block = block_indices[block_count - 1 - offset]
            load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()
            load_V(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()

    return kv_producer_state


# SM100-specific tile processor using SM100 helpers
@cute.jit
def produce_block_sparse_loads_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
    q_stage: cutlass.Constexpr,
    q_producer_phase: Int32,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    """SM100 entry point for sparse block iteration.

    SM100 uses PipelineTmaUmma which doesn't support extra_tx_count, so we use
    simplified block processing that just calls producer_acquire without extras.

    Args:
        m_block: which tile of m we are processing
        qhead_per_kvhead: Constexpr pack factor
    """
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    q_phase_flipped = False

    if mask_empty:
        # No masked blocks: process full list with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_full_block_idx,
            curr_full_block_cnt,
            load_q_with_first=True,
            m_block=m_block,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = not full_empty
    else:
        # Process masked blocks with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            load_q_with_first=True,
            m_block=m_block,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = True

        if not full_empty:
            # Process full blocks without Q loading
            kv_producer_state = load_block_list_sm100(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_q_with_first=False,
                m_block=m_block,
                q_stage=q_stage,
                kv_producer_state=kv_producer_state,
                load_Q=load_Q,
                load_K=load_K,
                load_V=load_V,
                pipeline_kv=pipeline_kv,
            )

    if q_phase_flipped:
        q_producer_phase ^= 1

    return kv_producer_state, q_producer_phase


@cute.jit
def get_total_block_count(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors
    if const_expr(full_block_cnt is not None):
        return (
            mask_block_cnt[batch_idx, head_idx, m_block_sparse]
            + full_block_cnt[batch_idx, head_idx, m_block_sparse]
        )
    else:
        return mask_block_cnt[batch_idx, head_idx, m_block_sparse]


@cute.jit
def handle_block_sparse_empty_tile_correction_sm100(
    tidx: Int32,
    q_stage: cutlass.Constexpr,
    m_block_size: cutlass.Constexpr,
    qhead_per_kvhead,
    pack_gqa: cutlass.Constexpr,
    is_split_kv: cutlass.Constexpr,
    learnable_sink,
    mLSE,
    seqlen,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    split_idx: Int32,
    sScale: cute.Tensor,
    stats: list,
    correction_epilogue: Callable,
    thr_mma_pv: cute.ThrMma,
    tOtOs: tuple[cute.Tensor],
    sO: cute.Tensor,
    mbar_ptr,
    mbar_softmax_corr_full_offset: Int32,
    mbar_softmax_corr_empty_offset: Int32,
    mbar_P_full_O_rescaled_offset: Int32,
    mbar_P_full_2_offset: Int32,
    mbar_corr_epi_full_offset: Int32,
    mbar_corr_epi_empty_offset: Int32,
    softmax_corr_consumer_phase: Int32,
    o_corr_consumer_phase: Int32,
    corr_epi_producer_phase: Int32,
    softmax_scale_log2: Float32,
    max_offset: Float32,
    max_offset_scale: Float32,
    mO_cur: Optional[cute.Tensor] = None,
    gO: Optional[cute.Tensor] = None,
    gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    mOutputAmax: Optional[cute.Tensor] = None,
    output_amax_chunk_seqlen: Int32 = 0,
):
    """Handle the block-sparse case where a tile is fully masked:
    * zero staged results
    * seed stats
    * satisfy the usual barrier protocol so downstream warps continue to make progress.
    """
    LOG2_E = Float32(math.log2(math.e))

    for stage in cutlass.range_constexpr(q_stage):
        row_sum_value = Float32(1.0)
        row_max_value = (
            -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None
        )
        if const_expr(learnable_sink is not None):
            sink_val = -Float32.inf
            if const_expr(not pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            elif tidx < m_block_size:
                q_head_idx = (
                    (q_stage * m_block + stage) * m_block_size + tidx
                ) % qhead_per_kvhead + head_idx * qhead_per_kvhead
                sink_val = Float32(learnable_sink[q_head_idx])
            if sink_val != -Float32.inf and (const_expr(not is_split_kv) or split_idx == 0):
                if row_max_value == -Float32.inf:
                    row_max_value = sink_val * (LOG2_E / softmax_scale_log2)
                    row_sum_value = max_offset_scale
                else:
                    row_sum_value = row_sum_value + cute.math.exp2(
                        sink_val * LOG2_E - row_max_value * softmax_scale_log2 + max_offset,
                        fastmath=True,
                    )
        if tidx < m_block_size:
            scale_row_idx = tidx + stage * m_block_size
            sScale[scale_row_idx] = row_sum_value
            if const_expr(mLSE is not None or learnable_sink is not None):
                sScale[scale_row_idx + q_stage * m_block_size] = row_max_value
        acc_flag = row_sum_value == Float32(0.0) or row_sum_value != row_sum_value
        stats[stage] = (row_sum_value, row_max_value, acc_flag)

        cute.arch.mbarrier_wait(
            mbar_ptr + mbar_softmax_corr_full_offset + stage,
            softmax_corr_consumer_phase,
        )
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_softmax_corr_empty_offset + stage)

        if const_expr(gmem_tiled_copy_O is None):
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_corr_epi_empty_offset + stage,
                corr_epi_producer_phase,
            )
        correction_epilogue(
            thr_mma_pv,
            tOtOs[stage],
            tidx,
            stage,
            m_block,
            seqlen.seqlen_q,
            Float32(0.0),  # zero scale ensures empty tile writes zeros into staged outputs
            sO[None, None, stage],
            mO_cur,
            gO,
            gmem_tiled_copy_O,
            mOutputAmax,
            output_amax_chunk_seqlen,
        )
        if const_expr(gmem_tiled_copy_O is None):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_corr_epi_full_offset + stage)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_P_full_O_rescaled_offset + stage)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_P_full_2_offset + stage)

    softmax_corr_consumer_phase ^= 1
    o_corr_consumer_phase ^= 1
    corr_epi_producer_phase ^= 1

    return (
        softmax_corr_consumer_phase,
        o_corr_consumer_phase,
        corr_epi_producer_phase,
    )


@cute.jit
def softmax_block_sparse_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    mbar_ptr,
    mbar_softmax_corr_full_offset: Int32,
    mbar_softmax_corr_empty_offset: Int32,
    mbar_P_full_O_rescaled_offset: Int32,
    mbar_P_full_2_offset: Int32,
    q_stage: cutlass.Constexpr,
    stage_idx: Int32,
    check_m_boundary: bool,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    total_block_cnt = curr_mask_block_cnt + curr_full_block_cnt

    if total_block_cnt == 0:
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_softmax_corr_full_offset + stage_idx)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_P_full_O_rescaled_offset + stage_idx)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_P_full_2_offset + stage_idx)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_softmax_corr_empty_offset + stage_idx)
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            (
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
                mask_n_block,
                is_first=True,
                mask_fn=partial(mask_fn, mask_seqlen=True, check_q_boundary=check_m_boundary),
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mask_n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=False, check_q_boundary=check_m_boundary),
                )

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=True,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=True, check_q_boundary=check_m_boundary
                    ),
                )
            else:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=False,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        total_block_cnt == 0,
    )
