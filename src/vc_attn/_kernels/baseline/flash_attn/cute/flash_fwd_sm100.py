# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
# Unsupported features that will be added later:
# - page size != 128
# - more hdim (192, 256)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import enum
import math
from typing import Tuple, Callable, Optional, Literal, NamedTuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.blackwell_helpers as sm100_utils_basic

from vc_attn._kernels.baseline.flash_attn.cute.paged_kv import PagedKVManager
import vc_attn._kernels.baseline.flash_attn.cute.utils as utils
from vc_attn._kernels.baseline.flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from vc_attn._kernels.baseline.flash_attn.cute import copy_utils
from vc_attn._kernels.baseline.flash_attn.cute.mask import AttentionMask
from vc_attn._kernels.baseline.flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner
from vc_attn._kernels.baseline.flash_attn.cute.seqlen_info import SeqlenInfoQK
from vc_attn._kernels.baseline.flash_attn.cute.block_info import BlockInfo
from vc_attn._kernels.baseline.flash_attn.cute.block_sparsity import BlockSparseTensors
from vc_attn._kernels.baseline.flash_attn.cute.block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
    handle_block_sparse_empty_tile_correction_sm100,
)
from vc_attn._kernels.baseline.flash_attn.cute.pack_gqa import PackGQA
from vc_attn._kernels.baseline.flash_attn.cute import blackwell_helpers as sm100_utils
from cutlass.cute import FastDivmodDivisor
from vc_attn._kernels.baseline.flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()


#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()
#     WarpSchedulerWG3 = enum.auto()
#     PFull = enum.auto()
#     PEmpty = enum.auto()


class DescaleTensors(NamedTuple):
    q_descale: Optional[cute.Tensor] = None
    k_descale: Optional[cute.Tensor] = None
    v_descale: Optional[cute.Tensor] = None

    def __new_from_mlir_values__(self, values):
        values = iter(values)
        return DescaleTensors(
            next(values) if self.q_descale is not None else None,
            next(values) if self.k_descale is not None else None,
            next(values) if self.v_descale is not None else None,
        )


class SvdCorrectionTensors(NamedTuple):
    raw_q: Optional[cute.Tensor] = None
    raw_k: Optional[cute.Tensor] = None
    raw_v: Optional[cute.Tensor] = None
    q_mean: Optional[cute.Tensor] = None
    q_low: Optional[cute.Tensor] = None
    k_coord: Optional[cute.Tensor] = None
    v_mean: Optional[cute.Tensor] = None
    v_basis: Optional[cute.Tensor] = None
    v_coord: Optional[cute.Tensor] = None

    def __new_from_mlir_values__(self, values):
        return SvdCorrectionTensors(
            *((*values, None, None, None, None, None, None, None, None, None)[:9])
        )


class FlashAttentionForwardSm100:
    arch = 100

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_sm103: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        skip_softmax_error: float = 0.0,
        skip_softmax_error_per_head: bool = False,
        skip_pv_gemm: bool = False,
        skip_counter_sample_q_stride: int = 1,
        skip_counter_sample_q_offset: int = 0,
        skip_softmax_disabled_head_mask: int = 0,
        head_index_count: int = 0,
        use_block_sparsity: bool = False,
        mid_window_blocks: Optional[int] = None,
        svd_topk: int = 0,
        svd_k_block: int = 64,
        svd_raw_head_dim: int = 0,
        svd_v_rank: int = 0,
        svd_num_k_blocks: int = 0,
        svd_delta_to_output: bool = False,
    ):
        self._use_block_sparsity = use_block_sparsity
        # Blackwell Ultra (sm_103): the exp2 polynomial-emulation path (`e2e`) is
        # disabled - its instruction-mix win was tuned on sm_100, so it falls back
        # to plain MUFU.EX2 here.
        self.is_sm103 = is_sm103
        self.use_tma_KV = not paged_kv_non_tma
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        # Skip-Softmax error budget ε; 0 disables.  Per-head mode: ε scalar acts
        # as a sentinel (>0 keeps the constexpr skip branches alive); the real
        # per-head ε is loaded from runtime tensor mSkipEps inside the kernel.
        self.skip_softmax_error = skip_softmax_error
        self.skip_softmax_error_per_head = skip_softmax_error_per_head
        self.skip_softmax_disabled_head_mask = skip_softmax_disabled_head_mask
        self.skip_counter_sample_q_stride = max(1, int(skip_counter_sample_q_stride))
        self.skip_counter_sample_q_offset = (
            int(skip_counter_sample_q_offset) % self.skip_counter_sample_q_stride
        )
        self.head_index_count = head_index_count
        # When True, MMA WG reads the per-warp sSkip flags set by the softmax WG
        # and drops P·V WGMMA for tiles where every softmax warp voted skip.
        # Requires skip_softmax_error > 0 (otherwise sSkip is never written).
        self.skip_pv_gemm = skip_pv_gemm
        self.svd_topk = max(0, int(svd_topk))
        self.svd_k_block = max(1, int(svd_k_block))
        self.svd_raw_head_dim = max(0, int(svd_raw_head_dim))
        self.svd_v_rank = max(0, int(svd_v_rank))
        self.svd_num_k_blocks = max(0, int(svd_num_k_blocks))
        self.svd_delta_to_output = bool(svd_delta_to_output)
        if self.svd_topk > 0:
            assert not pack_gqa, "SVD-CuTe correction MVP requires non-pack-GQA"
            assert not is_varlen_q, "SVD-CuTe correction MVP requires fixed-length Q"
            assert not paged_kv_non_tma, "SVD-CuTe correction MVP does not support paged KV"
            assert self.svd_raw_head_dim > 0, "SVD-CuTe correction requires raw Q/K head dim"
            if self.svd_delta_to_output:
                assert self.svd_v_rank > 0, "SVD direct raw-V delta requires V rank"
                assert self.svd_num_k_blocks > 0, "SVD direct raw-V delta requires K block count"
            assert self.svd_k_block <= n_block_size, (
                "SVD-CuTe correction block must fit in the FA4 N tile"
            )
            assert n_block_size % self.svd_k_block == 0, (
                "SVD-CuTe correction block must divide the FA4 N tile"
            )
        # Mid-out scan: when set, the dense path visits the K-block that is
        # `mid_window_blocks` to the right of the (m_block, n_block) diagonal
        # first, sweeps left to n_block_min, then sweeps the remaining tail.
        # Intent: prime row_max with the dominant near-diagonal scores so that
        # `skip_softmax_error` can drop more far-from-diagonal blocks downstream.
        # Dense path only - incompatible with causal/local/block-sparse.
        if mid_window_blocks is not None:
            assert not is_causal and not is_local, (
                "mid_window_blocks requires non-causal, non-local attention"
            )
            assert not use_block_sparsity, "mid_window_blocks is incompatible with block sparsity"
            assert mid_window_blocks >= 0, "mid_window_blocks must be non-negative"
        self.mid_window_blocks = mid_window_blocks
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        assert self.q_stage in [1, 2]

        # 2 Q tile per CTA
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q or self.svd_delta_to_output
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        self.q_subtile_factor = q_subtile_factor
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        # S0/S1 exp2 ping-pong. Off for hd128: profiled neutral-to-negative
        # (-2.3% cap6 fp8), doesn't raise eligible warps. Was only ever a win for
        # hd 64/96: self.head_dim_padded in [64,96] and not is_causal/is_local.
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or (
            self.head_dim_v_padded >= 128 and self.is_split_kv
        )
        if self.overlap_sO_sQ:
            self.is_persistent = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        if self.q_stage == 1:
            if not self.use_tma_KV:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            else:
                self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()

        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q:  # fallback
            self.epilogue_warp_ids = (13, 14)

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
            self.num_regs_correction = 64
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        elif use_block_sparsity:
            # Sparse path uses a different reg distribution from dense.
            # Empirical sweep on int8 sparse, head_dim=128, 3 passes/cfg:
            #     softmax/correction/other     480p Δ%   720p Δ%
            #     192/80/48 (dense default)    +85.5%    +84.4%   (correction spills 40 B)
            #     192/72/48                    +17.4%    +15.4%   (escape spill, ptxas dip)
            #     176/80/48                    + 2.7%    + 2.5%
            #     184/96/48                    + 2.8%    + 2.6%
            #     160/96/48                    + 0.3%    + 0.6%   <- chosen
            # In sparse, correction warp does extra work per iter (gmem-loaded
            # block count, different tile_block_count derivation) and wants 96
            # regs; softmax can be cut to 160 with no visible cost on this
            # code path, freeing budget to give correction more headroom.
            # Don't trust ptxas spill bytes alone - non-spilling configs
            # (172/76/48, 192/72/48, 176/80/48) all compile clean but perf
            # differs ≥6×. Must sweep AND time across the (softmax, correction)
            # plane jointly, not pick the first 0-byte-spill config.
            self.num_regs_softmax = 160 if not paged_kv_non_tma else 184
            self.num_regs_correction = 96
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        else:
            # 192/80 (softmax/correction) is the swept optimum for both fp8 and
            # int8 hd128. fp8: other splits within 0.3% (latency-bound). int8:
            # correction-heavy splits regress +3-4% (issue-bound softmax needs
            # the regs).
            self.num_regs_softmax = 192 if not paged_kv_non_tma else 184
            self.num_regs_correction = 80
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        self.num_regs_empty = 24

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        # INT8 K/V are both 1 byte, so a deeper K/V pipeline fits in SMEM and
        # hides more HBM/MMA latency.  FP8 stays at 4 - its 2-byte V makes SMEM
        # tight.
        is_int8 = self.q_dtype == cutlass.Int8
        self.kv_stage = (
            6
            if is_int8 and self.head_dim_padded <= 128 and self.head_dim_v_padded <= 128
            else 4
            if (
                self.q_dtype.width == 8
                or (self.is_nvf4_qk and self.v_dtype.width == 8)
                or self.q_stage == 1
            )
            and self.head_dim_padded <= 128
            and self.head_dim_v_padded <= 128
            else 3
        )
        # kv_stage depth swept neutral for both fp8 (4/5/6) and int8 (4/5/6/8):
        # the bottleneck is TMEM round-trip latency, not K/V load latency. fp8=4
        # (tighter SMEM from 2-byte V), int8=6.
        self.acc_stage = 1
        # For hdim 192,128, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.m_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mSkipCounter: Optional[
            cute.Tensor
        ] = None,  # diag: (num_heads, 3) int32 per-head [eval, softmax_skipped, pv_skipped]
        mSkipEps: Optional[
            cute.Tensor
        ] = None,  # (num_heads,) float32 per-head ε; indexed by head_idx (kv-head when pack_gqa)
        mHeadMap: Optional[
            cute.Tensor
        ] = None,  # (active_heads,) int32 logical group head -> real head
        svd_tensors: Optional[SvdCorrectionTensors] = None,
        mOutputAmax: Optional[cute.Tensor] = None,
        output_amax_chunk_seqlen: Int32 | int = 0,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        q_ptr_shape: Optional[tuple] = None,
        k_ptr_shape: Optional[tuple] = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        if const_expr(mSFQ is not None):
            assert q_ptr_shape is not None and k_ptr_shape is not None
            q_iter = mQ.iterator if hasattr(mQ, "iterator") else mQ
            k_iter = mK.iterator if hasattr(mK, "iterator") else mK
            mQ = cute.make_tensor(
                q_iter,
                cute.make_ordered_layout(
                    q_ptr_shape,
                    order=tuple(range(len(q_ptr_shape) - 1, -1, -1)),
                ),
            )
            mK = cute.make_tensor(
                k_iter,
                cute.make_ordered_layout(
                    k_ptr_shape,
                    order=tuple(range(len(k_ptr_shape) - 1, -1, -1)),
                ),
            )

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        self.is_nvf4_qk = const_expr(mSFQ is not None)
        self.sf_dtype = cutlass.Float8E4M3FN
        self.sf_vec_size = 16
        if const_expr(self.is_nvf4_qk):
            assert const_expr(mSFK is not None), "mSFK must be provided with mSFQ"
            assert const_expr(mCuSeqlensQ is None), "QK NVFP4 does not support varlen Q yet"
            assert const_expr(mCuSeqlensK is None), "QK NVFP4 does not support varlen K yet"
            assert const_expr(mSeqUsedQ is None), "QK NVFP4 does not support seqused_q yet"
            assert const_expr(mSeqUsedK is None), "QK NVFP4 does not support seqused_k yet"
            assert const_expr(mPageTable is None), "QK NVFP4 does not support paged KV yet"
            assert const_expr(not self.pack_gqa), "QK NVFP4 does not support PackGQA yet"
            assert const_expr(self.use_tma_KV), "QK NVFP4 currently requires TMA KV"
            self.sf_dtype = mSFQ.element_type
            assert const_expr(self.sf_dtype is cutlass.Float8E4M3FN), "QK NVFP4 requires E4M3 scale"
            self.sf_vec_size = 16
        self.is_int8 = const_expr(self.q_dtype == cutlass.Int8)
        self.is_fp8 = const_expr(self.q_dtype.width == 8 and self.q_dtype != cutlass.Int8)
        self.nvf4_early_scale_release = const_expr(
            self.is_nvf4_qk and self.skip_softmax_error > 0.0
        )
        self.tmem_vec_offset = self.tmem_s_offset
        if const_expr(self.is_nvf4_qk and self.q_stage == 2):
            self.tmem_vec_offset = [
                self.tmem_s_offset[stage] + self.tmem_s_to_p_offset - 1
                for stage in range(self.q_stage)
            ]
        # INT8/FP8/NVFP4 QK can land P in FP8 when V is FP8, so the max_offset
        # bias trick follows the PV operand dtype.
        self.p_is_fp8 = const_expr(self.v_dtype.width == 8)
        # cuDNN/upstream-FA4 parity: rescale deadband of 4, so the O-rescale
        # only fires when the running max moves by >4 in log2 domain (cuDNN
        # SASS: FSETP.GT diff,4 + FSEL on row_max, rescale rate 1.0% vs 4.8%
        # without).  max_offset drops 8 -> 4 to keep the P peak at
        # 2^(offset+deadband) = 256 inside e4m3 range.  Enabled for fp8-Q and
        # int8 per-block descale only when skip-softmax is off.  In skip-softmax
        # mode the deadband lowers skip hit-rate enough to cost more than the
        # saved O-rescales.  int8 per-head stays off as well: it is issue-bound
        # and its rescale rate is already low.
        # See notes/RESCALE_DEADBAND.md.
        int8_block_descale = const_expr(
            self.is_int8
            and descale_tensors is not None
            and descale_tensors.k_descale is not None
            and len(descale_tensors.k_descale.shape) == 3
        )
        skip_softmax_active = const_expr(self.skip_softmax_error > 0.0)
        fp8_without_skip_softmax = const_expr(self.is_fp8 and not skip_softmax_active)
        nvf4_qk_fp8_pv_without_skip_softmax = const_expr(
            self.is_nvf4_qk and self.p_is_fp8 and not skip_softmax_active
        )
        int8_block_without_skip_softmax = const_expr(int8_block_descale and not skip_softmax_active)
        use_rescale_deadband = const_expr(
            fp8_without_skip_softmax
            or nvf4_qk_fp8_pv_without_skip_softmax
            or int8_block_without_skip_softmax
        )
        # bf16-Q keeps the upstream threshold of 8 (P stays in fp32/bf16, no
        # e4m3 budget); 8-bit Q uses the deadband of 4 when enabled above.
        self.rescale_threshold = (
            8.0
            if const_expr(self.q_dtype.width == 16)
            else (4.0 if const_expr(use_rescale_deadband) else 0.0)
        )
        self.p_fp8_max_offset = 4 if const_expr(use_rescale_deadband) else 8

        if const_expr(
            (self.is_fp8 or (self.is_nvf4_qk and self.p_is_fp8))
            and not self.skip_softmax_error > 0.0
        ):
            self.num_regs_softmax = 192
            self.num_regs_correction = 88
            self.num_regs_other = 40
            fp8_sparse_descale = const_expr(
                self.is_fp8
                and self._use_block_sparsity
                and descale_tensors is not None
                and descale_tensors.k_descale is not None
            )
            if const_expr(fp8_sparse_descale):
                if const_expr(len(descale_tensors.k_descale.shape) == 3):
                    # Sparse per-block scaling puts the softmax warps on the
                    # producer-consumer critical path.
                    self.num_regs_softmax = 208
                    self.num_regs_correction = 56
                    self.num_regs_other = 40
                elif const_expr(len(descale_tensors.k_descale.shape) == 2):
                    # Per-head scaling shifts the critical path toward the
                    # correction warps instead.
                    self.num_regs_softmax = 184
                    self.num_regs_correction = 96
                    self.num_regs_other = 48

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        if const_expr(self.is_nvf4_qk):
            mSFQ = cute.make_tensor(
                mSFQ.iterator, cute.select(mSFQ.layout, mode=Q_layout_transpose)
            )
            mSFK = cute.make_tensor(
                mSFK.iterator, cute.select(mSFK.layout, mode=KV_layout_transpose)
            )
        self.num_head = mQ.shape[2]
        self.num_head_kv = mK.shape[2]
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        if const_expr(self.is_nvf4_qk):
            self.q_major_mode = tcgen05.OperandMajorMode.K
            self.k_major_mode = tcgen05.OperandMajorMode.K
        else:
            self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
            self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        # INT8 Q/K with FP8 V is allowed (SageAttention-style INT8 QK MMA + FP8 PV MMA)
        if const_expr(
            self.q_dtype != self.v_dtype and self.q_dtype != cutlass.Int8 and not self.is_nvf4_qk
        ):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        # p_dtype: the dtype used for attention weights P in the PV MMA.
        # For INT8 Q/K + FP8 V, P must be FP8 (v_dtype) since PV MMA uses v_dtype.
        self.p_dtype = self.v_dtype
        self._setup_attributes()
        self.use_tma_O = (
            self.arch >= 90
            and mCuSeqlensQ is None
            and mSeqUsedQ is None
            and not self.use_correction_warps_for_epi
        )
        # exp2 emulation knobs (apply_exp2_convert, `e2e=True` path).
        # Block-pattern gate: of every `e2e_freq` k-iterations, the last `e2e_res`
        # use polynomial emulation; the last `e2e_frg_limit` frgs are forced to MUFU.
        self.e2e_freq = 16
        if const_expr(self.is_fp8):
            # res=4 is the ptxas optimum (res=6 hits a scheduling dead-zone);
            # ~6-7.5% faster on cap2/cap6, cosine unchanged. Non-monotonic.
            self.e2e_res = 4
            self.e2e_frg_limit = 2
        elif const_expr(self.is_int8):
            self.e2e_res = 2
            self.e2e_frg_limit = 2
        elif const_expr(self.is_nvf4_qk and self.is_sm103):
            # One polynomial pair per eligible fragment. Extra pairs or a different
            # fragment limit increase the softmax critical path on SM103.
            self.e2e_freq = 30
            self.e2e_res = 2
            self.e2e_frg_limit = 1
        else:
            self.e2e_res = 4
            self.e2e_frg_limit = 1
        if const_expr(
            self.head_dim_padded > 64 and not self.is_causal and not self.is_local and self.pack_gqa
        ):
            self.e2e_freq = 32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else 10
        # exp2-dose sweep re-confirmed (2026-06-05 fp8 / 2026-06-06 int8, cap6):
        # fp8 freq=16/res=4/frg=2 optimum; int8 res=2 optimum and res>=4 SPILLS
        # catastrophically (+30%) from int8's extra I2F/dequant regs. Both tuned.
        # NVFP4 SM103 (2026-07-20, cap6): freq=30/res=2/frg=1 is the optimum.

        cta_group = tcgen05.CtaGroup.ONE
        self.cta_group = cta_group
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        if const_expr(self.is_nvf4_qk):
            tiled_mma_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.qk_acc_dtype,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.mma_tiler_pv[:2]

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.p_dtype,
            self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        sfq_smem_layout = None
        sfk_smem_layout = None
        if const_expr(self.is_nvf4_qk):
            mma_inst_k_qk = cute.size(tiled_mma_qk.shape_mnk, mode=[2])
            mma_inst_tile_k_qk = self.mma_tiler_qk[2] // mma_inst_k_qk
            sfq_smem_layout = sm100_utils.make_smem_layout_sfa(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                self.q_stage,
                mma_tile_inst_k=mma_inst_tile_k_qk,
            )
            sfk_smem_layout = sm100_utils.make_smem_layout_sfb(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                self.kv_stage,
                mma_tile_inst_k=mma_inst_tile_k_qk,
            )
            sfq_smem_layout_stage = cute.slice_(sfq_smem_layout, (None, None, None, 0))
            sfk_smem_layout_stage = cute.slice_(sfk_smem_layout, (None, None, None, 0))
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.q_stage,
        )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (self.qhead_per_kvhead, mQ.shape[0]),
                mQ.shape[1],
                mK.shape[2],
                *mQ.shape[3:],
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mO.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed)
                )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        if const_expr(self.is_nvf4_qk):
            self.tma_copy_bytes["Q"] += cute.size_in_bytes(mSFQ.element_type, sfq_smem_layout_stage)
            self.tma_copy_bytes["K"] += cute.size_in_bytes(mSFK.element_type, sfk_smem_layout_stage)

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        mQ_tma_shape = mQ.shape
        mK_tma_shape = mK.shape

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                self.cluster_layout_vmnk.shape,
            )
        else:
            tma_atom_K = None
            tma_atom_V = None

        tma_atom_sfq = None
        tma_tensor_sfq = None
        tma_atom_sfk = None
        tma_tensor_sfk = None
        tiled_mma_sfb_qk = None
        mma_tiler_sfb_qk = None
        if const_expr(self.is_nvf4_qk):
            sfq_layout = cute.tile_to_shape(
                blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout,
                mQ_tma_shape,
                (2, 1, 3, 4),
            )
            sfk_layout = cute.tile_to_shape(
                blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout,
                mK_tma_shape,
                (2, 1, 3, 4),
            )
            mSFQ = cute.make_tensor(mSFQ.iterator, sfq_layout)
            mSFK = cute.make_tensor(mSFK.iterator, sfk_layout)
            tma_atom_sfq, tma_tensor_sfq = cute.nvgpu.make_tiled_tma_atom_A(
                sm100_utils_basic.cluster_shape_to_tma_atom_A(
                    self.cluster_shape_mn, tiled_mma_qk.thr_id
                ),
                mSFQ,
                sfq_smem_layout_stage,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            mma_inst_k = cute.size(tiled_mma_qk.shape_mnk, mode=[2])
            mma_inst_tile_k = self.mma_tiler_qk[2] // mma_inst_k
            mma_inst_shape_mnk_sfb_qk = (
                self.mma_tiler_qk[0],
                cute.round_up(self.mma_tiler_qk[1], 128),
                mma_inst_k,
            )
            mma_tiler_sfb_qk = (
                mma_inst_shape_mnk_sfb_qk[0],
                mma_inst_shape_mnk_sfb_qk[1],
                mma_inst_shape_mnk_sfb_qk[2] * mma_inst_tile_k,
            )
            tiled_mma_sfb_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.k_dtype,
                self.k_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                tcgen05.CtaGroup.ONE,
                mma_inst_shape_mnk_sfb_qk[:2],
            )
            cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_sfb_qk.thr_id.shape,),
            )
            tma_atom_sfk, tma_tensor_sfk = cute.nvgpu.make_tiled_tma_atom_B(
                sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                    self.cluster_shape_mn, tiled_mma_sfb_qk.thr_id
                ),
                mSFK,
                sfk_smem_layout_stage,
                mma_tiler_sfb_qk,
                tiled_mma_sfb_qk,
                cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        o_cta_v_layout = cute.composition(cute.make_identity_layout(mO.shape), self.epi_tile)

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                TileScheduler = SingleTileLPTScheduler
            else:
                TileScheduler = (
                    SingleTileScheduler
                    if const_expr(not self.is_persistent)
                    else StaticPersistentTileScheduler
                )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            self.head_index_count
            if const_expr(self.head_index_count > 0)
            else cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=max(self.k_dtype.width // 8, 1),
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + self.q_stage
        self.mbar_O_full_offset = self.mbar_S_full_offset + self.q_stage
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + self.q_stage
        self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + self.q_stage
        self.mbar_corr_epi_full_offset = self.mbar_softmax_corr_empty_offset + self.q_stage
        self.mbar_corr_epi_empty_offset = self.mbar_corr_epi_full_offset + self.q_stage
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + self.q_stage
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        self.mbar_sfqk_load_offset = self.mbar_P_full_2_offset + self.q_stage
        self.mbar_total = (
            self.mbar_sfqk_load_offset + self.q_stage
            if const_expr(self.is_nvf4_qk)
            else self.mbar_P_full_2_offset + self.q_stage
        )

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout)
            if const_expr(not self.overlap_sO_sQ)
            else cutlass.max(
                cute.cosize(sQ_layout),
                cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
            )
        )
        sK_size = cute.cosize(sK_layout) if const_expr(not self.is_nvf4_qk) else 1
        sV_size = cute.cosize(sV_layout) if const_expr(self.is_nvf4_qk) else 1
        sfq_size = cute.cosize(sfq_smem_layout) if const_expr(self.is_nvf4_qk) else 1
        sfk_size = cute.cosize(sfk_smem_layout) if const_expr(self.is_nvf4_qk) else 1

        # Size-one placeholders still consume their alignment in CuTe structs.
        # Keep NVFP4-only buffers out of BF16 storage, which already reaches the
        # SM103 shared-memory limit for some head configurations.
        @cute.struct
        class SharedStorageNvfp4:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            # Per-warp skip flags: q_stage stages × num_softmax_warps_per_stage warps.
            # Each warp writes its own slot; MMA WG ANDs all slots to decide whether
            # to skip gemm_Pi.  Using per-warp slots avoids the race condition that
            # arises when different warps (covering different M-row ranges) have
            # different skip decisions and all write to a single sSkip[stage].
            sSkip: cute.struct.MemRange[Int32, self.q_stage * len(self.softmax0_warp_ids)]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, sK_size],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, sV_size],
                self.buffer_align_bytes,
            ]
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfq_size],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfk_size],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            tmem_holding_buf: Int32
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            sSkip: cute.struct.MemRange[Int32, self.q_stage * len(self.softmax0_warp_ids)]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorageNvfp4 if const_expr(self.is_nvf4_qk) else SharedStorage

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softmax_scale = None
        else:
            # NB: If a users passes in a score mod, we want to apply the score-mod in the sm_scaled qk
            # But in the original base 10. We hijack softmax_scale_log2 to just be the change of base
            # and correctly apply the softmax_scale prior to score_mod in the softmax step
            softmax_scale_log2 = LOG2_E
            softmax_scale = softmax_scale

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            )
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        head_divmod = None
        if cutlass.const_expr(self.pack_gqa):
            head_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError("Block sparsity + paged KV not supported on SM100")

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            descale_tensors,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            head_divmod,
            mSkipCounter,
            mSkipEps,
            mHeadMap,
            svd_tensors,
            mOutputAmax,
            Int32(output_amax_chunk_seqlen),
            tma_atom_sfq,
            tma_tensor_sfq,
            tma_atom_sfk,
            tma_tensor_sfk,
            sfq_smem_layout,
            sfk_smem_layout,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mSkipCounter: Optional[cute.Tensor] = None,
        mSkipEps: Optional[cute.Tensor] = None,
        mHeadMap: Optional[cute.Tensor] = None,
        svd_tensors: Optional[SvdCorrectionTensors] = None,
        mOutputAmax: Optional[cute.Tensor] = None,
        output_amax_chunk_seqlen: Int32 = 0,
        tma_atom_sfq: Optional[cute.CopyAtom] = None,
        tma_tensor_sfq: Optional[cute.Tensor] = None,
        tma_atom_sfk: Optional[cute.CopyAtom] = None,
        tma_tensor_sfk: Optional[cute.Tensor] = None,
        sfq_smem_layout: Optional[cute.Layout] = None,
        sfk_smem_layout: Optional[cute.Layout] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(tma_atom_K is not None):
                cpasync.prefetch_descriptor(tma_atom_K)
            if const_expr(tma_atom_V is not None):
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)
            if const_expr(tma_atom_sfq is not None):
                cpasync.prefetch_descriptor(tma_atom_sfq)
            if const_expr(tma_atom_sfk is not None):
                cpasync.prefetch_descriptor(tma_atom_sfk)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        # Use the first N warps to initialize barriers
        if warp_idx == 1:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_full_offset + i, 1)
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == 2:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + i, cute.arch.WARP_SIZE * 4
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + i, cute.arch.WARP_SIZE * 4
                )
        if warp_idx == 3:
            if const_expr(self.s0_s1_barrier):
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE
                    )
        if const_expr(not self.use_correction_warps_for_epi) and warp_idx == 4:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_full_offset + i,
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_empty_offset + i,
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids),
                )
        if warp_idx == 5:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_O_rescaled_offset + i,
                    cute.arch.WARP_SIZE
                    * (len(self.softmax0_warp_ids) + len(self.correction_warp_ids)),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id])
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == 6:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_2_offset + i,
                    cute.arch.WARP_SIZE * len(self.softmax0_warp_ids),
                )
        if warp_idx == 7:
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        # Non-NVFP4 storage ends at mbar_P_full_2; initializing the sfqk
        # barriers there would write past the mbar array into
        # tmem_holding_buf/sScale and leak the tmem allocation at dealloc.
        if const_expr(self.is_nvf4_qk):
            if warp_idx == 8:
                for i in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_sfqk_load_offset + i,
                        len(self.softmax0_warp_ids)
                        * (1 if self.nvf4_early_scale_release else cute.arch.WARP_SIZE),
                    )
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        pipeline_kv = self.make_and_init_load_kv_pipeline(mbar_ptr + self.mbar_load_kv_full_offset)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        if const_expr(self.is_nvf4_qk):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stride_sK_aligned = const_expr(stride_sV * self.v_dtype.width // self.k_dtype.width)
            sK_outer_aligned = cute.make_layout(
                sK_layout.outer.shape,
                stride=(*sK_layout.outer.stride[:-1], stride_sK_aligned),
            )
            sK = storage.sV.get_tensor(
                sK_outer_aligned, swizzle=sK_layout.inner, dtype=self.k_dtype
            )
        else:
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            # (MMA, MMA_K, MMA_D, PIPE)
            # Strip swizzle info to reuse smem
            sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer
            )

        sScale = storage.sScale.get_tensor(cute.make_layout(self.q_stage * self.m_block_size * 2))
        sSkip = storage.sSkip.get_tensor(
            cute.make_layout(self.q_stage * len(self.softmax0_warp_ids))
        )
        sSFQ = None
        sSFK = None
        if const_expr(self.is_nvf4_qk):
            sSFQ = storage.sSFQ.get_tensor(sfq_smem_layout)
            sSFK = storage.sSFK.get_tensor(sfk_smem_layout)

        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM
        thr_mma_sfb_qk = None
        mma_tiler_sfb_qk_kernel = None
        if const_expr(self.is_nvf4_qk):
            mma_inst_k_sfb_qk = cute.size(tiled_mma_qk.shape_mnk, mode=[2])
            mma_inst_tile_k_sfb_qk = self.mma_tiler_qk[2] // mma_inst_k_sfb_qk
            mma_inst_shape_mnk_sfb_qk_kernel = (
                self.mma_tiler_qk[0],
                cute.round_up(self.mma_tiler_qk[1], 128),
                mma_inst_k_sfb_qk,
            )
            mma_tiler_sfb_qk_kernel = (
                mma_inst_shape_mnk_sfb_qk_kernel[0],
                mma_inst_shape_mnk_sfb_qk_kernel[1],
                mma_inst_shape_mnk_sfb_qk_kernel[2] * mma_inst_tile_k_sfb_qk,
            )
            tiled_mma_sfb_qk_kernel = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.k_dtype,
                self.k_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                tcgen05.CtaGroup.ONE,
                mma_inst_shape_mnk_sfb_qk_kernel[:2],
            )
            thr_mma_sfb_qk = tiled_mma_sfb_qk_kernel.get_slice(0)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout)
            for stage in range(self.q_stage)
        )
        tOtOs = tuple(
            cute.make_tensor(tOtO.iterator + self.tmem_o_offset[stage], tOtO.layout)
            for stage in range(self.q_stage)
        )

        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

        tOrPs = [
            cute.make_tensor(
                tOrP.iterator
                + self.qk_acc_dtype.width // self.p_dtype.width * self.tmem_p_offset[stage],
                tOrP.layout,
            )
            for stage in range(self.q_stage)
        ]
        tCtSFQs = [None] * self.q_stage
        tCtSFKs = [None] * self.q_stage
        if const_expr(self.is_nvf4_qk):
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                cute.slice_(sfq_smem_layout, (None, None, None, 0)),
            )
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                cute.slice_(sfk_smem_layout, (None, None, None, 0)),
            )
            sfq_tmem_ptrs_f32 = [
                cute.make_ptr(
                    Float32,
                    self.tmem_total
                    if const_expr(self.q_stage == 1)
                    else self.tmem_s_offset[self.q_stage - 1 - stage],
                    mem_space=cute.AddressSpace.tmem,
                    assumed_align=16,
                )
                for stage in range(self.q_stage)
            ]
            tCtSFQs = [
                cute.make_tensor(
                    cute.recast_ptr(sfq_tmem_ptrs_f32[stage], dtype=self.sf_dtype),
                    tCtSFQ_layout,
                )
                for stage in range(self.q_stage)
            ]
            mma_inst_tile_k = self.mma_tiler_qk[2] // cute.size(tiled_mma_qk.shape_mnk, mode=[2])
            sfq_tmem_cols = (
                self.mma_tiler_qk[0] // cute.size(tiled_mma_qk.thr_id.shape) // 32
            ) * mma_inst_tile_k
            sfk_tmem_cols = (cute.round_up(self.mma_tiler_qk[1], 128) // 32) * mma_inst_tile_k
            if const_expr(self.q_stage == 1):
                assert self.tmem_total + sfq_tmem_cols + sfk_tmem_cols <= self.tmem_alloc_cols
            else:
                assert sfq_tmem_cols + sfk_tmem_cols <= self.tmem_s_to_p_offset - 1
            tCtSFKs = [
                cute.make_tensor(
                    cute.recast_ptr(sfq_tmem_ptrs_f32[stage] + sfq_tmem_cols, dtype=self.sf_dtype),
                    tCtSFK_layout,
                )
                for stage in range(self.q_stage)
            ]

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                thr_mma_sfb_qk,
                mma_tiler_sfb_qk_kernel,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_sfq,
                tma_tensor_sfq,
                tma_atom_sfk,
                tma_tensor_sfk,
                sSFQ,
                sSFK,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                mHeadMap,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
                cute.arch.sync_warp()

            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStSs,
                tOtOs,
                tOrPs,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                sSFQ,
                sSFK,
                tCtSFQs,
                tCtSFKs,
                sSkip=sSkip,
                mSkipCounter=mSkipCounter,
                mHeadMap=mHeadMap,
            )

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    mbar_ptr,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    mHeadMap,
                    mOutputAmax,
                    output_amax_chunk_seqlen,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if (const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]) or (
            const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1]
        ):
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                descale_tensors=descale_tensors,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                sSkip=sSkip,
                mLSE=mLSE,
                learnable_sink=learnable_sink,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                blocksparse_tensors=blocksparse_tensors,
                mSkipCounter=mSkipCounter,
                mSkipEps=mSkipEps,
                mHeadMap=mHeadMap,
                svd_tensors=svd_tensors,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(
                    0
                    if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0]
                    else 1
                )
                softmax_loop(
                    stage=stage,
                    tStSi=cute.make_tensor(
                        tStS.iterator
                        + (self.tmem_s_offset[0] if stage == 0 else self.tmem_s_offset[1]),
                        tStS.layout,
                    ),
                )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout)
                    softmax_loop(stage=0, tStSi=tStSi)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[1], tStS.layout)
                    softmax_loop(stage=1, tStSi=tStSi)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtOs,
                sScale,
                mO,
                mLSE,
                sO,
                learnable_sink,
                descale_tensors,
                gmem_tiled_copy_O,
                tma_atom_O,
                mbar_ptr,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                mHeadMap,
                svd_tensors,
                mOutputAmax,
                Int32(output_amax_chunk_seqlen),
            )
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        thr_mma_sfb_qk: Optional[cute.ThrMma],
        mma_tiler_sfb_qk: cutlass.Constexpr,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_sfq: Optional[cute.CopyAtom],
        tma_tensor_sfq: Optional[cute.Tensor],
        tma_atom_sfk: Optional[cute.CopyAtom],
        tma_tensor_sfk: Optional[cute.Tensor],
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        mHeadMap: Optional[cute.Tensor] = None,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            head_idx = self._real_head_idx(mHeadMap, head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                    mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            load_SFQ_fn = None
            if const_expr(self.is_nvf4_qk):
                tma_tensor_sfq_cur = tma_tensor_sfq[None, None, head_idx, batch_idx]
                gSFQ = cute.local_tile(
                    tma_tensor_sfq_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)
                )
                tSgSFQ = thr_mma_qk.partition_A(gSFQ)
                load_SFQ_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_sfq,
                    0,
                    cute.make_layout(1),
                    tSgSFQ,
                    sSFQ,
                    filter_zeros=True,
                )

            tKsSFK, tKgSFK = None, None
            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                if const_expr(self.is_nvf4_qk):
                    if const_expr(mPageTable is None):
                        if const_expr(not seqlen.has_cu_seqlens_k):
                            tma_tensor_sfk_cur = tma_tensor_sfk[None, None, head_idx_kv, batch_idx]
                        else:
                            tma_tensor_sfk_cur = cute.domain_offset(
                                (seqlen.offset_k, 0), tma_tensor_sfk[None, None, head_idx_kv]
                            )
                        gSFK = cute.local_tile(
                            tma_tensor_sfk_cur,
                            cute.select(mma_tiler_sfb_qk, mode=[1, 2]),
                            (None, 0),
                        )
                    else:
                        tma_tensor_sfk_cur = tma_tensor_sfk[None, None, head_idx_kv, None]
                        gSFK = cute.local_tile(
                            tma_tensor_sfk_cur,
                            cute.select(mma_tiler_sfb_qk, mode=[1, 2]),
                            (None, 0, None),
                        )
                    tSgSFK = thr_mma_sfb_qk.partition_B(gSFK)
                    tKsSFK, tKgSFK = cpasync.tma_partition(
                        tma_atom_sfk,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sSFK, 0, 3),
                        cute.group_modes(tSgSFK, 0, 3),
                    )
                    tKsSFK = cute.filter_zeros(tKsSFK)
                    tKgSFK = cute.filter_zeros(tKgSFK)
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmodDivisor(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            load_Q = partial(
                self.load_Q,
                load_Q_fn,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
                load_SFQ_fn=load_SFQ_fn,
            )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
                tma_atom_sf=tma_atom_sfk,
                tXgSF=tKgSFK,
                tXsSF=tKsSFK,
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
            )

            if const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                        load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                    if const_expr(self.mid_window_blocks is None):
                        first_block = n_block_max - 1 if n_block_max > 0 else 0
                    else:
                        # Diagonal expressed in n_block units; use the lower
                        # M-block of the cluster (the upper stage is at most
                        # q_stage-1 above, well within mid_window_blocks).
                        diag_n_block = (
                            self.q_stage * m_block * self.m_block_size
                        ) // self.n_block_size
                        first_block = cutlass.min(
                            diag_n_block + Int32(self.mid_window_blocks),
                            n_block_max - 1,
                        )
                    page_idx = (
                        mPageTable[batch_idx, first_block]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(first_block)
                    load_K(
                        block=first_block, producer_state=kv_producer_state, page_idx=page_idx
                    )  # K0
                    kv_producer_state.advance()
                    if const_expr(self.q_stage == 2) and (
                        const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE
                    ):
                        load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
                    q_producer_phase ^= 1
                    load_V(
                        block=first_block, producer_state=kv_producer_state, page_idx=page_idx
                    )  # V0
                    kv_producer_state.advance()

                    if const_expr(self.mid_window_blocks is None):
                        for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                            n_block = n_block_max - 2 - i
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None and self.use_tma_KV)
                                else None
                            )
                            if const_expr(not self.use_tma_KV):
                                paged_kv_manager.load_page_table(n_block)
                            # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("n_block = {}, page_idx = {}", n_block, page_idx)
                            load_K(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )  # Ki
                            kv_producer_state.advance()
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )  # Vi
                            kv_producer_state.advance()
                    else:
                        # Mid-out scan. Phase 1 sweeps left from first_block - 1
                        # down to n_block_min; Phase 2 sweeps left from
                        # n_block_max - 1 down to first_block + 1. When
                        # first_block == n_block_max - 1 (window covers the
                        # right edge), Phase 2 is empty and the order matches
                        # the original right-to-left scan.
                        phase1_count = first_block - n_block_min
                        phase2_count = n_block_max - 1 - first_block
                        for i in cutlass.range(phase1_count, unroll=1):
                            n_block = first_block - 1 - i
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None and self.use_tma_KV)
                                else None
                            )
                            if const_expr(not self.use_tma_KV):
                                paged_kv_manager.load_page_table(n_block)
                            load_K(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                        for i in cutlass.range(phase2_count, unroll=1):
                            n_block = n_block_max - 1 - i
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None and self.use_tma_KV)
                                else None
                            )
                            if const_expr(not self.use_tma_KV):
                                paged_kv_manager.load_page_table(n_block)
                            load_K(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        tCtSFQs: tuple,
        tCtSFKs: tuple,
        sSkip: Optional[cute.Tensor] = None,
        mSkipCounter: Optional[cute.Tensor] = None,
        mHeadMap: Optional[cute.Tensor] = None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

        if const_expr(self.is_nvf4_qk):
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_partial_fp4,
                    qk_mma_op,
                    self.tmem_s_offset[stage],
                    tSrQs[stage],
                    sA=sQ[None, None, None, stage],
                    tScaleA=tCtSFQs[stage],
                    tScaleB=tCtSFKs[stage],
                    zero_init=True,
                )
                for stage in range(self.q_stage)
            ]
        else:
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    qk_mma_op,
                    self.tmem_s_offset[stage],
                    tSrQs[stage],
                    sA=sQ[None, None, None, stage],
                    zero_init=True,
                )
                for stage in range(self.q_stage)
            ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrPs[stage],
                sA=None,
                mbar_wait_fraction_num=(
                    1
                    if const_expr(
                        (self.is_fp8 or (self.is_nvf4_qk and self.p_is_fp8)) and self.is_sm103
                    )
                    else 3
                ),
                mbar_wait_fraction_den=(
                    2
                    if const_expr(
                        (self.is_fp8 or (self.is_nvf4_qk and self.p_is_fp8)) and self.is_sm103
                    )
                    else 4
                ),
                # NVFP4: producer P-store split and consumer PV-consume boundary must
                # correspond in the SAME (v-dtype-keyed) units; pass the absolute tile
                # count computed by _mbar_p_split so both sides use the same boundary.
                pre_mbar_tiles=(
                    self._mbar_p_split(cute.size(tOrPs[stage].shape[2]))
                    if const_expr(self.is_nvf4_qk)
                    else None
                ),
            )
            for stage in range(self.q_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)
        if const_expr(self.is_nvf4_qk):
            tiled_copy_s2t_sfq_staged = [
                self.mainloop_s2t_copy_and_partition(sSFQ, tCtSFQs[stage])
                for stage in range(self.q_stage)
            ]
            tiled_copy_s2t_sfk_staged = [
                self.mainloop_s2t_copy_and_partition(sSFK, tCtSFKs[stage])
                for stage in range(self.q_stage)
            ]
            tiled_copy_s2t_sfq, tCsSFQ_compact_s2t, _ = tiled_copy_s2t_sfq_staged[0]
            tiled_copy_s2t_sfk, tCsSFK_compact_s2t, _ = tiled_copy_s2t_sfk_staged[0]
        else:
            tiled_copy_s2t_sfq_staged = []
            tiled_copy_s2t_sfk_staged = []
            tiled_copy_s2t_sfq = None
            tCsSFQ_compact_s2t = None
            tiled_copy_s2t_sfk = None
            tCsSFK_compact_s2t = None

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        mma_sfqk_producer_phase = Int32(0)
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            head_idx = self._real_head_idx(mHeadMap, head_idx)
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            if process_tile:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase
                    )
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                    if const_expr(self.is_nvf4_qk):
                        self.mainloop_copy_qk_scale_to_tmem(
                            tiled_copy_s2t_sfq_staged,
                            tiled_copy_s2t_sfk_staged,
                            stage,
                            mma_kv_consumer_state.index,
                            mbar_ptr,
                            mma_sfqk_producer_phase,
                            wait_sfqk_empty=True,
                        )
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(
                            sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        )
                    gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    # 4. release S0 / S1
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                mma_q_consumer_phase ^= 1
                if const_expr(self.is_nvf4_qk):
                    mma_sfqk_producer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile has finished.
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                            P_full_O_rescaled_phase,
                        )
                        # 3. gemm (or skip if softmax WG zeroed P for this block)
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        if const_expr(self.skip_pv_gemm and self.skip_softmax_error > 0.0):
                            # AND all per-warp skip slots: skip iff every warp agreed.
                            _num_sw = const_expr(len(self.softmax0_warp_ids))
                            _skip_all = sSkip[stage * _num_sw + 0]
                            for _w in cutlass.range_constexpr(1, _num_sw):
                                _skip_all = _skip_all & sSkip[stage * _num_sw + _w]
                            should_skip_mma = _skip_all != Int32(0)
                        else:
                            should_skip_mma = False
                        if should_skip_mma:
                            # All M-row ranges agreed P≈0; skip WGMMA, drain mbar_P_full_2.
                            if const_expr(mSkipCounter is not None):
                                with cute.arch.elect_one():
                                    utils.atomic_add_i32(
                                        1, utils.elem_pointer(mSkipCounter, (head_idx, 2))
                                    )
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_P_full_2_offset + stage,
                                P_full_O_rescaled_phase,
                            )
                        else:
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_should_accumulate,
                                mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                                mbar_phase=P_full_O_rescaled_phase,
                            )
                        # 4. release accumulated O0_partial / O1_partial
                        # Don't need to signal O_full to the correction warps anymore since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # with cute.arch.elect_one():
                        #     tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                        # 5. release V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrK[None, None, None, Ki_index], zero_init=True)
                        if const_expr(self.is_nvf4_qk):
                            self.mainloop_copy_qk_scale_to_tmem(
                                tiled_copy_s2t_sfq_staged,
                                tiled_copy_s2t_sfk_staged,
                                stage,
                                mma_kv_consumer_state.index,
                                mbar_ptr,
                                mma_sfqk_producer_phase,
                                wait_sfqk_empty=True,
                            )
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                        # 3. release S0
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    if const_expr(self.is_nvf4_qk):
                        mma_sfqk_producer_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.q_stage):
                        tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # 2. acquire corrected Oi_partial and Pi
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                        P_full_O_rescaled_phase,
                    )
                    # 3. gemm (or skip if softmax WG zeroed P for this block)
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    if const_expr(self.skip_pv_gemm and self.skip_softmax_error > 0.0):
                        _num_sw = const_expr(len(self.softmax0_warp_ids))
                        _skip_all = sSkip[stage * _num_sw + 0]
                        for _w in cutlass.range_constexpr(1, _num_sw):
                            _skip_all = _skip_all & sSkip[stage * _num_sw + _w]
                        should_skip_mma = _skip_all != Int32(0)
                    else:
                        should_skip_mma = False
                    if should_skip_mma:
                        # P was zeroed; skip WGMMA and just drain mbar_P_full_2.
                        if const_expr(mSkipCounter is not None):
                            with cute.arch.elect_one():
                                utils.atomic_add_i32(
                                    1, utils.elem_pointer(mSkipCounter, (head_idx, 2))
                                )
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_2_offset + stage,
                            P_full_O_rescaled_phase,
                        )
                    else:
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished compute
                    # the row sum of the current tile. It does not guarantee that the 1st tile
                    # of the next work tile has been computed yet.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t,
            thr_copy_s2t.partition_S(tCsSF_compact),
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def _mbar_p_split(self, k: int) -> int:
        """Boundary (in P K-tiles) for the softmax-store / PV-consume handoff.

        Keyed on the PV operand (V) dtype, matching the reference fp4 kernel. Both
        the producer (softmax P-store) and consumer (PV gemm) call this on their own
        partition's tile count so the split lands on the same physical boundary; the
        max(1, min(k-1, ...)) clamp guarantees a nonempty first chunk (a zero-length
        first chunk would signal P before storing anything -> PV reads garbage/NaN).
        """
        if self.v_dtype.width == 8 and k > 1:
            num, den = (3, 4) if self.head_dim_v_padded <= 64 else (1, 2)
            return max(1, min(k - 1, k * num // den))
        elif self.v_dtype.width > 8:
            return k // 4 * 3
        else:
            return k // 2

    @cute.jit
    def mainloop_copy_qk_scale_to_tmem(
        self,
        tiled_copy_s2t_sfq_staged: tuple,
        tiled_copy_s2t_sfk_staged: tuple,
        q_stage: cutlass.Constexpr,
        kv_stage: Int32,
        mbar_ptr: cute.Pointer,
        mma_sfqk_producer_phase: Int32,
        wait_sfqk_empty: cutlass.Constexpr,
    ):
        sm100_utils.tcgen05_after_thread_sync()
        if const_expr(wait_sfqk_empty):
            cute.arch.mbarrier_wait(
                mbar_ptr + self.mbar_sfqk_load_offset + q_stage,
                mma_sfqk_producer_phase,
            )
        tiled_copy_s2t_sfq, tCsSFQ_compact_s2t, tCtSFQ_compact_s2t = tiled_copy_s2t_sfq_staged[
            q_stage
        ]
        tCsSFQ_cur = tCsSFQ_compact_s2t[None, None, None, None, q_stage]
        cute.copy(tiled_copy_s2t_sfq, tCsSFQ_cur, tCtSFQ_compact_s2t)
        tiled_copy_s2t_sfk, tCsSFK_compact_s2t, tCtSFK_compact_s2t = tiled_copy_s2t_sfk_staged[
            q_stage
        ]
        tCsSFK_cur = tCsSFK_compact_s2t[None, None, None, None, kv_stage]
        cute.copy(tiled_copy_s2t_sfk, tCsSFK_cur, tCtSFK_compact_s2t)

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        """Map query-head tile index -> KV-head index (FA3 descale semantics)."""
        if cutlass.const_expr(self.pack_gqa):
            return head_idx
        return head_idx // self.qhead_per_kvhead

    @cute.jit
    def _real_head_idx(self, mHeadMap: Optional[cute.Tensor], head_idx: Int32) -> Int32:
        """Map a logical group-head tile index to the real tensor head."""
        if cutlass.const_expr(self.head_index_count > 0):
            return Int32(mHeadMap[head_idx])
        return head_idx

    @cute.jit
    def _load_effective_descales(
        self,
        descale_tensors: Optional[DescaleTensors],
        batch_idx: Int32,
        kv_head_idx: Int32,
        head_idx: Int32 = Int32(0),
        m_block_seq: Int32 = Int32(0),
    ) -> Tuple[Float32, Float32]:
        """Load effective QK and V descales, defaulting unspecified tensors to identity.

        Supports both 2D (per-tensor-per-head) and 3D (per-block) descale layouts:
          2D: descale[batch_idx, kv_head_idx]          -> scalar folded into scale.
          3D Q: descale[batch_idx, head_idx, m_block_seq] -> per m-block scalar (baked here).
          3D K: descale[batch_idx, kv_head_idx, n_block]  -> applied per-n_block
                inside softmax_step; skipped here.
        """
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if cutlass.const_expr(descale_tensors is not None):
            if cutlass.const_expr(descale_tensors.q_descale is not None):
                if cutlass.const_expr(len(descale_tensors.q_descale.shape) == 3):
                    qk_descale = qk_descale * Float32(
                        descale_tensors.q_descale[batch_idx, head_idx, m_block_seq]
                    )
                else:
                    qk_descale = qk_descale * Float32(
                        descale_tensors.q_descale[batch_idx, kv_head_idx]
                    )
            if cutlass.const_expr(descale_tensors.k_descale is not None):
                if cutlass.const_expr(len(descale_tensors.k_descale.shape) == 2):
                    qk_descale = qk_descale * Float32(
                        descale_tensors.k_descale[batch_idx, kv_head_idx]
                    )
                # 3D: applied per-n_block in softmax_step (no bake here).
            if cutlass.const_expr(descale_tensors.v_descale is not None):
                v_descale = Float32(descale_tensors.v_descale[batch_idx, kv_head_idx])
        return qk_descale, v_descale

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        descale_tensors: Optional[DescaleTensors],
        thr_mma_qk: cute.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        sSkip: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mSkipCounter: Optional[cute.Tensor] = None,
        mSkipEps: Optional[cute.Tensor] = None,
        mHeadMap: Optional[cute.Tensor] = None,
        svd_tensors: Optional[SvdCorrectionTensors] = None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )

        tmem_vec_delta = 0
        if const_expr(self.is_nvf4_qk and self.q_stage == 2):
            tmem_vec_delta = self.tmem_s_to_p_offset - 1
        tStScale_base = cute.make_tensor(tStSi.iterator + tmem_vec_delta, tStSi.layout)
        tStScale = cute.composition(tStScale_base, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        tStP_layout = cute.composition(
            tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)),
            Float32,
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(
            tidx
        )

        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(
                    8
                    if const_expr(
                        (self.q_dtype.width == 8 or (self.is_nvf4_qk and self.p_is_fp8))
                        and (not self.is_fp8 or not self.is_sm103)
                    )
                    else 16
                )
            ),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)
        if const_expr(self.is_nvf4_qk and self.q_stage == 2):
            if stage == 1:
                if const_expr(self.nvf4_early_scale_release):
                    if cute.arch.lane_idx() == 0:
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + 0)
                else:
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            head_idx = self._real_head_idx(mHeadMap, head_idx)
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=self.q_stage * m_block + stage,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=aux_tensors,
            )

            # Recompute fastdiv_mods if necessary
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )

            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            qk_descale, _ = self._load_effective_descales(
                descale_tensors,
                batch_idx,
                kv_head_idx,
                head_idx=head_idx,
                m_block_seq=self.q_stage * m_block + stage,
            )

            # Fold max_offset into the exp2 argument so P lands in
            # [0, 2^(max_offset + rescale_deadband)] - must stay below the FP8
            # E4M3 max 448, no extra per-element FMUL (rides the FMA bias
            # slot).  BF16 accumulator stays at 0.
            max_offset = self.p_fp8_max_offset if const_expr(self.p_is_fp8) else 0
            if const_expr(self.score_mod is None):
                softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
                softmax_scale_eff = None
            else:
                softmax_scale_log2_eff = softmax_scale_log2
                softmax_scale_eff = softmax_scale * qk_descale

            # Pre-translate ε into Δ-space so should_skip_block stays a cheap
            # sub+compare.  exp2(s·Δ) < ε/L  ⇔  Δ < log2(ε/L)/s.  Dividing by
            # softmax_scale_log2_eff (which has qk_descale folded in) lands the
            # threshold in the same frame row_max occupies across all descale
            # paths - no separate qk_descale correction needed.
            skip_threshold_val = Float32(0.0)
            skip_softmax_enabled = False
            if const_expr(self.skip_softmax_error > 0.0):
                if const_expr(self.skip_softmax_error_per_head):
                    eps_val = Float32(mSkipEps[head_idx])
                    skip_softmax_enabled = eps_val > Float32(0.0)
                else:
                    eps_val = Float32(self.skip_softmax_error)
                    skip_softmax_enabled = True
                if const_expr(self.skip_softmax_disabled_head_mask != 0):
                    for _h in cutlass.range_constexpr(64):
                        if const_expr(self.skip_softmax_disabled_head_mask & (1 << _h)):
                            skip_softmax_enabled = skip_softmax_enabled & (head_idx != Int32(_h))
                if skip_softmax_enabled:
                    skip_threshold_val = (
                        cute.math.log2(
                            eps_val / Float32(seqlen.seqlen_k),
                            fastmath=True,
                        )
                        / softmax_scale_log2_eff
                    )
                else:
                    skip_threshold_val = Float32(0.0)
            softmax = SoftmaxSm100.create(
                softmax_scale_log2_eff,
                rescale_threshold=self.rescale_threshold,
                softmax_scale=softmax_scale_eff,
                max_offset=max_offset,
                has_skip_softmax=const_expr(self.skip_softmax_error > 0.0),
                skip_threshold=skip_threshold_val,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                sSkip=sSkip,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                kv_head_idx=kv_head_idx,
                m_block=self.q_stage * m_block + stage,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                descale_tensors=descale_tensors,
                mSkipCounter=mSkipCounter,
                skip_softmax_enabled=skip_softmax_enabled,
                svd_tensors=svd_tensors,
            )

            if has_work:
                # Softmax acts as the producer: wait until correction signals the stage is empty
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
                )
                si_corr_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                # When aux_tensors exist, Q indices beyond seqlen_q must be wrapped to avoid
                # OOB aux_tensor access. Only edge tiles (where m_tile_end > seqlen_q) need this.
                if const_expr(aux_tensors is not None):
                    m_tile_end = (self.q_stage * m_block + stage + 1) * self.m_block_size
                    check_m_boundary = m_tile_end > seqlen.seqlen_q
                else:
                    check_m_boundary = False
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mbar_ptr,
                    self.mbar_softmax_corr_full_offset,
                    self.mbar_softmax_corr_empty_offset,
                    self.mbar_P_full_O_rescaled_offset,
                    self.mbar_P_full_2_offset,
                    self.q_stage,
                    Int32(stage),
                    check_m_boundary,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                if not empty_tile:
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    # if tidx == 0:
                    #     cute.printf("softmax row sum stage %d: %f, row_max = %f\n", stage, softmax.row_sum[0], softmax.row_max[0])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
                    # if tidx == 0: cute.printf("softmax row sum stage %d: %f\n", stage, softmax.row_sum[0])
            else:
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    if const_expr(self.mid_window_blocks is None):
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                            softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block_max - 1,
                                is_first=True,
                                mask_fn=partial(mask_fn, mask_seqlen=True),
                            )
                        )
                        n_block_max -= 1
                        # Next couple of iterations with causal masking
                        if const_expr(self.is_causal or self.is_local):
                            n_block_min_causal_local_mask = (
                                block_info.get_n_block_min_causal_local_mask(
                                    seqlen, m_block, n_block_min
                                )
                            )
                            for n_tile in cutlass.range(
                                n_block_max - n_block_min_causal_local_mask, unroll=1
                            ):
                                n_block = n_block_max - 1 - n_tile
                                (
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                ) = softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                        # The remaining iterations have no masking (but may still need mask_mod)
                        n_block_min_before_local_mask = (
                            block_info.get_n_block_min_before_local_mask(
                                seqlen, m_block, n_block_min
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_before_local_mask, unroll=1
                        ):
                            n_block = n_block_max - n_tile - 1
                            if const_expr(self.mask_mod is not None):
                                (
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                ) = softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
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
                                    n_block,
                                )
                        # Separate iterations with local masking on the left
                        if const_expr(self.is_local and block_info.window_size_left is not None):
                            n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                            for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                                n_block = n_block_max - 1 - n_tile
                                (
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                ) = softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                                # Now that we no longer already have the 1st iteration, need mask_seqlen=True here
                    else:
                        # Mid-out scan (must match the producer's iteration order):
                        #   first_block = min(diag + window, n_block_max - 1)
                        #   Phase 1: first_block, first_block - 1, ..., n_block_min
                        #   Phase 2: n_block_max - 1, ..., first_block + 1
                        # When first_block == n_block_max - 1 the order collapses
                        # to the original right-to-left scan.
                        diag_n_block = (
                            self.q_stage * m_block * self.m_block_size
                        ) // self.n_block_size
                        first_block = cutlass.min(
                            diag_n_block + Int32(self.mid_window_blocks),
                            n_block_max - 1,
                        )
                        # Interior (non-rightmost) blocks only need an explicit mask
                        # under causal/local geometry or a custom mask_mod; for plain
                        # attention the mask there is a no-op AND passing it disables
                        # the exp2 emulation path (e2e gate keys off `mask_fn is None`).
                        # Drop it for those blocks, matching the standard r-to-l scan.
                        interior_mask_fn = (
                            partial(mask_fn, mask_seqlen=False)
                            if const_expr(
                                self.is_causal or self.is_local or self.mask_mod is not None
                            )
                            else None
                        )
                        if first_block == n_block_max - 1:
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    first_block,
                                    is_first=True,
                                    mask_fn=partial(mask_fn, mask_seqlen=True),
                                )
                            )
                            for n_tile in cutlass.range(first_block - n_block_min, unroll=1):
                                n_block = first_block - 1 - n_tile
                                (
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                ) = softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=interior_mask_fn,
                                )
                        else:
                            # Phase 1: first iteration is the near-diagonal block,
                            # not at the right edge - no seqlen mask required.
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    first_block,
                                    is_first=True,
                                    mask_fn=interior_mask_fn,
                                )
                            )
                            for n_tile in cutlass.range(first_block - n_block_min, unroll=1):
                                n_block = first_block - 1 - n_tile
                                (
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                ) = softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=interior_mask_fn,
                                )
                            # Phase 2 first iteration is the rightmost block -
                            # this is the only one that may extend past seqlen_k.
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block_max - 1,
                                    mask_fn=partial(mask_fn, mask_seqlen=True),
                                )
                            )
                            for n_tile in cutlass.range(n_block_max - 2 - first_block, unroll=1):
                                n_block = n_block_max - 2 - n_tile
                                (
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                ) = softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=interior_mask_fn,
                                )

                    # Dense path always writes scale / signals
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + cute.math.log2(softmax.row_sum[0], fastmath=True)) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        sSkip: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        kv_head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        descale_tensors: Optional[DescaleTensors] = None,
        mSkipCounter: Optional[cute.Tensor] = None,
        skip_softmax_enabled=True,
        svd_tensors: Optional[SvdCorrectionTensors] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))

        # Wait for Si
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        # SM103: fuse the row-max into the S TMEM load via tcgen05.ld.red (the TMEM
        # controller computes the per-tile max at zero ALU cost).  Only on the plain
        # fp8 path -- int8 reduces in Int32, and score_mod/svd/mask rewrite the tile
        # after the load (the hw max would be stale).  Excluding mask_fn here keeps the
        # masked boundary blocks on a plain Ld32x32b (no dead .MAX -> no RZ in SASS).
        use_hw_rowmax = const_expr(
            self.is_sm103
            and not self.is_int8
            and self.score_mod is None
            and self.svd_topk == 0
            and mask_fn is None
        )
        hw_max = None
        if const_expr(use_hw_rowmax):
            if const_expr(self.nvf4_early_scale_release and self.q_stage == 2):
                sfqk_stage = self.q_stage - 1 - stage
                hw_max = sm100_utils.tmem_ld_red_max(
                    tStS_t2r,
                    tSrS_t2r,
                    mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage,
                )
            else:
                hw_max = sm100_utils.tmem_ld_red_max(tStS_t2r, tSrS_t2r)
        else:
            if const_expr(self.nvf4_early_scale_release and self.q_stage == 2):
                num_s_segments = cute.size(tStS_t2r.shape[1])
                cute.copy(
                    thr_tmem_load,
                    tStS_t2r[None, 0, None, None],
                    tSrS_t2r[None, 0, None, None],
                )
                cute.arch.fence_view_async_tmem_load()
                sfqk_stage = self.q_stage - 1 - stage
                if cute.arch.lane_idx() == 0:
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)
                for s_segment in cutlass.range_constexpr(1, num_s_segments):
                    cute.copy(
                        thr_tmem_load,
                        tStS_t2r[None, s_segment, None, None],
                        tSrS_t2r[None, s_segment, None, None],
                    )
            else:
                cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if const_expr(self.is_nvf4_qk and self.q_stage == 2):
            cute.arch.fence_view_async_tmem_load()
            if const_expr(not self.nvf4_early_scale_release):
                sfqk_stage = self.q_stage - 1 - stage
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)
        # INT8 MMA stores Int32 in TMEM; the copy above leaves raw Int32 bits
        # in tSrS_t2r's F32 storage.  update_row_max reduces in Int32 and
        # apply_exp2_convert fuses I2F with scale-subtract + exp2 per pair, so
        # the full F32 tile is never materialised.
        if const_expr(self.is_int8):
            tSrS_int_view = cute.recast_tensor(tSrS_t2r, Int32)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
            )
        if cutlass.const_expr(svd_tensors is not None and self.svd_topk > 0):
            self.apply_svd_exact_topk_score(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                svd_tensors,
                batch_idx,
                head_idx,
                m_block,
                n_block,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        # Per-block K descale (3D k_descale only): instead of multiplying the
        # entire S tile by k_descale_n (O(tile_size) FMULs), fold it into the
        # softmax scale - only the scalar row_max_tile and the FMA multiplier
        # in scale_subtract_rowmax need adjusting (O(1) scalar ops).
        if const_expr(self.is_int8):
            tSrS_rowmax_ssa = tSrS_int_view.load()
        else:
            tSrS_rowmax_ssa = tSrS_t2r.load()
        k_descale_n = None
        if const_expr(
            descale_tensors is not None
            and descale_tensors.k_descale is not None
            and len(descale_tensors.k_descale.shape) == 3
        ):
            k_descale_n = Float32(descale_tensors.k_descale[batch_idx, kv_head_idx, n_block])
        # SM103: hand the tcgen05.ld.red hardware max straight to update_row_max,
        # skipping the software fmax_reduce.  Only valid when the tile is unmodified
        # post-load -- use_hw_rowmax already excludes mask_fn/score_mod/svd, so hw_max
        # is None on those paths and the software reduction runs as before.
        row_max, acc_scale, row_max_tile = softmax.update_row_max(
            tSrS_rowmax_ssa,
            is_first,
            k_descale=k_descale_n,
            int_input=const_expr(self.is_int8),
            precomputed_tile_max=hw_max,
        )

        # should_skip_softmax = const_expr(not is_first) and softmax.should_skip_block(row_max_tile)

        # Short-circuit: when has_skip_softmax=False this is compile-time False and
        # vote_all_sync is never emitted.  When True, vote_all_sync ensures all threads
        # in the warp agree on the skip decision (no warp divergence).
        should_skip_softmax = False
        if const_expr(not is_first) and const_expr(softmax.has_skip_softmax):
            if skip_softmax_enabled:
                should_skip_softmax = cute.arch.vote_all_sync(
                    softmax.should_skip_block(row_max_tile)
                )

        # Diagnostic skip counter - opt-in, zero overhead when disabled
        # (const_expr folds the whole block out).
        if const_expr(mSkipCounter is not None and softmax.has_skip_softmax and not is_first):
            counter_in_sample = True
            if const_expr(self.skip_counter_sample_q_stride > 1):
                q_block_seq = self.q_stage * m_block + stage
                counter_in_sample = (
                    (q_block_seq + self.skip_counter_sample_q_offset)
                    % self.skip_counter_sample_q_stride
                ) == Int32(0)
            if skip_softmax_enabled:
                if counter_in_sample:
                    with cute.arch.elect_one():
                        utils.atomic_add_i32(1, utils.elem_pointer(mSkipCounter, (head_idx, 0)))
                        if should_skip_softmax:
                            utils.atomic_add_i32(1, utils.elem_pointer(mSkipCounter, (head_idx, 1)))

        if const_expr(not is_first):
            if const_expr(self.is_sm103):
                tSrScale_r2t = cute.make_rmem_tensor(
                    thr_tmem_store_scale.partition_S(tScScale).shape, Float32
                )
                tSrScale_r2t[0] = acc_scale
                cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
                cute.arch.fence_view_async_tmem_store()
            else:
                thread_idx = thr_tmem_load.thr_idx
                sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.p_dtype),
            tSrS_t2r.layout,
        )
        if should_skip_softmax:
            if const_expr(self.s0_s1_barrier):
                cute.arch.mbarrier_wait(
                    mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase
                )
            # Zero out P so that if other warps don't skip (and the MMA WG
            # runs the WGMMA), this warp's rows contribute nothing.
            tSrP_r2t_f32.fill(0.0)
        else:
            # INT8 fuses scale-subtract into apply_exp2_convert; non-INT8 keeps
            # the two-step path.
            has_3d_k_descale = const_expr(
                descale_tensors is not None
                and descale_tensors.k_descale is not None
                and len(descale_tensors.k_descale.shape) == 3
            )
            if const_expr(not self.is_int8):
                if const_expr(has_3d_k_descale):
                    softmax.scale_subtract_rowmax(tSrS_t2r, row_max, k_descale=k_descale_n)
                else:
                    softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
            if const_expr(self.s0_s1_barrier):
                cute.arch.mbarrier_wait(
                    mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase
                )
            softmax.apply_exp2_convert(
                tSrS_t2r,
                tSrP_r2t,
                # Mixed hardware/polynomial exp2. Non-SM103 FP8 uses a 25% dose;
                # SM103 NVFP4 uses a lighter dose because heavier emulation adds
                # more FMA work than the exposed softmax path can hide. NVFP4 skip
                # keeps hardware exp2 because emulation lengthens its control path.
                # Emulation pays off only when the kernel is issue-starved (low IPC)
                # with an SFU bottleneck. fp8 is (IPC 1.64, idle slots) -> moving exp2
                # off the SFU into idle FMA slots nets -6%. int8 is NOT: its I2F +
                # dequant FMAs already fill the issue slots (IPC 2.03), so the extra
                # polynomial instructions add cycles instead of filling idle ones
                # Both fp8 and int8 emulate, but at different doses (set above):
                # fp8 res=4 (heavy, -6%); int8 res=2 (light, -2.6%, spill-free).
                # The heavy fp8 dose would spill on int8 (extra I2F/dequant regs);
                # the light int8 dose underuses fp8's idle FMA slots. Each dose was
                # tuned to its dtype's register headroom + bottleneck.
                e2e=(
                    mask_fn is None
                    and self.head_dim_padded <= 128
                    and (
                        not self.is_sm103 or (self.is_nvf4_qk and not self.skip_softmax_error > 0.0)
                    )
                ),
                e2e_freq=self.e2e_freq,
                e2e_res=self.e2e_res,
                e2e_frg_limit=self.e2e_frg_limit,
                # Explicit 4-wide E4M3 pack of P -> ptxas emits F2FP.MERGE_C
                # instead of ~192 PRMT/launch: ~10% faster fp8, -3.4% int8_block,
                # cosine identical. On whenever P is fp8 (both fp8 and int8 paths).
                pack_fp8=const_expr(self.p_is_fp8),
                int_input=const_expr(self.is_int8),
                row_max=row_max if const_expr(self.is_int8) else None,
                k_descale=(k_descale_n if const_expr(self.is_int8 and has_3d_k_descale) else None),
            )

        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4)
        # Copy P registers -> TMEM, then signal barriers in two halves.
        if const_expr(self.is_nvf4_qk):
            # Match the PV-consumer boundary (see gemm_Pi pre_mbar_tiles): both sides
            # use _mbar_p_split so the store split covers exactly what PV consumes
            # before its P_full_2 wait. Keeps the softmax-store / PV-gemm overlap.
            p_tmem_store_split_k = self._mbar_p_split(cute.size(tStP_r2t.shape[2]))
        else:
            p_tmem_store_split_k = cute.size(tStP_r2t.shape[2]) // (
                2 if const_expr(self.is_fp8 and self.is_sm103) else 4
            )
            if const_expr(not (self.is_fp8 and self.is_sm103)):
                p_tmem_store_split_k *= 3
        for i in cutlass.range_constexpr(p_tmem_store_split_k):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Write per-warp skip flag (must precede mbarrier_arrive(P_full_O_rescaled)
        # so the MMA WG sees all writes after mbarrier_wait).
        # Different warps may cover different M-row ranges with different skip
        # outcomes; each writes its own slot so there is no write race.
        # The MMA WG ANDs all slots to decide whether to skip gemm_Pi.
        # Only the MMA WG reads sSkip (gated on self.skip_pv_gemm); if PV skipping
        # is off the writes are dead code.  Gate both ends symmetrically so the
        # write block compiles away when skip_pv_gemm is False.
        if const_expr(self.skip_pv_gemm and softmax.has_skip_softmax):
            _tidx = thr_tmem_load.thr_idx
            _warp_in_wg = _tidx // cute.arch.WARP_SIZE
            _num_sw = const_expr(len(self.softmax0_warp_ids))
            _skip_slot = stage * _num_sw + _warp_in_wg
            if _tidx % cute.arch.WARP_SIZE == 0:
                sSkip[_skip_slot] = Int32(1) if should_skip_softmax else Int32(0)
        # Notify mma warp that P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(p_tmem_store_split_k, cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
        )
        if not should_skip_softmax:
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        else:
            # Skipped block: P ≈ 0, so row_sum gets no new contribution.
            # Only rescale existing row_sum by acc_scale (≈1.0 since max didn't change).
            softmax.row_sum[0] = softmax.row_sum[0] * acc_scale
        # acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mHeadMap: Optional[cute.Tensor] = None,
        svd_tensors: Optional[SvdCorrectionTensors] = None,
        mOutputAmax: Optional[cute.Tensor] = None,
        output_amax_chunk_seqlen: Int32 = 0,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScales = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
            self.qk_acc_dtype,
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

        tStScales_t2r = [
            thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)
        ]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        for stage in cutlass.range_constexpr(self.q_stage):
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

        softmax_corr_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            head_idx = self._real_head_idx(mHeadMap, head_idx)
            kv_head_idx = self._kv_head_idx(head_idx)
            # For 3D q_descale the scale varies per m-block within this work tile.
            # softmax_scale_log2_eff is only consumed by learnable_sink + LSE
            # computation here; use stage 0's m-block (real seq m = q_stage * m_block)
            # - LSE/sink precision across stages is matched by Q-uniform scale
            # when all Q tiles share a descale, which is an accepted MVP limitation.
            qk_descale, v_descale = self._load_effective_descales(
                descale_tensors,
                batch_idx,
                kv_head_idx,
                head_idx=head_idx,
                m_block_seq=self.q_stage * m_block,
            )
            if const_expr(self.score_mod is None):
                softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
            else:
                softmax_scale_log2_eff = softmax_scale_log2

            max_offset = (
                Float32(self.p_fp8_max_offset) if const_expr(self.p_is_fp8) else Float32(0.0)
            )
            max_offset_scale = (
                Float32(2.0**self.p_fp8_max_offset) if const_expr(self.p_is_fp8) else Float32(1.0)
            )
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                    None, None, head_idx, split_idx
                ]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))

            # Default LSE to -inf for invalid split_idx tiles
            stats = [
                (
                    0.0,
                    -Float32.inf
                    if const_expr(mLSE is not None or learnable_sink is not None)
                    else None,
                    True,
                )
            ] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + 0, softmax_corr_consumer_phase
                )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 0)
                if const_expr(self.q_stage == 2):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 1,
                        softmax_corr_consumer_phase,
                    )
                softmax_corr_consumer_phase ^= 1

                tSrScale_t2r = cute.make_rmem_tensor(tSrScale_t2r_shape, Float32)
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait for S0 / S1
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                            softmax_corr_consumer_phase,
                        )
                        if const_expr(self.is_sm103):
                            cute.copy(thr_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                            cute.arch.fence_view_async_tmem_load()
                            scale = tSrScale_t2r[0]
                        else:
                            scale = sScale[tidx + stage * self.m_block_size]
                        if const_expr(self.q_stage == 2):
                            cute.arch.mbarrier_arrive(
                                mbar_ptr + self.mbar_softmax_corr_empty_offset + (1 - stage)
                            )
                        else:
                            cute.arch.mbarrier_arrive(
                                mbar_ptr + self.mbar_softmax_corr_empty_offset + stage
                            )
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        # should_rescale = True
                        # if tidx == 0: cute.printf("Correction scale i = %d, for stage %d: %f, should_rescale = %d\n", i, stage, scale, should_rescale)
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # cute.arch.mbarrier_wait(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
                        if should_rescale:
                            self.correction_rescale(thr_mma_pv, tOtOs[stage], tidx, scale)
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage
                        )

                    softmax_corr_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                if const_expr(self.q_stage == 2):
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                (self.q_stage * m_block + stage) * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                        softmax_corr_consumer_phase,
                    )
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[
                            tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                        ]
                    else:
                        row_max = None
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + stage
                    )
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2_eff)
                                row_sum = max_offset_scale
                            else:
                                row_sum += cute.math.exp2(
                                    sink_val * LOG2_E
                                    - row_max * softmax_scale_log2_eff
                                    + max_offset,
                                    fastmath=True,
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(
                        row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0
                    )
                    scale = scale * v_descale
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage,
                            corr_epi_producer_phase,
                        )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtOs[stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O,
                        mOutputAmax,
                        output_amax_chunk_seqlen,
                    )
                    if const_expr(self.svd_delta_to_output):
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierFwd.Epilogue),
                            number_of_threads=len(self.correction_warp_ids) * cute.arch.WARP_SIZE,
                        )
                        self.add_svd_raw_v_delta_gmem(
                            svd_tensors,
                            mO_cur,
                            tidx,
                            batch_idx,
                            head_idx,
                            self.q_stage * m_block + stage,
                            seqlen.seqlen_q,
                            row_max,
                            scale,
                            softmax_scale_log2_eff,
                            max_offset,
                        )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_full_offset + stage)
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
                    # if tidx == 0: cute.printf("Correction final scale for stage %d: %f\n", stage, scale)

                o_corr_consumer_phase ^= 1
                softmax_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                # WARNING: we need some code before the const_expr, see https://github.com/NVIDIA/cutlass/issues/2781
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                else:
                    gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_block_sparsity):
                    (
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtOs,
                        sO,
                        mbar_ptr,
                        self.mbar_softmax_corr_full_offset,
                        self.mbar_softmax_corr_empty_offset,
                        self.mbar_P_full_O_rescaled_offset,
                        self.mbar_P_full_2_offset,
                        self.mbar_corr_epi_full_offset,
                        self.mbar_corr_epi_empty_offset,
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2_eff,
                        max_offset,
                        max_offset_scale,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                        mOutputAmax,
                        output_amax_chunk_seqlen,
                    )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    gLSE = cute.local_tile(
                        mLSE_cur, (self.m_block_size,), (self.q_stage * m_block + stage,)
                    )
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    # if tidx == 0 and stage <= 1:
                    #     cute.printf("row_sum = {}, row_max = {}, acc_O_mn_row_is_zero_or_nan = {}\n", row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    LN2 = math.log(2.0)
                    lse = (
                        (
                            row_max * softmax_scale_log2_eff
                            + (cute.math.log2(row_sum, fastmath=True) - max_offset)
                        )
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if tidx < seqlen_q - (self.q_stage * m_block + stage) * self.m_block_size:
                        # This actually just works with PackGQA too
                        gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
        mOutputAmax: Optional[cute.Tensor] = None,
        output_amax_chunk_seqlen: Int32 = 0,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0]).get_slice(
            tidx
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range_constexpr(self.head_dim_v_padded // corr_tile_size):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOrO_frg_cvt = cute.make_rmem_tensor(tOrO_frg.shape, self.o_dtype)
            tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
            cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_r2s_i)
        # fence view async shared
        cute.arch.fence_view_async_shared()

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O
            assert gmem_tiled_copy_O is not None
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = utils.predicate_k(tOcO, limit=mO_cur.shape[1])
            pack_gqa = PackGQA(
                self.m_block_size,
                self.head_dim_v_padded,
                self.check_hdim_v_oob,
                self.qhead_per_kvhead,
            )

            # load acc O from smem to rmem for wider vectorization
            tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
            cute.autovec_copy(tOsO, tOrO)
            # copy acc O from rmem to gmem
            if const_expr(not self.pack_gqa):
                tile_row_base = (self.q_stage * m_block + stage) * self.m_block_size
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if t0OcO[0, rest_m, 0][0] < seqlen_q - tile_row_base - tOcO[0][0]:
                        if const_expr(mOutputAmax is not None):
                            global_row = tile_row_base + tOcO[0][0] + t0OcO[0, rest_m, 0][0]
                            chunk_id = Int32(0)
                            if output_amax_chunk_seqlen > Int32(0):
                                chunk_id = cutlass.min(
                                    global_row // output_amax_chunk_seqlen,
                                    cute.size(mOutputAmax) - 1,
                                )
                            for ki in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                                elem_valid = True
                                if const_expr(self.check_hdim_v_oob):
                                    elem_valid = tOpO[0, rest_m, ki]
                                if elem_valid:
                                    fval = tOrO[0, rest_m, ki].to(Float32)
                                    fabs = utils.fmax(fval, -fval)
                                    utils.atomic_max_fp32(
                                        fabs, utils.elem_pointer(mOutputAmax, (chunk_id,))
                                    )
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                pack_gqa.store_O(
                    mO_cur,
                    tOrO,
                    gmem_tiled_copy_O,
                    tidx,
                    self.q_stage * m_block + stage,
                    seqlen_q,
                    mOutputAmax,
                    output_amax_chunk_seqlen,
                )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mHeadMap: Optional[cute.Tensor] = None,
        mOutputAmax: Optional[cute.Tensor] = None,
        output_amax_chunk_seqlen: Int32 = 0,
    ):
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            head_idx = self._real_head_idx(mHeadMap, head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                        None, None, head_idx, split_idx
                    ]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
                if const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=self.q_stage * m_block + stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        if const_expr(self.q_stage == 2):
                            cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                        else:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage
                        )
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                    pack_gqa = PackGQA(
                        self.m_block_size,
                        self.head_dim_v_padded,
                        self.check_hdim_v_oob,
                        self.qhead_per_kvhead,
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        # load acc O from smem to rmem for wider vectorization
                        tOrO = cute.make_fragment_like(tOsO[None, None, None, 0], self.o_dtype)
                        cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                        # copy acc O from rmem to gmem
                        if const_expr(not self.pack_gqa):
                            tile_row_base = (self.q_stage * m_block + stage) * self.m_block_size
                            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                                if (
                                    t0OcO[0, rest_m, 0][0]
                                    < seqlen.seqlen_q - tile_row_base - tOcO[0][0]
                                ):
                                    if const_expr(mOutputAmax is not None):
                                        global_row = (
                                            tile_row_base + tOcO[0][0] + t0OcO[0, rest_m, 0][0]
                                        )
                                        chunk_id = Int32(0)
                                        if output_amax_chunk_seqlen > Int32(0):
                                            chunk_id = cutlass.min(
                                                global_row // output_amax_chunk_seqlen,
                                                cute.size(mOutputAmax) - 1,
                                            )
                                        for ki in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                                            elem_valid = True
                                            if const_expr(self.check_hdim_v_oob):
                                                elem_valid = tOpO[0, rest_m, ki]
                                            if elem_valid:
                                                fval = tOrO[0, rest_m, ki].to(Float32)
                                                fabs = utils.fmax(fval, -fval)
                                                utils.atomic_max_fp32(
                                                    fabs,
                                                    utils.elem_pointer(mOutputAmax, (chunk_id,)),
                                                )
                                    cute.copy(
                                        gmem_tiled_copy_O,
                                        tOrO[None, rest_m, None],
                                        tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                                        pred=tOpO[None, rest_m, None]
                                        if const_expr(self.check_hdim_v_oob)
                                        else None,
                                    )
                        else:
                            pack_gqa.store_O(
                                mO_cur,
                                tOrO,
                                gmem_tiled_copy_O,
                                tidx,
                                self.q_stage * m_block + stage,
                                seqlen.seqlen_q,
                                mOutputAmax,
                                output_amax_chunk_seqlen,
                            )
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage
                        )

                epi_consumer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
        load_SFQ_fn: Optional[Callable] = None,
    ):
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes["Q"])
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)
        if const_expr(load_SFQ_fn is not None):
            load_SFQ_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        tma_atom_sf: Optional[cute.CopyAtom] = None,
        tXgSF: Optional[cute.Tensor] = None,
        tXsSF: Optional[cute.Tensor] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)

        if const_expr(self.use_tma_KV):
            assert tXgX is not None and tXsX is not None and tma_atom is not None
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full_ptr + stage,
                    self.tma_copy_bytes[K_or_V],
                )
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
            tXgX_cur = (
                tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
            )
            cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)
            if const_expr(tma_atom_sf is not None):
                assert tXgSF is not None and tXsSF is not None
                tXsSF_cur = tXsSF[None, stage]
                tXgSF_cur = (
                    tXgSF[None, block] if const_expr(page_idx is None) else tXgSF[None, 0, page_idx]
                )
                cute.copy(tma_atom_sf, tXgSF_cur, tXsSF_cur, tma_bar_ptr=mbar_full_ptr + stage)
        else:
            assert paged_kv_manager is not None
            paged_kv_manager.load_KV(block, sX[None, None, None, stage], K_or_V)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_mbarrier_arrive_noinc(mbar_full_ptr + stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        if self.use_tma_KV:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids)
            )
            return cutlass.pipeline.PipelineTmaUmma.create(
                barrier_storage=load_kv_mbar_ptr,
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
            )
        else:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids) * cute.arch.WARP_SIZE
            )
            return cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                barrier_storage=load_kv_mbar_ptr,
            )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def apply_svd_exact_topk_score(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        svd_tensors: SvdCorrectionTensors,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        n_block: Int32,
    ) -> None:
        """Overwrite proxy scores for per-block top-k entries with exact raw QK.

        FA4 still produces the low-rank proxy S tile through the normal QK WGMMA.
        The softmax warp then picks top-k proxy entries inside each Exp16 K
        block and recomputes only those scores from raw Q/K before rowmax and
        online softmax.  The loop computes one exact dot per selected token,
        not one dot per S element, to keep CuTe IR size bounded.
        """
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        n_vals = cutlass.const_expr(cute.size(tSrS_t2r.shape))
        scores = cute.make_rmem_tensor(tSrS_t2r.shape, Float32)
        picked = cute.make_rmem_tensor(tSrS_t2r.shape, Int32)
        scores.store(tSrS_t2r.load())
        picked.fill(0)
        q_idx = tScS_t2r[0][0]
        q_bias = Float32(svd_tensors.q_mean[batch_idx, q_idx, head_idx])
        for i in cutlass.range_constexpr(n_vals):
            scores[i] = scores[i] + q_bias
            tSrS_t2r[i] = scores[i]

        for block_start in cutlass.range_constexpr(0, self.n_block_size, self.svd_k_block):
            for _top_i in cutlass.range_constexpr(self.svd_topk):
                best = -Float32.inf
                for i in cutlass.range_constexpr(n_vals):
                    kv_idx = tScS_t2r[i][1]
                    local_k = kv_idx - n_block * self.n_block_size
                    in_svd_block = (local_k >= Int32(block_start)) & (
                        local_k < Int32(block_start + self.svd_k_block)
                    )
                    available = in_svd_block & (picked[i] == Int32(0))
                    cand = scores[i] if available else -Float32.inf
                    best = utils.fmax(best, cand)

                sel_kv = Int32(0)
                selected_any = False
                for i in cutlass.range_constexpr(n_vals):
                    kv_idx = tScS_t2r[i][1]
                    local_k = kv_idx - n_block * self.n_block_size
                    in_svd_block = (local_k >= Int32(block_start)) & (
                        local_k < Int32(block_start + self.svd_k_block)
                    )
                    is_pick = in_svd_block & (picked[i] == Int32(0)) & (scores[i] == best)
                    sel_kv = kv_idx if is_pick else sel_kv
                    selected_any = selected_any | is_pick
                    picked[i] = Int32(1) if is_pick else picked[i]

                exact = Float32(0.0)
                for d in cutlass.range_constexpr(self.svd_raw_head_dim):
                    qv = Float32(svd_tensors.raw_q[batch_idx, q_idx, head_idx, d])
                    kv = Float32(svd_tensors.raw_k[batch_idx, sel_kv, head_idx, d])
                    exact = exact + qv * kv

                for i in cutlass.range_constexpr(n_vals):
                    kv_idx = tScS_t2r[i][1]
                    local_k = kv_idx - n_block * self.n_block_size
                    in_svd_block = (local_k >= Int32(block_start)) & (
                        local_k < Int32(block_start + self.svd_k_block)
                    )
                    is_selected = selected_any & in_svd_block & (kv_idx == sel_kv)
                    tSrS_t2r[i] = exact if is_selected else tSrS_t2r[i]

    @cute.jit
    def add_svd_raw_v_delta_gmem(
        self,
        svd_tensors: SvdCorrectionTensors,
        mO_cur: cute.Tensor,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block_seq: Int32,
        seqlen_q: Int32,
        row_max: Float32,
        final_scale: Float32,
        softmax_scale_log2_eff: Float32,
        max_offset: Float32,
    ) -> None:
        """Add normalized selected raw-V residual directly to O in global memory.

        This is the first direct CuTe path for the final SVD correction: FA4 PV
        accumulates the full-D low-rank reconstruction V_hat, and this epilogue
        patch adds sum_selected softmax_exact * (raw_v - V_hat).  It is row-wise
        for now; the next optimization is to tile/vectorize this residual path.
        """
        q_local = tidx
        q_idx = m_block_seq * self.m_block_size + q_local
        if q_local < Int32(self.m_block_size) and q_idx < seqlen_q:
            q_bias = Float32(svd_tensors.q_mean[batch_idx, q_idx, head_idx])
            picked = cute.make_rmem_tensor((self.svd_k_block,), Int32)
            scores = cute.make_rmem_tensor((self.svd_k_block,), Float32)
            for block_i in cutlass.range_constexpr(self.svd_num_k_blocks):
                k_start = Int32(block_i * self.svd_k_block)
                picked.fill(0)
                for kk in cutlass.range_constexpr(self.svd_k_block):
                    kv_idx = k_start + Int32(kk)
                    proxy = q_bias
                    for r in cutlass.range_constexpr(self.head_dim_padded):
                        qv = Float32(svd_tensors.q_low[batch_idx, q_idx, head_idx, r])
                        kv = (
                            Float32(svd_tensors.k_coord[batch_idx, kv_idx, head_idx, r])
                            if kv_idx < seqlen_q
                            else Float32(0.0)
                        )
                        proxy = proxy + qv * kv
                    scores[kk] = proxy if kv_idx < seqlen_q else -Float32.inf

                for _top_i in cutlass.range_constexpr(self.svd_topk):
                    best = -Float32.inf
                    sel_local = Int32(0)
                    selected_any = False
                    for kk in cutlass.range_constexpr(self.svd_k_block):
                        available = picked[kk] == Int32(0)
                        cand = scores[kk] if available else -Float32.inf
                        is_better = cand > best
                        best = cand if is_better else best
                        sel_local = Int32(kk) if is_better else sel_local
                        selected_any = selected_any | available
                    picked[sel_local] = Int32(1)
                    sel_k = k_start + sel_local
                    if selected_any and sel_k < seqlen_q:
                        exact = Float32(0.0)
                        for d_raw in cutlass.range_constexpr(self.svd_raw_head_dim):
                            qv = Float32(svd_tensors.raw_q[batch_idx, q_idx, head_idx, d_raw])
                            kv = Float32(svd_tensors.raw_k[batch_idx, sel_k, head_idx, d_raw])
                            exact = exact + qv * kv
                        w = (
                            cute.math.exp2(
                                exact * softmax_scale_log2_eff
                                - row_max * softmax_scale_log2_eff
                                + max_offset,
                                fastmath=True,
                            )
                            * final_scale
                        )
                        for d_out in cutlass.range_constexpr(self.svd_raw_head_dim):
                            vhat = Float32(svd_tensors.v_mean[head_idx, d_out])
                            for rv in cutlass.range_constexpr(self.svd_v_rank):
                                vc = Float32(svd_tensors.v_coord[batch_idx, sel_k, head_idx, rv])
                                vb = Float32(svd_tensors.v_basis[head_idx, d_out, rv])
                                vhat = vhat + vc * vb
                            raw_v = Float32(svd_tensors.raw_v[batch_idx, sel_k, head_idx, d_out])
                            old = Float32(mO_cur[d_out, q_idx])
                            mO_cur[d_out, q_idx] = (old + w * (raw_v - vhat)).to(self.o_dtype)

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            assert head_divmod is not None
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical, head_offset = divmod(q_physical, head_divmod)
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = divmod(q_idx_logical, seqlen_q_divmod)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
        )
