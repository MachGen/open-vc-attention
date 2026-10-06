# Copyright (c) 2025, Tri Dao.

from typing import Optional, Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import layout_utils
import open_vc_attn._kernels.blackwell.flash_attn.cute.utils as utils
from open_vc_attn._kernels.blackwell.flash_attn.cute.seqlen_info import SeqlenInfoQK


@cute.jit
def mask_r2p(X: cute.Tensor, col_limit: Int32, rank1: bool = False) -> None:
    # Bit manipulation, compiles down to the R2P instruction
    # For sm100: we know that tScS_t2r[i][1] == i, for the particular tmem copy atom we're using.
    col_limit_transformed = col_limit
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    # Ideally we'd move by 32 instead of 24, but mask >> i isn't correct for i == 31
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        # Don't need to clamp to 32 since the shr.u32 instruction does that already
        col_limit_right_s = max(col_limit_transformed - s * 24, 0)
        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
        mask = (1 << col_limit_right_s) - 1
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask & (1 << i))
            c = s * 24 + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
                # This is the equivalent of:
                # X[s * 24 + i] = X[s * 24 + i] if col_limit_right_s <= i else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def mask_r2p_dual_bound(
    X: cute.Tensor,
    col_limit_left: Int32,  # Inclusive lower bound
    col_limit_right: Int32,  # Exclusive upper bound
) -> None:
    """
    Dual-bound masking using two bitmasks for SM100, following mask_r2p.
    Masks elements where: NOT (col_limit_left <= col < col_limit_right)

    Uses bit manipulation to create a range mask:
        mask_right = (1 << right) - 1  -> bits (right-1)..0 are 1
        mask_left  = (1 << left) - 1   -> bits (left-1)..0 are 1
        mask_range = mask_range = mask_right & ~ mask_left -> bits (right-1)..left are 1
    """
    ncol = const_expr(cute.size(X.shape))

    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        right_s = max(col_limit_right - s * 24, 0)
        left_s = max(col_limit_left - s * 24, 0)

        # otherwise cute dsl complains about python int too large to convert into c long
        right_s = min(right_s, 24)
        left_s = min(left_s, 24)

        # bits (right-1)..left are 1
        mask_right = (1 << right_s) - 1
        mask_left = (1 << left_s) - 1
        mask_range = mask_right & ~mask_left

        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask_range & (1 << i))
            c = s * 24 + i
            X[c] = X[c] if in_bound else -Float32.inf


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # only pass in if we're doing PackGQA
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n
        r2p = True
        if const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        # if tScS_t2r[i][1] >= seqlenk_col_limit:
                        #     acc_S[i] = -Float32.inf
                        # For some reason the 2 lines above generate really bad SASS
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= seqlenk_col_limit else acc_S[i]
                else:
                    mask_r2p(acc_S, seqlenk_col_limit, rank1=True)

        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # Block sparse case w/ mask_mod
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)

            ncol = const_expr(cute.size(tScS_t2r.shape))
            for i in cutlass.range_constexpr(ncol):
                row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
                col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
                global_row = row_coord + m_block * self.tile_m
                global_col = col_coord + n_block * self.tile_n

                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert head_divmod is not None
                    mask_row, head_offset = divmod(global_row, head_divmod)
                    head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
                else:
                    head_idx_for_mod = head_idx
                    mask_row = global_row

                mask_row_for_mod = mask_row
                if const_expr(has_fastdiv and aux_tensors is not None):
                    if check_q_boundary:
                        _, mask_row_for_mod = divmod(mask_row, fastdiv_mods[0])
                global_col_for_mod = global_col
                if const_expr(has_fastdiv and mask_seqlen and aux_tensors is not None):
                    _, global_col_for_mod = divmod(global_col, fastdiv_mods[1])

                head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                mask_row_ssa = utils.scalar_to_ssa(mask_row_for_mod, cutlass.Int32)
                kv_idx_ssa = utils.scalar_to_ssa(global_col_for_mod, cutlass.Int32)
                mask_value = mask_mod(
                    batch_idx_ssa,
                    head_idx_ssa,
                    mask_row_ssa,
                    kv_idx_ssa,
                    self.seqlen_info,
                    aux_tensors,
                )
                cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                acc_S[i] = acc_S[i] if cond else -Float32.inf
                if const_expr(mask_seqlen):
                    acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]
                if check_q_boundary:
                    acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]

        else:  # Causal or local
            causal_row_offset = 1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                # if cute.arch.thread_idx()[0] % 32 == 0:
                #     cute.printf("tidx = %d, tidx tmem = %d, row_idx = %d, col_limit_right = %d, causal_row_offset = %d\n", cute.arch.thread_idx()[0], thr_tmem_load.thr_idx, row_idx, col_limit_right, causal_row_offset)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= col_limit_right else acc_S[i]
                else:
                    mask_r2p(acc_S, col_limit_right, rank1=True)
            else:
                local_row_offset_right = (
                    causal_row_offset + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - 1 - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                if const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                else:
                    col_limit_right = self.tile_n
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if const_expr(self.window_size_left is not None)
                    else 0
                )
                if const_expr(not r2p):
                    # if cute.arch.thread_idx()[0] == 0 or cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", m_block, n_block, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        col_idx = tScS_t2r[i][1]
                        acc_S[i] = (
                            -Float32.inf
                            if col_idx >= col_limit_right or col_idx < col_limit_left
                            else acc_S[i]
                        )
                else:
                    # XOR-based R2P dual bound masking
                    mask_r2p_dual_bound(acc_S, col_limit_left, col_limit_right)
