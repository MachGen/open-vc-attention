# Copyright (c) 2025, Tri Dao.
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass._mlir.dialects import llvm

import vc_attn._kernels.v2.flash_attn.cute.mma_sm100_desc as sm100_desc
from vc_attn._kernels.v2.flash_attn.cute.utils import parse_swizzle_from_pointer


# CUTLASS 4.4 compatibility: MmaF8F6F4Op / MmaMXF8F6F4Op were added in a later
# release.  Use getattr so isinstance() simply never matches on older builds.
_MmaF8F6F4Op = getattr(tcgen05.mma, "MmaF8F6F4Op", type("_MmaF8F6F4Op", (), {}))
_MmaMXF8F6F4Op = getattr(tcgen05.mma, "MmaMXF8F6F4Op", type("_MmaMXF8F6F4Op", (), {}))


def _tcgen05_mma_kind(op: cute.nvgpu.tcgen05.mma.MmaOp) -> str:
    if isinstance(op, tcgen05.mma.MmaF16BF16Op):
        return "f16"
    if isinstance(op, tcgen05.mma.MmaTF32Op):
        return "tf32"
    if isinstance(op, tcgen05.mma.MmaI8Op):
        return "i8"
    if isinstance(op, (tcgen05.mma.MmaFP8Op, _MmaF8F6F4Op)):
        return "f8f6f4"
    if isinstance(op, (tcgen05.mma.MmaMXF8Op, _MmaMXF8F6F4Op)):
        return "mxf8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF4Op):
        return "mxf4"
    if isinstance(op, tcgen05.mma.MmaMXF4NVF4Op):
        return "mxf4nvf4"
    raise TypeError(f"Unsupported tcgen05 MMA op kind: {type(op).__name__}")


@cute.jit
def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    swap_AB: bool = False,
) -> None:
    if const_expr(swap_AB):
        return gemm_w_idx(
            tiled_mma, acc, tCrB, tCrA, B_idx, A_idx, zero_init=zero_init, swap_AB=False
        )
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
            cute.gemm(mma_atom, acc, rA[None, None, k], rB[None, None, k], acc)


@cute.jit
def gemm_ptx_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    **kwargs,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    sA_cur = None
    if const_expr(sA is not None):
        sA_cur = sA if const_expr(A_idx is None) else sA[None, None, None, A_idx]
    sB_cur = sB if const_expr(B_idx is None) else sB[None, None, None, B_idx]
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    acc_tmem_addr = acc.iterator.toint()
    gemm_ptx_partial(
        mma_atom.op, acc_tmem_addr, rA, rB, sA_cur, sB_cur, zero_init=zero_init, **kwargs
    )


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> cute.TiledMma:
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        tiled_mma.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
    return tiled_mma


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    """Convert a 64-bit integer to a tuple of two 32-bit integers."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


def _smem_desc_base(op_dtype, major_mode, smem: cute.Tensor) -> Tuple[int, int]:
    """Compile-time (lo, hi) words of a shared-memory matrix descriptor."""
    base = sm100_desc.make_smem_desc_base(
        cute.recast_layout(128, op_dtype.width, smem.layout[0]),
        parse_swizzle_from_pointer(smem.iterator),
        sm100_desc.Major.K
        if major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else sm100_desc.Major.MN,
    )
    return i64_to_i32x2(base)


def _mma_descriptors(op, tCrA, tCrB, sA, sB):
    """Static instruction descriptor, descriptor high words and per-k operand offsets.

    A comes from shared memory (SS) or tensor memory (TS, sA is None).
    """
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if not is_ts:
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    idesc = sm100_desc.mma_op_to_idesc(op)
    a_hi = None if is_ts else _smem_desc_base(op.a_dtype, op.a_major_mode, sA)[1]
    b_hi = _smem_desc_base(op.b_dtype, op.b_major_mode, sB)[1]
    tCrA_layout = (
        cute.recast_layout(32, tCrA.element_type.width, tCrA.layout) if is_ts else tCrA.layout
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    return is_ts, idesc, a_hi, b_hi, offset_a, offset_b


def _mma_start_descriptors(op, tCrA, sA, sB, is_ts):
    """Runtime low words of the A/B descriptors at k = 0 (A is None for TS)."""
    start_a = None
    if not is_ts:
        a_lo = _smem_desc_base(op.a_dtype, op.a_major_mode, sA)[0]
        start_a = Int32(a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
    b_lo = _smem_desc_base(op.b_dtype, op.b_major_mode, sB)[0]
    start_b = Int32(b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    return start_a, start_b


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    tA_addr: Optional[Int32] = None,
    mbar_wait_fraction_num: cutlass.Constexpr[int] = 3,
    mbar_wait_fraction_den: cutlass.Constexpr[int] = 4,
    pre_mbar_tiles: Optional[cutlass.Constexpr[int]] = None,
    extra_cols: cutlass.Constexpr[int] = 0,
) -> None:
    is_ts, idesc, smem_desc_a_hi, smem_desc_b_hi, offset_a, offset_b = _mma_descriptors(
        op, tCrA, tCrB, sA, sB
    )
    if const_expr(extra_cols):
        # Extend PV into the constant shared-memory operand beside V.
        idesc = const_expr((idesc & ~(0x3F << 17)) | (((op.shape_mnk[1] + extra_cols) // 8) << 17))
    kind = _tcgen05_mma_kind(op)
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))]
    smem_desc_start_a_lo, smem_desc_start_b_lo = _mma_start_descriptors(op, tCrA, sA, sB, is_ts)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        # For TS gemm, somehow tCrA.iterator.toint() returns 0 no matter what, so we need to
        # explicitly pass in the tA_addr for correctness.
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        mbar_wait_k = cute.size(tCrA.shape[2])
        if const_expr(mbar_ptr is not None):
            if const_expr(pre_mbar_tiles is not None):
                mbar_wait_k = pre_mbar_tiles
            else:
                mbar_wait_k = (
                    cute.size(tCrA.shape[2]) // mbar_wait_fraction_den * mbar_wait_fraction_num
                )
        # The fused 32/96 handoff reaches the resumed segment before the prefix
        # loop initializes this descriptor. Preserve other paths' instruction stream.
        init_b_desc = const_expr(
            "mov.b32 smem_desc_b_lo, smem_desc_b_lo_start;\n\t" if extra_cols else ""
        )
        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            + init_b_desc
            + f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2]) if const_expr(mbar_ptr is None) else mbar_wait_k,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(mbar_wait_k, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def tmem_ld_red_max(
    tStS: cute.Tensor,
    tSrS: cute.Tensor,
    early_release_mbar_ptr: Optional[cutlass.Pointer] = None,
) -> cutlass.Float32:
    """SM103: fused TMEM load + row-max via raw tcgen05.ld.red PTX.

    Drop-in replacement for cute.copy(thr_tmem_load, src, dst) that also
    returns the row max. Same instruction count as baseline (one x32 load
    per tile), with the max computed in the TMEM controller at zero ALU cost.

    has_side_effects=True is load-bearing: it makes MLIR treat the load as
    impure so LICM cannot hoist it out of the KV-block loop (cute-dsl models
    tcgen05.ld.red as side-effect-free, which would otherwise read a stale
    block-0 S tile).
    """
    from cutlass._mlir import ir as _ir

    # tStS = thr_tmem_load.partition_S(...) has layout (((32,32),1), nseg, 1, 1):
    # mode-0 is the 32-wide datapath (+ 32 warp lanes), mode-1 the column-segments
    # (stride 32).  One x32 ld.red per segment yields this lane's 32 datapath values
    # plus the segment max; nseg of them tile the full per-thread row.
    num_tiles = cute.size(tStS.shape[1])
    f32_ty = _ir.F32Type.get()
    row_max = cutlass.Float32(0.0)
    if cutlass.const_expr(num_tiles % 2 == 0):
        struct_ty = llvm.StructType.get_literal([f32_ty] * 65)
        asm_str = (
            "tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max"
            " {" + ", ".join(f"${i}" for i in range(64)) + "}"
            ", $64, [$65];\n"
        )
        for k in cutlass.range_constexpr(num_tiles // 2):
            result = llvm.inline_asm(
                struct_ty,
                [tStS[None, k * 2, None, None].iterator.toint().ir_value()],
                asm_str,
                "=f," * 65 + "r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
            for i in cutlass.range_constexpr(64):
                tSrS[k * 64 + i] = cutlass.Float32(llvm.extractvalue(f32_ty, result, [i]))
            tile_max = cutlass.Float32(llvm.extractvalue(f32_ty, result, [64]))
            row_max = tile_max if cutlass.const_expr(k == 0) else cute.arch.fmax(row_max, tile_max)
            if cutlass.const_expr(early_release_mbar_ptr is not None and k == 0):
                cute.arch.fence_view_async_tmem_load()
                if cute.arch.lane_idx() == 0:
                    cute.arch.mbarrier_arrive(early_release_mbar_ptr)
    else:
        struct_ty = llvm.StructType.get_literal([f32_ty] * 33)
        asm_str = (
            "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max"
            " {" + ", ".join(f"${i}" for i in range(32)) + "}"
            ", $32, [$33];\n"
        )
        for k in cutlass.range_constexpr(num_tiles):
            result = llvm.inline_asm(
                struct_ty,
                [tStS[None, k, None, None].iterator.toint().ir_value()],
                asm_str,
                "=f," * 33 + "r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
            for i in cutlass.range_constexpr(32):
                tSrS[k * 32 + i] = cutlass.Float32(llvm.extractvalue(f32_ty, result, [i]))
            tile_max = cutlass.Float32(llvm.extractvalue(f32_ty, result, [32]))
            row_max = tile_max if cutlass.const_expr(k == 0) else cute.arch.fmax(row_max, tile_max)
            if cutlass.const_expr(early_release_mbar_ptr is not None and k == 0):
                cute.arch.fence_view_async_tmem_load()
                if cute.arch.lane_idx() == 0:
                    cute.arch.mbarrier_arrive(early_release_mbar_ptr)
    return row_max


def _is_pure_fp8_mma(op: cute.nvgpu.tcgen05.mma.MmaOp) -> bool:
    return (
        not hasattr(op, "sf_dtype")
        and op.a_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
        and op.b_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
    )


def _mma_inst_kind(op: cute.nvgpu.tcgen05.mma.MmaOp) -> str:
    if hasattr(op, "sf_dtype"):
        if op.a_dtype is cutlass.Float4E2M1FN:
            return "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        return "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
    if _is_pure_fp8_mma(op):
        return "tcgen05.mma.cta_group::1.kind::f8f6f4"
    return "tcgen05.mma.cta_group::1.kind::f16"


@cute.jit
def gemm_ptx_partial_fp4(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    tScaleA: cute.Tensor,
    tScaleB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    tA_addr: Optional[Int32] = None,
    pre_mbar_tiles: Optional[cutlass.Constexpr[int]] = None,
) -> None:
    is_ts, idesc, smem_desc_a_hi, smem_desc_b_hi, offset_a, offset_b = _mma_descriptors(
        op, tCrA, tCrB, sA, sB
    )
    scale_A_base_col = tcgen05.find_tmem_tensor_col_offset(tScaleA[None, None, 0])
    scale_B_base_col = tcgen05.find_tmem_tensor_col_offset(tScaleB[None, None, 0])
    offset_sfa = [
        tcgen05.find_tmem_tensor_col_offset(tScaleA[None, None, k]) - scale_A_base_col
        for k in range(cute.size(tCrA.shape[2]))
    ]
    offset_sfb = [
        tcgen05.find_tmem_tensor_col_offset(tScaleB[None, None, k]) - scale_B_base_col
        for k in range(cute.size(tCrB.shape[2]))
    ]
    smem_desc_start_a_lo, smem_desc_start_b_lo = _mma_start_descriptors(op, tCrA, sA, sB, is_ts)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        num_k = const_expr(cute.size(tCrA.shape[2]))
        scale_A_addrs = [
            Int32(cute.arch.make_warp_uniform(tScaleA[None, None, k].iterator.toint())).ir_value()
            for k in range(num_k)
        ]
        scale_B_addrs = [
            Int32(cute.arch.make_warp_uniform(tScaleB[None, None, k].iterator.toint())).ir_value()
            for k in range(num_k)
        ]
        mma_inst_str = const_expr(_mma_inst_kind(op))
        sfa_op_base = 4
        sfb_op_base = 4 + num_k
        input_args = (
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ]
            + scale_A_addrs
            + scale_B_addrs
        )

        k0_desc_setup = (
            "mov.b64 smem_desc_a, {smem_desc_a_lo_start, smem_desc_a_hi};\n\t"
            "mov.b64 smem_desc_b, {smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
        )

        def _kk_desc_setup(kk):
            return (
                f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[kk])};\n\t"
                f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[kk])};\n\t"
                f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            )

        def _mma_k_block(kk):
            pred = pred_str if kk == 0 else "1"
            desc_setup = k0_desc_setup if kk == 0 else _kk_desc_setup(kk)
            return (
                f"mov.b32 tmem_scale_a, ${sfa_op_base + kk};\n\t"
                f"mov.b32 tmem_scale_b, ${sfb_op_base + kk};\n\t"
                f"mov.b32 idesc, {hex(idesc)};\n\t"
                "and.b32 sf_id_bits, tmem_scale_a, 0xC0000000;\n\t"
                "shr.u32 sf_id_bits, sf_id_bits, 1;\n\t"
                "or.b32 idesc, idesc, sf_id_bits;\n\t"
                "and.b32 sf_id_bits, tmem_scale_b, 0xC0000000;\n\t"
                "shr.u32 sf_id_bits, sf_id_bits, 26;\n\t"
                "or.b32 idesc, idesc, sf_id_bits;\n\t"
                + desc_setup
                + f"@leader_thread {mma_inst_str} [tmem_acc], smem_desc_a, smem_desc_b, "
                f"idesc, [tmem_scale_a], [tmem_scale_b], {pred};\n\t"
            )

        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 sf_id_bits;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_scale_a;\n\t"
            ".reg .b32 tmem_scale_b;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t" + "".join(_mma_k_block(k) for k in range(num_k)) + "}\n",
            ",".join(["r"] * len(input_args)),
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        scale_A_base_addr = tScaleA[None, None, 0].iterator.toint()
        scale_B_base_addr = tScaleB[None, None, 0].iterator.toint()
        mma_inst_str = const_expr(_mma_inst_kind(op))
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(scale_A_base_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(scale_B_base_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$6], $7, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""

        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 sf_id_bits;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            ".reg .b32 tmem_scale_a;\n\t"
            ".reg .b32 tmem_scale_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 tmem_scale_a, $4;\n\t"
            f"mov.b32 tmem_scale_b, $5;\n\t"
            "and.b32 sf_id_bits, tmem_scale_a, 0xC0000000;\n\t"
            "shr.u32 sf_id_bits, sf_id_bits, 1;\n\t"
            "or.b32 idesc, idesc, sf_id_bits;\n\t"
            "and.b32 sf_id_bits, tmem_scale_b, 0xC0000000;\n\t"
            "shr.u32 sf_id_bits, sf_id_bits, 26;\n\t"
            "or.b32 idesc, idesc, sf_id_bits;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread {mma_inst_str} [tmem_acc], [tmem_a], smem_desc_b, idesc, "
            f"[tmem_scale_a], [tmem_scale_b], {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread {mma_inst_str} [tmem_acc], [tmem_a + {hex(offset_a[k])}], "
                    f"smem_desc_b, idesc, [tmem_scale_a + {hex(offset_sfa[k])}], "
                    f"[tmem_scale_b + {hex(offset_sfb[k])}], 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2])
                    if const_expr(mbar_ptr is None)
                    else (
                        pre_mbar_tiles
                        if const_expr(pre_mbar_tiles is not None)
                        else cute.size(tCrA.shape[2]) // 4 * 3
                    ),
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread {mma_inst_str} [tmem_acc], [tmem_a + {hex(offset_a[k])}], "
                        f"smem_desc_b, idesc, [tmem_scale_a + {hex(offset_sfa[k])}], "
                        f"[tmem_scale_b + {hex(offset_sfb[k])}], 1;\n\t"
                    )
                    for k in range(
                        max(
                            1,
                            pre_mbar_tiles
                            if const_expr(pre_mbar_tiles is not None)
                            else cute.size(tCrA.shape[2]) // 4 * 3,
                        ),
                        cute.size(tCrA.shape[2]),
                    )
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@dataclass(frozen=True)
class BlockScaledBasicChunk:
    sf_vec_size: int
    major_mode: OperandMajorMode = OperandMajorMode.K
    _layout: cute.Layout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.major_mode == OperandMajorMode.K:
            atom_shape = ((32, 4), (self.sf_vec_size, 4))
            atom_stride = ((16, 4), (0, 1))
        else:
            atom_shape = ((self.sf_vec_size, 4), (32, 4))
            atom_stride = ((0, 1), (16, 4))
        object.__setattr__(self, "_layout", cute.make_layout(atom_shape, stride=atom_stride))

    @property
    def layout(self) -> cute.Layout:
        return self._layout


@dsl_user_op
def make_smem_layout_sfa(
    tiled_mma,
    mma_tiler_mnk,
    sf_vec_size,
    num_stages,
    *,
    loc=None,
    ip=None,
    mma_tile_inst_k=4,
):
    sfa_tile_shape = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[2],
    )
    smem_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout,
        sfa_tile_shape,
        (2, 1),
    )
    sfa_tile_shape = cute.shape_div(sfa_tile_shape, (1, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfa_tile_shape)
    smem_layout = cute.logical_divide(smem_layout, ((128, sf_vec_size),))
    return cute.append(
        smem_layout,
        cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))),
    )


@dsl_user_op
def make_smem_layout_sfb(
    tiled_mma,
    mma_tiler_mnk,
    sf_vec_size,
    num_stages,
    *,
    loc=None,
    ip=None,
    mma_tile_inst_k=4,
    atom_n=128,
):
    sfb_tile_shape = (
        cute.round_up(mma_tiler_mnk[1], atom_n),
        mma_tiler_mnk[2],
    )
    smem_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout,
        sfb_tile_shape,
        (2, 1),
    )
    sfb_tile_shape = cute.shape_div(sfb_tile_shape, (1, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfb_tile_shape)
    smem_layout = cute.logical_divide(smem_layout, ((atom_n, sf_vec_size),))
    return cute.append(
        smem_layout,
        cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))),
    )


@cute.jit
def tcgen05_after_thread_sync():
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
