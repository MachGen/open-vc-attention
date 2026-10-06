# Copyright (c) 2025, Tri Dao.

import math
import hashlib
import inspect
import re
from typing import Type, Callable, Optional, Tuple, overload

import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cute.runtime import from_dlpack


import quack.activation

_MIXER_ATTRS = ("__vec_size__",)


def _compute_base_hash(func: Callable) -> str:
    """Compute hash from source code or bytecode and closure values."""
    try:
        data = inspect.getsource(func).encode()
    except (OSError, TypeError):
        if hasattr(func, "__code__") and func.__code__ is not None:
            data = func.__code__.co_code
        else:
            data = repr(func).encode()

    hasher = hashlib.sha256(data)

    if hasattr(func, "__closure__") and func.__closure__ is not None:
        for cell in func.__closure__:
            hasher.update(repr(cell.cell_contents).encode())

    return hasher.hexdigest()


def hash_callable(
    func: Callable, mixer_attrs: Tuple[str] = _MIXER_ATTRS, set_cute_hash: bool = True
) -> str:
    """Hash a callable based on the source code or bytecode and closure values.
    Fast-path: if the callable (or its __wrapped__ base) has a ``__cute_hash__``
    attribute, that value is returned immediately as the base hash, then
    metadata dunders are mixed in to produce the final dict-key hash.
    set_cute_hash: whether or not to set func.__cute_hash__
    """
    # Resolve base hash
    if hasattr(func, "__cute_hash__"):
        base_hash = func.__cute_hash__
    else:
        # Unwrap decorated functions (e.g., cute.jit wrappers).
        base_func = getattr(func, "__wrapped__", func)

        if hasattr(base_func, "__cute_hash__"):
            base_hash = base_func.__cute_hash__
        else:
            base_hash = _compute_base_hash(base_func)

            if set_cute_hash:
                base_func.__cute_hash__ = base_hash

    # Mix in mutable metadata dunders
    mixer_values = tuple(getattr(func, attr, None) for attr in mixer_attrs)

    if all(v is None for v in mixer_values):
        return base_hash

    hasher = hashlib.sha256(base_hash.encode())

    for attr, val in zip(_MIXER_ATTRS, mixer_values):
        hasher.update(f"{attr}={val!r}".encode())

    return hasher.hexdigest()


def create_softcap_scoremod(softcap_val):
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_premask_fn(
        acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        return softcap_val * cute.math.tanh(acc_S_SSA * inv_softcap, fastmath=True)

    return scoremod_premask_fn


def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility
        )
    )


def convert_from_dlpack_leading_static(
    x, leading_dim, alignment=16, static_modes=None, stride_order=None
) -> cute.Tensor:
    if stride_order is None:
        stride_order = x.dim_order()
    x_ = from_dlpack(x, assumed_align=alignment)
    for i in range(x.ndim):
        if i != leading_dim and (static_modes is None or i not in static_modes):
            x_ = x_.mark_compact_shape_dynamic(mode=i, stride_order=stride_order)
    return x_


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def mma_make_fragment_A(
    smem: cute.Tensor, thr_mma: cute.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_B(smem, thr_mma)
    else:
        return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
    smem: cute.Tensor, thr_mma: cute.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_A(smem, thr_mma)
    else:
        return thr_mma.make_fragment_B(thr_mma.partition_B(smem))


def get_smem_store_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def parse_swizzle_from_pointer(ptr: cute.Pointer) -> cute.Swizzle:
    """Extract swizzle parameters from a pointer's swizzle_type.

    The swizzle_type string has the form '!cute.swizzle<"S<b,m,s>">' where
    b, m, s are the swizzle parameters (bits, base, shift).

    Returns:
        A cute.Swizzle object constructed from the extracted parameters

    Raises:
        ValueError: If the swizzle_type string cannot be parsed
    """
    # Ideally there should be a better API to get swizzle parameters, but we'll just parse
    # the string here.
    swizzle_str = str(ptr.type.swizzle_type)
    # Extract the inner part "S<b,m,s>"
    match = re.search(r"S<(\d+),(\d+),(\d+)>", swizzle_str)
    if match:
        b, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return cute.make_swizzle(b, m, s)
    else:
        raise ValueError(f"Could not parse swizzle_type: {swizzle_str}")


@dsl_user_op
def fmax(
    a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None
) -> Float32:
    return Float32(
        nvvm.fmax(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        # if const_expr(init_val is None):
        #     init_val = -cutlass.Float32.if
        # return x.reduce(cute.ReductionOp.MAX, init_val, 0)
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        # local_max = [res[0], res[1]]
        # for i in cutlass.range_constexpr(2, cute.size(x.shape), 2):
        #     local_max[0] = fmax(local_max[0], res[i + 0])
        #     local_max[1] = fmax(local_max[1], res[i + 1])
        # local_max[0] = fmax(local_max[0], local_max[1])
        # return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
    else:
        # [2025-06-15] x.reduce only seems to use 50% 3-input max and 50% 2-input max
        # We instead force the 3-input max.
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_max_0 = (
            fmax(init_val, res[0], res[1])
            if const_expr(init_val is not None)
            else fmax(res[0], res[1])
        )
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])


@cute.jit
def imax_reduce(x: cute.TensorSSA, init_val: cutlass.Int32 | None = None) -> cutlass.Int32:
    """Int32 analog of fmax_reduce: max-reduces without I2F on the tile.

    Uses 3-input cutlass.max so the compiler can emit IMAX3.U32.MAX (≈1 op
    for 3 operands, halving the instruction count of the reduction chain).
    Requires x.size() % 8 == 0; falls back to 2-input pairs otherwise.
    """
    res = cute.make_rmem_tensor(x.shape, cutlass.Int32)
    res.store(x)
    if const_expr(cute.size(x.shape) % 8 == 0):
        # Same 4-lane structure as fmax_reduce's arch>=100 branch, but with
        # 3-input max folding two elements per lane per step.
        local_max_0 = (
            cutlass.max(init_val, res[0], res[1])
            if const_expr(init_val is not None)
            else cutlass.max(res[0], res[1])
        )
        local_max = [
            local_max_0,
            cutlass.max(res[2], res[3]),
            cutlass.max(res[4], res[5]),
            cutlass.max(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = cutlass.max(local_max[0], res[i], res[i + 1])
            local_max[1] = cutlass.max(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = cutlass.max(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = cutlass.max(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = cutlass.max(local_max[0], local_max[1])
        return cutlass.max(local_max[0], local_max[2], local_max[3])
    local_max = [res[0], res[1], res[2], res[3]]
    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        local_max[0] = cutlass.max(local_max[0], res[i + 0])
        local_max[1] = cutlass.max(local_max[1], res[i + 1])
        local_max[2] = cutlass.max(local_max[2], res[i + 2])
        local_max[3] = cutlass.max(local_max[3], res[i + 3])
    local_max[0] = cutlass.max(local_max[0], local_max[1])
    local_max[2] = cutlass.max(local_max[2], local_max[3])
    local_max[0] = cutlass.max(local_max[0], local_max[2])
    if const_expr(init_val is None):
        return local_max[0]
    return cutlass.max(local_max[0], init_val)


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
        # res = cute.make_fragment(x.shape, Float32)
        # res.store(x)
        # local_sum = [res[0], res[1], res[2], res[3]]
        # for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        #     local_sum[0] += res[i + 0]
        #     local_sum[1] += res[i + 1]
        #     local_sum[2] += res[i + 2]
        #     local_sum[3] += res[i + 3]
        # local_sum[0] += local_sum[1]
        # local_sum[2] += local_sum[3]
        # local_sum[0] += local_sum[2]
        # return local_sum[0] if const_expr(init_val is None) else local_sum[0] + init_val
    else:
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_sum_0 = (
            cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            # cute.arch.add_packed_f32x2((init_val / 2, init_val / 2), (res[0], res[1]))
            if const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], (res[i + 0], res[i + 1]))
            local_sum[1] = cute.arch.add_packed_f32x2(local_sum[1], (res[i + 2], res[i + 3]))
            local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], (res[i + 4], res[i + 5]))
            local_sum[3] = cute.arch.add_packed_f32x2(local_sum[3], (res[i + 6], res[i + 7]))
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]


@dsl_user_op
def atomic_add_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Atomic add on a gmem Int32 location."""
    llvm.inline_asm(
        None,
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value(), Int32(a).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def float_to_ordered_i32_bits(a: float | Float32, *, loc=None, ip=None) -> Int32:
    """Bitcast non-negative f32 to i32 for ordered atomic max."""
    return Int32(llvm.bitcast(T.i32(), Float32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def atomic_max_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Atomic max on a gmem Int32 location."""
    nvvm.atomicrmw(
        res=T.i32(), op=nvvm.AtomicOpKind.MAX, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
    )


@dsl_user_op
def atomic_max_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Atomic max on a gmem Int32-backed Float32 location via i32 bitcast."""
    atomic_max_i32(float_to_ordered_i32_bits(a, loc=loc, ip=ip), gmem_ptr, loc=loc, ip=ip)


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    # gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    # # cache_hint = cutlass.Int64(0x12F0000000000000)
    # llvm.inline_asm(
    #     None,
    #     [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip)],
    #     # [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip), cache_hint.ir_value()],
    #     "red.global.add.f32 [$0], $1;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, 0x12F0000000000000;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, $2;",
    #     "l,f",
    #     # "l,f,l",
    #     has_side_effects=True,
    #     is_align_stack=False,
    #     asm_dialect=llvm.AsmDialect.AD_ATT,
    # )
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value()
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if const_expr(sync):
        warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx


# @dsl_user_op
# def warp_vote_any_lt(a: float | Float32, b: float | Float32, *, loc=None, ip=None) -> cutlass.Boolean:
#     mask = cutlass.Int32(-1)
#     return cutlass.Boolean(
#         llvm.inline_asm(
#             T.i32(),
#             [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip), mask.ir_value(loc=loc, ip=ip)],
#             ".pred p1, p2;\n"
#             "setp.lt.f32 p1, $1, $2;\n"
#             "vote.sync.any.pred p2, p1, $3;\n"
#             "selp.u32 $0, 1, 0, p2;",
#             # "selp.u32 $0, 1, 0, p1;",
#             "=r,f,f,r",
#             has_side_effects=False,
#             is_align_stack=False,
#             asm_dialect=llvm.AsmDialect.AD_ATT,
#         )
#     )


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    # important: need stride 1 and not 0 for recast_tensor to work
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@dsl_user_op
def shr_u32(val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.s32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def warp_prefix_sum(val: cutlass.Int32, lane: Optional[cutlass.Int32] = None) -> cutlass.Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, val = %d", cute.arch.thread_idx()[0] % 32, val)
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
        # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, partial_sum = %d, val = %d", cute.arch.thread_idx()[0] % 32, partial_sum, val)
    return val


@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None
) -> cutlass.Int32:
    assert to_dtype in [cutlass.BFloat16, cutlass.Float16], "to_dtype must be BFloat16 or Float16"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@overload
def cvt_f16(src: cute.Tensor, dst: cute.Tensor) -> None: ...


@overload
def cvt_f16(src: cute.Tensor, dtype: Type[cute.Numeric]) -> cute.Tensor: ...


@cute.jit
def cvt_f16(src: cute.Tensor, dst_or_dtype):
    """Convert Float32 tensor to Float16/BFloat16.

    Args:
        src: Source tensor with Float32 element type
        dst_or_dtype: Either a destination tensor or a dtype (Float16/BFloat16)

    Returns:
        None if dst is a tensor, or a new tensor if dtype is provided
    """
    if const_expr(isinstance(dst_or_dtype, type)):
        # dtype variant: create new tensor and call the tensor variant
        dtype = dst_or_dtype
        dst = cute.make_rmem_tensor(src.shape, dtype)
        cvt_f16(src, dst)
        return dst
    else:
        # tensor variant: write to dst
        dst = dst_or_dtype
        assert cute.size(dst.shape) == cute.size(src.shape), "dst and src must have the same size"
        assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
        assert dst.element_type in [cutlass.BFloat16, cutlass.Float16], (
            "dst must be BFloat16 or Float16"
        )
        assert src.element_type is Float32, "src must be Float32"
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
        for i in cutlass.range_constexpr(cute.size(dst_i32)):
            dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


@dsl_user_op
@cute.jit
def evaluate_polynomial(x: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None) -> Float32:
    deg = len(poly) - 1
    out = poly[deg]
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = out * x + poly[i]
    return out


@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: Float32, y: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = cute.arch.fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


@dsl_user_op
def add_round_down(x: float | Float32, y: float | Float32, *, loc=None, ip=None) -> Float32:
    # There's probably a way to call llvm or nvvm to do this instead of ptx
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Float32(y).ir_value(loc=loc, ip=ip)],
            "add.rm.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def combine_int_frac_ex2(x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None) -> Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            # add.u32 generates IMAD instruction and add.s32 generates LEA instruction
            # IMAD uses the FMA pipeline and LEA uses the ALU pipeline, afaik
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
            "mov.b32 $0, out_i;\n\t"
            "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_emulation(x: Float32, *, loc=None, ip=None) -> Float32:
    # We assume x <= 127.0
    poly_ex2_deg3 = (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    )
    fp32_round_int = float(2**23 + 2**22)
    x_clamped = cute.arch.fmax(x, -127.0)
    # We want to round down here, so that the fractional part is in [0, 1)
    x_rounded = add_round_down(x_clamped, fp32_round_int, loc=loc, ip=ip)
    # The integer floor of x is now in the last 8 bits of x_rounded
    # We assume the next 2 ops round to nearest even. The rounding mode is important.
    x_rounded_back = x_rounded - fp32_round_int
    x_frac = x_clamped - x_rounded_back
    x_frac_ex2 = evaluate_polynomial(x_frac, poly_ex2_deg3, loc=loc, ip=ip)
    return combine_int_frac_ex2(x_rounded, x_frac_ex2, loc=loc, ip=ip)


@dsl_user_op
def pack4_e4m3(f0: Float32, f1: Float32, f2: Float32, f3: Float32, *, loc=None, ip=None) -> Float32:
    """Pack 4 f32 -> one 32-bit word of 4 E4M3 (bytes [f0,f1,f2,f3]), bits returned as f32.

    Explicit `mov.b32 {lo,hi}` hands ptxas a register that is already the packed
    store word, steering it toward F2FP...MERGE_C instead of cvt.e4m3x2 ->.b16
    pairs that ptxas then re-assembles with a separate PRMT.
    """
    out = llvm.inline_asm(
        T.f32(),
        [
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
            Float32(f2).ir_value(loc=loc, ip=ip),
            Float32(f3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t"
        "}\n",
        "=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


@dsl_user_op
def fma_sat_f32(x: Float32, scale: Float32, bias: Float32, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Float32(f).ir_value(loc=loc, ip=ip) for f in (x, scale, bias)],
        "fma.rn.sat.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


@dsl_user_op
def expcast_bias_round_down(
    row_max: Float32, code_scale: Float32, *, subnormal: bool = False, loc=None, ip=None
) -> Float32:
    """Bound the affine peak despite cancellation in the bias."""
    # 119.65 * 2^-24 for subnormal codes, or 1024 + 119.65 for biased codes.
    peak = "0f36EF4CCD" if subnormal else "0f448EF4CD"
    out = llvm.inline_asm(
        T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in (-row_max, code_scale)],
        f"fma.rm.f32 $0, $1, $2, {peak};",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


@dsl_user_op
def pack4_expcast_half_codes(
    f0: Float32, f1: Float32, f2: Float32, f3: Float32, *, loc=None, ip=None
) -> Float32:
    """Round biased codes in FP16, clamp below 1024, and pack their low bytes."""
    out = llvm.inline_asm(
        T.f32(),
        [Float32(f).ir_value(loc=loc, ip=ip) for f in (f0, f1, f2, f3)],
        "{\n.reg .b32 lo, hi, lower;\n"
        "mov.b32 lower, 0x64006400;\n"
        "cvt.rn.f16x2.f32 lo, $2, $1;\n"
        "cvt.rn.f16x2.f32 hi, $4, $3;\n"
        "max.f16x2 lo, lo, lower;\n"
        "max.f16x2 hi, hi, lower;\n"
        "prmt.b32 $0, lo, hi, 0x6420;\n}\n",
        "=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


@dsl_user_op
def pack_expcast_tile(
    scale: Float32,
    bias: Float32,
    skip: Int32,
    *scores: Float32,
    subnormal: bool = False,
    loc=None,
    ip=None,
) -> Tuple[Float32, ...]:
    """Encode 32 to 128 scores; skip must be uniform across the calling warp."""
    assert len(scores) in (32, 64, 96, 128)
    words = len(scores) // 4
    scale_arg = words + len(scores)
    bias_arg, skip_arg = scale_arg + 1, scale_arg + 2
    # FP16 subnormal bits encode round(code) when the affine uses units of 2^-24.
    # RELU supplies the zero lower bound without separate packed max instructions.
    convert = "cvt.rn.relu.f16x2.f32" if subnormal else "cvt.rn.f16x2.f32"
    # Keep the entire conversion behind one uniform branch. Splitting it into
    # small asm calls can make ptxas predicate hundreds of instructions on skips.
    assembly = (
        "{\n.reg .b64 xp, sc, bc, result;\n.reg .f32 f0,f1,f2,f3;\n"
        ".reg .b32 lo,hi,lower;\n.reg .pred p;\n"
        f"setp.ne.u32 p,${skip_arg},0;\n@p bra.uni ZERO;\n"
        f"mov.b64 sc,{{${scale_arg},${scale_arg}}};\n"
        f"mov.b64 bc,{{${bias_arg},${bias_arg}}};\n"
        "mov.b32 lower,0x64006400;\n"
    )
    for i in range(words):
        a = words + 4 * i
        assembly += (
            f"mov.b64 xp,{{${a},${a + 1}}};\nfma.rn.f32x2 result,xp,sc,bc;\n"
            "mov.b64 {f0,f1},result;\n"
            f"mov.b64 xp,{{${a + 2},${a + 3}}};\nfma.rn.f32x2 result,xp,sc,bc;\n"
            "mov.b64 {f2,f3},result;\n"
            f"{convert} lo,f1,f0;\n{convert} hi,f3,f2;\n"
        )
        if not subnormal:
            assembly += "max.f16x2 lo,lo,lower;\nmax.f16x2 hi,hi,lower;\n"
        assembly += f"prmt.b32 ${i},lo,hi,0x6420;\n"
    assembly += "bra.uni DONE;\nZERO:\n"
    assembly += "".join(f"mov.f32 ${i},0f00000000;\n" for i in range(words))
    assembly += "DONE:\n}\n"
    operands = [Float32(x).ir_value(loc=loc, ip=ip) for x in (*scores, scale, bias)]
    operands.append(Int32(skip).ir_value(loc=loc, ip=ip))
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * words),
        operands,
        assembly,
        ",".join(["=f"] * words + ["f"] * (len(scores) + 2) + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(words)
    )


@dsl_user_op
def pack4_expcast_e4m3(
    f0: Float32, f1: Float32, f2: Float32, f3: Float32, *, loc=None, ip=None
) -> Float32:
    """Round four clamped code/120 values and pack their E4M3 bytes."""
    # The FMA rounds codes into the low mantissa bits without an integer conversion.
    f0, f1 = cute.arch.fma_packed_f32x2((f0, f1), (120.0, 120.0), (8388608.0, 8388608.0))
    f2, f3 = cute.arch.fma_packed_f32x2((f2, f3), (120.0, 120.0), (8388608.0, 8388608.0))
    out = llvm.inline_asm(
        T.f32(),
        [Float32(f).ir_value(loc=loc, ip=ip) for f in (f0, f1, f2, f3)],
        "{\n"
        ".reg .b32 c<4>, lo, hi;\n"
        + "".join(f"mov.b32 c{i}, ${i + 1};\n" for i in range(4))
        + "prmt.b32 lo, c0, c1, 0x4440;\n"
        "prmt.b32 hi, c2, c3, 0x4440;\n"
        "prmt.b32 $0, lo, hi, 0x5410;\n"
        "}\n",
        "=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(out)


@dsl_user_op
def add16_e4m3_to_f32(
    p0: Float32,
    p1: Float32,
    p2: Float32,
    p3: Float32,
    a0: Float32,
    a1: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """Sum two groups of eight E4M3 values in FP16, then accumulate in FP32."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Float32(f).ir_value(loc=loc, ip=ip) for f in (p0, p1, p2, p3, a0, a1)],
        "{\n.reg .b16 h<8>;\n.reg .b32 x<8>;\n"
        + "".join(
            f"mov.b32 {{h{2 * i}, h{2 * i + 1}}}, ${i + 2};\n"
            f"cvt.rn.f16x2.e4m3x2 x{2 * i}, h{2 * i};\n"
            f"cvt.rn.f16x2.e4m3x2 x{2 * i + 1}, h{2 * i + 1};\n"
            for i in range(4)
        )
        + "add.rn.f16x2 x0, x0, x1;\nadd.rn.f16x2 x2, x2, x3;\n"
        "add.rn.f16x2 x4, x4, x5;\nadd.rn.f16x2 x6, x6, x7;\n"
        "add.rn.f16x2 x0, x0, x2;\nadd.rn.f16x2 x4, x4, x6;\n"
        "add.rn.f16x2 x0, x0, x4;\nmov.b32 {h0, h1}, x0;\n"
        "add.f32.f16 $0, h0, $6;\nadd.f32.f16 $1, h1, $7;\n}\n",
        "=f,=f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(2))


# TODO: check that the ex2_emulation_2 produces the same SASS as the ptx version
@dsl_user_op
def ex2_emulation_2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    # We assume x <= 127.0 and y <= 127.0
    poly_ex2_deg3 = (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    )
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm")
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even. The rounding mode is important.
    xy_rounded_back = quack.activation.sub_packed_f32x2(
        xy_rounded, (fp32_round_int, fp32_round_int)
    )
    xy_frac = quack.activation.sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, poly_ex2_deg3, loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def e2e_asm2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    out_f32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip), Float32(y, loc=loc, ip=ip).ir_value()],
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, $2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, $3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 $0, r7;\n\t"
        "mov.b32 $1, r8;\n\t"
        "}\n",
        "=r,=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [0], loc=loc, ip=ip))
    out1 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [1], loc=loc, ip=ip))
    return out0, out1


@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    # We assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    """Convert a scalar to a cute TensorSSA of shape (1,) and given dtype"""
    vec = cute.make_rmem_tensor(1, dtype)
    vec[0] = a
    return vec.load()


def ssa_to_scalar(val):
    """Could inline but nice for reflecting the above api"""
    return val[0]


@dsl_user_op
def expcast_tree_max(*scores: Float32, loc=None, ip=None) -> Float32:
    """Reduce 128 scores, optionally including the running max, in five levels."""
    assert len(scores) in (128, 129)

    def reduce(values: Tuple[Float32, ...]) -> Float32:
        size = len(values)
        if size == 1:
            return values[0]
        if size <= 3:
            return fmax(*values, loc=loc, ip=ip)
        first = 1
        while first * 3 < size:
            first *= 3
        second = 1
        while second * 3 < size - first:
            second *= 3
        return fmax(
            reduce(values[:first]),
            reduce(values[first : first + second]),
            reduce(values[first + second :]),
            loc=loc,
            ip=ip,
        )

    return reduce(scores)
