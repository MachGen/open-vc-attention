"""Regression coverage for the VC-attn 8aa761e snapshot."""

import importlib.metadata
import os

import pytest

torch = pytest.importorskip("torch")

from vc_attn.api import attention_fp8, interface, prepare_fp8, raw_forward  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="Requires Blackwell SM100 or SM103",
    ),
]


def inputs(length, heads):
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(length, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    gain = torch.linspace(0.25, 3, (length + 127) // 128, device="cuda").repeat_interleave(128)
    q.mul_(gain[:length, None, None])
    k.mul_(gain[:length].flip(0)[:, None, None])
    return q, k, v


def require_dsl():
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.6.0":
        pytest.skip("The imported scheduling gates require DSL 4.6.0")


@pytest.mark.large
@pytest.mark.skipif(os.environ.get("VC_ATTN_TEST_LARGE") != "1", reason="Set VC_ATTN_TEST_LARGE=1")
def test_default_b200_fusedpipe_without_d_matches_benchmark(monkeypatch):
    require_dsl()
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Source-level fusedpipe is B200-specific")
    import cutlass.cute as cute

    from vc_attn import _sass_runtime
    from vc_attn._kernels.v4.flash_attn.cute import v_layout
    from vc_attn.api import attention
    from vc_attn.benchmark import GPUOwnership, make_call
    from vc_attn.registry import DEFAULT_BACKENDS

    guard = GPUOwnership(torch.cuda.current_device())
    guard.check()
    q, k, v = inputs(32769, 7)
    prepared = prepare_fp8(q, k, v)
    interface("v4")._flash_attn_fwd.compile_cache.clear()
    compile_original, pack_original = cute.compile, v_layout.pack_v
    schedules, packed = [], []

    def compile_record(kernel, *args, **kwargs):
        result = compile_original(kernel, *args, **kwargs)
        schedules.append(kernel)
        return result

    def pack_record(value):
        packed.append(tuple(value.shape))
        return pack_original(value)

    def reject_d(*args, **kwargs):
        pytest.fail("The default path attempted experimental D binary export")

    monkeypatch.setattr(cute, "compile", compile_record)
    monkeypatch.setattr(v_layout, "pack_v", pack_record)
    monkeypatch.setattr(_sass_runtime, "_export_patched", reject_d)
    output = attention(q, k, v)
    assert torch.isfinite(output).all()
    assert packed, "Low-head-count B200 input must select V packing"
    kernels = [kernel for kernel in schedules if hasattr(kernel, "mid_window_blocks")]
    assert kernels and all(kernel.mid_window_blocks == 4 for kernel in kernels)
    # Constructor options remain inspectable even on a compiler disk-cache hit.
    assert all(
        kernel.inline_rescale and kernel.expcast and kernel.q_stage == 2 for kernel in kernels
    )
    assert all(kernel.m_block_size == kernel.n_block_size == 128 for kernel in kernels)
    assert all(not kernel.is_sm103 and not kernel.use_sm103_schedule for kernel in kernels)
    for scope in ("attention", "quantize-attention"):
        call, metadata = make_call(DEFAULT_BACKENDS[-1], q, k, v, prepared, scope=scope)
        assert torch.equal(output, call())
        assert metadata == {"mid_window_blocks": 4, "sass_d_enabled": False}
    assert torch.equal(output, attention_fp8(prepared, mid_window_blocks=4))
    assert all(
        not hasattr(compiled, "sass_d_metadata")
        for compiled in interface("v4")._flash_attn_fwd.compile_cache.values()
    )
    torch.cuda.synchronize()
    guard.check()


@pytest.mark.large
@pytest.mark.skipif(os.environ.get("VC_ATTN_TEST_LARGE") != "1", reason="Set VC_ATTN_TEST_LARGE=1")
def test_unscaled_original_scan_packs_v_and_matches_lse_fallback(monkeypatch):
    require_dsl()
    from vc_attn._kernels.v4.flash_attn.cute import v_layout

    q, k, v = [x.to(torch.float8_e4m3fn) for x in inputs(32769, 32)]
    cu = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    kwargs = dict(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        expcast=True,
        mid_window_blocks=None,
        version="v4",
    )
    packed = []
    original = v_layout.pack_v

    def record_pack(value):
        packed.append(value.shape)
        return original(value)

    monkeypatch.setattr(v_layout, "pack_v", record_pack)
    output, _ = raw_forward(q, k, v, **kwargs)
    assert len(packed) == 1
    reference, lse = raw_forward(q, k, v, **kwargs, return_lse=True)
    assert len(packed) == 1, "LSE must retain the non-packed fallback"
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()
    assert (output.float() - reference.float()).norm() / reference.float().norm() < 0.003


@pytest.mark.large
@pytest.mark.skipif(os.environ.get("VC_ATTN_TEST_LARGE") != "1", reason="Set VC_ATTN_TEST_LARGE=1")
@pytest.mark.parametrize("mid", [None, 4])
def test_scaled_expcast_preserves_previous_snapshot(mid):
    require_dsl()
    prepared = prepare_fp8(*inputs(32769, 32))
    previous = attention_fp8(prepared, version="scaled", mid_window_blocks=mid)
    current = attention_fp8(prepared, version="v4", mid_window_blocks=mid)
    assert torch.isfinite(current).all()
    assert torch.equal(current, previous)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_sm103_ordinary_fp8_matches_previous_snapshot(scaled, return_lse):
    require_dsl()
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Ordinary-FP8 scheduling changes are SM103-only")
    prepared = prepare_fp8(*inputs(257, 4))
    tensors = (prepared.q, prepared.k, prepared.v)
    kwargs = dict(prepared.layout, expcast=False, return_lse=return_lse)
    if scaled:
        kwargs.update(
            q_descale=prepared.q_descale, k_descale=prepared.k_descale, v_descale=prepared.v_descale
        )
    outputs = [raw_forward(*tensors, version=version, **kwargs) for version in ("scaled", "v4")]
    previous, current = [out[0] for out in outputs]
    assert torch.isfinite(current).all()
    torch.testing.assert_close(current, previous, atol=0.002, rtol=0.002)
    if return_lse:
        torch.testing.assert_close(outputs[1][1], outputs[0][1], atol=0.0001, rtol=0.0001)
    # Both snapshots must retain separate caches and module identities.
    assert interface("scaled") is not interface("v4")
