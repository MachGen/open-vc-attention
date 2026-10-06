"""Fused preparation must preserve codes, scales, outputs and graph semantics."""

import os

import pytest

torch = pytest.importorskip("torch")

from open_vc_attn import attention, attention_fp8, prepare_fp8, prepare_fp8_fused  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.large,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="Fused preparation is validated on B200",
    ),
    pytest.mark.skipif(
        os.environ.get("OPEN_VC_ATTN_TEST_LARGE") != "1", reason="Set OPEN_VC_ATTN_TEST_LARGE=1"
    ),
]


@pytest.mark.parametrize(
    "length,dtype,heads",
    [
        (32768, "float16", 2),
        (32769, "bfloat16", 2),
        (73397, "bfloat16", 7),
        (73397, "bfloat16", 56),
    ],
)
def test_fused_codes_scales_outputs_and_launch_count(length, dtype, heads, monkeypatch):
    from open_vc_attn._kernels.blackwell.flash_attn.cute import v_layout

    torch.manual_seed(414)
    q, k, v = [
        torch.randn(length, heads, 128, device="cuda", dtype=getattr(torch, dtype))
        for _ in range(3)
    ]
    # Different block scales and a zero head exercise reductions and tail masks.
    q[:128] *= 16
    k[-129:] *= 0.125
    v[:, 0] = 0
    plain = prepare_fp8(q, k, v)
    cu = torch.arange(2, device="cuda", dtype=torch.int32) * length
    fused = prepare_fp8_fused(q, k, v, cu_seqlens=cu)
    for name in ("q", "k", "v"):
        assert torch.equal(
            getattr(plain, name).contiguous().view(torch.uint8),
            getattr(fused, name).contiguous().view(torch.uint8),
        ), name
    blocks = (length + 127) // 128
    assert torch.equal(plain.q_descale, fused.q_descale[..., :blocks])
    assert torch.equal(plain.k_descale, fused.k_descale)
    assert torch.equal(plain.v_descale, fused.v_descale)
    assert fused.layout["cu_seqlens_q"] is cu
    assert fused.layout["cu_seqlens_k"] is cu
    assert fused.v_prepacked and fused.v.stride(0) == 1
    expected = attention_fp8(plain)

    def unexpected_pack(_):
        raise AssertionError("Prepacked V must not be packed again")

    monkeypatch.setattr(v_layout, "pack_v", unexpected_pack)
    assert torch.equal(expected, attention_fp8(fused))
    assert torch.equal(expected, attention(q, k, v))
    assert torch.equal(expected, attention(q, k, v, preparation="fused"))
    assert torch.equal(expected, attention(q[None], k[None], v[None])[0])
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        prepare_fp8_fused(q, k, v, cu_seqlens=cu)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert len(kernels) == 3, kernels
    for name in ("_quantize_qk", "_per_head_amax_kernel", "_cast_pack_v"):
        assert sum(name in item for item in kernels) == 1, kernels


def test_fused_graph_reads_updated_inputs_and_default_opt_out():
    torch.manual_seed(909)
    q, k, v = [torch.randn(32769, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            attention(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = attention(q, k, v)
    graph.replay()
    before = output.clone()
    q.mul_(1.5)
    v.add_(0.25)
    graph.replay()
    assert not torch.equal(before, output)
    assert torch.equal(output, attention(q, k, v))
    assert torch.equal(output, attention(q, k, v, preparation="unfused"))
    assert torch.equal(
        attention(q, k, v, mid_window_blocks=None),
        attention(q, k, v, mid_window_blocks=None, preparation="unfused"),
    )


def test_prepacked_contract_rejects_incompatible_options():
    from dataclasses import replace

    q = torch.zeros(32768, 1, 128, device="cuda", dtype=torch.bfloat16)
    prepared = prepare_fp8_fused(q, q, q)
    for options in (
        {"causal": True},
        {"return_lse": True},
    ):
        with pytest.raises(ValueError, match="Prepacked V"):
            attention_fp8(prepared, **options)
    with pytest.raises(ValueError, match="v_prepacked"):
        attention_fp8(replace(prepared, v=prepared.v.contiguous()))
    with pytest.raises(ValueError, match="requires version"):
        attention_fp8(prepared, version="reference")
    for options in ({"causal": True}, {"return_lse": True}):
        with pytest.raises(ValueError, match="Fused preparation requires"):
            attention(q, q, q, preparation="fused", **options)
    with pytest.raises(ValueError, match="cu_seqlens"):
        prepare_fp8_fused(q, q, q, cu_seqlens=torch.zeros(3, device="cuda", dtype=torch.int32))
