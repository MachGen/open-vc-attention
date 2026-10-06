"""Operator checks. Approximation tolerances are not model-quality acceptance gates."""

import os

import pytest

torch = pytest.importorskip("torch")

from vc_attn.api import attention, attention_fp8, prepare_fp8  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="Requires an exclusively allocated Blackwell GPU",
    ),
]


def oracle(q, k, v, scale=None, causal=False):
    qh, kh, vh = [x.float().transpose(0, 1) for x in (q, k, v)]
    score = (qh @ kh.transpose(-1, -2)) * (scale or q.shape[-1] ** -0.5)
    if causal:
        mask = torch.ones(q.shape[0], k.shape[0], device=q.device, dtype=torch.bool).triu(1)
        score.masked_fill_(mask, -float("inf"))
    return (score.softmax(-1) @ vh).transpose(0, 1)


@pytest.mark.parametrize("s", [1, 129, 1024])
def test_versions_against_fp32_and_repeatability(s):
    torch.manual_seed(700 + s)
    q, k, v = [torch.randn(s, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    ref = oracle(q, k, v)
    prepared = prepare_fp8(q, k, v)
    outputs = {}
    for version in ("baseline", "v1", "v2", "v3", "scaled", "v4"):

        def call():
            return attention_fp8(prepared, version=version, expcast=version != "baseline")

        out = call().clone()
        assert torch.isfinite(out).all()
        assert torch.equal(out, call())
        error = (out.float() - ref).norm() / ref.norm().clamp_min(1e-12)
        assert error < 0.10, (version, s, error.item())
        outputs[version] = out
    assert torch.equal(outputs["v1"], outputs["v2"])
    assert torch.equal(outputs["v2"], outputs["v3"])
    bf16 = attention(q, k, v, version="baseline", mode="bf16")
    assert (bf16.float() - ref).norm() / ref.norm().clamp_min(1e-12) < 0.01


def test_quantization_contract_and_batched_isolation():
    torch.manual_seed(919)
    x = torch.randn(2, 129, 2, 128, device="cuda", dtype=torch.bfloat16)
    x[0] = 0
    p = prepare_fp8(x, x, x)
    assert p.q_descale.shape == (2, 2, 2)
    assert p.v_descale.shape == (2, 2)
    assert torch.isfinite(p.q.float()).all() and (p.q_descale > 0).all()
    out = attention_fp8(p)
    assert torch.count_nonzero(out[0]) == 0
    single = attention(x[1], x[1], x[1])
    assert torch.allclose(out[1].float(), single.float(), atol=0.01, rtol=0.01)


def test_cross_attention_scale_and_validation():
    torch.manual_seed(441)
    q = torch.randn(31, 2, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(129, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    ref = oracle(q, k, v, scale=0.2)
    out = attention(q, k, v, mode="bf16", softmax_scale=0.2)
    assert (out.float() - ref).norm() / ref.norm() < 0.01
    with pytest.raises(ValueError):
        attention(q, k, v, mode="unknown")
    with pytest.raises(ValueError):
        attention(q.requires_grad_(), k, v)


def test_diffusion_adapter_packed_sequences_and_padding(monkeypatch):
    from vc_attn.integrations.diffusion import DenseDiffusionAdapter

    monkeypatch.setenv("VC_ATTN_VERSION", "scaled")
    monkeypatch.setenv("VC_ATTN_MODE", "expcast")
    impl = DenseDiffusionAdapter(2, 128, 128**-0.5)
    torch.manual_seed(513)
    q, k, v = [torch.randn(258, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    bounds = (0, 129, 258)
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    out = impl.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=129, cu_seqlens_host=bounds)
    for a, b in zip(bounds, bounds[1:]):
        assert torch.equal(out[a:b], impl.forward(q[a:b], k[a:b], v[a:b]))
    padded = DenseDiffusionAdapter(2, 128, 128**-0.5, packed_trailing_padding=True)
    out = padded.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=129, cu_seqlens_host=bounds)
    assert torch.count_nonzero(out[129:]) == 0
    assert torch.equal(out[:129], impl.forward(q[:129], k[:129], v[:129]))
    with pytest.raises(ValueError):
        impl.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=129)


def test_graph_replay_uses_current_activations():
    torch.manual_seed(319)
    q, k, v = [torch.randn(129, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            attention(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = attention(q, k, v)
    graph.replay()
    assert torch.equal(out, attention(q, k, v))
    q.mul_(1.5)
    v.add_(0.25)
    graph.replay()
    assert torch.equal(out, attention(q, k, v))


@pytest.mark.large
@pytest.mark.skipif(os.environ.get("VC_ATTN_TEST_LARGE") != "1", reason="Set VC_ATTN_TEST_LARGE=1")
def test_large_scaled_fastpath():
    torch.manual_seed(91)
    q, k, v = [torch.randn(32768, 32, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    p = prepare_fp8(q, k, v)
    old = attention_fp8(p, version="v3")
    new = attention_fp8(p, version="scaled")
    assert torch.isfinite(new).all()
    assert (new.float() - old.float()).norm() / old.float().norm() < 0.01


@pytest.mark.parametrize("s", [1, 129, 1025])
@pytest.mark.skipif(not os.environ.get("VC_ATTN_NATIVE_LIBRARY"), reason="Native DSO is opt-in")
def test_native_tail_and_scale_layout(s):
    from vc_attn.native_plan import NativePlan

    torch.manual_seed(210 + s)
    q, k, v = [torch.randn(s, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    # Deliberately vary scales across blocks; the last block can be partial.
    factors = torch.linspace(0.25, 2, (s + 127) // 128, device="cuda").repeat_interleave(128)[:s]
    q *= factors[:, None, None]
    k *= factors.flip(0)[:, None, None]
    ref = oracle(q, k, v)
    plan = NativePlan(
        prepare_fp8(q, k, v),
        os.environ["VC_ATTN_NATIVE_LIBRARY"],
    )
    try:
        out = plan().clone()
        assert torch.isfinite(out).all()
        assert torch.equal(out, plan())
        assert (out.float() - ref).norm() / ref.norm().clamp_min(1e-12) < 0.10
    finally:
        plan.close()
