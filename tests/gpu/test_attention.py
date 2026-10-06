"""Operator checks. Approximation tolerances are not model-quality acceptance gates."""

import pytest

torch = pytest.importorskip("torch")

from open_vc_attn.api import attention, attention_fp8, prepare_fp8  # noqa: E402

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
def test_open_vc_against_fp32_and_repeatability(s):
    torch.manual_seed(700 + s)
    q, k, v = [torch.randn(s, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    ref = oracle(q, k, v)
    prepared = prepare_fp8(q, k, v)
    out = attention_fp8(prepared).clone()
    assert torch.isfinite(out).all()
    assert torch.equal(out, attention_fp8(prepared))
    error = (out.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert error < 0.10, (s, error.item())
    bf16 = attention(q, k, v, version="reference", mode="bf16")
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
    out = attention(q, k, v, version="reference", mode="bf16", softmax_scale=0.2)
    assert (out.float() - ref).norm() / ref.norm() < 0.01
    with pytest.raises(ValueError):
        attention(q, k, v, mode="unknown")
    with pytest.raises(ValueError):
        attention(q.requires_grad_(), k, v)


def test_diffusion_adapter_packed_sequences_and_padding(monkeypatch):
    from open_vc_attn.integrations.diffusion import DenseDiffusionAdapter

    monkeypatch.setenv("OPEN_VC_ATTN_IMPLEMENTATION", "open-vc")
    monkeypatch.setenv("OPEN_VC_ATTN_MODE", "expcast")
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


@pytest.mark.parametrize("s", [129, 1024])
def test_vc_baseline_against_fp32(s):
    from open_vc_attn.baselines import attention_vc, prepare_vc, vc_attention

    torch.manual_seed(900 + s)
    q, k, v = [torch.randn(s, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    ref = oracle(q, k, v)
    out = vc_attention(q, k, v)
    assert out.shape == q.shape and torch.isfinite(out).all()
    error = (out.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert error < 0.10, (s, error.item())
    grouped = prepare_vc(q, k, v)
    reused = prepare_vc(q, k, v, permutation=grouped.permutation, check_permutation=False)
    assert torch.equal(
        attention_vc(grouped, output_shape=q.shape), attention_vc(reused, output_shape=q.shape)
    )


def test_vc_complete_call_is_graph_capturable():
    from open_vc_attn.baselines import attention_vc, prepare_vc

    torch.manual_seed(77)
    q, k, v = [torch.randn(1024, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    grouped = prepare_vc(q, k, v)

    def call():
        prepared = prepare_vc(q, k, v, permutation=grouped.permutation, check_permutation=False)
        return attention_vc(prepared, output_shape=q.shape)

    eager = call().clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_vc_tuned_v_smooth_path_matches_untuned_path():
    """Long sequences take the tuned V-Smooth path (FP32 means, mean prefetch, original scan)."""
    from open_vc_attn._kernels.blackwell.v_smooth import apply_v_smooth, prepare_v_smooth
    from open_vc_attn.api import _layout, raw_forward
    from open_vc_attn.baselines import attention_vc, prepare_vc

    torch.manual_seed(1201)
    q, k, v = [torch.randn(8192, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    tuned = prepare_vc(q, k, v)
    assert tuned.means.dtype == torch.float32
    untuned = prepare_v_smooth(q, k, v, permutation=tuned.permutation)  # BF16 means
    _, _, _, layout, _ = _layout(untuned.q, untuned.k, untuned.v)
    slow = raw_forward(
        untuned.q,
        untuned.k,
        untuned.v,
        **layout,
        expcast=True,
        mid_window_blocks=None,
        return_lse=False,
        **untuned.forward_kwargs(),
    )[0]
    fast = attention_vc(tuned, output_shape=q.shape)
    ref = oracle(q, k, v)
    for out in (fast, slow):
        assert torch.isfinite(out).all()
        assert (out.float() - ref).norm() / ref.norm() < 0.10
    assert (fast.float() - slow.float()).norm() / slow.float().norm() < 0.01
    # The BF16-means preparation is the PyTorch implementation; its K/V codes are the reference.
    fused = apply_v_smooth(q, k, v, tuned.permutation)
    assert torch.equal(fused.k.view(torch.uint8), untuned.k.view(torch.uint8))
    differing = (fused.v.view(torch.uint8) != untuned.v.view(torch.uint8)).float().mean()
    assert differing < 1e-5, differing
