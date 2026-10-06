"""V residual repair integration: denominator, accuracy and replay contracts."""

import os

import pytest

torch = pytest.importorskip("torch")

from open_vc_attn import attention, attention_v_repair, prepare_v_repair  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.large,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="V repair is validated on B200",
    ),
    pytest.mark.skipif(
        os.environ.get("OPEN_VC_ATTN_TEST_LARGE") != "1", reason="Set OPEN_VC_ATTN_TEST_LARGE=1"
    ),
]


@pytest.mark.parametrize("window", [None, 4])
def test_repair_denominator_scales_and_outlier_accuracy(window):
    torch.manual_seed(33)
    q, k, v = [torch.randn(32769, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    gains = torch.linspace(0.25, 3, 257, device="cuda").repeat_interleave(128)[:32769]
    k.mul_(gains[:, None, None])
    base = attention(q, k, v, mid_window_blocks=window)
    plain = prepare_v_repair(q, k, v, budget=0.0)
    assert plain.repair_tokens == plain.selected_tokens == 0
    assert plain.v_prepacked
    assert torch.equal(attention_v_repair(plain, mid_window_blocks=window), base)
    zeroed = prepare_v_repair(q, k, v, budget=0.04)
    assert zeroed.selected_tokens == round(0.04 * 32769)
    assert zeroed.repair_tokens % 128 == 0
    zeroed.v[: zeroed.repair_tokens].zero_()
    assert torch.equal(attention_v_repair(zeroed, mid_window_blocks=window), base)

    v = v.float()
    v[torch.randperm(v.shape[0], device="cuda")[:300]] *= 40
    v = v.bfloat16()
    ref = attention(q, k, v, version="reference", mode="bf16").float()
    plain_out = attention(q, k, v, mid_window_blocks=window)
    prepared = prepare_v_repair(q, k, v, budget=0.02)
    output = attention_v_repair(prepared, mid_window_blocks=window)
    plain_error = (plain_out.float() - ref).norm() / ref.norm()
    repair_error = (output.float() - ref).norm() / ref.norm()
    assert torch.isfinite(output).all()
    assert repair_error < plain_error, (repair_error.item(), plain_error.item())
    assert torch.equal(output, attention_v_repair(prepared, mid_window_blocks=window))
    assert torch.equal(
        output,
        attention_v_repair(
            prepare_v_repair(q[None], k[None], v[None], budget=0.02), mid_window_blocks=window
        )[0],
    )


def test_repair_graph_and_budget_validation():
    torch.manual_seed(133)
    q, k, v = [torch.randn(32768, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

    def call():
        return attention_v_repair(prepare_v_repair(q, k, v, budget=0.005))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    before = output.clone()
    q.mul_(1.25)
    v.add_(0.5)
    graph.replay()
    assert not torch.equal(output, before)
    assert torch.equal(output, call())
    for budget in (-0.1, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="budget"):
            prepare_v_repair(q, k, v, budget=budget)
    with pytest.raises(ValueError, match="S >= 32768"):
        prepare_v_repair(q[:129], k[:129], v[:129], budget=0.02)
