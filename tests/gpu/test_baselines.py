"""VC baseline device handling: preparation must launch on the inputs' device."""

import pytest

torch = pytest.importorskip("torch")

from open_vc_attn.baselines import attention_vc, prepare_vc  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Requires two CUDA devices",
    ),
    pytest.mark.skipif(
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and torch.cuda.get_device_capability(1) not in ((10, 0), (10, 3)),
        reason="Requires Blackwell SM100 or SM103 on device 1",
    ),
]


def _vc(q, k, v, **kwargs):
    out = attention_vc(prepare_vc(q, k, v, **kwargs), output_shape=q.shape)
    torch.cuda.synchronize(q.device)
    return out


def _relative_error(out, ref):
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def test_vc_preparation_runs_on_input_device_not_current_device():
    torch.manual_seed(7)
    q, k, v = [torch.randn(129, 2, 128, device="cuda:1", dtype=torch.bfloat16) for _ in range(3)]
    with torch.cuda.device(1):
        grouping = prepare_vc(q, k, v).permutation
        fresh_ref = _vc(q, k, v)
        reused_ref = _vc(q, k, v, permutation=grouping)
        unchecked_ref = _vc(q, k, v, permutation=grouping, check_permutation=False)

    torch.cuda.set_device(0)
    fresh = _vc(q, k, v)
    reused = _vc(q, k, v, permutation=grouping)
    unchecked = _vc(q, k, v, permutation=grouping, check_permutation=False)
    assert torch.cuda.current_device() == 0

    for out in (fresh, reused, unchecked):
        assert out.device == q.device and bool(torch.isfinite(out).all())
    # A reused grouping is deterministic, so it must match the device-1 context exactly.
    assert torch.equal(reused, reused_ref)
    assert torch.equal(unchecked, unchecked_ref)
    # Fresh k-means accumulates centroids with float atomics; groupings can differ slightly.
    assert _relative_error(fresh, fresh_ref) < 0.05
