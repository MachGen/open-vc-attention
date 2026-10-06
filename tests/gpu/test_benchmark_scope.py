"""Attention-scope benchmark calls launch exactly one kernel, with V packed beforehand."""

import os

import pytest

torch = pytest.importorskip("torch")

from open_vc_attn.api import attention_fp8, prepare_fp8  # noqa: E402
from open_vc_attn.benchmarking.runner import _prepack_v, make_call  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.large,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="Requires Blackwell SM100 or SM103",
    ),
    pytest.mark.skipif(
        os.environ.get("OPEN_VC_ATTN_TEST_LARGE") != "1", reason="Set OPEN_VC_ATTN_TEST_LARGE=1"
    ),
]


def _kernels(fn):
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def _inputs(length, heads):
    torch.manual_seed(11)
    return [torch.randn(length, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]


def test_generic_preparation_is_packed_before_timing():
    # 32769 x 32 also passes the SM103 packed-path work threshold.
    q, k, v = _inputs(32769, 32)
    prepared = prepare_fp8(q, k, v)
    assert not prepared.v_prepacked
    packed = _prepack_v(prepared)
    assert packed.v_prepacked
    assert torch.equal(attention_fp8(packed), attention_fp8(prepared))
    names = _kernels(lambda: attention_fp8(packed))
    assert len(names) == 1, names
    assert not any("transpose_v" in name for name in names)


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="V repair requires B200",
)
def test_repair_attention_scope_launches_one_kernel():
    q, k, v = _inputs(32769, 2)
    call, metadata = make_call(
        "open-vc", q, k, v, prepare_fp8(q, k, v), scope="attention", repair_budget=0.005
    )
    assert metadata["selected_tokens_per_head"] == 164
    names = _kernels(call)
    assert len(names) == 1, names
    assert not any("transpose_v" in name for name in names)
