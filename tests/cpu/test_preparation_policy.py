"""Preparation policy selection without a CUDA runtime."""

import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api():
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.cuda = types.SimpleNamespace(get_device_capability=Mock(return_value=(10, 0)))
    module = types.ModuleType("open_vc_attn.api")
    module.__package__ = "open_vc_attn"
    with patch.dict(sys.modules, {"torch": torch, "open_vc_attn.api": module}):
        exec(
            compile((ROOT / "src/open_vc_attn/api.py").read_text(), "api.py", "exec"),
            module.__dict__,
        )
        module.validate_qkv = Mock()
        module._packed_dsl_available = Mock(return_value=True)
        module.prepare_fp8 = Mock(return_value="general")
        module.prepare_fp8_fused = Mock(return_value="packed")
        module.attention_fp8 = Mock(return_value="output")
        yield module


def tensor(shape=(32769, 2, 128), contiguous=True):
    return types.SimpleNamespace(
        ndim=len(shape), shape=shape, device="cuda:0", is_contiguous=lambda: contiguous
    )


def test_default_fuses_and_explicit_unfused_preserves_general_api(api):
    q = tensor()
    api.attention(q, q, q)
    api.prepare_fp8_fused.assert_called_once_with(q, q, q)
    api.prepare_fp8.assert_not_called()
    api.attention_fp8.assert_called_once()
    assert api.attention_fp8.call_args.args == ("packed",)
    api.attention(q, q, q, preparation="unfused")
    api.prepare_fp8.assert_called_once_with(q, q, q)
    assert api.attention_fp8.call_args.args == ("general",)


@pytest.mark.parametrize(
    "case",
    ["short", "batch", "cross", "strided", "causal", "lse", "sm103", "dsl"],
)
def test_ineligible_calls_preserve_general_preparation_and_forced_fusion_fails(api, case):
    q = k = v = tensor()
    options = {}
    if case == "short":
        q = k = v = tensor((32767, 2, 128))
    elif case == "batch":
        q = k = v = tensor((2, 32769, 2, 128))
    elif case == "cross":
        k = v = tensor((33000, 2, 128))
    elif case == "strided":
        v = tensor(contiguous=False)
    elif case == "causal":
        options["causal"] = True
    elif case == "lse":
        options["return_lse"] = True
    elif case == "sm103":
        api.torch.cuda.get_device_capability.return_value = (10, 3)
    elif case == "dsl":
        api._packed_dsl_available.return_value = False
    api.attention(q, k, v, **options)
    api.prepare_fp8.assert_called_once_with(q, k, v)
    api.prepare_fp8_fused.assert_not_called()
    with pytest.raises(ValueError, match="Fused preparation requires"):
        api.attention(q, k, v, preparation="fused", **options)


@pytest.mark.parametrize(
    "options",
    [{"mode": "fp8"}, {"mode": "bf16"}, {"version": "reference"}],
)
def test_only_open_vc_expcast_and_reference_bf16_are_accepted(api, options):
    q = tensor()
    with pytest.raises(ValueError):
        api.attention(q, q, q, **options)
