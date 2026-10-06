"""Release defaults must agree without requiring a CUDA installation."""

import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from open_vc_attn._dispatch import DEFAULT_MID_WINDOW_BLOCKS, DEFAULT_VERSION
from open_vc_attn.benchmarking.backends import DEFAULT_BACKENDS
from open_vc_attn.benchmarking.runner import make_call

ROOT = Path(__file__).resolve().parents[2]


class ReleaseDefaults(unittest.TestCase):
    def test_public_api_and_benchmark_scan_defaults_agree(self):
        # Load the real API with a type-only Torch stub, then observe which
        # low-level options the public and benchmark routes actually pass.
        torch = types.ModuleType("torch")
        torch.Tensor = object
        source = (ROOT / "src/open_vc_attn/api.py").read_text()
        api = types.ModuleType("open_vc_attn.api")
        api.__package__ = "open_vc_attn"
        # Packing V before timing is checked on GPU (byte-identical outputs); keep the
        # stubbed prepared inputs as they are here.
        with (
            patch.dict(sys.modules, {"torch": torch, "open_vc_attn.api": api}),
            patch("open_vc_attn.benchmarking.runner._prepack_v", side_effect=lambda p: p),
        ):
            exec(compile(source, "api.py", "exec"), api.__dict__)
            prepared = types.SimpleNamespace(
                q=object(),
                k=object(),
                v=object(),
                q_descale=1,
                k_descale=2,
                v_descale=3,
                layout={},
                output_shape=(129, 2, 128),
            )
            api.validate_qkv = Mock()
            api._fused_preparation_eligible = Mock(return_value=False)
            api.prepare_fp8 = Mock(return_value=prepared)
            api.raw_forward = Mock(return_value=(Mock(), None))
            api.attention(object(), object(), object())
            public_options = api.raw_forward.call_args.kwargs
            self.assertEqual(public_options["version"], DEFAULT_VERSION)
            self.assertEqual(public_options["mid_window_blocks"], DEFAULT_MID_WINDOW_BLOCKS)
            for scope in ("attention", "quantize-attention"):
                with self.subTest(scope=scope):
                    call, metadata = make_call(
                        DEFAULT_BACKENDS[-1], object(), object(), object(), prepared, scope=scope
                    )
                    call()
                    self.assertEqual(api.raw_forward.call_args.kwargs, public_options)
                    self.assertEqual(metadata["mid_window_blocks"], 4)
            api.attention_fp8(prepared, mid_window_blocks=None)
            self.assertIsNone(api.raw_forward.call_args.kwargs["mid_window_blocks"])
            api.attention_fp8(prepared, causal=True)
            self.assertIsNone(api.raw_forward.call_args.kwargs["mid_window_blocks"])

    def test_vc_baseline_uses_v_smooth_expcast_and_original_scan(self):
        torch = types.ModuleType("torch")
        torch.Tensor = object
        torch.bfloat16 = "bf16"
        torch.no_grad = lambda: lambda fn: fn
        torch.cuda = types.SimpleNamespace(device=Mock(return_value=contextlib.nullcontext()))
        grouped = types.SimpleNamespace(
            q=object(),
            k=object(),
            v=object(),
            permutation="perm",
            forward_kwargs=lambda: {"v_smooth": True, "v_smooth_means": 1, "v_smooth_scale": 2},
        )
        smooth = types.ModuleType("open_vc_attn._kernels.blackwell.v_smooth")
        smooth.prepare_v_smooth = Mock(return_value=grouped)
        api = types.ModuleType("open_vc_attn.api")
        api.validate_qkv = Mock()
        api._layout = Mock(return_value=(None, None, None, {"max_seqlen_q": 129}, None))
        api.prepare_fp8 = Mock()
        out = Mock()
        api.raw_forward = Mock(return_value=(out, None))
        modules = {
            "torch": torch,
            "open_vc_attn.api": api,
            "open_vc_attn._kernels.blackwell.v_smooth": smooth,
        }
        with patch.dict(sys.modules, modules):
            baselines = types.ModuleType("open_vc_attn.baselines")
            baselines.__package__ = "open_vc_attn"
            source = (ROOT / "src/open_vc_attn/baselines.py").read_text()
            exec(compile(source, "baselines.py", "exec"), baselines.__dict__)
            q = types.SimpleNamespace(
                dtype="bf16", shape=(129, 2, 128), ndim=3, device="cuda:1", reshape=lambda *a: q
            )
            baselines.vc_attention(q, q, q)
        # Preparation launches Triton kernels, which use the current device: select the inputs'.
        torch.cuda.device.assert_called_once_with("cuda:1")
        options = api.raw_forward.call_args.kwargs
        self.assertIs(options["expcast"], True)
        self.assertIsNone(options["mid_window_blocks"])
        self.assertIs(options["v_smooth"], True)
        self.assertNotIn("version", options)
        out.reshape.assert_called_once_with((129, 2, 128))
