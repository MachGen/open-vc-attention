"""Release defaults must agree without requiring a CUDA installation."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vc_attn._sass_runtime import maybe_patch_compiled
from vc_attn.benchmark import make_call
from vc_attn.registry import DEFAULT_BACKENDS, DEFAULT_MID_WINDOW_BLOCKS, DEFAULT_VERSION

ROOT = Path(__file__).resolve().parents[2]


class ReleaseDefaults(unittest.TestCase):
    def test_public_api_and_benchmark_scan_defaults_agree(self):
        # Load the real API with a type-only Torch stub, then observe which
        # low-level options the public and benchmark routes actually pass.
        torch = types.ModuleType("torch")
        torch.Tensor = object
        source = (ROOT / "src/vc_attn/api.py").read_text()
        api = types.ModuleType("vc_attn.api")
        api.__package__ = "vc_attn"
        with patch.dict(sys.modules, {"torch": torch, "vc_attn.api": api}):
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
                    self.assertIs(metadata["sass_d_enabled"], False)
            historical, _ = make_call(
                "vc_v4", object(), object(), object(), prepared, scope="attention"
            )
            historical()
            self.assertIsNone(api.raw_forward.call_args.kwargs["mid_window_blocks"])
            api.attention_fp8(prepared, causal=True)
            self.assertIsNone(api.raw_forward.call_args.kwargs["mid_window_blocks"])

    def test_default_never_inspects_or_exports_d_binary(self):
        compiled = object()
        with (
            patch("vc_attn._sass_runtime.importlib.metadata.version") as version,
            patch("vc_attn._sass_runtime._export_patched") as export,
        ):
            for window in (None, 0, 4, 8, 1024):
                self.assertIs(maybe_patch_compiled(compiled, mid_window_blocks=window), compiled)
            version.assert_not_called()
            export.assert_not_called()
            version.return_value = "unsupported"
            self.assertIs(maybe_patch_compiled(compiled, enable=True), compiled)
            version.assert_called_once_with("nvidia-cutlass-dsl")
            export.assert_not_called()

    def test_public_kernel_hook_does_not_opt_into_experimental_d(self):
        path = ROOT / "src/vc_attn/_kernels/v4/flash_attn/cute/interface.py"
        calls = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "maybe_patch_compiled"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("enable", [arg.arg for arg in calls[0].keywords])


if __name__ == "__main__":
    unittest.main()
