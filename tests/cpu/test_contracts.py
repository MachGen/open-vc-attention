import ast
import subprocess
import sys
import unittest
from pathlib import Path

from open_vc_attn._dispatch import DEFAULT_VERSION
from open_vc_attn.benchmarking.backends import DEFAULT_BACKENDS, get_backend
from open_vc_attn.benchmarking.measurement import parse_shape, speedup, summarize
from open_vc_attn.integrations.packed import segments

ROOT = Path(__file__).resolve().parents[2]


class Contracts(unittest.TestCase):
    def test_default_implementation_and_reference_identity(self):
        self.assertEqual(DEFAULT_VERSION, "open-vc")
        self.assertEqual(DEFAULT_BACKENDS, ("bf16", "vc", "open-vc"))
        self.assertEqual(get_backend("open-vc").mode, "expcast")
        self.assertEqual(get_backend("vc").mode, "vsmooth")
        self.assertEqual(get_backend("bf16").version, "reference")

    def test_gpu_uuid_forms(self):
        from open_vc_attn.benchmarking.runner import _nvidia_uuid

        self.assertEqual(_nvidia_uuid("abc"), "GPU-abc")
        self.assertEqual(_nvidia_uuid("GPU-abc"), "GPU-abc")
        self.assertEqual(_nvidia_uuid("MIG-abc"), "MIG-abc")

    def test_provenance_covers_kernel_sources(self):
        import shutil
        import tempfile

        from open_vc_attn.benchmarking.runner import _runtime_provenance

        tuning = "_kernels/blackwell/flash_attn/cute/fp8_tuning.py"
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "open_vc_attn"
            shutil.copytree(ROOT / "src/open_vc_attn", copy)
            before = _runtime_provenance(copy)
            self.assertIn(tuning, before["source_sha256"])
            self.assertIn("_kernels/blackwell/v_smooth.py", before["source_sha256"])
            path = copy / tuning
            text = path.read_text()
            self.assertIn("split_encoding = tuned and expcast", text)
            path.write_text(
                text.replace("split_encoding = tuned and expcast", "split_encoding = False")
            )
            after = _runtime_provenance(copy)
        self.assertNotEqual(before["package_sha256"], after["package_sha256"])
        self.assertNotEqual(before["source_sha256"][tuning], after["source_sha256"][tuning])

    def test_pairing_is_not_ratio_of_independent_medians(self):
        result = speedup([1, 2, 100], [1, 100, 2])
        self.assertEqual(result["ratio_of_medians"], 1)
        self.assertEqual(result["paired_ratios"], [1, 0.02, 50])
        with self.assertRaises(ValueError):
            speedup([1], [1, 2])
        with self.assertRaises(ValueError):
            summarize([float("nan")])
        with self.assertRaises(ValueError):
            summarize([0])

    def test_shape_and_backend_names_fail_closed(self):
        self.assertEqual(parse_shape("188214x7x128"), (188214, 7, 128))
        for bad in ("0x7x128", "4096x7x64", "4096", "x7x128"):
            with self.assertRaises(ValueError):
                parse_shape(bad)
        with self.assertRaises(ValueError):
            get_backend("fastest")
        with self.assertRaises(ValueError):
            get_backend("fp8")
        self.assertEqual(get_backend("bf16").mode, "bf16")

    def test_packed_boundaries_do_not_merge_sequences(self):
        self.assertEqual(segments((0, 129, 386), 386, 257), [(0, 129, False), (129, 386, False)])
        self.assertEqual(
            segments((0, 129, 256), 256, 129, trailing_padding=True),
            [(0, 129, False), (129, 256, True)],
        )
        for bounds in (None, (1, 256), (0, 129, 128, 256), (0, 0, 256)):
            with self.assertRaises(ValueError):
                segments(bounds, 256, 256)

    def test_import_and_help_need_no_cuda_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys,open_vc_attn; from open_vc_attn.benchmarking.backends import BACKENDS; "
                "assert 'torch' not in sys.modules; assert len(BACKENDS)==3",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, "-m", "open_vc_attn.cli.bench", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_metadata_and_invalid_arguments_without_torch(self):
        import json

        result = subprocess.run(
            [sys.executable, "-m", "open_vc_attn.cli.info"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(json.loads(result.stdout)["backends"]), ["bf16", "vc", "open-vc"])
        for args in (
            ["--shapes", "0x7x128"],
            ["--backends", "fp8"],
            ["--repair-budget", "nan"],
            ["--repair-budget", "1"],
            ["--preparation", "fused"],
            ["--backends", "bf16", "vc", "--repair-budget", "0.01"],
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-m", "open_vc_attn.cli.bench", *args],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2, result.stderr)

    def test_runtime_has_no_platform_dependency(self):
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [x.name for x in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                self.assertFalse(
                    any(x.startswith(("machgen", "flash_attention_plus")) for x in names), path
                )


if __name__ == "__main__":
    unittest.main()
