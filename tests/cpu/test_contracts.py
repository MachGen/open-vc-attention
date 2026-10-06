import ast
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

from vc_attn.integrations.packed import segments
from vc_attn.measurement import parse_shape, speedup, summarize
from vc_attn.registry import DEFAULT_BACKENDS, DEFAULT_VERSION, get_backend, source_manifest

ROOT = Path(__file__).resolve().parents[2]


class Contracts(unittest.TestCase):
    def test_default_selects_synced_snapshot_and_retains_historical_backends(self):
        self.assertEqual(DEFAULT_VERSION, "v4")
        self.assertEqual(DEFAULT_BACKENDS[-1], "vc_v4_mid4")
        self.assertEqual(get_backend(DEFAULT_BACKENDS[-1]).mode, "expcast_mid4")
        self.assertEqual(get_backend("vc_v4").mode, "expcast")
        self.assertEqual(get_backend("vc_v4").version, DEFAULT_VERSION)
        self.assertEqual(get_backend("fp8_v4").version, DEFAULT_VERSION)
        self.assertEqual(get_backend("vc_scaled").version, "scaled")
        manifest = source_manifest()["versions"]
        self.assertEqual(
            manifest["v4"]["source_revision"], "8aa761eac845d734dc5dbe48a6196c1fe1b0a7cf"
        )
        self.assertEqual(
            manifest["scaled"]["source_revision"], "4245ca87a02a476e89b793a5541a1f0576684b01"
        )

    def test_gpu_uuid_forms(self):
        from vc_attn.benchmark import _nvidia_uuid

        self.assertEqual(_nvidia_uuid("abc"), "GPU-abc")
        self.assertEqual(_nvidia_uuid("GPU-abc"), "GPU-abc")
        self.assertEqual(_nvidia_uuid("MIG-abc"), "MIG-abc")

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
        self.assertEqual(get_backend("fp8_ref").mode, "fp8")
        self.assertEqual(get_backend("bf16_ref").mode, "bf16")

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
                "import sys,vc_attn; from vc_attn.registry import BACKENDS; "
                "assert 'torch' not in sys.modules; assert len(BACKENDS)>5",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, "-m", "vc_attn.benchmark", "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_kernel_provenance(self):
        manifest = source_manifest()
        for version in manifest["versions"].values():
            self.assertEqual(len(version["source_revision"]), 40)
            for name, record in version["files"].items():
                self.assertEqual(
                    hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), record["sha256"], name
                )

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
