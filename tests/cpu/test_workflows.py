import copy
import importlib.metadata
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from vc_attn.check import inspect_installation
from vc_attn.report import render

ROOT = Path(__file__).resolve().parents[2]


class UserWorkflows(unittest.TestCase):
    def test_report_selects_new_candidate_without_breaking_historical_reports(self):
        report = json.loads((ROOT / "docs/benchmarks/b200-main.json").read_text())
        self.assertIn("/ vc_scaled median", render(report))
        for row in report["shapes"]:
            row["timings"]["vc_v4"] = copy.deepcopy(row["timings"]["vc_scaled"])
            row["accuracy"]["vc_v4"] = copy.deepcopy(row["accuracy"]["vc_scaled"])
        self.assertIn("/ vc_v4 median", render(report))
        self.assertIn("/ vc_scaled median", render(report, candidate="vc_scaled"))
        for row in report["shapes"]:
            row["timings"]["vc_v4_mid4"] = copy.deepcopy(row["timings"]["vc_scaled"])
            row["accuracy"]["vc_v4_mid4"] = copy.deepcopy(row["accuracy"]["vc_scaled"])
        self.assertIn("/ vc_v4_mid4 median", render(report))
        for row in report["shapes"]:
            for key in ("timings", "accuracy"):
                del row[key]["vc_v4"]
                del row[key]["vc_scaled"]
        self.assertIn("/ vc_v4_mid4 median", render(report))

    def test_published_report_can_change_denominator_without_retiming(self):
        report = json.loads((ROOT / "docs/benchmarks/b200-main.json").read_text())
        self.assertIn("1.466x", render(report))
        self.assertIn("1.833x", render(report, baseline="bf16_ref"))
        self.assertIn("5.665%", render(report))
        # Cached summary fields must not override the preserved raw samples.
        report["shapes"][1]["timings"]["vc_scaled"]["median_ms"] = 1
        self.assertIn("54.080", render(report))

    def test_incomplete_or_invalid_results_cannot_be_presented_as_complete(self):
        report = json.loads((ROOT / "docs/benchmarks/b200-main.json").read_text())
        for status in ("running", "invalid"):
            bad = dict(report, status=status)
            with self.assertRaises(ValueError):
                render(bad)
        bad = copy.deepcopy(report)
        bad["shapes"][0]["timings"]["fp8_ref"]["samples_ms"].pop()
        with self.assertRaises(ValueError):
            render(bad)
        with self.assertRaises(ValueError):
            render(report, baseline="native_v6")

    def test_invalid_bench_commands_fail_before_importing_torch(self):
        import builtins

        from vc_attn.benchmark import main

        original = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("Invalid command tried to initialize Torch")
            return original(name, *args, **kwargs)

        invalid = [
            ["--shapes", "0x7x128"],
            ["--backends", "native_v6", "vc_scaled", "--baseline", "native_v6"],
            [
                "--backends",
                "native_v6",
                "vc_scaled",
                "--baseline",
                "native_v6",
                "--scope",
                "quantize-attention",
            ],
            ["--warm-seconds", "nan"],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), patch("builtins.__import__", no_torch):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    main(argv)
                self.assertEqual(exc.exception.code, 2)

    def test_install_check_is_actionable_without_cuda_imports(self):
        def versions(name):
            if name == "cuda-python":
                raise importlib.metadata.PackageNotFoundError(name)
            return {"nvidia-cutlass-dsl": "4.5.0", "quack-kernels": "0.6.1"}.get(name, "1.0")

        with (
            patch("vc_attn.check.importlib.metadata.version", versions),
            patch("vc_attn.check.platform.system", return_value="Linux"),
            patch("vc_attn.check.sys.version_info", (3, 12)),
        ):
            report = inspect_installation()
        self.assertEqual(len(report["errors"]), 2)
        self.assertIn("Missing cuda-python", report["errors"][0])
        self.assertIn("must be 4.6.0", report["errors"][1])
        for module in ("vc_attn.check", "vc_attn.report"):
            proc = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
