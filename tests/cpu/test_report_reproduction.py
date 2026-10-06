import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "report_common", ROOT / "tools/report_reproduction/common.py"
)
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


class ReportReproduction(unittest.TestCase):
    def test_balanced_order_and_paired_statistics(self):
        routes = ["pre_fusedpipe", "fusedpipe", "bf16", "d"]
        orders = common.balanced_orders(routes, 12, 20261004)
        for i in range(4):
            for route in routes:
                self.assertEqual(sum(order[i] == route for order in orders), 3)
        self.assertEqual(orders, common.balanced_orders(routes, 12, 20261004))
        for bad in (0, 3, 11):
            with self.assertRaises(ValueError):
                common.balanced_orders(routes, bad, 1)
        samples = {"pre_fusedpipe": [2, 4, 6], "fusedpipe": [1, 2, 3], "bf16": [4, 8, 12]}
        result = common.summarize(samples)
        self.assertAlmostEqual(
            result["comparisons"]["pre_fusedpipe_over_fusedpipe"]["paired_geometric_mean"], 2
        )
        self.assertEqual(result["comparisons"]["bf16_over_fusedpipe"]["wins"], 3)
        for invalid in (
            {},
            {**samples, "d": [1]},
            {**samples, "bf16": [0, 1, 2]},
            {**samples, "fusedpipe": [float("nan"), 2, 3]},
        ):
            with self.assertRaises(ValueError):
                common.summarize(invalid)

    def test_reject_incomplete_or_failed_evidence(self):
        routes = ["pre_fusedpipe", "fusedpipe", "bf16"]
        report = dict(
            schema="vc-report-replay-v1",
            status="complete",
            guard_failure=None,
            byte_checks=[{"equal": True}],
            d_enabled=False,
            raw_samples_ms={r: [1, 2, 3] for r in routes},
            single_kernel={r: "flash_attention" for r in routes},
            settings={"rounds": 3},
            orders=common.balanced_orders(routes, 3, 1),
        )
        common.validate_result(report)
        invalid = [
            {"status": "running"},
            {"guard_failure": "foreign PID"},
            {"byte_checks": []},
            {"byte_checks": [{"equal": False}]},
            {"single_kernel": {}},
            {"d_enabled": True},
            {"orders": report["orders"][:2]},
            {"orders": [routes] * 3},
        ]
        for replacement in invalid:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                common.validate_result({**copy.deepcopy(report), **replacement})

    def test_controls_are_isolated_and_unknown_gate_rejected(self):
        original = common.tree_hashes(ROOT / "src/vc_attn")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "source/vc_attn"
            proof = common.stage_source(ROOT / "src/vc_attn", dest)
            self.assertEqual(common.tree_hashes(ROOT / "src/vc_attn"), original)
            self.assertEqual(proof["input_files"], original)
            kernel = Path("_kernels/v4/flash_attn/cute/flash_fwd_sm100.py")
            before, after = (ROOT / "src/vc_attn" / kernel).read_text(), (dest / kernel).read_text()
            self.assertEqual(after.replace("False and descale_tensors", "descale_tensors"), before)
            self.assertNotEqual(after, before)
            # A second transformation must fail instead of silently changing an unknown family.
            with self.assertRaisesRegex(ValueError, "Unknown fusedpipe gate"):
                common.stage_source(dest, Path(tmp) / "repeated")


if __name__ == "__main__":
    unittest.main()
