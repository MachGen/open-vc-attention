"""Published benchmark records: assembly from runner outputs and the schema check CI runs."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("records", ROOT / "tools/reports/records.py")
records = importlib.util.module_from_spec(spec)
spec.loader.exec_module(records)

PROVENANCE = {
    "package_sha256": "a" * 64,
    "source_sha256": {"api.py": "b" * 64, "_kernels/blackwell/v_smooth.py": "c" * 64},
    "source_revision": {"commit": "d" * 40, "dirty": False},
}


def _row(shape, names, *, repair=0.0):
    timings = {
        n: {"median_ms": 2.0 if n == "bf16" else 1.0, "samples_ms": [1.0, 2.0]} for n in names
    }
    row = {
        "shape": list(shape),
        "accuracy": {n: {"relative_l2": 0.03, "rmse": 0.2, "max_abs": 9.0} for n in names},
        "timings": timings,
        "speedups": {n: {"ratio_of_medians": 2.0, "median_paired_ratio": 2.0} for n in names},
        "orders": [list(names)],
        "backend_metadata": {"open-vc": {"selected_tokens_per_head": round(repair * shape[0])}},
    }
    if "vc" in names:
        row["vc_complete_call_events_ms"] = {
            "fresh_grouping": [3.0] * 5,
            "reused_grouping": [2.0] * 5,
        }
    return row


def _run(scope, shape, names, *, repair=0.0, provenance=PROVENANCE):
    return {
        "status": "complete",
        "isolation_checked": True,
        "timing": "graph",
        "scope": scope,
        "baseline": "bf16",
        "backends": {n: n for n in names},
        "gpu": "NVIDIA B200",
        "versions": {"torch": "2.11.0"},
        "device_configuration_start": {},
        "input_sha256": f"input-{shape[1]}",
        "source_provenance": provenance,
        "settings": {"rounds": 12, "repeats": 10, "warm_calls": 6, "repair_budget": repair},
        "shapes": [_row(shape, names, repair=repair)],
    }


def _write(directory, runs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, run in enumerate(runs):
        path = directory / f"run{i}.json"
        path.write_text(json.dumps(run))
        paths.append(path)
    return paths


class Records(unittest.TestCase):
    def _build(self, tmp, provenance=PROVENANCE):
        trio = ("bf16", "vc", "open-vc")
        comparison = [
            _run(scope, (32768, h, 128), trio, provenance=provenance)
            for h in (7, 56)
            for scope in ("attention", "quantize-attention")
        ]
        repair = [
            _run(scope, (32768, 56, 128), ("bf16", "open-vc"), repair=b, provenance=provenance)
            for b in (0.0, 0.005)
            for scope in ("attention", "quantize-attention")
        ]
        return (
            records.build_comparison(_write(Path(tmp) / "cmp", comparison), "test input"),
            records.build_repair(_write(Path(tmp) / "rep", repair), "test input"),
        )

    def test_built_records_pass_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmp, rep = self._build(tmp)
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "comparison.json").write_text(json.dumps(cmp))
            (out / "repair.json").write_text(json.dumps(rep))
            self.assertEqual(records.check(out), [])
        self.assertEqual([r["shape"][1] for r in cmp["rows"]], [7, 56])
        grouping = cmp["rows"][0]["vc_complete_call_events_ms"]
        self.assertEqual(grouping["fresh_grouping"]["median_ms"], 3.0)
        self.assertEqual(cmp["source_provenance"], PROVENANCE)
        self.assertEqual([b["budget"] for b in rep["budgets"]], [0.0, 0.005])

    def test_dirty_or_mixed_sources_are_refused(self):
        dirty = {**PROVENANCE, "source_revision": {"commit": "d" * 40, "dirty": True}}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            self._build(tmp, provenance=dirty)

    def test_old_records_fail_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "comparison.json").write_text(
                json.dumps({"schema": "open-vc-comparison-v2", "rows": [{"vc_grouping_ms": {}}]})
            )
            (out / "repair.json").write_text(json.dumps({"schema": "open-vc-repair-v1"}))
            errors = records.check(out)
        self.assertTrue(any("kernel source hashes" in e for e in errors))
        self.assertTrue(any("vc_complete_call_events_ms" in e for e in errors))
