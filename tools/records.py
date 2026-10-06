"""Build and check the published B200 benchmark records from open-vc-attn-bench outputs.

``build`` assembles ``comparison.json`` (BF16, VC and Open-VC; one attention-scope and one
quantize-attention run per shape) and ``repair.json`` (Open-VC V repair budgets; one run per
budget and scope). It refuses incomplete, non-isolated or mixed-source runs, and carries the
runs' source provenance (per-file hashes, package hash, git revision) into the records.

``check`` validates the published records' schema and source provenance; the release audit runs it.

    python tools/records.py build --comparison cmp_*.json --repair rep_*.json \\
        --input-description "..." --output-dir benchmarks/results/b200
    python tools/records.py check
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "benchmarks/results/b200"
BACKENDS = ("bf16", "vc", "open-vc")
SCOPES = {"attention": "attention", "quantize-attention": "complete_call"}
GROUP_FRACTION = 0.25

TIMING = {
    "unit": "milliseconds per CUDA Graph replay",
    "order": "paired rounds in randomized order (seeded)",
    "statistic": "speedup = BF16 median / candidate median, per scope",
    "attention": (
        "one attention kernel per replay; inputs, including packed V, prepared beforehand "
        "(Open-VC: fused or packed preparation, V repair: packed repair rows; VC: grouped, "
        "permuted and quantized)"
    ),
    "complete_call": (
        "the full call from BF16 inputs, including metadata and FP8 preparation "
        "(VC: V-Smooth preparation with the reused grouping)"
    ),
}
ISOLATION = "GPU ownership checked before and after every sample; no other compute process"
V_SMOOTH = {
    "implementation": (
        "tuned kernel path (FP32 means, mean prefetch, tensor-core mean restoration), original "
        "key scan; fused two-pass Triton preparation with a reused grouping; k-means "
        "(64 groups, 4 iterations) with tensor-core centroid updates"
    ),
    "grouping": (
        "vc_complete_call_events_ms per row: VC complete calls with a fresh and with a reused "
        "grouping, interleaved, CUDA events, 5 samples each. Their median difference is the "
        "grouping cost; VC-Attention groups on the first quarter of denoising steps and reuses "
        "the grouping afterwards, so timed calls reuse it."
    ),
    "grouping_step_fraction": GROUP_FRACTION,
}


def _load(paths):
    runs = []
    for path in paths:
        run = json.loads(Path(path).read_text())
        if run.get("status") != "complete" or not run.get("isolation_checked"):
            raise SystemExit(f"{path}: not a complete, isolated run")
        if run.get("timing") != "graph" or len(run["shapes"]) != 1:
            raise SystemExit(f"{path}: expected one shape timed with CUDA graphs")
        runs.append(run)
    identities = {json.dumps(r["source_provenance"], sort_keys=True) for r in runs}
    if len(identities) != 1:
        raise SystemExit("Runs come from different sources; rerun them from one revision")
    revision = runs[0]["source_provenance"].get("source_revision")
    if not revision or revision.get("dirty"):
        raise SystemExit("Publish only runs from a clean git checkout")
    return runs


def _settings(run):
    s = run["settings"]
    return {
        "rounds": s["rounds"],
        "repeats_per_sample": s["repeats"],
        "warm_calls_per_sample": s["warm_calls"],
    }


def _header(runs, schema, description):
    first = runs[0]
    if any(_settings(r) != _settings(first) or r["gpu"] != first["gpu"] for r in runs):
        raise SystemExit("Runs use different GPUs or timing settings")
    return {
        "schema": schema,
        "gpu": first["gpu"],
        "input": description,
        "timing": {**TIMING, **_settings(first)},
        "isolation": ISOLATION,
        "versions": first["versions"],
        "device_start": first["device_configuration_start"],
        "source_provenance": first["source_provenance"],
    }


def _summary(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"median_ms": median, "samples_ms": list(values)}


def _scope(row, names):
    return {
        "median_ms": {n: row["timings"][n]["median_ms"] for n in names},
        "samples_ms": {n: row["timings"][n]["samples_ms"] for n in names},
        "speedup_vs_bf16": {n: row["speedups"][n]["ratio_of_medians"] for n in names[1:]},
        "median_paired_speedup_vs_bf16": {
            n: row["speedups"][n]["median_paired_ratio"] for n in names[1:]
        },
        "orders": row["orders"],
    }


def build_comparison(paths, description):
    runs = _load(paths)
    by_shape = {}
    for run in runs:
        if set(run["backends"]) != set(BACKENDS) or run["baseline"] != "bf16":
            raise SystemExit("Comparison runs must time bf16, vc and open-vc against bf16")
        if run["settings"].get("repair_budget"):
            raise SystemExit("Comparison runs must not use V repair")
        key = tuple(run["shapes"][0]["shape"])
        if run["scope"] in by_shape.setdefault(key, {}):
            raise SystemExit(f"Duplicate {run['scope']} run for shape {key}")
        by_shape[key][run["scope"]] = run
    rows = []
    for shape, scopes in sorted(by_shape.items(), key=lambda item: item[0][1]):
        if set(scopes) != set(SCOPES):
            raise SystemExit(f"Shape {shape} needs one run per scope")
        attention = scopes["attention"]["shapes"][0]
        complete = scopes["quantize-attention"]["shapes"][0]
        if scopes["attention"]["input_sha256"] != scopes["quantize-attention"]["input_sha256"]:
            raise SystemExit(f"Shape {shape} scopes used different inputs")
        rows.append(
            {
                "shape": list(shape),
                "input_sha256": scopes["attention"]["input_sha256"],
                "accuracy_vs_bf16": {
                    n: {k: attention["accuracy"][n][k] for k in ("relative_l2", "rmse", "max_abs")}
                    for n in BACKENDS[1:]
                },
                "attention": _scope(attention, BACKENDS),
                "complete_call": _scope(complete, BACKENDS),
                "vc_complete_call_events_ms": {
                    name: _summary(samples)
                    for name, samples in complete["vc_complete_call_events_ms"].items()
                },
            }
        )
    record = _header(runs, "open-vc-comparison-v3", description)
    record.update(backends=runs[0]["backends"], v_smooth=V_SMOOTH, rows=rows)
    return record


def build_repair(paths, description):
    runs = _load(paths)
    budgets = {}
    for run in runs:
        if set(run["backends"]) != {"bf16", "open-vc"} or run["baseline"] != "bf16":
            raise SystemExit("Repair runs must time bf16 and open-vc against bf16")
        budget = run["settings"]["repair_budget"]
        if run["scope"] in budgets.setdefault(budget, {}):
            raise SystemExit(f"Duplicate {run['scope']} run for budget {budget}")
        budgets[budget][run["scope"]] = run
    shapes = {tuple(r["shapes"][0]["shape"]) for r in runs}
    inputs = {r["input_sha256"] for r in runs}
    if len(shapes) != 1 or len(inputs) != 1:
        raise SystemExit("Repair runs must share one shape and input")
    entries = []
    for budget, scopes in sorted(budgets.items()):
        if set(scopes) != set(SCOPES):
            raise SystemExit(f"Budget {budget} needs one run per scope")
        attention = scopes["attention"]["shapes"][0]
        complete = scopes["quantize-attention"]["shapes"][0]
        metadata = attention["backend_metadata"]["open-vc"]
        entries.append(
            {
                "budget": budget,
                "attention_median_ms": attention["timings"]["open-vc"]["median_ms"],
                "attention_samples_ms": attention["timings"]["open-vc"]["samples_ms"],
                "bf16_attention_median_ms": attention["timings"]["bf16"]["median_ms"],
                "selected_tokens_per_head": metadata.get("selected_tokens_per_head", 0),
                "repair_tokens_per_head": metadata.get("repair_tokens_per_head", 0),
                "relative_l2_vs_bf16": attention["accuracy"]["open-vc"]["relative_l2"],
                "complete_call_median_ms": complete["timings"]["open-vc"]["median_ms"],
                "complete_call_samples_ms": complete["timings"]["open-vc"]["samples_ms"],
            }
        )
    if entries[0]["budget"] != 0:
        raise SystemExit("Repair records need the zero-budget reference")
    record = _header(runs, "open-vc-repair-v2", description)
    record.update(shape=list(shapes.pop()), input_sha256=inputs.pop(), budgets=entries)
    return record


def _require(condition, message, errors):
    if not condition:
        errors.append(message)


def check(directory=RECORDS):
    """Return a list of problems with the published records; empty when they are usable."""
    errors = []
    cmp = json.loads((directory / "comparison.json").read_text())
    rep = json.loads((directory / "repair.json").read_text())
    for name, record in (("comparison.json", cmp), ("repair.json", rep)):
        provenance = record.get("source_provenance") or {}
        revision = provenance.get("source_revision") or {}
        _require(
            len(provenance.get("package_sha256", "")) == 64
            and any(p.startswith("_kernels/") for p in provenance.get("source_sha256", {})),
            f"{name}: missing kernel source hashes",
            errors,
        )
        _require(
            bool(revision.get("commit")) and revision.get("dirty") is False,
            f"{name}: missing a clean source revision",
            errors,
        )
    _require(cmp.get("schema") == "open-vc-comparison-v3", "comparison.json: schema", errors)
    _require(rep.get("schema") == "open-vc-repair-v2", "repair.json: schema", errors)
    _require(
        cmp.get("v_smooth", {}).get("grouping_step_fraction") == GROUP_FRACTION,
        "comparison.json: grouping_step_fraction",
        errors,
    )
    for row in cmp.get("rows", []):
        where = f"comparison.json row {row.get('shape')}"
        for scope in SCOPES.values():
            medians = row.get(scope, {}).get("median_ms", {})
            _require(set(medians) == set(BACKENDS), f"{where}: {scope} medians", errors)
        timed = row.get("vc_complete_call_events_ms", {})
        _require(
            {"fresh_grouping", "reused_grouping"} <= set(timed),
            f"{where}: vc_complete_call_events_ms",
            errors,
        )
        _require(
            set(row.get("accuracy_vs_bf16", {})) == set(BACKENDS[1:]), f"{where}: accuracy", errors
        )
    _require(bool(cmp.get("rows")), "comparison.json: no rows", errors)
    budgets = [b.get("budget") for b in rep.get("budgets", [])]
    _require(bool(budgets) and budgets[0] == 0, "repair.json: zero-budget reference", errors)
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build", help="Assemble records from open-vc-attn-bench outputs")
    make.add_argument("--comparison", nargs="+", required=True, type=Path)
    make.add_argument("--repair", nargs="+", required=True, type=Path)
    make.add_argument("--input-description", required=True)
    make.add_argument("--output-dir", type=Path, default=RECORDS)
    sub.add_parser("check", help="Validate the published records")
    args = parser.parse_args(argv)
    if args.command == "build":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, record in (
            ("comparison.json", build_comparison(args.comparison, args.input_description)),
            ("repair.json", build_repair(args.repair, args.input_description)),
        ):
            (args.output_dir / name).write_text(json.dumps(record, indent=1) + "\n")
            print(f"Wrote {args.output_dir / name}")
    errors = check(args.output_dir if args.command == "build" else RECORDS)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("Records OK")


if __name__ == "__main__":
    main()
