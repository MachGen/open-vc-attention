"""Render completed standalone benchmark JSON without importing CUDA libraries."""

import argparse
import json
from pathlib import Path

from .measurement import summarize
from .registry import DEFAULT_BACKENDS


def render(report, *, baseline=None, candidate=None):
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("Expected a complete standalone benchmark report (schema_version=1)")
    baseline = baseline or report.get("baseline")
    rows = report.get("shapes", [])
    if not rows:
        raise ValueError("Report contains no measured shapes")
    backends = list(rows[0].get("timings", {}))
    if not backends:
        raise ValueError("Report contains no measured backends")
    if candidate is None:
        candidate = next(
            (name for name in (DEFAULT_BACKENDS[-1], "vc_v4", "vc_scaled") if name in backends),
            backends[-1],
        )
    if baseline not in backends or candidate not in backends:
        raise ValueError(f"Both {baseline!r} and {candidate!r} must be measured backends")
    lines = [
        f"GPU: {report.get('gpu', 'unknown')}; scope: {report.get('scope')}; "
        f"timer: {report.get('timing')}; isolation checked: {report.get('isolation_checked')}",
        f"Speedup = {baseline} median / {candidate} median. "
        "Approximate attention; errors are relative to bf16_ref.",
        "",
        "| S x H x D | "
        + " | ".join(f"{name} ms" for name in backends)
        + f" | {candidate} speedup | {candidate} rel L2 |",
        "|---|" + "---:|" * (len(backends) + 2),
    ]
    for row in rows:
        timings = row.get("timings", {})
        if set(timings) != set(backends):
            raise ValueError("Measured backend sets differ between shapes")
        medians = {}
        rounds = report.get("settings", {}).get("rounds")
        for name in backends:
            samples = timings[name].get("samples_ms", [])
            if rounds is not None and len(samples) != rounds:
                raise ValueError("Incomplete sample count in a completed report")
            medians[name] = summarize(samples)["median_ms"]
        accuracy = row.get("accuracy", {}).get(candidate, {})
        if accuracy.get("finite") is not True:
            raise ValueError("Candidate did not record finite output")
        error = accuracy.get("relative_l2")
        error_text = f"{100 * error:.3f}%" if isinstance(error, (int, float)) else "unavailable"
        shape = " x ".join(str(x) for x in row["shape"])
        lines.append(
            f"| {shape} | "
            + " | ".join(f"{medians[name]:.3f}" for name in backends)
            + f" | {medians[baseline] / medians[candidate]:.3f}x | {error_text} |"
        )
    if report.get("isolation_checked") is not True:
        lines += ["", "Shared-GPU result: do not use as an isolated performance claim."]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--baseline", help="Choose any backend measured in this same run")
    parser.add_argument(
        "--candidate",
        help="Prefers vc_v4_mid4, then historical vc_v4/vc_scaled, then the last backend",
    )
    args = parser.parse_args(argv)
    try:
        print(
            render(
                json.loads(args.result.read_text()),
                baseline=args.baseline,
                candidate=args.candidate,
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
