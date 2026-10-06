"""CPU-only statistical helpers shared by benchmarking and result validation."""

import math
import statistics


def summarize(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Timing samples must be finite and positive")
    mean = statistics.mean(samples)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": mean,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "cv_percent": 100 * statistics.pstdev(samples) / mean,
        "samples_ms": list(samples),
    }


def speedup(baseline, candidate):
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired samples must have equal nonzero length")
    summarize(baseline)
    summarize(candidate)
    ratios = [a / b for a, b in zip(baseline, candidate)]
    return {
        "ratio_of_medians": statistics.median(baseline) / statistics.median(candidate),
        "median_paired_ratio": statistics.median(ratios),
        "paired_ratios": ratios,
        "latency_reduction_percent": 100
        * (1 - statistics.median(candidate) / statistics.median(baseline)),
    }


def parse_shape(value):
    try:
        result = tuple(int(x) for x in value.replace(",", "x").split("x"))
    except ValueError as exc:
        raise ValueError("Shape must be SxHx128") from exc
    if len(result) != 3 or min(result) < 1 or result[2] != 128:
        raise ValueError("Shape must be SxHx128 with positive dimensions")
    return result
