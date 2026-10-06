"""Data-only contracts shared by the report builder, runner and CPU tests."""

import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPORT_SHAPES = ((188214, 7, 128), (188214, 56, 128), (262144, 7, 128), (262144, 56, 128))
SMOKE_SHAPE = (32769, 7, 128)
CONFIG = dict(
    mid_window_blocks=4, expcast=True, return_lse=False, skip_softmax_error=0.0, skip_pv_gemm=False
)
PACKAGES = (
    "torch",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
    "triton",
    "quack-kernels",
    "cuda-python",
    "apache-tvm-ffi",
    "numpy",
    "einops",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def tree_hashes(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(Path(root).rglob("*.py"))
        if not p.name.startswith("._")
    }


def native_identity(path):
    """Identify the sole CUDA entry, ignoring only its namespace spelling."""
    from vc_attn._sass_d import _elf, _metadata_fingerprint

    data = Path(path).read_bytes()
    candidates = []
    cursor = 0
    while (cursor := data.find(b"\x7fELF", cursor)) >= 0:
        if cursor + 20 <= len(data) and struct.unpack_from("<H", data, cursor + 18)[0] == 190:
            elf = _elf(data[cursor:])
            cubin = data[cursor : cursor + elf["extent"]]
            texts = [s for s in elf["sections"] if s["name"].startswith(".text.")]
            if len(texts) != 1:
                raise ValueError("Expected one CUDA text entry")
            section = texts[0]
            candidates.append(
                {
                    "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
                    "text_bytes": section["size"],
                    "text_sha256": hashlib.sha256(
                        cubin[section["offset"] : section["offset"] + section["size"]]
                    ).hexdigest(),
                    "metadata_sha256": _metadata_fingerprint(cubin, elf, section["name"][6:]),
                }
            )
        cursor += 4
    if len(candidates) != 1:
        raise ValueError("Expected one embedded CUDA ELF")
    return candidates[0]


def tool_identity(name):
    path = shutil.which(name)
    if path is None:
        return {"available": False}
    result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
    return {
        "available": True,
        "name": Path(path).name,
        "sha256": sha(path),
        "version": (result.stdout + result.stderr).strip(),
        "returncode": result.returncode,
    }


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Some compilers bundle ptxas rather than using PATH. Preserve every candidate
    # with an explicit identity; do not claim that a PATH binary was actually used.
    candidates = {}
    for name in ("nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-cu13", "triton"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for entry in dist.files or []:
            if Path(str(entry)).name in ("ptxas", "ptxas-blackwell"):
                path = Path(dist.locate_file(entry))
                result = subprocess.run(
                    [str(path), "--version"], capture_output=True, text=True, timeout=20
                )
                candidates[name + ":" + str(entry)] = {
                    "sha256": sha(path),
                    "version": (result.stdout + result.stderr).strip(),
                }
    return {
        "python": sys.version,
        "packages": packages,
        "ptxas_candidates": candidates,
        "tools": {name: tool_identity(name) for name in ("gcc", "objcopy", "nvcc", "ptxas")},
    }


def balanced_orders(labels, rounds, seed):
    if (
        len(labels) != len(set(labels))
        or not labels
        or rounds < len(labels)
        or rounds % len(labels)
    ):
        raise ValueError("Rounds must be a positive multiple of the number of routes")
    rng = random.Random(seed)
    orders = []
    for _ in range(rounds // len(labels)):
        order = list(labels)
        rng.shuffle(order)
        block = [order[i:] + order[:i] for i in range(len(order))]
        rng.shuffle(block)
        orders.extend(block)
    return orders


def summarize(samples):
    required = {"pre_fusedpipe", "fusedpipe", "bf16"}
    if set(samples) not in (required, required | {"d"}):
        raise ValueError("Expected the complete three-route or four-route cohort")
    lengths = {len(values) for values in samples.values()}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError("Incomplete paired samples")
    if any(not math.isfinite(v) or v <= 0 for values in samples.values() for v in values):
        raise ValueError("Timing samples must be finite and positive")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    pairs = [("pre_fusedpipe", "fusedpipe"), ("bf16", "fusedpipe")]
    if "d" in samples:
        pairs += [("fusedpipe", "d"), ("bf16", "d")]
    comparisons = {}
    for control, candidate in pairs:
        ratios = [a / b for a, b in zip(samples[control], samples[candidate])]
        comparisons[control + "_over_" + candidate] = {
            "ratio_of_medians": medians[control] / medians[candidate],
            "paired_geometric_mean": math.exp(sum(map(math.log, ratios)) / len(ratios)),
            "wins": sum(x > 1 for x in ratios),
            "rounds": len(ratios),
        }
    return {"median_ms": medians, "comparisons": comparisons}


def validate_result(report):
    if report.get("schema") != "vc-report-replay-v1" or report.get("status") != "complete":
        raise ValueError("Only complete report-reproduction results are accepted")
    if (
        report.get("guard_failure", "missing") is not None
        or not report.get("byte_checks")
        or any(item.get("equal") is not True for item in report["byte_checks"])
    ):
        raise ValueError("Missing or failed validation evidence")
    samples = report["raw_samples_ms"]
    routes = {"pre_fusedpipe", "fusedpipe", "bf16"} | ({"d"} if report["d_enabled"] else set())
    if set(samples) != routes or set(report["single_kernel"]) != routes:
        raise ValueError("Missing route or kernel-count evidence")
    rounds = report["settings"]["rounds"]
    if (
        rounds < len(routes)
        or rounds % len(routes)
        or len(report["orders"]) != rounds
        or any(len(values) != rounds for values in samples.values())
    ):
        raise ValueError("Incomplete paired rounds")
    if any(len(order) != len(routes) or set(order) != routes for order in report["orders"]):
        raise ValueError("Invalid route order")
    for position in range(len(routes)):
        if any(
            sum(order[position] == route for order in report["orders"]) != rounds // len(routes)
            for route in routes
        ):
            raise ValueError("Unbalanced route order")
    return summarize(samples)


def stage_source(source, destination):
    """Create private namespaces; the installed package and frozen files are untouched."""
    source, destination = Path(source), Path(destination)
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*")
    )
    kernel = destination / "_kernels/v4/flash_attn/cute/flash_fwd_sm100.py"
    original = kernel.read_text()
    marker = "use_fusedpipe = const_expr(\n            descale_tensors is not None"
    if original.count(marker) != 1:
        raise ValueError("Unknown fusedpipe gate; review the source transformation")
    fused = destination / "_kernels/fusedpipe"
    shutil.copytree(destination / "_kernels/v4", fused)
    for path in fused.rglob("*.py"):
        path.write_text(
            path.read_text().replace("vc_attn._kernels.v4", "vc_attn._kernels.fusedpipe")
        )
    kernel.write_text(
        original.replace(marker, marker.replace("descale_tensors", "False and descale_tensors", 1))
    )
    registry = destination / "registry.py"
    registry.write_text(registry.read_text() + '\nVERSIONS = (*VERSIONS, "fusedpipe")\n')
    return {
        "input_files": tree_hashes(source),
        "staged_files": tree_hashes(destination),
        "transformations": [
            "Clone v4 as fusedpipe with namespace-only import substitutions.",
            "Set only use_fusedpipe=False in staged v4; keep mid4 and every other option identical.",
            "Register the private fusedpipe namespace only in the staged package.",
            "Both controls retain the same current B200 H7 packing eligibility.",
        ],
    }
