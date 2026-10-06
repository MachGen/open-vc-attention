"""Paired benchmark entry point. No model data, host addresses or private launchers."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .backends import BACKENDS, CANDIDATE, DEFAULT_BACKENDS, DEFAULT_BASELINE, get_backend
from .measurement import parse_shape, speedup, summarize


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nvidia_uuid(value):
    """PyTorch versions expose either a bare UUID or NVIDIA's prefixed form."""
    value = str(value)
    return value if value.startswith(("GPU-", "MIG-")) else "GPU-" + value


class GPUOwnership:
    """Fail on other compute PIDs. Scheduler reservation is still the caller's job."""

    def __init__(self, device, enabled=True):
        import torch

        self.enabled = enabled
        self.uuid = _nvidia_uuid(torch.cuda.get_device_properties(device).uuid)
        self.samples = 0

    def check(self):
        if not self.enabled:
            return
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                self.uuid,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        pids = {int(x.strip()) for x in result.stdout.splitlines() if x.strip()}
        foreign = pids - {os.getpid()}
        if foreign:
            raise RuntimeError(f"Other CUDA processes detected: {sorted(foreign)}; reject this run")
        self.samples += 1


def _accuracy(out, ref):
    import torch

    diff = out.float() - ref.float()
    finite = bool(torch.isfinite(out).all())
    if not finite:
        raise RuntimeError("Non-finite attention output")
    return {
        "finite": finite,
        "relative_l2": (diff.norm() / ref.float().norm().clamp_min(1e-12)).item(),
        "max_abs": diff.abs().max().item(),
        "rmse": diff.square().mean().sqrt().item(),
    }


def _prepack_v(prepared):
    """Pack V before timing when the dense dispatch would otherwise pack it per call.

    Fused preparation and V repair already return packed V. Generic preparation leaves
    V unpacked, and eligible packed-path dispatches (for example large B300 shapes,
    which the B200-only fused preparation does not cover) would pack it inside every
    timed call. Ineligible dispatches never pack and are returned unchanged.
    """
    from dataclasses import replace

    from .._kernels.blackwell.flash_attn.cute.v_layout import pack_v
    from ..api import attention_fp8

    if prepared.v_prepacked or prepared.v.ndim != 3:
        return prepared
    packed = replace(prepared, v=pack_v(prepared.v), v_prepacked=True)
    try:
        attention_fp8(packed)
    except ValueError as exc:
        if "v_prepacked" not in str(exc):
            raise
        return prepared
    return packed


def make_call(name, q, k, v, prepared, *, scope, repair_budget=0.0, preparation="auto"):
    """Return a zero-argument callable for one backend and its metadata.

    ``attention`` scope times one attention kernel on inputs prepared beforehand
    (Open-VC uses fused, pre-packed preparation when eligible); ``quantize-attention``
    includes all per-call preparation. In ``attention`` scope V is always packed before
    timing, so each timed call launches exactly one attention kernel. The VC baseline
    computes its V grouping once per shape and reuses the permutation, as VC-Attention
    does after its first denoising steps.
    """
    from ..api import (
        DEFAULT_VERSION,
        _fused_preparation_eligible,
        _layout,
        attention,
        attention_fp8,
        prepare_fp8_fused,
        raw_forward,
    )

    spec = get_backend(name)
    if name == "bf16":
        if scope == "attention":
            q0, k0, v0, layout, _ = _layout(q, k, v)
            return lambda: raw_forward(q0, k0, v0, version="reference", **layout, return_lse=False)[
                0
            ], {"implementation": spec.description}
        return lambda: attention(q, k, v, version="reference", mode="bf16"), {
            "implementation": spec.description
        }
    if name == "vc":
        from ..baselines import attention_vc, prepare_vc

        grouped = prepare_vc(q, k, v)
        metadata = {"implementation": spec.description, "mid_window_blocks": None}
        if scope == "attention":
            _, _, _, layout, _ = _layout(grouped.q, grouped.k, grouped.v)
            return lambda: attention_vc(grouped, output_shape=q.shape, layout=layout), metadata
        return lambda: attention_vc(
            prepare_vc(q, k, v, permutation=grouped.permutation, check_permutation=False),
            output_shape=q.shape,
        ), metadata
    metadata = {"implementation": spec.description, "mid_window_blocks": 4}
    if repair_budget > 0:
        from ..v_repair import attention_v_repair, prepare_v_repair

        selected = round(repair_budget * q.shape[-3])
        metadata.update(
            repair_budget=repair_budget,
            selected_tokens_per_head=selected,
            repair_tokens_per_head=((selected + 127) // 128) * 128,
        )
        if selected:
            if scope == "attention":
                repaired = prepare_v_repair(q, k, v, budget=repair_budget)
                return lambda: attention_v_repair(repaired), metadata
            return lambda: attention_v_repair(
                prepare_v_repair(q, k, v, budget=repair_budget)
            ), metadata
    if scope == "quantize-attention":
        return lambda: attention(q, k, v, preparation=preparation), metadata
    if _fused_preparation_eligible(
        q, k, v, version=DEFAULT_VERSION, mode="expcast", causal=False, return_lse=False
    ):
        prepared = prepare_fp8_fused(q, k, v)
    else:
        prepared = _prepack_v(prepared)
    return lambda: attention_fp8(prepared), metadata


def _measure(fn, repeats):
    import torch

    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _measure_vc_grouping(q, k, v, samples=5):
    """Time VC complete calls with a fresh and with a reused grouping, interleaved.

    V-Smooth's k-means grouping runs only on early denoising steps. Both variants include
    the same preparation and attention and are timed the same way (CUDA events, one call
    per sample), so their difference is the incremental grouping cost to amortize.
    """
    from ..baselines import attention_vc, prepare_vc

    grouping = prepare_vc(q, k, v).permutation
    calls = {
        "fresh_grouping": lambda: attention_vc(prepare_vc(q, k, v), output_shape=q.shape),
        "reused_grouping": lambda: attention_vc(
            prepare_vc(q, k, v, permutation=grouping, check_permutation=False),
            output_shape=q.shape,
        ),
    }
    times = {name: [] for name in calls}
    for _ in range(samples):
        for name, call in calls.items():
            times[name].append(_measure(call, 1))
    return times


def _device_metadata(uuid):
    fields = ["driver_version", "pstate", "clocks.sm", "clocks.mem", "power.limit"]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                uuid,
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        values = [x.strip() for x in result.stdout.strip().split(",")]
        if len(values) != len(fields):
            raise ValueError("Unexpected GPU metadata columns")
        return dict(zip(fields, values))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"unavailable": type(exc).__name__}


def _source_revision(root):
    """Git commit and cleanliness of the checkout the package was imported from, if any.

    A wheel install has no checkout; its identity is then the per-file hashes alone.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.split()
        dirty = subprocess.run(
            ["git", "-C", top[0], "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, IndexError, subprocess.SubprocessError):
        return None
    return {"commit": top[1], "dirty": bool(dirty)}


def _runtime_provenance(root=None):
    """Hash every Python source of the imported package, kernels included.

    ``package_sha256`` digests the sorted (path, file hash) pairs, so any change to the
    attention kernel, its dispatch and tuning, V-Smooth or the wrappers changes it.
    """
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    files = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(f"{name}\0{value}\n".encode())
    return {
        "package_sha256": digest.hexdigest(),
        "source_sha256": files,
        "source_revision": _source_revision(root),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["4096x7x128"])
    parser.add_argument("--preset", choices=["video", "smoke"])
    parser.add_argument(
        "--backends", nargs="+", choices=list(BACKENDS), default=list(DEFAULT_BACKENDS)
    )
    parser.add_argument("--baseline", choices=list(BACKENDS), default=DEFAULT_BASELINE)
    parser.add_argument("--scope", choices=["attention", "quantize-attention"], default="attention")
    parser.add_argument(
        "--repair-budget",
        type=float,
        default=0.0,
        help="Optional Open-VC V repair fraction per head; B200 long single sequence only",
    )
    parser.add_argument(
        "--preparation",
        choices=["auto", "unfused", "fused"],
        default="auto",
        help="Open-VC preparation for quantize-attention scope without V repair",
    )
    parser.add_argument("--timing", choices=["events", "graph"], default="events")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warm-calls", type=int, default=6)
    parser.add_argument("--warm-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--input", type=Path, help="Optional tensor-only .pt mapping with q/k/v; never uploaded"
    )
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Disable PID guard; marks results non-isolated",
    )
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args(argv)
    if (
        min(args.rounds, args.repeats) < 1
        or args.warm_calls < 0
        or args.warm_seconds < 0
        or not math.isfinite(args.warm_seconds)
    ):
        parser.error("Rounds/repeats must be positive; warmup must be nonnegative")
    if len(set(args.backends)) != len(args.backends):
        parser.error("Duplicate backend names")
    if args.baseline not in args.backends:
        parser.error("The named baseline must be included in --backends")
    if not 0.0 <= args.repair_budget < 1.0:
        parser.error("--repair-budget must be finite and in [0, 1)")
    if args.repair_budget and CANDIDATE not in args.backends:
        parser.error(f"--repair-budget requires the {CANDIDATE} backend")
    if args.preparation != "auto" and (
        args.scope != "quantize-attention" or args.repair_budget or CANDIDATE not in args.backends
    ):
        parser.error(
            f"--preparation requires quantize-attention scope with {CANDIDATE} and no repair"
        )
    if args.device < 0:
        parser.error("--device must be a nonnegative visible CUDA device index")
    shapes = args.shapes
    if args.preset == "video":
        shapes = [f"{s}x{h}x128" for s in (32768, 73426, 188214) for h in (7, 56)]
    elif args.preset == "smoke":
        shapes = ["129x2x128", "1024x7x128"]
    try:
        shapes = [parse_shape(s) for s in shapes]
    except ValueError as exc:
        parser.error(str(exc))
    if args.repair_budget and any(s < 32768 for s, _, _ in shapes):
        parser.error("--repair-budget requires sequence lengths >= 32768")
    if "vc" in args.backends and any(s < 128 for s, _, _ in shapes):
        parser.error("The vc backend requires sequence lengths >= 128")
    import torch

    from ..api import attention, prepare_fp8

    if not torch.cuda.is_available():
        parser.error(
            "CUDA unavailable; install CUDA-enabled PyTorch and run open-vc-attn-check --smoke"
        )
    torch.cuda.set_device(args.device)
    torch.set_grad_enabled(False)
    guard = GPUOwnership(args.device, enabled=not args.allow_shared_gpu)
    versions = {}
    for package in (
        "open-vc-attn",
        "torch",
        "triton",
        "nvidia-cutlass-dsl",
        "quack-kernels",
        "flash-attn-4",
        "cuda-python",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed as a distribution"
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_provenance": _runtime_provenance(),
        "device_configuration_start": _device_metadata(guard.uuid),
        "status": "running",
        "baseline": args.baseline,
        "scope": args.scope,
        "timing": args.timing,
        "fp8_default_quantization": "Q/K per-block128; V per-head; E4M3 absmax/448",
        "backends": {name: BACKENDS[name].description for name in args.backends},
        # Attention scope packs V before timing on every path (see make_call).
        "v_packing_included": args.scope == "quantize-attention",
        "input_kind": "local_capture" if args.input else "synthetic",
        "seed": args.seed,
        "versions": versions,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cuda_runtime": torch.version.cuda,
        "isolation_checked": guard.enabled,
        "settings": {k: v for k, v in vars(args).items() if k not in ("input", "output")},
        "shapes": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output)

    try:
        guard.check()
        captured = (
            torch.load(args.input, map_location="cpu", weights_only=True) if args.input else None
        )
        if captured is not None:
            if not isinstance(captured, dict) or any(
                not isinstance(captured.get(x), torch.Tensor) for x in ("q", "k", "v")
            ):
                raise ValueError("--input requires a tensor-only q/k/v mapping")
            report["input_sha256"] = _sha(args.input)
        for s, h, d in shapes:
            torch.manual_seed(args.seed + s + h)
            if captured is None:
                q, k, v = [
                    torch.randn(s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)
                ]
            else:
                if any(tuple(captured[x].shape) != (s, h, d) for x in ("q", "k", "v")):
                    raise ValueError(
                        "Capture must match the exact requested shape; no implicit slicing"
                    )
                q, k, v = [
                    captured[x].to(device="cuda", dtype=torch.bfloat16).contiguous()
                    for x in ("q", "k", "v")
                ]
            prepared = prepare_fp8(q, k, v)
            ref = attention(q, k, v, version="reference", mode="bf16").clone()
            row = {
                "shape": [s, h, d],
                "accuracy_reference": "bf16",
                "accuracy": {},
                "backend_metadata": {},
                "orders": [],
                "timings": {},
                "speedups": {},
            }
            report["shapes"].append(row)
            calls, graphs, graph_outputs = {}, {}, {}
            for name in args.backends:
                guard.check()
                call, metadata = make_call(
                    name,
                    q,
                    k,
                    v,
                    prepared,
                    scope=args.scope,
                    repair_budget=args.repair_budget,
                    preparation=args.preparation,
                )
                row["backend_metadata"][name] = metadata
                output = call()
                row["accuracy"][name] = _accuracy(output, ref)
                if name == "vc":
                    row["vc_complete_call_events_ms"] = _measure_vc_grouping(q, k, v)
                torch.cuda.synchronize()
                deadline = time.monotonic() + args.warm_seconds
                while time.monotonic() < deadline:
                    call()
                    torch.cuda.synchronize()
                calls[name] = call
                print(f"READY {s}x{h}x{d} {name}", flush=True)
            del output, ref
            if args.timing == "graph":
                for name, fn in calls.items():
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_outputs[name] = fn()
                    graphs[name] = graph
                measured = {name: g.replay for name, g in graphs.items()}
            else:
                measured = calls
            samples = {name: [] for name in measured}
            rng = random.Random(args.seed + s + h)
            for rnd in range(args.rounds):
                order = list(measured)
                rng.shuffle(order)
                row["orders"].append(order)
                for name in order:
                    guard.check()
                    for _ in range(args.warm_calls):
                        measured[name]()
                    samples[name].append(_measure(measured[name], args.repeats))
                    guard.check()
                row["timings"] = {name: summarize(vv) for name, vv in samples.items()}
                row["speedups"] = {
                    name: speedup(samples[args.baseline], vv) for name, vv in samples.items()
                }
                print(f"ROUND {s}x{h} {rnd + 1}/{args.rounds}", flush=True)
                save()
            torch.cuda.synchronize()
            measured.clear()
            graphs.clear()
            graph_outputs.clear()
            for call in calls.values():
                if hasattr(call, "close"):
                    call.close()
            calls.clear()
            del q, k, v, prepared
            gc.collect()
            torch.cuda.empty_cache()
        report["device_configuration_end"] = _device_metadata(guard.uuid)
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["status"] = "complete"
        report["ownership_checks"] = guard.samples
    except BaseException as exc:
        report["status"] = "invalid"
        # Do not record local filesystem paths from arbitrary exception messages.
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        save()
    from .report import render

    print(render(report))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
