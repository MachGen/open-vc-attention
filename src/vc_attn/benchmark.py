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

from .measurement import parse_shape, speedup, summarize
from .registry import BACKENDS, DEFAULT_BACKENDS, get_backend, source_manifest


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


def make_call(name, q, k, v, prepared, *, scope, native_library=None):
    import torch

    from .api import (
        _layout,
        attention,
        attention_fp8,
        prepare_v_smooth,
        quantize_nvfp4,
        raw_forward,
    )

    spec = get_backend(name)
    if spec.version == "native":
        if scope != "attention" or not native_library:
            raise ValueError("Native backends need --native-library and --scope attention")
        from .native_plan import NativePlan

        plan = NativePlan(prepared, native_library)
        return plan, {"native_library_sha256": _sha(native_library), "source_variant": name}
    if spec.version == "torch":

        def call():
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
            )[0].transpose(0, 1)

        return call, {"implementation": "torch SDPA automatic dispatch"}
    if spec.version == "upstream":
        import flash_attn.cute.interface as upstream

        fn = getattr(upstream, "flash_attn_fwd", None) or upstream._flash_attn_fwd
        q0, k0, v0, layout, _ = _layout(q, k, v)
        return lambda: fn(q0, k0, v0, **layout, return_lse=False)[0], {
            "interface_sha256": _sha(upstream.__file__),
            "package_version": importlib.metadata.version("flash-attn-4"),
        }
    mode = spec.mode
    if mode in ("vsmooth", "nvfp4"):

        def prepare_special():
            if mode == "vsmooth":
                p = prepare_v_smooth(q, k, v, version=spec.version)
                return (p.q, p.k, p.v), {**prepared.layout, **p.forward_kwargs()}
            qn, sq = quantize_nvfp4(q[None], version=spec.version)
            kn, sk = quantize_nvfp4(k[None], version=spec.version)
            # NVFP4 uses fixed-length layout and its own Q/K scales.
            return (qn, kn, v[None].to(torch.float8_e4m3fn)), {"mSFQ": sq, "mSFK": sk}

        tensors, kwargs = prepare_special()

        def call_special():
            x, kw = prepare_special() if scope == "quantize-attention" else (tensors, kwargs)
            return raw_forward(
                *x, version=spec.version, expcast=True, mid_window_blocks=4, return_lse=False, **kw
            )[0].reshape(q.shape)

        return call_special, {
            "experimental": True,
            "quantization": "NVFP4 Q/K per16, unscaled E4M3 V"
            if mode == "nvfp4"
            else "V-Smooth grouping and residual quantization",
        }
    mid = 4 if mode == "expcast_mid4" else None
    actual_mode = "expcast" if mode == "expcast_mid4" else mode
    metadata = {"mid_window_blocks": mid, "sass_d_enabled": False}
    if scope == "attention" and mode == "bf16":
        q0, k0, v0, layout, _ = _layout(q, k, v)
        return lambda: raw_forward(q0, k0, v0, version=spec.version, **layout, return_lse=False)[
            0
        ], metadata
    if scope == "quantize-attention" or mode == "bf16":
        return lambda: attention(
            q, k, v, version=spec.version, mode=actual_mode, mid_window_blocks=mid
        ), metadata
    return lambda: attention_fp8(
        prepared, version=spec.version, expcast=actual_mode == "expcast", mid_window_blocks=mid
    ), metadata


def _measure(fn, repeats):
    import torch

    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


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


def _runtime_provenance():
    root = Path(__file__).parent
    names = (
        "api.py",
        "benchmark.py",
        "native_plan.py",
        "quantization.py",
        "measurement.py",
        "registry.py",
        "_sass_runtime.py",
        "_sass_d.py",
    )
    return {
        "manifest_sha256": _sha(root / "source_manifest.json"),
        "runtime_sha256": {name: _sha(root / name) for name in names},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["4096x7x128"])
    parser.add_argument("--preset", choices=["minimax", "smoke"])
    parser.add_argument(
        "--backends", nargs="+", choices=list(BACKENDS), default=list(DEFAULT_BACKENDS)
    )
    parser.add_argument("--baseline", choices=list(BACKENDS), default="fp8_ref")
    parser.add_argument("--scope", choices=["attention", "quantize-attention"], default="attention")
    parser.add_argument("--timing", choices=["events", "graph"], default="events")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warm-calls", type=int, default=6)
    parser.add_argument("--warm-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--native-library")
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
    if args.device < 0:
        parser.error("--device must be a nonnegative visible CUDA device index")
    if "native_v6" in args.backends:
        if args.scope != "attention":
            parser.error("Native v6 supports --scope attention only")
        if not args.native_library or not Path(args.native_library).is_file():
            parser.error(
                "Build the B300 native library, then pass --native-library path/to/library.so"
            )
    if (
        args.timing == "graph"
        and args.scope == "quantize-attention"
        and set(args.backends) & {"vc_vsmooth", "vc_nvfp4"}
    ):
        parser.error(
            "Frozen V-Smooth/NVFP4 preparation is not graph-capturable; use --timing events "
            "or --scope attention for these experimental backends"
        )
    shapes = args.shapes
    if args.preset == "minimax":
        shapes = [f"{s}x{h}x128" for s in (32768, 73426, 188214) for h in (7, 56)]
    elif args.preset == "smoke":
        shapes = ["129x2x128", "1024x7x128"]
    try:
        shapes = [parse_shape(s) for s in shapes]
    except ValueError as exc:
        parser.error(str(exc))
    import torch

    from .api import attention, prepare_fp8

    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; install CUDA-enabled PyTorch and run vc-attn-check --smoke")
    torch.cuda.set_device(args.device)
    if "native_v6" in args.backends and torch.cuda.get_device_capability(args.device) != (10, 3):
        parser.error("native_v6 requires a B300 / SM103 GPU")
    torch.set_grad_enabled(False)
    guard = GPUOwnership(args.device, enabled=not args.allow_shared_gpu)
    versions = {}
    for package in (
        "vc-attention",
        "torch",
        "triton",
        "nvidia-cutlass-dsl",
        "quack-kernels",
        "cuda-python",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed as a distribution"
    manifest = source_manifest()
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
        "v_packing_included": True,
        "input_kind": "local_capture" if args.input else "synthetic",
        "seed": args.seed,
        "versions": versions,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cuda_runtime": torch.version.cuda,
        "isolation_checked": guard.enabled,
        "source_revisions": {k: v["source_revision"] for k, v in manifest["versions"].items()},
        "settings": {
            k: v for k, v in vars(args).items() if k not in ("input", "output", "native_library")
        },
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
            ref = attention(q, k, v, version="baseline", mode="bf16").clone()
            row = {
                "shape": [s, h, d],
                "accuracy_reference": "bf16_ref",
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
                    name, q, k, v, prepared, scope=args.scope, native_library=args.native_library
                )
                row["backend_metadata"][name] = metadata
                output = call()
                row["accuracy"][name] = _accuracy(output, ref)
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
