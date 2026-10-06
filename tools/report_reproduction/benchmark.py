"""Replay one prebuilt attention kernel; preparation and profiling are outside timing."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

from common import (
    CONFIG,
    REPORT_SHAPES,
    SMOKE_SHAPE,
    balanced_orders,
    environment,
    now,
    save,
    sha,
    summarize,
    tree_hashes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="32769x7x128")
    parser.add_argument(
        "--include-d",
        action="store_true",
        help="Explicitly run experimental D from an opt-in build",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warm-calls", type=int, default=2)
    parser.add_argument("--warm-seconds", type=float, default=1)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Disable exclusive PID checks; results cannot establish isolated performance",
    )
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.split("x")))
    if shape not in (*REPORT_SHAPES, SMOKE_SHAPE):
        parser.error("Choose the smoke shape or one of the four report shapes")
    if args.output.exists():
        parser.error("Fresh output path required; retain interrupted attempts separately")
    if (
        args.repeats < 1
        or args.warm_calls < 0
        or not math.isfinite(args.warm_seconds)
        or args.warm_seconds < 0
    ):
        parser.error("Invalid repeat/warmup settings")
    labels = ["pre_fusedpipe", "fusedpipe", "bf16"] + (["d"] if args.include_d else [])
    orders = balanced_orders(labels, args.rounds, args.seed)
    build = args.build.resolve()
    manifest = json.loads((build / "manifest.json").read_text())
    if manifest.get("schema") != "vc-report-aot-v1" or manifest.get("status") != "complete":
        parser.error("Build manifest is incomplete or invalid")
    if args.include_d and not manifest["d_enabled"]:
        parser.error("Rebuild with --include-d to run the experimental fourth route")
    source = build / "source/vc_attn"
    if tree_hashes(source) != manifest["source"]["staged_files"]:
        parser.error("Staged source changed; rebuild")
    sys.path.insert(0, str(source.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "idle_benchmark"))
    sys.argv[:] = [sys.argv[0]]
    os.environ["OMP_NUM_THREADS"] = "1"
    import cutlass.cute as cute
    import torch
    from cuda.bindings import driver as cuda
    from key_format import decode, encode
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    from vc_attn.api import _layout, interface, prepare_fp8, raw_forward
    from vc_attn.benchmark import GPUOwnership, _device_metadata

    for package, version in manifest["environment"]["packages"].items():
        if version is not None and importlib.metadata.version(package) != version:
            raise RuntimeError("Build/runtime dependency mismatch: " + package)
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("Report reproduction requires B200 / SM100")
    guard = GPUOwnership(args.device, enabled=not args.allow_shared_gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema="vc-report-replay-v1",
        status="running",
        started_at=now(),
        shape=list(shape),
        gpu=torch.cuda.get_device_name(),
        seed=args.seed,
        configuration=CONFIG,
        build_manifest_sha256=sha(build / "manifest.json"),
        harness_files=tree_hashes(Path(__file__).resolve().parent),
        source=manifest["source"],
        environment=environment(),
        build_environment=manifest["environment"],
        isolation_checked=not args.allow_shared_gpu,
        d_enabled=args.include_d,
        settings={k: getattr(args, k) for k in ("rounds", "repeats", "warm_calls", "warm_seconds")},
        scope="One prebuilt attention kernel per CUDA Graph replay",
        excluded=[
            "input generation",
            "quantization",
            "V packing",
            "allocation",
            "compilation",
            "validation",
            "profiling",
            "model execution",
            "communication",
        ],
        byte_checks=[],
        single_kernel={},
        raw_samples_ms={k: [] for k in labels},
        orders=[],
    )
    modules, captured = [], {}

    def require_equal(label, actual, expected):
        valid = bool(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
        report["byte_checks"].append({"label": label, "equal": valid, "elements": actual.numel()})
        if not valid:
            raise RuntimeError("Complete output byte equality failed: " + label)

    def load_record(record):
        library = build / "artifacts" / record["library"]
        if library.parent != build / "artifacts" or sha(library) != record["sha256"]:
            raise RuntimeError("Artifact identity mismatch")
        module = cute.runtime.load_module(str(library), enable_tvm_ffi=True)
        modules.append(module)
        wrapped = make_kwargs_wrapper(
            getattr(module, record["name"]),
            arg_names=record["arg_names"],
            arg_defaults=tuple(record["arg_defaults"]),
            map_dataclass_to_tuple=record["dataclass_names"],
        )
        if record["fp8"]:

            def typed(*values, _fn=wrapped):
                return _fn(*(tuple(t.view(torch.float8_e4m3fn) for t in values[:3]) + values[3:]))

            return typed
        return wrapped

    def graph_call(label, fn, values, record, output, expected):
        values = list(values)
        stream_index = record["arg_names"].index("stream")

        def direct():
            values[stream_index] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            fn(*values)

        output.fill_(float("nan"))
        direct()
        torch.cuda.synchronize()
        require_equal(label + "/direct", output, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            direct()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        require_equal(label + "/graph", output, expected)
        captured[label] = dict(
            graph=graph,
            output=output,
            expected=expected,
            record=record,
            fn=fn,
            values=values,
            direct=direct,
        )

    try:
        guard.check()
        report["device_start"] = _device_metadata(guard.uuid)
        save(args.output, report)
        # Inputs exactly follow the archived CPU BF16 generation protocol.
        cpu = []
        report["input_sha256"] = {}
        for offset, name in enumerate(("q", "k", "v")):
            x = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(args.seed + offset),
            )
            digest = hashlib.sha256()
            for chunk in x.split(1024):
                digest.update(chunk.contiguous().view(torch.uint8).numpy().tobytes())
            report["input_sha256"][name] = digest.hexdigest()
            cpu.append(x)
        archive_path = (
            Path(__file__).resolve().parents[2]
            / "docs/reports/VC-Attention-Long-Sequence-Results.json"
        )
        archive = json.loads(archive_path.read_text())
        expected_rows = [
            r
            for r in archive["rows"]
            if r["shape"] == list(shape) and r["input"]["seed"] == args.seed
        ]
        if expected_rows and report["input_sha256"] != expected_rows[0]["input"]["qkv_sha256"]:
            raise RuntimeError("Generated inputs differ from the archived hashes")
        report["archived_input_hashes_match"] = True if expected_rows else None
        for record in manifest["records"]:
            interface(record["version"])._flash_attn_fwd.compile_cache[
                decode(record["cache_key"])
            ] = load_record(record)

        def reject_compile(*values, **kwargs):
            raise RuntimeError("Attention cache miss; rebuild the CPU artifacts")

        cute.compile = reject_compile
        q, k, v = [x.to("cuda").contiguous() for x in cpu]
        del cpu, x
        prep = prepare_fp8(q, k, v)
        qd = prep.q_descale
        if qd.shape[-1] % 2:
            qd = torch.nn.functional.pad(qd, (0, 1), value=1.0)
        scales = dict(q_descale=qd, k_descale=prep.k_descale, v_descale=prep.v_descale)
        for label, version in (
            ("pre_fusedpipe", "v4"),
            ("fusedpipe", "fusedpipe"),
            ("bf16", "baseline"),
        ):
            guard.check()
            cache = interface(version)._flash_attn_fwd.compile_cache
            original, observed = dict(cache), []

            def recorder(key, fn):
                def call(*values):
                    observed.append((key, fn, values))
                    return fn(*values)

                return call

            try:
                cache.update({key: recorder(key, fn) for key, fn in original.items()})
                if version == "baseline":
                    qb, kb, vb, layout, _ = _layout(q, k, v)
                    output = raw_forward(qb, kb, vb, version=version, **layout, return_lse=False)[0]
                else:
                    output = raw_forward(
                        prep.q, prep.k, prep.v, version=version, **prep.layout, **scales, **CONFIG
                    )[0]
                torch.cuda.synchronize()
            finally:
                cache.clear()
                cache.update(original)
            if len(observed) != 1:
                raise RuntimeError("Expected exactly one native attention callable")
            key, fn, values = observed[0]
            records = [
                r
                for r in manifest["records"]
                if r["version"] == version and r["cache_key"] == encode(key)
            ]
            if len(records) != 1:
                raise RuntimeError("Native callable does not match build manifest")
            graph_call(label, fn, values, records[0], output, output.clone())
        if args.include_d:
            control = captured["fusedpipe"]
            record = {**control["record"], **control["record"]["d"]}
            fn = load_record(record)
            values = list(control["values"])
            index = record["arg_names"].index("mO")
            values[index] = torch.empty_like(values[index])
            output = values[index].view_as(control["output"])
            graph_call("d", fn, values, record, output, control["expected"])
        for label, state in captured.items():
            guard.check()
            if not bool(torch.isfinite(state["output"]).all()):
                raise RuntimeError("Nonfinite output: " + label)
            if label != "bf16":
                require_equal(
                    label + "/pre_fusedpipe", state["output"], captured["pre_fusedpipe"]["expected"]
                )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                state["graph"].replay()
                torch.cuda.synchronize()
            kernels = [
                event
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            if len(kernels) != 1 or "flash" not in kernels[0].name.lower():
                raise RuntimeError("Graph must contain exactly one attention kernel: " + label)
            report["single_kernel"][label] = kernels[0].name
        report["launches"] = {label: state["record"] for label, state in captured.items()}
        save(args.output, report)
        print("CORRECTNESS_AND_SINGLE_KERNEL_PASS", flush=True)
        for state in captured.values():
            until = time.monotonic() + args.warm_seconds
            while time.monotonic() < until:
                guard.check()
                state["graph"].replay()
                torch.cuda.synchronize()
        for order in orders:
            for label in order:
                guard.check()
                graph = captured[label]["graph"]
                for _ in range(args.warm_calls):
                    graph.replay()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(args.repeats):
                    graph.replay()
                end.record()
                end.synchronize()
                guard.check()
                report["raw_samples_ms"][label].append(start.elapsed_time(end) / args.repeats)
            report["orders"].append(order)
            save(args.output, report)
            print("ROUND", len(report["orders"]), flush=True)
        for label, state in captured.items():
            state["output"].fill_(float("nan"))
            state["graph"].replay()
            torch.cuda.synchronize()
            require_equal(label + "/post-timing", state["output"], state["expected"])
        if (
            tree_hashes(source) != manifest["source"]["staged_files"]
            or sha(build / "manifest.json") != report["build_manifest_sha256"]
        ):
            raise RuntimeError("Source or build identity changed during replay")
        for record in manifest["records"]:
            for artifact in (record, *([record["d"]] if args.include_d and "d" in record else [])):
                if sha(build / "artifacts" / artifact["library"]) != artifact["sha256"]:
                    raise RuntimeError("AOT library changed during replay")
        report["device_end"] = _device_metadata(guard.uuid)
        guard.check()
        report.update(
            summarize(report["raw_samples_ms"]),
            status="complete",
            completed_at=now(),
            ownership_checks=guard.samples,
            guard_failure=None,
        )
        print("COMPLETE", args.output, flush=True)
    except BaseException as exc:
        report.update(
            status="invalid", error=f"{type(exc).__name__}: {exc}", partial_timings_accepted=False
        )
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()
