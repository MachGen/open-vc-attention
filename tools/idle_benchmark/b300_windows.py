# ruff: noqa: E402
"""Bounded B300 paired sampling; keeps accepted rounds while waiting for idle."""

import argparse
import gc
import hashlib
import importlib.abc
import importlib.metadata
import json
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml as nv

p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--minutes", type=int, default=45)
p.add_argument("--rounds", type=int, default=12)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--warm-calls", type=int, default=2)
p.add_argument(
    "--shapes", nargs="+", default=["188214x7x128", "188214x56x128", "73426x7x128", "73426x56x128"]
)
p.add_argument("--output", default="results/b300-windows.json")
args = p.parse_args()
for name in ("minutes", "rounds", "repeats"):
    if getattr(args, name) <= 0:
        p.error(f"--{name} must be positive")
if args.warm_calls < 0 or args.gpu < 0:
    p.error("--warm-calls and --gpu must be non-negative")
supported = {"188214x7x128", "188214x56x128", "73426x7x128", "73426x56x128", "129x7x128"}
if not set(args.shapes) <= supported or len(args.shapes) != len(set(args.shapes)):
    p.error("Choose distinct precompiled shapes: " + ", ".join(sorted(supported)))
sys.argv = [sys.argv[0]]
root = Path(os.environ.get("VC_ATTN_WORKDIR", "build/idle-benchmark")).resolve()
root.mkdir(parents=True, exist_ok=True)
out = root / args.output
if out.exists():
    p.error(f"Output already exists: {out}; choose a new --output")
nv.nvmlInit()
handle = nv.nvmlDeviceGetHandleByIndex(args.gpu)
uuid = nv.nvmlDeviceGetUUID(handle)
os.environ.update(
    CUDA_VISIBLE_DEVICES=uuid,
    CUTE_DSL_ARCH="sm_103a",
    TRITON_CACHE_DIR=str(root / "cache/triton"),
    CUTE_DSL_CACHE_DIR=str(root / "cache/cute"),
    OMP_NUM_THREADS="4",
)
os.environ.pop("PYTHONPATH", None)


class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("machgen", "machgen_kernels", "flash_attention_plus"):
            raise RuntimeError("Forbidden application import")


sys.meta_path.insert(0, Boundary())

import torch
from aot_load import load

from vc_attn.api import attention, prepare_fp8
from vc_attn.benchmark import _accuracy, _device_metadata, _runtime_provenance, make_call
from vc_attn.measurement import parse_shape, speedup, summarize
from vc_attn.registry import source_manifest

torch.cuda.set_device(0)
if torch.cuda.get_device_capability() != (10, 3):
    raise RuntimeError("This AOT helper requires B300 / SM103")
torch.set_grad_enabled(False)
torch.set_num_threads(4)
artifact_manifest = load(root / "aot-sm103")
print(
    "AOT_LOADED", len(artifact_manifest["records"]), "GPU", args.gpu, "PID", os.getpid(), flush=True
)
deadline = time.monotonic() + 60 * args.minutes
backends = ["bf16_ref", "fp8_ref", "native_v6", "vc_scaled"]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def foreign():
    return {x.pid for x in nv.nvmlDeviceGetComputeRunningProcesses(handle)} - {os.getpid()}


def activity(start, end=None):
    try:
        xs = nv.nvmlDeviceGetProcessUtilization(handle, int(start * 1e6))
    except nv.NVMLError_NotFound:
        xs = []
    return [
        {"timestamp": x.timeStamp, "sm": x.smUtil, "mem": x.memUtil}
        for x in xs
        if x.pid != os.getpid()
        and (x.smUtil or x.memUtil)
        and (end is None or x.timeStamp <= int(end * 1e6))
    ]


class Overlap(Exception):
    pass


class Timeout(Exception):
    pass


def wait_idle():
    stable = None
    next_log = time.monotonic() + 30
    while time.monotonic() < deadline:
        busy = bool(activity(time.time() - 2)) or nv.nvmlDeviceGetUtilizationRates(handle).gpu > 0
        if busy:
            stable = None
        else:
            stable = stable or time.monotonic()
        if stable and time.monotonic() - stable >= 2:
            return
        if time.monotonic() > next_log:
            print("WAIT_IDLE", stamp(), flush=True)
            next_log = time.monotonic() + 30
        time.sleep(0.2)
    raise Timeout("Wall-clock budget reached")


class Window:
    def __enter__(self):
        wait_idle()
        self.begin = time.time()
        self.pids = foreign()
        self.stop = threading.Event()
        self.bad = threading.Event()
        self.reason = None

        def monitor():
            try:
                while not self.stop.wait(0.1):
                    if activity(self.begin):
                        self.reason = "foreign GPU activity"
                        self.bad.set()
                        return
                    if foreign() != self.pids:
                        self.reason = "foreign process set changed"
                        self.bad.set()
                        return
            except BaseException:
                self.reason = "monitor failed"
                self.bad.set()

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def check(self):
        if self.bad.is_set():
            raise Overlap(self.reason)
        if time.monotonic() >= deadline:
            raise Timeout("Wall-clock budget reached")

    def gpu(self, fn):
        self.check()
        value = fn()
        torch.cuda.synchronize()
        self.check()
        return value

    def __exit__(self, kind, exc, tb):
        torch.cuda.synchronize()
        self.end = time.time()
        self.stop.set()
        self.thread.join()
        # Wait for delayed process samples; include a conservative 1-second tail.
        if kind is None:
            time.sleep(4)
            late = activity(self.begin, self.end + 1)
            if self.bad.is_set() or late or foreign() != self.pids:
                raise Overlap(self.reason or "delayed foreign activity")


report = {
    "schema_version": 1,
    "status": "running",
    "started_at_utc": stamp(),
    "baseline": "fp8_ref",
    "scope": "attention",
    "timing": "events",
    "gpu": torch.cuda.get_device_name(),
    "capability": [10, 3],
    "cuda_runtime": torch.version.cuda,
    "isolation_checked": False,
    "input_kind": "synthetic",
    "seed": 20260928,
    "fp8_default_quantization": "Q/K per-block128; V per-head; E4M3 absmax/448",
    "v_packing_included": True,
    "versions": {
        n: importlib.metadata.version(n)
        for n in [
            "vc-attention",
            "torch",
            "triton",
            "nvidia-cutlass-dsl",
            "quack-kernels",
            "cuda-python",
        ]
    },
    "source_provenance": _runtime_provenance(),
    "source_revisions": {k: v["source_revision"] for k, v in source_manifest()["versions"].items()},
    "device_configuration_start": _device_metadata(uuid),
    "settings": {
        "rounds": args.rounds,
        "repeats": args.repeats,
        "warm_calls": args.warm_calls,
        "warm_seconds": 0,
        "backends": backends,
        "aot": True,
    },
    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "aot_artifacts": [
        {"name": r["name"], "sha256": r["sha256"], "version": r["version"]}
        for r in artifact_manifest["records"]
    ],
    "guard_protocol": {
        "isolation": "shared_gpu_idle_windows",
        "fixed_gpu": True,
        "gpu_uuid_sha256": hashlib.sha256(uuid.encode()).hexdigest(),
        "scope": "selected physical GPU",
        "precheck": "2 seconds stable idle after trailing 2-second utilization check",
        "monitor_seconds": 0.1,
        "delayed_check_seconds": 4,
        "sample_tail_seconds": 1,
        "action_on_interference": "stop enqueueing; discard complete paired round; wait for idle",
        "resident_models": True,
    },
    "rejections": [],
    "shapes": [],
}


def save():
    out.parent.mkdir(exist_ok=True, parents=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(out)


def reject(phase, reason):
    report["rejections"].append({"at_utc": stamp(), "phase": phase, "reason": str(reason)})
    print("REJECT", phase, str(reason), flush=True)
    save()


def guarded(phase, fn):
    while True:
        try:
            with Window() as w:
                value = fn(w)
            return value
        except Overlap as exc:
            reject(phase, exc)


def run():
    calls = {}
    try:
        # Check AOT outputs against a small FP32 dense oracle before timing.
        def validate(w):
            torch.manual_seed(101)
            q, k, v = w.gpu(
                lambda: [
                    torch.randn(129, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
                ]
            )
            prepared = w.gpu(lambda: prepare_fp8(q, k, v))
            ref = w.gpu(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q.float().transpose(0, 1)[None],
                    k.float().transpose(0, 1)[None],
                    v.float().transpose(0, 1)[None],
                )[0].transpose(0, 1)
            )
            results = {}
            for name in backends:
                fn, _ = w.gpu(
                    lambda: make_call(
                        name,
                        q,
                        k,
                        v,
                        prepared,
                        scope="attention",
                        native_library=root / "libvc_native_v6.so",
                    )
                )
                try:
                    results[name] = w.gpu(lambda: _accuracy(fn(), ref))
                finally:
                    if hasattr(fn, "close"):
                        fn.close()
                assert results[name]["relative_l2"] < (0.02 if name == "bf16_ref" else 0.12), (
                    results[name]
                )
            return results

        report["aot_validation"] = guarded("small-correctness", validate)
        save()
        print("AOT_VALIDATED", report["aot_validation"], flush=True)
        for shape in args.shapes:
            s, h, d = parse_shape(shape)
            if nv.nvmlDeviceGetMemoryInfo(handle).free < 80 * 2**30:
                raise RuntimeError("Insufficient free HBM")

            def prepare(w):
                torch.manual_seed(20260928 + s + h)
                tensors = w.gpu(
                    lambda: [
                        torch.randn(s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)
                    ]
                )
                prepared = w.gpu(lambda: prepare_fp8(*tensors))
                return tensors, prepared

            tensors, prepared = guarded(shape + "/prepare", prepare)
            q, k, v = tensors
            row = {
                "shape": [s, h, d],
                "accuracy_reference": "bf16_ref",
                "accuracy": {},
                "backend_metadata": {},
                "orders": [],
                "timings": {},
                "speedups": {},
                "accepted_windows": [],
                "completed": False,
            }
            report["shapes"].append(row)
            save()
            ref = guarded(
                shape + "/reference",
                lambda w: w.gpu(
                    lambda: attention(q, k, v, version="baseline", mode="bf16").clone()
                ),
            )
            for name in backends:
                fn, meta = guarded(
                    shape + "/" + name + "/setup",
                    lambda w: w.gpu(
                        lambda: make_call(
                            name,
                            q,
                            k,
                            v,
                            prepared,
                            scope="attention",
                            native_library=root / "libvc_native_v6.so",
                        )
                    ),
                )
                calls[name] = fn
                row["backend_metadata"][name] = meta
                row["accuracy"][name] = guarded(
                    shape + "/" + name + "/accuracy", lambda w: w.gpu(lambda: _accuracy(fn(), ref))
                )
                print("READY", shape, name, flush=True)
                save()
            ref = None
            samples = {name: [] for name in backends}
            rng = random.Random(20260928 + s + h)
            for rnd in range(args.rounds):
                order = list(backends)
                rng.shuffle(order)
                while True:
                    try:
                        pair = {}
                        with Window() as w:
                            for name in order:
                                for _ in range(args.warm_calls):
                                    w.gpu(calls[name])
                                start, end = [
                                    torch.cuda.Event(enable_timing=True) for _ in range(2)
                                ]
                                w.check()
                                start.record()
                                for _ in range(args.repeats):
                                    w.check()
                                    calls[name]()
                                end.record()
                                end.synchronize()
                                w.check()
                                pair[name] = start.elapsed_time(end) / args.repeats
                        break
                    except Overlap as exc:
                        reject(shape + "/round" + str(rnd + 1), exc)
                for name in backends:
                    samples[name].append(pair[name])
                row["orders"].append(order)
                row["accepted_windows"].append(
                    {"started_at": w.begin, "ended_at": w.end, "guard_passed": True}
                )
                row["timings"] = {k: summarize(vs) for k, vs in samples.items()}
                row["speedups"] = {k: speedup(samples["fp8_ref"], vs) for k, vs in samples.items()}
                print("ACCEPT", shape, rnd + 1, args.rounds, pair, flush=True)
                save()
            row["completed"] = True
            for fn in calls.values():
                if hasattr(fn, "close"):
                    fn.close()
            calls.clear()
            fn = q = k = v = prepared = tensors = None
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report["completed_at_utc"] = stamp()
        report["status"] = "complete"
    except Timeout as exc:
        report["status"] = "partial"
        report["stop_reason"] = str(exc)
    except BaseException as exc:
        report["status"] = "invalid"
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        for fn in calls.values():
            if hasattr(fn, "close"):
                fn.close()
        report["ended_at_utc"] = stamp()
        report["device_configuration_end"] = _device_metadata(uuid)
        report["platform_imports"] = [
            n
            for n in sys.modules
            if n.split(".")[0] in ("machgen", "machgen_kernels", "flash_attention_plus")
        ]
        save()
        print("FINISHED", report["status"], flush=True)


if __name__ == "__main__":
    run()
    if report["status"] != "complete":
        raise SystemExit(2)
