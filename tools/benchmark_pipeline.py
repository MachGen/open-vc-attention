"""Measure warm synchronous SGLang generation with explicit backend identity."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from vc_attn.measurement import summarize
from vc_attn.registry import DEFAULT_VERSION


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def workload_identity(config):
    common = copy.deepcopy(config)
    common["server_args"].pop("attention_backend", None)
    return fingerprint(common)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--backend", choices=["fa", "vc_attn"], required=True)
    p.add_argument("--vc-version", default=DEFAULT_VERSION)
    p.add_argument("--vc-mode", choices=["bf16", "fp8", "expcast"], default="expcast")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--baseline-json", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        p.error("warmup must be nonnegative and repeats positive")
    import torch

    from vc_attn.registry import VERSIONS

    if args.vc_version not in VERSIONS:
        p.error(f"Choose --vc-version from {VERSIONS}")
    config = json.loads(args.config.read_text())
    if set(config) != {"server_args", "sampling_params"}:
        p.error("config requires exactly server_args and sampling_params")
    config["server_args"]["attention_backend"] = args.backend
    if config["server_args"].get("ring_degree", 1) != 1:
        p.error("The VC adapter does not support ring attention")
    os.environ["VC_ATTN_VERSION"] = args.vc_version
    os.environ["VC_ATTN_MODE"] = args.vc_mode
    identity = workload_identity(config)
    baseline = None
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text())
        if baseline.get("status") != "complete" or baseline.get("workload_sha256") != identity:
            p.error("Baseline must be complete with identical non-attention settings")
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "warm synchronous generate wall time",
        "attention_backend": args.backend,
        "vc_version": args.vc_version,
        "vc_mode": args.vc_mode,
        "workload_sha256": identity,
        "warmup_count": args.warmup,
        "timings": {},
        "load_excluded": True,
        "encoding_and_output_io": "as configured in sampling_params",
        "sglang_version": importlib.metadata.version("sglang"),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ],
        },
    }
    generator = None
    try:
        from sglang.multimodal_gen import DiffGenerator

        started = time.perf_counter()
        generator = DiffGenerator.from_pretrained(
            server_args=config["server_args"], local_mode=True
        )
        report["load_seconds"] = time.perf_counter() - started
        samples = []
        for index in range(args.warmup + args.repeats):
            started = time.perf_counter()
            result = generator.generate(
                sampling_params_kwargs=copy.deepcopy(config["sampling_params"])
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            if result is None or (
                isinstance(result, list) and (not result or any(x is None for x in result))
            ):
                raise RuntimeError("Generation failed; reject this measurement")
            if index >= args.warmup:
                samples.append(elapsed_ms)
            print(
                f"{'WARMUP' if index < args.warmup else 'MEASURE'} {elapsed_ms:.3f} ms", flush=True
            )
        report["timings"] = summarize(samples)
        if baseline is not None:
            if baseline["sglang_version"] != report["sglang_version"]:
                raise ValueError("Baseline and candidate SGLang versions differ")
            if baseline.get("environment") != report["environment"]:
                raise ValueError("Baseline and candidate GPU/Torch/CUDA environments differ")
            report["baseline_backend"] = {
                k: baseline[k] for k in ("attention_backend", "vc_version", "vc_mode")
            }
            report["unpaired_ratio_of_medians"] = (
                baseline["timings"]["median_ms"] / report["timings"]["median_ms"]
            )
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "invalid"
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        try:
            if generator is not None:
                generator.shutdown()
        except BaseException as exc:
            report["status"] = "invalid"
            report["shutdown_failure_type"] = type(exc).__name__
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
