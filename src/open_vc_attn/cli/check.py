"""Check installation metadata; optionally run a tiny correctness smoke on an allocated GPU."""

import argparse
import importlib.metadata
import json
import platform
import sys

from .. import __version__

REQUIRED = (
    "torch",
    "triton",
    "nvidia-cutlass-dsl",
    "quack-kernels",
    "flash-attn-4",
    "cuda-python",
    "einops",
)
EXACT = {"nvidia-cutlass-dsl": "4.6.2", "quack-kernels": "0.6.4", "flash-attn-4": "4.0.0b33"}


def inspect_installation():
    versions, errors = {}, []
    for name in REQUIRED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            errors.append(f"Missing {name}; install this package with the documented constraints")
    for name, expected in EXACT.items():
        if versions[name] is not None and versions[name] != expected:
            errors.append(f"{name} must be {expected}; found {versions[name]}")
    if platform.system() != "Linux":
        errors.append("GPU execution requires Linux; CPU documentation/report tools work elsewhere")
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")
    return {
        "open_vc_attn": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "versions": versions,
        "errors": errors,
        "gpu_smoke": "not requested",
    }


def smoke(device):
    import torch

    from ..api import attention, attention_fp8, prepare_fp8
    from ..benchmarking.runner import GPUOwnership

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and CUDA-enabled PyTorch")
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(f"Requires B200 (SM100) or B300 (SM103); found {capability}")
    guard = GPUOwnership(device)
    guard.check()
    with torch.inference_mode():
        torch.manual_seed(1729)
        q, k, v = [torch.randn(129, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        out = attention(q, k, v)
        prepared = attention_fp8(prepare_fp8(q, k, v))
        torch.testing.assert_close(out, prepared, atol=0, rtol=0)
        qh, kh, vh = [t.float().transpose(0, 1) for t in (q, k, v)]
        ref = ((qh @ kh.transpose(-1, -2) / 128**0.5).softmax(-1) @ vh).transpose(0, 1)
        error = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        if not torch.isfinite(out).all() or error >= 0.10:
            raise RuntimeError(f"Tiny FP32-reference smoke failed: relative L2={error}")
        torch.cuda.synchronize()
    guard.check()
    return {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "cuda_runtime": torch.version.cuda,
        "shape": [129, 2, 128],
        "relative_l2_vs_fp32": error,
        "note": "Tiny operator smoke, not model-quality or performance validation",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Compile/run on a reserved B200/B300; first use can take minutes",
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    report = inspect_installation()
    if args.smoke and not report["errors"]:
        try:
            report["gpu_smoke"] = smoke(args.device)
        except Exception as exc:
            report["errors"].append(f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
