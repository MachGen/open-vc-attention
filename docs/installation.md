# Installation

GPU execution requires Linux, NVIDIA Blackwell SM100 or SM103, Python >=3.10 and the CUDA 13 stack in `requirements-tested.txt`. CUDA-enabled PyTorch is a separate first step; do not install the constraints file with `pip -r`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -c requirements-tested.txt .
open-vc-attn-info
open-vc-attn-check
open-vc-attn-check --smoke
```

The smoke command requires a reserved GPU and rejects another compute PID. It checks a small operator against FP32; it does not certify all shapes or model quality. First invocation compiles CuTe/Triton kernels and can take minutes. Allow a writable compiler cache and warm all shapes before timing or CUDA graph capture.

`open-vc-attn-info`, CLI `--help`, report parsing and CPU tests do not import Torch or initialize CUDA. For CPU-only development, use `pip install --no-deps -e .` and install `pytest`, `ruff`, `build`, `setuptools>=77`, `wheel`. This is not a GPU installation.

A built wheel includes Python/CuTe source and compiles kernels at runtime. It does not bundle a platform-specific attention binary. Install a local wheel with `pip install -c requirements-tested.txt dist/open_vc_attn-*.whl`; use the same constraints and CUDA PyTorch installation.
