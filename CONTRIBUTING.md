# Contributing

Read README and `docs/architecture.md` before changing code. Keep public API, backend labels and timing boundaries consistent. Device-kernel modifications require Blackwell correctness validation; CPU CI alone does not validate CUDA execution. Preserve attribution and document source transformations.

```bash
python -m pip install --no-deps -e .
python -m pip install pytest ruff build 'setuptools>=77' wheel
ruff check src tests tools examples integrations/sglang/install.py
ruff format --check src tests tools examples integrations/sglang/install.py
python -m pytest tests/cpu
python tools/audit_release.py
python -m build
python -m pytest tests/packaging
# On a reserved Blackwell GPU with GPU dependencies installed:
python -m pytest tests/gpu
OPEN_VC_ATTN_TEST_LARGE=1 python -m pytest tests/gpu -m large
```

Publish only complete timing runs from a reserved GPU, with raw samples. When the benchmark records change, update the technical report's Markdown and PDF to match. Public artifacts must contain no private hosts, paths or user data.
