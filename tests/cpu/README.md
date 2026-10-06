# CPU contract tests

[test_contracts.py](test_contracts.py) is a `unittest` suite also collected by pytest.
It covers GPU UUID normalization, paired timing statistics, invalid shapes and
backend names, packed-sequence boundaries, lazy imports/CLI help, frozen kernel
hashes, and the absence of application namespace imports.

Run from the repository root after installation:

```bash
python -m pytest tests/cpu
# Standard-library runner:
python -m unittest discover -s tests/cpu -v
```

These tests need the source checkout for provenance checks but do not require
CUDA or import Torch through the public package initializer. GPU math is tested
separately in [test_gpu.py](../test_gpu.py). Use
[tools/audit_release.py](../../tools/audit_release.py) for the broader release boundary audit.

[Repository](../../README.md) · [Parent directory](../README.md)

[test_workflows.py](test_workflows.py) covers new-user diagnostics, invalid CLI
commands before CUDA initialization, and report integrity/denominator selection.

[test_defaults.py](test_defaults.py) checks agreement between public API and
benchmark scan options, preserved historical backend behavior, and that default
calls never inspect or export a D binary. Explicit experimental opt-in remains guarded.

[test_report_reproduction.py](test_report_reproduction.py) checks balanced paired
sampling, rejection of incomplete/failed evidence and isolated source-control
construction with an unknown-gate failure. GPU replay is validated separately.
