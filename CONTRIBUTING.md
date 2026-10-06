# Contributing

Keep changes within attention kernels, input preparation, integration adapters,
and measurement tooling. Do not add model weights, application services, private
captures, host addresses, authentication material or deployment configuration.

Follow [the clean installation guide](docs/installation.md) first.

```bash
python -m pip install -e '.[dev]' -c requirements-tested.txt
ruff check src/vc_attn tests tools examples
ruff format --check src/vc_attn tests tools examples
python -m pytest tests/cpu
python tools/audit_release.py
python -m build
```

On an exclusively allocated Blackwell GPU:

```bash
python -m pytest tests/test_gpu.py tests/test_v4_gpu.py
VC_ATTN_TEST_LARGE=1 python -m pytest tests/test_gpu.py tests/test_v4_gpu.py -m large
vc-attn-bench --preset smoke --output results/smoke.json
```

Frozen `_kernels` snapshots are excluded from formatting and lint rewrites.
Add a new version rather than changing an old numerical baseline. If a source
update is intentional, record original and transformed file hashes and explain
the import/algorithm differences. Do not regenerate hashes to hide an accidental
change. CPU tests cannot establish GPU kernel correctness or performance.

For performance changes, report hardware, software, inputs, baseline, precision,
timer boundary, raw paired samples and numerical differences. For model claims,
also report matched checkpoints, seeds, inference settings and output quality.

Contributions are under BSD-3-Clause unless a file carries another retained
third-party license. Preserve author attribution and third-party notices.
