# Installation and first run

Use Linux, Python 3.10 or newer (tested with 3.12), an NVIDIA B200 or B300,
and a driver compatible with CUDA 13. A CUDA Toolkit compiler is needed only
for the optional native v6 baseline. The CuTe API JIT-compiles on first use.

## 1. Install into a clean environment

From a clone or an unpacked source release:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[dev]' -c requirements-tested.txt
python -m pip check
vc-attn-check
```

`requirements-tested.txt` is a **constraints file**, not a list of packages to
install. `-c` pins selected dependencies while `.[dev]` chooses which packages
are installed. Its plain package names intentionally have no extras; the Cu13
extras are declared in `pyproject.toml`. Optional upstream FlashAttention is
only installed when you select `.[upstream]`.

For a deployment that does not need tests or editable source, replace `-e
'.[dev]'` with `.` or a release wheel. PyPI publication is separate from GitHub
releases; do not assume `pip install vc-attention` resolves this repository.

The metadata-only `vc-attn-check` does not import Torch or initialize a GPU.
It reports missing dependencies, an unsupported platform and incompatible
DSL/quack versions. A successful metadata check does not prove GPU execution.

## 2. Run a tiny correctness check

Reserve a GPU using your environment's scheduler. Select it through
`CUDA_VISIBLE_DEVICES`; device indices used by these commands are logical
indices within that visible set.

```bash
vc-attn-check --smoke
python examples/quickstart.py
```

The smoke uses `[129,2,128]` BF16 Q/K/V, checks the two API routes agree, and
compares VC against a small FP32 dense reference. It refuses a GPU with other
compute processes. Expect first-use compilation to take longer than subsequent
calls. Success prints `"status": "passed"`; the example prints
`torch.Size([4096, 7, 128]) torch.bfloat16`.

This checks a small operator case. It is not a performance measurement or
model-quality acceptance. For broader GPU coverage:

```bash
python -m pytest tests/cpu
VC_ATTN_TEST_LARGE=1 python -m pytest tests/test_gpu.py tests/test_v4_gpu.py
```

## 3. Measure performance

```bash
vc-attn-bench --preset smoke --output results/smoke.json
vc-attn-report results/smoke.json
vc-attn-bench --shapes 4096x8x128 32768x32x128 --output results/attention.json
```

The benchmark prints a table and saves all raw samples. Its default is
attention-only timing of fixed BF16, fixed ordinary FP8 and `vc_v4_mid4`:
the same dense scan as the public API, with eligible B200 fusedpipe and D disabled.
Use `--scope quantize-attention` to include input preparation. First-run
compilation and warmup are excluded. Larger shapes can take several minutes;
see [the protocol](benchmarking.md) before quoting performance numbers. The CLI
measures noncausal self-attention for one sequence; its shape list is independent
of any model or framework.

## 4. Optional B300 native baseline

Use **B300 / SM103**, CUDA Toolkit 13 and an allocated GPU:

```bash
# Set CUDA_HOME to your CUDA 13 toolkit if nvcc is not already on PATH.
python tools/build_native.py --arch sm_103a --output build/libvc_native_v6.so
VC_ATTN_NATIVE_LIBRARY="$PWD/build/libvc_native_v6.so" \
  python -m pytest tests/test_gpu.py -k native
vc-attn-bench --shapes 188214x7x128 \
  --backends bf16_ref fp8_ref native_v6 vc_scaled --baseline fp8_ref \
  --native-library build/libvc_native_v6.so --output results/b300.json
vc-attn-report results/b300.json --baseline native_v6
```

Native inputs are already quantized. The wrapper owns reusable output buffers;
plan construction is excluded from timing. A plan's next call overwrites its
previous output. Clone outputs you need to retain. The CuTe call includes any
automatic V packing and output allocation inside its call; both are timed as
implemented. Read [native sources](../src/vc_attn/native/README.md).

For failures, start with [Troubleshooting](troubleshooting.md). For model
integration, use [the API guide](integration.md) or [SGLang](sglang.md).

[Repository](../README.md) · [Documentation](README.md)
