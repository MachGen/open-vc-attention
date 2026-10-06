# Reproducing the fusedpipe report

The release default is dense FP8 `mid_window_blocks=4`, with fusedpipe on
eligible B200 calls. **D is experimental and disabled by default.** This workflow
reconstructs the report's matched controls in a separate build directory. It does
not modify the installed package or its frozen source files.

## What is compared

| Result name | Construction | Scan |
|---|---|---|
| `pre_fusedpipe` | Copy current v4; disable only its `use_fusedpipe` source gate | mid4 |
| `fusedpipe` | Copy current v4 into a private namespace | mid4 |
| `bf16` | The vendored BF16 baseline, measured in the same run | Baseline scan |
| `d` (opt-in) | Exact guarded 10-slot SASS derivative of the fusedpipe object | mid4 |

Both FP8 source controls use identical descales, packing eligibility, tiles,
ExpCast and no skipping. The public backend name `vc_v4` retains its historical
original scan (`mid_window_blocks=None`); **it is not `pre_fusedpipe`**.
The historical H7 packing adaptation is already present in current v4.

The CPU builder records source hashes, transformation descriptions, exported
ABI/cache keys, native text and normalized executable metadata hashes, compiler
options and linked-library hashes. D has an independent exported function and
library: all associated host symbols are renamed before linking, and the exact
patched cubin is verified afterward. A failed D compatibility check aborts the
build; it never silently substitutes fusedpipe under a D label.

## Environment and setup

Use Linux x86-64, Python 3.12 and an NVIDIA B200 / SM100 GPU for replay. Other
architectures are rejected. CPU compilation needs the CUDA 13 CuTe DSL libraries,
GCC and GNU binutils (`nm`, `objcopy`); it does not initialize CUDA. The scripts
must be run from a source checkout or extracted sdist, not a wheel alone.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -c requirements-tested.txt -e .
python -m pip check
```

The verified runtime used Torch 2.11.0+cu130, CUDA runtime 13.0, CuTe DSL 4.6.0,
quack-kernels 0.6.1, Triton 3.6.0 and cuda-python 13.0.3. Check the generated
manifest, [historical environment](historical-environment.json) and
[current validation](reproduction-validation.json) for exact scope and provenance.
Requirements are a compatibility baseline; the manifest records what actually ran.

## First run: three routes, D off

Build on CPU, then run the replay command inside your site's allocated GPU job.
Use an idle GPU exclusively for performance measurements. The runner rejects
foreign CUDA PIDs at preparation and sample boundaries; this is not continuous
process monitoring and is not a replacement for scheduler ownership.

```bash
python tools/report_reproduction/build.py --output build/report
python tools/report_reproduction/benchmark.py \
  --build build/report --shape 32769x7x128 \
  --output results/report-smoke.json
python tools/report_reproduction/summarize.py results/report-smoke.json
```

Expected output includes `CPU_COMPLETE 3 CUDA_INITIALIZED False`, then
`CORRECTNESS_AND_SINGLE_KERNEL_PASS`, 12 rounds and `COMPLETE`. The result must
have `status="complete"`, `guard_failure=null`, successful byte checks and one
profiled attention kernel for every route. Fresh build/output paths are required.
The tail-block smoke validates execution, not the long-sequence speedup.

## Reproduce the four historical shapes, including D

The fourth route requires opt-in at **both** build and replay. This does not
enable D in `attention()`, `attention_fp8()` or `vc-attn-bench`.

```bash
python tools/report_reproduction/build.py --include-d --output build/report-d
for shape in 188214x7x128 188214x56x128 262144x7x128 262144x56x128; do
  python tools/report_reproduction/benchmark.py \
    --build build/report-d --include-d --shape "$shape" \
    --seed 20261004 --rounds 12 --repeats 5 \
    --warm-calls 2 --warm-seconds 1 \
    --output "results/report-d-$shape.json" || break
done
python tools/report_reproduction/summarize.py results/report-d-*.json
```

The largest shape needs substantial host memory and GPU headroom; the validated
configuration and memory observations are in the validation record. Each case is
a separate process. Its CPU BF16 Q/K/V alone occupy 10.5 GiB, before runtime
overhead; input generation and hashing can take over a minute. The validation
guard required at least 75 GiB of free GPU memory before launch and observed a
minimum of 26.43 GiB remaining during the largest case. Host peak memory was not
measured. Keep interrupted or rejected attempts separately; only
completed records count. On shared hardware, `--allow-shared-gpu` is available
for correctness/runner smoke and is explicitly marked nonisolated in the result.
It does not authorize disturbing another workload or establish performance.

## Inputs, timing and correctness

Q/K/V are contiguous CPU BF16 `torch.randn` tensors generated with independent
seeds `seed`, `seed+1`, `seed+2`, using one CPU thread, then transferred to the
GPU. For the four archived shapes and seed 20261004 the runner requires all three
full input SHA256 hashes to match the [published JSON](VC-Attention-Long-Sequence-Results.json).
Changing the seed creates a new cohort without an archived input-hash comparison.
FP8 Q/K descales and V scales are prepared once; Q descales retain even block
padding for this AOT family. All routes use dense noncausal D=128 with no skipping
and no LSE output. Typed FP8 views adapt the exported ABI; this does not establish
general real-Uint8 FFI compatibility for D.

Preparation, allocation, quantization, V packing, compilation, validation and
profiling all occur outside timing. The runner obtains the native callable and
arguments from the staged source interfaces, captures exactly that one attention
launch in a CUDA Graph and verifies its kernel count with the Torch profiler.
The ordinary `vc-attn-bench` attention scope includes V packing and measures a
different boundary.

Outputs are poisoned before direct calls, before graph replay and after timing.
The complete output bytes must match the corresponding source-interface result;
pre-fusedpipe, fusedpipe and optional D must also match one another. All outputs
must be finite. BF16 is a separate algorithmic reference: it is not expected to
be byte-equal to FP8, and this runner does not establish model-quality equivalence.

Timing uses CUDA events around repeated graph replay. Route order is randomized
in balanced rotation blocks, with equal counts at each position. Rounds must be
a positive multiple of the route count (three normally, four with D). The summary
recomputes medians, paired geometric-mean ratios and paired win counts from raw
samples. Sources, build manifest and libraries are checked again after replay.

## Interpreting results and failures

- A different native fingerprint creates a new compiler/kernel cohort. Compare
  native text **and** executable metadata before claiming the historical binary
  was reconstructed. Host objects/cubins can differ merely because names differ.
- D rejection is expected for an unrecognized compiler family. Keep D off or
  investigate the mismatch; do not relax its fingerprint checks to get a result.
- An attention cache miss aborts instead of compiling during replay. Rebuild in
  the matching environment. Triton preparation may compile outside the timer.
- A byte mismatch, nonfinite result, graph with extra kernels, changed artifact,
  foreign PID or incomplete run is invalid. Partial timing is never accepted.
- Clock and power observations are recorded at start/end; clocks are not locked.
  They are observations, not proof of stable clocks throughout the run.
- Historical ptxas identity and the complete compiler environment were not
  recorded. Recovered fields are labeled as such; present-day versions must not
  be inserted into historical records. New manifests list discovered ptxas
  candidates, without claiming which candidate a compiler actually invoked.

The published timing tables remain unchanged. New runs are separate experiments,
and a single fixed-seed cohort does not establish a universal scheduling gain.
Private captured-model tensors from the earlier cohort are not distributed;
only the synthetic long-sequence inputs are fully regenerable here.

[Reports](README.md) · [Tool implementation](../../tools/report_reproduction/README.md)
