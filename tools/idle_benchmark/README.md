# CPU compilation and B300 idle-window sampling

These optional scripts move compilation off the GPU and collect complete paired
rounds between other workloads. Use them only when shared-GPU testing is allowed
by the machine owner. They do not reserve a GPU, stop other processes or change
clocks. An exclusive allocation remains the preferred measurement environment.

| File | Role |
|---|---|
| [cpu_compile.py](cpu_compile.py) | Compile four CuTe attention variants for SM103 on CPU and export shared libraries |
| [cpu_triton.py](cpu_triton.py) | Compile quantization and V-packing kernels on CPU |
| [aot_load.py](aot_load.py) | Verify source/artifact hashes and load locally built kernels |
| [key_format.py](key_format.py) | Encode dispatch keys as data-only JSON |
| [b300_windows.py](b300_windows.py) | Validate outputs and collect paired rounds in monitored idle windows |

## Prepare without running GPU kernels

Use the [tested installation](../../docs/installation.md), a CUDA 13 toolkit,
GCC, and `nvidia-ml-py` for NVML monitoring. Run these commands from the repository
root. Build and load artifacts within the same environment; they are not portable
release binaries.

```bash
python -m pip install nvidia-ml-py
export VC_ATTN_WORKDIR="$PWD/build/idle-benchmark"
export TRITON_CACHE_DIR="$VC_ATTN_WORKDIR/cache/triton"
mkdir -p "$VC_ATTN_WORKDIR"
CUDA_VISIBLE_DEVICES= python tools/build_native.py --arch sm_103a \
  --output "$VC_ATTN_WORKDIR/libvc_native_v6.so"
python tools/idle_benchmark/cpu_compile.py
python tools/idle_benchmark/cpu_triton.py
```

Both compilation scripts hide GPUs, target `sm_103a`, and assert that PyTorch
has not initialized CUDA. CuTe uses its supported fake-tensor/TVM-FFI
[AOT interface](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/tvm_ffi_compilation.html).
Each library has a unique exported symbol. The loader reinterprets FP8 byte
views for the AOT type contract without copying or changing values. Kernel
source bodies, descales and dispatch settings remain unchanged. These measurements
characterize the precompiled AOT path; first-use JIT compilation is not measured.

Compilation stages the dense, single-sequence D=128 configurations used below:
S=73426/188214 and H=7/56, plus a small validation call. This is a focused
benchmark helper, not a general AOT inference API.

## Collect complete paired rounds

```bash
python tools/idle_benchmark/b300_windows.py --gpu 0 --minutes 45 \
  --shapes 188214x7x128 188214x56x128 73426x7x128 73426x56x128 \
  --rounds 12 --warm-calls 2 --repeats 3
```

`--gpu` is the physical NVML device index. The process binds CUDA to that GPU's
UUID and keeps the same GPU and seeded inputs for each shape. It requires at
least 80 GiB of free HBM before starting each shape and retains its allocations
while waiting. It compares
`bf16_ref`, `fp8_ref`, `native_v6` and `vc_scaled`. Small FP32-oracle checks and
full-output BF16 comparisons precede timing. Internal V packing is timed;
quantization, compilation and the rest of the model are excluded.

The monitor waits for two seconds of stable idle after a trailing two-second
utilization check. It polls foreign process utilization on the **selected GPU**
every 0.1 seconds, checks process-set changes, and waits four seconds for delayed
samples after each window, with a conservative one-second tail. If interference
appears, it stops enqueueing additional work, discards the whole paired round,
and waits. Already enqueued work can finish before the pause. Accepted rounds
remain in the checkpoint while the process waits; no partial round is accepted.

The result is `results/b300-windows.json` under `VC_ATTN_WORKDIR`. `status=complete`
requires every requested shape and round to finish. The time limit writes
`status=partial` with a nonzero exit; errors write `status=invalid`. Starting a new process starts a
new run; an existing result path is rejected, so choose a new `--output`. A partial or
invalid report must not be passed off as a complete comparison.

This is **shared-GPU idle-window sampling**, not exclusive isolation. NVML has
finite sampling resolution, and resident model allocations remain present.
The shorter warmup/repeat protocol differs from the main exclusive B200 run.
Report those differences and do not infer cross-card hardware speedups from the
absolute latencies. Short kernels can also be sensitive to the brief warmup;
retain raw samples and use a separately labeled longer-warmup rerun when needed.
No clock, power or scheduler setting is modified.

[Tools](../README.md) · [Benchmark protocol](../../docs/benchmarking.md)
