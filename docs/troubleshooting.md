# Troubleshooting

Start with `python -m pip check` and `vc-attn-check`. Run
`vc-attn-check --smoke` only after reserving a supported GPU.

| Symptom | Likely cause and next step |
|---|---|
| `CUDA unavailable` | Check `nvidia-smi`, your driver, GPU visibility and that Torch is a CUDA build. Follow the CUDA 13 install command in the [installation guide](installation.md). |
| Missing package or incompatible DSL/quack | Install with `python -m pip install -e '.[dev]' -c requirements-tested.txt` in the same environment as the command. Do not silently upgrade the pinned DSL/quack pair. |
| `Constraints cannot have extras` | Update to the current constraints file. Extras belong in package requirements; `-c` entries are plain names and version pins. |
| First call appears slow | The kernels compile on first use for a new specialization. Test a small shape first and allow compilation to finish. For B300 benchmark preparation without GPU execution, see [CPU compilation](../tools/idle_benchmark/README.md). Benchmark timers exclude compilation and warmup. |
| `Other CUDA processes detected` | Reserve an idle GPU and check process ownership. Do not stop unrelated processes. `--allow-shared-gpu` labels results non-isolated and is unsuitable for published isolated-performance claims. |
| GPU memory exhausted | Use a smaller S/H shape first, check resident allocations, and retain failure logs. The harness stores inputs, outputs and an accuracy reference; it needs more memory than a single attention call. |
| `nvcc` missing / unsupported architecture | Native v6 needs a CUDA 13 toolkit. Set `CUDA_HOME` to that toolkit; it takes precedence over PATH. Native v6 runs only on B300 / SM103. |
| Hopper, Ada, CPU, or another architecture | The supported GPU targets are B200 and B300. A similarly named low-level kernel file does not establish package support on other devices. |
| GQA/MQA or D other than 128 rejected | The public wrapper currently supports equal Q/K/V head counts and D=128. See the [support matrix](integration.md#supported-contract). |
| Unexpected numerical difference | FP8 and ExpCast are approximate. Compare full-output errors with a fixed BF16/FP32 reference and validate matched model outputs; low microbenchmark error is not video-quality acceptance. |
| A large shape has less speedup | Verify the baseline, input distribution, timing scope and dispatch conditions. The fused path has a size/feature gate; requesting LSE, causal or batched attention can select another path. |
| A result JSON is rejected by `vc-attn-report` | Only completed standalone benchmark reports with all raw samples are accepted. Interrupted or shared runs must not be relabeled as isolated successful measurements. Historical imported B300 JSON has a separate schema. |
| SGLang patch reports an unexpected file | Use the exact pinned revision and clean preimages from [the SGLang guide](sglang.md); the installer intentionally refuses unknown source. |

For a bug report include the repository revision, installed package versions,
GPU/driver, shape, backend, complete traceback and a minimal synthetic example.
For performance also attach the result JSON, timer scope, isolation and baseline.
Review logs for private paths, prompts or credentials before sharing.

[Repository](../README.md) · [Documentation](README.md)
