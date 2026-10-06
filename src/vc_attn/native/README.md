# Native CUDA v6 — B300 only

The files directly in this directory are the original **B300 / SM103** v6 implementation.

- [fa_fp8_v6.cu](fa_fp8_v6.cu) implements FP8 attention and the C ABI for plan creation, execution and destruction.
- [tmem_ops.cuh](tmem_ops.cuh) provides inline PTX Tensor Memory load/store and reduction helpers.
- [../native_plan.py](../native_plan.py) binds the C ABI, retains buffers and manages plan lifetime.

Build from the repository root with CUDA 13, then benchmark on B300:

```bash
python tools/build_native.py --arch sm_103a --output build/libvc_native_v6.so
vc-attn-bench --shapes 188214x7x128 --backends fp8_ref native_v6 vc_scaled \
  --baseline native_v6 --native-library build/libvc_native_v6.so
```

To call it directly, prepare FP8 inputs once and keep the plan alive for repeated
calls with those same tensors:

```python
from contextlib import closing
from vc_attn import prepare_fp8
from vc_attn.native_plan import NativePlan

# q, k, v: CUDA BF16 tensors on B300, each shaped [S, H, 128].
prepared = prepare_fp8(q, k, v)
with closing(NativePlan(prepared, "build/libvc_native_v6.so")) as plan:
    output = plan().clone()  # The plan reuses its output buffer on later calls.
```

Rebuild preparation and the plan for new input tensors. The plan retains its
input buffers and must be closed after use. This direct path is single-sequence,
equal-head self-attention; it does not implement a model or KV-cache interface.

The original uses SM103-only `tcgen05.ld.red` and is not runnable on B200.
No prebuilt DSO is distributed. Native plans time attention on already prepared
FP8 inputs; plan creation and quantization are excluded. See
[baseline details](../../../docs/appendix/versions.md) and [the protocol](../../../docs/benchmarking.md).

The [fresh B300 performance table](../../../docs/performance.md) compares this
original native implementation with VC and the fixed BF16/ordinary-FP8 references
on S=73426/188214 and H=7/56. Read its shared-GPU and warmup qualification.

[Repository](../../../README.md) · [Parent directory](../README.md)
