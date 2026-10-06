# ruff: noqa: E402
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUTE_DSL_ARCH"] = "sm_103a"
from pathlib import Path

root = Path(os.environ.get("VC_ATTN_WORKDIR", "build/idle-benchmark")).resolve()
root.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(root / "cache/triton")
import json
from types import SimpleNamespace

import torch
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.runtime.driver import driver

from vc_attn import quantization as q
from vc_attn._kernels.scaled.flash_attn.cute.v_layout import transpose_v

driver.set_active(
    SimpleNamespace(
        get_current_device=lambda: 0,
        get_current_stream=lambda d: 0,
        get_current_target=lambda: GPUTarget("cuda", 103, 32),
    )
)
records = []


def record(**kw):
    records.append(kw["compile"]["specialization_data"])


knobs.runtime.jit_post_compile_hook = record
for s, h in [(188214, 7), (188214, 56), (73426, 7), (73426, 56), (129, 7)]:
    n = (s + 127) // 128
    q._per_block_quant_kernel.warmup(
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float32,
        s,
        h,
        128,
        n,
        IS_INT8=False,
        QMAX=448.0,
        BLOCK_L=128,
        BLOCK_D=128,
        num_warps=4,
        grid=(1, n, h),
    )
    q._per_head_amax_kernel.warmup(
        torch.bfloat16,
        torch.float32,
        s,
        h,
        128,
        BLOCK_L=128,
        BLOCK_D=128,
        num_warps=4,
        grid=(1, n, h),
    )
    q._per_head_scale_cast_kernel.warmup(
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float32,
        s,
        h,
        128,
        IS_INT8=False,
        QMAX=448.0,
        BLOCK_L=128,
        BLOCK_D=128,
        num_warps=4,
        grid=(1, n, h),
    )
    transpose_v.warmup(
        torch.uint8,
        torch.uint8,
        s,
        h * 128,
        n * 128,
        64,
        128,
        num_warps=4,
        grid=((n * 128 + 63) // 64, h, 1),
    )
(root / "aot-sm103/triton-specs.json").write_text(json.dumps(records, indent=2) + "\n")
assert not torch.cuda.is_initialized()
print(
    "TRITON_CPU_COMPLETE", len(records), "CUDA_INITIALIZED", torch.cuda.is_initialized(), flush=True
)
