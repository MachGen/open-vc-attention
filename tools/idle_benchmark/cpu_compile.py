# ruff: noqa: E402
"""Offline SM103 compilation; no CUDA allocations or kernel execution."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUTE_DSL_ARCH"] = "sm_103a"
os.environ["OMP_NUM_THREADS"] = "4"
import dataclasses
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_tensor
from key_format import encode
from torch._subclasses.fake_tensor import FakeTensorMode

from vc_attn.api import interface
from vc_attn.benchmark import _runtime_provenance

root = Path(os.environ.get("VC_ATTN_WORKDIR", "build/idle-benchmark")).resolve()
root.mkdir(parents=True, exist_ok=True)
dest = root / "aot-sm103"
dest.mkdir(exist_ok=True)
records = []
real_compile = cute.compile


def tensor_descriptor(t, assumed_align=16, leading_dim=-1, **kwargs):
    leading_dim = t.ndim - 1 if leading_dim == -1 else leading_dim
    dtype = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    }[t.dtype]
    return make_fake_tensor(
        dtype,
        tuple(cute.sym_int32() for _ in t.shape),
        tuple(1 if i == leading_dim else cute.sym_int64() for i in range(t.ndim)),
        memspace=cute.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def fake_pack(v):
    s, h, d = v.shape
    padded = (s + 127) // 128 * 128
    return (
        torch.empty((1, h, d, padded), dtype=v.dtype, device=v.device)[..., :s]
        .movedim(-1, -3)
        .squeeze(0)
    )


def compile_export(fn, *args, **kwargs):
    start = time.time()
    name = "attn_" + str(len(records))
    print("CPU_COMPILE", name, flush=True)
    compiled = real_compile(fn, *args, **kwargs)
    obj = dest / (name + ".o")
    so = dest / (name + ".so")
    compiled.export_to_c(str(obj), function_name=name)
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-o",
            str(so),
            str(obj),
            *cute.runtime.find_runtime_libraries(enable_tvm_ffi=True),
        ],
        check=True,
    )
    signature = inspect.signature(fn.__call__)
    params = list(signature.parameters.values())
    record = {
        "name": name,
        "library": so.name,
        "sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
        "arg_names": [p.name for p in params],
        "arg_defaults": [p.default for p in params if p.default is not inspect.Parameter.empty],
        "dataclass_names": [p.name for p, a in zip(params, args) if dataclasses.is_dataclass(a)],
        "seconds": time.time() - start,
    }
    records.append(record)
    print("CPU_COMPILED", name, record["seconds"], flush=True)
    # The host wrapper's launch is deliberately a no-op during metadata staging.
    return lambda *a, **kw: None


for version in ("baseline", "scaled"):
    mod = interface(version)
    vmod = (
        __import__(f"vc_attn._kernels.{version}.flash_attn.cute.v_layout", fromlist=["pack_v"])
        if version == "scaled"
        else None
    )
    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(patch.object(mod, "_get_device_capability", lambda: 10))
        stack.enter_context(patch.object(mod, "_get_device_capability_minor", lambda: 3))
        stack.enter_context(patch.object(mod, "to_cute_tensor", tensor_descriptor))
        stack.enter_context(
            patch.object(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))
        )
        stack.enter_context(patch.object(cute, "compile", compile_export))
        if vmod:
            stack.enter_context(patch.object(vmod, "pack_v", fake_pack))
        for s, h in [(188214, 7), (188214, 56), (73426, 7), (73426, 56)]:
            for fp8 in [False, True] if version == "baseline" else [True]:
                with FakeTensorMode():
                    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
                    q, k, v = [
                        torch.empty((s, h, 128), device="cuda:0", dtype=dtype) for _ in range(3)
                    ]
                    cu = torch.empty(2, device="cuda:0", dtype=torch.int32)
                    kw = dict(
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=s,
                        max_seqlen_k=s,
                        return_lse=False,
                    )
                    if fp8:
                        kw.update(
                            q_descale=torch.empty((1, h, (s + 127) // 128), device="cuda:0"),
                            k_descale=torch.empty((1, h, (s + 127) // 128), device="cuda:0"),
                            v_descale=torch.empty((1, h), device="cuda:0"),
                        )
                    if version == "scaled":
                        kw["expcast"] = True
                    before = set(mod._flash_attn_fwd.compile_cache)
                    mod._flash_attn_fwd(q, k, v, **kw)
                    added = set(mod._flash_attn_fwd.compile_cache) - before
                    if added:
                        assert len(added) == 1
                        records[-1].update(
                            version=version,
                            example_shape=[s, h, 128],
                            fp8=fp8,
                            cache_key=encode(added.pop()),
                        )
    assert not torch.cuda.is_initialized(), "Offline compilation initialized CUDA"
(dest / "manifest.json").write_text(
    json.dumps(
        {
            "gpu_arch": "sm_103a",
            "cuda_initialized": torch.cuda.is_initialized(),
            "source_provenance": _runtime_provenance(),
            "records": records,
        },
        indent=2,
    )
    + "\n"
)
print("CPU_COMPLETE", len(records), "CUDA_INITIALIZED", torch.cuda.is_initialized(), flush=True)
