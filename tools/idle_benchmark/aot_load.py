# ruff: noqa: E402
"""Load attention-only AOT artifacts built locally from the installed snapshot."""

import hashlib
import importlib
import json
from pathlib import Path

import cutlass.cute as cute
from key_format import decode
from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

from vc_attn.api import interface
from vc_attn.benchmark import _runtime_provenance

modules = []


def load(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source_provenance"] != _runtime_provenance():
        raise ValueError("Installed source differs from the compiled source; recompile")
    for r in manifest["records"]:
        path = root / r["library"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"AOT library hash mismatch: {path.name}")
        module = cute.runtime.load_module(str(path), enable_tvm_ffi=True)
        modules.append(module)
        fn = make_kwargs_wrapper(
            getattr(module, r["name"]),
            arg_names=r["arg_names"],
            arg_defaults=tuple(r["arg_defaults"]),
            map_dataclass_to_tuple=r["dataclass_names"],
        )
        if r["fp8"]:
            # The original from_dlpack ABI uses byte views; the supported fake
            # tensor AOT ABI names FP8 explicitly. Reinterpret without copying.
            import torch

            def typed(*args, _fn=fn):
                return _fn(*(tuple(t.view(torch.float8_e4m3fn) for t in args[:3]) + args[3:]))

            fn = typed
        # Data-only key format: no executable pickle payloads.
        key = decode(r["cache_key"])
        interface(r["version"])._flash_attn_fwd.compile_cache[key] = fn
    for spec in json.loads((root / "triton-specs.json").read_text()):
        d = json.loads(spec)
        modname, name = d["name"].rsplit(".", 1)
        getattr(importlib.import_module(modname), name).preload(spec)
    return manifest
