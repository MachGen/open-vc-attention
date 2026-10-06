"""Build matched SM100 report controls on CPU; D requires --include-d."""

import argparse
import dataclasses
import inspect
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from common import (
    CONFIG,
    REPORT_SHAPES,
    SMOKE_SHAPE,
    environment,
    native_identity,
    now,
    save,
    sha,
    stage_source,
    tree_hashes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh build directory")
    parser.add_argument(
        "--include-d",
        action="store_true",
        help="Build experimental D; never enables it in the public API",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error("Use a fresh output directory")
    root.mkdir(parents=True)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        CUTE_DSL_ARCH="sm_100a",
        OMP_NUM_THREADS="1",
        CUTE_DSL_CACHE_DIR=str(root / "cache/cute"),
    )
    sys.dont_write_bytecode = True
    sys.argv[:] = [sys.argv[0]]
    repo = Path(__file__).resolve().parents[2]
    source = root / "source/vc_attn"
    provenance = stage_source(repo / "src/vc_attn", source)
    sys.path.insert(0, str(source.parent))
    sys.path.insert(0, str(repo / "tools/idle_benchmark"))
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import make_fake_tensor
    from key_format import encode
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vc_attn._sass_d import patch_host_object
    from vc_attn.api import interface

    dest = root / "artifacts"
    dest.mkdir()
    manifest = dict(
        schema="vc-report-aot-v1",
        status="building",
        started_at=now(),
        architecture="sm_100a",
        configuration=CONFIG,
        source=provenance,
        harness_files=tree_hashes(Path(__file__).resolve().parent),
        environment=environment(),
        compiler_options="--enable-tvm-ffi",
        shapes=[list(SMOKE_SHAPE), *map(list, REPORT_SHAPES)],
        d_enabled=args.include_d,
        cuda_initialized=False,
        records=[],
    )
    save(root / "manifest.json", manifest)
    records = manifest["records"]
    real_compile = cute.compile
    libraries = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
    manifest["runtime_libraries"] = [{"name": Path(p).name, "sha256": sha(p)} for p in libraries]

    def descriptor(t, assumed_align=16, leading_dim=-1, **kwargs):
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
        return (
            torch.empty((1, h, d, (s + 127) // 128 * 128), dtype=v.dtype, device=v.device)[..., :s]
            .movedim(-1, -3)
            .squeeze(0)
        )

    def link(obj, so):
        subprocess.run(["gcc", "-shared", "-o", str(so), str(obj), *libraries], check=True)

    def export(kernel, *values, **kwargs):
        name = f"report_{version}_{len(records)}"
        print("CPU_COMPILE", name, flush=True)
        compiled = real_compile(kernel, *values, **kwargs)
        obj, so = dest / (name + ".o"), dest / (name + ".so")
        compiled.export_to_c(str(obj), function_name=name)
        link(obj, so)
        params = list(inspect.signature(kernel.__call__).parameters.values())
        fields = (
            "inline_rescale",
            "fused_denominator",
            "tensor_core_denominator",
            "expcast_handoff",
            "q_stage",
            "m_block_size",
            "n_block_size",
            "mid_window_blocks",
            "threads_per_cta",
        )
        record = dict(
            name=name,
            library=so.name,
            sha256=sha(so),
            object_sha256=sha(obj),
            native=native_identity(obj),
            version=version,
            fp8=version != "baseline",
            arg_names=[p.name for p in params],
            arg_defaults=[p.default for p in params if p.default is not inspect.Parameter.empty],
            dataclass_names=[
                p.name for p, value in zip(params, values) if dataclasses.is_dataclass(value)
            ],
            kernel_config={key: getattr(kernel, key, None) for key in fields},
        )
        if version != "baseline":
            if not all(
                record["kernel_config"][key]
                for key in ("inline_rescale", "fused_denominator", "expcast_handoff")
            ):
                raise RuntimeError("Control failed to select packed-V fused denominator path")
        if version == "fusedpipe" and args.include_d:
            result = patch_host_object(obj.read_bytes(), mid_window_blocks=4)
            if result is None or result[1]["applied"] is not True:
                raise RuntimeError("Canonical D guard rejected this build; no substitute D emitted")
            patched, proof = result
            dname = "report_d_" + str(len(records))
            raw, renamed, dso = (
                dest / (dname + ".raw.o"),
                dest / (dname + ".o"),
                dest / (dname + ".so"),
            )
            raw.write_bytes(patched)
            symbols = subprocess.check_output(
                ["nm", "--defined-only", "--format=posix", str(obj)], text=True
            )
            names = {line.split()[0] for line in symbols.splitlines() if line.split()}
            mapping = {n: n.replace(name, dname) for n in names if name in n}
            if not mapping or any(
                n.startswith("__cute_internal_") and n not in mapping for n in names
            ):
                raise RuntimeError("Unrecognized internal host symbols")
            symbol_map = dest / (dname + ".symbols")
            symbol_map.write_text("".join(f"{old} {new}\n" for old, new in sorted(mapping.items())))
            subprocess.run(
                ["objcopy", "--redefine-syms=" + str(symbol_map), str(raw), str(renamed)],
                check=True,
            )
            link(renamed, dso)
            symbols_after = subprocess.check_output(
                ["nm", "--defined-only", "--format=posix", str(dso)], text=True
            )
            if name in symbols_after or not all(new in symbols_after for new in mapping.values()):
                raise RuntimeError("D host symbols were not isolated")
            cubin = patched[proof["cubin_offset"] : proof["cubin_offset"] + proof["cubin_size"]]
            if dso.read_bytes().count(cubin) != 1:
                raise RuntimeError("Linked D library did not preserve its exact patched cubin")
            record["d"] = dict(
                name=dname,
                library=dso.name,
                sha256=sha(dso),
                proof=proof,
                host_symbol_map=mapping,
                derived_from_object_sha256=sha(obj),
                native=native_identity(renamed),
            )
        records.append(record)
        return lambda *values, **kwargs: None

    try:
        # Larger shape first: it compiles the same dynamic family as the report.
        for version in ("baseline", "v4", "fusedpipe"):
            mod = interface(version)
            with ExitStack() as stack:
                stack.enter_context(patch.object(mod, "_get_device_capability", lambda: 10))
                stack.enter_context(patch.object(mod, "_get_device_capability_minor", lambda: 0))
                stack.enter_context(patch.object(mod, "to_cute_tensor", descriptor))
                stack.enter_context(
                    patch.object(
                        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0)
                    )
                )
                stack.enter_context(patch.object(cute, "compile", export))
                if version != "baseline":
                    vmod = __import__(
                        f"vc_attn._kernels.{version}.flash_attn.cute.v_layout", fromlist=["pack_v"]
                    )
                    stack.enter_context(patch.object(vmod, "pack_v", fake_pack))
                for s, h, d in (*REPORT_SHAPES, SMOKE_SHAPE):
                    with FakeTensorMode():
                        fp8 = version != "baseline"
                        q, k, v = [
                            torch.empty(
                                (s, h, d),
                                device="cuda:0",
                                dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
                            )
                            for _ in range(3)
                        ]
                        cu = torch.empty(2, device="cuda:0", dtype=torch.int32)
                        options = dict(
                            cu_seqlens_q=cu,
                            cu_seqlens_k=cu,
                            max_seqlen_q=s,
                            max_seqlen_k=s,
                            return_lse=False,
                        )
                        if fp8:
                            options.update(
                                CONFIG,
                                q_descale=torch.empty(
                                    (1, h, ((s + 255) // 256) * 2), device="cuda:0"
                                ),
                                k_descale=torch.empty((1, h, (s + 127) // 128), device="cuda:0"),
                                v_descale=torch.empty((1, h), device="cuda:0"),
                            )
                        before = set(mod._flash_attn_fwd.compile_cache)
                        mod._flash_attn_fwd(q, k, v, **options)
                        added = set(mod._flash_attn_fwd.compile_cache) - before
                        if added:
                            if len(added) != 1:
                                raise RuntimeError("Expected one attention specialization")
                            records[-1].update(
                                cache_key=encode(added.pop()), example_shape=[s, h, d]
                            )
            if torch.cuda.is_initialized():
                raise RuntimeError("Offline compilation initialized CUDA")
        manifest.update(status="complete", completed_at=now(), gpu_validation="not run")
    except BaseException as exc:
        manifest.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(root / "manifest.json", manifest)
    print("CPU_COMPLETE", len(records), "CUDA_INITIALIZED", torch.cuda.is_initialized(), flush=True)


if __name__ == "__main__":
    main()
