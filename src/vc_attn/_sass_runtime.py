"""Private, per-function CuTe 4.6 adapter for the guarded B200 SASS D patch.

Disabled in the public API and benchmark. Experimental tooling must pass
enable=True explicitly; D has unresolved issues and is not a release default.
The caller must gate the supported v4 specialization. This adapter additionally
requires an exact accepted cubin. It never changes a DSL class, the original
compiled function, or the DSL's persistent cache. Unsupported layouts and host
construction failures return the original callable before any kernel is run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import tempfile
import types
import uuid
from pathlib import Path


def _clone_kwargs_wrapper(wrapper, original_target, replacement):
    """Preserve the generated ABI adapter, including dataclass/default rules."""
    if not isinstance(wrapper, types.FunctionType) or wrapper.__closure__ is not None:
        return None
    target_name = "__i_target_func"
    if (
        wrapper.__globals__.get(target_name) is not original_target
        or target_name not in wrapper.__code__.co_names
    ):
        return None
    namespace = dict(wrapper.__globals__)
    namespace[target_name] = replacement
    copied = types.FunctionType(wrapper.__code__, namespace, wrapper.__name__, wrapper.__defaults__)
    copied.__kwdefaults__ = (
        dict(wrapper.__kwdefaults__) if wrapper.__kwdefaults__ is not None else None
    )
    copied.__annotations__ = dict(wrapper.__annotations__)
    copied.__doc__ = wrapper.__doc__
    copied.__module__ = wrapper.__module__
    copied.__qualname__ = wrapper.__qualname__
    return copied


def _export_patched(compiled, destination, name, *, mid_window_blocks=4, **options):
    from ._sass_d import patch_host_object

    compiled.export_to_c(str(destination), function_name=name, **options)
    original = Path(destination).read_bytes()
    result = patch_host_object(original, mid_window_blocks=mid_window_blocks)
    if result is None:
        return None
    patched, info = result
    if not isinstance(patched, bytes) or len(patched) != len(original):
        return None
    return patched, dict(info)


class _PatchedCompiled:
    """One independently owned host object; all executable entrypoints use D."""

    def __init__(self, original, name, function, adapter, engine, object_bytes, info):
        self._original = original
        self._function = function
        self._adapter = adapter
        self._engine = engine
        # The engine and FFI function retain their own object/runtime lifetime;
        # retaining bytes also makes this ownership explicit for introspection.
        self._object_bytes = object_bytes
        self._mid_window_blocks = info["mid_window_blocks"]
        self.function_name = name
        self.sass_d_metadata = dict(
            info,
            runtime_adapter="cute-4.6-per-function-object-v1",
            execution="tvm_ffi" if function is not None else "aot_export_only",
            host_export_name=name,
            host_object_sha256=hashlib.sha256(object_bytes).hexdigest(),
        )

    def __call__(self, *args, **kwargs):
        if self._adapter is None:
            raise RuntimeError("This cross-compiled D function must be exported before execution")
        # Do not catch runtime/kernel errors or retry a failed launch as baseline.
        return self._adapter(*args, **kwargs)

    def to(self, device=None):
        # TVM-FFI performs its own per-device lazy initialization.
        return self

    def __tvm_ffi_object__(self):
        return self._function

    def export_to_c(
        self,
        object_file_path,
        function_name=None,
        *,
        enable_pic=True,
        export_only_tvm_ffi_symbols=False,
    ):
        """Export a fresh uniquely named D object; never silently emit baseline."""
        destination = Path(object_file_path)
        if destination.suffix != ".o" or not function_name:
            raise ValueError("D export requires an .o path and explicit function_name")
        # Original export clones its IR and renames host functions/references.
        # Patch that newly named object again; names can affect cubin strings.
        with tempfile.TemporaryDirectory(
            prefix="vc-attn-sass-export-", dir=destination.parent
        ) as tmp:
            staged = Path(tmp) / "export.o"
            result = _export_patched(
                self._original,
                staged,
                function_name,
                mid_window_blocks=self._mid_window_blocks,
                enable_pic=enable_pic,
                export_only_tvm_ffi_symbols=export_only_tvm_ffi_symbols,
            )
            if result is None:
                raise RuntimeError(
                    "D cubin guard rejected AOT export; no unpatched object was emitted"
                )
            staged.write_bytes(result[0])
            staged.replace(destination)


def maybe_patch_compiled(compiled, *, mid_window_blocks=4, enable=False):
    """Keep the original callable unless experimental tooling explicitly opts in.

    The public v4 interface does not opt in. The default exits before consulting
    the compiler, exporting an object or attempting any binary mutation.

    Call immediately after ``cute.compile`` and before publishing the interface
    cache entry. Normal JIT/file-cache hits are handled identically. Pure CPU
    cross-compiles retain a patched ``export_to_c`` route without loading a
    CUDA library. Loading pre-existing external AOT modules bypasses this hook.

    A successful wrapper exposes ``sass_d_metadata``. Absence means this
    optimization was not applied. The original object's artifacts/IR are never
    relabeled as having been compiled from the edited machine instructions.
    """
    if enable is not True:
        return compiled._original if isinstance(compiled, _PatchedCompiled) else compiled
    if isinstance(compiled, _PatchedCompiled):
        return (
            compiled
            if type(mid_window_blocks) is int and compiled._mid_window_blocks == mid_window_blocks
            else compiled._original
        )
    if type(mid_window_blocks) is not int or not 0 <= mid_window_blocks <= 0x7FFFFFFF:
        return compiled
    try:
        if importlib.metadata.version("nvidia-cutlass-dsl") != "4.6.0":
            return compiled
        from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunctionWithKwargs

        if type(compiled) is not TVMFFIJitCompiledFunctionWithKwargs:
            return compiled
        if compiled.execution_args.has_pointer_address_arg_specs:
            return compiled
        has_engine = compiled.engine is not None
        previous_wrapper = compiled._kwargs_wrapper
        previous_function = compiled._tvm_ffi_function
        if has_engine and (
            previous_function is None
            or _clone_kwargs_wrapper(previous_wrapper, previous_function, previous_function) is None
        ):
            return compiled
        if not has_engine and (previous_wrapper is not None or previous_function is not None):
            return compiled
        name = "vc_attn_sass_d_" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="vc-attn-sass-d-") as tmp:
            result = _export_patched(
                compiled, Path(tmp) / "kernel.o", name, mid_window_blocks=mid_window_blocks
            )
        if result is None:
            return compiled
        object_bytes, info = result
        if not has_engine:
            return _PatchedCompiled(compiled, name, None, None, None, object_bytes, info)

        import cutlass.cute as cute
        import tvm_ffi
        from cutlass._mlir._mlir_libs._cutlass_ir._execution_engine import BinaryExecutionEngine

        engine = BinaryExecutionEngine(
            object_bytes, cute.runtime.find_runtime_libraries(enable_tvm_ffi=True), True
        )
        pointer = engine.lookup("__tvm_ffi_" + name)
        if not pointer:
            return compiled
        function = tvm_ffi.Function.__from_extern_c__(pointer, keep_alive_object=engine)
        adapter = _clone_kwargs_wrapper(previous_wrapper, previous_function, function)
        if adapter is None:
            return compiled
        return _PatchedCompiled(compiled, name, function, adapter, engine, object_bytes, info)
    except Exception:
        # Construction has not called the FFI function. Returning the original
        # preserves supported operation when private DSL APIs/tools differ.
        return compiled
