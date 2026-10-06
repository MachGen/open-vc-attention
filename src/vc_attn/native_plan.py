"""Optional CUDA Toolkit-only native v6 baseline with explicit buffer lifetime."""

import ctypes
from pathlib import Path

import torch


class NativePlan:
    def __init__(self, prepared, library):
        if torch.cuda.get_device_capability(prepared.q.device) != (10, 3):
            raise ValueError("Native v6 requires B300 / SM103")
        if prepared.q.ndim != 3 or prepared.q.shape != prepared.k.shape:
            raise ValueError("Native v6 supports one self-attention sequence [S,H,128]")
        self.prepared = prepared
        self.out = torch.empty_like(prepared.q, dtype=torch.bfloat16)
        self.lse = torch.empty(
            (prepared.q.shape[1], prepared.q.shape[0]),
            device=prepared.q.device,
            dtype=torch.float32,
        )
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.fa_error.restype = ctypes.c_char_p
        self.lib.fa_create_scaled.restype = ctypes.c_void_p
        self.lib.fa_create_scaled.argtypes = (
            [ctypes.c_int, ctypes.c_int]
            + [ctypes.c_void_p] * 6
            + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            + [ctypes.c_void_p] * 3
        )
        self.lib.fa_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.fa_run.restype = ctypes.c_int
        self.lib.fa_destroy.argtypes = [ctypes.c_void_p]
        self.lib.fa_destroy.restype = None
        self.lib.fa_set_lse.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.fa_set_lse.restype = None
        self.handle = None
        with torch.cuda.device(prepared.q.device):
            self.handle = self.lib.fa_create_scaled(
                1,
                0,
                prepared.q.data_ptr(),
                prepared.k.data_ptr(),
                prepared.v.data_ptr(),
                self.out.data_ptr(),
                self.lse.data_ptr(),
                None,
                prepared.q.shape[0],
                prepared.q.shape[1],
                torch.cuda.current_stream().cuda_stream,
                prepared.q_descale.data_ptr(),
                prepared.k_descale.data_ptr(),
                prepared.v_descale.data_ptr(),
            )
        if not self.handle:
            raise RuntimeError(self.lib.fa_error().decode())
        self.lib.fa_set_lse(self.handle, None)

    def __call__(self):
        if not self.handle:
            raise RuntimeError("Native plan is closed")
        with torch.cuda.device(self.prepared.q.device):
            status = self.lib.fa_run(self.handle, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(self.lib.fa_error().decode())
        return self.out.reshape(self.prepared.output_shape)

    def close(self):
        if self.handle:
            with torch.cuda.device(self.prepared.q.device):
                torch.cuda.synchronize(self.prepared.q.device)
                self.lib.fa_destroy(self.handle)
                self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()
