"""Separate preparation from attention for fixed activations."""

import torch

from open_vc_attn import attention_fp8, prepare_fp8

q, k, v = [torch.randn(4096, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
prepared = prepare_fp8(q, k, v)
out = attention_fp8(prepared)
print(out.shape, out.dtype)
