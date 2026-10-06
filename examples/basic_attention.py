"""Run after installing on a reserved Blackwell GPU."""

import torch

from open_vc_attn import attention

q, k, v = [torch.randn(4096, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
out = attention(q, k, v)
print(out.shape, out.dtype)
