"""Compare BF16, the VC-Attention baseline and Open-VC on one input (reserved Blackwell GPU)."""

import torch

from open_vc_attn import attention
from open_vc_attn.baselines import vc_attention

torch.manual_seed(7)
q, k, v = [torch.randn(32768, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
reference = attention(q, k, v, version="reference", mode="bf16").float()
for name, output in (
    ("VC (ExpCast + V-Smooth)", vc_attention(q, k, v)),
    ("Open-VC", attention(q, k, v)),
):
    error = (output.float() - reference).norm() / reference.norm().clamp_min(1e-12)
    print(f"{name}: relative L2 vs BF16 = {error.item():.4%}")
print("For timing, use: open-vc-attn-bench --preset video --timing graph")
