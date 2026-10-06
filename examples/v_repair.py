"""Compare Open-VC with and without V residual repair on a reserved B200."""

import torch

from open_vc_attn import attention, attention_v_repair, prepare_v_repair

torch.manual_seed(33)
q, k, v = [torch.randn(32769, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
plain = attention(q, k, v)
prepared = prepare_v_repair(q, k, v, budget=0.005)
repaired = attention_v_repair(prepared)
reference = attention(q, k, v, version="reference", mode="bf16").float()
for name, output in (("Open-VC", plain), ("Open-VC + V repair", repaired)):
    error = (output.float() - reference).norm() / reference.norm().clamp_min(1e-12)
    print(f"{name}: relative L2 = {error.item():.6f}")
print(f"Selected/padded tokens per head: {prepared.selected_tokens}/{prepared.repair_tokens}")
