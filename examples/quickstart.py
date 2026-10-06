import torch

from vc_attn import attention, attention_fp8, prepare_fp8

q, k, v = [torch.randn(4096, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

# Drop-in inference call: includes Q/K/V quantization.
out = attention(q, k, v, mode="expcast")

# Explicit preparation: useful when the caller already owns FP8 activations.
prepared = prepare_fp8(q, k, v)
out_again = attention_fp8(prepared, expcast=True)
torch.testing.assert_close(out, out_again, atol=0, rtol=0)
print(out.shape, out.dtype)
