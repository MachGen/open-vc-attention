import torch


def e4m3_scale_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.int8:
        return tensor.view(torch.float8_e4m3fn)
    return tensor


_NVF4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def swizzle_nvf4_sf_for_kernel(sf: torch.Tensor) -> torch.Tensor:
    """Put (B, S, H, D/16) scales in tcgen05 layout_128x4 storage order."""
    bsz, seqlen, nheads, sf_k = sf.shape
    if sf_k % 4 != 0:
        raise ValueError(f"NVFP4 scale layout requires D/16 divisible by 4, got {sf_k}")
    seqlen_padded = ((seqlen + 127) // 128) * 128
    if seqlen_padded != seqlen:
        padded = torch.empty(
            bsz, seqlen_padded, nheads, sf_k, device=sf.device, dtype=sf.dtype
        )
        padded[:, :seqlen] = sf
        padded[:, seqlen:] = 0
        sf = padded
    return (
        sf.reshape(bsz, seqlen_padded // 128, 4, 32, nheads, sf_k // 4, 4)
        .permute(0, 4, 1, 5, 3, 2, 6)
        .contiguous()
        .view(bsz, seqlen_padded, nheads, sf_k)
    )


def quantize_nvf4(x: torch.Tensor, sf_vec: int = 16):
    """Quantize bf16 to packed NVFP4 + per-16 E4M3 scales.

    This runs during benchmark setup, so its overhead is intentionally excluded
    from timed attention kernel measurements.
    """
    bsz, seqlen, nheads, headdim = x.shape
    if headdim % sf_vec != 0:
        raise ValueError(f"NVFP4 requires headdim divisible by {sf_vec}, got {headdim}")
    levels = _NVF4_LEVELS.to(x.device)
    xb = x.reshape(bsz, seqlen, nheads, headdim // sf_vec, sf_vec)
    amax = xb.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale_e4m3 = (amax / 6.0).to(torch.float8_e4m3fn)
    scale_r = scale_e4m3.float()
    xn = xb.float() / scale_r
    sign = torch.sign(xn)
    idx = (xn.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    code = (idx.to(torch.uint8) | ((sign < 0).to(torch.uint8) << 3)).reshape(
        bsz, seqlen, nheads, headdim
    )
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    sf = swizzle_nvf4_sf_for_kernel(scale_e4m3.squeeze(-1).contiguous())
    return packed.view(torch.float4_e2m1fn_x2), sf
