"""Framework-neutral dense/packed diffusion attention adapter."""

import os

import torch

from open_vc_attn._dispatch import DEFAULT_VERSION, VERSIONS
from open_vc_attn.api import attention

from .packed import segments


class DenseDiffusionAdapter:
    def __init__(
        self,
        num_heads,
        head_size,
        softmax_scale,
        causal=False,
        num_kv_heads=None,
        prefix="",
        **extra_impl_args,
    ):
        if head_size != 128 or num_kv_heads not in (None, num_heads):
            raise ValueError("Open-VC diffusion adapter requires MHA with head_size=128")
        if causal or extra_impl_args.get("dropout_p", 0):
            raise ValueError("Open-VC diffusion adapter supports dense noncausal inference only")
        self.version = os.environ.get("OPEN_VC_ATTN_IMPLEMENTATION", DEFAULT_VERSION)
        self.mode = os.environ.get("OPEN_VC_ATTN_MODE", "expcast")
        if self.version not in VERSIONS or self.mode not in ("bf16", "expcast"):
            raise ValueError("Invalid OPEN_VC_ATTN_IMPLEMENTATION or OPEN_VC_ATTN_MODE")
        if (self.mode == "bf16") != (self.version == "reference"):
            raise ValueError("Use reference with bf16, or open-vc with expcast")
        self.scale = softmax_scale
        self.trailing_padding = bool(extra_impl_args.get("packed_trailing_padding", False))

    def forward(self, query, key, value, attn_metadata=None):
        return attention(
            query, key, value, version=self.version, mode=self.mode, softmax_scale=self.scale
        )

    def forward_varlen(self, query, key, value, *, cu_seqlens, max_seqlen, cu_seqlens_host=None):
        if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
            raise ValueError("Packed Open-VC adapter expects matching [T,H,128] self-attention")
        ranges = segments(
            cu_seqlens_host, query.shape[0], max_seqlen, trailing_padding=self.trailing_padding
        )
        if cu_seqlens.numel() != len(ranges) + 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError("Device and host sequence metadata must describe the same bounds")
        if cu_seqlens.device != query.device:
            raise ValueError("Device sequence metadata must reside with Q/K/V")
        # Quantize each sequence separately so no scale block crosses a boundary.
        outputs = [
            torch.zeros_like(query[a:b]) if pad else self.forward(query[a:b], key[a:b], value[a:b])
            for a, b, pad in ranges
        ]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
