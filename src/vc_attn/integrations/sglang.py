"""Opt-in SGLang diffusion adapter. Requires the pinned registration patch.

Quantization is included. Sequence-parallel communication remains in SGLang.
"""

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

from .diffusion import DenseDiffusionAdapter


class VCAttentionBackend(AttentionBackend):
    accept_output_buffer = False

    @staticmethod
    def get_enum():
        return AttentionBackendEnum.VC_ATTN

    @staticmethod
    def get_supported_head_sizes():
        return [128]

    @staticmethod
    def get_impl_cls():
        return VCAttentionImpl

    @staticmethod
    def get_metadata_cls():
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None

    @classmethod
    def supports_ring_rotation(cls):
        return False


class VCAttentionImpl(DenseDiffusionAdapter, AttentionImpl):
    pass
