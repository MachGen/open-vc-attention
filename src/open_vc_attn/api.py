"""Small inference API with explicit preparation and timing boundaries."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import torch

from ._dispatch import DEFAULT_MID_WINDOW_BLOCKS, DEFAULT_VERSION, interface


def raw_forward(q, k, v, *, version=DEFAULT_VERSION, **kwargs):
    """Advanced kernel interface. Uses the selected kernel's exact argument contract.

    Forward only: gradients are rejected. Returns (output, optional LSE).
    Caller owns layout, quantization metadata and low-level feature compatibility.
    """
    if any(t.requires_grad for t in (q, k, v)):
        raise ValueError("open_vc_attn is an inference API; detach inputs explicitly")
    module = interface(version)
    fn = getattr(module, "flash_attn_fwd", None) or module._flash_attn_fwd
    with torch.cuda.device(q.device):
        return fn(q, k, v, **kwargs)[:2]


def validate_qkv(q, k, v, *, floating=True):
    if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("Expected [S,H,128] or [B,S,H,128] Q/K/V")
    if any(t.shape[-1] != 128 or min(t.shape) < 1 for t in (q, k, v)):
        raise ValueError("The convenience API requires nonempty D=128 inputs")
    if k.shape != v.shape or q.shape[-2:] != k.shape[-2:]:
        raise ValueError("The convenience API requires MHA and matching K/V shapes")
    if q.ndim == 4 and q.shape[0] != k.shape[0]:
        raise ValueError("Q/K/V batch sizes must match")
    if any(not t.is_cuda or t.device != q.device for t in (q, k, v)):
        raise ValueError("Q/K/V must reside on the same CUDA device")
    if any(t.requires_grad for t in (q, k, v)):
        raise ValueError("Forward inference only; detach inputs explicitly")
    allowed = (torch.float16, torch.bfloat16) if floating else (torch.float8_e4m3fn,)
    if q.dtype not in allowed or any(t.dtype != q.dtype for t in (k, v)):
        raise ValueError(f"Expected matching dtypes from {allowed}")
    if torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
        raise ValueError("This release validates Blackwell SM100/SM103 only")


def _layout(q, k, v):
    original_shape = tuple(q.shape)
    if q.ndim == 3 or q.shape[0] == 1:
        q, k, v = [t.reshape(-1, t.shape[-2], t.shape[-1]).contiguous() for t in (q, k, v)]
        sq, sk = q.shape[0], k.shape[0]
        kwargs = {
            # Build static metadata on device; pageable host copies break graph capture.
            "cu_seqlens_q": torch.arange(2, device=q.device, dtype=torch.int32) * sq,
            "cu_seqlens_k": torch.arange(2, device=q.device, dtype=torch.int32) * sk,
            "max_seqlen_q": sq,
            "max_seqlen_k": sk,
        }
    else:
        q, k, v = [t.contiguous() for t in (q, k, v)]
        kwargs = {}
    return q, k, v, kwargs, original_shape


@dataclass(frozen=True)
class PreparedFP8:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_descale: torch.Tensor
    k_descale: torch.Tensor
    v_descale: torch.Tensor
    layout: dict
    output_shape: tuple
    v_prepacked: bool = False


@lru_cache(None)
def _packed_dsl_available():
    try:
        return package_version("nvidia-cutlass-dsl") == "4.6.2"
    except PackageNotFoundError:
        return False


def _fused_preparation_eligible(q, k, v, *, version, mode, causal, return_lse):
    """Mirror the public subset of the packed B200 dispatch without loading CUDA code."""
    return (
        version == DEFAULT_VERSION
        and mode == "expcast"
        and not causal
        and not return_lse
        and (q.ndim == 3 or (q.ndim == 4 and q.shape[0] == 1))
        and q.shape == k.shape == v.shape
        and q.shape[-3] >= 32768
        and all(t.is_contiguous() for t in (q, k, v))
        and torch.cuda.get_device_capability(q.device) == (10, 0)
        and _packed_dsl_available()
    )


def prepare_fp8(q, k, v):
    """Q/K: per 128 tokens per head; V: per sequence per head; E4M3 max 448.

    The returned tensors retain descales in FP32. Preparation is stream ordered.
    Reuse the result for attention-only benchmarks, not for changing activations.
    """
    validate_qkv(q, k, v)
    from .quantization import fp8_quantize_per_block_triton, fp8_quantize_per_head_triton

    with torch.cuda.device(q.device):
        q, k, v, layout, shape = _layout(q, k, v)
        q8, qs = fp8_quantize_per_block_triton(q, block_l=128)
        k8, ks = fp8_quantize_per_block_triton(k, block_l=128)
        v8, vs = fp8_quantize_per_head_triton(v)
        if q.ndim == 3:
            q8, k8, v8 = q8[0], k8[0], v8[0]
        return PreparedFP8(q8, k8, v8, qs, ks, vs, layout, shape)


def prepare_fp8_fused(q, k, v, *, cu_seqlens=None):
    """Three-kernel B200 preparation; returns V already in the packed layout.

    Dense single-sequence self-attention only, S >= 32768, D=128, DSL 4.6.2.
    Optional contiguous int32 CUDA metadata must contain [0, S]; the caller
    owns its values. Without metadata, device operations remain graph-safe.
    """
    from .preparation import prepare_fp8_fused as implementation

    return implementation(q, k, v, cu_seqlens=cu_seqlens)


def attention_fp8(
    prepared,
    *,
    version=DEFAULT_VERSION,
    causal=False,
    softmax_scale=None,
    mid_window_blocks=DEFAULT_MID_WINDOW_BLOCKS,
    return_lse=False,
):
    """Run prepared FP8 ExpCast attention, using a four-block dense scan by default.

    The execution pipeline is selected internally. None restores the original scan; causal
    calls always use the masked traversal. Generic prepared inputs pack V inside
    this call; fused prepared inputs already contain packed V.
    """
    if version != DEFAULT_VERSION:
        raise ValueError(f"Prepared FP8 attention requires version={DEFAULT_VERSION!r}")
    flags = {"expcast": True}
    if getattr(prepared, "v_prepacked", False):
        if causal or return_lse:
            raise ValueError("Prepacked V requires dense Open-VC ExpCast without LSE")
        flags["v_prepacked"] = True
    out, lse = raw_forward(
        prepared.q,
        prepared.k,
        prepared.v,
        version=version,
        q_descale=prepared.q_descale,
        k_descale=prepared.k_descale,
        v_descale=prepared.v_descale,
        **prepared.layout,
        **flags,
        causal=causal,
        softmax_scale=softmax_scale,
        mid_window_blocks=None if causal else mid_window_blocks,
        return_lse=return_lse,
    )
    out = out.reshape(prepared.output_shape)
    return (out, lse) if return_lse else out


def attention(
    q,
    k,
    v,
    *,
    version=DEFAULT_VERSION,
    mode="expcast",
    causal=False,
    softmax_scale=None,
    mid_window_blocks=DEFAULT_MID_WINDOW_BLOCKS,
    return_lse=False,
    preparation="auto",
):
    """BF16/FP16 in, attention out. Includes quantization for the ExpCast mode.

    mode is expcast (Open-VC, the default) or bf16 (the FlashAttention-4 BF16 reference,
    which requires version="reference"). No implicit fallback.
    A 3D tensor denotes one sequence; 4D tensors denote independent batch items.
    Dense low-bit calls default to mid_window_blocks=4; None restores the original
    scan. Causal calls use masked traversal, and BF16 calls keep their native scan.
    preparation='auto' fuses preparation for eligible B200 calls; 'unfused'
    retains general preparation and 'fused' requires the packed contract.
    """
    validate_qkv(q, k, v)
    if mode not in ("bf16", "expcast"):
        raise ValueError("mode must be bf16 or expcast")
    if (mode == "bf16") != (version == "reference"):
        raise ValueError('mode="bf16" requires version="reference"; ExpCast requires Open-VC')
    if preparation not in ("auto", "unfused", "fused"):
        raise ValueError("preparation must be auto, unfused, or fused")
    fuse = preparation != "unfused" and _fused_preparation_eligible(
        q, k, v, version=version, mode=mode, causal=causal, return_lse=return_lse
    )
    if preparation == "fused" and not fuse:
        raise ValueError(
            "Fused preparation requires contiguous B200 single-sequence self-attention, "
            "S >= 32768, DSL 4.6.2, dense Open-VC ExpCast and no LSE"
        )
    if mode != "bf16":
        return attention_fp8(
            prepare_fp8_fused(q, k, v) if fuse else prepare_fp8(q, k, v),
            version=version,
            causal=causal,
            softmax_scale=softmax_scale,
            mid_window_blocks=mid_window_blocks,
            return_lse=return_lse,
        )
    with torch.cuda.device(q.device):
        q, k, v, layout, shape = _layout(q, k, v)
        out, lse = raw_forward(
            q,
            k,
            v,
            version=version,
            **layout,
            causal=causal,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
        )
    out = out.reshape(shape)
    return (out, lse) if return_lse else out
