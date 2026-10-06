"""Small inference API with explicit preparation and timing boundaries."""

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module

import torch

from .registry import DEFAULT_MID_WINDOW_BLOCKS, DEFAULT_VERSION, VERSIONS


@lru_cache(maxsize=None)
def interface(version):
    if version not in VERSIONS:
        raise ValueError(f"Unknown version {version!r}; choose from {VERSIONS}")
    return import_module(f"vc_attn._kernels.{version}.flash_attn.cute.interface")


def raw_forward(q, k, v, *, version=DEFAULT_VERSION, **kwargs):
    """Advanced kernel interface. Uses that snapshot's exact argument contract.

    Forward only: gradients are rejected. Returns (output, optional LSE).
    Caller owns layout, quantization metadata and low-level feature compatibility.
    """
    if any(t.requires_grad for t in (q, k, v)):
        raise ValueError("vc_attn is an inference API; detach inputs explicitly")
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
    """Opt-in B200 preparation fusion; see preparation.py for the narrow contract."""
    from .preparation import prepare_fp8_fused as implementation

    return implementation(q, k, v, cu_seqlens=cu_seqlens)


def attention_fp8(
    prepared,
    *,
    version=DEFAULT_VERSION,
    expcast=True,
    causal=False,
    softmax_scale=None,
    mid_window_blocks=DEFAULT_MID_WINDOW_BLOCKS,
    return_lse=False,
):
    """Run prepared FP8 attention, using a four-block dense scan by default.

    Eligible B200 calls use fusedpipe without D. None restores the original scan; causal
    calls always use the masked traversal. Automatic V packing is inside this call.
    """
    if version == "baseline" and expcast:
        raise ValueError("The ordinary FP8 baseline does not implement ExpCast")
    flags = {"expcast": True} if expcast else {}
    if getattr(prepared, "v_prepacked", False):
        if version != "v4":
            raise ValueError("Prepacked V is supported only by the v4 interface")
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
        skip_softmax_error=0.0,
        skip_pv_gemm=False,
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
):
    """BF16/FP16 in, attention out. Includes quantization for low-bit modes.

    mode is bf16 (no conversion), fp8, or expcast. No implicit fallback.
    A 3D tensor denotes one sequence; 4D tensors denote independent batch items.
    Dense low-bit calls default to mid_window_blocks=4; None restores the original
    scan. Causal calls use masked traversal, and BF16 calls keep their native scan.
    """
    validate_qkv(q, k, v)
    if mode not in ("bf16", "fp8", "expcast"):
        raise ValueError("mode must be bf16, fp8, or expcast")
    if mode != "bf16":
        return attention_fp8(
            prepare_fp8(q, k, v),
            version=version,
            expcast=mode == "expcast",
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


def prepare_v_smooth(q, k, v, *, version=DEFAULT_VERSION, **kwargs):
    if version == "baseline":
        raise ValueError("V-Smooth is not part of the ordinary baseline")
    return import_module(f"vc_attn._kernels.{version}.v_smooth").prepare_v_smooth(q, k, v, **kwargs)


def quantize_nvfp4(x, *, version=DEFAULT_VERSION):
    return import_module(f"vc_attn._kernels.{version}.nvfp4").quantize_nvf4(x)
