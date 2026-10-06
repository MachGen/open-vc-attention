"""VC-Attention baseline: ExpCast with V-Smooth, without Open-VC's optimizations.

VC-Attention's method on the same Blackwell kernel family: ExpCast probabilities, V-Smooth
(online k-means value grouping, K/V permutation and per-128-token-block V demeaning with
means restored in the kernel), and the original key scan. None of Open-VC's mid-window
traversal, packed V, fused ExpCast preparation or B200 score/scale overlap is used.

``prepare_vc`` groups values (k-means) and prepares FP8 inputs; pass its ``permutation``
back to reuse the grouping, as VC-Attention does after its first denoising steps. Reuse
runs a fused two-pass Triton preparation. ``attention_vc`` runs the kernel.
"""

import torch

from .api import _layout, raw_forward, validate_qkv


@torch.no_grad()
def prepare_vc(q, k, v, *, clusters=64, iterations=4, permutation=None, check_permutation=True):
    """Group V tokens with k-means, permute K/V by group and demean V per 128-token block.

    Accepts BF16 self-attention inputs [S,H,128] or [1,S,H,128]. Pass the returned
    ``permutation`` back in to reuse the grouping across calls, as VC-Attention does
    after its first denoising steps. ``check_permutation=False`` skips validating a reused
    permutation, which synchronizes with the host and so cannot run in CUDA graph capture.
    """
    from ._kernels.blackwell.v_smooth import prepare_v_smooth

    validate_qkv(q, k, v)
    if q.dtype != torch.bfloat16 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("The VC baseline expects matching BF16 self-attention Q/K/V")
    if q.ndim == 4 and q.shape[0] != 1:
        raise ValueError("The VC baseline supports one sequence")
    q, k, v = [t.reshape(-1, t.shape[-2], t.shape[-1]) for t in (q, k, v)]
    # Triton launches on the current device and stream, not the inputs' device.
    with torch.cuda.device(q.device):
        return prepare_v_smooth(
            q,
            k,
            v,
            clusters=clusters,
            iterations=iterations,
            permutation=permutation,
            check_permutation=check_permutation,
            expand_means=True,
        )


def attention_vc(prepared, *, output_shape=None, softmax_scale=None, layout=None):
    """Run ExpCast attention with V-Smooth restoration and the original key scan.

    Pass ``layout`` (sequence metadata from a previous call) to reuse it across calls.
    """
    q, k, v = prepared.q, prepared.k, prepared.v
    if layout is None:
        _, _, _, layout, _ = _layout(q, k, v)
    out, _ = raw_forward(
        q,
        k,
        v,
        **layout,
        expcast=True,
        softmax_scale=softmax_scale,
        mid_window_blocks=None,
        return_lse=False,
        **prepared.forward_kwargs(),
    )
    return out if output_shape is None else out.reshape(output_shape)


def vc_attention(q, k, v, *, clusters=64, iterations=4, permutation=None):
    """BF16 in, attention out: VC grouping and quantization followed by attention."""
    prepared = prepare_vc(
        q, k, v, clusters=clusters, iterations=iterations, permutation=permutation
    )
    return attention_vc(prepared, output_shape=q.shape)
