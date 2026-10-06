"""Host-side packed sequence validation, without CUDA synchronization."""


def segments(bounds, total, max_seqlen, *, trailing_padding=False):
    if bounds is None:
        raise ValueError("cu_seqlens_host is required; CUDA metadata is not copied to CPU")
    bounds = tuple(bounds)
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != total:
        raise ValueError("Packed bounds must start at zero and end at the tensor length")
    if any(not isinstance(x, int) for x in bounds) or any(
        a >= b for a, b in zip(bounds, bounds[1:])
    ):
        raise ValueError("Packed bounds must be strictly increasing integers")
    result = [(a, b, False) for a, b in zip(bounds, bounds[1:])]
    if trailing_padding and len(bounds) == 3 and bounds[1] == max_seqlen:
        result[-1] = (bounds[1], bounds[2], True)
    if max_seqlen < 1 or any(b - a > max_seqlen for a, b, pad in result if not pad):
        raise ValueError("max_seqlen is inconsistent with packed bounds")
    return result
