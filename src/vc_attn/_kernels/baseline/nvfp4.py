import torch


def e4m3_scale_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.int8:
        return tensor.view(torch.float8_e4m3fn)
    return tensor
