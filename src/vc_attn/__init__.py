"""Forward attention for Blackwell GPUs; importing the package does not initialize CUDA."""

__version__ = "0.1.2"


def attention(*args, **kwargs):
    from .api import attention as implementation

    return implementation(*args, **kwargs)


def prepare_fp8(*args, **kwargs):
    from .api import prepare_fp8 as implementation

    return implementation(*args, **kwargs)


def prepare_fp8_fused(*args, **kwargs):
    from .api import prepare_fp8_fused as implementation

    return implementation(*args, **kwargs)


def attention_fp8(*args, **kwargs):
    from .api import attention_fp8 as implementation

    return implementation(*args, **kwargs)


def raw_forward(*args, **kwargs):
    from .api import raw_forward as implementation

    return implementation(*args, **kwargs)


__all__ = ["attention", "attention_fp8", "prepare_fp8", "prepare_fp8_fused", "raw_forward"]
