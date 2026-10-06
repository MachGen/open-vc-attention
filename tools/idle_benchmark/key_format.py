"""Data-only encoding of the frozen attention dispatch cache keys."""

import cutlass

TYPES = {name: getattr(cutlass, name) for name in ("BFloat16", "Float8E4M3FN")}


def encode(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, tuple):
        return {"tuple": [encode(item) for item in value]}
    for name, dtype in TYPES.items():
        if value is dtype:
            return {"cutlass_type": name}
    raise TypeError(f"Unsupported cache-key type: {type(value)}")


def decode(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict) and set(value) == {"tuple"}:
        return tuple(decode(item) for item in value["tuple"])
    if isinstance(value, dict) and set(value) == {"cutlass_type"}:
        return TYPES[value["cutlass_type"]]
    raise ValueError("Unsupported cache-key encoding")
