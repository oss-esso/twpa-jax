"""Small HDF5 decoding helpers shared by readers and smoke tests."""

from __future__ import annotations

from typing import Any


def decode_h5_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return value


def decode_h5_string(value: Any) -> str:
    return str(decode_h5_scalar(value))
