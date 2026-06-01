"""Numerical runtime configuration helpers."""

from __future__ import annotations


def enable_x64() -> None:
    """Enable JAX 64-bit mode before arrays are created."""
    import jax

    jax.config.update("jax_enable_x64", True)
