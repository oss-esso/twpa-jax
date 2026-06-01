"""Repository-local Python startup hooks."""

from __future__ import annotations

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except Exception:
    pass
