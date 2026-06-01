"""Capability reporting for optional WSL2 accelerator experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class AcceleratorCapabilities:
    """Installed optional backends and the active JAX platform."""

    jax_backend: str
    jax_devices: tuple[str, ...]
    cudaq_available: bool
    cuquantum_available: bool

    @property
    def has_jax_gpu(self) -> bool:
        return self.jax_backend in {"gpu", "cuda"}

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "has_jax_gpu": self.has_jax_gpu}


def detect_accelerator_capabilities() -> AcceleratorCapabilities:
    """
    Detect optional accelerator packages.

    CUDA-Q and cuQuantum are intentionally not imported. Importing this module
    remains safe on the supported Windows CPU baseline.
    """
    import jax

    return AcceleratorCapabilities(
        jax_backend=str(jax.default_backend()),
        jax_devices=tuple(str(device) for device in jax.devices()),
        cudaq_available=find_spec("cudaq") is not None,
        cuquantum_available=find_spec("cuquantum") is not None,
    )


__all__ = ["AcceleratorCapabilities", "detect_accelerator_capabilities"]

