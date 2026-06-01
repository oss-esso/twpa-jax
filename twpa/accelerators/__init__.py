"""Optional accelerator capability detection without import-time GPU dependencies."""

from .backend import AcceleratorCapabilities, detect_accelerator_capabilities

__all__ = ["AcceleratorCapabilities", "detect_accelerator_capabilities"]

