"""Finite-tone drives and affine continuation paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex


@dataclass(frozen=True)
class MultiToneDrive:
    """A single positive-phasor current drive."""

    tone: ToneIndex
    node_index: int
    current_a: float | complex

    def to_coeffs(self, basis: MultiToneBasis, n_nodes: int) -> np.ndarray:
        """Return this drive as a coefficient array using the 0.5 convention."""
        coefficients = np.zeros((basis.n_tones, n_nodes), dtype=np.complex128)
        coefficients[basis.index_of(self.tone), self.node_index] = (
            0.5 * self.current_a
        )
        return coefficients


@dataclass(frozen=True)
class AffineSourcePath:
    """Source path ``S(tau) = source_start + tau * source_delta``."""

    source_start: np.ndarray
    source_delta: np.ndarray

    def __post_init__(self) -> None:
        start = np.asarray(self.source_start, dtype=np.complex128)
        delta = np.asarray(self.source_delta, dtype=np.complex128)
        if start.shape != delta.shape or start.ndim != 2:
            raise ValueError("affine source arrays must have equal 2D shapes")
        object.__setattr__(self, "source_start", start)
        object.__setattr__(self, "source_delta", delta)

    def source(self, tau: float) -> np.ndarray:
        return self.source_start + float(tau) * self.source_delta

    def derivative(self, tau: float = 0.0) -> np.ndarray:
        del tau
        return np.array(self.source_delta, copy=True)

    @classmethod
    def pump_turn_on(cls, source_p: np.ndarray) -> "AffineSourcePath":
        source = np.asarray(source_p, dtype=np.complex128)
        return cls(np.zeros_like(source), source)

    @classmethod
    def signal_turn_on(
        cls, source_p: np.ndarray, source_s: np.ndarray
    ) -> "AffineSourcePath":
        pump = np.asarray(source_p, dtype=np.complex128)
        signal = np.asarray(source_s, dtype=np.complex128)
        return cls(np.zeros_like(pump), pump + signal)

    @classmethod
    def signal_substep(
        cls,
        source_p: np.ndarray,
        source_s_old: np.ndarray,
        source_s_target: np.ndarray,
    ) -> "AffineSourcePath":
        pump = np.asarray(source_p, dtype=np.complex128)
        old = np.asarray(source_s_old, dtype=np.complex128)
        target = np.asarray(source_s_target, dtype=np.complex128)
        return cls(pump + old, target - old)
