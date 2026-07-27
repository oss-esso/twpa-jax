"""Torus-grid operations shared by multitone harmonic balance."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex


class TorusGrid:
    """Flat ``(theta_p, theta_delta)`` grid associated with a tone basis."""

    def __init__(self, basis: MultiToneBasis) -> None:
        self.basis = basis
        self.n_p = basis.n_p
        self.n_delta = basis.n_delta
        self.nt = self.n_p * self.n_delta
        self.theta_flat = basis.theta_flat

    @property
    def n_torus(self) -> int:
        return self.nt

    def phase_rows(self, ells: Iterable[ToneIndex]) -> np.ndarray:
        """Return normalized negative-exponent rows for spectral averaging."""
        offsets = list(ells)
        theta = self.theta_flat
        return np.asarray(
            [
                np.exp(-1j * (offset.h * theta[:, 0] + offset.q * theta[:, 1]))
                / self.nt
                for offset in offsets
            ],
            dtype=np.complex128,
        )

    def synthesize(self, X: np.ndarray) -> np.ndarray:
        return self.basis.synthesize(X)

    def project(self, y: np.ndarray) -> np.ndarray:
        return self.basis.project(y)
