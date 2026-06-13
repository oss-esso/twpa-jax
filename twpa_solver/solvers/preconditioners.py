"""Preconditioner builders for TWPA nonlinear residuals."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from twpa_solver.residuals.aft_hb import PumpAFTResidual


def build_linear_passive_preconditioner(residual: PumpAFTResidual) -> LinearOperator:
    """Build a dense block inverse of the passive linear HB operator."""
    matrix = build_linear_passive_operator_matrix(residual)
    inverse = np.linalg.pinv(matrix)
    return LinearOperator(
        matrix.shape,
        matvec=lambda value: inverse @ np.asarray(value, dtype=float),
        dtype=float,
    )


def build_linear_passive_operator_matrix(residual: PumpAFTResidual) -> np.ndarray:
    """Return the passive linearized residual matrix in cos/sin coefficient order."""
    model = residual.model
    n = model.num_nodes
    total = residual.size
    matrix = np.zeros((total, total), dtype=float)
    k0 = model.zero_signal_stiffness()
    for h in range(1, residual.harmonics + 1):
        omega_h = h * residual.omega
        a = k0 - omega_h**2 * model.capacitance_f
        g = omega_h * model.conductance_s
        block = np.block([[a, g], [-g, a]]) / residual.config.residual_scale_a
        start = (h - 1) * 2 * n
        stop = start + 2 * n
        matrix[start:stop, start:stop] = block
    return matrix
