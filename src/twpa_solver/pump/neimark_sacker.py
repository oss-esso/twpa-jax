"""Nonlinear criticality contraction for a harmonic-balance NS candidate."""

from __future__ import annotations

import numpy as np

from twpa_solver.pump.bifurcation import d2n_coeffs, d3n_coeffs


def _reshape_hill_vector(vector: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if values.size == int(np.prod(shape)):
        return values.reshape(shape)
    if values.size == 2 * int(np.prod(shape)):
        size = int(np.prod(shape))
        return (values[:size] + 1j * values[size:]).reshape(shape)
    raise ValueError("hill_vector size does not match the harmonic-balance state")


def first_lyapunov_coefficient(
    problem: object,
    state: np.ndarray,
    hill_vector: np.ndarray,
    omega_p: float,
    *,
    adjoint_vector: np.ndarray | None = None,
) -> float:
    """Return the cubic NS criticality coefficient in the pinned sign convention.

    The convention is ``l1 < 0`` for a supercritical branch.  The supplied
    Hill and adjoint vectors are normalized in the real-packed Euclidean
    metric.  The coefficient uses the exact second and third directional
    derivatives already used by the pump bifurcation diagnostics; this avoids
    a finite-difference tensor and keeps the contraction on the production
    nonlinear operator.  A missing adjoint uses the critical vector as the
    test covector, which is a valid diagnostic only for a self-adjoint local
    reference problem and is reported by callers as such.
    """
    if omega_p <= 0.0:
        raise ValueError("omega_p must be positive")
    X = np.asarray(state, dtype=np.complex128)
    if X.ndim != 2:
        raise ValueError("state must be a two-dimensional coefficient array")
    vector = _reshape_hill_vector(hill_vector, X.shape)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        raise ValueError("hill_vector must be nonzero")
    vector = vector / vector_norm
    if adjoint_vector is None:
        adjoint = vector
    else:
        adjoint = _reshape_hill_vector(adjoint_vector, X.shape)
        adjoint_norm = float(np.linalg.norm(adjoint))
        if adjoint_norm == 0.0:
            raise ValueError("adjoint_vector must be nonzero")
        adjoint = adjoint / adjoint_norm

    second = d2n_coeffs(problem, X, vector)
    third = d3n_coeffs(problem, X, vector)
    quadratic_projection = np.vdot(adjoint, second)
    cubic_projection = np.vdot(adjoint, third)
    # The quadratic projection contributes through the phase-normalization
    # correction.  Keeping it explicit makes the convention and diagnostics
    # reproducible even when the adjoint is supplied by a Hill backend.
    corrected = cubic_projection - 2.0 * abs(quadratic_projection) ** 2
    return float(np.real(corrected) / (2.0 * omega_p))
