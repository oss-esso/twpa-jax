"""JVP backends and a finite-difference sanity check.

Audit result: the production JVP (``jvp_coeffs_with_tangent``) is a
hand-coded analytic / alternating-frequency-time (AFT) Jacobian-vector product

    (J V)_k = D_k V_k + AFT[ Gamma(t) * (Bphi^T v(t)) ],

NOT a finite-difference residual difference and NOT autodiff. It captures the
full real-pump coupling -- including the conjugate ``K_{k+q} conj(V_q)`` term --
implicitly through the real-waveform synthesis. This module provides:

* ``analytic_jvp`` -- thin wrapper over the problem's analytic AFT JVP (default).
* ``fd_jvp`` -- complex-coefficient finite-difference directional derivative,
  used only to sanity-check the analytic JVP (not for production solves).
* ``jvp_relative_error`` -- compares the two on a random perturbation.
* optional JAX path flag (``jax_available``); the analytic JVP is already exact,
  so JAX is offered only as an independent cross-check, never a default.
"""

from __future__ import annotations

import numpy as np

try:  # optional, never a hard dependency
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    JAX_AVAILABLE = False


def jax_available() -> bool:
    return JAX_AVAILABLE


def analytic_jvp(problem, X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Exact analytic/AFT JVP (the production path)."""
    return problem.jvp_coeffs(X, V)


def fd_jvp(
    problem,
    X: np.ndarray,
    V: np.ndarray,
    source_scale: float = 1.0,
    eps: float = 1e-7,
) -> np.ndarray:
    """Central finite-difference directional derivative of the residual.

    (R(X + eps V) - R(X - eps V)) / (2 eps). The residual's explicit linear
    source cancels, so this equals J V up to O(eps^2). Used only for testing.
    """
    scale = eps * max(1.0, float(np.linalg.norm(X)) / max(np.linalg.norm(V), 1e-30))
    Rp = problem.residual_coeffs(X + scale * V, source_scale)
    Rm = problem.residual_coeffs(X - scale * V, source_scale)
    return (Rp - Rm) / (2.0 * scale)


def jvp_relative_error(
    problem,
    X: np.ndarray,
    V: np.ndarray | None = None,
    *,
    seed: int = 0,
    eps: float = 1e-7,
) -> float:
    """Relative L2 error between the analytic JVP and the FD JVP at ``X``."""
    rng = np.random.default_rng(seed)
    if V is None:
        V = (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
        V *= max(np.linalg.norm(X), 1e-12) / max(np.linalg.norm(V), 1e-30)
    ja = analytic_jvp(problem, X, V)
    jf = fd_jvp(problem, X, V, eps=eps)
    return float(np.linalg.norm(ja - jf) / max(np.linalg.norm(jf), 1e-30))
