"""Tests for twpa_solver.signal.stability: the Tier-1 Floquet stability proxy.

estimate_sigma_min is pure linear algebra (no circuit/pump state), so it is
tested directly against a dense SVD ground truth on synthetic sparse
matrices -- no CircuitMatrices/pump fixtures needed.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.signal.stability import (
    estimate_sigma_min,
    local_minima,
    refine_complex_resonance,
    refine_singular_omega,
)


def _random_complex_sparse(n: int, seed: int, density: float = 0.6) -> sp.csc_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    mask = rng.random((n, n)) < density
    dense = dense * mask
    # Keep the matrix well-conditioned but non-singular: push the diagonal up.
    dense += 3.0 * np.eye(n)
    return sp.csc_matrix(dense)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_estimate_sigma_min_matches_dense_svd(seed: int) -> None:
    A = _random_complex_sparse(n=12, seed=seed)
    expected = np.linalg.svd(A.toarray(), compute_uv=False).min()

    est = estimate_sigma_min(A, iters=25, seed=seed)

    assert est.matrix_size == 12
    assert est.convergence_ratio == pytest.approx(1.0, abs=0.05)
    assert est.sigma_min == pytest.approx(expected, rel=1e-3)


def test_estimate_sigma_min_near_singular_matrix_is_small() -> None:
    n = 8
    A = sp.identity(n, format="lil", dtype=np.complex128)
    A[0, 0] = 1e-8
    A = A.tocsc()

    est = estimate_sigma_min(A, iters=20, seed=0)

    assert est.sigma_min == pytest.approx(1e-8, rel=1e-2)


def test_estimate_sigma_min_is_deterministic_for_fixed_seed() -> None:
    A = _random_complex_sparse(n=10, seed=7)
    est1 = estimate_sigma_min(A, iters=10, seed=42)
    est2 = estimate_sigma_min(A, iters=10, seed=42)
    assert est1.sigma_min == est2.sigma_min


def test_local_minima_finds_dip_and_excludes_endpoints() -> None:
    values = [5.0, 4.0, 1.0, 3.0, 0.5, 2.0, 4.0]
    idx = local_minima(values, k=8)
    assert 4 in idx  # the deepest interior dip (value 0.5)
    assert idx[0] == 4  # ranked shallowest-first by depth (smallest first)
    assert 0 not in idx and len(values) - 1 not in idx


def test_local_minima_respects_k_limit() -> None:
    values = [5.0, 1.0, 5.0, 2.0, 5.0, 0.5, 5.0]
    idx = local_minima(values, k=1)
    assert len(idx) == 1
    assert idx[0] == 5  # the global minimum among interior points


def test_refine_singular_omega_finds_known_complex_eigenvalue() -> None:
    # A(omega) = omega*I - M is singular exactly at omega = eigenvalue of M --
    # a linear-in-omega matrix pencil, same singularity structure as the
    # physical (quadratic-in-omega) conversion matrix, ground-truthable via
    # a dense eig.
    rng = np.random.default_rng(3)
    n = 10
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    eigvals = np.linalg.eigvals(M)
    target = complex(eigvals[np.argmax(np.abs(eigvals.imag))])

    def assemble(omega: complex) -> sp.csc_matrix:
        return sp.csc_matrix(omega * np.eye(n, dtype=np.complex128) - M)

    omega0 = target + (0.05 + 0.05j)
    omega1 = target + (0.1 - 0.03j)
    result = refine_singular_omega(assemble, omega0, omega1, max_iters=50, tol=1e-10)

    assert result.converged
    assert abs(result.omega - target) < 1e-6
    assert result.residual < 1e-6


def test_refine_complex_resonance_rejects_non_analytic_loss_model() -> None:
    with pytest.raises(ValueError, match="not analytic"):
        refine_complex_resonance(
            circuit=None,
            khat=None,
            omega_p=1.0,
            ms=[0],
            signal_ghz_guess=1.0,
            loss_model="conductance_abs_omega",
        )
