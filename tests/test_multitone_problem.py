from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive


def _circuit() -> CircuitMatrices:
    return CircuitMatrices(
        C=sp.eye(2, format="csr") * 1e-15,
        G=sp.eye(2, format="csr") * 1e-3,
        K=sp.csr_matrix([[2e9, -1e9], [-1e9, 2e9]]),
        Bphi=sp.csr_matrix([[1.0], [-1.0]]),
        Ic=np.array([1e-6]),
        port_to_index={1: 0},
    )


def _problem() -> FullMultiToneProblem:
    basis = build_three_tone_basis(2.0e10, 1.0e9)
    drive = MultiToneDrive(basis.pump_tone, 0, 1e-9).to_coeffs(basis, 2)
    path = AffineSourcePath.pump_turn_on(drive)
    return FullMultiToneProblem(_circuit(), basis, path)


def test_affine_sources_preserve_drive_normalization() -> None:
    problem = _problem()
    source = problem.source_coeffs(1.0)

    assert source[problem.basis.index_of(problem.basis.pump_tone), 0] == 0.5e-9
    np.testing.assert_array_equal(problem.source_coeffs(0.0), np.zeros_like(source))
    np.testing.assert_array_equal(problem.source_delta_coeffs(), source)
    np.testing.assert_allclose(
        problem.source_time(1.0)[:, 0],
        1e-9 * np.cos(problem.basis.theta_flat[:, 0]),
    )


def test_multitone_jvp_matches_central_difference() -> None:
    problem = _problem()
    rng = np.random.default_rng(12)
    X = rng.normal(size=(problem.H, problem.n)) * 1e-18
    V = (rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape)) * 1e-18
    tangent = problem.tangent_state(X)
    analytic = problem.jvp_coeffs_with_tangent(V, tangent)
    step = 1e-3
    finite_difference = (
        problem.residual_coeffs(X + step * V, 0.0)
        - problem.residual_coeffs(X - step * V, 0.0)
    ) / (2.0 * step)

    relative = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(finite_difference)
    assert relative < 1e-6


def test_spectral_tangent_matches_aft_jvp() -> None:
    problem = _problem()
    rng = np.random.default_rng(5)
    X = rng.normal(size=(problem.H, problem.n)) * 1e-18
    V = rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape)
    tangent = problem.tangent_state(X)
    spectral = problem.spectral_tangent_state(tangent)

    aft = problem.jvp_coeffs_with_tangent(V, tangent)
    spectral_jvp = problem.jvp_coeffs_with_spectral_tangent(V, spectral)

    assert spectral.khat
    np.testing.assert_allclose(spectral_jvp, aft, rtol=1e-10, atol=1e-18)
