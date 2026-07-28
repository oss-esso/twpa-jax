from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.schur import build_multitone_schur_problem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver, NewtonKrylovSettings


def _problems() -> tuple[FullMultiToneProblem, object]:
    circuit = CircuitMatrices(
        C=sp.eye(3, format="csr") * 1e-15,
        G=sp.eye(3, format="csr") * 1e-3,
        K=sp.csr_matrix([[2e9, -1e9, 0], [-1e9, 2e9, -1e9], [0, -1e9, 2e9]]),
        Bphi=sp.csr_matrix([[1.0], [0.0], [-1.0]]),
        Ic=np.array([1e-6]),
        port_to_index={1: 0, 2: 2},
    )
    basis = build_three_tone_basis(2.0e10, 1.0e9)
    source = MultiToneDrive(basis.pump_tone, 0, 1e-9).to_coeffs(basis, 3)
    full = FullMultiToneProblem(circuit, basis, AffineSourcePath.pump_turn_on(source))
    return full, build_multitone_schur_problem(full, [0, 2])


def _settings() -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10,
        max_newton=20,
        gmres_rtol=1e-8,
        gmres_atol=0.0,
        gmres_restart=20,
        gmres_maxiter=40,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )


def test_multitone_schur_root_reconstructs_full_root() -> None:
    full, schur = _problems()
    solver = HarmonicNewtonKrylovSolver(_settings())
    full_state, full_reports = solver.solve_continuation(full, continuation_steps=4)
    schur_state, schur_reports = solver.solve_continuation(schur, continuation_steps=4)

    assert full_reports[-1].converged
    assert schur_reports[-1].converged
    reconstructed = schur.reconstruct_full(schur_state)
    np.testing.assert_allclose(reconstructed, full_state, rtol=1e-8, atol=1e-18)
    assert schur.full_time_residual_rel(schur_state, 1.0) < 1e-8


def test_multitone_schur_jvp_matches_finite_difference() -> None:
    full, schur = _problems()
    rng = np.random.default_rng(2)
    state = rng.normal(size=(full.H, schur.n)) * 1e-18
    direction = rng.normal(size=state.shape) * 1e-18
    analytic = schur.jvp_coeffs(state, direction)
    step = 1e-3
    finite_difference = (
        schur.residual_coeffs(state + step * direction, 0.0)
        - schur.residual_coeffs(state - step * direction, 0.0)
    ) / (2.0 * step)
    assert np.linalg.norm(analytic - finite_difference) / np.linalg.norm(analytic) < 1e-6
