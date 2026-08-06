"""Tests for the advanced intra-cell continuation methods added to the pump
solver: the tangent (Euler) predictor, pseudo-transient continuation, and
pseudo-arclength continuation.

A tiny one-node LC + Josephson problem (shared with
``test_exp08_seed_adaptive_warmstart``) runs the full Newton-Krylov stack in
milliseconds while exercising the real residual / JVP / real-coupled
preconditioner / arclength code paths.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from twpa_solver.pump import hb as exp08  # noqa: E402


def _build_problem(pump_current: float, *, omega: float = 0.37):
    C = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    G = sp.csr_matrix(np.array([[0.01]], dtype=np.complex128))
    K = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Bphi = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
    grid = exp08.HarmonicGrid(modes=np.array([1, 2, 3]), nt=16, omega=omega)
    branch = exp08.JosephsonBranchArray(Ic=np.array([1.0], dtype=np.float64), phi0=1.0)
    return exp08.FullIPMPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi, branch=branch, grid=grid,
        pump_node_index=0, pump_current_a=pump_current,
    )


def _settings() -> exp08.NewtonKrylovSettings:
    return exp08.NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=30, gmres_rtol=1e-9, gmres_atol=0.0,
        gmres_restart=40, gmres_maxiter=200, min_alpha=1.0 / 1024.0,
        preconditioner="mean_tangent", compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
    )


def _solver() -> exp08.HarmonicNewtonKrylovSolver:
    return exp08.HarmonicNewtonKrylovSolver(_settings())


def test_tangent_predictor_beats_copy_near_the_branch() -> None:
    # Converge at lambda=0.5, then predict lambda=0.6. The exact tangent should
    # give a lower residual at the new lambda than simply copying the state.
    problem = _build_problem(pump_current=0.5)
    solver = _solver()
    X_half, rep = solver.solve_one(problem, problem.zeros(), 0.5)
    assert rep.converged
    d_lambda = 0.1
    pred = solver.tangent_predictor(problem, X_half, d_lambda)
    r_copy = problem.norms(X_half, 0.6, False)["coeff_rel"]
    r_tan = problem.norms(pred, 0.6, False)["coeff_rel"]
    assert r_tan < r_copy


def test_pseudo_transient_converges_from_zero() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X, reports = solver.solve_pseudo_transient(problem, problem.zeros(), delta0=1.0)
    assert reports[-1].converged
    assert problem.norms(X, 1.0, False)["coeff_rel"] < 1e-7


def test_arclength_reaches_target_lambda() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80)
    assert info["reached_target"]
    assert abs(lam - 1.0) < 1e-9
    # The target endpoint is linearly interpolated between the two straddling
    # arclength points, so its residual is small but not at Newton tolerance
    # (it is consumed as a warm guess for a final target solve downstream).
    assert problem.norms(X, 1.0, False)["coeff_rel"] < 1e-3


def test_arclength_matches_direct_solution() -> None:
    # The arclength solution at lambda=1 must equal the ordinary solve at full
    # drive (same branch, easy current).
    problem = _build_problem(pump_current=0.4)
    solver = _solver()
    X_direct, rep = solver.solve_one(problem, problem.zeros(), 1.0)
    assert rep.converged
    # ds=0.1 leaves ~7e-7 of linear-interpolation truncation at the target
    # crossing (interpolated between the two straddling arclength points, not
    # itself Newton-polished); ds=0.02 shrinks that below 1e-7 while still
    # converging in well under max_steps.
    X_arc, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0, max_steps=200)
    assert info["reached_target"]
    np.testing.assert_allclose(X_arc, X_direct, atol=1e-7)


def _build_scaled_problem(pump_current: float, s: float, *, omega: float = 0.37):
    # Residual D(w)X + Bphi*Ic*sin(psi/phi0) - S is exactly covariant under
    # X -> s*X when phi0 -> s*phi0, Ic -> s*Ic, pump_current -> s*pump_current
    # (C, G, K, lambda unchanged): psi/phi0 is invariant so the nonlinear term
    # scales by s via Ic, the linear term scales by s via X, and the source
    # scales by s via pump_current_a. So the solution scales by exactly s and
    # every accepted lambda along the branch is identical to the s=1 problem.
    C = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    G = sp.csr_matrix(np.array([[0.01]], dtype=np.complex128))
    K = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Bphi = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
    grid = exp08.HarmonicGrid(modes=np.array([1, 2, 3]), nt=16, omega=omega)
    branch = exp08.JosephsonBranchArray(Ic=np.array([s], dtype=np.float64), phi0=s)
    return exp08.FullIPMPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi, branch=branch, grid=grid,
        pump_node_index=0, pump_current_a=s * pump_current,
    )


def test_arclength_is_invariant_under_state_rescaling() -> None:
    solver = _solver()
    problem_unit = _build_scaled_problem(0.6, 1.0)
    X_unit, lam_unit, info_unit = solver.solve_arclength(
        problem_unit, problem_unit.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
    )

    s = 1e-15
    problem_scaled = _build_scaled_problem(0.6, s)
    X_scaled, lam_scaled, info_scaled = solver.solve_arclength(
        problem_scaled, problem_scaled.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
    )

    assert info_unit["reached_target"] == info_scaled["reached_target"]
    assert info_unit["reached_target"]
    assert abs(lam_scaled - lam_unit) < 1e-9
    np.testing.assert_allclose(X_scaled, s * X_unit, rtol=1e-4, atol=s * 1e-8)


def test_arclength_reports_state_scale() -> None:
    solver = _solver()
    problem = _build_scaled_problem(0.6, 1e-15)
    _X, _lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
    )
    assert info["state_scale"] is not None
    assert math.isfinite(info["state_scale"])
    assert info["state_scale"] > 0.0
    # The scale must actually track the state's units, not sit pinned at 1.0.
    assert info["state_scale"] < 1e-10


def test_scaled_two_point_arclength_reaches_higher_drive() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X0, rep0 = solver.solve_one(problem, problem.zeros(), 0.2)
    X1, rep1 = solver.solve_one(problem, X0, 0.3)
    assert rep0.converged and rep1.converged

    points, info = solver.trace_arclength_from_two_points(
        problem, X0, 0.2, X1, 0.3, ds=0.05, max_steps=30,
    )

    assert len(points) > 2
    assert max(lam for _X, lam in points) >= 1.0
    assert info["state_scale"] > 0.0
