"""Tests for the advanced intra-cell continuation methods added to the pump
solver: the tangent (Euler) predictor, pseudo-transient continuation, and
pseudo-arclength continuation.

A tiny one-node LC + Josephson problem (shared with
``test_exp08_seed_adaptive_warmstart``) runs the full Newton-Krylov stack in
milliseconds while exercising the real residual / JVP / real-coupled
preconditioner / arclength code paths.
"""

from __future__ import annotations

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
    X_arc, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80)
    assert info["reached_target"]
    np.testing.assert_allclose(X_arc, X_direct, atol=1e-5)


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
