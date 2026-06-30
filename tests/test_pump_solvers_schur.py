"""Tests for the Schur-reduced / matrix-free pump-solver backends.

Covers, per the backend spec:
  * Schur algebra on a small random sparse linear system (full vs reduced agree).
  * Schur pump fixture on a toy ladder (full pump == reconstructed Schur pump on
    retained AND eliminated nodes).
  * JVP: analytic AFT == finite-difference directional derivative.
  * Metadata: a reconstructed Schur pump solution writes the same .npz exp09
    loads (full-node X_real/X_imag + pump_modes).
The legacy exp08 tests are exercised separately (test_exp08_*).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EXP = Path(__file__).resolve().parents[1] / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import exp08_full_ipm_pump_solve as exp08  # noqa: E402
from pump_solvers import jvp_backends as jb  # noqa: E402
from pump_solvers.schur_operators import SchurReducedProblem  # noqa: E402
from pump_solvers.schur_partition import (  # noqa: E402
    assemble_schur_complements,
    back_substitute_full,
    build_partition,
    reduced_linear_apply,
)


# --------------------------------------------------------------------------- #
# Toy pump fixture: a tiny LC ladder with one Josephson branch.
# --------------------------------------------------------------------------- #
def _toy_problem(current_a: float = 8e-7, nt: int = 16):
    """3-node chain: node 0 (port, JJ-incident), node 1 (internal linear),
    node 2 (JJ-incident). The single JJ branch spans nodes 0 and 2, so node 1 is
    purely linear-internal and must be eliminated by the Schur partition."""
    n = 3
    K = sp.csr_matrix(np.array([
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0],
    ]) * 1e9)
    C = sp.eye(n, format="csr") * 1e-15
    G = sp.eye(n, format="csr") * 1e-3
    # One Josephson branch between nodes 0 and 2 -> node 1 is eliminable.
    Bphi = sp.csr_matrix(np.array([[1.0], [0.0], [-1.0]]))
    branch = exp08.JosephsonBranchArray(Ic=np.array([1.0e-6]), phi0=3.29e-16)
    omega = 2.0 * np.pi * 7e9
    grid = exp08.HarmonicGrid(modes=np.array([1, 3, 5]), nt=nt, omega=omega)
    full = exp08.FullIPMPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi, branch=branch, grid=grid,
        pump_node_index=0, pump_current_a=current_a, source_mode=1,
    )
    return full


def _settings(precond: str) -> exp08.NewtonKrylovSettings:
    return exp08.NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=30, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=40, gmres_maxiter=60, min_alpha=1.0 / 1024.0,
        preconditioner=precond, compute_time_residual=False, verbose=False,
        continuation_predictor="none", jvp_mode="aft",
    )


# --------------------------------------------------------------------------- #
# 1. Schur algebra on a random sparse linear system.
# --------------------------------------------------------------------------- #
def test_schur_algebra_matches_full_linear_solve():
    rng = np.random.default_rng(0)
    n, p = 40, 25
    A = sp.random(n, n, density=0.15, random_state=rng, format="csr")
    A = (A + A.T) + sp.eye(n) * 5.0  # well-conditioned
    A = A.tocsc().astype(np.complex128)
    retained = np.arange(n - p)
    eliminated = np.arange(n - p, n)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b[eliminated] = 0.0  # source only on retained, as in the pump problem

    x_full = spla.spsolve(A, b)

    Ann = A[retained][:, retained]
    Ane = A[retained][:, eliminated]
    Aen = A[eliminated][:, retained]
    Aee = A[eliminated][:, eliminated].tocsc()
    lu = spla.splu(Aee)
    # Schur complement solve for retained, then back-substitute eliminated.
    Sc = Ann - Ane @ sp.csr_matrix(lu.solve(Aen.toarray()))
    xn = spla.spsolve(sp.csc_matrix(Sc), b[retained])
    xe = lu.solve(-(Aen @ xn))

    assert np.linalg.norm(xn - x_full[retained]) / np.linalg.norm(x_full[retained]) < 1e-10
    assert np.linalg.norm(xe - x_full[eliminated]) / np.linalg.norm(x_full[eliminated]) < 1e-10


# --------------------------------------------------------------------------- #
# 2. Schur pump fixture: full pump == reconstructed Schur pump.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("apply_mode", ["assembled", "matrix_free"])
def test_schur_pump_matches_full_pump(apply_mode):
    full = _toy_problem()
    part = build_partition(full._linear_blocks, full.Bphi, [0, 2])
    assert part.eliminated.tolist() == [1]  # node 1 is the only eliminable node

    Xf, repF = exp08.HarmonicNewtonKrylovSolver(
        _settings("real_coupled")).solve_continuation(full, continuation_steps=12)
    assert repF[-1].converged

    schur = SchurReducedProblem(full=full, partition=part, linear_apply_mode=apply_mode)
    Xn, repS = exp08.HarmonicNewtonKrylovSolver(
        _settings("mean_tangent")).solve_continuation(schur, continuation_steps=12)
    assert repS[-1].converged
    X_recon = schur.reconstruct_full(Xn)

    rel = np.linalg.norm(X_recon - Xf) / np.linalg.norm(Xf)
    assert rel < 1e-8, f"reconstructed pump differs from full by {rel:.2e}"
    # eliminated node reconstructed, not just retained.
    assert np.linalg.norm(X_recon[:, part.eliminated] - Xf[:, part.eliminated]) \
        / np.linalg.norm(Xf[:, part.eliminated]) < 1e-8
    # full time residual of the reconstructed solution is tiny.
    assert schur.full_time_residual_rel(Xn, 1.0) < 1e-6


def test_assembled_and_matrixfree_linear_apply_agree():
    full = _toy_problem()
    part = build_partition(full._linear_blocks, full.Bphi, [0, 2])
    assemble_schur_complements(part)
    rng = np.random.default_rng(3)
    for h in range(full.H):
        vn = rng.standard_normal(part.m) + 1j * rng.standard_normal(part.m)
        a = part.schur[h] @ vn
        b = reduced_linear_apply(part, h, vn)
        assert np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30) < 1e-10


# --------------------------------------------------------------------------- #
# 3. JVP analytic vs finite difference.
# --------------------------------------------------------------------------- #
def test_analytic_jvp_matches_fd_full():
    # Evaluate at the converged (physical, sub-Ic) solution: the Josephson phase
    # psi/phi0 is O(1) there, so the finite difference is well posed.
    full = _toy_problem()
    X, rep = exp08.HarmonicNewtonKrylovSolver(
        _settings("real_coupled")).solve_continuation(full, continuation_steps=12)
    assert rep[-1].converged
    assert jb.jvp_relative_error(full, X, seed=2) < 1e-6


def test_analytic_jvp_matches_fd_schur():
    full = _toy_problem()
    part = build_partition(full._linear_blocks, full.Bphi, [0, 2])
    schur = SchurReducedProblem(full=full, partition=part)
    Xn, rep = exp08.HarmonicNewtonKrylovSolver(
        _settings("mean_tangent")).solve_continuation(schur, continuation_steps=12)
    assert rep[-1].converged
    assert jb.jvp_relative_error(schur, Xn, seed=5) < 1e-6


# --------------------------------------------------------------------------- #
# 4. Metadata: reconstructed Schur pump writes an exp09-loadable .npz.
# --------------------------------------------------------------------------- #
def test_schur_solution_writes_exp09_compatible_npz(tmp_path):
    full = _toy_problem()
    part = build_partition(full._linear_blocks, full.Bphi, [0, 2])
    schur = SchurReducedProblem(full=full, partition=part)
    Xn, reports = exp08.HarmonicNewtonKrylovSolver(
        _settings("mean_tangent")).solve_continuation(schur, continuation_steps=12)
    X_recon = schur.reconstruct_full(Xn)
    meta = {"pump_modes": [1, 3, 5], "pump_basis": "positive_phasor"}
    exp08.write_results(tmp_path, X_recon, reports,
                        exp08.summarize_solution(full, X_recon), meta)

    npz = np.load(tmp_path / "pump_solution.npz")
    # exp09 expects full-node X_real/X_imag plus the pump-mode list.
    assert npz["X_real"].shape == (full.H, full.n)
    assert npz["X_imag"].shape == (full.H, full.n)
    assert npz["pump_modes"].tolist() == [1, 3, 5]
    X_loaded = npz["X_real"] + 1j * npz["X_imag"]
    assert np.allclose(X_loaded, X_recon)
