"""Validation tests for the matrix-free periodic-orbit Floquet layer."""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scripts.h1_transient_branch_transfer import TransientSystem
from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.core.nonlinear import JosephsonBranchLaw
from twpa_solver.stability import (
    build_hb_periodic_orbit,
    build_monodromy_operator,
    classify_multiplier,
    compute_floquet_multipliers,
    track_multiplier_branches,
)


def _linear_system(
    c_values: list[float],
    g_matrix: np.ndarray,
    k_matrix: np.ndarray,
    bphi: np.ndarray | None = None,
) -> TransientSystem:
    n = len(c_values)
    if bphi is None:
        bphi = np.zeros((n, 1))
    circuit = CircuitMatrices(
        C=sp.diags(c_values, format="csr"),
        G=sp.csr_matrix(g_matrix),
        K=sp.csr_matrix(k_matrix),
        Bphi=sp.csr_matrix(bphi),
        Ic=np.zeros(bphi.shape[1]),
        phi0=1.0,
    )
    differential = np.flatnonzero(np.diff(circuit.C.indptr) != 0)
    algebraic = np.setdiff1d(np.arange(n), differential)
    c_factor = spla.splu(circuit.C[differential][:, differential].tocsc())
    g_factor = (
        spla.splu(circuit.G[algebraic][:, algebraic].tocsc())
        if algebraic.size
        else None
    )
    return TransientSystem(
        circuit=circuit,
        branch=JosephsonBranchLaw(circuit.Ic, circuit.phi0),
        omega=3.0,
        pump_node=0,
        differential=differential,
        algebraic=algebraic,
        c_factor=c_factor,
        g_alg_factor=g_factor,
    )


def _operator(system: TransientSystem, steps: int = 64):
    orbit = build_hb_periodic_orbit(
        np.zeros((1, system.n), dtype=complex),
        [1],
        system.omega,
        system.phi0,
        steps_per_period=steps,
    )
    return build_monodromy_operator(
        system,
        orbit,
        max_step_theta=2.0 * math.pi / steps,
    )


def test_linear_monodromy_action_matches_exact_matrix_exponential() -> None:
    system = _linear_system(
        [1.0, 0.7],
        [[0.2, 0.0], [0.0, 0.4]],
        [[2.0, 0.0], [0.0, 1.4]],
    )
    operator = _operator(system, steps=512)
    theta_matrix = np.block(
        [
            [np.zeros((2, 2)), np.eye(2)],
            [
                -np.diag([2.0, 1.4]) @ np.diag([1.0, 1.0 / 0.7]) / 3.0**2,
                -np.diag([0.2, 0.4]) @ np.diag([1.0, 1.0 / 0.7]) / 3.0,
            ],
        ]
    )
    vector = np.array([0.3, -0.5, 0.7, -0.2])
    expected = la.expm(theta_matrix * 2.0 * math.pi) @ vector
    actual = operator.matvec(vector)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_matrix_free_floquet_eigenvalues_match_linear_reference() -> None:
    system = _linear_system(
        [1.0, 0.7],
        [[0.2, 0.0], [0.0, 0.4]],
        [[2.0, 0.0], [0.0, 1.4]],
    )
    operator = _operator(system, steps=96)
    result = compute_floquet_multipliers(operator, eigenvalues=2, tol=1e-10)
    theta_matrix = np.block(
        [
            [np.zeros((2, 2)), np.eye(2)],
            [
                -np.diag([2.0, 1.4]) @ np.diag([1.0, 1.0 / 0.7]) / 3.0**2,
                -np.diag([0.2, 0.4]) @ np.diag([1.0, 1.0 / 0.7]) / 3.0,
            ],
        ]
    )
    expected = np.linalg.eigvals(la.expm(theta_matrix * 2.0 * math.pi))
    expected = expected[np.argsort(-np.abs(expected))][:2]
    np.testing.assert_allclose(
        np.sort_complex(result.multipliers),
        np.sort_complex(expected),
        rtol=5e-4,
        atol=5e-6,
    )
    assert result.matvecs > 0
    assert result.spectral_radius < 1.0


def test_finite_difference_monodromy_action_matches_linear_flow() -> None:
    system = _linear_system([1.0], [[0.2]], [[2.0]])
    operator = _operator(system, steps=512)
    vector = np.array([0.4, -0.8])
    theta_matrix = np.array([[0.0, 1.0], [-2.0 / 9.0, -0.2 / 3.0]])
    flow = la.expm(theta_matrix * 2.0 * math.pi)
    eps = 1.0e-7
    base = np.zeros(2)
    finite_difference = (flow @ (base + eps * vector) - flow @ base) / eps
    np.testing.assert_allclose(
        operator.matvec(vector), finite_difference, rtol=2e-5, atol=2e-5
    )


def test_trapezoid_monodromy_converges_with_phase_step_refinement() -> None:
    system = _linear_system([1.0], [[0.2]], [[2.0]])
    coarse = _operator(system, steps=64)
    fine = _operator(system, steps=128)
    vector = np.array([0.4, -0.8])
    theta_matrix = np.array([[0.0, 1.0], [-2.0 / 9.0, -0.2 / 3.0]])
    exact = la.expm(theta_matrix * 2.0 * math.pi) @ vector
    coarse_error = np.linalg.norm(coarse.matvec(vector) - exact)
    fine_error = np.linalg.norm(fine.matvec(vector) - exact)
    assert fine_error < coarse_error / 3.0


def test_dae_tangent_vectors_are_projected_to_the_algebraic_constraint() -> None:
    system = _linear_system(
        [1.0, 0.0],
        [[0.2, 0.0], [0.0, 1.0]],
        [[2.0, -0.5], [-0.5, 1.5]],
    )
    operator = _operator(system, steps=64)
    vector = np.array([0.3, -0.4, 0.8])
    consistent = operator.state_space.make_consistent(vector, operator.orbit.q[0])
    dq, _ = operator.state_space.unpack(consistent)
    constraint = system.circuit.K[system.algebraic] @ dq
    assert np.linalg.norm(constraint) < 1.0e-12
    mapped = operator.matvec(vector)
    mapped_consistent = operator.state_space.make_consistent(
        mapped, operator.orbit.q[0]
    )
    np.testing.assert_allclose(mapped, mapped_consistent, atol=1.0e-12)


def test_tracker_preserves_continuous_complex_branches() -> None:
    previous = np.array([0.99 + 0.01j, 0.5 - 0.1j])
    current = np.array([0.49 - 0.09j, 0.98 + 0.03j])
    tracked = track_multiplier_branches(previous, current)
    np.testing.assert_allclose(tracked, np.array([current[1], current[0]]))


def test_multiplier_crossing_classifier_is_conservative() -> None:
    assert classify_multiplier(0.8 + 0.0j) == "STABLE_MODE"
    assert classify_multiplier(1.01 + 0.0j) == "UNSTABLE_MODE"
    assert classify_multiplier(-1.0 + 0.0j) == "-1_CROSSING_CANDIDATE"
    assert classify_multiplier(1.0j) == "COMPLEX_UNIT_CIRCLE_CANDIDATE"
