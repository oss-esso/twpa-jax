from __future__ import annotations

import numpy as np
import pytest

from tests.test_multitone_problem import _circuit
from twpa_solver.multitone.basis import (
    ToneIndex,
    build_autonomous_torus_basis,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.schur import build_multitone_schur_problem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.multitone.torus import TorusProblem
from twpa_solver.pump.solver import bordered_solve_refined


def _torus_problem() -> TorusProblem:
    basis = build_autonomous_torus_basis(2.0e10, 1.0e9, [1], 1)
    drive = MultiToneDrive(basis.pump_tone, 0, 1.0e-9).to_coeffs(basis, 2)
    base = FullMultiToneProblem(
        _circuit(), basis, AffineSourcePath.pump_turn_on(drive)
    )
    return TorusProblem(base, (1,), 1, 1.0e9)


def test_torus_problem_exposes_square_unbordered_jacobian() -> None:
    problem = _torus_problem()
    state = problem.full_problem().zeros()

    residual = problem.residual_vector(state)
    jacobian = problem.jacobian(state)

    assert residual.size == problem.unknown_size
    assert jacobian.shape == (problem.unknown_size - 1, problem.unknown_size - 1)
    assert ToneIndex(0, 1) in problem.basis.tones


def test_torus_anchor_is_the_imaginary_generator_coordinate() -> None:
    problem = _torus_problem()
    state = problem.full_problem().zeros()
    state[problem.basis.index_of(ToneIndex(0, 1)), 0] = 1.0 + 2.0j

    assert problem.anchor(state) == 2.0


def test_phase_anchor_alone_admits_the_period1_branch() -> None:
    """Pin the defect the amplitude constraint exists to remove.

    With the q != 0 sector exactly zero the anchor is satisfied identically,
    the residual carries no q != 0 content, and the residual does not depend
    on omega_a at all.  The period-1 state is therefore an exact root of the
    phase-anchored bordered system for any generator frequency.
    """
    problem = _torus_problem()
    rng = np.random.default_rng(0)
    state = problem.full_problem().zeros()
    pump_rows = [i for i, t in enumerate(problem.basis.tones) if t.q == 0]
    generator_rows = problem.generator_rows()
    for row in pump_rows:
        state[row] = rng.normal(size=state.shape[1]) * 1e-12

    assert problem.anchor(state) == 0.0
    residual = problem.residual_coeffs(state)
    assert np.linalg.norm(residual[generator_rows]) == 0.0

    far = problem.residual_coeffs(state, 0.37 * problem.basis.omega_p)
    np.testing.assert_array_equal(far, residual)


def test_amplitude_constraint_rejects_the_period1_branch() -> None:
    problem = _torus_problem()
    state = problem.full_problem().zeros()

    assert problem.amplitude_residual(state, 1.0e-9) == -1.0e-9

    state[problem.basis.index_of(ToneIndex(0, 1)), 0] = 1.0e-9
    assert problem.amplitude_residual(state, 1.0e-9) == 0.0


def test_amplitude_solve_refuses_a_period1_start() -> None:
    problem = _torus_problem()
    state = problem.full_problem().zeros()

    with pytest.raises(ValueError, match="period-1 branch"):
        problem.solve_newton_amplitude(state, 1.0e-9)


def test_torus_bordered_step_matches_dense_reference() -> None:
    matrix = np.array(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.0, 0.3],
            [0.0, -0.2, 1.5],
        ]
    )
    parameter_column = np.array([0.4, -0.3, 0.2])
    anchor_row = np.array([0.1, -0.6, 0.5])
    residual = np.array([0.8, -0.4, 0.2])
    anchor_value = 0.35
    bordered = np.block(
        [[matrix, parameter_column[:, None]], [anchor_row[None, :], np.zeros((1, 1))]]
    )
    expected = np.linalg.solve(
        bordered, np.concatenate((-residual, np.asarray([-anchor_value])))
    )

    result = bordered_solve_refined(
        lambda value: matrix @ value,
        lambda value: np.linalg.solve(matrix, value),
        residual,
        anchor_value,
        -parameter_column,
        lambda value: float(anchor_row @ value),
        0.0,
    )

    assert result is not None
    actual = np.concatenate((result[0], np.asarray([result[1]])))
    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10)


def test_torus_accepts_schur_problem_and_records_anchor_mapping() -> None:
    full = _torus_problem().base_problem
    reduced = build_multitone_schur_problem(full, [0])
    problem = TorusProblem(reduced, (1,), 1, 1.0e9, node_ref=1)

    assert problem.is_schur
    assert problem.full_problem().n == reduced.n
    assert problem.anchor_full_node == 1
