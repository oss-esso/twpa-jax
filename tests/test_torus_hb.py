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
from twpa_solver.multitone.torus import (
    TorusProblem,
    apply_border_aware_preconditioner,
    apply_one_border_preconditioner,
    build_branch_lock_geometry,
    lattice_label_audit,
)
from twpa_solver.pump.solver import bordered_solve_refined
from twpa_solver.signal.stability import audit_loss_convention


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


def test_branch_lock_separates_finite_torus_from_period1() -> None:
    problem = _torus_problem()
    predictor = problem.full_problem().zeros()
    predictor[problem.basis.index_of(ToneIndex(1, 0)), 0] = 1.0e-12
    predictor[problem.basis.index_of(ToneIndex(0, 1)), 0] = 1.0e-10 + 2.0e-11j
    q_values = np.asarray([tone.q for tone in problem.basis.tones])
    geometry = build_branch_lock_geometry(predictor, q_values)

    period1 = predictor.copy()
    period1[problem.basis.index_of(ToneIndex(0, 1)), 0] = 0.0

    assert geometry.value(predictor) == pytest.approx(0.0, abs=1.0e-15)
    assert abs(geometry.value(period1)) > 1.0e-3
    assert abs(geometry.phase_projection) > 0.1
    assert abs(geometry.radial_projection) > 0.1


def test_branch_switch_uses_the_critical_direction_before_arclength(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _torus_problem()
    state = problem.full_problem().zeros()
    state[problem.basis.index_of(ToneIndex(1, 0)), 0] = 1.0e-12
    mode = np.zeros_like(state)
    mode[problem.basis.index_of(ToneIndex(0, 1)), 0] = 1.0 + 0.25j
    captured: dict[str, object] = {}

    def fake_arclength(
        X0: np.ndarray,
        *,
        previous_X: np.ndarray,
        previous_omega_a: float,
        previous_source_tau: float,
        tangent: np.ndarray,
        step_size: float,
        **kwargs: object,
    ) -> tuple[np.ndarray, float, float, dict[str, object], np.ndarray]:
        captured["X0"] = X0
        captured["previous_X"] = previous_X
        captured["tangent"] = tangent
        captured["step_size"] = step_size
        captured["phase_reference"] = kwargs["phase_reference"]
        return X0, previous_omega_a, previous_source_tau, {
            "converged": True,
        }, tangent

    monkeypatch.setattr(problem, "solve_torus_arclength", fake_arclength)
    result = problem.solve_torus_branch_switch(
        state,
        omega_a_ns=problem.omega_a,
        source_tau_ns=1.0,
        perturbation=mode,
        step_size=0.05,
    )

    assert result[3]["converged"] is True
    np.testing.assert_array_equal(captured["previous_X"], state)
    assert float(captured["step_size"]) == 0.05
    predicted = np.asarray(captured["X0"])
    assert np.linalg.norm(predicted[problem.generator_rows()]) > 0.0
    np.testing.assert_array_equal(captured["phase_reference"], mode)


def test_omitted_q_residual_is_evaluated_on_a_larger_lattice() -> None:
    problem = _torus_problem()
    state = problem.full_problem().zeros()
    state[problem.basis.index_of(ToneIndex(1, 0)), 0] = 1.0e-12
    state[problem.basis.index_of(ToneIndex(0, 1)), 0] = 2.0e-13

    result = problem.omitted_q_residual(state, evaluation_q_max=2)

    assert result["omitted_q_max"] == 2.0
    assert result["omitted_q_residual_abs"] >= 0.0
    assert result["omitted_q_residual_rel"] >= 0.0


def test_omitted_q_residual_supports_schur_coordinates() -> None:
    full = _torus_problem().base_problem
    reduced = build_multitone_schur_problem(full, [0])
    problem = TorusProblem(reduced, (1,), 1, 1.0e9, node_ref=1)
    state = problem.full_problem().zeros()
    state[problem.basis.index_of(ToneIndex(1, 0)), 0] = 1.0e-12
    state[problem.basis.index_of(ToneIndex(0, 1)), 0] = 2.0e-13

    result = problem.omitted_q_residual(state, evaluation_q_max=2)

    assert result["omitted_q_max"] == 2.0
    assert result["omitted_q_residual_abs"] >= 0.0
    assert result["omitted_q_residual_rel"] >= 0.0


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


def test_border_aware_preconditioner_matches_dense_solve() -> None:
    state_matrix = np.array(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.0, 0.3],
            [0.0, -0.2, 1.5],
        ]
    )
    border_columns = np.array(
        [
            [0.4, -0.2],
            [-0.3, 0.5],
            [0.2, 0.1],
        ]
    )
    constraint_rows = np.array(
        [
            [0.1, -0.6, 0.5],
            [0.2, 0.3, -0.4],
        ]
    )
    border_matrix = np.array([[0.0, 0.0], [0.7, -0.1]])
    rhs = np.array([0.8, -0.4, 0.2, 0.1, -0.3])
    expected_matrix = np.block(
        [
            [state_matrix, border_columns],
            [constraint_rows, border_matrix],
        ]
    )
    expected = np.linalg.solve(expected_matrix, rhs)

    actual = apply_border_aware_preconditioner(
        lambda value: np.linalg.solve(state_matrix, value),
        rhs[:3],
        rhs[3:],
        border_columns,
        constraint_rows,
        border_matrix,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10)


def test_one_border_preconditioner_matches_dense_solve() -> None:
    state_matrix = np.array(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.0, 0.3],
            [0.0, -0.2, 1.5],
        ]
    )
    border_column = np.array([0.4, -0.3, 0.2])
    constraint_row = np.array([0.1, -0.6, 0.5])
    rhs = np.array([0.8, -0.4, 0.2, 0.1])
    expected_matrix = np.block(
        [
            [state_matrix, border_column[:, None]],
            [constraint_row[None, :], np.zeros((1, 1))],
        ]
    )
    expected = np.linalg.solve(expected_matrix, rhs)

    actual = apply_one_border_preconditioner(
        lambda value: np.linalg.solve(state_matrix, value),
        rhs[:3],
        float(rhs[3]),
        border_column,
        constraint_row,
        0.0,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10)


def test_torus_accepts_schur_problem_and_records_anchor_mapping() -> None:
    full = _torus_problem().base_problem
    reduced = build_multitone_schur_problem(full, [0])
    problem = TorusProblem(reduced, (1,), 1, 1.0e9, node_ref=1)

    assert problem.is_schur
    assert problem.full_problem().n == reduced.n
    assert problem.anchor_full_node == 1


def test_torus_fast_preconditioner_keeps_exact_lattice_labels() -> None:
    problem = _torus_problem()
    state = problem.full_problem().zeros()
    state[problem.basis.index_of(ToneIndex(1, 0)), 0] = 1.0e-12
    state[problem.basis.index_of(ToneIndex(0, 1)), 0] = 2.0e-13
    current = problem.full_problem()
    _, preconditioner = problem._linearization(current, state)

    audit = lattice_label_audit(current, preconditioner)

    assert audit["preconditioner_uses_exact_lattice_keys"] is True
    assert audit["jvp_difference_keys_match"] is True
    assert audit["jvp_sum_keys_match"] is True
    assert audit["scalar_frequency_rounding_would_collapse"] is True


def test_torus_linear_fidelity_report_separates_state_and_border() -> None:
    base = _torus_problem().base_problem
    problem = TorusProblem(
        base,
        (1,),
        1,
        1.0e9,
        factor_backend="superlu",
    )
    state = problem.full_problem().zeros()
    state[problem.basis.index_of(ToneIndex(1, 0)), 0] = 1.0e-12
    state[problem.basis.index_of(ToneIndex(0, 1)), 0] = 2.0e-13
    tangent = np.zeros(2 * state.size + 2)
    tangent[-1] = 1.0

    report = problem.linear_fidelity_report(
        state,
        omega_a=problem.omega_a,
        source_tau=1.0,
        previous_X=state,
        previous_omega_a=problem.omega_a,
        previous_source_tau=1.0,
        tangent=tangent,
        phase_reference=state,
        gmres_maxiter=1,
        gmres_restart=2,
    )

    assert "state_preconditioner_fidelity" in report
    assert "augmented_preconditioner_fidelity" in report
    assert "state_only_gmres" in report
    assert "augmented_gmres" in report
    assert "phase_frequency_gmres" in report
    assert "diagonal_augmented_gmres" in report
    assert "phase_equivariance_relative" in report
    assert "phase_frequency_fd_errors" in report
    assert "phase_frequency_preconditioner_fidelity" in report
    assert "direct_bordered_residual" in report
    assert all(
        np.isfinite(value)
        for value in report["phase_frequency_fd_errors"].values()
    )
    assert all(
        np.isfinite(value)
        for value in report["direct_bordered_residual"].values()
    )
    assert report["phase_null_action_relative"] >= 0.0
    assert np.isfinite(report["border_schur_condition"])


def test_loss_convention_audit_is_nonblocking_for_analytic_stability_model() -> None:
    problem = _torus_problem()
    audit = audit_loss_convention(
        problem.base_problem.circuit,
        "current_complex_c",
    )

    assert audit["loss_model"] == "current_complex_c"
    assert audit["analytic_in_omega"] is True
    assert audit["conjugate_symmetric"] is False
    assert "do not publish gain" in str(audit["interpretation"])
