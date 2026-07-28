from __future__ import annotations

import math

import numpy as np
import pytest

from twpa_solver.builders.jc_doc import build_jpa
from twpa_solver.core import CircuitMatrices
from twpa_solver.core.linear import port_s_from_unit_current_response
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
)
from twpa_solver.signal.floquet import (
    build_signal_schur_partition,
    solve_gain_one,
    solve_gain_one_schur,
)
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.io import PumpSolution
from twpa_solver.pump.basis import PumpBasis


def _jpa_problem() -> tuple[CircuitMatrices, FullPumpProblem]:
    builder, metadata = build_jpa()
    arrays = builder.assemble()
    circuit = CircuitMatrices(
        C=arrays["C"], G=arrays["G"], K=arrays["K"], Bphi=arrays["Bphi"],
        Ic=arrays["Ic"], port_to_index=arrays["ports"],
    )
    omega = 2.0 * math.pi * 4.75001e9
    problem = FullPumpProblem(
        C=circuit.C,
        G=circuit.G,
        K=circuit.K,
        Bphi=circuit.Bphi,
        branch=JosephsonBranchArray(circuit.Ic, circuit.phi0),
        grid=HarmonicGrid(np.array([1, 3, 5]), nt=16, omega=omega),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=metadata["pump_sources"][0]["current_a"],
    )
    return circuit, problem


def _settings(preconditioner: str = "real_coupled") -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=30, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=40, gmres_maxiter=60, min_alpha=1.0 / 1024.0,
        preconditioner=preconditioner, compute_time_residual=False, verbose=False,
        continuation_predictor="none", jvp_mode="aft",
    )


def test_jpa_pump_golden_is_stable() -> None:
    _circuit, problem = _jpa_problem()
    X, reports = HarmonicNewtonKrylovSolver(_settings()).solve_continuation(
        problem, continuation_steps=12
    )
    report = reports[-1]

    assert report.coeff_rel == pytest.approx(1.7399049460865922e-15, rel=1e-12)
    assert report.newton_iterations == 3
    assert float(np.sum(X.real) + np.sum(X.imag)) == pytest.approx(
        1.8893037472379403e-17, rel=1e-12, abs=1e-30
    )


def test_source_normalization_and_port_conversion_are_frozen() -> None:
    _circuit, problem = _jpa_problem()
    source = problem.source_coeffs(1.0)
    assert source[problem.source_row, problem.pump_node_index] == 0.5 * problem.pump_current_a
    assert problem.source_time(1.0)[0, problem.pump_node_index] == problem.pump_current_a
    np.testing.assert_array_equal(problem.source_coeffs(0.0), np.zeros_like(source))
    assert port_s_from_unit_current_response(2.5, source_port=1, out_port=2) == 0.1 + 0j
    assert port_s_from_unit_current_response(2.5, source_port=1, out_port=1) == -0.9 + 0j


def test_jpa_gain_full_and_schur_are_frozen() -> None:
    circuit, problem = _jpa_problem()
    X, _reports = HarmonicNewtonKrylovSolver(_settings()).solve_continuation(
        problem, continuation_steps=12
    )
    omega = problem.grid.omega
    pump = PumpSolution(
        X=X, omega_p=omega, pump_freq_ghz=4.75001, harmonics=3,
        nt_original=16, metadata={}, modes=[1, 3, 5],
        basis=PumpBasis([1, 3, 5], "dense_real", omega),
    )
    khat = build_khat(circuit.Bphi, compute_gamma_hat(circuit, pump, 10, 32), 1e-30)
    kwargs = dict(
        circuit=circuit, khat=khat, khat_off_0=khat[0], omega_p=omega,
        signal_ghz=4.5, sidebands=1, signal_m=0, idler_m=1,
        source_index=circuit.port_to_index[1], out_index=circuit.port_to_index[1],
        source_current_a=1e-9, source_port=1, out_port=1, z0_ohm=50.0,
    )
    full = solve_gain_one(**kwargs)
    reduced = solve_gain_one_schur(**kwargs, schur_part=build_signal_schur_partition(
        circuit, omega, 4.5, 1, circuit.port_to_index[1], circuit.port_to_index[1]
    ))
    for result in (full, reduced):
        assert result.gain_db == pytest.approx(-1.9286549331065743e-15, abs=1e-12)
        assert result.gain_vs_off_db == pytest.approx(0.0, abs=1e-12)
        assert result.linear_rel_residual == pytest.approx(2.147211942242553e-16, rel=1e-12)
    assert full.gain_db == pytest.approx(reduced.gain_db, abs=1e-12)
