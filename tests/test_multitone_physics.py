from __future__ import annotations

import math

import numpy as np

from twpa_solver.builders.jc_doc import build_jpa
from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.observables import tone_s21
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import PumpBasis
from twpa_solver.signal.floquet import solve_gain_one
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.io import PumpSolution


def _jpa() -> tuple[CircuitMatrices, dict[str, object]]:
    builder, metadata = build_jpa()
    arrays = builder.assemble()
    circuit = CircuitMatrices(
        C=arrays["C"],
        G=arrays["G"],
        K=arrays["K"],
        Bphi=arrays["Bphi"],
        Ic=arrays["Ic"],
        port_to_index=arrays["ports"],
    )
    return circuit, metadata


def _settings() -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10,
        max_newton=30,
        gmres_rtol=1e-8,
        gmres_atol=0.0,
        gmres_restart=40,
        gmres_maxiter=60,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )


def _pump(circuit: CircuitMatrices, metadata: dict[str, object]):
    omega_p = 2.0 * math.pi * 4.75001e9
    problem = FullPumpProblem(
        C=circuit.C,
        G=circuit.G,
        K=circuit.K,
        Bphi=circuit.Bphi,
        branch=JosephsonBranchArray(circuit.Ic, circuit.phi0),
        grid=HarmonicGrid(np.array([1]), nt=16, omega=omega_p),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=float(metadata["pump_sources"][0]["current_a"]),
    )
    state, reports = HarmonicNewtonKrylovSolver(_settings()).solve_continuation(
        problem, continuation_steps=4
    )
    assert reports[-1].converged
    solution = PumpSolution(
        X=state,
        omega_p=omega_p,
        pump_freq_ghz=4.75001,
        harmonics=1,
        nt_original=16,
        metadata={},
        modes=[1],
        basis=PumpBasis([1], "dense_real", omega_p),
    )
    khat = build_khat(
        circuit.Bphi,
        compute_gamma_hat(circuit, solution, 3, 32),
        1e-30,
    )
    return problem, state, solution, khat


def _finite_signal(
    circuit: CircuitMatrices,
    metadata: dict[str, object],
    pump_source: np.ndarray,
    signal_current_a: float,
    state: np.ndarray,
):
    omega_p = 2.0 * math.pi * 4.75001e9
    basis = build_three_tone_basis(omega_p, 2.0 * math.pi * 0.25001e9)
    signal = MultiToneDrive(
        basis.signal_tone,
        circuit.port_to_index[1],
        signal_current_a,
    ).to_coeffs(basis, circuit.C.shape[0])
    problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.signal_turn_on(pump_source, signal),
    )
    initial = np.zeros((basis.n_tones, state.shape[1]), dtype=np.complex128)
    initial[basis.index_of(basis.pump_tone)] = state[0]
    result, report = HarmonicNewtonKrylovSolver(_settings()).solve_one(
        problem, initial, 1.0
    )
    assert report.converged
    return basis, result


def test_small_signal_parity_uses_real_reconstruction() -> None:
    circuit, metadata = _jpa()
    pump_problem, pump_state, pump_solution, khat = _pump(circuit, metadata)
    del pump_solution
    basis = build_three_tone_basis(
        2.0 * math.pi * 4.75001e9, 2.0 * math.pi * 0.25001e9
    )
    pump_source = MultiToneDrive(
        basis.pump_tone,
        circuit.port_to_index[1],
        pump_problem.pump_current_a,
    ).to_coeffs(basis, circuit.C.shape[0])
    basis, state = _finite_signal(
        circuit, metadata, pump_source, 1e-12, pump_state
    )
    multitone_db = 20.0 * math.log10(
        abs(
            tone_s21(
                state,
                basis,
                circuit,
                signal_tone=basis.signal_tone,
                source_port=1,
                out_port=1,
                source_current_a=1e-12,
            )
        )
    )
    floquet = solve_gain_one(
        circuit=circuit,
        khat=khat,
        khat_off_0=khat[0],
        omega_p=2.0 * math.pi * 4.75001e9,
        signal_ghz=4.5,
        sidebands=1,
        signal_m=0,
        idler_m=1,
        source_index=circuit.port_to_index[1],
        out_index=circuit.port_to_index[1],
        source_current_a=1e-12,
        source_port=1,
        out_port=1,
        z0_ohm=50.0,
    )
    assert abs(multitone_db - floquet.gain_db) < 0.05
    assert abs(multitone_db - 0.0000471115653308916) < 1e-8


def test_tone_s21_matches_floquet_gain_result() -> None:
    circuit, metadata = _jpa()
    del metadata
    basis = build_three_tone_basis(
        2.0 * math.pi * 4.75001e9, 2.0 * math.pi * 0.25001e9
    )
    source = MultiToneDrive(
        basis.signal_tone, circuit.port_to_index[1], 1e-12
    ).to_coeffs(basis, circuit.C.shape[0])
    problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.signal_turn_on(np.zeros_like(source), source),
    )
    state, report = HarmonicNewtonKrylovSolver(_settings()).solve_one(
        problem, problem.zeros(), 1.0
    )
    assert report.converged
    omega_p = 2.0 * math.pi * 4.75001e9
    zero = np.zeros((1, circuit.C.shape[0]), dtype=np.complex128)
    solution = PumpSolution(
        X=zero,
        omega_p=omega_p,
        pump_freq_ghz=4.75001,
        harmonics=1,
        nt_original=16,
        metadata={},
        modes=[1],
        basis=PumpBasis([1], "dense_real", omega_p),
    )
    khat = build_khat(circuit.Bphi, compute_gamma_hat(circuit, solution, 1, 16), 1e-30)
    result = solve_gain_one(
        circuit=circuit,
        khat=khat,
        khat_off_0=khat[0],
        omega_p=omega_p,
        signal_ghz=4.5,
        sidebands=1,
        signal_m=0,
        idler_m=1,
        source_index=circuit.port_to_index[1],
        out_index=circuit.port_to_index[1],
        source_current_a=1e-12,
        source_port=1,
        out_port=1,
        z0_ohm=50.0,
    )
    measured = 20.0 * math.log10(
        abs(
            tone_s21(
                state,
                basis,
                circuit,
                signal_tone=basis.signal_tone,
                source_port=1,
                out_port=1,
                source_current_a=1e-12,
            )
        )
    )
    assert abs(measured - result.gain_db) < 0.05


def test_pump_only_limit_matches_full_pump_solution() -> None:
    circuit, metadata = _jpa()
    pump_problem, pump_state, _solution, _khat = _pump(circuit, metadata)
    basis = build_three_tone_basis(
        2.0 * math.pi * 4.75001e9, 2.0 * math.pi * 0.25001e9
    )
    source = MultiToneDrive(
        basis.pump_tone,
        circuit.port_to_index[1],
        pump_problem.pump_current_a,
    ).to_coeffs(basis, circuit.C.shape[0])
    problem = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(source)
    )
    state, reports = HarmonicNewtonKrylovSolver(_settings()).solve_continuation(
        problem, continuation_steps=4
    )
    assert reports[-1].converged
    np.testing.assert_allclose(
        state[basis.index_of(basis.pump_tone)], pump_state[0], rtol=0.0, atol=1e-10
    )
    off_q = [i for i, tone in enumerate(basis.tones) if tone.q != 0]
    assert float(np.max(np.abs(state[off_q]))) < 1e-12


def test_pump_off_matches_khat_off_zero_reference() -> None:
    circuit, metadata = _jpa()
    del metadata
    omega_p = 2.0 * math.pi * 4.75001e9
    basis = build_three_tone_basis(omega_p, 2.0 * math.pi * 0.25001e9)
    source = MultiToneDrive(
        basis.signal_tone, circuit.port_to_index[1], 1e-12
    ).to_coeffs(basis, circuit.C.shape[0])
    problem = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.signal_turn_on(np.zeros_like(source), source)
    )
    state, report = HarmonicNewtonKrylovSolver(_settings()).solve_one(
        problem, problem.zeros(), 1.0
    )
    assert report.converged
    zero = np.zeros((1, circuit.C.shape[0]), dtype=np.complex128)
    solution = PumpSolution(
        X=zero,
        omega_p=omega_p,
        pump_freq_ghz=4.75001,
        harmonics=1,
        nt_original=16,
        metadata={},
        modes=[1],
        basis=PumpBasis([1], "dense_real", omega_p),
    )
    khat = build_khat(circuit.Bphi, compute_gamma_hat(circuit, solution, 1, 16), 1e-30)
    reference = solve_gain_one(
        circuit=circuit,
        khat=khat,
        khat_off_0=khat[0],
        omega_p=omega_p,
        signal_ghz=4.5,
        sidebands=1,
        signal_m=0,
        idler_m=1,
        source_index=circuit.port_to_index[1],
        out_index=circuit.port_to_index[1],
        source_current_a=1e-12,
        source_port=1,
        out_port=1,
        z0_ohm=50.0,
    )
    measured = tone_s21(
        state,
        basis,
        circuit,
        signal_tone=basis.signal_tone,
        source_port=1,
        out_port=1,
        source_current_a=1e-12,
    )
    assert abs(20.0 * math.log10(abs(measured)) - reference.gain_db) < 1e-6
