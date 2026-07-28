from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from twpa_solver.builders.jc_doc import build_jpa
from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import (
    MultiToneBasis,
    ToneIndex,
    build_sideband_matched_basis,
)
from twpa_solver.multitone.observables import tone_s21
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.seed import promote_pump_solution
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import PumpBasis
from twpa_solver.signal.floquet import GainResult, solve_gain_one
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.io import PumpSolution


OMEGA_P = 2.0 * math.pi * 4.75001e9
PUMP_MODES = [1, 3, 5]
SIGNAL_CURRENT_A = 1e-12


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


def _pump(
    circuit: CircuitMatrices, metadata: dict[str, object]
) -> tuple[FullPumpProblem, np.ndarray, PumpSolution, dict[int, sp.csr_matrix]]:
    problem = FullPumpProblem(
        C=circuit.C,
        G=circuit.G,
        K=circuit.K,
        Bphi=circuit.Bphi,
        branch=JosephsonBranchArray(circuit.Ic, circuit.phi0),
        grid=HarmonicGrid(np.array(PUMP_MODES), nt=16, omega=OMEGA_P),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=float(metadata["pump_sources"][0]["current_a"]),
    )
    state, reports = HarmonicNewtonKrylovSolver(_settings()).solve_continuation(
        problem, continuation_steps=12
    )
    assert reports[-1].converged
    solution = PumpSolution(
        X=state,
        omega_p=OMEGA_P,
        pump_freq_ghz=4.75001,
        harmonics=5,
        nt_original=16,
        metadata={},
        modes=PUMP_MODES,
        basis=PumpBasis(PUMP_MODES, "dense_real", OMEGA_P),
    )
    khat = build_khat(
        circuit.Bphi,
        compute_gamma_hat(circuit, solution, 10, 32),
        1e-30,
    )
    return problem, state, solution, khat


def _pump_off_khat(circuit: CircuitMatrices) -> sp.csr_matrix:
    return (
        circuit.Bphi
        @ sp.diags(circuit.Ic / circuit.phi0)
        @ circuit.Bphi.T
    ).tocsr()


def _basis(signal_ghz: float, sidebands: int = 2) -> MultiToneBasis:
    delta = OMEGA_P - 2.0 * math.pi * signal_ghz * 1e9
    return build_sideband_matched_basis(
        PUMP_MODES, sidebands, OMEGA_P, delta, OMEGA_P * 6.0
    )


def _pump_source(
    problem: FullPumpProblem, basis: MultiToneBasis
) -> np.ndarray:
    source = np.zeros((basis.n_tones, problem.n), dtype=np.complex128)
    pump = problem.source_coeffs(1.0)
    for row, mode in enumerate(PUMP_MODES):
        source[basis.index_of(ToneIndex(mode, 0))] = pump[row]
    return source


def _solve_signal(
    circuit: CircuitMatrices,
    basis: MultiToneBasis,
    pump_source: np.ndarray,
    signal_current_a: float,
    initial: np.ndarray,
) -> np.ndarray:
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
    state, report = HarmonicNewtonKrylovSolver(_settings()).solve_one(
        problem, initial, 1.0
    )
    assert report.converged
    return state


def _floquet(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    signal_ghz: float,
    khat_off_0: sp.csr_matrix,
) -> GainResult:
    return solve_gain_one(
        circuit=circuit,
        khat=khat,
        khat_off_0=khat_off_0,
        omega_p=OMEGA_P,
        signal_ghz=signal_ghz,
        sidebands=2,
        signal_m=0,
        idler_m=-2,
        source_index=circuit.port_to_index[1],
        out_index=circuit.port_to_index[1],
        source_current_a=SIGNAL_CURRENT_A,
        source_port=1,
        out_port=1,
        z0_ohm=50.0,
    )


def _multitone_gain_vs_off(
    circuit: CircuitMatrices,
    basis: MultiToneBasis,
    pump_state: np.ndarray,
    pump_source: np.ndarray,
    signal_ghz: float,
) -> float:
    pump_initial = promote_pump_solution(
        pump_state,
        PumpBasis(PUMP_MODES, "dense_real", OMEGA_P),
        basis,
    )
    on = _solve_signal(
        circuit, basis, pump_source, SIGNAL_CURRENT_A, pump_initial
    )
    off = _solve_signal(
        circuit,
        basis,
        np.zeros_like(pump_source),
        SIGNAL_CURRENT_A,
        np.zeros_like(pump_initial),
    )
    row = basis.index_of(basis.signal_tone)
    node = circuit.port_to_index[1]
    return float(10.0 * math.log10(abs(on[row, node] / off[row, node]) ** 2))


def test_small_signal_parity_at_real_gain_point() -> None:
    circuit, metadata = _jpa()
    problem, pump_state, _solution, khat = _pump(circuit, metadata)
    basis = _basis(4.8)
    pump_source = _pump_source(problem, basis)
    measured = _multitone_gain_vs_off(
        circuit, basis, pump_state, pump_source, 4.8
    )
    reference = _floquet(circuit, khat, 4.8, _pump_off_khat(circuit))
    np.testing.assert_allclose(
        reference.gain_vs_off_db, 15.591267314309897, atol=0.05
    )
    assert abs(measured - reference.gain_vs_off_db) < 0.05


def test_small_signal_parity_matches_full_sideband_reference() -> None:
    """The JPA gate also uses the complete ``m=-10..10`` reference set."""
    circuit, metadata = _jpa()
    problem, pump_state, _solution, khat = _pump(circuit, metadata)
    delta = OMEGA_P - 2.0 * math.pi * 4.8e9
    basis = build_sideband_matched_basis(
        PUMP_MODES, 10, OMEGA_P, delta, OMEGA_P * 12.0
    )
    assert basis.covered_sidebands() == set(range(-10, 11))
    pump_source = _pump_source(problem, basis)
    measured = _multitone_gain_vs_off(
        circuit, basis, pump_state, pump_source, 4.8
    )
    reference = solve_gain_one(
        circuit=circuit,
        khat=khat,
        khat_off_0=_pump_off_khat(circuit),
        omega_p=OMEGA_P,
        signal_ghz=4.8,
        sidebands=10,
        signal_m=0,
        idler_m=-2,
        source_index=circuit.port_to_index[1],
        out_index=circuit.port_to_index[1],
        source_current_a=SIGNAL_CURRENT_A,
        source_port=1,
        out_port=1,
        z0_ohm=50.0,
    )
    assert abs(measured - reference.gain_vs_off_db) < 0.5


def test_weak_point_is_explicitly_no_gain_limit() -> None:
    circuit, metadata = _jpa()
    problem, pump_state, _solution, khat = _pump(circuit, metadata)
    basis = _basis(4.5)
    pump_source = _pump_source(problem, basis)
    measured = _multitone_gain_vs_off(
        circuit, basis, pump_state, pump_source, 4.5
    )
    reference = _floquet(circuit, khat, 4.5, _pump_off_khat(circuit))
    assert abs(reference.gain_vs_off_db) < 0.01
    assert abs(measured - reference.gain_vs_off_db) < 0.05


def test_tone_s21_matches_floquet_gain_result_at_gain_point() -> None:
    circuit, metadata = _jpa()
    problem, pump_state, _solution, khat = _pump(circuit, metadata)
    basis = _basis(4.8)
    pump_source = _pump_source(problem, basis)
    pump_initial = promote_pump_solution(
        pump_state,
        PumpBasis(PUMP_MODES, "dense_real", OMEGA_P),
        basis,
    )
    state = _solve_signal(
        circuit, basis, pump_source, SIGNAL_CURRENT_A, pump_initial
    )
    reference = _floquet(circuit, khat, 4.8, _pump_off_khat(circuit))
    measured = tone_s21(
        state,
        basis,
        circuit,
        signal_tone=basis.signal_tone,
        source_port=1,
        out_port=1,
        source_current_a=SIGNAL_CURRENT_A,
    )
    measured_db = 20.0 * math.log10(abs(measured))
    assert abs(measured_db - reference.gain_db) < 1e-6


def test_pump_only_limit_matches_full_pump_solution() -> None:
    circuit, metadata = _jpa()
    problem, pump_state, _solution, _khat = _pump(circuit, metadata)
    basis = _basis(4.8)
    source = _pump_source(problem, basis)
    multitone = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(source)
    )
    state, reports = HarmonicNewtonKrylovSolver(_settings()).solve_continuation(
        multitone, continuation_steps=12
    )
    assert reports[-1].converged
    np.testing.assert_allclose(
        state[basis.index_of(ToneIndex(1, 0))], pump_state[0], rtol=0.0, atol=1e-10
    )
    for mode_row, mode in enumerate(PUMP_MODES[1:], start=1):
        np.testing.assert_allclose(
            state[basis.index_of(ToneIndex(mode, 0))],
            pump_state[mode_row],
            atol=1e-10,
        )
    off_q = [i for i, tone in enumerate(basis.tones) if tone.q != 0]
    assert float(np.max(np.abs(state[off_q]))) < 1e-12


def test_pump_off_matches_khat_off_zero_reference() -> None:
    circuit, metadata = _jpa()
    problem, _pump_state, _solution, _khat = _pump(circuit, metadata)
    basis = _basis(4.5)
    source = MultiToneDrive(
        basis.signal_tone, circuit.port_to_index[1], SIGNAL_CURRENT_A
    ).to_coeffs(basis, circuit.C.shape[0])
    off_problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.signal_turn_on(np.zeros_like(source), source),
    )
    state, report = HarmonicNewtonKrylovSolver(_settings()).solve_one(
        off_problem, off_problem.zeros(), 1.0
    )
    assert report.converged
    zero = np.zeros((len(PUMP_MODES), circuit.C.shape[0]), dtype=np.complex128)
    zero_solution = PumpSolution(
        X=zero,
        omega_p=OMEGA_P,
        pump_freq_ghz=4.75001,
        harmonics=5,
        nt_original=16,
        metadata={},
        modes=PUMP_MODES,
        basis=PumpBasis(PUMP_MODES, "dense_real", OMEGA_P),
    )
    khat = build_khat(
        circuit.Bphi,
        compute_gamma_hat(circuit, zero_solution, 10, 32),
        1e-30,
    )
    reference = _floquet(circuit, khat, 4.5, _pump_off_khat(circuit))
    measured = tone_s21(
        state,
        basis,
        circuit,
        signal_tone=basis.signal_tone,
        source_port=1,
        out_port=1,
        source_current_a=SIGNAL_CURRENT_A,
    )
    assert abs(20.0 * math.log10(abs(measured)) - reference.gain_db) < 1e-6
