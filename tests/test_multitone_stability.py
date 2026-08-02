from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from twpa_solver.builders.jc_doc import build_jpa
from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import (
    MultiToneBasis,
    build_sideband_matched_basis,
    build_three_tone_basis,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.multitone.stability import (
    _q0_linearization,
    assess_multitone_stability,
)
from twpa_solver.pump import HarmonicNewtonKrylovSolver, NewtonKrylovSettings


def _problem(scale: int = 1) -> FullMultiToneProblem:
    circuit = CircuitMatrices(
        C=sp.eye(1, format="csr") * 1e-12,
        G=sp.eye(1, format="csr") * 1e-3,
        K=sp.eye(1, format="csr") * 1e9,
        Bphi=sp.csr_matrix([[1.0]]),
        Ic=np.array([0.0]),
        port_to_index={1: 0},
    )
    basis = build_three_tone_basis(2.0e10, 1.0e9)
    if scale != 1:
        basis = MultiToneBasis(
            list(basis.tones), basis.omega_p, basis.delta,
            basis.n_p * scale, basis.n_delta * scale,
        )
    source = np.zeros((basis.n_tones, 1), dtype=complex)
    return FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(source)
    )


def _jpa_settings() -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=30, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=40, gmres_maxiter=60, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
    )


def _jpa_problem(signal_ghz: float) -> FullMultiToneProblem:
    """A real Josephson device, so khat actually carries a nonlinearity.

    The Ic=0 fixture above is a linear circuit: every stability answer it
    gives is the same answer it would give with no solver at all.
    """
    builder, _ = build_jpa()
    arrays = builder.assemble()
    circuit = CircuitMatrices(
        C=arrays["C"], G=arrays["G"], K=arrays["K"], Bphi=arrays["Bphi"],
        Ic=arrays["Ic"], port_to_index=arrays["ports"],
    )
    omega_p = 2.0 * math.pi * 4.75001e9
    delta = omega_p - 2.0 * math.pi * signal_ghz * 1e9
    basis = build_sideband_matched_basis(
        [1, 3, 5, 7, 9], 2, omega_p, delta, omega_p * 12.0
    )
    pump = MultiToneDrive(
        basis.pump_tone, circuit.port_to_index[1], 1.13e-8
    ).to_coeffs(basis, circuit.node_count)
    return FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(pump)
    )


def _converged(problem: FullMultiToneProblem) -> np.ndarray:
    state, report = HarmonicNewtonKrylovSolver(_jpa_settings()).solve_one(
        problem, problem.zeros(), 1.0
    )
    assert report.converged, "pump state for the stability test did not converge"
    return state


def test_known_passive_small_signal_state_is_stable() -> None:
    problem = _problem()
    result = assess_multitone_stability(problem, problem.zeros(), refine=False)
    assert result.status == "STABLE_PROXY"
    assert result.sigma_min > 0.0


def test_stability_resolution_metadata_is_explicit() -> None:
    problem = _problem(2)
    result = assess_multitone_stability(problem, problem.zeros(), refine=False)
    assert result.torus_resolution == (problem.basis.n_p, problem.basis.n_delta)
    assert result.matrix_size > 0


def test_sideband_ladder_is_symmetric_and_contiguous() -> None:
    """``ms`` indexes sidebands; khat is keyed by ``ell = m - q``.

    Returning the khat keys as the ladder produced a ragged, asymmetric set
    (it ran -8..18 on the jpa fixture), which is a different axis entirely.
    """
    problem = _jpa_problem(6.6)
    _, ms = _q0_linearization(problem, problem.zeros())

    assert ms == list(range(-max(ms), max(ms) + 1))
    assert ms == sorted(ms)


def test_near_dc_sideband_is_reported_inconclusive_not_stable() -> None:
    """A degenerate operating point must not be scored as stable.

    With the signal 10 kHz off the pump, the m=-1 sideband lands at -10 kHz.
    The conversion matrix is nearly singular there because the circuit has no
    DC path, and the resulting sigma_min is bit-identical with the pump on or
    off -- so a STABLE verdict would be describing the linear circuit.
    """
    problem = _jpa_problem(4.75)
    result = assess_multitone_stability(problem, _converged(problem))

    assert result.status == "INCONCLUSIVE"
    assert "near-DC" in result.reason


def test_verdict_responds_to_the_pump_state() -> None:
    """The whole point of a finite-signal check is finite-signal sensitivity.

    Every previously committed test ran on a linear circuit at a zero state,
    so all of them pass against a function that ignores its state argument
    entirely. This one does not.
    """
    problem = _jpa_problem(6.6)
    state = _converged(problem)

    off = assess_multitone_stability(problem, problem.zeros(), refine=False)
    on = assess_multitone_stability(problem, state * 1.0e6, refine=False)

    assert off.sigma_min is not None and on.sigma_min is not None
    shift = abs(on.sigma_min - off.sigma_min) / abs(off.sigma_min)
    assert shift > 1.0e-6, (
        f"sigma_min moved {shift:.3e} between a zero state and a strongly "
        "driven one -- the diagnostic is not reading the state"
    )


def test_verdict_is_invariant_to_torus_resolution() -> None:
    """The plan's criterion, on a quantity that can actually vary.

    khat comes from an FFT over the torus, so a resolution that is too coarse
    would move it. The previous resolution test only asserted that a metadata
    tuple echoed the basis back.
    """
    coarse = _jpa_problem(6.6)
    fine_basis = MultiToneBasis(
        list(coarse.basis.tones), coarse.basis.omega_p, coarse.basis.delta,
        coarse.basis.n_p * 2, coarse.basis.n_delta * 2,
    )
    fine = FullMultiToneProblem(
        coarse.circuit, fine_basis, coarse.source_path
    )

    coarse_result = assess_multitone_stability(coarse, _converged(coarse))
    fine_result = assess_multitone_stability(fine, _converged(fine))

    assert coarse_result.status == fine_result.status
    np.testing.assert_allclose(
        coarse_result.sigma_min, fine_result.sigma_min, rtol=1e-6
    )
    # The exponent is compared against the pump rate, not against itself at a
    # tight relative tolerance: both resolutions land near 3e-15 s^-1, and two
    # numbers that are each 25 orders below omega_p agree physically however
    # much their last digits differ.
    for result in (coarse_result, fine_result):
        assert abs(result.dominant_exponent_per_s) < 1e-6 * coarse.basis.omega_p
