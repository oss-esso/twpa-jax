from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import ToneIndex, build_three_tone_basis
from twpa_solver.multitone import observables
from twpa_solver.multitone.observables import power_balance
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver, NewtonKrylovSettings


def _circuit(loss: float) -> CircuitMatrices:
    return CircuitMatrices(
        C=sp.eye(2, format="csr") * 1e-15,
        G=sp.eye(2, format="csr") * (1.0 / 50.0 + loss),
        K=sp.csr_matrix([[2e9, -1e9], [-1e9, 2e9]]),
        Bphi=sp.csr_matrix([[1.0], [-1.0]]),
        Ic=np.array([1e-6]),
        port_to_index={1: 0, 2: 1},
    )


def _state(loss: float):
    circuit = _circuit(loss)
    basis = build_three_tone_basis(2.0e10, 1.0e9)
    source = MultiToneDrive(
        basis.pump_tone, 0, 1e-8
    ).to_coeffs(basis, circuit.C.shape[0])
    source += MultiToneDrive(
        basis.signal_tone, 0, 1e-10
    ).to_coeffs(basis, circuit.C.shape[0])
    problem = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(source)
    )
    settings = NewtonKrylovSettings(
        newton_tol=1e-10,
        max_newton=20,
        gmres_rtol=1e-8,
        gmres_atol=0.0,
        gmres_restart=20,
        gmres_maxiter=40,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )
    state, reports = HarmonicNewtonKrylovSolver(settings).solve_continuation(
        problem, continuation_steps=4
    )
    assert reports[-1].converged
    assert np.linalg.norm(state) > 0.0
    return state, basis, circuit


def test_power_balance_lossless_nonzero_state_and_manley_rowe() -> None:
    state, basis, circuit = _state(0.0)
    result = power_balance(state, basis, circuit)
    assert result["supplied_power"] == pytest.approx(
        result["dissipated_power"], abs=1e-30
    )
    assert result["power_balance_rel_err"] < 1e-12
    assert result["external_manley_rowe_evaluable"] == 0.0
    assert "pump_net_power_w" in result
    assert "pump_depletion_all_port_db" in result
    assert result["pump_depletion_all_port_db"] is None
    assert set(result["port_power_by_tone_w"]) >= {"h1_q-1", "h1_q0", "h1_q1"}


def test_power_balance_lossy_closes_with_dissipation() -> None:
    state, basis, circuit = _state(1e-3)
    result = power_balance(state, basis, circuit)
    assert result["dissipated_power"] > 0.0
    assert result["power_balance_rel_err"] < 1e-12
    assert np.isfinite(result["external_power_balance_rel_err"])


def test_manley_rowe_ignores_pump_harmonic_conversion(monkeypatch) -> None:
    """Harmonic-only conversion must not make the signal gate evaluable."""
    circuit = _circuit(0.0)
    basis = build_three_tone_basis(2.0e10, 1.0e9)
    basis = type(basis)(
        tones=[*basis.tones, (3, 0)],
        omega_p=basis.omega_p,
        delta=basis.delta,
        n_p=14,
        n_delta=basis.n_delta,
    )
    zero = np.zeros((basis.n_tones, 2), dtype=np.complex128)
    powers = {
        (tone, port): (1e-20 if tone == ToneIndex(3, 0) else 0.0)
        for tone in basis.tones
        for port in circuit.port_to_index
    }

    def fake_waves(*args, **kwargs):
        return {
            "a_power": powers,
            "b_power": {key: 0.0 for key in powers},
        }

    monkeypatch.setattr(observables, "extract_port_waves", fake_waves)
    result = power_balance(zero, basis, circuit)
    assert result["external_manley_rowe_photon_scale"] == 0.0
    assert result["external_manley_rowe_evaluable"] == 0.0
