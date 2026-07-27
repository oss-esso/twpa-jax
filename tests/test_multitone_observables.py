from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

from twpa_solver.core import CircuitMatrices, port_waves
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.observables import extract_port_waves, junction_diagnostics
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver, NewtonKrylovSettings


def _circuit() -> CircuitMatrices:
    return CircuitMatrices(
        C=sp.eye(2, format="csr"),
        G=sp.eye(2, format="csr"),
        K=sp.eye(2, format="csr"),
        Bphi=sp.csr_matrix([[1.0], [-1.0]]),
        Ic=np.array([1.0]),
        port_to_index={1: 0, 2: 1},
    )


def test_port_waves_and_observable_shapes() -> None:
    a, b = port_waves(3.0 + 2.0j, 0.01 - 0.02j)
    assert a == (3.0 + 2.0j + 50.0 * (0.01 - 0.02j)) / (2.0 * np.sqrt(50.0))
    assert b == (3.0 + 2.0j - 50.0 * (0.01 - 0.02j)) / (2.0 * np.sqrt(50.0))

    basis = build_three_tone_basis(10.0, 1.0)
    state = np.ones((basis.n_tones, 2), dtype=np.complex128) * 1e-3
    waves = extract_port_waves(state, basis, _circuit(), [1, 2])
    assert len(waves["a"]) == basis.n_tones * 2
    assert len(junction_diagnostics(state, basis, _circuit())) == 1


def test_passive_port_waves_use_network_current_and_power_balance() -> None:
    circuit = _circuit()
    circuit = CircuitMatrices(
        C=circuit.C,
        G=circuit.G,
        K=circuit.K,
        Bphi=circuit.Bphi,
        Ic=np.array([0.0]),
        port_to_index=circuit.port_to_index,
    )
    basis = build_three_tone_basis(10.0, 1.0)
    source = MultiToneDrive(basis.signal_tone, 0, 1e-9).to_coeffs(
        basis, circuit.C.shape[0]
    )
    problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.signal_turn_on(np.zeros_like(source), source),
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
    state, report = HarmonicNewtonKrylovSolver(settings).solve_one(
        problem, problem.zeros(), 1.0
    )
    assert report.converged
    waves = extract_port_waves(state, basis, circuit, [1], z0_ohm=50.0)
    for row, tone in enumerate(basis.tones):
        voltage = 2.0 * 1j * basis.omegas[row] * state[row, 0]
        source_current = 1e-9 if tone == basis.signal_tone else 0.0
        network_current = source_current - voltage / 50.0
        key = (tone, 1)
        assert waves["b_power"][key] <= waves["a_power"][key] + 1e-20
        delivered = float(np.real(voltage * np.conj(network_current)))
        assert waves["a_power"][key] - waves["b_power"][key] == pytest.approx(
            delivered, abs=1e-25
        )
