from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.core.constants import PHI0_REDUCED
from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex
from twpa_solver.multitone.observables import _port_current_coefficients
from twpa_solver.pump.basis import PumpBasis
from twpa_solver.signal.gamma import compute_gamma_hat, synthesize_real_from_positive_harmonics
from twpa_solver.signal.io import PumpSolution
from twpa_solver.core.nonlinear import JosephsonBranchLaw


def _jj_circuit() -> CircuitMatrices:
    return CircuitMatrices(
        C=sp.csr_matrix([[1.0e-12]]),
        G=sp.csr_matrix([[1.0e-6]]),
        K=sp.csr_matrix([[1.0e6]]),
        Bphi=sp.csr_matrix([[1.0]]),
        Ic=np.array([2.0e-6]),
        phi0=PHI0_REDUCED,
        port_to_index={1: 0},
        branch_law=JosephsonBranchLaw(np.array([2.0e-6]), PHI0_REDUCED),
    )


def test_gamma_hat_matches_legacy_josephson_formula() -> None:
    circuit = _jj_circuit()
    omega = 2.0 * math.pi * 5.0e9
    pump = PumpSolution(
        X=np.array([[0.03 * PHI0_REDUCED + 0.01j * PHI0_REDUCED]]),
        omega_p=omega,
        pump_freq_ghz=5.0,
        harmonics=1,
        nt_original=32,
        metadata={},
        modes=[1],
        basis=PumpBasis([1], "dense_real", omega),
    )
    got = compute_gamma_hat(circuit, pump, 2, 32)
    t, x_t = synthesize_real_from_positive_harmonics(pump.X, omega, 32, modes=[1])
    psi = x_t[:, 0]
    expected = {
        ell: np.mean((circuit.Ic[0] / PHI0_REDUCED) * np.cos(psi / PHI0_REDUCED)
                    * np.exp(-1j * ell * omega * t))
        for ell in range(-2, 3)
    }
    for ell in expected:
        np.testing.assert_allclose(got[ell][0], expected[ell], rtol=2e-13, atol=1e-6)


def test_port_currents_are_unchanged_at_zero_dc_flux() -> None:
    circuit = _jj_circuit()
    basis = MultiToneBasis(
        [ToneIndex(1, 0), ToneIndex(1, -1), ToneIndex(1, 1)],
        omega_p=2.0 * math.pi * 10.0e9,
        delta=2.0 * math.pi * 1.0e9,
        n_p=8,
        n_delta=8,
    )
    rng = np.random.default_rng(4)
    X = (rng.normal(size=(basis.n_tones, circuit.node_count))
         + 1j * rng.normal(size=(basis.n_tones, circuit.node_count))) * 1e-18
    zero = _port_current_coefficients(X, basis, circuit)
    explicit_zero = _port_current_coefficients(
        X, basis, circuit, np.zeros(circuit.branch_count)
    )
    np.testing.assert_array_equal(zero, explicit_zero)
