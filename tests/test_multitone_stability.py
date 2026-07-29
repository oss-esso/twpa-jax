from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath
from twpa_solver.multitone.stability import assess_multitone_stability


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
        from twpa_solver.multitone.basis import MultiToneBasis
        basis = MultiToneBasis(
            list(basis.tones), basis.omega_p, basis.delta,
            basis.n_p * scale, basis.n_delta * scale,
        )
    source = np.zeros((basis.n_tones, 1), dtype=complex)
    return FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(source)
    )


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
