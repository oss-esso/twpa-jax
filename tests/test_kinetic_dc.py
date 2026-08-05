from __future__ import annotations

import numpy as np
import pytest

from twpa_solver.builders.kimpa import build_kimpa
from twpa_solver.core import kinetic_dc_branch_flux
from twpa_solver.pump.problem import FullPumpProblem, HarmonicGrid
from twpa_solver.core.nonlinear import make_branch_law


def test_dc_branch_flux_inverts_to_requested_current() -> None:
    circuit = build_kimpa("kimpa_measured_seed", cells=(1, 1, 1))
    for current in (0.0, 300e-6, 550e-6, 600e-6):
        flux = kinetic_dc_branch_flux(circuit, current)
        recovered = circuit.branch_law.current(flux[None, :])[0]
        assert recovered[-1] == pytest.approx(current, rel=1e-14, abs=1e-18)


def test_dc_flux_is_zero_on_non_kinetic_branches() -> None:
    circuit = build_kimpa("kimpa_measured_seed", cells=(1, 1, 1))
    flux = kinetic_dc_branch_flux(circuit, 550e-6)
    assert np.count_nonzero(flux[:-1]) == 0


def test_residual_removes_static_dc_current() -> None:
    circuit = build_kimpa("kimpa_fabricated_nominal", cells=(1, 1, 1))
    dc = kinetic_dc_branch_flux(circuit, 550e-6)
    problem = FullPumpProblem(
        C=circuit.C,
        G=circuit.G,
        K=circuit.K,
        Bphi=circuit.Bphi,
        branch=make_branch_law(circuit),
        grid=HarmonicGrid(np.array([1]), nt=8, omega=2.0 * np.pi * 16.94e9),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=0.0,
        dc_branch_flux=dc,
    )
    np.testing.assert_allclose(problem.residual_coeffs(problem.zeros(), 0.0), 0.0, atol=1e-30)
