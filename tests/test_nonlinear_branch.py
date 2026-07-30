from __future__ import annotations

import numpy as np
import pytest

from twpa_solver.core.constants import PHI0_REDUCED
from twpa_solver.core.nonlinear import EffectiveSnailBranchLaw, JosephsonBranchLaw
from twpa_solver.builders.le_gal_2025 import build_effective_snail_line


def test_josephson_branch_law_matches_legacy_formula() -> None:
    law = JosephsonBranchLaw(np.array([1.4e-6]), PHI0_REDUCED)
    flux = np.array([[0.0], [0.2 * PHI0_REDUCED]])
    np.testing.assert_allclose(
        law.current(flux), 1.4e-6 * np.sin(flux / PHI0_REDUCED)
    )
    np.testing.assert_allclose(
        law.tangent(flux), 1.4e-6 / PHI0_REDUCED * np.cos(flux / PHI0_REDUCED)
    )


def test_effective_snail_half_flux_equilibrium_and_inductance() -> None:
    phi_ext = np.array([np.pi * PHI0_REDUCED])
    law = EffectiveSnailBranchLaw(
        np.array([1.4e-6]), np.array([0.062]), phi_ext, PHI0_REDUCED
    )
    equilibrium = np.array([[np.pi * PHI0_REDUCED]])
    assert abs(law.current(equilibrium)[0, 0]) < 1e-20
    inductance = 1.0 / law.tangent(equilibrium)[0, 0]
    assert inductance / 866.4e-12 == pytest.approx(1.0, rel=0.01)


def test_shifted_snail_law_is_zero_and_odd_at_equilibrium() -> None:
    circuit = build_effective_snail_line(cells=2)
    law = circuit.branch_law
    zeros = np.zeros((1, 2))
    assert np.max(np.abs(law.current(zeros))) < 1e-16 * 1.4e-6
    flux = np.array([[0.01 * PHI0_REDUCED, -0.02 * PHI0_REDUCED]])
    np.testing.assert_allclose(law.current(-flux), -law.current(flux), atol=1e-18)
    np.testing.assert_allclose(
        1.0 / law.tangent(zeros)[0],
        circuit.metadata["linear_inductance_h"],
        rtol=1e-12,
    )
