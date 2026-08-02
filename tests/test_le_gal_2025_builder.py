from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from twpa_solver.builders.le_gal_2025 import build_effective_snail_line
from twpa_solver.core import load_circuit, save_circuit


def test_effective_snail_line_contract() -> None:
    circuit = build_effective_snail_line(cells=20)
    assert circuit.branch_count == 20
    assert circuit.port_to_index == {1: 0, 2: 20}
    assert circuit.node_count == 21
    assert circuit.metadata["snail_capacitance_f"] == 31e-15
    assert circuit.metadata["ground_capacitance_f"] == 223.5e-15
    assert np.isclose(circuit.metadata["linear_inductance_h"], 866.4e-12, rtol=0.01)
    assert np.isclose(
        np.sqrt(circuit.metadata["linear_inductance_h"] / 223.5e-15),
        62.3765,
        rtol=5e-3,
    )


def test_effective_snail_save_load_preserves_branch_law(tmp_path: Path) -> None:
    circuit = build_effective_snail_line(cells=4)
    save_circuit(circuit, tmp_path)
    loaded = load_circuit(tmp_path)
    assert loaded.branch_law is not None
    assert loaded.branch_law.metadata["type"] == "effective_snail"
    np.testing.assert_allclose(loaded.branch_law.ratio, circuit.branch_law.ratio)
    np.testing.assert_allclose(
        loaded.branch_law.equilibrium_flux, circuit.branch_law.equilibrium_flux
    )


def test_alternate_cpr_convention_is_explicit() -> None:
    circuit = build_effective_snail_line(
        cells=2, external_flux_on_small_junction=True
    )
    assert circuit.metadata["external_flux_on_small_junction"] is True
    assert circuit.metadata["equilibrium_flux_rad"][0] == pytest.approx(0.0, abs=1e-12)
