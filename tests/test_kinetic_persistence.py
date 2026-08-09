from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from twpa_solver.builders.jc_doc import CircuitBuilder
from twpa_solver.builders.kimpa import build_kimpa
from twpa_solver.core import load_circuit, save_circuit
from twpa_solver.core.nonlinear import make_branch_law


def test_kinetic_circuit_round_trips(tmp_path) -> None:
    circuit = build_kimpa("kimpa_fabricated_nominal", cells=(2, 2, 2))
    save_circuit(circuit, tmp_path)
    arrays = np.load(tmp_path / "ipm_arrays.npz")
    assert set(("branch_law_kind", "ki_lk", "ki_istar2", "ki_istar4")) <= set(arrays.files)
    loaded = load_circuit(tmp_path)
    assert loaded.metadata["branch_law"]["Istar2_a"][0] == pytest.approx(0.00325)
    assert loaded.metadata["branch_law"]["Istar4_a"][0] == pytest.approx(0.0017)
    assert loaded.branch_law.metadata["type"] == "kinetic_inductor"
    np.testing.assert_array_equal(loaded.branch_law.kinetic_inductance_h, circuit.branch_law.kinetic_inductance_h)
    flux = np.array([[0.0], [1.0e-15], [-2.0e-15]])
    np.testing.assert_array_equal(
        make_branch_law(loaded).current(flux), make_branch_law(circuit).current(flux)
    )


def test_mixed_jj_ki_circuit_round_trips(tmp_path) -> None:
    builder = CircuitBuilder("mixed_persist")
    builder.port("P1", "a", "0", 1)
    builder.josephson_inductor("jj", "a", "0", 1.0e-9)
    builder.kinetic_inductor("ki", "b", "0", 1.0e-9, 1.0e-3)
    original = builder.assemble()["branch_law"]
    builder.write(tmp_path, {})
    arrays = np.load(tmp_path / "ipm_arrays.npz")
    np.testing.assert_array_equal(arrays["branch_law_kind"], [0, 1])
    loaded = load_circuit(tmp_path)
    assert loaded.branch_law.metadata["type"] == "composite"
    flux = np.array([[0.0, 0.0], [1.0e-15, -2.0e-15]])
    np.testing.assert_allclose(
        make_branch_law(loaded).current(flux), original.current(flux), rtol=0.0, atol=0.0,
    )


def test_legacy_design_without_kind_array_still_loads() -> None:
    loaded = load_circuit("designs/ipm_2c_fixed")
    assert loaded.branch_law is None


def test_quartic_disabled_round_trips_as_none(tmp_path) -> None:
    circuit = build_kimpa("kimpa_fabricated_nominal", cells=(1, 1, 1))
    circuit = replace(circuit, branch_law=replace(circuit.branch_law, istar4_a=None))
    save_circuit(circuit, tmp_path)
    assert load_circuit(tmp_path).branch_law.istar4_a is None


def test_hung_fixture_preserves_explicit_scales(tmp_path) -> None:
    circuit = build_kimpa("kimpa_hung_2025", cells=(1, 1, 1))
    save_circuit(circuit, tmp_path)
    loaded = load_circuit(tmp_path)
    np.testing.assert_array_equal(loaded.branch_law.kinetic_inductance_h, [835e-12])
    np.testing.assert_array_equal(loaded.branch_law.critical_current_a, [1.15e-3])
    np.testing.assert_array_equal(loaded.branch_law.istar2_a, [3.25e-3])
    np.testing.assert_array_equal(loaded.branch_law.istar4_a, [1.70e-3])
