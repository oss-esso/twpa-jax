from __future__ import annotations

import math

import pytest

from twpa_solver.core import load_circuit
from twpa_solver.ports import (
    LEGACY_TW_OFFSET_DB,
    port_available_power_w,
    port_current_from_power_a,
)

DESIGN_DIR = "designs/ipm_2c_fixed"


def test_norton_available_power_is_i_squared_z0_over_8() -> None:
    current_a = 7.3803e-6
    z0_ohm = 50.0
    expected = current_a * current_a * z0_ohm / 8.0
    assert port_available_power_w(
        current_a, z0_ohm, convention="norton"
    ) == pytest.approx(expected)


def test_legacy_available_power_is_i_squared_z0_over_2() -> None:
    current_a = 7.3803e-6
    z0_ohm = 50.0
    expected = current_a * current_a * z0_ohm / 2.0
    assert port_available_power_w(current_a, z0_ohm) == pytest.approx(expected)


def test_legacy_offset_is_exactly_6p02_db() -> None:
    current_a = 3.1e-7
    z0_ohm = 50.0
    norton_w = port_available_power_w(current_a, z0_ohm, convention="norton")
    legacy_w = port_available_power_w(
        current_a, z0_ohm, convention="legacy_traveling_wave"
    )
    offset_db = 10.0 * math.log10(legacy_w / norton_w)
    assert offset_db == pytest.approx(LEGACY_TW_OFFSET_DB, abs=1e-12)
    assert LEGACY_TW_OFFSET_DB == pytest.approx(6.0205999133, abs=1e-9)


@pytest.mark.parametrize("convention", ["norton", "legacy_traveling_wave"])
def test_round_trip_current_power_current(convention: str) -> None:
    current_a = 4.2e-6
    z0_ohm = 50.0
    power_w = port_available_power_w(current_a, z0_ohm, convention=convention)
    recovered = port_current_from_power_a(power_w, z0_ohm, convention=convention)
    assert recovered == pytest.approx(current_a, rel=1e-12)


def test_invalid_convention_rejected() -> None:
    with pytest.raises(ValueError):
        port_available_power_w(1.0, 50.0, convention="bogus")
    with pytest.raises(ValueError):
        port_current_from_power_a(1.0, 50.0, convention="bogus")


def test_nonpositive_z0_rejected() -> None:
    with pytest.raises(ValueError):
        port_available_power_w(1.0, 0.0)
    with pytest.raises(ValueError):
        port_current_from_power_a(1.0, -1.0)


def test_mutation_swapping_8_to_2_fails_norton_check() -> None:
    """Guards the exact denominator: a stray divide-by-8 must fail this."""
    current_a = 1.0e-6
    z0_ohm = 50.0
    correct = port_available_power_w(current_a, z0_ohm)
    wrong_denominator_8 = current_a * current_a * z0_ohm / 8.0
    assert correct != pytest.approx(wrong_denominator_8)
    assert correct == pytest.approx(wrong_denominator_8 * 4.0)


def test_netlist_ports_are_matched_wave_ports_at_1_over_z0() -> None:
    """designs/ipm_2c_fixed's G matrix has exactly one 1/Z0 nonzero per port.

    Each drive port is an ideal current source in parallel with Z0 -- a
    matched wave port, where the injected current is the incident wave's own
    amplitude (Z0 only absorbs reflections correctly), not a Norton
    generator whose short-circuit current splits across a separate matched
    load.
    """
    circuit = load_circuit(DESIGN_DIR)
    z0_ohm = 50.0
    g_coo = circuit.G.tocoo()
    nonzero_diag = {
        int(r): float(v)
        for r, c, v in zip(g_coo.row, g_coo.col, g_coo.data)
        if r == c and abs(v) > 1e-15
    }
    for port_node in circuit.port_to_index.values():
        assert int(port_node) in nonzero_diag
        assert nonzero_diag[int(port_node)] == pytest.approx(1.0 / z0_ohm, rel=1e-9)
    off_diag_nonzero = sum(
        1
        for r, c, v in zip(g_coo.row, g_coo.col, g_coo.data)
        if r != c and abs(v) > 1e-15
    )
    assert off_diag_nonzero == 0
