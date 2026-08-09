from __future__ import annotations

from types import SimpleNamespace

import pytest

from twpa_solver.port_roles import resolve_mixing_order, resolve_port_roles


@pytest.mark.parametrize(
    ("ports", "expected"),
    [
        ((1,), {"pump_port": 1, "source_port": 1, "out_port": 1}),
        ((1, 2), {"pump_port": 1, "source_port": 1, "out_port": 2}),
        ((1, 2, 3, 4), {"pump_port": 4, "source_port": 1, "out_port": 2}),
    ],
)
def test_port_roles_follow_available_port_count(ports, expected):
    circuit = SimpleNamespace(port_to_index={port: port for port in ports})
    assert resolve_port_roles(circuit) == expected


def test_explicit_port_roles_are_preserved():
    assert resolve_port_roles({2: 0, 4: 1}, pump_port=2, source_port=4, out_port=2) == {
        "pump_port": 2, "source_port": 4, "out_port": 2,
    }


def test_mixing_order_auto_uses_external_bias():
    assert resolve_mixing_order("auto", dc_current_a=0.0) == 4
    assert resolve_mixing_order("auto", dc_current_a=1e-6) == 3
    assert resolve_mixing_order("auto", dc_branch_flux_over_phi0=0.33) == 3
    assert resolve_mixing_order("auto", design_meta={"features": {"dc_bias": True}}) == 3
    assert resolve_mixing_order("auto", dc_current_a=1e-6, dc_branch_flux_over_phi0=None) == 3


def test_mixing_order_explicit_override():
    assert resolve_mixing_order("4", dc_current_a=1e-6) == 4
    assert resolve_mixing_order(3, dc_current_a=0.0) == 3
