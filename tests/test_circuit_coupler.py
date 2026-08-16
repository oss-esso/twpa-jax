"""Phase 5 tests for directional couplers and explicit CPW geometry."""

from __future__ import annotations

import re
from typing import Any

import pytest

from twpa_solver.builders.cpw_coupler import CPWConformalCoupler, CPWModeResult
from twpa_solver.builders.ipm import (
    Element,
    add_edge_coupled_directional_coupler,
    calculate_conformal_discrete_params,
)
from twpa_solver.circuit import Circuit, ExplicitCouplerGeometry


FAB_GEOMETRY = ExplicitCouplerGeometry(
    gaps_um=[5.5, 5.0, 5.0, 5.5],
    widths_um=[9.186, 15.0, 9.186],
    length_um=100.0,
)


def _fab_discrete(length_um: float) -> Any:
    conformal = CPWConformalCoupler(
        list(FAB_GEOMETRY.gaps_um),
        list(FAB_GEOMETRY.widths_um),
        length_um,
    )
    parameters = conformal.parameters()
    result = CPWModeResult(
        gaps_um=tuple(FAB_GEOMETRY.gaps_um),
        widths_um=tuple(FAB_GEOMETRY.widths_um),
        length_um=length_um,
        coupling_db=float(parameters["coupling_db"]),
        z_eff_ohm=float(parameters["Z_eff"]),
        model="three_line",
    )
    return calculate_conformal_discrete_params(result, cell_length_um=10.0)


def _translate(value: int | str, mapping: dict[int, int]) -> int | str:
    if isinstance(value, int):
        return mapping.get(value, value)

    def replace(match: re.Match[str]) -> str:
        number = int(match.group())
        return str(mapping.get(number, number))

    return re.sub(r"\d+", replace, value)


def _canonical_element(element: Element, mapping: dict[int, int]) -> dict[str, object]:
    data = element.__dict__.copy()
    data["name"] = _translate(str(data["name"]), mapping)
    data["n1"] = _translate(data["n1"], mapping)
    data["n2"] = _translate(data["n2"], mapping)
    return data


def test_coupler_matches_legacy_element_emission_without_calling_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discrete = _fab_discrete(length_um=30.0)
    legacy: list[Element] = []
    add_edge_coupled_directional_coupler(legacy, 1, 100, 0, discrete)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("new coupler called the legacy emission function")

    monkeypatch.setattr(
        "twpa_solver.builders.ipm.add_edge_coupled_directional_coupler",
        fail_if_called,
    )
    circuit = Circuit("coupler_parity")
    signal = circuit.path("signal")
    pump = circuit.path("pump")
    handle = circuit.add_directional_coupler(
        signal,
        pump,
        geometry=ExplicitCouplerGeometry(
            FAB_GEOMETRY.gaps_um,
            FAB_GEOMETRY.widths_um,
            30.0,
        ),
    )
    signal_nodes = [handle.signal_in, *[cell.signal.right for cell in handle.cells]]
    pump_nodes = [handle.pump_in, *[cell.pump.right for cell in handle.cells]]
    mapping = {0: 0}
    mapping.update({1 + index: node.uid for index, node in enumerate(signal_nodes)})
    mapping.update({100 + index: node.uid for index, node in enumerate(pump_nodes)})

    actual = [_canonical_element(element, {}) for element in circuit.compile().elements]
    expected = [_canonical_element(element, mapping) for element in legacy]
    assert actual == expected


def test_explicit_fab_geometry_bypasses_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("explicit geometry called the optimizer")

    monkeypatch.setattr(
        "twpa_solver.circuit.blocks.coupler.make_coupler_discrete",
        fail_if_called,
    )
    circuit = Circuit("explicit_geometry")
    signal = circuit.path("signal")
    pump = circuit.path("pump")
    handle = circuit.add_directional_coupler(signal, pump, geometry=FAB_GEOMETRY)

    assert handle.metadata["model"] == "three_line"
    assert handle.metadata["coupling_db"] < 0.0


def test_explicit_geometry_dimension_errors_are_raised() -> None:
    with pytest.raises(ValueError, match="one more"):
        ExplicitCouplerGeometry([5.0, 5.0], [9.0, 9.0, 9.0], 100.0)
    with pytest.raises(ValueError, match="two or three"):
        ExplicitCouplerGeometry([5.0, 5.0, 5.0, 5.0], [9.0], 100.0)


def test_coupler_advances_both_paths_and_exposes_terminals() -> None:
    circuit = Circuit("coupler_handles")
    signal = circuit.path("signal")
    pump = circuit.path("pump")
    signal_start = signal.start
    pump_start = pump.start
    handle = circuit.add_directional_coupler(
        signal,
        pump,
        geometry=ExplicitCouplerGeometry(
            FAB_GEOMETRY.gaps_um,
            FAB_GEOMETRY.widths_um,
            30.0,
        ),
    )

    assert len(handle.cells) == 3
    assert signal.start is signal_start
    assert pump.start is pump_start
    assert signal.end is handle.signal_out
    assert pump.end is handle.pump_out
    assert handle.port(1) is handle.signal_in
    assert handle.port(2) is handle.signal_out
    assert handle.port(3) is handle.pump_in
    assert handle.port(4) is handle.pump_out
    assert handle.cell(2).signal.right is signal.end
    assert handle.cell(2).pump.right is pump.end


def test_same_path_is_rejected() -> None:
    circuit = Circuit("same_path")
    signal = circuit.path("signal")
    with pytest.raises(ValueError, match="two distinct paths"):
        circuit.add_directional_coupler(signal, signal, geometry=FAB_GEOMETRY)


def test_foreign_path_is_rejected() -> None:
    first = Circuit("first")
    second = Circuit("second")
    signal = first.path("signal")
    pump = second.path("pump")
    with pytest.raises(ValueError, match="another Circuit"):
        first.add_directional_coupler(signal, pump, geometry=FAB_GEOMETRY)


def test_coupler_emission_is_deterministic() -> None:
    def build() -> list[dict[str, object]]:
        circuit = Circuit("deterministic_coupler")
        signal = circuit.path("signal")
        pump = circuit.path("pump")
        circuit.add_directional_coupler(
            signal,
            pump,
            geometry=ExplicitCouplerGeometry(
                FAB_GEOMETRY.gaps_um,
                FAB_GEOMETRY.widths_um,
                30.0,
            ),
        )
        return [element.__dict__.copy() for element in circuit.compile().elements]

    assert build() == build()
