"""Line-scoped YAML topology contracts."""

from __future__ import annotations

import pytest

from twpa_solver.design import compile_design, load_design
from twpa_solver.design.errors import DesignParameterError, DesignSchemaError


def _line_design() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "coupled_lines",
        "parameters": {
            "config": "ipm_default",
            "Lj": "$Lj$ / 2",
            "Cg": "2 * $Cg$",
        },
        "coupler_mode": "ideal",
        "topology": [
            {
                "line": "signal",
                "port_in": 1,
                "port_out": 2,
                "blocks": [
                    {"type": "input_ports", "name": "input_signal"},
                    {"type": "cpw", "name": "signal_cpw", "cells": 2},
                    {
                        "type": "directional_coupler",
                        "name": "coupler_1",
                        "port_in_signal": 1,
                        "port_in_pump": 3,
                        "port_out_signal": 2,
                        "port_out_pump": 4,
                    },
                    {
                        "type": "jtl",
                        "name": "jtl_1",
                        "rows": 2,
                        "jj_number": 4,
                    },
                    {"type": "output_ports", "name": "output_signal"},
                ],
            },
            {
                "line": "pump",
                "port_in": 3,
                "port_out": 4,
                "blocks": [
                    {"type": "input_ports", "name": "input_pump"},
                    {"type": "cpw", "name": "pump_cpw", "cells": 3},
                    {"type": "directional_coupler", "name": "coupler_1"},
                    {"type": "jtl", "name": "jtl_1"},
                    {"type": "output_ports", "name": "output_pump"},
                ],
            },
        ],
    }


def test_line_topology_orders_both_cpws_before_shared_coupler() -> None:
    design = compile_design(_line_design(), strict=True)

    blocks = {block.path: block for block in design.blocks}
    coupler = blocks["coupler_1"]

    assert set(design.ports) == {1, 2, 3, 4}
    assert blocks["signal_cpw"].end_nodes["signal"] == coupler.start_nodes["signal"]
    assert blocks["pump_cpw"].end_nodes["pump"] == coupler.start_nodes["pump"]
    assert [
        block.path for block in design.blocks if block.path.startswith("jtl_1.row[")
    ] == ["jtl_1.row[0]", "jtl_1.row[1]"]


def test_line_topology_applies_base_expressions_and_local_jtl_override() -> None:
    spec = _line_design()
    signal = spec["topology"][0]
    signal["blocks"][3]["Lj"] = "${Lj} / 3"
    design = compile_design(spec)

    assert design.metadata["parameters"]["Lj"] == pytest.approx(61.95e-12)
    assert design.metadata["parameters"]["Cg"] == pytest.approx(132.0e-15)
    first_lj = design.resolve_element("jtl_1.row[0].cell[0].Lj")
    element = next(item for item in design.elements if item.name == first_lj)
    assert element.value == pytest.approx(20.65e-12)


def test_line_topology_rejects_shared_endpoint_without_connection() -> None:
    spec = _line_design()
    spec["topology"][1]["port_out"] = 2

    with pytest.raises(DesignSchemaError, match="shared output_ports name"):
        compile_design(spec)


def test_shared_jtl_joins_two_lines_into_three_port_network() -> None:
    spec = _line_design()
    signal = spec["topology"][0]
    pump = spec["topology"][1]
    signal["blocks"][-1]["name"] = "output"
    pump["port_out"] = 2
    pump["blocks"][-1]["name"] = "output"
    signal["blocks"][2]["port_out_pump"] = 2

    design = compile_design(spec)
    blocks = {block.path: block for block in design.blocks}

    assert set(design.ports) == {1, 2, 3}
    assert design.cursors["signal"] == design.cursors["pump"]
    assert blocks["coupler_1"].end_nodes["signal"] == (
        blocks["coupler_1"].end_nodes["pump"]
    )
    assert blocks["jtl_1.row[0]"].start_nodes["signal"] == (
        blocks["coupler_1"].end_nodes["signal"]
    )


def test_line_topology_rejects_duplicate_detailed_block() -> None:
    spec = _line_design()
    spec["topology"][1]["blocks"][2]["cursors"] = ["signal", "pump"]

    with pytest.raises(DesignSchemaError, match="already defined"):
        compile_design(spec)


def test_parameter_expression_rejects_calls_and_cycles() -> None:
    call_spec = _line_design()
    call_spec["parameters"]["Lj"] = "abs($Lj$)"
    with pytest.raises(DesignParameterError, match="only numeric literals"):
        compile_design(call_spec)

    cycle_spec = _line_design()
    cycle_spec["parameters"].update({"left": "${right}", "right": "${left}"})
    with pytest.raises(DesignParameterError, match="parameter cycle"):
        compile_design(cycle_spec)


def test_documented_coupled_cpw_jtl_design_compiles() -> None:
    design = compile_design(
        load_design("designs/coupled_cpw_jtl_example.yaml"),
        coupler_mode="ideal",
    )

    assert set(design.ports) == {1, 2, 3, 4}
    assert design.metadata["parameters"]["Lj"] == pytest.approx(61.95e-12)
    assert design.metadata["parameters"]["Cg"] == pytest.approx(132.0e-15)
    assert sum(
        block.count
        for block in design.blocks
        if block.path.startswith("jtl_1.row[")
    ) == 3 * 435
    assert sum(
        block.count
        for block in design.blocks
        if block.path.startswith("jtl_2.row[")
    ) == 2 * 100


def test_documented_three_port_design_compiles() -> None:
    design = compile_design(
        load_design("designs/three_port_cpw_jtl_example.yaml"),
        coupler_mode="ideal",
    )

    assert set(design.ports) == {1, 2, 3}
    assert design.cursors["signal"] == design.cursors["pump"]


def test_line_scoped_2c_matches_compact_2c_element_for_element() -> None:
    line_scoped = compile_design(load_design("designs/ipm_2c_line_scoped.yaml"))
    compact = compile_design(load_design("designs/ipm_2c.yaml"))

    assert [element.__dict__ for element in line_scoped.elements] == [
        element.__dict__ for element in compact.elements
    ]
    assert line_scoped.cursors == compact.cursors
    assert {
        number: port.node for number, port in line_scoped.ports.items()
    } == {
        number: port.node for number, port in compact.ports.items()
    }
