"""Line-scoped YAML topology contracts."""

from __future__ import annotations

import copy

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


def test_line_topology_allows_local_coupler_parameter_overrides() -> None:
    local = _line_design()
    local["topology"][0]["blocks"][2].update({
        "coupling_dB": -16.0,
        "coupler_freq_hz": 10.0e9,
        "Z0": 55.0,
        "cell_length_um": 12.0,
    })
    global_values = _line_design()
    global_values["parameters"].update({
        "coupling_dB": -16.0,
        "coupler_freq_hz": 10.0e9,
        "Z0": 55.0,
        "cell_length_um": 12.0,
    })

    local_design = compile_design(local, coupler_mode="ideal")
    global_design = compile_design(global_values, coupler_mode="ideal")

    assert [element.__dict__ for element in local_design.elements] == [
        element.__dict__ for element in global_design.elements
    ]


def test_line_topology_allows_commenting_detailed_shared_coupler() -> None:
    spec = _line_design()
    # Leave the reference on the pump line to model commenting only the
    # detailed signal-side declaration.
    spec["topology"][0]["blocks"].pop(2)

    compiled = compile_design(spec, coupler_mode="ideal")

    assert all(block.type != "directional_coupler" for block in compiled.blocks)


def test_line_topology_allows_commenting_both_shared_coupler_occurrences() -> None:
    spec = _line_design()
    spec["topology"][0]["blocks"].pop(2)
    spec["topology"][1]["blocks"].pop(2)

    compiled = compile_design(spec, coupler_mode="ideal")

    assert all(block.type != "directional_coupler" for block in compiled.blocks)


def test_parameter_expression_rejects_calls_and_cycles() -> None:
    call_spec = _line_design()
    call_spec["parameters"]["Lj"] = "abs($Lj$)"
    with pytest.raises(DesignParameterError, match="only numeric literals"):
        compile_design(call_spec)

    cycle_spec = _line_design()
    cycle_spec["parameters"].update({"left": "${right}", "right": "${left}"})
    with pytest.raises(DesignParameterError, match="parameter cycle"):
        compile_design(cycle_spec)


def test_documented_three_port_design_compiles() -> None:
    design = compile_design(
        load_design("designs/three_port_cpw_jtl_example.yaml"),
        coupler_mode="ideal",
    )

    assert set(design.ports) == {1, 2, 3}
    assert design.cursors["signal"] == design.cursors["pump"]


def test_line_scoped_2c_matches_compact_2c_element_for_element() -> None:
    line_scoped = compile_design(load_design("designs/ipm_2c.yaml"))
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


def test_repeat_can_contain_ipm_composites() -> None:
    spec = {
        "schema_version": 1,
        "name": "repeated_ipm",
        "cursors": {"signal": 1, "pump": 10000},
        "ground": 0,
        "parameters": {"config": "ipm_default"},
        "coupler_mode": "ideal",
        "topology": [
            {"type": "port", "name": "input_signal", "cursor": "signal", "port": 1},
            {"type": "port", "name": "input_pump", "cursor": "pump", "port": 3},
            {"type": "directional_coupler", "name": "coupler_in"},
            {"repeat": {
                "name": "period",
                "count": 2,
                "topology": [
                    {"type": "ipm_line", "name": "line", "rows": 1, "cells": 4},
                    {"type": "directional_coupler", "name": "coupler"},
                ],
            }},
            {"type": "ipm_tail", "name": "tail", "rows": 1, "cells": 4},
            {"type": "port", "name": "output_signal", "cursor": "signal", "port": 2},
            {"type": "port", "name": "output_pump", "cursor": "pump", "port": 4},
        ],
    }

    design = compile_design(spec)

    assert sum(block.type == "directional_coupler" for block in design.blocks) == 3
    assert any(block.path == "period[0].line.row[0].array[0]" for block in design.blocks)
    assert any(block.path == "period[1].line.row[0].array[0]" for block in design.blocks)


def test_ipm_defaults_preserve_explicit_two_coupler_netlist() -> None:
    explicit = load_design("designs/ipm_2c.yaml")
    omitted = copy.deepcopy(explicit)
    for line in omitted["topology"]:
        for block in line["blocks"]:
            if block.get("type") == "ipm_line":
                for key in (
                    "rows", "cells", "between", "trailing_signal_cpw_cells",
                    "trailing_pump_cpw_cells",
                ):
                    block.pop(key, None)
            elif block.get("type") == "ipm_tail":
                for key in ("rows", "cells", "between", "final_array"):
                    block.pop(key, None)

    expected = compile_design(explicit, coupler_mode="ideal")
    actual = compile_design(omitted, coupler_mode="ideal")

    assert [element.__dict__ for element in actual.elements] == [
        element.__dict__ for element in expected.elements
    ]


def test_global_and_local_loss_scatter_settings_are_editable() -> None:
    global_spec = _line_design()
    global_spec["parameters"].update({
        "tan_delta": 1.0e-3,
        "lj_scatter_sigma": 0.01,
        "scatter_seed": 17,
    })
    global_design = compile_design(global_spec)

    assert global_design.component_plan is not None
    assert global_design.component_plan.metadata["scatter_specs"]["Lj"]["sigma"] == pytest.approx(0.01)
    assert global_design.metadata["loss"]["default"] == pytest.approx(1.0e-3)

    local_spec = _line_design()
    local_spec["topology"][0]["blocks"][3].update({
        "tan_delta": 2.0e-3,
        "lj_scatter_sigma": 0.01,
        "scatter_seed": 19,
    })
    local_design = compile_design(local_spec)

    assert any(
        element.role.startswith("jtl_cg@jtl_1.row[0]")
        for element in local_design.elements
        if element.kind == "capacitor"
    )
    assert any(
        key.startswith("jtl_cg@jtl_1.row[0]")
        and value == pytest.approx(2.0e-3)
        for key, value in local_design.metadata["loss"]["by_role"].items()
    )
