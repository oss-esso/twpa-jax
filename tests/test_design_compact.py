"""Compact design composition and design-owned profile contracts."""

from __future__ import annotations

import copy

import numpy as np

from twpa_solver.builders.profiles import parse_profile_shorthand
from twpa_solver.design import compile_design, load_design


def test_mixed_granularity_exposes_internal_nodes_and_accepts_local_capacitor():
    spec = {
        "schema_version": 1,
        "name": "mixed",
        "technology": "qanova_ipm_v1",
        "parameters": {"Lj": 123.9e-12, "Cj": 145e-15, "Cg": 66e-15},
        "topology": [
            {"type": "port", "name": "in", "cursor": "signal", "port": 1},
            {"type": "ipm_line", "name": "line", "cursor": "signal", "arrays": 2},
            {"type": "capacitor", "name": "local", "nodes": [
                "line.array[1].cell[3].right", "ground"
            ], "C": 98e-15},
            {"action": "set", "target": "line.array[0].cell[1].Lj",
             "value": 140e-12},
        ],
    }
    compiled = compile_design(spec)
    node = compiled.resolve_node("line.array[1].cell[3].right")
    local = next(element for element in compiled.elements if element.name == "local")
    assert local.n1 == node
    assert compiled.resolve_element("line.array[0].cell[1].Lj")
    assert any(element.value == 140e-12 for element in compiled.elements)


def test_design_owned_linear_profile_matches_profile_engine():
    design = compile_design(load_design("designs/ipm_2c_linear.yaml"))
    expected = np.array([100e-12 + (150e-12 - 100e-12) * t
                         for t in np.linspace(0.0, 1.0, 418)])
    first_row = np.array([
        next(element.value for element in design.elements
             if element.name == design.resolve_element(
                 f"period[0].row[0].array.cell[{cell}].Lj"))
        for cell in range(418)
    ])
    assert np.allclose(first_row, expected)


def test_named_half_sine_matches_legacy_expression():
    design = compile_design(load_design("designs/ipm_2c_half_sine.yaml"))
    expected = (parse_profile_shorthand(
        "all:custom:100p->150p:domain=per_row,expression=sin(pi*t/2)"
    ))
    from twpa_solver.builders.profiles import evaluate_profile
    values = evaluate_profile([expected], n_cells=418, cells_per_row=418,
                              base_value=123.9e-12)
    actual = np.array([
        next(element.value for element in design.elements
             if element.name == design.resolve_element(
                 f"period[0].row[0].array.cell[{cell}].Lj"))
        for cell in range(418)
    ])
    assert np.allclose(actual, values)


def test_technology_defaults_match_explicit_design_values():
    explicit = load_design("designs/ipm_2c.yaml")
    preset = copy.deepcopy(explicit)
    preset["technology"] = "ipm_default"
    left = compile_design(explicit)
    right = compile_design(preset)
    assert [element.__dict__ for element in left.elements] == [
        element.__dict__ for element in right.elements
    ]
