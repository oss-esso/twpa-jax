import copy

import pytest
import scipy.sparse as sp

from twpa_solver.builders.ipm import (
    IPMParams,
    make_coupler_discrete,
    make_ipm,
    parse_args,
)
from twpa_solver.builders.ipm import build_matrices
from twpa_solver.design import compile_design, load_design
from twpa_solver.design.errors import DesignSchemaError


def test_legacy_builder_defaults_to_auto_and_rejects_cached_mode() -> None:
    assert parse_args([]).coupler_mode == "auto"
    with pytest.raises(SystemExit):
        parse_args(["--coupler-mode", "cached"])


def test_ipm_yaml_is_element_for_element_legacy_parity():
    params = IPMParams(start_node_bot=10_000)
    legacy, ends = make_ipm(params, make_coupler_discrete(params, "auto"))
    compiled = compile_design(load_design("designs/ipm_2c_line_scoped.yaml"))
    assert [element.__dict__ for element in compiled.elements] == [
        element.__dict__ for element in legacy
    ]
    assert compiled.cursors == {"signal": ends["top_end_node"], "pump": ends["bottom_end_node"]}


def test_ipm_yaml_matrices_match_legacy_and_stored_structure():
    params = IPMParams(start_node_bot=10_000)
    coupler = make_coupler_discrete(params, "auto")
    legacy, _ = make_ipm(params, coupler)
    compiled = compile_design(load_design("designs/ipm_2c_line_scoped.yaml"))
    left, right = build_matrices(legacy), build_matrices(compiled.elements)
    for name in ("C", "G", "K", "Bphi"):
        assert (left[name] != right[name]).nnz == 0
        stored = sp.load_npz(f"tests/data/ipm_2c_reference/{name}.npz")
        assert (right[name] != stored).nnz == 0
    assert (left["Ic"] == right["Ic"]).all()


def test_nested_repeat_registers_stable_paths():
    compiled = compile_design(load_design("designs/ipm_2c_line_scoped.yaml"))
    assert compiled.resolve_node("line_1.row[1].array[0].cell[10].left") == 888


def test_repeat_depth_three_is_rejected():
    spec = {
        "schema_version": 1,
        "name": "too_deep",
        "ground": 0,
        "parameters": {},
        "cursors": {"signal": 1},
        "topology": [{"repeat": {"count": 1, "name": "a", "topology": [
            {"repeat": {"count": 1, "name": "b", "topology": [
                {"repeat": {"count": 1, "name": "c", "topology": [
                    {"type": "port", "name": "p", "cursor": "signal", "port": 1}
                ]}}
            ]}}
        ]}}],
    }
    with pytest.raises(DesignSchemaError, match="deeper than 2"):
        compile_design(spec)


def test_compile_is_deterministic_and_repeat_mutation_changes_topology():
    spec = load_design("designs/ipm_2c_line_scoped.yaml")
    first = compile_design(spec)
    second = compile_design(copy.deepcopy(spec))
    assert [element.__dict__ for element in first.elements] == [
        element.__dict__ for element in second.elements
    ]
    mutated = copy.deepcopy(spec)
    mutated["topology"][0]["blocks"][2]["rows"] = 4
    changed = compile_design(mutated)
    assert len(changed.elements) != len(first.elements)
