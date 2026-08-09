import pytest

from twpa_solver.design import compile_design
from twpa_solver.design.errors import DesignResolutionError, DesignSchemaError


def _spec(patches=None):
    return {"schema_version": 1, "name": "patchable", "ground": 0,
            "parameters": {}, "cursors": {"signal": 1}, "topology": [
                {"type": "jj_line", "name": "array", "cursor": "signal",
                 "cells": 2, "Lj": 1e-10, "Cj": 1e-13, "Cg": 1e-14}
            ], "patches": patches or []}


def test_set_patch_changes_only_target_value():
    base = compile_design(_spec())
    patched = compile_design(_spec([{"action": "set", "target": "array.cell[0].right",
                                    "value": 2e-10}]))
    assert patched.elements[1].value == 2e-10
    assert len(patched.elements) == len(base.elements)
    assert [e.name for e in patched.elements] == [e.name for e in base.elements]


def test_remove_and_add_path_patches_are_exact():
    removed = compile_design(_spec([{"action": "remove", "target": "array.cell[0].right"}]))
    added = compile_design(_spec([{"action": "add", "name": "extra",
                                  "nodes": ["array.cell[0].left", 0],
                                  "value": 1e-15, "kind": "capacitor"}]))
    assert len(removed.elements) == 6
    assert len(added.elements) == 8


def test_missing_patch_target_is_rejected():
    with pytest.raises(DesignSchemaError, match="found 0"):
        compile_design(_spec([{"action": "set", "target": "missing", "value": 1}]))


def test_missing_node_path_is_rejected():
    design = compile_design(_spec())
    with pytest.raises(DesignResolutionError):
        design.resolve_node("array.cell[99].left")


def test_mutual_raw_element_requires_linear_inductor_endpoints():
    spec = _spec()
    spec["topology"].append({"type": "raw_element", "name": "bad", "nodes": [
        "array.cell[0].right", "array.cell[1].right"], "value": 0.1,
        "kind": "mutual_inductor_k"})
    with pytest.raises(DesignSchemaError, match="linear inductors"):
        compile_design(spec)


def test_cursor_collision_is_a_hard_error():
    spec = _spec()
    spec["cursors"]["pump"] = 1
    spec["topology"].append({"type": "port", "name": "pump_port",
                              "cursor": "pump", "port": 3})
    with pytest.raises(Exception, match="collision"):
        compile_design(spec)
