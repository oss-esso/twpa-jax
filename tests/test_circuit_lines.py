"""Phase 3 tests for repeated lines and targeted edits."""

from __future__ import annotations

import pytest

from twpa_solver.builders.blocks import BuildContext, build_rf_squid_line
from twpa_solver.builders.ipm import Element, add_jtl_element, add_tl_element
from twpa_solver.circuit import Circuit


def test_transmission_line_matches_legacy_cell_emission() -> None:
    circuit = Circuit("tl_line")
    path = circuit.path("signal")
    circuit.add_transmission_line(path, cells=3, L=2e-12, C=3e-15)
    compiled = circuit.compile()

    expected: list[Element] = []
    for node in range(1, 4):
        add_tl_element(expected, node, 0, 2e-12, 3e-15)

    assert [element.__dict__ for element in compiled.elements] == [
        element.__dict__ for element in expected
    ]


def test_jj_line_matches_legacy_cell_emission_and_boundary_capacitance() -> None:
    circuit = Circuit("jj_line")
    path = circuit.path("signal")
    circuit.add_jj_line(path, cells=3, Lj=123.9e-12, Cj=145e-15, Cg=66e-15)
    compiled = circuit.compile()

    expected: list[Element] = []
    for index, node in enumerate(range(1, 4)):
        add_jtl_element(
            expected,
            node,
            0,
            33e-15 if index == 0 else 66e-15,
            123.9e-12,
            145e-15,
            cell_index=index,
        )
    expected.append(
        Element(
            "C4_0_JTL_end",
            4,
            0,
            33e-15,
            "capacitor",
            "jtl_cg",
            2,
        )
    )

    assert [element.__dict__ for element in compiled.elements] == [
        element.__dict__ for element in expected
    ]


def test_jj_line_handles_expose_cell_205_and_path_end() -> None:
    circuit = Circuit("handles")
    path = circuit.path("signal")
    line = circuit.add_jj_line(
        path,
        cells=206,
        Lj=123.9e-12,
        Cj=145e-15,
        Cg=66e-15,
    )
    cell = line.cell(205)

    assert line.input is path.start
    assert line.output is path.end
    assert line.node(206) is path.end
    assert cell.left is line.node(205)
    assert cell.right is line.node(206)
    assert cell.Lj is not None
    assert cell.Cj is not None
    assert cell.Cg is not None
    assert cell.Lj.cell_index == 205
    assert cell.Cj.cell_index == 205
    assert cell.Cg.cell_index == 205


def test_targeted_addition_changes_only_the_requested_local_element() -> None:
    circuit = Circuit("targeted")
    path = circuit.path("signal")
    line = circuit.add_jj_line(
        path,
        cells=3,
        Lj=123.9e-12,
        Cj=145e-15,
        Cg=66e-15,
    )
    before = [element.__dict__.copy() for element in circuit.compile().elements]
    circuit.add_capacitor(line.cell(1).right, circuit.ground, 98e-15)
    after = [element.__dict__.copy() for element in circuit.compile().elements]

    assert len(after) == len(before) + 1
    assert after[:-1] == before
    assert after[-1]["n1"] == line.cell(1).right.uid
    assert after[-1]["n2"] == 0


def test_set_value_and_remove_change_only_the_requested_cell_element() -> None:
    circuit = Circuit("edits")
    path = circuit.path("signal")
    line = circuit.add_jj_line(
        path,
        cells=3,
        Lj=123.9e-12,
        Cj=145e-15,
        Cg=66e-15,
    )
    target = line.cell(1).Cg
    removed = line.cell(1).Cj
    assert target is not None
    assert removed is not None
    before = {
        element.name: element.__dict__.copy()
        for element in circuit.compile().elements
    }

    circuit.set_value(target, 98e-15)
    circuit.remove(removed)
    after = {
        element.name: element.__dict__.copy()
        for element in circuit.compile().elements
    }

    assert len(after) == len(before) - 1
    assert removed.name not in after
    assert after[target.name]["value"] == 98e-15
    unchanged = set(before) - {target.name, removed.name}
    assert all(after[name] == before[name] for name in unchanged)


def test_line_cell_counts_and_indices_are_validated() -> None:
    circuit = Circuit("invalid_lines")
    path = circuit.path("signal")
    with pytest.raises(ValueError, match="positive integer"):
        circuit.add_jj_line(path, cells=0, Lj=1e-12, Cj=1e-15, Cg=1e-15)
    with pytest.raises(ValueError, match="non-negative integer"):
        circuit.add_transmission_line(path, cells=-1, L=1e-12, C=1e-15)


def test_rf_squid_line_builds_with_patterned_ground_loading() -> None:
    circuit = Circuit("rf_line")
    path = circuit.path("signal")
    line = circuit.add_rf_squid_line(
        path,
        cells=4,
        Ic=1e-6,
        Lm=2e-12,
        Lw=3e-12,
        Lpar=4e-12,
        Cj=5e-15,
        Cg_pattern=[6e-15, 8e-15],
        Cg_pattern_counts=[1, 1],
    )
    compiled = circuit.compile()

    assert len(line.cells) == 4
    assert line.output is path.end
    assert len(compiled.elements) == 4 * 7
    assert compiled.matrices()["Bphi"].shape[1] == 4


def test_rf_squid_line_matches_legacy_block_emission() -> None:
    config = {
        "cursor": "signal",
        "cells": 4,
        "Ic": 1e-6,
        "Lm": 2e-12,
        "Lw": 3e-12,
        "Lpar": 4e-12,
        "Cj": 5e-15,
        "Cg_pattern": [6e-15, 8e-15],
        "Cg_pattern_counts": [1, 1],
    }
    legacy_context = BuildContext([], {"signal": 1}, 0, 0)
    build_rf_squid_line(legacy_context, config, "rf")

    circuit = Circuit("rf_line_parity")
    path = circuit.path("signal")
    circuit.add_rf_squid_line(
        path,
        cells=4,
        Ic=1e-6,
        Lm=2e-12,
        Lw=3e-12,
        Lpar=4e-12,
        Cj=5e-15,
        Cg_pattern=[6e-15, 8e-15],
        Cg_pattern_counts=[1, 1],
        name="rf",
    )

    assert [element.__dict__ for element in circuit.compile().elements] == [
        element.__dict__ for element in legacy_context.circuit
    ]


def test_line_emission_is_deterministic() -> None:
    def build() -> list[dict[str, object]]:
        circuit = Circuit("deterministic_line")
        path = circuit.path("signal")
        circuit.add_jj_line(
            path,
            cells=3,
            Lj=123.9e-12,
            Cj=145e-15,
            Cg=66e-15,
        )
        return [element.__dict__.copy() for element in circuit.compile().elements]

    assert build() == build()
