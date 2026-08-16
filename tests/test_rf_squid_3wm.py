from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from twpa_solver.builders.ipm import build_matrices
from twpa_solver.design import compile_design
from scripts.h1_transient_branch_transfer import checkpoint_dc_flux


def _design(cells: int = 25) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "rf_squid_test",
        "ground": 0,
        "cursors": {"signal": 1, "pump": 1000},
        "parameters": {},
        "coupler_mode": "cached",
        "topology": [
            {"type": "port", "name": "input", "cursor": "signal", "port": 1},
            {"type": "rf_squid_line", "name": "line", "cursor": "signal",
             "cells": cells, "Ic": 0.93e-6, "Lm": 58.6e-12,
             "Lw": 37.0e-12, "Lpar": 8.9e-12, "Cj": 20e-15,
             "Cg_pattern": [10.5e-15, 68.2e-15, 10.5e-15, 50.4e-15],
             "Cg_pattern_counts": [6, 6, 6, 6]},
            {"type": "port", "name": "output", "cursor": "signal", "port": 2},
        ],
    }


def test_rf_squid_line_expands_to_one_junction_per_cell():
    design = compile_design(_design())
    junctions = [element for element in design.elements
                 if element.kind == "josephson_inductor"]
    assert len(junctions) == 25
    assert design.named_nodes["line.cell[0].left"] == 1
    assert design.named_nodes["line.cell[24].right"] == 76


def test_rf_squid_loading_repeats_and_truncates_deterministically():
    design = compile_design(_design(25))
    capacitors = {
        element.cell_index: float(element.value)
        for element in design.elements
        if element.role == "rf_squid_cg" and element.name.endswith("Cg_right")
    }
    assert capacitors[0] == 5.25e-15
    assert capacitors[6] == 34.1e-15
    assert capacitors[12] == 5.25e-15
    assert capacitors[18] == 25.2e-15
    assert capacitors[24] == 5.25e-15


def test_rf_squid_period_stamps_full_ground_capacitance():
    design = compile_design(_design(24))
    total = sum(
        float(element.value)
        for element in design.elements
        if element.role == "rf_squid_cg"
    )
    assert total == 837.6e-15


def test_biased_josephson_tangent_contains_a_first_pump_harmonic():
    design = compile_design(_design(1))
    matrices = build_matrices(design.elements)
    phi0 = 2.067833848e-15 / (2.0 * np.pi)
    dc = 0.33 * 2.0 * np.pi * phi0
    phase = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    tangent = np.cos((dc + 0.05 * phi0 * np.cos(phase)) / phi0)
    first = np.mean(tangent * np.exp(-1j * phase))
    assert matrices["Bphi"].shape[1] == 1
    assert abs(first) > 1e-4


def test_td_checkpoint_uses_production_dc_flux_over_nominal_fallback():
    """The TD handoff must preserve the exact HB bias convention."""
    circuit = SimpleNamespace(branch_count=3)
    report = {"metadata": {"dc_branch_flux": [1.0, 2.0, 3.0]}}
    resolved = checkpoint_dc_flux(report, circuit, np.full(3, 9.0))
    np.testing.assert_array_equal(resolved, [1.0, 2.0, 3.0])


def test_td_checkpoint_scalar_dc_flux_broadcasts_to_all_branches():
    circuit = SimpleNamespace(branch_count=2)
    report = {"metadata": {"dc_branch_flux_wb": 4.5}}
    np.testing.assert_array_equal(
        checkpoint_dc_flux(report, circuit, np.zeros(2)), [4.5, 4.5]
    )

