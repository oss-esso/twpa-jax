"""Phase 3 tests for physical cell builders."""

from __future__ import annotations

from twpa_solver.circuit import Circuit


def test_tl_cell_is_composed_from_capacitor_and_inductor_primitives() -> None:
    circuit = Circuit("tl_cell")
    left = circuit.node("left")
    right = circuit.node("right")

    cell = circuit.add_tl_cell(left, right, L=2e-12, C=3e-15, name="tl.cell[0]")

    assert cell.left is left
    assert cell.right is right
    assert cell.Cg is not None
    assert cell.Cg.kind == "capacitor"
    assert cell.Cg.value == 3e-15
    assert cell.Cg.path == "tl.cell[0].C"
    assert cell.extras["L"].kind == "linear_inductor"
    assert cell.extras["L"].value == 2e-12


def test_jj_cell_exposes_the_expected_handles_and_roles() -> None:
    circuit = Circuit("jj_cell")
    left = circuit.node("left")
    right = circuit.node("right")

    cell = circuit.add_jj_cell(
        left,
        right,
        Lj=123.9e-12,
        Cj=145e-15,
        Cg=66e-15,
        name="line.cell[205]",
        cell_index=205,
    )

    assert cell.left is left
    assert cell.right is right
    assert cell.Lj is not None
    assert cell.Cj is not None
    assert cell.Cg is not None
    assert cell.Lj.value == 123.9e-12
    assert cell.Cj.value == 145e-15
    assert cell.Cg.value == 66e-15
    assert cell.Lj.role == "jj_lj"
    assert cell.Cj.role == "jj_cj"
    assert cell.Cg.role == "jtl_cg"
    assert cell.Lj.path == "line.cell[205].Lj"
    assert cell.Cj.path == "line.cell[205].Cj"
    assert cell.Cg.path == "line.cell[205].Cg"
    assert cell.Lj.cell_index == 205


def test_rf_squid_cell_exposes_internal_branch_elements() -> None:
    circuit = Circuit("rf_squid_cell")
    left = circuit.node("left")
    right = circuit.node("right")

    cell = circuit.add_rf_squid_cell(
        left,
        right,
        Lw=1e-12,
        Lm=2e-12,
        Lpar=3e-12,
        Lj=4e-12,
        Cj=5e-15,
        Cg=6e-15,
        name="rf.cell[0]",
        cell_index=0,
    )

    assert cell.extras["Lw"].role == "rf_squid_lw"
    assert cell.extras["Lm"].role == "rf_squid_lm"
    assert cell.extras["Lpar"].role == "rf_squid_lpar"
    assert cell.extras["Cg_left"].value == 3e-15
    assert cell.Cg is not None
    assert cell.Cg.value == 3e-15
    assert cell.Lj is not None
    assert cell.Cj is not None
