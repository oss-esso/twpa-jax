"""Phase 8 structural gates for the v2 and v3 fabrication architectures."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from designs.python.ipm_v2 import IPMv2Config, _coupling_db, build_ipm_v2
from designs.python.ipm_v3 import build_ipm_v3
from twpa_solver.builders.ipm import build_matrices
from twpa_solver.circuit import Circuit, coupler_leakage_db
from twpa_solver.core import CircuitMatrices, solve_linear_scattering


def _compiled_matrices(circuit: Circuit) -> dict[str, Any]:
    """Compile a symbolic circuit and assemble its unchanged solver matrices."""

    return build_matrices(circuit.compile().elements)


def test_v2_lumped_arrays_add_no_nodes_and_preserve_v1_shape() -> None:
    """The array count changes values, not symbolic topology or node count."""

    default = build_ipm_v2()
    single = build_ipm_v2(IPMv2Config(junction_array_count=1))
    default_elements = default.compile().elements
    junctions = [
        element for element in default_elements
        if element.kind == "josephson_inductor"
    ]

    assert len(default.graph.nodes) == len(single.graph.nodes)
    assert len(junctions) == 20 * 15 * 3 * 2
    assert all(element.value == pytest.approx(3.0 * IPMv2Config().Lj)
               for element in junctions)


def test_v2_array_capacitance_is_one_third() -> None:
    """Every effective v2 junction capacitor uses the series-array scaling."""

    circuit = build_ipm_v2()
    refs = [
        element for element in circuit.graph.elements
        if element.kind == "josephson_inductor"
    ]

    assert len(refs) == 1800
    assert all(
        ref.companion is not None
        and ref.companion.value == pytest.approx(IPMv2Config().Cj / 3.0)
        for ref in refs
    )


def test_coupler_leakage_is_explicit_and_optional() -> None:
    """The Prometheus correction is available but is not silently inherited."""

    nominal = -14.0
    expected = 10.0 * np.log10(
        10.0 ** (nominal / 10.0) /
        (1.0 - 2.0 * 10.0 ** (nominal / 10.0))
    )
    assert coupler_leakage_db(nominal, 2) == pytest.approx(expected)

    nominal_config = IPMv2Config()
    leakage_config = IPMv2Config(apply_coupler_leakage=True)
    assert _coupling_db(nominal_config, 1) == -14.0
    assert _coupling_db(nominal_config, 2) == -14.0
    assert _coupling_db(leakage_config, 1) == pytest.approx(
        coupler_leakage_db(-14.0, 1)
    )
    assert _coupling_db(leakage_config, 2) == pytest.approx(
        coupler_leakage_db(-14.0, 2)
    )


def test_v3_has_six_rows_three_explicit_three_line_couplers() -> None:
    """The v3 gate checks topology and explicit model selection, not fab parity."""

    circuit = build_ipm_v3()
    compiled = circuit.compile()
    junctions = [
        element for element in compiled.elements
        if element.kind == "josephson_inductor"
    ]
    couplers = [
        data for path, data in circuit.graph.hierarchy.items()
        if path.startswith("coupler[")
    ]

    assert len(junctions) == 36 * 18 * 6
    assert len(couplers) == 3
    assert all(data["model"] == "three_line" for data in couplers)
    assert all(np.isfinite(data["coupling_db"]) for data in couplers)
    assert len({
        key.split(".")[0] for key in circuit.graph.named_nodes
        if key.startswith("row[")
    }) == 6


def test_v3_uses_fab_geometry_optimizer_then_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optimized fab cross-section must reach the explicit path."""

    def fail_optimizer(*args: object, **kwargs: object) -> None:
        raise AssertionError("v3 unexpectedly used the generic discrete route")

    monkeypatch.setattr(
        "twpa_solver.circuit.blocks.coupler.make_coupler_discrete",
        fail_optimizer,
    )
    circuit = build_ipm_v3()
    couplers = [
        data for path, data in circuit.graph.hierarchy.items()
        if path.startswith("coupler[")
    ]

    assert [data["coupling_db"] for data in couplers] == pytest.approx(
        [-25.0] * 3, abs=1e-2
    )


@pytest.mark.parametrize("builder", [build_ipm_v2, build_ipm_v3])
def test_variant_compile_and_matrix_shapes(builder: Any) -> None:
    """New variants lower to valid Element[] and self-consistent matrices."""

    circuit = builder()
    netlist = circuit.export_netlist()
    matrices = _compiled_matrices(circuit)

    assert netlist
    assert matrices["C"].shape == matrices["G"].shape
    assert matrices["C"].shape == matrices["K"].shape
    assert matrices["Bphi"].shape[0] == matrices["C"].shape[0]
    assert matrices["Bphi"].shape[1] == matrices["Ic"].size


def test_v3_linear_scattering_is_finite() -> None:
    """The v3 passive reduction supports the existing linear scattering solve."""

    matrices = _compiled_matrices(build_ipm_v3())
    circuit = CircuitMatrices(
        C=matrices["C"],
        G=matrices["G"],
        K=matrices["K"],
        Bphi=matrices["Bphi"],
        Ic=matrices["Ic"],
        Lj=matrices["Lj"],
        nodes=matrices["nodes"],
        port_to_index=matrices["port_vectors"],
    )
    result = solve_linear_scattering(
        circuit,
        frequency_hz=10.0e9,
        source_port=1,
        out_port=2,
    )

    assert np.isfinite(result.s_abs)
    assert np.isfinite(result.s_db)


@pytest.mark.parametrize("builder", [build_ipm_v2, build_ipm_v3])
def test_variant_emission_is_deterministic(builder: Any) -> None:
    """Repeated construction preserves names, ordering, and node numbering."""

    first = builder().compile()
    second = builder().compile()

    assert [element.__dict__ for element in first.elements] == [
        element.__dict__ for element in second.elements
    ]
    first_nodes = sorted(
        (node.uid, node.name, node.path, number)
        for node, number in first.node_map.items()
    )
    second_nodes = sorted(
        (node.uid, node.name, node.path, number)
        for node, number in second.node_map.items()
    )
    assert first_nodes == second_nodes


def test_variant_api_has_no_squared_kinetic_inductance_parameter() -> None:
    """Variant authoring accepts physical Lk values, never a squared alias."""

    assert "Lk_sq" not in inspect.signature(build_ipm_v2).parameters
    assert "Lk_sq" not in inspect.signature(build_ipm_v3).parameters
    assert "Lk_sq" not in inspect.signature(coupler_leakage_db).parameters
