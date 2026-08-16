"""Phase 1 tests for the symbolic circuit graph and primitive builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twpa_solver.builders.ipm import Element, build_matrices
from twpa_solver.circuit import Circuit, Node
from twpa_solver.circuit.netlist_export import export_netlist


def _make_basic_circuit(port_numbers: tuple[int, int] = (1, 3)) -> Circuit:
    circuit = Circuit("basic")
    left = circuit.node("left")
    right = circuit.node("right")
    circuit.add_resistor(left, circuit.ground, 50.0)
    circuit.add_capacitor(left, right, 2.0e-15)
    circuit.add_inductor(right, circuit.ground, 1.0e-12)
    circuit.add_port(left, number=port_numbers[0])
    circuit.add_port(right, number=port_numbers[1])
    return circuit


def test_primitives_compile_with_expected_handles_and_matrix_stamp() -> None:
    circuit = Circuit("primitives")
    left = circuit.node("left")
    right = circuit.node("right")
    resistor = circuit.add_resistor(left, circuit.ground, 50.0, name="R_input")
    capacitor = circuit.add_capacitor(left, right, 2.0e-15, name="C_link")
    inductor = circuit.add_inductor(right, circuit.ground, 1.0e-12, name="L_line")
    second_inductor = circuit.add_inductor(
        left, right, 2e-12, name="L_second"
    )
    mutual = circuit.add_mutual_inductor(inductor, second_inductor, 0.2)
    jj = circuit.add_jj(left, right, 1e-12, 3e-15, name="JJ_main")

    assert resistor.n1 is left
    assert resistor.n2 is circuit.ground
    assert capacitor.value == 2.0e-15
    assert capacitor.kind == "capacitor"
    assert capacitor.role == "capacitor"
    assert inductor.kind == "linear_inductor"
    assert mutual.kind == "mutual_inductor_k"
    assert jj.kind == "josephson_inductor"
    assert jj.role == "jj_lj"
    assert jj.companion is not None
    assert jj.companion.role == "jj_cj"

    compiled = circuit.compile()
    expected = [
        Element("R_input", 1, 0, 50.0, "resistor", "resistor"),
        Element("C_link", 1, 2, 2e-15, "capacitor", "capacitor"),
        Element("L_line", 2, 0, 1e-12, "linear_inductor", "tl_l"),
        Element("L_second", 1, 2, 2e-12, "linear_inductor", "tl_l"),
        Element("K1", "L_line", "L_second", 0.2, "mutual_inductor_k", "mutual_k"),
        Element("JJ_main.Lj", 1, 2, 1e-12, "josephson_inductor", "jj_lj"),
        Element("JJ_main.Cj", 1, 2, 3e-15, "capacitor", "jj_cj"),
    ]
    assert [element.__dict__ for element in compiled.elements] == [
        element.__dict__ for element in expected
    ]
    actual_matrices = compiled.matrices()
    expected_matrices = build_matrices(expected)
    for name in ("C", "G", "K", "Bphi"):
        assert (actual_matrices[name] != expected_matrices[name]).nnz == 0
    assert np.array_equal(actual_matrices["Ic"], expected_matrices["Ic"])


def test_arbitrary_graph_and_ground_compile_without_paths() -> None:
    circuit = Circuit("branched")
    a = circuit.node("a")
    b = circuit.node("b")
    branch = circuit.node("branch")
    circuit.add_capacitor(a, branch, 1e-15)
    circuit.add_inductor(branch, b, 1e-12)
    circuit.add_resistor(branch, circuit.ground, 50.0)

    compiled = circuit.compile()

    assert compiled.node_map[circuit.ground] == 0
    assert compiled.node_map[a] == 1
    assert compiled.node_map[b] == 2
    assert compiled.node_map[branch] == 3
    assert all(
        0 <= int(endpoint) <= 3
        for element in compiled.elements
        for endpoint in (element.n1, element.n2)
        if isinstance(endpoint, int)
    )


def test_compilation_is_deterministic() -> None:
    first = _make_basic_circuit((1, 3)).compile()
    second = _make_basic_circuit((7, 42)).compile()

    assert [element.__dict__ for element in first.elements[:3]] == [
        element.__dict__ for element in second.elements[:3]
    ]
    assert [node.uid for node in first.node_map] == [
        node.uid for node in second.node_map
    ]


def test_legacy_numbering_compiles_with_compatibility_nodes() -> None:
    compiled = _make_basic_circuit().compile(node_numbering="legacy")

    assert compiled.node_map[compiled.reverse_node_map[1]] == 1
    assert sorted(compiled.node_map.values()) == [0, 1, 2]
    assert compiled.metadata["node_numbering"] == "legacy"


def test_graph_membership_uses_node_identity_not_field_equality() -> None:
    circuit = Circuit("identity")
    node = circuit.node("node")
    twin = Node(node.uid, node.owner_id, node.name, node.path)
    circuit.add_capacitor(twin, circuit.ground, 1.0e-15)

    with pytest.raises(ValueError, match="node is not registered"):
        circuit.compile()


def test_legacy_numbering_preserves_explicit_element_names() -> None:
    circuit = Circuit("explicit_name")
    signal = circuit.path("signal")
    circuit.set_legacy_path_bases({"signal": 100})
    resistor = circuit.add_resistor(
        signal.start,
        circuit.ground,
        50.0,
        name="R_input",
    )

    compiled = circuit.compile(node_numbering="legacy")

    assert resistor.auto_name is False
    assert compiled.elements[0].name == "R_input"


def test_port_numbers_do_not_change_node_allocation() -> None:
    first = _make_basic_circuit((1, 3)).compile()
    second = _make_basic_circuit((7, 42)).compile()

    first_nodes = [node.uid for node in first.reverse_node_map.values()]
    second_nodes = [node.uid for node in second.reverse_node_map.values()]
    assert first_nodes == second_nodes
    assert [port.node for port in first.ports.values()] == [1, 2]
    assert [port.node for port in second.ports.values()] == [1, 2]


def test_duplicate_names_ports_and_cross_circuit_nodes_raise() -> None:
    first = Circuit("first")
    node = first.node("node")
    with pytest.raises(ValueError, match="duplicate symbolic node name"):
        first.node("node")
    first.add_capacitor(node, first.ground, 1e-15, name="cap")
    with pytest.raises(ValueError, match="duplicate explicit element name"):
        first.add_resistor(node, first.ground, 50.0, name="cap")
    first.add_port(node, number=1)
    with pytest.raises(ValueError, match="duplicate port number"):
        first.add_port(node, number=1)

    second = Circuit("second")
    with pytest.raises(ValueError, match="another Circuit"):
        second.add_capacitor(node, second.ground, 1e-15)


def test_remove_and_dangling_mutual_reference_validation() -> None:
    circuit = Circuit("validation")
    a = circuit.node("a")
    b = circuit.node("b")
    first = circuit.add_inductor(a, b, 1e-12, name="L_first")
    second = circuit.add_inductor(b, circuit.ground, 2e-12, name="L_second")
    mutual = circuit.add_mutual_inductor(first, second, 0.1, name="K_pair")
    circuit.remove(first)
    with pytest.raises(ValueError, match="removed element"):
        circuit.compile()
    circuit.remove(mutual)
    with pytest.raises(ValueError, match="already been removed"):
        circuit.remove(mutual)


def test_netlist_export_round_trips_element_count_and_endpoints(tmp_path: Path) -> None:
    circuit = Circuit("netlist")
    a = circuit.node("a")
    b = circuit.node("b")
    circuit.add_capacitor(a, b, 1e-15, name="C_test")
    compiled = circuit.compile()

    text = export_netlist(compiled)
    target = tmp_path / "netlist.cir"
    assert circuit.export_netlist(str(target)) == text
    lines = [line.split() for line in text.splitlines()]
    assert len(lines) == len(compiled.elements)
    assert lines[0][:3] == ["C_test", "1", "2"]
    assert target.read_text(encoding="utf-8") == text


def test_invalid_numbering_and_values_report_the_relevant_path() -> None:
    circuit = Circuit("invalid")
    node = circuit.node("node")
    with pytest.raises(ValueError, match="resistor: R must be finite and positive"):
        circuit.add_resistor(node, circuit.ground, 0.0)
    with pytest.raises(ValueError, match=r"port\[0\]"):
        circuit.add_port(node, number=0)
    with pytest.raises(ValueError, match="node_numbering"):
        circuit.compile(node_numbering="unknown")  # type: ignore[arg-type]
