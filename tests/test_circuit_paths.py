"""Phase 2 tests for symbolic circuit paths."""

from __future__ import annotations

import pytest

from twpa_solver.circuit import Circuit


def test_path_extension_advances_end_and_preserves_start() -> None:
    circuit = Circuit("paths")
    signal = circuit.path("signal")
    start = signal.start
    first = circuit.node("signal.node[1]")
    second = circuit.node("signal.node[2]")

    signal.extend(first)
    signal.extend(second)

    assert signal.start is start
    assert signal.end is second
    assert signal.nodes == [start, first, second]
    assert len(signal) == 3


def test_path_node_205_is_the_205th_appended_node() -> None:
    circuit = Circuit("long_path")
    signal = circuit.path("signal")
    appended = [circuit.node(f"signal.node[{index}]") for index in range(1, 206)]

    for node in appended:
        signal.extend(node)

    assert signal.node(205) is appended[204]
    assert signal.end is appended[-1]
    assert len(signal.nodes) == 206


def test_multiple_paths_do_not_change_existing_path_semantics() -> None:
    circuit = Circuit("multiple_paths")
    signal = circuit.path("signal")
    signal_start = signal.start
    pump = circuit.path("pump")
    bias = circuit.path("bias")
    signal_node = circuit.node("signal.next")
    signal.extend(signal_node)

    assert signal.start is signal_start
    assert signal.end is signal_node
    assert pump.start.uid != signal.start.uid
    assert bias.start.uid != pump.start.uid
    assert signal.nodes == [signal_start, signal_node]


def test_duplicate_or_empty_path_names_raise() -> None:
    circuit = Circuit("duplicate_paths")
    circuit.path("signal")

    with pytest.raises(ValueError, match="duplicate symbolic path name"):
        circuit.path("signal")
    with pytest.raises(ValueError, match="path name must not be empty"):
        circuit.path("")


def test_six_paths_and_twelve_ports_compile() -> None:
    circuit = Circuit("many_paths")
    paths = [circuit.path(f"path_{index}") for index in range(6)]
    for index, path in enumerate(paths):
        circuit.add_port(path.start, number=2 * index + 1)
        circuit.add_port(path.end, number=2 * index + 2)

    compiled = circuit.compile()

    assert len(compiled.ports) == 12
    assert compiled.metadata["node_count"] == 6
    assert [port.node for port in compiled.ports.values()] == [
        node for path in paths for node in (path.start.uid, path.end.uid)
    ]


def test_path_cannot_extend_with_a_node_from_another_circuit() -> None:
    first = Circuit("first")
    second = Circuit("second")
    path = first.path("signal")

    with pytest.raises(ValueError, match="another Circuit"):
        path.extend(second.node("foreign"))


def test_path_node_bounds_and_types_report_the_path() -> None:
    path = Circuit("bounds").path("signal")

    with pytest.raises(IndexError, match=r"signal\.node\[1\]"):
        path.node(1)
    with pytest.raises(TypeError, match="node index must be an integer"):
        path.node("one")  # type: ignore[arg-type]


def test_repeated_path_construction_is_deterministic() -> None:
    def build_path() -> tuple[list[int], list[int]]:
        circuit = Circuit("deterministic")
        signal = circuit.path("signal")
        signal.extend(circuit.node("signal.node[1]"))
        signal.extend(circuit.node("signal.node[2]"))
        return (
            [node.uid for node in signal.nodes],
            [node.uid for node in circuit.graph.nodes],
        )

    assert build_path() == build_path()
