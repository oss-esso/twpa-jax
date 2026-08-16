"""Compilation from symbolic graph objects to the existing Element IR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from twpa_solver.builders.ipm import Element, build_matrices

from .elements import ElementRef
from .graph import CircuitGraph
from .nodes import Node
from .ports import CompiledPort
from .validation import validate_graph


NodeNumbering = Literal["creation", "legacy"]


@dataclass
class CompiledCircuit:
    """Solver-facing result of compiling a symbolic circuit graph."""

    elements: list[Element]
    node_map: dict[Node, int]
    reverse_node_map: dict[int, Node]
    ports: dict[int, CompiledPort]
    hierarchy: dict[str, dict[str, Any]]
    metadata: dict[str, Any]

    def matrices(self) -> dict[str, Any]:
        """Assemble matrices through the unchanged legacy matrix assembler."""

        return build_matrices(self.elements)


def _allocate_nodes(
    graph: CircuitGraph,
    node_numbering: NodeNumbering,
) -> dict[Node, int]:
    if node_numbering == "legacy":
        node_map: dict[Node, int] = {graph.ground: 0}
        used_numbers = {0}
        for node, solver_number in graph.legacy_node_numbers.items():
            if node is graph.ground or solver_number == 0:
                raise ValueError("ground cannot receive a non-ground legacy number")
            if solver_number in used_numbers:
                raise ValueError(f"legacy solver node {solver_number} is assigned twice")
            node_map[node] = solver_number
            used_numbers.add(solver_number)
        next_number = 1
        for path_name, path_nodes in graph.path_nodes.items():
            base = graph.legacy_path_bases.get(path_name)
            if base is None:
                while next_number in used_numbers:
                    next_number += 1
                base = next_number
            for offset, node in enumerate(path_nodes):
                solver_number = graph.legacy_node_numbers.get(node, base + offset)
                if solver_number in used_numbers and node_map.get(node) != solver_number:
                    raise ValueError(
                        f"{path_name}: legacy path numbering overlaps solver node "
                        f"{solver_number}"
                    )
                existing = node_map.get(node)
                if existing is not None and existing != solver_number:
                    raise ValueError(
                        f"{node.path}: assigned two legacy solver nodes "
                        f"{existing} and {solver_number}"
                    )
                node_map[node] = solver_number
                used_numbers.add(solver_number)
            next_number = max(next_number, base + len(path_nodes))
        for node in graph.nodes:
            if node not in node_map:
                while next_number in used_numbers:
                    next_number += 1
                node_map[node] = next_number
                used_numbers.add(next_number)
                next_number += 1
        return node_map
    return {
        node: solver_number
        for solver_number, node in enumerate(graph.nodes)
    }


def _compile_endpoint(
    endpoint: Node | ElementRef,
    node_map: dict[Node, int],
    element_names: dict[int, str],
) -> int | str:
    if isinstance(endpoint, Node):
        return node_map[endpoint]
    return element_names[id(endpoint)]


def _legacy_element_name(
    ref: ElementRef,
    node_to_solver: dict[int, int],
) -> str:
    """Rewrite generated node-bearing names for historical compatibility."""

    if not ref.auto_name:
        return ref.name

    if ref.kind in ("port", "resistor"):
        prefix = "P" if ref.kind == "port" else "R"
        if (
            isinstance(ref.n1, Node)
            and isinstance(ref.n2, Node)
            and re.fullmatch(rf"{prefix}\d+", ref.name)
        ):
            return (
                f"{prefix}{node_to_solver[ref.n1.uid]}_"
                f"{node_to_solver[ref.n2.uid]}"
            )
        return ref.name

    if ref.kind == "mutual_inductor_k":
        if not isinstance(ref.n1, ElementRef) or not isinstance(ref.n2, ElementRef):
            return ref.name
        if not isinstance(ref.n1.n1, Node) or not isinstance(ref.n2.n1, Node):
            return ref.name
        match = re.fullmatch(r"K(\d+)_(\d+)(.*)", ref.name)
        if match is None:
            return ref.name
        if (int(match.group(1)), int(match.group(2))) != (
            ref.n1.n1.uid,
            ref.n2.n1.uid,
        ):
            return ref.name
        return (
            f"K{node_to_solver[ref.n1.n1.uid]}_"
            f"{node_to_solver[ref.n2.n1.uid]}{match.group(3)}"
        )

    prefixes = {
        "capacitor": "C",
        "coupling_capacitor": "Cc",
        "linear_inductor": "L",
        "josephson_inductor": "Lj",
    }
    prefix = prefixes.get(ref.kind)
    if prefix is None:
        return ref.name
    if not isinstance(ref.n1, Node) or not isinstance(ref.n2, Node):
        return ref.name
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)_(\d+)(.*)", ref.name)
    if match is None:
        return ref.name
    if (int(match.group(1)), int(match.group(2))) != (
        ref.n1.uid,
        ref.n2.uid,
    ):
        return ref.name
    return (
        f"{prefix}{node_to_solver[ref.n1.uid]}_"
        f"{node_to_solver[ref.n2.uid]}{match.group(3)}"
    )


def compile_graph(
    graph: CircuitGraph,
    name: str,
    node_numbering: NodeNumbering = "creation",
) -> CompiledCircuit:
    """Compile a graph while preserving element creation order."""

    if node_numbering not in ("creation", "legacy"):
        raise ValueError(
            "node_numbering must be 'creation' or 'legacy', "
            f"got {node_numbering!r}"
        )
    validate_graph(graph)
    node_map = _allocate_nodes(graph, node_numbering)
    reverse_node_map = {number: node for node, number in node_map.items()}
    node_to_solver = {node.uid: number for node, number in node_map.items()}
    element_names = {
        id(ref): (
            _legacy_element_name(ref, node_to_solver)
            if node_numbering == "legacy"
            else ref.name
        )
        for ref in graph.elements
    }
    elements: list[Element] = []

    for ref in graph.elements:
        if ref.removed:
            continue
        elements.append(
            Element(
                name=element_names[id(ref)],
                n1=_compile_endpoint(ref.n1, node_map, element_names),
                n2=_compile_endpoint(ref.n2, node_map, element_names),
                value=ref.value,
                kind=ref.kind,
                role=ref.role,
                cell_index=ref.cell_index,
            )
        )


    ports = {
        number: CompiledPort(
            number=number,
            node=node_map[port.node],
            impedance=port.impedance,
        )
        for number, port in sorted(graph.ports.items())
    }
    metadata = {
        "name": name,
        "node_numbering": node_numbering,
        "node_count": len(graph.nodes) - 1,
        "element_count": len(elements),
    }
    return CompiledCircuit(
        elements=elements,
        node_map=node_map,
        reverse_node_map=reverse_node_map,
        ports=ports,
        hierarchy=dict(graph.hierarchy),
        metadata=metadata,
    )
