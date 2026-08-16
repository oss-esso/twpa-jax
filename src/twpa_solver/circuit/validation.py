"""Validation for symbolic circuit graphs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .elements import ElementRef
from .graph import CircuitGraph
from .nodes import Node


def validate_positive(value: float | int, label: str, path: str) -> float:
    """Return a finite positive value or raise a path-specific error."""

    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{path}: {label} must be finite and positive")
    return numeric


def validate_node(node: Node, owner_id: int, path: str) -> None:
    """Validate that a node belongs to the current circuit."""

    if not isinstance(node, Node):
        raise TypeError(f"{path}: expected a Node")
    if node.owner_id != owner_id:
        raise ValueError(f"{path}: node belongs to another Circuit")


def validate_element_ref(ref: ElementRef, owner_id: int, path: str) -> None:
    """Validate that an element handle belongs to the current circuit."""

    if not isinstance(ref, ElementRef):
        raise TypeError(f"{path}: expected an ElementRef")
    if ref.owner_id != owner_id:
        raise ValueError(f"{path}: element belongs to another Circuit")


def _validate_endpoint(
    endpoint: Node | ElementRef,
    graph: CircuitGraph,
    element_path: str,
    node_ids: set[int],
    element_ids: set[int],
) -> None:
    if isinstance(endpoint, Node):
        validate_node(endpoint, graph.owner_id, element_path)
        if id(endpoint) not in node_ids:
            raise ValueError(f"{element_path}: node is not registered in the graph")
        return

    validate_element_ref(endpoint, graph.owner_id, element_path)
    if id(endpoint) not in element_ids:
        raise ValueError(f"{element_path}: element reference is not registered")
    if endpoint.removed:
        raise ValueError(f"{element_path}: references removed element {endpoint.path}")
    if endpoint.kind != "linear_inductor":
        raise ValueError(
            f"{element_path}: mutual inductor endpoint must be a linear inductor"
        )


def validate_graph(graph: CircuitGraph) -> None:
    """Validate ownership, endpoints, ports, and active references."""

    node_ids = {id(node) for node in graph.nodes}
    element_ids = {id(element) for element in graph.elements}
    for node in graph.nodes:
        validate_node(node, graph.owner_id, node.path or "node")

    for element in graph.elements:
        validate_element_ref(element, graph.owner_id, element.path)
        if element.removed:
            continue
        _validate_endpoint(element.n1, graph, element.path, node_ids, element_ids)
        _validate_endpoint(element.n2, graph, element.path, node_ids, element_ids)

    for number, port in graph.ports.items():
        if number != port.number:
            raise ValueError(f"port[{number}]: key does not match port number")
        validate_node(port.node, graph.owner_id, f"port[{number}]")
        if id(port.node) not in node_ids:
            raise ValueError(f"port[{number}]: node is not registered in the graph")
        validate_positive(port.impedance, "impedance", f"port[{number}]")


def ensure_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    """Validate a mapping used for hierarchy metadata."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected a mapping")
    return value
