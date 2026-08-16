"""Ordered symbolic graph storage for circuit construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .elements import ElementRef
from .nodes import Node
from .ports import Port


@dataclass
class CircuitGraph:
    """Own ordered nodes, elements, ports, and hierarchy metadata."""

    owner_id: int
    nodes: list[Node] = field(default_factory=list)
    elements: list[ElementRef] = field(default_factory=list)
    ports: dict[int, Port] = field(default_factory=dict)
    named_nodes: dict[str, Node] = field(default_factory=dict)
    named_elements: dict[str, ElementRef] = field(default_factory=dict)
    hierarchy: dict[str, dict[str, Any]] = field(default_factory=dict)
    path_nodes: dict[str, list[Node]] = field(default_factory=dict)
    legacy_path_bases: dict[str, int] = field(default_factory=dict)
    legacy_node_numbers: dict[Node, int] = field(default_factory=dict)
    ground: Node = field(init=False)

    def __post_init__(self) -> None:
        self.ground = Node(uid=0, owner_id=self.owner_id, name="ground", path="ground")
        self.nodes.append(self.ground)
        self.named_nodes[self.ground.path] = self.ground
