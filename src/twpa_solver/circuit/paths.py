"""Ordered symbolic paths over the circuit graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from .nodes import Node
from .validation import validate_node


@dataclass
class Path:
    """An ordered route through symbolic nodes.

    The path is an ergonomic view. Nodes and elements remain owned by the
    circuit graph and are the authoritative representation.
    """

    name: str
    owner_id: int
    _nodes: list[Node] = field(default_factory=list, repr=False)

    @property
    def start(self) -> Node:
        """Return the first node; it never changes after construction."""

        return self._nodes[0]

    @property
    def end(self) -> Node:
        """Return the current final node."""

        return self._nodes[-1]

    @property
    def nodes(self) -> list[Node]:
        """Return a copy of the ordered path nodes."""

        return list(self._nodes)

    def node(self, index: int) -> Node:
        """Return a node by its path index."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"{self.name}: node index must be an integer")
        try:
            return self._nodes[index]
        except IndexError as error:
            raise IndexError(f"{self.name}.node[{index}] is out of range") from error

    def extend(self, node: Node) -> Node:
        """Append a circuit-local node and make it the new endpoint."""

        validate_node(node, self.owner_id, self.name)
        self._nodes.append(node)
        return node

    def __len__(self) -> int:
        return len(self._nodes)
