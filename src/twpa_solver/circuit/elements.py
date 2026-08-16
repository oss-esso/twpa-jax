"""Symbolic element handles."""

from __future__ import annotations

from dataclasses import dataclass, field

from .nodes import Node


@dataclass
class ElementRef:
    """A mutable handle for an element owned by one :class:`Circuit`.

    Mutual-inductor endpoints are references to the two linear-inductor
    handles. All other endpoints are symbolic :class:`Node` objects.
    """

    n1: Node | ElementRef
    n2: Node | ElementRef
    value: float | int | str
    kind: str
    role: str
    name: str
    path: str
    cell_index: int | None = None
    owner_id: int = 0
    removed: bool = False
    auto_name: bool = False
    companion: ElementRef | None = field(default=None, repr=False, compare=False)
