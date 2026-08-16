"""Structured handles returned by Phase 3 builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .elements import ElementRef
from .nodes import Node


@dataclass
class BlockHandle:
    """Base handle carrying a stable hierarchy path."""

    path: str
    children: dict[str, BlockHandle] = field(default_factory=dict)


@dataclass
class CellHandle(BlockHandle):
    """Handle for one physical cell and its primitive elements."""

    left: Node | None = None
    right: Node | None = None
    Lj: ElementRef | None = None
    Cj: ElementRef | None = None
    Cg: ElementRef | None = None
    extras: dict[str, ElementRef] = field(default_factory=dict)


@dataclass
class LineHandle(BlockHandle):
    """Handle for a repeated line and its ordered cells."""

    input: Node | None = None
    output: Node | None = None
    _nodes: list[Node] = field(default_factory=list, repr=False)
    cells: list[CellHandle] = field(default_factory=list)

    def node(self, index: int) -> Node:
        """Return a line node by its zero-based line index."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"{self.path}: node index must be an integer")
        try:
            return self._nodes[index]
        except IndexError as error:
            raise IndexError(f"{self.path}.node[{index}] is out of range") from error

    def cell(self, index: int) -> CellHandle:
        """Return a generated cell by its zero-based cell index."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"{self.path}: cell index must be an integer")
        try:
            return self.cells[index]
        except IndexError as error:
            raise IndexError(f"{self.path}.cell[{index}] is out of range") from error


@dataclass
class CouplerCellHandle:
    """Handle for one paired signal/pump coupler cell."""

    signal: CellHandle
    pump: CellHandle
    coupling: ElementRef
    mutual: ElementRef


@dataclass
class CouplerHandle(BlockHandle):
    """Handle for a two-path directional coupler."""

    signal_in: Node | None = None
    signal_out: Node | None = None
    pump_in: Node | None = None
    pump_out: Node | None = None
    cells: list[CouplerCellHandle] = field(default_factory=list)
    geometry: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def cell(self, index: int) -> CouplerCellHandle:
        """Return a coupled signal/pump cell by zero-based index."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"{self.path}: cell index must be an integer")
        try:
            return self.cells[index]
        except IndexError as error:
            raise IndexError(f"{self.path}.cell[{index}] is out of range") from error

    def port(self, number: int) -> Node:
        """Return a terminal using the fabrication port-number convention."""

        terminals: dict[int, Node | None] = {
            1: self.signal_in,
            2: self.signal_out,
            3: self.pump_in,
            4: self.pump_out,
        }
        if number not in terminals:
            raise ValueError(f"{self.path}: port number must be one of 1, 2, 3, 4")
        terminal = terminals[number]
        if terminal is None:
            raise RuntimeError(f"{self.path}: terminal {number} is not initialized")
        return terminal


@dataclass
class IPMArrayHandle(BlockHandle):
    """Handle for one repeated Josephson array in an IPM row."""

    cells: list[CellHandle] = field(default_factory=list)

    def cell(self, index: int) -> CellHandle:
        """Return one Josephson cell by zero-based index."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"{self.path}: cell index must be an integer")
        try:
            return self.cells[index]
        except IndexError as error:
            raise IndexError(f"{self.path}.cell[{index}] is out of range") from error


@dataclass
class IPMRowHandle(BlockHandle):
    """Handle for one IPM row containing one or more arrays."""

    array: list[IPMArrayHandle] = field(default_factory=list)


@dataclass
class IPMSectionHandle(BlockHandle):
    """Handle for a repeated IPM section and its row/array hierarchy."""

    row: list[IPMRowHandle] = field(default_factory=list)
    coupler: CouplerHandle | None = None
    next_cell_index: int = 0
