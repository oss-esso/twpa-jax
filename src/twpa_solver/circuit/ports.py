"""Symbolic and compiled circuit ports."""

from __future__ import annotations

from dataclasses import dataclass

from .nodes import Node


@dataclass(frozen=True, slots=True)
class Port:
    """An external interface attached to a symbolic node."""

    number: int
    node: Node
    impedance: float


@dataclass(frozen=True, slots=True)
class CompiledPort:
    """A port after its symbolic node has received a solver number."""

    number: int
    node: int
    impedance: float
