"""Symbolic circuit nodes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Node:
    """A circuit-local symbolic electrical connection point.

    ``uid`` is only a construction-order identifier. It is never a solver
    node number; compilation assigns solver numbers separately.
    """

    uid: int
    owner_id: int
    name: str | None = None
    path: str = ""
