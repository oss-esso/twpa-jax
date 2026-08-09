from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twpa_solver.builders.blocks import BlockRecord
from twpa_solver.builders.ipm import Element
from twpa_solver.design.errors import DesignResolutionError


@dataclass(frozen=True)
class PortRecord:
    number: int
    node: int


@dataclass(frozen=True)
class CompiledDesign:
    name: str
    elements: list[Element]
    cursors: dict[str, int]
    named_nodes: dict[str, int]
    named_elements: dict[str, str]
    ports: dict[int, PortRecord]
    blocks: list[BlockRecord]
    metadata: dict[str, Any]
    component_plan: Any = None

    def _resolve(self, table: dict[str, Any], path: str, kind: str) -> Any:
        if path in table:
            return table[path]
        choices = [key for key in table if key.startswith(path.rsplit(".", 1)[0])][:3]
        hint = f"; closest paths: {choices}" if choices else ""
        raise DesignResolutionError(f"{kind} path {path!r} not found{hint}")

    def resolve_node(self, path: str) -> int:
        return int(self._resolve(self.named_nodes, path, "node"))

    def resolve_element(self, path: str) -> str:
        return str(self._resolve(self.named_elements, path, "element"))
