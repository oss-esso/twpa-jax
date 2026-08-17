"""Repeated transmission-line builder."""

from __future__ import annotations

from ..handles import LineHandle
from ..paths import Path
from ..validation import validate_positive


class TransmissionLineBuilders:
    """Build transmission lines by composing TL cells."""

    TECHNOLOGY_DEFAULTS = {"L": "Ll", "C": "Cl"}

    def add_transmission_line(
        self,
        path: Path,
        *,
        cells: int,
        L: float | None = None,
        C: float | None = None,
        name: str | None = None,
    ) -> LineHandle:
        """Append ``cells`` transmission-line cells to ``path``."""

        if path.owner_id != self.graph.owner_id:
            raise ValueError(f"{path.name}: path belongs to another Circuit")
        if not isinstance(cells, int) or isinstance(cells, bool) or cells < 0:
            raise ValueError(f"{path.name}: cells must be a non-negative integer")
        inductance = validate_positive(
            float(self._resolve_builder_parameter(
                "L", L, self.TECHNOLOGY_DEFAULTS, path.name
            )),
            "L",
            path.name,
        )
        capacitance = validate_positive(
            float(self._resolve_builder_parameter(
                "C", C, self.TECHNOLOGY_DEFAULTS, path.name
            )),
            "C",
            path.name,
        )
        block_path = name or f"{path.name}.transmission_line"
        start = path.end
        nodes = [start]
        handles = []
        current = start
        for index in range(cells):
            right = self.node(f"{block_path}.cell[{index}].right")
            cell = self.add_tl_cell(
                current,
                right,
                L=inductance,
                C=capacitance,
                name=f"{block_path}.cell[{index}]",
            )
            handles.append(cell)
            path.extend(right)
            nodes.append(right)
            current = right
        return LineHandle(
            path=block_path,
            input=start,
            output=current,
            _nodes=nodes,
            cells=handles,
        )
