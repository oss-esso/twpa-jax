"""Transmission-line cell builder."""

from __future__ import annotations

from ..handles import CellHandle
from ..nodes import Node
from ..validation import validate_positive


class TLCellBuilders:
    """Build transmission-line cells from primitive elements."""

    def add_tl_cell(
        self,
        n1: Node,
        n2: Node,
        *,
        L: float,
        C: float,
        name: str | None = None,
        cell_index: int | None = None,
    ) -> CellHandle:
        """Add one shunt capacitor and series linear inductor cell."""

        inductance = validate_positive(L, "L", name or "tl_cell")
        capacitance = validate_positive(C, "C", name or "tl_cell")
        cell_path = name or f"cell[{cell_index if cell_index is not None else 0}]"
        capacitor = self.add_capacitor(
            n1,
            self.ground,
            capacitance,
            name=f"C{n1.uid}_{self.ground.uid}",
            cell_index=cell_index,
            path=f"{cell_path}.C",
            auto_name=True,
        )
        inductor = self.add_inductor(
            n1,
            n2,
            inductance,
            name=f"L{n1.uid}_{n2.uid}",
            cell_index=cell_index,
            path=f"{cell_path}.L",
            auto_name=True,
        )
        return CellHandle(
            path=cell_path,
            left=n1,
            right=n2,
            Cg=capacitor,
            extras={"L": inductor},
        )
