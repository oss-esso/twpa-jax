"""Josephson transmission-line cell builder."""

from __future__ import annotations

from ..handles import CellHandle
from ..nodes import Node
from ..validation import validate_positive


class JJCellBuilders:
    """Build JJ cells from the primitive capacitor and JJ builders."""

    def add_jj_cell(
        self,
        n1: Node,
        n2: Node,
        *,
        Lj: float,
        Cj: float,
        Cg: float,
        name: str | None = None,
        cell_index: int | None = None,
    ) -> CellHandle:
        """Add one ground capacitor and one Josephson junction pair."""

        lj = validate_positive(Lj, "Lj", name or "jj_cell")
        cj = validate_positive(Cj, "Cj", name or "jj_cell")
        cg = validate_positive(Cg, "Cg", name or "jj_cell")
        cell_path = name or f"cell[{cell_index if cell_index is not None else 0}]"
        capacitor = self.add_capacitor(
            n1,
            self.ground,
            cg,
            name=f"C{n1.uid}_{self.ground.uid}",
            role="jtl_cg",
            cell_index=cell_index,
            path=f"{cell_path}.Cg",
            auto_name=True,
        )
        junction = self.add_jj(
            n1,
            n2,
            lj,
            cj,
            name=f"JJ{n1.uid}_{n2.uid}",
            cell_index=cell_index,
            path=cell_path,
            lj_name=f"Lj{n1.uid}_{n2.uid}",
            cj_name=f"C{n1.uid}_{n2.uid}",
            auto_name=True,
        )
        return CellHandle(
            path=cell_path,
            left=n1,
            right=n2,
            Lj=junction,
            Cj=junction.companion,
            Cg=capacitor,
        )
