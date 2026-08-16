"""RF-SQUID cell builder."""

from __future__ import annotations

from ..handles import CellHandle
from ..nodes import Node
from ..validation import validate_positive


class RFSquidCellBuilders:
    """Build RF-SQUID cells from primitive elements."""

    def add_rf_squid_cell(
        self,
        left: Node,
        right: Node | None,
        *,
        Lw: float,
        Lm: float,
        Lpar: float,
        Lj: float,
        Cj: float,
        Cg: float,
        name: str,
        cell_index: int,
    ) -> CellHandle:
        """Add one RF-SQUID cell and its two split ground capacitors."""

        values = {
            "Lw": validate_positive(Lw, "Lw", name),
            "Lm": validate_positive(Lm, "Lm", name),
            "Lpar": validate_positive(Lpar, "Lpar", name),
            "Lj": validate_positive(Lj, "Lj", name),
            "Cj": validate_positive(Cj, "Cj", name),
            "Cg": validate_positive(Cg, "Cg", name),
        }
        wire = self.node(f"{name}.wire")
        branch = self.node(f"{name}.branch")
        if right is None:
            right = self.node(f"{name}.right")
        lw = self.add_inductor(
            left,
            wire,
            values["Lw"],
            name=f"{name}.Lw",
            role="rf_squid_lw",
            cell_index=cell_index,
            path=f"{name}.Lw",
            auto_name=True,
        )
        lm = self.add_inductor(
            wire,
            right,
            values["Lm"],
            name=f"{name}.Lm",
            role="rf_squid_lm",
            cell_index=cell_index,
            path=f"{name}.Lm",
            auto_name=True,
        )
        lpar = self.add_inductor(
            wire,
            branch,
            values["Lpar"],
            name=f"{name}.Lpar",
            role="rf_squid_lpar",
            cell_index=cell_index,
            path=f"{name}.Lpar",
            auto_name=True,
        )
        junction = self.add_jj(
            branch,
            right,
            values["Lj"],
            values["Cj"],
            name=f"{name}.JJ",
            cell_index=cell_index,
            path=name,
            lj_name=f"Lj{branch.uid}_{right.uid}",
            cj_name=f"C{branch.uid}_{right.uid}",
            auto_name=True,
        )
        cg_left = self.add_capacitor(
            left,
            self.ground,
            values["Cg"] / 2.0,
            name=f"{name}.Cg_left",
            role="rf_squid_cg",
            cell_index=cell_index,
            path=f"{name}.Cg_left",
            auto_name=True,
        )
        cg_right = self.add_capacitor(
            right,
            self.ground,
            values["Cg"] / 2.0,
            name=f"{name}.Cg_right",
            role="rf_squid_cg",
            cell_index=cell_index,
            path=f"{name}.Cg_right",
            auto_name=True,
        )
        return CellHandle(
            path=name,
            left=left,
            right=right,
            Lj=junction,
            Cj=junction.companion,
            Cg=cg_right,
            extras={
                "Lw": lw,
                "Lm": lm,
                "Lpar": lpar,
                "Cg_left": cg_left,
            },
        )
