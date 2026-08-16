"""Repeated Josephson transmission-line builder."""

from __future__ import annotations

from ..handles import LineHandle
from ..paths import Path
from ..profiles import Profile, evaluate_profile
from ..validation import validate_positive


class JJLineBuilders:
    """Build JJ lines by composing JJ cells and boundary capacitors."""

    def add_jj_line(
        self,
        path: Path,
        *,
        cells: int,
        Lj: float | Profile,
        Cj: float | Profile,
        Cg: float | Profile,
        name: str | None = None,
        boundary_caps: bool = True,
        cell_index_start: int = 0,
    ) -> LineHandle:
        """Append a JJ line with half-ground-capacitance end boundaries."""

        if path.owner_id != self.graph.owner_id:
            raise ValueError(f"{path.name}: path belongs to another Circuit")
        if not isinstance(cells, int) or isinstance(cells, bool) or cells <= 0:
            raise ValueError(f"{path.name}: cells must be a positive integer")
        if not isinstance(boundary_caps, bool):
            raise TypeError(f"{path.name}: boundary_caps must be a boolean")
        if not isinstance(cell_index_start, int) or isinstance(cell_index_start, bool):
            raise TypeError(f"{path.name}: cell_index_start must be an integer")
        if cell_index_start < 0:
            raise ValueError(f"{path.name}: cell_index_start must not be negative")
        lj_values = self._profile_values(Lj, cells, path.name, "Lj")
        cj_values = self._profile_values(Cj, cells, path.name, "Cj")
        cg_values = self._profile_values(Cg, cells, path.name, "Cg")
        block_path = name or f"{path.name}.jj_line"
        start = path.end
        nodes = [start]
        handles = []
        current = start
        for index in range(cells):
            right = self.node(f"{block_path}.cell[{index}].right")
            cell = self.add_jj_cell(
                current,
                right,
                Lj=lj_values[index],
                Cj=cj_values[index],
                Cg=(
                    cg_values[index] / 2.0
                    if boundary_caps and index == 0
                    else cg_values[index]
                ),
                name=f"{block_path}.cell[{index}]",
                cell_index=cell_index_start + index,
            )
            handles.append(cell)
            path.extend(right)
            nodes.append(right)
            current = right
        if boundary_caps:
            self.add_capacitor(
                current,
                self.ground,
                cg_values[-1] / 2.0,
                name=f"C{current.uid}_{self.ground.uid}_JTL_end",
                role="jtl_cg",
                cell_index=cell_index_start + cells - 1,
                path=f"{block_path}.end.Cg",
                auto_name=True,
            )
        return LineHandle(
            path=block_path,
            input=start,
            output=current,
            _nodes=nodes,
            cells=handles,
        )

    def _profile_values(
        self,
        value: float | Profile,
        cells: int,
        path: str,
        parameter: str,
    ) -> list[float]:
        """Resolve one scalar or profile into validated per-cell values."""

        if not isinstance(value, Profile):
            return [validate_positive(value, parameter, path)] * cells
        try:
            values = evaluate_profile(
                [value.to_segment()],
                n_cells=cells,
                cells_per_row=cells,
                base_value=float(value.start),
            )
        except ValueError as error:
            raise ValueError(f"{path}.{parameter}: {error}") from error
        return [validate_positive(item, parameter, path) for item in values]
