"""Repeated RF-SQUID line builder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..handles import LineHandle
from ..paths import Path
from ..validation import validate_positive


class RFSquidLineBuilders:
    """Build RF-SQUID lines by composing RF-SQUID cells."""

    def add_rf_squid_line(
        self,
        path: Path,
        *,
        cells: int,
        Ic: float,
        Lm: float,
        Lw: float,
        Lpar: float,
        Cj: float,
        Cg: float | None = None,
        Cg_pattern: Sequence[float] | None = None,
        Cg_pattern_counts: Sequence[int] | None = None,
        Lj: float | None = None,
        name: str | None = None,
    ) -> LineHandle:
        """Append an RF-SQUID line matching the legacy block topology."""

        if path.owner_id != self.graph.owner_id:
            raise ValueError(f"{path.name}: path belongs to another Circuit")
        if not isinstance(cells, int) or isinstance(cells, bool) or cells <= 0:
            raise ValueError(f"{path.name}: cells must be a positive integer")
        current_ic = validate_positive(Ic, "Ic", path.name)
        # Deliberately NOT PHI0_REDUCED / current_ic.  The legacy builder
        # (builders/blocks.py:149) evaluates phi0 / (2*pi*Ic); regrouping as
        # (phi0/2*pi) / Ic differs in the last bit for ~46% of Ic values and
        # breaks element-for-element parity.  Do not "simplify" this.
        current_lj = (
            2.067833848e-15 / (2.0 * np.pi * current_ic)
            if Lj is None
            else validate_positive(Lj, "Lj", path.name)
        )
        values = self._resolve_cg_values(
            Cg,
            Cg_pattern,
            Cg_pattern_counts,
            path.name,
        )
        block_path = name or f"{path.name}.rf_squid_line"
        start = path.end
        nodes = [start]
        handles = []
        current = start
        for index in range(cells):
            cell = self.add_rf_squid_cell(
                current,
                None,
                Lw=Lw,
                Lm=Lm,
                Lpar=Lpar,
                Lj=current_lj,
                Cj=Cj,
                Cg=values[index % len(values)],
                name=f"{block_path}.cell[{index}]",
                cell_index=index,
            )
            right = cell.right
            assert right is not None
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

    def _resolve_cg_values(
        self,
        Cg: float | None,
        Cg_pattern: Sequence[float] | None,
        Cg_pattern_counts: Sequence[int] | None,
        path: str,
    ) -> list[float]:
        if Cg_pattern is None:
            if Cg is None:
                raise ValueError(f"{path}: provide Cg or Cg_pattern")
            return [validate_positive(Cg, "Cg", path)]
        if not Cg_pattern:
            raise ValueError(f"{path}.Cg_pattern must not be empty")
        counts = Cg_pattern_counts or [1] * len(Cg_pattern)
        if len(counts) != len(Cg_pattern):
            raise ValueError(f"{path}.Cg_pattern_counts must match Cg_pattern")
        values: list[float] = []
        for value, count in zip(Cg_pattern, counts):
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(f"{path}: Cg pattern counts must be positive integers")
            values.extend([validate_positive(value, "Cg", path)] * count)
        return values
