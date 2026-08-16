"""Public builders for repeated IPM sections."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..handles import IPMArrayHandle, IPMRowHandle, IPMSectionHandle
from ..paths import Path
from ..validation import validate_positive


def coupler_leakage_db(coupling_db: float, coupler_number: int) -> float:
    """Return the Prometheus per-coupler leakage correction in dB.

    ``coupling_db`` is the nominal power coupling and ``coupler_number`` is
    one-based.  The correction is opt-in at the design level; callers that do
    not request it retain the nominal coupling unchanged, as the v3 layout
    does.
    """

    nominal = float(coupling_db)
    if not math.isfinite(nominal) or nominal >= 0.0:
        raise ValueError("coupling_db must be finite and negative")
    if not isinstance(coupler_number, int) or isinstance(coupler_number, bool):
        raise TypeError("coupler_number must be an integer")
    if coupler_number <= 0:
        raise ValueError("coupler_number must be positive")
    coupling = 10.0 ** (nominal / 10.0)
    leakage = 1.0 - coupler_number * coupling
    if leakage <= 0.0:
        raise ValueError("coupler leakage correction is undefined")
    return 10.0 * math.log10(coupling / leakage)


class IPMBuilders:
    """Compose IPM sections from the public line and coupler builders."""

    def add_ipm_section(
        self,
        signal: Path,
        pump: Path,
        *,
        rows: int,
        array_length: int,
        Lj: float,
        Cj: float,
        Cg: float,
        short_tl_cells: int,
        long_tl_cells: int,
        coupler_section_cells: int,
        coupler: Mapping[str, Any] | None = None,
        name: str | None = None,
        tl_L: float | None = None,
        tl_C: float | None = None,
        cell_index_start: int = 0,
    ) -> IPMSectionHandle:
        """Append IPM rows, optional routing, and an optional coupler."""

        self._validate_ipm_paths(signal, pump)
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise ValueError("rows must be a positive integer")
        if not isinstance(array_length, int) or isinstance(array_length, bool):
            raise TypeError("array_length must be an integer")
        if array_length <= 0:
            raise ValueError("array_length must be a positive integer")
        for label, value in (
            ("short_tl_cells", short_tl_cells),
            ("long_tl_cells", long_tl_cells),
            ("coupler_section_cells", coupler_section_cells),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(cell_index_start, int) or isinstance(cell_index_start, bool):
            raise TypeError("cell_index_start must be an integer")
        if cell_index_start < 0:
            raise ValueError("cell_index_start must not be negative")
        lj = validate_positive(Lj, "Lj", signal.name)
        cj = validate_positive(Cj, "Cj", signal.name)
        cg = validate_positive(Cg, "Cg", signal.name)
        if short_tl_cells or long_tl_cells or coupler_section_cells:
            if tl_L is None or tl_C is None:
                raise ValueError("tl_L and tl_C are required for IPM routing cells")
            line_l = validate_positive(tl_L, "tl_L", signal.name)
            line_c = validate_positive(tl_C, "tl_C", signal.name)
        else:
            line_l = line_c = 1.0

        block_path = name or f"{signal.name}.ipm_section"
        row_handles: list[IPMRowHandle] = []
        for row_index in range(rows):
            line = self.add_jj_line(
                signal,
                cells=array_length,
                Lj=lj,
                Cj=cj,
                Cg=cg,
                boundary_caps=True,
                cell_index_start=cell_index_start + row_index * array_length,
                name=f"{block_path}.row[{row_index}].array[0]",
            )
            array = IPMArrayHandle(
                path=f"{block_path}.row[{row_index}].array[0]",
                cells=line.cells,
            )
            row_handles.append(
                IPMRowHandle(
                    path=f"{block_path}.row[{row_index}]",
                    array=[array],
                )
            )
            if row_index < rows - 1 or coupler is None:
                self.add_transmission_line(
                    signal,
                    cells=short_tl_cells,
                    L=line_l,
                    C=line_c,
                    name=f"{block_path}.row[{row_index}].short_tl",
                )

        coupler_handle = None
        if coupler is not None:
            if not isinstance(coupler, Mapping):
                raise TypeError("coupler must be a mapping or None")
            self.add_transmission_line(
                signal,
                cells=long_tl_cells,
                L=line_l,
                C=line_c,
                name=f"{block_path}.long_tl.signal",
            )
            self.add_transmission_line(
                pump,
                cells=coupler_section_cells,
                L=line_l,
                C=line_c,
                name=f"{block_path}.long_tl.pump",
            )
            coupler_handle = self.add_directional_coupler(
                signal,
                pump,
                **dict(coupler),
                name=f"{block_path}.coupler",
            )
        self.graph.hierarchy[block_path] = {
            "rows": rows,
            "array_length": array_length,
            "coupler": coupler_handle is not None,
        }
        return IPMSectionHandle(
            path=block_path,
            row=row_handles,
            coupler=coupler_handle,
            next_cell_index=cell_index_start + rows * array_length,
        )

    def _validate_ipm_paths(self, signal: Path, pump: Path) -> None:
        """Validate the two paths used by an IPM section."""

        if signal is pump:
            raise ValueError("IPM section requires two distinct paths")
        if signal.owner_id != self.graph.owner_id:
            raise ValueError(f"{signal.name}: path belongs to another Circuit")
        if pump.owner_id != self.graph.owner_id:
            raise ValueError(f"{pump.name}: path belongs to another Circuit")
