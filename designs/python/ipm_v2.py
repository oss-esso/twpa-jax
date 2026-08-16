"""Fabrication v2 IPM: three-junction lumped arrays in the v1 topology."""

from __future__ import annotations

from dataclasses import dataclass

from twpa_solver.circuit import Circuit, coupler_leakage_db
from twpa_solver.circuit.paths import Path


@dataclass(frozen=True)
class IPMv2Config:
    """Electrical and topological parameters for the v2 fabrication shape."""

    cell_count: int = 20
    arrays_per_row: int = 15
    rows_per_section: int = 3
    sections: int = 2
    junction_array_count: int = 3
    Lj: float = 123.9e-12
    Cj: float = 145.0e-15
    Cg: float = 66.0e-15
    coupling_db: float = -14.0
    coupler_frequency_hz: float = 8.0e9
    cell_length_um: float = 10.0
    apply_coupler_leakage: bool = False


def _add_junction_array_line(
    circuit: Circuit,
    path: Path,
    *,
    cells: int,
    array_count: int,
    Lj: float,
    Cj: float,
    Cg: float,
    name: str,
    cell_index_start: int,
) -> int:
    """Append one fabrication array and return the next cell index."""

    current = path.end
    for index in range(cells):
        right = circuit.node(f"{name}.cell[{index}].right")
        circuit.add_capacitor(
            current,
            circuit.ground,
            Cg,
            name=f"{name}.cell[{index}].Cg",
            role="jtl_cg",
            cell_index=cell_index_start + index,
        )
        circuit.add_jj_array(
            current,
            right,
            Lj=Lj,
            Cj=Cj,
            count=array_count,
            name=f"{name}.cell[{index}].JJ",
            cell_index=cell_index_start + index,
        )
        path.extend(right)
        current = right
    return cell_index_start + cells


def _coupling_db(config: IPMv2Config, coupler_number: int) -> float:
    """Select nominal or explicitly requested v1-style leakage correction."""

    if not config.apply_coupler_leakage:
        return config.coupling_db
    return coupler_leakage_db(config.coupling_db, coupler_number)


def build_ipm_v2(config: IPMv2Config | None = None) -> Circuit:
    """Build the v2 shape: ``20 x 15 x 3 x 2`` array junctions.

    Each physical three-junction series array is represented by one lumped
    Josephson branch.  The two paths and two couplers retain the v1 routing
    shape; leakage correction is opt-in through ``apply_coupler_leakage``.
    """

    settings = config or IPMv2Config()
    circuit = Circuit("ipm_v2")
    signal = circuit.path("signal")
    pump = circuit.path("pump")
    circuit.add_port(signal.start, number=1)
    circuit.add_resistor(signal.start, circuit.ground, 50.0)
    circuit.add_port(pump.start, number=3)
    circuit.add_resistor(pump.start, circuit.ground, 50.0)

    cell_index = 0
    coupler_number = 0
    for section in range(settings.sections):
        coupler_number += 1
        circuit.add_directional_coupler(
            signal,
            pump,
            coupling_db=_coupling_db(settings, coupler_number),
            frequency=settings.coupler_frequency_hz,
            mode="cached",
            cell_length_um=settings.cell_length_um,
            name=f"section[{section}].coupler",
        )
        for row in range(settings.rows_per_section):
            for array in range(settings.arrays_per_row):
                cell_index = _add_junction_array_line(
                    circuit,
                    signal,
                    cells=settings.cell_count,
                    array_count=settings.junction_array_count,
                    Lj=settings.Lj,
                    Cj=settings.Cj,
                    Cg=settings.Cg,
                    name=f"section[{section}].row[{row}].array[{array}]",
                    cell_index_start=cell_index,
                )

    circuit.add_resistor(signal.end, circuit.ground, 50.0)
    circuit.add_port(signal.end, number=2)
    circuit.add_resistor(pump.end, circuit.ground, 50.0)
    circuit.add_port(pump.end, number=4)
    return circuit


__all__ = ["IPMv2Config", "build_ipm_v2"]
