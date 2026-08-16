"""Fabrication v3 IPM with explicit three-conductor couplers."""

from __future__ import annotations

from dataclasses import dataclass

from twpa_solver.builders.cpw_coupler import optimize_cpw_coupler
from twpa_solver.circuit import Circuit, ExplicitCouplerGeometry, coupler_leakage_db
from twpa_solver.circuit.paths import Path


@dataclass(frozen=True)
class IPMv3Config:
    """Readable v3 dimensions and electrical defaults.

    The coupler dimensions are Prometheus fabrication starting dimensions for
    the conformal optimizer. ``coupler_length_um`` is retained as the
    explicit, unoptimized fallback length when optimization is disabled.
    """

    cell_count: int = 36
    arrays_per_row: int = 18
    rows: int = 6
    Lj: float = 123.9e-12
    Cj: float = 145.0e-15
    Cg: float = 66.0e-15
    coupling_db: float = -25.0
    coupler_frequency_hz: float = 10.0e9
    coupler_length_um: float = 2738.2160926784595
    coupler_z0_ohm: float = 50.0
    optimize_coupler: bool = True
    cell_length_um: float = 10.0
    apply_coupler_leakage: bool = False
    coupler_gaps_um: tuple[float, ...] = (5.5, 5.0, 5.0, 5.5)
    coupler_widths_um: tuple[float, ...] = (9.186, 15.0, 9.186)


def _add_junction_array_line(
    circuit: Circuit,
    path: Path,
    *,
    cells: int,
    Lj: float,
    Cj: float,
    Cg: float,
    name: str,
    cell_index_start: int,
) -> int:
    """Append one 36-cell lumped-array segment to a v3 row."""

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
        circuit.add_jj(
            current,
            right,
            Lj,
            Cj,
            name=f"{name}.cell[{index}].JJ",
            cell_index=cell_index_start + index,
        )
        path.extend(right)
        current = right
    return cell_index_start + cells


def _coupling_db(config: IPMv3Config, coupler_number: int) -> float:
    """Select nominal v3 coupling or explicit Prometheus leakage correction."""

    if not config.apply_coupler_leakage:
        return config.coupling_db
    return coupler_leakage_db(config.coupling_db, coupler_number)


def build_ipm_v3(config: IPMv3Config | None = None) -> Circuit:
    """Build six v3 rows, with one explicit coupler per two rows.

    The row dimensions are ``36 x 18 x 6``.  Signal chaining is direct from
    row to row; the pump path advances only at the three couplers, matching the
    v3 fabrication routing. The fabrication starting cross-section is passed
    through the bounded conformal optimizer and then emitted through
    :class:`ExplicitCouplerGeometry`. Prometheus uses one straight section
    plus meanders; the discrete model represents the resulting unrolled
    length only and does not model bends or bend discontinuities.
    """

    settings = config or IPMv3Config()
    circuit = Circuit("ipm_v3")
    signal = circuit.path("signal")
    pump = circuit.path("pump")
    circuit.add_port(signal.start, number=1)
    circuit.add_resistor(signal.start, circuit.ground, 50.0)
    circuit.add_port(pump.start, number=3)
    circuit.add_resistor(pump.start, circuit.ground, 50.0)

    if settings.optimize_coupler:
        optimized = optimize_cpw_coupler(
            coupling_db=settings.coupling_db,
            frequency_hz=settings.coupler_frequency_hz,
            z0=settings.coupler_z0_ohm,
            model="three_line",
            initial_gaps_um=settings.coupler_gaps_um,
            initial_widths_um=settings.coupler_widths_um,
        )
        geometry = ExplicitCouplerGeometry(
            gaps_um=optimized.gaps_um,
            widths_um=optimized.widths_um,
            length_um=optimized.length_um,
        )
    else:
        geometry = ExplicitCouplerGeometry(
            gaps_um=settings.coupler_gaps_um,
            widths_um=settings.coupler_widths_um,
            length_um=settings.coupler_length_um,
        )
    cell_index = 0
    coupler_number = 0
    for row in range(settings.rows):
        for array in range(settings.arrays_per_row):
            cell_index = _add_junction_array_line(
                circuit,
                signal,
                cells=settings.cell_count,
                Lj=settings.Lj,
                Cj=settings.Cj,
                Cg=settings.Cg,
                name=f"row[{row}].array[{array}]",
                cell_index_start=cell_index,
            )
        if row % 2 == 1:
            coupler_number += 1
            circuit.add_directional_coupler(
                signal,
                pump,
                coupling_db=_coupling_db(settings, coupler_number),
                frequency=settings.coupler_frequency_hz,
                geometry=geometry,
                cell_length_um=settings.cell_length_um,
                name=f"coupler[{coupler_number - 1}]",
            )

    circuit.add_resistor(signal.end, circuit.ground, 50.0)
    circuit.add_port(signal.end, number=2)
    circuit.add_resistor(pump.end, circuit.ground, 50.0)
    circuit.add_port(pump.end, number=4)
    return circuit


__all__ = ["IPMv3Config", "build_ipm_v3"]
