"""Two-path directional-coupler builder."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from twpa_solver.builders.cpw_coupler import CPWConformalCoupler, CPWModeResult
from twpa_solver.builders.ipm import (
    CouplerDiscrete,
    IPMParams,
    calculate_conformal_discrete_params,
    make_coupler_discrete,
)

from ..handles import CouplerCellHandle, CouplerHandle
from ..paths import Path
from ..validation import validate_positive


@dataclass(frozen=True)
class ExplicitCouplerGeometry:
    """Explicit CPW geometry for the two- or three-conductor model."""

    gaps_um: Sequence[float]
    widths_um: Sequence[float]
    length_um: float

    def __post_init__(self) -> None:
        gaps = tuple(float(value) for value in self.gaps_um)
        widths = tuple(float(value) for value in self.widths_um)
        if len(widths) not in (2, 3):
            raise ValueError("widths_um must contain two or three conductors")
        if len(gaps) != len(widths) + 1:
            raise ValueError("gaps_um must contain one more value than widths_um")
        if any(not math.isfinite(value) or value <= 0.0 for value in (*gaps, *widths)):
            raise ValueError("explicit coupler gaps and widths must be positive")
        length = float(self.length_um)
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("explicit coupler length_um must be positive")
        object.__setattr__(self, "gaps_um", gaps)
        object.__setattr__(self, "widths_um", widths)
        object.__setattr__(self, "length_um", length)


class CouplerBuilders:
    """Build directional couplers from existing discrete coupler physics."""

    def add_directional_coupler(
        self,
        signal: Path,
        pump: Path,
        *,
        coupling_db: float | None = None,
        frequency: float | None = None,
        z0: float = 50.0,
        geometry: ExplicitCouplerGeometry | None = None,
        mode: str = "auto",
        cell_length_um: float = 10.0,
        name: str | None = None,
    ) -> CouplerHandle:
        """Append a discrete directional coupler to two distinct paths."""

        self._validate_paths(signal, pump)
        target_db = -14.0 if coupling_db is None else float(coupling_db)
        target_frequency = 8.0e9 if frequency is None else float(frequency)
        cell_length = validate_positive(cell_length_um, "cell_length_um", signal.name)
        impedance = validate_positive(z0, "z0", signal.name)
        discrete = self._resolve_discrete(
            target_db,
            target_frequency,
            impedance,
            geometry,
            mode,
            cell_length,
        )
        block_path = name or f"{signal.name}.directional_coupler"
        signal_in = signal.end
        pump_in = pump.end
        signal_current = signal_in
        pump_current = pump_in
        cells: list[CouplerCellHandle] = []

        for index in range(discrete.N_coupled):
            signal_right = self.node(f"{block_path}.signal.cell[{index}].right")
            signal_cell = self.add_tl_cell(
                signal_current,
                signal_right,
                L=discrete.L_cell,
                C=discrete.C_gnd_cell / (2.0 if index == 0 else 1.0),
                name=f"{block_path}.signal.cell[{index}]",
            )
            signal.extend(signal_right)

            pump_right = self.node(f"{block_path}.pump.cell[{index}].right")
            pump_cell = self.add_tl_cell(
                pump_current,
                pump_right,
                L=discrete.L_cell,
                C=discrete.C_gnd_cell / (2.0 if index == 0 else 1.0),
                name=f"{block_path}.pump.cell[{index}]",
            )
            pump.extend(pump_right)

            coupling = self.add_coupling_capacitor(
                signal_current,
                pump_current,
                discrete.Cc_cell / (2.0 if index == 0 else 1.0),
                name=f"Cc{signal_current.uid}_{pump_current.uid}",
                path=f"{block_path}.cell[{index}].coupling",
                auto_name=True,
            )
            signal_inductor = signal_cell.extras["L"]
            pump_inductor = pump_cell.extras["L"]
            mutual = self.add_mutual_inductor(
                signal_inductor,
                pump_inductor,
                discrete.K_ind,
                name=f"K{signal_current.uid}_{pump_current.uid}",
                path=f"{block_path}.cell[{index}].mutual",
                auto_name=True,
            )
            cells.append(
                CouplerCellHandle(
                    signal=signal_cell,
                    pump=pump_cell,
                    coupling=coupling,
                    mutual=mutual,
                )
            )
            signal_current = signal_right
            pump_current = pump_right

        self.add_capacitor(
            signal_current,
            self.ground,
            discrete.C_gnd_cell / 2.0,
            name=f"C{signal_current.uid}_{self.ground.uid}_end",
            path=f"{block_path}.signal.end",
            auto_name=True,
        )
        self.add_capacitor(
            pump_current,
            self.ground,
            discrete.C_gnd_cell / 2.0,
            name=f"C{pump_current.uid}_{self.ground.uid}_end",
            path=f"{block_path}.pump.end",
            auto_name=True,
        )
        self.add_coupling_capacitor(
            signal_current,
            pump_current,
            discrete.Cc_cell / 2.0,
            name=f"Cc{signal_current.uid}_{pump_current.uid}_end",
            path=f"{block_path}.end.coupling",
            auto_name=True,
        )
        metadata = {
            "coupling_db": float(discrete.geometry.k_db),
            "model": discrete.geometry.model,
            "N_coupled": discrete.N_coupled,
            "N_uncoupled": discrete.N_uncoupled,
        }
        self.graph.hierarchy[block_path] = metadata
        return CouplerHandle(
            path=block_path,
            signal_in=signal_in,
            signal_out=signal_current,
            pump_in=pump_in,
            pump_out=pump_current,
            cells=cells,
            geometry=discrete.geometry,
            metadata=metadata,
        )

    def _validate_paths(self, signal: Path, pump: Path) -> None:
        """Validate two distinct paths owned by this circuit."""

        if signal is pump:
            raise ValueError("directional coupler requires two distinct paths")
        if signal.owner_id != self.graph.owner_id:
            raise ValueError(f"{signal.name}: path belongs to another Circuit")
        if pump.owner_id != self.graph.owner_id:
            raise ValueError(f"{pump.name}: path belongs to another Circuit")

    def _resolve_discrete(
        self,
        coupling_db: float,
        frequency: float,
        z0: float,
        geometry: ExplicitCouplerGeometry | None,
        mode: str,
        cell_length_um: float,
    ) -> CouplerDiscrete:
        """Resolve explicit or optimized geometry through existing builders."""

        if geometry is not None:
            conformal = CPWConformalCoupler(
                list(geometry.gaps_um),
                list(geometry.widths_um),
                geometry.length_um,
            )
            parameters = conformal.parameters()
            result = CPWModeResult(
                gaps_um=tuple(geometry.gaps_um),
                widths_um=tuple(geometry.widths_um),
                length_um=geometry.length_um,
                coupling_db=float(parameters["coupling_db"]),
                z_eff_ohm=float(parameters["Z_eff"]),
                model="three_line" if len(geometry.widths_um) == 3 else "two_line",
            )
            return calculate_conformal_discrete_params(result, cell_length_um)
        params = IPMParams(
            coupling_dB=coupling_db,
            coupler_freq_hz=frequency,
            Z0=z0,
            cell_length_um=cell_length_um,
        )
        return make_coupler_discrete(params, mode)
