"""
twpa.nonlinear.supercell_hb
===========================

Supercell and periodic-surrogate harmonic-balance utilities for KI-TWPA lines.

This module provides a bridge between:

    full long layout
        -> repeated / averaged supercell representation
        -> reduced nonlinear HB solve
        -> convergence diagnostics versus supercell size and period count

The current implementation is a production-facing reference layer. It uses the
existing dense distributed HB backend on reduced surrogate layouts. It does not
yet implement a true Bloch-periodic nonlinear HB boundary condition with complex
phase-twisted unknowns. The API is designed so that such a backend can replace
the surrogate solver later.

Use cases
---------
1. Periodic loading studies:
       Extract or average one repeated period from a long layout.

2. Dense-HB reduction:
       Tile a representative supercell for a manageable number of periods.

3. Convergence studies:
       Sweep cells_per_supercell and n_periods_for_surrogate.

4. Industrial 100 mm path:
       Validate nonlinear behavior on periodic reduced surrogates before
       moving to block-banded / Newton-Krylov full-layout HB.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout, make_layout_from_arrays
from twpa.core.params import NonlinearParams
from twpa.linear.dispersion import (
    DispersionConfig,
    DispersionResult,
    extract_layout_dispersion,
)
from twpa.linear.cells import CellModelConfig
from twpa.linear.cascade import CascadeConfig
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    PumpHBLadderResult,
    solve_pump_hb_ladder,
)


ArrayLike = Any


class SupercellBoundaryKind(str, Enum):
    """
    Supercell nonlinear boundary model.

    OPEN_SURROGATE:
        Tile the supercell a finite number of times and solve the finite line.

    SINGLE_PERIOD_OPEN:
        Solve one extracted/averaged supercell as a short finite line.

    BLOCH_PHASE_PLACEHOLDER:
        Reserved for a future true phase-twisted periodic nonlinear HB backend.
        Currently raises if used for solving.
    """

    OPEN_SURROGATE = "open_surrogate"
    SINGLE_PERIOD_OPEN = "single_period_open"
    BLOCH_PHASE_PLACEHOLDER = "bloch_phase_placeholder"


class SupercellConstructionMethod(str, Enum):
    """
    How to construct the representative supercell.
    """

    EXTRACT_FIRST = "extract_first"
    EXTRACT_AT_INDEX = "extract_at_index"
    AVERAGE_PERIODS = "average_periods"


@dataclass(frozen=True)
class SupercellExtractionConfig:
    """
    Configuration for extracting or averaging a representative supercell.

    Parameters
    ----------
    cells_per_supercell:
        Number of fine cells per supercell.
    start_cell:
        Start index used for EXTRACT_AT_INDEX.
    method:
        Extraction/construction method.
    include_partial_final_period:
        Whether AVERAGE_PERIODS may use a final incomplete period. For periodic
        devices this should usually be False.
    name_suffix:
        Suffix appended to generated layout names.
    """

    cells_per_supercell: int
    start_cell: int = 0
    method: SupercellConstructionMethod = SupercellConstructionMethod.AVERAGE_PERIODS
    include_partial_final_period: bool = False
    name_suffix: str = "supercell"

    def __post_init__(self) -> None:
        if int(self.cells_per_supercell) <= 0:
            raise ValueError("cells_per_supercell must be positive")
        if int(self.start_cell) < 0:
            raise ValueError("start_cell must be non-negative")
        object.__setattr__(self, "cells_per_supercell", int(self.cells_per_supercell))
        object.__setattr__(self, "start_cell", int(self.start_cell))
        object.__setattr__(self, "method", SupercellConstructionMethod(self.method))

    def with_updates(self, **kwargs: Any) -> "SupercellExtractionConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells_per_supercell": self.cells_per_supercell,
            "start_cell": self.start_cell,
            "method": self.method.value,
            "include_partial_final_period": self.include_partial_final_period,
            "name_suffix": self.name_suffix,
        }


@dataclass(frozen=True)
class SupercellSurrogateConfig:
    """
    Configuration for finite-line surrogate construction from a supercell.

    Parameters
    ----------
    extraction:
        Supercell extraction/averaging config.
    boundary_kind:
        Nonlinear boundary model.
    n_periods_for_surrogate:
        Number of repeated periods in the finite surrogate line.
    preserve_total_length:
        If True, tile enough periods to approximately preserve the original
        length when boundary_kind is OPEN_SURROGATE and n_periods_for_surrogate
        is None.
    """

    extraction: SupercellExtractionConfig
    boundary_kind: SupercellBoundaryKind = SupercellBoundaryKind.OPEN_SURROGATE
    n_periods_for_surrogate: int | None = 8
    preserve_total_length: bool = False
    name: str = "supercell_surrogate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_kind", SupercellBoundaryKind(self.boundary_kind))
        if self.n_periods_for_surrogate is not None and int(self.n_periods_for_surrogate) <= 0:
            raise ValueError("n_periods_for_surrogate must be positive when provided")
        if self.n_periods_for_surrogate is not None:
            object.__setattr__(self, "n_periods_for_surrogate", int(self.n_periods_for_surrogate))

    def with_updates(self, **kwargs: Any) -> "SupercellSurrogateConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction": self.extraction.to_dict(),
            "boundary_kind": self.boundary_kind.value,
            "n_periods_for_surrogate": self.n_periods_for_surrogate,
            "preserve_total_length": self.preserve_total_length,
            "name": self.name,
        }


@dataclass(frozen=True)
class SupercellSurrogateResult:
    """
    Result of constructing a supercell surrogate layout.
    """

    source_layout: LineLayout
    supercell_layout: LineLayout
    surrogate_layout: LineLayout
    config: SupercellSurrogateConfig
    n_source_periods: int
    n_surrogate_periods: int
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_layout": self.source_layout.summary(),
            "supercell_layout": self.supercell_layout.summary(),
            "surrogate_layout": self.surrogate_layout.summary(),
            "config": self.config.to_dict(),
            "n_source_periods": self.n_source_periods,
            "n_surrogate_periods": self.n_surrogate_periods,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SupercellPumpHBConfig:
    """
    Supercell pump-HB workflow config.
    """

    surrogate: SupercellSurrogateConfig
    pump: PumpHBLadderConfig = PumpHBLadderConfig()
    name: str = "supercell_pump_hb"

    def with_updates(self, **kwargs: Any) -> "SupercellPumpHBConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surrogate": self.surrogate.to_dict(),
            "pump": self.pump.to_dict(),
            "name": self.name,
        }


@dataclass(frozen=True)
class SupercellPumpHBResult:
    """
    Pump-HB result on a supercell surrogate layout.
    """

    surrogate: SupercellSurrogateResult
    pump_result: PumpHBLadderResult
    config: SupercellPumpHBConfig
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.pump_result.converged

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "surrogate": self.surrogate.to_dict(),
            "pump_result": self.pump_result.to_dict(),
            "config": self.config.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SupercellConvergencePoint:
    """
    One point in a supercell convergence sweep.
    """

    cells_per_supercell: int
    n_periods_for_surrogate: int
    result: SupercellPumpHBResult

    @property
    def converged(self) -> bool:
        return self.result.converged

    def scalar_metrics(self) -> dict[str, float | int | bool]:
        profile = self.result.pump_result.profile
        residual = self.result.pump_result.residual
        return {
            "cells_per_supercell": self.cells_per_supercell,
            "n_periods_for_surrogate": self.n_periods_for_surrogate,
            "surrogate_n_cells": self.result.surrogate.surrogate_layout.n_cells,
            "converged": self.converged,
            "residual_norm": float(residual.norm),
            "max_I_over_Istar": float(profile.max_pump_current_ratio),
            "max_node_voltage_abs_V": float(profile.max_node_voltage_abs_V),
            "max_branch_current_peak_A": float(profile.max_branch_current_peak_time_A),
            "pump_output_input_gain_db": float(profile.output_to_input_voltage_gain_db),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scalar_metrics(),
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class SupercellConvergenceSweepResult:
    """
    Sweep over supercell sizes and finite-surrogate period counts.
    """

    points: tuple[SupercellConvergencePoint, ...]
    metadata: Mapping[str, Any] | None = None

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_converged(self) -> int:
        return sum(1 for p in self.points if p.converged)

    @property
    def converged(self) -> bool:
        return self.n_converged == self.n_points

    def metric_array(self, key: str) -> jax.Array:
        return jnp.asarray([p.scalar_metrics()[key] for p in self.points])

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "n_converged": self.n_converged,
            "converged": self.converged,
            "points": [p.to_dict() for p in self.points],
            "metadata": dict(self.metadata or {}),
        }


def n_complete_supercells(layout: LineLayout, cells_per_supercell: int) -> int:
    """
    Number of complete supercells in a layout.
    """
    if cells_per_supercell <= 0:
        raise ValueError("cells_per_supercell must be positive")
    return int(layout.n_cells // cells_per_supercell)


def extract_supercell_layout(
    layout: LineLayout,
    *,
    cells_per_supercell: int,
    start_cell: int = 0,
    name: str | None = None,
) -> LineLayout:
    """
    Extract one contiguous supercell from a layout.
    """
    c = int(cells_per_supercell)
    s = int(start_cell)

    if c <= 0:
        raise ValueError("cells_per_supercell must be positive")
    if s < 0:
        raise ValueError("start_cell must be non-negative")
    if s + c > layout.n_cells:
        raise ValueError(
            f"Requested cells [{s}, {s + c}) exceed layout size {layout.n_cells}"
        )

    sl = slice(s, s + c)

    return make_layout_from_arrays(
        length_m=layout.length_m[sl],
        L_series_H=layout.L_series_H[sl],
        C_shunt_F=layout.C_shunt_F[sl],
        R_series_ohm=layout.R_series_ohm[sl],
        G_shunt_S=layout.G_shunt_S[sl],
        C_stub_F=layout.C_stub_F[sl],
        L_res_H=layout.L_res_H[sl],
        C_res_F=layout.C_res_F[sl],
        C_couple_F=layout.C_couple_F[sl],
        z0_ohm=layout.z0_ohm,
        name=name or f"{layout.name}_cell{s}_to_{s + c}",
        metadata={
            **dict(layout.metadata or {}),
            "source": "extract_supercell_layout",
            "source_layout": layout.name,
            "start_cell": s,
            "cells_per_supercell": c,
        },
    )


def average_periodic_supercell_layout(
    layout: LineLayout,
    *,
    cells_per_supercell: int,
    include_partial_final_period: bool = False,
    name: str | None = None,
) -> LineLayout:
    """
    Average corresponding cell positions across all periods.

    This is useful when the layout is intended to be periodic but contains
    weak fabrication disorder or small generated variations. For each position
    inside the period, parameters are averaged over all complete periods.
    """
    c = int(cells_per_supercell)
    if c <= 0:
        raise ValueError("cells_per_supercell must be positive")

    n_complete = layout.n_cells // c
    remainder = layout.n_cells % c

    if n_complete == 0:
        raise ValueError("Layout is shorter than one supercell")

    use_n = n_complete * c
    if include_partial_final_period and remainder:
        # Pad the final incomplete period with NaNs and nanmean it.
        n_periods = n_complete + 1

        def average_array(arr: jax.Array) -> jax.Array:
            base = np.asarray(arr[:use_n]).reshape(n_complete, c)
            partial = np.full((1, c), np.nan, dtype=base.dtype)
            partial[0, :remainder] = np.asarray(arr[use_n:])
            stacked = np.concatenate([base, partial], axis=0)
            return jnp.asarray(np.nanmean(stacked, axis=0), dtype=arr.dtype)

    else:
        n_periods = n_complete

        def average_array(arr: jax.Array) -> jax.Array:
            base = jnp.reshape(arr[:use_n], (n_complete, c))
            return jnp.mean(base, axis=0)

    return make_layout_from_arrays(
        length_m=average_array(layout.length_m),
        L_series_H=average_array(layout.L_series_H),
        C_shunt_F=average_array(layout.C_shunt_F),
        R_series_ohm=average_array(layout.R_series_ohm),
        G_shunt_S=average_array(layout.G_shunt_S),
        C_stub_F=average_array(layout.C_stub_F),
        L_res_H=average_array(layout.L_res_H),
        C_res_F=average_array(layout.C_res_F),
        C_couple_F=average_array(layout.C_couple_F),
        z0_ohm=layout.z0_ohm,
        name=name or f"{layout.name}_avg_period_{c}",
        metadata={
            **dict(layout.metadata or {}),
            "source": "average_periodic_supercell_layout",
            "source_layout": layout.name,
            "cells_per_supercell": c,
            "n_periods_averaged": n_periods,
            "include_partial_final_period": include_partial_final_period,
            "remainder_cells": int(remainder),
        },
    )


def tile_supercell_layout(
    supercell: LineLayout,
    *,
    n_periods: int,
    name: str | None = None,
) -> LineLayout:
    """
    Tile a supercell layout into a finite surrogate line.
    """
    n = int(n_periods)
    if n <= 0:
        raise ValueError("n_periods must be positive")

    def tile(arr: jax.Array) -> jax.Array:
        return jnp.tile(arr, (n,))

    return make_layout_from_arrays(
        length_m=tile(supercell.length_m),
        L_series_H=tile(supercell.L_series_H),
        C_shunt_F=tile(supercell.C_shunt_F),
        R_series_ohm=tile(supercell.R_series_ohm),
        G_shunt_S=tile(supercell.G_shunt_S),
        C_stub_F=tile(supercell.C_stub_F),
        L_res_H=tile(supercell.L_res_H),
        C_res_F=tile(supercell.C_res_F),
        C_couple_F=tile(supercell.C_couple_F),
        z0_ohm=supercell.z0_ohm,
        name=name or f"{supercell.name}_x{n}",
        metadata={
            **dict(supercell.metadata or {}),
            "source": "tile_supercell_layout",
            "supercell_layout": supercell.name,
            "n_periods": n,
        },
    )


def build_supercell_layout(
    layout: LineLayout,
    config: SupercellExtractionConfig,
) -> LineLayout:
    """
    Construct a representative supercell layout.
    """
    c = config.cells_per_supercell

    if config.method == SupercellConstructionMethod.EXTRACT_FIRST:
        return extract_supercell_layout(
            layout,
            cells_per_supercell=c,
            start_cell=0,
            name=f"{layout.name}_{config.name_suffix}_{c}_first",
        )

    if config.method == SupercellConstructionMethod.EXTRACT_AT_INDEX:
        return extract_supercell_layout(
            layout,
            cells_per_supercell=c,
            start_cell=config.start_cell,
            name=f"{layout.name}_{config.name_suffix}_{c}_start{config.start_cell}",
        )

    if config.method == SupercellConstructionMethod.AVERAGE_PERIODS:
        return average_periodic_supercell_layout(
            layout,
            cells_per_supercell=c,
            include_partial_final_period=config.include_partial_final_period,
            name=f"{layout.name}_{config.name_suffix}_{c}_avg",
        )

    raise ValueError(f"Unsupported supercell construction method {config.method}")


def build_supercell_surrogate_layout(
    layout: LineLayout,
    config: SupercellSurrogateConfig,
) -> SupercellSurrogateResult:
    """
    Build the finite layout that will be passed to the nonlinear HB backend.
    """
    if config.boundary_kind == SupercellBoundaryKind.BLOCH_PHASE_PLACEHOLDER:
        raise NotImplementedError(
            "True Bloch-periodic nonlinear HB is not implemented yet. "
            "Use OPEN_SURROGATE or SINGLE_PERIOD_OPEN."
        )

    supercell = build_supercell_layout(layout, config.extraction)
    n_source_periods = n_complete_supercells(layout, config.extraction.cells_per_supercell)

    if config.boundary_kind == SupercellBoundaryKind.SINGLE_PERIOD_OPEN:
        surrogate = supercell
        n_surrogate_periods = 1

    elif config.boundary_kind == SupercellBoundaryKind.OPEN_SURROGATE:
        if config.preserve_total_length:
            n_periods = max(1, int(round(layout.n_cells / supercell.n_cells)))
        else:
            n_periods = 1 if config.n_periods_for_surrogate is None else config.n_periods_for_surrogate

        surrogate = tile_supercell_layout(
            supercell,
            n_periods=n_periods,
            name=f"{layout.name}_{config.name}_period{supercell.n_cells}_x{n_periods}",
        )
        n_surrogate_periods = n_periods

    else:
        raise ValueError(f"Unsupported boundary kind {config.boundary_kind}")

    return SupercellSurrogateResult(
        source_layout=layout,
        supercell_layout=supercell,
        surrogate_layout=surrogate,
        config=config,
        n_source_periods=n_source_periods,
        n_surrogate_periods=n_surrogate_periods,
        metadata={
            "source": "build_supercell_surrogate_layout",
        },
    )


def solve_supercell_pump_hb(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    *,
    drive: PumpDriveConfig,
    config: SupercellPumpHBConfig,
    metadata: Mapping[str, Any] | None = None,
) -> SupercellPumpHBResult:
    """
    Build a supercell surrogate and solve pump-only HB on it.
    """
    surrogate = build_supercell_surrogate_layout(layout, config.surrogate)

    pump = solve_pump_hb_ladder(
        surrogate.surrogate_layout,
        nonlinear_params,
        drive=drive,
        pump_config=config.pump,
        metadata={
            "driver": "solve_supercell_pump_hb",
            "supercell_surrogate": surrogate.to_dict(),
            **dict(metadata or {}),
        },
    )

    return SupercellPumpHBResult(
        surrogate=surrogate,
        pump_result=pump,
        config=config,
        metadata=dict(metadata or {}),
    )


def sweep_supercell_pump_hb(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    *,
    drive: PumpDriveConfig,
    base_pump_config: PumpHBLadderConfig,
    cells_per_supercell_values: Sequence[int],
    n_periods_for_surrogate_values: Sequence[int] = (1, 2, 4, 8),
    construction_method: SupercellConstructionMethod = SupercellConstructionMethod.AVERAGE_PERIODS,
    boundary_kind: SupercellBoundaryKind = SupercellBoundaryKind.OPEN_SURROGATE,
) -> SupercellConvergenceSweepResult:
    """
    Sweep supercell size and surrogate length for pump-HB convergence studies.
    """
    points: list[SupercellConvergencePoint] = []

    for cells_per_supercell in cells_per_supercell_values:
        for n_periods in n_periods_for_surrogate_values:
            extraction = SupercellExtractionConfig(
                cells_per_supercell=int(cells_per_supercell),
                method=construction_method,
            )
            surrogate_cfg = SupercellSurrogateConfig(
                extraction=extraction,
                boundary_kind=boundary_kind,
                n_periods_for_surrogate=int(n_periods),
                preserve_total_length=False,
            )
            cfg = SupercellPumpHBConfig(
                surrogate=surrogate_cfg,
                pump=base_pump_config,
                name=f"supercell_pump_c{cells_per_supercell}_p{n_periods}",
            )

            result = solve_supercell_pump_hb(
                layout,
                nonlinear_params,
                drive=drive,
                config=cfg,
                metadata={
                    "sweep": "sweep_supercell_pump_hb",
                    "cells_per_supercell": int(cells_per_supercell),
                    "n_periods_for_surrogate": int(n_periods),
                },
            )

            points.append(
                SupercellConvergencePoint(
                    cells_per_supercell=int(cells_per_supercell),
                    n_periods_for_surrogate=int(n_periods),
                    result=result,
                )
            )

    return SupercellConvergenceSweepResult(
        points=tuple(points),
        metadata={
            "drive": drive.to_dict(),
            "base_pump_config": base_pump_config.to_dict(),
            "construction_method": SupercellConstructionMethod(construction_method).value,
            "boundary_kind": SupercellBoundaryKind(boundary_kind).value,
        },
    )


def compare_supercell_layout_dispersion(
    full_layout: LineLayout,
    surrogate: SupercellSurrogateResult,
    frequency_hz: ArrayLike,
    *,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    dispersion_config: DispersionConfig | None = None,
) -> dict[str, Any]:
    """
    Compare dispersion of the full layout and a supercell surrogate.

    This is a diagnostic to ensure that a reduced periodic surrogate did not
    destroy the relevant pump/signal/idler linear physics.
    """
    f = jnp.asarray(frequency_hz, dtype=jnp.float64)

    full_disp = extract_layout_dispersion(
        f,
        full_layout,
        cell_model=cell_model or CellModelConfig(),
        cascade_config=cascade_config or CascadeConfig(),
        dispersion_config=dispersion_config or DispersionConfig(),
    )

    sur_disp = extract_layout_dispersion(
        f,
        surrogate.surrogate_layout,
        cell_model=cell_model or CellModelConfig(),
        cascade_config=cascade_config or CascadeConfig(),
        dispersion_config=dispersion_config or DispersionConfig(),
    )

    beta_full = full_disp.beta_preferred_rad_per_m
    beta_sur = sur_disp.beta_preferred_rad_per_m

    alpha_full = full_disp.alpha_preferred_np_per_m
    alpha_sur = sur_disp.alpha_preferred_np_per_m

    beta_err = beta_sur - beta_full
    alpha_err = alpha_sur - alpha_full

    return {
        "frequency_hz": {
            "min": float(f[0]),
            "max": float(f[-1]),
            "n": int(f.shape[0]),
        },
        "full_layout": full_layout.summary(),
        "surrogate_layout": surrogate.surrogate_layout.summary(),
        "beta_abs_error_max_rad_per_m": float(jnp.nanmax(jnp.abs(beta_err))),
        "beta_rms_error_rad_per_m": float(jnp.sqrt(jnp.nanmean(beta_err**2))),
        "alpha_abs_error_max_np_per_m": float(jnp.nanmax(jnp.abs(alpha_err))),
        "alpha_rms_error_np_per_m": float(jnp.sqrt(jnp.nanmean(alpha_err**2))),
        "full_dispersion": full_disp.to_dict(),
        "surrogate_dispersion": sur_disp.to_dict(),
        "surrogate": surrogate.to_dict(),
    }


def estimate_supercell_bloch_phase(
    dispersion: DispersionResult,
    *,
    cells_per_supercell: int,
    cell_length_m: float,
) -> jax.Array:
    """
    Estimate Bloch phase advance per supercell from beta(f).

    phi(f) = beta(f) * cells_per_supercell * cell_length_m
    """
    if cells_per_supercell <= 0:
        raise ValueError("cells_per_supercell must be positive")
    if cell_length_m <= 0.0:
        raise ValueError("cell_length_m must be positive")

    return (
        dispersion.beta_preferred_rad_per_m
        * float(cells_per_supercell)
        * float(cell_length_m)
    )


def convergence_sweep_table(result: SupercellConvergenceSweepResult) -> str:
    """
    Markdown table for a supercell pump convergence sweep.
    """
    lines = [
        "| cells/supercell | surrogate periods | surrogate cells | status | residual | max I/I* | pump gain dB |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]

    for point in result.points:
        m = point.scalar_metrics()
        status = "pass" if point.converged else "fail"
        lines.append(
            f"| {m['cells_per_supercell']} | {m['n_periods_for_surrogate']} | "
            f"{m['surrogate_n_cells']} | {status} | "
            f"{m['residual_norm']:.6e} | {m['max_I_over_Istar']:.6e} | "
            f"{m['pump_output_input_gain_db']:.6g} |"
        )

    return "\n".join(lines)


__all__ = [
    "SupercellBoundaryKind",
    "SupercellConstructionMethod",
    "SupercellExtractionConfig",
    "SupercellSurrogateConfig",
    "SupercellSurrogateResult",
    "SupercellPumpHBConfig",
    "SupercellPumpHBResult",
    "SupercellConvergencePoint",
    "SupercellConvergenceSweepResult",
    "n_complete_supercells",
    "extract_supercell_layout",
    "average_periodic_supercell_layout",
    "tile_supercell_layout",
    "build_supercell_layout",
    "build_supercell_surrogate_layout",
    "solve_supercell_pump_hb",
    "sweep_supercell_pump_hb",
    "compare_supercell_layout_dispersion",
    "estimate_supercell_bloch_phase",
    "convergence_sweep_table",
]