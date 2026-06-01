"""
twpa.workflows.industrial_100mm
===============================

Production-facing workflow for a 100 mm / 20,000-cell KI-TWPA simulator.

This module orchestrates the full simulation stack produced so far:

    layout/disorder
        -> linear RF validation
        -> dispersion extraction
        -> coarsening hierarchy
        -> pump-only nonlinear HB
        -> small-signal gain linearization
        -> operating-map reports

Important
---------
This file is the workflow layer. It does not replace the low-level solvers.

The current nonlinear backend is still dense/reference. Therefore, a true
20,000-cell nonlinear HB solve is not yet expected to run through the dense
solver. The intended industrial path is:

    1. run full 20,000-cell linear cascade / dispersion,
    2. run nonlinear HB on validated reduced/coarsened layouts,
    3. compare convergence over N_eff,
    4. later swap dense HB internals for block-banded / Newton-Krylov backend.

This workflow makes that separation explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import json

import jax
import jax.numpy as jnp
import numpy as np

from twpa.io.reports import jsonify

from twpa.core.layout import LineLayout, make_layout_from_arrays
from twpa.core.params import NonlinearParams
from twpa.linear.cells import (
    CellModelConfig,
    layout_cell_parameter_summary,
    validate_layout_cells,
)
from twpa.linear.cascade import (
    CascadeConfig,
    CascadeStrategy,
    LinearScanResult,
    run_linear_scan,
    validate_cascade,
)
from twpa.linear.coarsening import (
    CoarseningHierarchy,
    CoarseningHierarchyConfig,
    CoarseningMethod,
    compare_hierarchy_dispersion,
    generate_coarsening_hierarchy,
)
from twpa.linear.dispersion import (
    DispersionConfig,
    DispersionResult,
    StopbandMetric,
    compute_dp4wm_phase_matching,
    detect_stopbands,
    extract_layout_dispersion,
    nonlinear_delta_beta_dp4wm_simple,
    validate_dispersion_result,
)
from twpa.nonlinear.gain import (
    GainOperatingMap,
    GainSweepConfig,
    GainSweepResult,
    solve_gain_sweep_from_pump,
)
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpFrequencySweepResult,
    PumpHBLadderConfig,
    PumpHBLadderResult,
    solve_pump_hb_ladder,
    sweep_pump_frequency,
)


ArrayLike = Any
TargetPlanFactory = Callable[[PumpHBLadderResult], Any]
SweepConfigFactory = Callable[[Any], GainSweepConfig]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IndustrialStageStatus(str, Enum):
    """Workflow-stage status."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    PARTIAL = "partial"


class IndustrialRunMode(str, Enum):
    """Top-level workflow mode."""

    LINEAR_ONLY = "linear_only"
    PUMP_ONLY = "pump_only"
    GAIN = "gain"
    FULL = "full"


# ---------------------------------------------------------------------------
# JSON / serialization helpers
# ---------------------------------------------------------------------------

def _jsonify(obj: Any) -> Any:
    """
    Convert common numerical/JAX/Python objects into JSON-friendly data.
    """
    return jsonify(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, complex):
        return {
            "real": float(np.real(obj)),
            "imag": float(np.imag(obj)),
            "abs": float(abs(obj)),
        }
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _jsonify(obj.to_dict())
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            if np.iscomplexobj(arr):
                return _jsonify(complex(arr))
            return float(arr)
        return {
            "array_shape": tuple(int(s) for s in arr.shape),
            "array_dtype": str(arr.dtype),
            "min_abs": float(np.nanmin(np.abs(arr))) if arr.size else None,
            "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else None,
        }
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonify(jnp.asarray(obj))
    return obj


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """
    Write JSON with stable formatting.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_npz(path: str | Path, **arrays: Any) -> Path:
    """
    Write NPZ arrays using NumPy conversion.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in arrays.items()})
    return out


# ---------------------------------------------------------------------------
# Layout specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndustrialLayoutSpec:
    """
    Specification for a 100 mm / 20,000-cell synthetic KI-TWPA layout.

    Parameters
    ----------
    length_m:
        Physical line length. Industrial target is 0.1 m.
    n_cells:
        Number of lumped cells. Industrial target is 20,000.
    z0_ohm:
        Target characteristic impedance used when L_per_m/C_per_m are omitted.
    phase_velocity_m_per_s:
        Target phase velocity used when L_per_m/C_per_m are omitted.
    L_per_m_H, C_per_m_F:
        Optional explicit line parameters.
    R_per_m_ohm, G_per_m_S:
        Optional distributed loss.
    stub_period_cells:
        Period of capacitive loading. 0 disables periodic stub loading.
    stub_offset:
        Loaded-cell offset inside the period.
    C_stub_loaded_F:
        Absolute stub capacitance added to loaded cells.
    C_stub_loaded_fraction_of_base:
        Alternative relative stub capacitance on loaded cells.
    """

    length_m: float = 0.100
    n_cells: int = 20_000
    z0_ohm: float = 50.0
    phase_velocity_m_per_s: float = 1.20e8
    L_per_m_H: float | None = None
    C_per_m_F: float | None = None
    R_per_m_ohm: float = 0.0
    G_per_m_S: float = 0.0
    stub_period_cells: int = 0
    stub_offset: int = 0
    C_stub_loaded_F: float = 0.0
    C_stub_loaded_fraction_of_base: float = 0.0
    name: str = "industrial_100mm_20000cell"

    def __post_init__(self) -> None:
        if self.length_m <= 0.0:
            raise ValueError("length_m must be positive")
        if int(self.n_cells) <= 0:
            raise ValueError("n_cells must be positive")
        if self.z0_ohm <= 0.0:
            raise ValueError("z0_ohm must be positive")
        if self.phase_velocity_m_per_s <= 0.0:
            raise ValueError("phase_velocity_m_per_s must be positive")
        if self.L_per_m_H is not None and self.L_per_m_H <= 0.0:
            raise ValueError("L_per_m_H must be positive when provided")
        if self.C_per_m_F is not None and self.C_per_m_F <= 0.0:
            raise ValueError("C_per_m_F must be positive when provided")
        if self.R_per_m_ohm < 0.0:
            raise ValueError("R_per_m_ohm must be non-negative")
        if self.G_per_m_S < 0.0:
            raise ValueError("G_per_m_S must be non-negative")
        if int(self.stub_period_cells) < 0:
            raise ValueError("stub_period_cells must be non-negative")
        object.__setattr__(self, "n_cells", int(self.n_cells))
        object.__setattr__(self, "stub_period_cells", int(self.stub_period_cells))
        object.__setattr__(self, "stub_offset", int(self.stub_offset))

    @property
    def dx_m(self) -> float:
        return self.length_m / self.n_cells

    @property
    def effective_L_per_m_H(self) -> float:
        if self.L_per_m_H is not None:
            return self.L_per_m_H
        return self.z0_ohm / self.phase_velocity_m_per_s

    @property
    def effective_C_per_m_F(self) -> float:
        if self.C_per_m_F is not None:
            return self.C_per_m_F
        return 1.0 / (self.z0_ohm * self.phase_velocity_m_per_s)

    def with_updates(self, **kwargs: Any) -> "IndustrialLayoutSpec":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_m": self.length_m,
            "n_cells": self.n_cells,
            "dx_m": self.dx_m,
            "z0_ohm": self.z0_ohm,
            "phase_velocity_m_per_s": self.phase_velocity_m_per_s,
            "L_per_m_H": self.L_per_m_H,
            "C_per_m_F": self.C_per_m_F,
            "effective_L_per_m_H": self.effective_L_per_m_H,
            "effective_C_per_m_F": self.effective_C_per_m_F,
            "R_per_m_ohm": self.R_per_m_ohm,
            "G_per_m_S": self.G_per_m_S,
            "stub_period_cells": self.stub_period_cells,
            "stub_offset": self.stub_offset,
            "C_stub_loaded_F": self.C_stub_loaded_F,
            "C_stub_loaded_fraction_of_base": self.C_stub_loaded_fraction_of_base,
            "name": self.name,
        }


def build_industrial_layout(spec: IndustrialLayoutSpec) -> LineLayout:
    """
    Build a vectorized LineLayout from IndustrialLayoutSpec.
    """
    n = spec.n_cells
    dx = spec.dx_m

    L_cell = spec.effective_L_per_m_H * dx
    C_cell = spec.effective_C_per_m_F * dx
    R_cell = spec.R_per_m_ohm * dx
    G_cell = spec.G_per_m_S * dx

    length_m = jnp.full((n,), dx, dtype=jnp.float64)
    L_series_H = jnp.full((n,), L_cell, dtype=jnp.float64)
    C_shunt_F = jnp.full((n,), C_cell, dtype=jnp.float64)
    R_series_ohm = jnp.full((n,), R_cell, dtype=jnp.float64)
    G_shunt_S = jnp.full((n,), G_cell, dtype=jnp.float64)
    C_stub_F = jnp.zeros((n,), dtype=jnp.float64)

    if spec.stub_period_cells > 0:
        idx = jnp.arange(n)
        active = (idx % spec.stub_period_cells) == (spec.stub_offset % spec.stub_period_cells)
        stub_value = spec.C_stub_loaded_F
        if stub_value == 0.0 and spec.C_stub_loaded_fraction_of_base != 0.0:
            stub_value = spec.C_stub_loaded_fraction_of_base * C_cell
        C_stub_F = jnp.where(active, stub_value, 0.0)

    return make_layout_from_arrays(
        length_m=length_m,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
        R_series_ohm=R_series_ohm,
        G_shunt_S=G_shunt_S,
        C_stub_F=C_stub_F,
        L_res_H=0.0,
        C_res_F=0.0,
        C_couple_F=0.0,
        z0_ohm=spec.z0_ohm,
        name=spec.name,
        metadata={
            "source": "build_industrial_layout",
            "industrial_layout_spec": spec.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndustrialLinearStageConfig:
    """
    Linear validation and dispersion configuration.
    """

    frequency_min_hz: float = 1.0e9
    frequency_max_hz: float = 16.0e9
    n_frequency_points: int = 401
    cell_model: CellModelConfig = field(default_factory=CellModelConfig)
    cascade: CascadeConfig = field(
        default_factory=lambda: CascadeConfig(
            strategy=CascadeStrategy.AUTO,
            chunk_size=512,
            cells_per_supercell=1,
            allow_remainder=True,
        )
    )
    dispersion: DispersionConfig = field(default_factory=DispersionConfig)
    cutoff_safety_factor: float = 3.0
    expected_stopband: bool | None = None
    stopband_metric: StopbandMetric = StopbandMetric.BOTH
    stopband_s21_threshold_db: float = -10.0
    stopband_alpha_threshold_np_per_m: float = 1.0

    def __post_init__(self) -> None:
        if self.frequency_min_hz <= 0.0:
            raise ValueError("frequency_min_hz must be positive")
        if self.frequency_max_hz <= self.frequency_min_hz:
            raise ValueError("frequency_max_hz must exceed frequency_min_hz")
        if int(self.n_frequency_points) < 2:
            raise ValueError("n_frequency_points must be >= 2")
        object.__setattr__(self, "n_frequency_points", int(self.n_frequency_points))
        object.__setattr__(self, "stopband_metric", StopbandMetric(self.stopband_metric))

    def frequency_grid(self) -> jax.Array:
        return jnp.linspace(
            self.frequency_min_hz,
            self.frequency_max_hz,
            self.n_frequency_points,
            dtype=jnp.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency_min_hz": self.frequency_min_hz,
            "frequency_max_hz": self.frequency_max_hz,
            "n_frequency_points": self.n_frequency_points,
            "cell_model": self.cell_model.to_dict(),
            "cascade": self.cascade.to_dict(),
            "dispersion": self.dispersion.to_dict(),
            "cutoff_safety_factor": self.cutoff_safety_factor,
            "expected_stopband": self.expected_stopband,
            "stopband_metric": self.stopband_metric.value,
            "stopband_s21_threshold_db": self.stopband_s21_threshold_db,
            "stopband_alpha_threshold_np_per_m": self.stopband_alpha_threshold_np_per_m,
        }


@dataclass(frozen=True)
class IndustrialCoarseningStageConfig:
    """
    Effective-layout hierarchy for nonlinear convergence studies.
    """

    enabled: bool = True
    hierarchy: CoarseningHierarchyConfig = field(
        default_factory=lambda: CoarseningHierarchyConfig(
            target_cell_counts=(50, 100, 200, 500, 1000, 2000, 5000),
            method=CoarseningMethod.EXACT_GROUP_SUM,
            preserve_supercells=True,
            cells_per_supercell=1,
            include_original=True,
        )
    )
    compare_dispersion: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hierarchy": self.hierarchy.to_dict() if hasattr(self.hierarchy, "to_dict") else _jsonify(self.hierarchy),
            "compare_dispersion": self.compare_dispersion,
        }


@dataclass(frozen=True)
class IndustrialPumpStageConfig:
    """
    Pump-HB stage configuration.

    The dense backend should use pump_layout_target_n_cells to select a reduced
    layout. Full 20,000-cell nonlinear solves require a later structured backend.
    """

    enabled: bool = True
    pump_layout_target_n_cells: int = 200
    pump_drive: PumpDriveConfig = field(
        default_factory=lambda: PumpDriveConfig.from_current_rms(
            pump_frequency_hz=8.0e9,
            current_rms_A=1e-7,
        )
    )
    pump_config: PumpHBLadderConfig = field(default_factory=PumpHBLadderConfig)
    nonlinear_params: NonlinearParams | None = None
    sweep_frequencies_hz: tuple[float, ...] = ()
    reuse_previous_solution_in_sweep: bool = True

    def __post_init__(self) -> None:
        if int(self.pump_layout_target_n_cells) <= 0:
            raise ValueError("pump_layout_target_n_cells must be positive")
        object.__setattr__(self, "pump_layout_target_n_cells", int(self.pump_layout_target_n_cells))
        object.__setattr__(self, "sweep_frequencies_hz", tuple(float(x) for x in self.sweep_frequencies_hz))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pump_layout_target_n_cells": self.pump_layout_target_n_cells,
            "pump_drive": self.pump_drive.to_dict(),
            "pump_config": self.pump_config.to_dict(),
            "nonlinear_params": None if self.nonlinear_params is None else self.nonlinear_params.to_dict(),
            "sweep_frequencies_hz": list(self.sweep_frequencies_hz),
            "reuse_previous_solution_in_sweep": self.reuse_previous_solution_in_sweep,
        }


@dataclass(frozen=True)
class IndustrialGainStageConfig:
    """
    Gain stage configuration.

    The gain stage requires two callables passed to run_industrial_100mm_workflow:

        target_plan_factory(pump_result) -> FrequencyPlan
        sweep_config_factory(target_plan) -> GainSweepConfig

    This keeps the workflow compatible with different frequency-plan conventions.
    """

    enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


@dataclass(frozen=True)
class Industrial100mmWorkflowConfig:
    """
    Full industrial workflow config.
    """

    mode: IndustrialRunMode = IndustrialRunMode.FULL
    layout: IndustrialLayoutSpec = field(default_factory=IndustrialLayoutSpec)
    linear: IndustrialLinearStageConfig = field(default_factory=IndustrialLinearStageConfig)
    coarsening: IndustrialCoarseningStageConfig = field(default_factory=IndustrialCoarseningStageConfig)
    pump: IndustrialPumpStageConfig = field(default_factory=IndustrialPumpStageConfig)
    gain: IndustrialGainStageConfig = field(default_factory=IndustrialGainStageConfig)
    output_dir: str | None = None
    save_artifacts: bool = False
    name: str = "industrial_100mm_workflow"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", IndustrialRunMode(self.mode))

    def with_updates(self, **kwargs: Any) -> "Industrial100mmWorkflowConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "layout": self.layout.to_dict(),
            "linear": self.linear.to_dict(),
            "coarsening": self.coarsening.to_dict(),
            "pump": self.pump.to_dict(),
            "gain": self.gain.to_dict(),
            "output_dir": self.output_dir,
            "save_artifacts": self.save_artifacts,
            "name": self.name,
        }


# ---------------------------------------------------------------------------
# Stage results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndustrialLinearStageResult:
    """
    Linear validation result for one layout.
    """

    layout: LineLayout
    frequency_hz: jax.Array
    scan: LinearScanResult
    dispersion: DispersionResult
    cell_report: Mapping[str, Any]
    cascade_report: Mapping[str, Any]
    dispersion_report: Mapping[str, Any]
    stopbands: tuple[Mapping[str, Any], ...]
    status: IndustrialStageStatus
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "layout": self.layout.summary(),
            "frequency_hz": _jsonify(self.frequency_hz),
            "scan": self.scan.to_dict(),
            "dispersion": self.dispersion.to_dict(),
            "cell_report": dict(self.cell_report),
            "cascade_report": dict(self.cascade_report),
            "dispersion_report": dict(self.dispersion_report),
            "stopbands": [dict(s) for s in self.stopbands],
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class IndustrialCoarseningStageResult:
    """
    Coarsening hierarchy result.
    """

    hierarchy: CoarseningHierarchy | None
    dispersion_comparisons: tuple[Mapping[str, Any], ...]
    selected_pump_layout: LineLayout | None
    status: IndustrialStageStatus
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "hierarchy": None if self.hierarchy is None else self.hierarchy.to_dict(),
            "dispersion_comparisons": [dict(x) for x in self.dispersion_comparisons],
            "selected_pump_layout": None if self.selected_pump_layout is None else self.selected_pump_layout.summary(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class IndustrialPumpStageResult:
    """
    Pump-HB stage result.
    """

    result: PumpHBLadderResult | None
    sweep: PumpFrequencySweepResult | None
    layout: LineLayout | None
    status: IndustrialStageStatus
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "layout": None if self.layout is None else self.layout.summary(),
            "result": None if self.result is None else self.result.to_dict(),
            "sweep": None if self.sweep is None else self.sweep.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class IndustrialGainStageResult:
    """
    Gain stage result.
    """

    sweep: GainSweepResult | None
    operating_map: GainOperatingMap | None
    status: IndustrialStageStatus
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sweep": None if self.sweep is None else self.sweep.to_dict(),
            "operating_map": None if self.operating_map is None else self.operating_map.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class Industrial100mmWorkflowResult:
    """
    Full workflow result.
    """

    config: Industrial100mmWorkflowConfig
    layout: LineLayout
    linear: IndustrialLinearStageResult
    coarsening: IndustrialCoarseningStageResult | None
    pump: IndustrialPumpStageResult | None
    gain: IndustrialGainStageResult | None
    artifact_paths: Mapping[str, str]
    status: IndustrialStageStatus
    metadata: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.status == IndustrialStageStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "config": self.config.to_dict(),
            "layout": self.layout.summary(),
            "linear": self.linear.to_dict(),
            "coarsening": None if self.coarsening is None else self.coarsening.to_dict(),
            "pump": None if self.pump is None else self.pump.to_dict(),
            "gain": None if self.gain is None else self.gain.to_dict(),
            "artifact_paths": dict(self.artifact_paths),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Linear stage
# ---------------------------------------------------------------------------

def run_linear_stage(
    layout: LineLayout,
    config: IndustrialLinearStageConfig,
) -> IndustrialLinearStageResult:
    """
    Run cell validation, full cascade, S-parameter scan, and dispersion.
    """
    f = config.frequency_grid()

    cell_report = validate_layout_cells(
        f,
        layout,
        config=config.cell_model,
        cutoff_safety_factor=config.cutoff_safety_factor,
    ).to_dict()

    cascade_report = validate_cascade(
        f,
        layout,
        cell_model=config.cell_model,
        cascade_config=config.cascade,
    ).to_dict()

    scan = run_linear_scan(
        f,
        layout,
        cell_model=config.cell_model,
        cascade_config=config.cascade,
    )

    dispersion = extract_layout_dispersion(
        f,
        layout,
        cell_model=config.cell_model,
        cascade_config=config.cascade,
        dispersion_config=config.dispersion,
    )

    dispersion_report = validate_dispersion_result(
        dispersion,
        layout_name=layout.name,
        expected_stopband=config.expected_stopband,
        stopband_metric=config.stopband_metric,
        s21_threshold_db=config.stopband_s21_threshold_db,
        alpha_threshold_np_per_m=config.stopband_alpha_threshold_np_per_m,
    ).to_dict()

    stopbands = tuple(
        sb.to_dict()
        for sb in detect_stopbands(
            dispersion,
            metric=config.stopband_metric,
            s21_threshold_db=config.stopband_s21_threshold_db,
            alpha_threshold_np_per_m=config.stopband_alpha_threshold_np_per_m,
        )
    )

    passed = (
        bool(cell_report.get("cutoff_guard_passed", False))
        and bool(cascade_report.get("passed", False))
        and bool(dispersion_report.get("passed", False))
    )

    return IndustrialLinearStageResult(
        layout=layout,
        frequency_hz=f,
        scan=scan,
        dispersion=dispersion,
        cell_report=cell_report,
        cascade_report=cascade_report,
        dispersion_report=dispersion_report,
        stopbands=stopbands,
        status=IndustrialStageStatus.PASS if passed else IndustrialStageStatus.FAIL,
        metadata={
            "cell_parameter_summary": layout_cell_parameter_summary(layout),
        },
    )


# ---------------------------------------------------------------------------
# Coarsening stage
# ---------------------------------------------------------------------------

def select_layout_closest_n_cells(
    layouts: Sequence[LineLayout],
    target_n_cells: int,
) -> LineLayout:
    """
    Select the layout with cell count closest to target_n_cells.
    """
    if not layouts:
        raise ValueError("No layouts available")
    return min(layouts, key=lambda x: abs(x.n_cells - int(target_n_cells)))


def run_coarsening_stage(
    full_layout: LineLayout,
    linear_config: IndustrialLinearStageConfig,
    coarsening_config: IndustrialCoarseningStageConfig,
    *,
    pump_target_n_cells: int,
) -> IndustrialCoarseningStageResult:
    """
    Generate coarsening hierarchy and select pump layout.
    """
    if not coarsening_config.enabled:
        return IndustrialCoarseningStageResult(
            hierarchy=None,
            dispersion_comparisons=tuple(),
            selected_pump_layout=full_layout,
            status=IndustrialStageStatus.SKIP,
            metadata={"message": "coarsening disabled; using full layout"},
        )

    hierarchy = generate_coarsening_hierarchy(
        full_layout,
        coarsening_config.hierarchy,
    )

    comparisons: tuple[Mapping[str, Any], ...] = tuple()
    if coarsening_config.compare_dispersion:
        try:
            comps = compare_hierarchy_dispersion(
                linear_config.frequency_grid(),
                hierarchy,
                reference=full_layout,
                cell_model=linear_config.cell_model,
                dispersion_config=linear_config.dispersion,
            )
            comparisons = tuple(c.to_dict() for c in comps)
        except Exception as exc:
            comparisons = (
                {
                    "error": str(exc),
                    "message": "coarsening dispersion comparison failed",
                },
            )

    selected = select_layout_closest_n_cells(
        hierarchy.layouts,
        pump_target_n_cells,
    )

    # We keep this stage PASS if hierarchy exists. Dispersion comparison is
    # diagnostic because some coarsening methods are intentionally surrogate-like.
    return IndustrialCoarseningStageResult(
        hierarchy=hierarchy,
        dispersion_comparisons=comparisons,
        selected_pump_layout=selected,
        status=IndustrialStageStatus.PASS,
        metadata={
            "pump_target_n_cells": pump_target_n_cells,
            "selected_n_cells": selected.n_cells,
        },
    )


# ---------------------------------------------------------------------------
# Pump stage
# ---------------------------------------------------------------------------

def _default_nonlinear_params_or_raise(params: NonlinearParams | None) -> NonlinearParams:
    if params is None:
        raise ValueError(
            "Industrial pump/gain stages require NonlinearParams. "
            "Pass IndustrialPumpStageConfig(nonlinear_params=...)."
        )
    return params


def run_pump_stage(
    layout: LineLayout,
    config: IndustrialPumpStageConfig,
) -> IndustrialPumpStageResult:
    """
    Run pump-only HB on the selected reduced layout.
    """
    if not config.enabled:
        return IndustrialPumpStageResult(
            result=None,
            sweep=None,
            layout=layout,
            status=IndustrialStageStatus.SKIP,
            metadata={"message": "pump stage disabled"},
        )

    nonlinear = _default_nonlinear_params_or_raise(config.nonlinear_params)

    result = solve_pump_hb_ladder(
        layout,
        nonlinear,
        drive=config.pump_drive,
        pump_config=config.pump_config,
        metadata={
            "workflow_stage": "industrial_pump",
            "layout_name": layout.name,
            "n_cells": layout.n_cells,
        },
    )

    sweep_result = None
    if config.sweep_frequencies_hz:
        if config.pump_drive.kind.value == "available_power_dbm":
            sweep_result = sweep_pump_frequency(
                layout,
                nonlinear,
                pump_frequencies_hz=config.sweep_frequencies_hz,
                pump_power_dbm=config.pump_drive.available_power_dbm,
                source_impedance_ohm=config.pump_drive.source_impedance_ohm,
                pump_config=config.pump_config,
                reuse_previous_solution=config.reuse_previous_solution_in_sweep,
            )
        else:
            sweep_result = sweep_pump_frequency(
                layout,
                nonlinear,
                pump_frequencies_hz=config.sweep_frequencies_hz,
                pump_current_rms_A=config.pump_drive.current_rms_A,
                source_impedance_ohm=config.pump_drive.source_impedance_ohm,
                pump_config=config.pump_config,
                reuse_previous_solution=config.reuse_previous_solution_in_sweep,
            )

    status = IndustrialStageStatus.PASS if result.converged else IndustrialStageStatus.FAIL

    return IndustrialPumpStageResult(
        result=result,
        sweep=sweep_result,
        layout=layout,
        status=status,
        metadata={
            "pump_drive": config.pump_drive.to_dict(),
            "pump_config": config.pump_config.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Gain stage
# ---------------------------------------------------------------------------

def run_gain_stage(
    pump_result: PumpHBLadderResult | None,
    config: IndustrialGainStageConfig,
    *,
    target_plan_factory: TargetPlanFactory | None = None,
    sweep_config_factory: SweepConfigFactory | None = None,
) -> IndustrialGainStageResult:
    """
    Run gain stage from a pump result.

    This requires target_plan_factory and sweep_config_factory because the
    exact signal/idler plan convention is project-specific.
    """
    if not config.enabled:
        return IndustrialGainStageResult(
            sweep=None,
            operating_map=None,
            status=IndustrialStageStatus.SKIP,
            metadata={"message": "gain stage disabled"},
        )

    if pump_result is None:
        return IndustrialGainStageResult(
            sweep=None,
            operating_map=None,
            status=IndustrialStageStatus.FAIL,
            metadata={"message": "gain stage requested but no pump result exists"},
        )

    if target_plan_factory is None or sweep_config_factory is None:
        return IndustrialGainStageResult(
            sweep=None,
            operating_map=None,
            status=IndustrialStageStatus.FAIL,
            metadata={
                "message": (
                    "gain stage requires target_plan_factory and "
                    "sweep_config_factory callables"
                )
            },
        )

    target_plan = target_plan_factory(pump_result)
    sweep_config = sweep_config_factory(target_plan)

    sweep = solve_gain_sweep_from_pump(
        pump_result,
        target_plan=target_plan,
        sweep_config=sweep_config,
    )

    return IndustrialGainStageResult(
        sweep=sweep,
        operating_map=None,
        status=IndustrialStageStatus.PASS if sweep.passed else IndustrialStageStatus.FAIL,
        metadata={
            "target_plan": target_plan.to_dict(),
            "sweep_config": sweep_config.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Phase-matching helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndustrialPhaseMatchingReport:
    """
    DP4WM phase-matching diagnostic from the linear dispersion.
    """

    pump_frequency_hz: float
    signal_frequency_hz: jax.Array
    report: Mapping[str, Any]
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pump_frequency_hz": self.pump_frequency_hz,
            "signal_frequency_hz": _jsonify(self.signal_frequency_hz),
            "report": dict(self.report),
            "metadata": dict(self.metadata or {}),
        }


def compute_industrial_phase_matching_report(
    dispersion: DispersionResult,
    *,
    pump_frequency_hz: float,
    signal_frequency_hz: ArrayLike,
    pump_current_peak_A: float | None = None,
    I_star_A: float | None = None,
) -> IndustrialPhaseMatchingReport:
    """
    Compute a DP4WM phase-matching diagnostic from extracted dispersion.

    If pump_current_peak_A and I_star_A are given, includes a simple nonlinear
    phase correction. Otherwise, the nonlinear correction is zero.
    """
    if pump_current_peak_A is not None and I_star_A is not None:
        beta_p = dispersion.beta_at_frequency(pump_frequency_hz)
        delta_nl = nonlinear_delta_beta_dp4wm_simple(
            beta_pump_rad_per_m=beta_p,
            pump_current_peak_A=pump_current_peak_A,
            I_star_A=I_star_A,
        )
    else:
        delta_nl = 0.0

    pm = compute_dp4wm_phase_matching(
        dispersion,
        pump_frequency_hz=pump_frequency_hz,
        signal_frequency_hz=signal_frequency_hz,
        nonlinear_delta_beta_rad_per_m=delta_nl,
    )

    return IndustrialPhaseMatchingReport(
        pump_frequency_hz=pump_frequency_hz,
        signal_frequency_hz=jnp.asarray(signal_frequency_hz),
        report=pm.to_dict(),
        metadata={
            "pump_current_peak_A": pump_current_peak_A,
            "I_star_A": I_star_A,
        },
    )


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def export_industrial_artifacts(
    result: Industrial100mmWorkflowResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Export compact JSON/NPZ artifacts for a workflow result.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    paths["summary_json"] = str(
        write_json(out / "industrial_100mm_summary.json", result.to_dict())
    )

    # Linear arrays: these are the most useful compact numerical artifacts.
    paths["linear_npz"] = str(
        write_npz(
            out / "industrial_100mm_linear.npz",
            frequency_hz=result.linear.frequency_hz,
            s=result.linear.scan.s,
            s21_db=result.linear.scan.s21_db,
            beta_eff_rad_per_m=result.linear.scan.beta_eff_rad_per_m,
            group_delay_s=result.linear.scan.group_delay_s,
            beta_preferred_rad_per_m=result.linear.dispersion.beta_preferred_rad_per_m,
            alpha_preferred_np_per_m=result.linear.dispersion.alpha_preferred_np_per_m,
        )
    )

    if result.pump is not None and result.pump.result is not None:
        pump = result.pump.result
        paths["pump_npz"] = str(
            write_npz(
                out / "industrial_100mm_pump_solution.npz",
                node_voltage_coeffs_V=pump.state.node_voltage_coeffs_V,
                branch_current_coeffs_A=pump.state.branch_current_coeffs_A,
                frequencies_hz=pump.frequency_plan.frequencies_hz,
                injected_current_coeffs_A=pump.distributed_result.injected_current_coeffs_A,
            )
        )

    return paths


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def _combine_statuses(statuses: Sequence[IndustrialStageStatus]) -> IndustrialStageStatus:
    active = [s for s in statuses if s != IndustrialStageStatus.SKIP]
    if not active:
        return IndustrialStageStatus.SKIP
    if all(s == IndustrialStageStatus.PASS for s in active):
        return IndustrialStageStatus.PASS
    if any(s == IndustrialStageStatus.PASS for s in active):
        return IndustrialStageStatus.PARTIAL
    return IndustrialStageStatus.FAIL


def run_industrial_100mm_workflow(
    config: Industrial100mmWorkflowConfig | None = None,
    *,
    target_plan_factory: TargetPlanFactory | None = None,
    sweep_config_factory: SweepConfigFactory | None = None,
) -> Industrial100mmWorkflowResult:
    """
    Run the industrial 100 mm / 20,000-cell workflow.

    Parameters
    ----------
    config:
        Workflow configuration.
    target_plan_factory:
        Optional gain-stage callable:
            target_plan_factory(pump_result) -> FrequencyPlan
    sweep_config_factory:
        Optional gain-stage callable:
            sweep_config_factory(target_plan) -> GainSweepConfig

    Returns
    -------
    Industrial100mmWorkflowResult
    """
    cfg = config or Industrial100mmWorkflowConfig()
    layout = build_industrial_layout(cfg.layout)

    linear = run_linear_stage(layout, cfg.linear)

    coarsening_result = None
    pump_layout = layout

    if cfg.mode in {
        IndustrialRunMode.PUMP_ONLY,
        IndustrialRunMode.GAIN,
        IndustrialRunMode.FULL,
    }:
        coarsening_result = run_coarsening_stage(
            layout,
            cfg.linear,
            cfg.coarsening,
            pump_target_n_cells=cfg.pump.pump_layout_target_n_cells,
        )
        if coarsening_result.selected_pump_layout is not None:
            pump_layout = coarsening_result.selected_pump_layout

    pump_result = None
    if cfg.mode in {
        IndustrialRunMode.PUMP_ONLY,
        IndustrialRunMode.GAIN,
        IndustrialRunMode.FULL,
    }:
        pump_result = run_pump_stage(pump_layout, cfg.pump)

    gain_result = None
    if cfg.mode in {IndustrialRunMode.GAIN, IndustrialRunMode.FULL}:
        gain_result = run_gain_stage(
            None if pump_result is None else pump_result.result,
            cfg.gain,
            target_plan_factory=target_plan_factory,
            sweep_config_factory=sweep_config_factory,
        )

    statuses = [linear.status]
    if coarsening_result is not None:
        statuses.append(coarsening_result.status)
    if pump_result is not None:
        statuses.append(pump_result.status)
    if gain_result is not None:
        statuses.append(gain_result.status)

    status = _combine_statuses(statuses)

    result = Industrial100mmWorkflowResult(
        config=cfg,
        layout=layout,
        linear=linear,
        coarsening=coarsening_result,
        pump=pump_result,
        gain=gain_result,
        artifact_paths={},
        status=status,
        metadata={
            "mode": cfg.mode.value,
            "jax_backend": jax.default_backend(),
        },
    )

    artifact_paths: dict[str, str] = {}
    if cfg.save_artifacts:
        if cfg.output_dir is None:
            raise ValueError("save_artifacts=True requires output_dir")
        artifact_paths = export_industrial_artifacts(result, cfg.output_dir)
        result = replace(result, artifact_paths=artifact_paths)

    return result


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_linear_only_100mm_config(
    *,
    output_dir: str | None = None,
    save_artifacts: bool = False,
) -> Industrial100mmWorkflowConfig:
    """
    Fast constructor for full 20,000-cell linear validation only.
    """
    return Industrial100mmWorkflowConfig(
        mode=IndustrialRunMode.LINEAR_ONLY,
        output_dir=output_dir,
        save_artifacts=save_artifacts,
        name="industrial_100mm_linear_only",
    )


def make_reduced_pump_100mm_config(
    *,
    nonlinear_params: NonlinearParams,
    pump_frequency_hz: float,
    pump_current_rms_A: float,
    pump_layout_target_n_cells: int = 200,
    output_dir: str | None = None,
    save_artifacts: bool = False,
) -> Industrial100mmWorkflowConfig:
    """
    Constructor for linear full-layout validation plus reduced-layout pump HB.
    """
    return Industrial100mmWorkflowConfig(
        mode=IndustrialRunMode.PUMP_ONLY,
        pump=IndustrialPumpStageConfig(
            enabled=True,
            pump_layout_target_n_cells=pump_layout_target_n_cells,
            pump_drive=PumpDriveConfig.from_current_rms(
                pump_frequency_hz=pump_frequency_hz,
                current_rms_A=pump_current_rms_A,
            ),
            nonlinear_params=nonlinear_params,
        ),
        output_dir=output_dir,
        save_artifacts=save_artifacts,
        name="industrial_100mm_reduced_pump",
    )


def summarize_workflow_markdown(result: Industrial100mmWorkflowResult) -> str:
    """
    Produce a compact markdown summary.
    """
    lines = [
        "# Industrial 100 mm / 20,000-cell workflow summary",
        "",
        f"- overall status: `{result.status.value}`",
        f"- mode: `{result.config.mode.value}`",
        f"- layout: `{result.layout.name}`",
        f"- cells: `{result.layout.n_cells}`",
        f"- length: `{result.layout.total_length_m:.6g} m`",
        "",
        "## Linear stage",
        "",
        f"- status: `{result.linear.status.value}`",
        f"- S21 dB min/max: `{result.linear.scan.to_dict()['s21_db_min']:.4g}` / `{result.linear.scan.to_dict()['s21_db_max']:.4g}`",
        f"- stopbands detected: `{len(result.linear.stopbands)}`",
    ]

    if result.coarsening is not None:
        lines += [
            "",
            "## Coarsening stage",
            "",
            f"- status: `{result.coarsening.status.value}`",
            f"- selected pump layout: `{None if result.coarsening.selected_pump_layout is None else result.coarsening.selected_pump_layout.name}`",
            f"- selected cells: `{None if result.coarsening.selected_pump_layout is None else result.coarsening.selected_pump_layout.n_cells}`",
        ]

    if result.pump is not None:
        lines += [
            "",
            "## Pump stage",
            "",
            f"- status: `{result.pump.status.value}`",
        ]
        if result.pump.result is not None:
            p = result.pump.result.profile
            lines += [
                f"- converged: `{result.pump.result.converged}`",
                f"- max |I|/I*: `{p.max_pump_current_ratio:.4g}`",
                f"- pump output/input voltage gain: `{p.output_to_input_voltage_gain_db:.4g} dB`",
                f"- residual norm: `{result.pump.result.residual.norm:.4e}`",
            ]

    if result.gain is not None:
        lines += [
            "",
            "## Gain stage",
            "",
            f"- status: `{result.gain.status.value}`",
        ]
        if result.gain.sweep is not None:
            lines += [
                f"- points: `{result.gain.sweep.n_points}`",
                f"- converged: `{result.gain.sweep.n_converged}`",
                f"- max signal gain: `{float(jnp.nanmax(result.gain.sweep.signal_gain_db_array)):.4g} dB`",
            ]

    if result.artifact_paths:
        lines += [
            "",
            "## Artifacts",
            "",
        ]
        for key, path in result.artifact_paths.items():
            lines.append(f"- `{key}`: `{path}`")

    return "\n".join(lines)


__all__ = [
    "IndustrialStageStatus",
    "IndustrialRunMode",
    "IndustrialLayoutSpec",
    "build_industrial_layout",
    "IndustrialLinearStageConfig",
    "IndustrialCoarseningStageConfig",
    "IndustrialPumpStageConfig",
    "IndustrialGainStageConfig",
    "Industrial100mmWorkflowConfig",
    "IndustrialLinearStageResult",
    "IndustrialCoarseningStageResult",
    "IndustrialPumpStageResult",
    "IndustrialGainStageResult",
    "Industrial100mmWorkflowResult",
    "run_linear_stage",
    "select_layout_closest_n_cells",
    "run_coarsening_stage",
    "run_pump_stage",
    "run_gain_stage",
    "IndustrialPhaseMatchingReport",
    "compute_industrial_phase_matching_report",
    "export_industrial_artifacts",
    "run_industrial_100mm_workflow",
    "make_linear_only_100mm_config",
    "make_reduced_pump_100mm_config",
    "summarize_workflow_markdown",
]
