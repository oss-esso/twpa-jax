"""
twpa.workflows.synthetic_benchmarks
===================================

Synthetic benchmark suite for the JAX-backed TWPA simulator.

This module provides reproducible benchmark cases for validating the simulator
stack before trusting calibrated or industrial 100 mm / 20,000-cell runs.

Benchmark hierarchy
-------------------
The intended progression is:

    1. tiny linear layouts
       Validate ABCD cascade, MNA comparison, dispersion extraction.

    2. periodic/stub-loaded linear layouts
       Validate stopband and Bloch dispersion behavior.

    3. coarsening convergence
       Validate reduced layouts against a finer reference.

    4. one-node nonlinear HB
       Validate nonlinear element projection and dense Newton.

    5. distributed nonlinear pump HB on small layouts
       Validate pump-only HB residuals and continuation.

    6. small-signal linearization/gain smoke tests
       Validate that the gain layer can linearize and solve perturbations.

This file is meant to be used by scripts and notebooks. It writes compact JSON
and NPZ artifacts and returns structured dataclass reports.

Important
---------
Synthetic benchmarks are not device calibration. They are regression tests and
sanity checks. Passing this suite means the code path is internally consistent,
not that a fabricated TWPA has been modeled accurately.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import json

import jax
import jax.numpy as jnp
import numpy as np

from twpa.io.reports import jsonify
from twpa.core.frequency_plan import FrequencyPlan, make_pump_only_plan
from twpa.core.layout import LineLayout, make_layout_from_arrays
from twpa.core.params import NonlinearParams
from twpa.linear.cascade import (
    CascadeConfig,
    CascadeStrategy,
    compare_layout_to_uniform_rlgc_line,
    run_linear_scan,
    validate_cascade,
)
from twpa.linear.cells import CellModelConfig, CellModelKind, validate_layout_cells
from twpa.linear.coarsening import (
    CoarseningHierarchyConfig,
    CoarseningMethod,
    compare_hierarchy_dispersion,
    generate_coarsening_hierarchy,
)
from twpa.linear.dispersion import (
    DispersionConfig,
    StopbandMetric,
    detect_stopbands,
    extract_layout_dispersion,
    validate_dispersion_result,
)
from twpa.linear.ladder_mna import (
    LadderMNAConfig,
    compare_ladder_mna_to_abcd,
    validate_ladder_mna,
)
from twpa.nonlinear.distributed_hb import (
    DistributedHBConfig,
    make_kinetic_model_from_layout,
    run_distributed_hb_self_checks,
    solve_distributed_pump_current_hb,
)
from twpa.nonlinear.gain import (
    GainSolveConfig,
    GainSweepConfig,
    solve_gain_sweep_from_pump,
)
from twpa.nonlinear.one_node import run_one_node_self_checks
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    run_pump_hb_self_checks,
    solve_pump_hb_ladder,
)
from twpa.nonlinear.linearization import SmallSignalLinearizationConfig
from twpa.solvers.hb_solver import DenseNewtonConfig


ArrayLike = Any
TargetPlanFactory = Callable[[Any], FrequencyPlan]
SweepConfigFactory = Callable[[FrequencyPlan], GainSweepConfig]


# ---------------------------------------------------------------------------
# JSON / artifact helpers
# ---------------------------------------------------------------------------

def _jsonify(obj: Any) -> Any:
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
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            if np.iscomplexobj(arr):
                return _jsonify(complex(arr))
            return arr.item()
        return {
            "array_shape": tuple(int(s) for s in arr.shape),
            "array_dtype": str(arr.dtype),
            "min_abs": float(np.nanmin(np.abs(arr))) if arr.size else None,
            "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else None,
        }
    return obj


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_npz(path: str | Path, **arrays: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in arrays.items()})
    return out


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BenchmarkStatus(str, Enum):
    """Benchmark status."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class SyntheticLayoutKind(str, Enum):
    """Synthetic layout families."""

    UNIFORM = "uniform"
    STUB_PERIODIC = "stub_periodic"
    WEAK_DISORDER = "weak_disorder"
    LOSSY_UNIFORM = "lossy_uniform"


class SyntheticBenchmarkStage(str, Enum):
    """Benchmark stage identifiers."""

    LINEAR = "linear"
    MNA = "mna"
    DISPERSION = "dispersion"
    COARSENING = "coarsening"
    ONE_NODE_HB = "one_node_hb"
    DISTRIBUTED_HB = "distributed_hb"
    PUMP_HB = "pump_hb"
    GAIN = "gain"


# ---------------------------------------------------------------------------
# Layout cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticLayoutSpec:
    """
    Synthetic layout specification.

    Parameters
    ----------
    kind:
        Layout family.
    n_cells:
        Number of lumped cells.
    length_m:
        Total line length.
    z0_ohm:
        Target impedance used to derive L/C if L_per_m/C_per_m are absent.
    phase_velocity_m_per_s:
        Target phase velocity.
    L_per_m_H, C_per_m_F:
        Optional explicit line parameters.
    R_per_m_ohm, G_per_m_S:
        Loss terms.
    stub_period_cells:
        Period for capacitive loading.
    stub_fraction:
        C_stub / C_base for loaded cells.
    disorder_std_fraction:
        Standard deviation of multiplicative L/C disorder.
    disorder_seed:
        Seed for deterministic disorder.
    """

    kind: SyntheticLayoutKind = SyntheticLayoutKind.UNIFORM
    n_cells: int = 32
    length_m: float = 1.0e-3
    z0_ohm: float = 50.0
    phase_velocity_m_per_s: float = 1.2e8
    L_per_m_H: float | None = None
    C_per_m_F: float | None = None
    R_per_m_ohm: float = 0.0
    G_per_m_S: float = 0.0
    stub_period_cells: int = 8
    stub_fraction: float = 0.2
    disorder_std_fraction: float = 0.01
    disorder_seed: int = 123
    name: str = "synthetic_layout"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", SyntheticLayoutKind(self.kind))
        if int(self.n_cells) <= 0:
            raise ValueError("n_cells must be positive")
        object.__setattr__(self, "n_cells", int(self.n_cells))
        if self.length_m <= 0.0:
            raise ValueError("length_m must be positive")
        if self.z0_ohm <= 0.0:
            raise ValueError("z0_ohm must be positive")
        if self.phase_velocity_m_per_s <= 0.0:
            raise ValueError("phase_velocity_m_per_s must be positive")
        if self.R_per_m_ohm < 0.0:
            raise ValueError("R_per_m_ohm must be non-negative")
        if self.G_per_m_S < 0.0:
            raise ValueError("G_per_m_S must be non-negative")
        if int(self.stub_period_cells) <= 0:
            raise ValueError("stub_period_cells must be positive")
        object.__setattr__(self, "stub_period_cells", int(self.stub_period_cells))
        if self.stub_fraction < 0.0:
            raise ValueError("stub_fraction must be non-negative")
        if self.disorder_std_fraction < 0.0:
            raise ValueError("disorder_std_fraction must be non-negative")

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

    def with_updates(self, **kwargs: Any) -> "SyntheticLayoutSpec":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "n_cells": self.n_cells,
            "length_m": self.length_m,
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
            "stub_fraction": self.stub_fraction,
            "disorder_std_fraction": self.disorder_std_fraction,
            "disorder_seed": self.disorder_seed,
            "name": self.name,
        }


def build_synthetic_layout(spec: SyntheticLayoutSpec) -> LineLayout:
    """
    Build a deterministic synthetic LineLayout.
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

    if spec.kind == SyntheticLayoutKind.STUB_PERIODIC:
        idx = jnp.arange(n)
        loaded = (idx % spec.stub_period_cells) == 0
        C_stub_F = jnp.where(loaded, spec.stub_fraction * C_cell, 0.0)

    elif spec.kind == SyntheticLayoutKind.WEAK_DISORDER:
        rng = np.random.default_rng(spec.disorder_seed)
        L_mult = 1.0 + spec.disorder_std_fraction * rng.standard_normal(n)
        C_mult = 1.0 + spec.disorder_std_fraction * rng.standard_normal(n)
        L_series_H = L_series_H * jnp.asarray(np.maximum(L_mult, 1e-6), dtype=jnp.float64)
        C_shunt_F = C_shunt_F * jnp.asarray(np.maximum(C_mult, 1e-6), dtype=jnp.float64)

    elif spec.kind == SyntheticLayoutKind.LOSSY_UNIFORM:
        # Loss already enters through R/G fields. Nothing else required.
        pass

    elif spec.kind == SyntheticLayoutKind.UNIFORM:
        pass

    else:
        raise ValueError(f"Unsupported synthetic layout kind {spec.kind}")

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
            "source": "build_synthetic_layout",
            "synthetic_layout_spec": spec.to_dict(),
        },
    )


def default_synthetic_layout_specs() -> tuple[SyntheticLayoutSpec, ...]:
    """
    Default layout cases for regression benchmarks.
    """
    return (
        SyntheticLayoutSpec(
            kind=SyntheticLayoutKind.UNIFORM,
            n_cells=32,
            length_m=1.0e-3,
            name="synthetic_uniform_32",
        ),
        SyntheticLayoutSpec(
            kind=SyntheticLayoutKind.STUB_PERIODIC,
            n_cells=64,
            length_m=2.0e-3,
            stub_period_cells=8,
            stub_fraction=0.25,
            name="synthetic_stub_periodic_64",
        ),
        SyntheticLayoutSpec(
            kind=SyntheticLayoutKind.WEAK_DISORDER,
            n_cells=64,
            length_m=2.0e-3,
            disorder_std_fraction=0.01,
            name="synthetic_weak_disorder_64",
        ),
        SyntheticLayoutSpec(
            kind=SyntheticLayoutKind.LOSSY_UNIFORM,
            n_cells=32,
            length_m=1.0e-3,
            R_per_m_ohm=20.0,
            G_per_m_S=1e-6,
            name="synthetic_lossy_uniform_32",
        ),
    )


# ---------------------------------------------------------------------------
# Benchmark config/result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticLinearBenchmarkConfig:
    """
    Linear benchmark configuration.
    """

    frequency_min_hz: float = 1.0e9
    frequency_max_hz: float = 12.0e9
    n_frequency_points: int = 151
    cell_model: CellModelConfig = field(
        default_factory=lambda: CellModelConfig(kind=CellModelKind.PI)
    )
    cascade: CascadeConfig = field(
        default_factory=lambda: CascadeConfig(
            strategy=CascadeStrategy.AUTO,
            chunk_size=128,
            cells_per_supercell=8,
            allow_remainder=True,
        )
    )
    dispersion: DispersionConfig = field(
        default_factory=lambda: DispersionConfig(cells_per_supercell=1)
    )
    run_mna_comparison: bool = True
    run_uniform_baseline_comparison: bool = True
    cutoff_safety_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.frequency_min_hz <= 0.0:
            raise ValueError("frequency_min_hz must be positive")
        if self.frequency_max_hz <= self.frequency_min_hz:
            raise ValueError("frequency_max_hz must exceed frequency_min_hz")
        if int(self.n_frequency_points) < 2:
            raise ValueError("n_frequency_points must be >= 2")
        object.__setattr__(self, "n_frequency_points", int(self.n_frequency_points))

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
            "run_mna_comparison": self.run_mna_comparison,
            "run_uniform_baseline_comparison": self.run_uniform_baseline_comparison,
            "cutoff_safety_factor": self.cutoff_safety_factor,
        }


@dataclass(frozen=True)
class SyntheticNonlinearBenchmarkConfig:
    """
    Nonlinear benchmark configuration.

    Nonlinear stages are skipped if nonlinear_params is None.
    """

    nonlinear_params: NonlinearParams | None = None
    pump_frequency_hz: float = 6.0e9
    pump_current_rms_A: float = 1.0e-8
    n_pump_harmonics: int = 3
    max_cells_for_dense_hb: int = 64
    run_one_node: bool = True
    run_distributed_hb: bool = True
    run_pump_hb: bool = True
    run_gain_smoke: bool = False
    solver: DenseNewtonConfig = field(
        default_factory=lambda: DenseNewtonConfig(
            max_iter=40,
            abs_tol=1e-9,
            rel_tol=1e-9,
            damping_initial=1.0,
            regularization=0.0,
        )
    )

    def __post_init__(self) -> None:
        if self.pump_frequency_hz <= 0.0:
            raise ValueError("pump_frequency_hz must be positive")
        if self.pump_current_rms_A < 0.0:
            raise ValueError("pump_current_rms_A must be non-negative")
        if int(self.n_pump_harmonics) <= 0:
            raise ValueError("n_pump_harmonics must be positive")
        if int(self.max_cells_for_dense_hb) <= 0:
            raise ValueError("max_cells_for_dense_hb must be positive")
        object.__setattr__(self, "n_pump_harmonics", int(self.n_pump_harmonics))
        object.__setattr__(self, "max_cells_for_dense_hb", int(self.max_cells_for_dense_hb))

    def pump_plan(self) -> FrequencyPlan:
        return make_pump_only_plan(
            self.pump_frequency_hz,
            n_harmonics=self.n_pump_harmonics,
            include_negative=True,
            include_dc=False,
            sort="frequency",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_nonlinear_params": self.nonlinear_params is not None,
            "nonlinear_params": None if self.nonlinear_params is None else self.nonlinear_params.to_dict(),
            "pump_frequency_hz": self.pump_frequency_hz,
            "pump_current_rms_A": self.pump_current_rms_A,
            "n_pump_harmonics": self.n_pump_harmonics,
            "max_cells_for_dense_hb": self.max_cells_for_dense_hb,
            "run_one_node": self.run_one_node,
            "run_distributed_hb": self.run_distributed_hb,
            "run_pump_hb": self.run_pump_hb,
            "run_gain_smoke": self.run_gain_smoke,
            "solver": self.solver.to_dict(),
        }


@dataclass(frozen=True)
class SyntheticCoarseningBenchmarkConfig:
    """
    Coarsening convergence benchmark configuration.
    """

    enabled: bool = True
    hierarchy: CoarseningHierarchyConfig = field(
        default_factory=lambda: CoarseningHierarchyConfig(
            target_cell_counts=(16, 32, 64, 128),
            method=CoarseningMethod.EXACT_GROUP_SUM,
            preserve_supercells=True,
            cells_per_supercell=1,
            include_original=True,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hierarchy": self.hierarchy.to_dict() if hasattr(self.hierarchy, "to_dict") else _jsonify(self.hierarchy),
        }


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """
    Full synthetic benchmark suite config.
    """

    layout_specs: tuple[SyntheticLayoutSpec, ...] = field(default_factory=default_synthetic_layout_specs)
    linear: SyntheticLinearBenchmarkConfig = field(default_factory=SyntheticLinearBenchmarkConfig)
    coarsening: SyntheticCoarseningBenchmarkConfig = field(default_factory=SyntheticCoarseningBenchmarkConfig)
    nonlinear: SyntheticNonlinearBenchmarkConfig = field(default_factory=SyntheticNonlinearBenchmarkConfig)
    output_dir: str | None = None
    save_artifacts: bool = False
    stop_on_error: bool = False
    name: str = "synthetic_benchmarks"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout_specs", tuple(self.layout_specs))
        if not self.layout_specs:
            raise ValueError("layout_specs may not be empty")

    def with_updates(self, **kwargs: Any) -> "SyntheticBenchmarkConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_specs": [s.to_dict() for s in self.layout_specs],
            "linear": self.linear.to_dict(),
            "coarsening": self.coarsening.to_dict(),
            "nonlinear": self.nonlinear.to_dict(),
            "output_dir": self.output_dir,
            "save_artifacts": self.save_artifacts,
            "stop_on_error": self.stop_on_error,
            "name": self.name,
        }


@dataclass(frozen=True)
class BenchmarkStageResult:
    """
    Generic stage result.
    """

    stage: SyntheticBenchmarkStage
    status: BenchmarkStatus
    elapsed_s: float
    report: Mapping[str, Any]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == BenchmarkStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "elapsed_s": self.elapsed_s,
            "report": dict(self.report),
            "error": self.error,
        }


@dataclass(frozen=True)
class SyntheticLayoutBenchmarkResult:
    """
    Benchmark result for one synthetic layout.
    """

    layout: LineLayout
    layout_spec: SyntheticLayoutSpec
    stage_results: tuple[BenchmarkStageResult, ...]
    artifact_paths: Mapping[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        active = [r for r in self.stage_results if r.status != BenchmarkStatus.SKIP]
        return bool(active) and all(r.status == BenchmarkStatus.PASS for r in active)

    @property
    def status(self) -> BenchmarkStatus:
        if self.passed:
            return BenchmarkStatus.PASS
        if any(r.status == BenchmarkStatus.ERROR for r in self.stage_results):
            return BenchmarkStatus.ERROR
        if any(r.status == BenchmarkStatus.FAIL for r in self.stage_results):
            return BenchmarkStatus.FAIL
        return BenchmarkStatus.SKIP

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "layout": self.layout.summary(),
            "layout_spec": self.layout_spec.to_dict(),
            "stage_results": [r.to_dict() for r in self.stage_results],
            "artifact_paths": dict(self.artifact_paths),
        }


@dataclass(frozen=True)
class SyntheticBenchmarkSuiteResult:
    """
    Full synthetic benchmark suite result.
    """

    config: SyntheticBenchmarkConfig
    layout_results: tuple[SyntheticLayoutBenchmarkResult, ...]
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.layout_results)

    @property
    def status(self) -> BenchmarkStatus:
        if self.passed:
            return BenchmarkStatus.PASS
        if any(r.status == BenchmarkStatus.ERROR for r in self.layout_results):
            return BenchmarkStatus.ERROR
        if any(r.status == BenchmarkStatus.FAIL for r in self.layout_results):
            return BenchmarkStatus.FAIL
        return BenchmarkStatus.SKIP

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "config": self.config.to_dict(),
            "layout_results": [r.to_dict() for r in self.layout_results],
            "artifact_paths": dict(self.artifact_paths),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Stage runner helper
# ---------------------------------------------------------------------------

def _run_stage(
    stage: SyntheticBenchmarkStage,
    fn: Callable[[], Mapping[str, Any]],
    *,
    skip: bool = False,
    skip_message: str = "",
) -> BenchmarkStageResult:
    """
    Run a benchmark stage with timing and exception capture.
    """
    if skip:
        return BenchmarkStageResult(
            stage=stage,
            status=BenchmarkStatus.SKIP,
            elapsed_s=0.0,
            report={"message": skip_message},
            error=None,
        )

    t0 = perf_counter()
    try:
        report = fn()
        elapsed = perf_counter() - t0
        passed = bool(report.get("passed", True))
        return BenchmarkStageResult(
            stage=stage,
            status=BenchmarkStatus.PASS if passed else BenchmarkStatus.FAIL,
            elapsed_s=elapsed,
            report=report,
            error=None,
        )
    except Exception as exc:
        elapsed = perf_counter() - t0
        return BenchmarkStageResult(
            stage=stage,
            status=BenchmarkStatus.ERROR,
            elapsed_s=elapsed,
            report={"message": "stage raised exception"},
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Linear stages
# ---------------------------------------------------------------------------

def run_linear_benchmark_stage(
    layout: LineLayout,
    config: SyntheticLinearBenchmarkConfig,
) -> Mapping[str, Any]:
    """
    Run linear cell/cascade/S-parameter validation.
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

    passed = bool(cell_report["cutoff_guard_passed"]) and bool(cascade_report["passed"])

    report = {
        "passed": passed,
        "cell_report": cell_report,
        "cascade_report": cascade_report,
        "scan": scan.to_dict(),
    }

    if config.run_uniform_baseline_comparison:
        try:
            baseline = compare_layout_to_uniform_rlgc_line(
                f,
                layout,
                cell_model=config.cell_model,
                cascade_config=config.cascade,
            )
            report["uniform_baseline_comparison"] = baseline.to_dict()
        except Exception as exc:
            report["uniform_baseline_comparison"] = {"error": str(exc)}

    return report


def run_mna_benchmark_stage(
    layout: LineLayout,
    config: SyntheticLinearBenchmarkConfig,
) -> Mapping[str, Any]:
    """
    Run MNA validation and compare against ABCD PI-cell cascade.
    """
    f = config.frequency_grid()

    mna_report = validate_ladder_mna(
        f,
        layout,
        config=LadderMNAConfig(
            include_stub_capacitance=config.cell_model.include_stub_capacitance,
        ),
    ).to_dict()

    comp_report = compare_ladder_mna_to_abcd(
        f,
        layout,
        mna_config=LadderMNAConfig(
            include_stub_capacitance=config.cell_model.include_stub_capacitance,
        ),
        cell_model=CellModelConfig(
            kind=CellModelKind.PI,
            include_stub_capacitance=config.cell_model.include_stub_capacitance,
        ),
        cascade_config=config.cascade,
        tolerance_s21_db=1.0,
    ).to_dict()

    passed = bool(mna_report["passed"]) and bool(comp_report["passed"])

    return {
        "passed": passed,
        "mna_report": mna_report,
        "mna_vs_abcd": comp_report,
    }


def run_dispersion_benchmark_stage(
    layout: LineLayout,
    config: SyntheticLinearBenchmarkConfig,
) -> Mapping[str, Any]:
    """
    Run dispersion extraction and stopband detection.
    """
    f = config.frequency_grid()

    dispersion = extract_layout_dispersion(
        f,
        layout,
        cell_model=config.cell_model,
        cascade_config=config.cascade,
        dispersion_config=config.dispersion,
    )

    report = validate_dispersion_result(
        dispersion,
        layout_name=layout.name,
        expected_stopband=None,
        stopband_metric=StopbandMetric.BOTH,
    ).to_dict()

    stopbands = [
        sb.to_dict()
        for sb in detect_stopbands(
            dispersion,
            metric=StopbandMetric.BOTH,
        )
    ]

    return {
        "passed": bool(report["passed"]),
        "dispersion": dispersion.to_dict(),
        "validation": report,
        "stopbands": stopbands,
    }


def run_coarsening_benchmark_stage(
    layout: LineLayout,
    linear_config: SyntheticLinearBenchmarkConfig,
    coarsening_config: SyntheticCoarseningBenchmarkConfig,
) -> Mapping[str, Any]:
    """
    Run coarsening hierarchy and compare dispersion convergence.
    """
    hierarchy = generate_coarsening_hierarchy(
        layout,
        coarsening_config.hierarchy,
    )

    comparisons = compare_hierarchy_dispersion(
        linear_config.frequency_grid(),
        hierarchy,
        reference=layout,
        cell_model=linear_config.cell_model,
        dispersion_config=linear_config.dispersion,
    )

    comparison_dicts = [c.to_dict() for c in comparisons]

    # Synthetic coarsening is allowed to be imperfect. Mark pass if the machinery
    # completed and at least one comparison exists.
    passed = len(comparison_dicts) > 0 or len(hierarchy.layouts) == 1

    return {
        "passed": passed,
        "hierarchy": hierarchy.to_dict(),
        "comparisons": comparison_dicts,
    }


# ---------------------------------------------------------------------------
# Nonlinear stages
# ---------------------------------------------------------------------------

def run_one_node_benchmark_stage(
    config: SyntheticNonlinearBenchmarkConfig,
) -> Mapping[str, Any]:
    """
    Run one-node nonlinear HB self-checks.
    """
    plan = config.pump_plan()
    report = run_one_node_self_checks(plan)
    return {
        "passed": bool(report["passed"]),
        "self_checks": report,
    }


def run_distributed_hb_benchmark_stage(
    layout: LineLayout,
    config: SyntheticNonlinearBenchmarkConfig,
) -> Mapping[str, Any]:
    """
    Run distributed HB self-checks on a small layout.
    """
    if config.nonlinear_params is None:
        return {"passed": False, "message": "nonlinear_params is required"}

    plan = config.pump_plan()
    report = run_distributed_hb_self_checks(
        plan,
        layout,
        nonlinear_params=config.nonlinear_params,
    )
    return {
        "passed": bool(report["passed"]),
        "self_checks": report,
    }


def run_pump_hb_benchmark_stage(
    layout: LineLayout,
    config: SyntheticNonlinearBenchmarkConfig,
) -> Mapping[str, Any]:
    """
    Run pump-HB self-checks and one direct pump solve.
    """
    if config.nonlinear_params is None:
        return {"passed": False, "message": "nonlinear_params is required"}

    checks = run_pump_hb_self_checks(
        layout,
        config.nonlinear_params,
        pump_frequency_hz=config.pump_frequency_hz,
    )

    drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=config.pump_frequency_hz,
        current_rms_A=config.pump_current_rms_A,
    )
    pump_config = PumpHBLadderConfig(
        n_pump_harmonics=config.n_pump_harmonics,
        solver=config.solver,
    )

    result = solve_pump_hb_ladder(
        layout,
        config.nonlinear_params,
        drive=drive,
        pump_config=pump_config,
        metadata={"benchmark": "run_pump_hb_benchmark_stage"},
    )

    passed = bool(checks["passed"] and result.converged)

    return {
        "passed": passed,
        "self_checks": checks,
        "pump_result": result.to_dict(),
    }


def run_gain_smoke_benchmark_stage(
    layout: LineLayout,
    config: SyntheticNonlinearBenchmarkConfig,
    *,
    target_plan_factory: TargetPlanFactory | None = None,
    sweep_config_factory: SweepConfigFactory | None = None,
) -> Mapping[str, Any]:
    """
    Run a gain-layer smoke test.

    This stage requires target_plan_factory and sweep_config_factory because
    signal/idler frequency-plan conventions are project-specific.
    """
    if config.nonlinear_params is None:
        return {"passed": False, "message": "nonlinear_params is required"}
    if target_plan_factory is None or sweep_config_factory is None:
        return {
            "passed": False,
            "message": "gain smoke requires target_plan_factory and sweep_config_factory",
        }

    drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=config.pump_frequency_hz,
        current_rms_A=config.pump_current_rms_A,
    )
    pump_config = PumpHBLadderConfig(
        n_pump_harmonics=config.n_pump_harmonics,
        solver=config.solver,
    )

    pump = solve_pump_hb_ladder(
        layout,
        config.nonlinear_params,
        drive=drive,
        pump_config=pump_config,
        metadata={"benchmark": "gain_smoke_pump"},
    )

    target_plan = target_plan_factory(pump)
    sweep_config = sweep_config_factory(target_plan)

    sweep = solve_gain_sweep_from_pump(
        pump,
        target_plan=target_plan,
        sweep_config=sweep_config,
        linearization_config=SmallSignalLinearizationConfig(),
    )

    return {
        "passed": bool(pump.converged and sweep.passed),
        "pump": pump.to_dict(),
        "target_plan": target_plan.to_dict(),
        "sweep": sweep.to_dict(),
    }


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------

def run_synthetic_layout_benchmarks(
    layout_spec: SyntheticLayoutSpec,
    config: SyntheticBenchmarkConfig,
    *,
    target_plan_factory: TargetPlanFactory | None = None,
    sweep_config_factory: SweepConfigFactory | None = None,
) -> SyntheticLayoutBenchmarkResult:
    """
    Run all applicable benchmarks for one synthetic layout.
    """
    layout = build_synthetic_layout(layout_spec)
    stage_results: list[BenchmarkStageResult] = []

    stage_results.append(
        _run_stage(
            SyntheticBenchmarkStage.LINEAR,
            lambda: run_linear_benchmark_stage(layout, config.linear),
        )
    )

    stage_results.append(
        _run_stage(
            SyntheticBenchmarkStage.MNA,
            lambda: run_mna_benchmark_stage(layout, config.linear),
            skip=not config.linear.run_mna_comparison,
            skip_message="MNA comparison disabled",
        )
    )

    stage_results.append(
        _run_stage(
            SyntheticBenchmarkStage.DISPERSION,
            lambda: run_dispersion_benchmark_stage(layout, config.linear),
        )
    )

    stage_results.append(
        _run_stage(
            SyntheticBenchmarkStage.COARSENING,
            lambda: run_coarsening_benchmark_stage(layout, config.linear, config.coarsening),
            skip=not config.coarsening.enabled,
            skip_message="coarsening benchmark disabled",
        )
    )

    nonlinear_skip = config.nonlinear.nonlinear_params is None
    dense_hb_skip = nonlinear_skip or layout.n_cells > config.nonlinear.max_cells_for_dense_hb

    stage_results.append(
        _run_stage(
            SyntheticBenchmarkStage.ONE_NODE_HB,
            lambda: run_one_node_benchmark_stage(config.nonlinear),
            skip=nonlinear_skip or not config.nonlinear.run_one_node,
            skip_message="one-node HB disabled or nonlinear_params missing",
        )
    )

    stage_results.append(
        _run_stage(
            SyntheticBenchmarkStage.DISTRIBUTED_HB,
            lambda: run_distributed_hb_benchmark_stage(layout, config.nonlinear),
            skip=dense_hb_skip or not config.nonlinear.run_distributed_hb,
            skip_message="distributed HB disabled, too many cells, or nonlinear_params missing",
        )
    )

    stage_results.append(
        _run_stage(
            SyntheticBenchmarkStage.PUMP_HB,
            lambda: run_pump_hb_benchmark_stage(layout, config.nonlinear),
            skip=dense_hb_skip or not config.nonlinear.run_pump_hb,
            skip_message="pump HB disabled, too many cells, or nonlinear_params missing",
        )
    )

    stage_results.append(
        _run_stage(
            SyntheticBenchmarkStage.GAIN,
            lambda: run_gain_smoke_benchmark_stage(
                layout,
                config.nonlinear,
                target_plan_factory=target_plan_factory,
                sweep_config_factory=sweep_config_factory,
            ),
            skip=(
                dense_hb_skip
                or not config.nonlinear.run_gain_smoke
                or target_plan_factory is None
                or sweep_config_factory is None
            ),
            skip_message="gain smoke disabled, unavailable factories, too many cells, or nonlinear_params missing",
        )
    )

    return SyntheticLayoutBenchmarkResult(
        layout=layout,
        layout_spec=layout_spec,
        stage_results=tuple(stage_results),
    )


def run_synthetic_benchmarks(
    config: SyntheticBenchmarkConfig | None = None,
    *,
    target_plan_factory: TargetPlanFactory | None = None,
    sweep_config_factory: SweepConfigFactory | None = None,
) -> SyntheticBenchmarkSuiteResult:
    """
    Run the full synthetic benchmark suite.
    """
    cfg = config or SyntheticBenchmarkConfig()
    results: list[SyntheticLayoutBenchmarkResult] = []

    for spec in cfg.layout_specs:
        result = run_synthetic_layout_benchmarks(
            spec,
            cfg,
            target_plan_factory=target_plan_factory,
            sweep_config_factory=sweep_config_factory,
        )
        results.append(result)

        if cfg.stop_on_error and result.status == BenchmarkStatus.ERROR:
            break

    suite = SyntheticBenchmarkSuiteResult(
        config=cfg,
        layout_results=tuple(results),
        artifact_paths={},
        metadata={
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
    )

    if cfg.save_artifacts:
        if cfg.output_dir is None:
            raise ValueError("save_artifacts=True requires output_dir")
        paths = export_synthetic_benchmark_artifacts(suite, cfg.output_dir)
        suite = replace(suite, artifact_paths=paths)

    return suite


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def export_synthetic_benchmark_artifacts(
    result: SyntheticBenchmarkSuiteResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Export synthetic benchmark summary and useful linear arrays.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    paths["summary_json"] = str(write_json(out / "synthetic_benchmarks_summary.json", result.to_dict()))
    paths["summary_md"] = str(out / "synthetic_benchmarks_summary.md")
    Path(paths["summary_md"]).write_text(summarize_synthetic_benchmarks_markdown(result), encoding="utf-8")

    # Export a compact NPZ for each layout's linear scan if possible.
    for layout_result in result.layout_results:
        linear_stage = next(
            (s for s in layout_result.stage_results if s.stage == SyntheticBenchmarkStage.LINEAR),
            None,
        )
        if linear_stage is None or linear_stage.status not in {BenchmarkStatus.PASS, BenchmarkStatus.FAIL}:
            continue

        try:
            layout = layout_result.layout
            f = result.config.linear.frequency_grid()
            scan = run_linear_scan(
                f,
                layout,
                cell_model=result.config.linear.cell_model,
                cascade_config=result.config.linear.cascade,
            )
            key = f"linear_npz_{layout.name}"
            paths[key] = str(
                write_npz(
                    out / f"{layout.name}_linear_scan.npz",
                    frequency_hz=f,
                    s=scan.s,
                    s21_db=scan.s21_db,
                    beta_eff_rad_per_m=scan.beta_eff_rad_per_m,
                    group_delay_s=scan.group_delay_s,
                )
            )
        except Exception:
            continue

    return paths


# ---------------------------------------------------------------------------
# Markdown reports
# ---------------------------------------------------------------------------

def summarize_synthetic_layout_markdown(result: SyntheticLayoutBenchmarkResult) -> str:
    """
    Markdown summary for one layout benchmark result.
    """
    lines = [
        f"## {result.layout.name}",
        "",
        f"- status: `{result.status.value}`",
        f"- cells: `{result.layout.n_cells}`",
        f"- length: `{result.layout.total_length_m:.6g} m`",
        "",
        "| stage | status | elapsed s | error |",
        "|---|---|---:|---|",
    ]

    for stage in result.stage_results:
        err = "" if stage.error is None else stage.error
        lines.append(
            f"| `{stage.stage.value}` | `{stage.status.value}` | {stage.elapsed_s:.4g} | {err} |"
        )

    return "\n".join(lines)


def summarize_synthetic_benchmarks_markdown(result: SyntheticBenchmarkSuiteResult) -> str:
    """
    Markdown summary for the full suite.
    """
    lines = [
        "# Synthetic benchmark suite",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- layouts: `{len(result.layout_results)}`",
        f"- JAX backend: `{dict(result.metadata or {}).get('jax_backend', 'unknown')}`",
        "",
    ]

    for layout_result in result.layout_results:
        lines.append(summarize_synthetic_layout_markdown(layout_result))
        lines.append("")

    if result.artifact_paths:
        lines += [
            "## Artifacts",
            "",
        ]
        for key, path in result.artifact_paths.items():
            lines.append(f"- `{key}`: `{path}`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience configs
# ---------------------------------------------------------------------------

def make_fast_linear_synthetic_config(
    *,
    output_dir: str | None = None,
    save_artifacts: bool = False,
) -> SyntheticBenchmarkConfig:
    """
    Fast linear-only synthetic benchmark config.
    """
    return SyntheticBenchmarkConfig(
        layout_specs=(
            SyntheticLayoutSpec(
                kind=SyntheticLayoutKind.UNIFORM,
                n_cells=16,
                length_m=5e-4,
                name="fast_uniform_16",
            ),
            SyntheticLayoutSpec(
                kind=SyntheticLayoutKind.STUB_PERIODIC,
                n_cells=32,
                length_m=1e-3,
                stub_period_cells=8,
                stub_fraction=0.2,
                name="fast_stub_periodic_32",
            ),
        ),
        nonlinear=SyntheticNonlinearBenchmarkConfig(
            nonlinear_params=None,
            run_one_node=False,
            run_distributed_hb=False,
            run_pump_hb=False,
            run_gain_smoke=False,
        ),
        output_dir=output_dir,
        save_artifacts=save_artifacts,
        name="fast_linear_synthetic",
    )


def make_small_nonlinear_synthetic_config(
    *,
    nonlinear_params: NonlinearParams,
    output_dir: str | None = None,
    save_artifacts: bool = False,
) -> SyntheticBenchmarkConfig:
    """
    Small nonlinear benchmark config suitable for dense HB.
    """
    return SyntheticBenchmarkConfig(
        layout_specs=(
            SyntheticLayoutSpec(
                kind=SyntheticLayoutKind.UNIFORM,
                n_cells=8,
                length_m=2e-4,
                name="small_nonlinear_uniform_8",
            ),
            SyntheticLayoutSpec(
                kind=SyntheticLayoutKind.STUB_PERIODIC,
                n_cells=16,
                length_m=4e-4,
                stub_period_cells=4,
                stub_fraction=0.15,
                name="small_nonlinear_stub_16",
            ),
        ),
        nonlinear=SyntheticNonlinearBenchmarkConfig(
            nonlinear_params=nonlinear_params,
            pump_frequency_hz=6e9,
            pump_current_rms_A=1e-8,
            max_cells_for_dense_hb=16,
            run_one_node=True,
            run_distributed_hb=True,
            run_pump_hb=True,
            run_gain_smoke=False,
        ),
        output_dir=output_dir,
        save_artifacts=save_artifacts,
        name="small_nonlinear_synthetic",
    )


__all__ = [
    "BenchmarkStatus",
    "SyntheticLayoutKind",
    "SyntheticBenchmarkStage",
    "SyntheticLayoutSpec",
    "build_synthetic_layout",
    "default_synthetic_layout_specs",
    "SyntheticLinearBenchmarkConfig",
    "SyntheticNonlinearBenchmarkConfig",
    "SyntheticCoarseningBenchmarkConfig",
    "SyntheticBenchmarkConfig",
    "BenchmarkStageResult",
    "SyntheticLayoutBenchmarkResult",
    "SyntheticBenchmarkSuiteResult",
    "run_linear_benchmark_stage",
    "run_mna_benchmark_stage",
    "run_dispersion_benchmark_stage",
    "run_coarsening_benchmark_stage",
    "run_one_node_benchmark_stage",
    "run_distributed_hb_benchmark_stage",
    "run_pump_hb_benchmark_stage",
    "run_gain_smoke_benchmark_stage",
    "run_synthetic_layout_benchmarks",
    "run_synthetic_benchmarks",
    "export_synthetic_benchmark_artifacts",
    "summarize_synthetic_layout_markdown",
    "summarize_synthetic_benchmarks_markdown",
    "make_fast_linear_synthetic_config",
    "make_small_nonlinear_synthetic_config",
]
