"""
twpa.nonlinear.gain
===================

Small-signal gain and conversion utilities for pumped TWPA simulations.

This module sits above:

    twpa.nonlinear.pump_hb_ladder
    twpa.nonlinear.linearization

and provides production-facing gain calculations:

    pump HB solution
        -> target small-signal frequency plan
        -> distributed HB linearization
        -> signal current injection
        -> voltage gain / idler conversion
        -> sweep/report objects

Important scope
---------------
This is a dense/reference gain layer. It is meant to define correct APIs,
validation quantities, and convergence checks before replacing the internal
linear solve with industrial block-banded or matrix-free backends.

The user-facing workflow is:

    pump_result = solve_pump_hb_ladder(...)
    lin = build_gain_linearization_from_pump(...)
    gain_point = solve_gain_point(...)
    sweep = solve_gain_sweep(...)

Frequency-plan responsibility
-----------------------------
This module does not force a single frequency-plan convention. The caller may
pass any FrequencyPlan containing the labels needed by the gain calculation,
typically:

    pump
    signal
    idler

The actual generation of such plans can later be centralized in
twpa.core.frequency_plan. Here we only require that labels exist in the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp

from twpa.core.frequency_plan import FrequencyPlan
from twpa.core.harmonics import coefficient_power_summary
from twpa.core.layout import LineLayout
from twpa.nonlinear.distributed_hb import DistributedHBSolveResult
from twpa.nonlinear.linearization import (
    DistributedHBLinearization,
    SmallSignalGainPoint,
    SmallSignalLinearizationConfig,
    SmallSignalSolveResult,
    SmallSignalSource,
    build_linearization_from_pump_result,
    compute_gain_point_from_solution,
    solve_linearized_small_signal,
    solve_signal_current_gain,
)
from twpa.nonlinear.pump_hb_ladder import PumpHBLadderResult


ArrayLike = Any


# ---------------------------------------------------------------------------
# Enums / scalar helpers
# ---------------------------------------------------------------------------

class GainQuantity(str, Enum):
    """Supported reported gain quantities."""

    VOLTAGE_GAIN = "voltage_gain"
    POWER_GAIN_MATCHED = "power_gain_matched"
    IDLER_CONVERSION = "idler_conversion"


class GainStatus(str, Enum):
    """Status of one gain calculation."""

    PASS = "pass"
    FAIL = "fail"


class GainInputKind(str, Enum):
    """Supported small-signal input conventions."""

    CURRENT_RMS = "current_rms"


def complex_abs_db(x: ArrayLike, *, floor: float = 1e-300) -> jax.Array:
    """
    20 log10 |x| for voltage/current-wave-like ratios.
    """
    return 20.0 * jnp.log10(jnp.maximum(jnp.abs(jnp.asarray(x)), floor))


def power_abs_db(x: ArrayLike, *, floor: float = 1e-300) -> jax.Array:
    """
    10 log10 |x| for power ratios.
    """
    return 10.0 * jnp.log10(jnp.maximum(jnp.abs(jnp.asarray(x)), floor))


def matched_power_gain_from_voltage_gain(
    voltage_gain: ArrayLike,
    *,
    input_impedance_ohm: float = 50.0,
    output_impedance_ohm: float = 50.0,
) -> jax.Array:
    """
    Convert voltage gain into a matched-load power-gain estimate.

    If input and output impedances are equal, power gain is |Vout/Vin|^2.

    More generally:

        Gp ≈ |Vout/Vin|^2 * Rin / Rout

    This is a diagnostic estimate, not a full microwave power-wave
    de-embedding calculation.
    """
    if input_impedance_ohm <= 0.0:
        raise ValueError("input_impedance_ohm must be positive")
    if output_impedance_ohm <= 0.0:
        raise ValueError("output_impedance_ohm must be positive")
    g = jnp.asarray(voltage_gain)
    return jnp.abs(g) ** 2 * input_impedance_ohm / output_impedance_ohm


def matched_power_gain_db_from_voltage_gain(
    voltage_gain: ArrayLike,
    *,
    input_impedance_ohm: float = 50.0,
    output_impedance_ohm: float = 50.0,
) -> float:
    """
    Matched-load power gain in dB from voltage gain.
    """
    gp = matched_power_gain_from_voltage_gain(
        voltage_gain,
        input_impedance_ohm=input_impedance_ohm,
        output_impedance_ohm=output_impedance_ohm,
    )
    return float(10.0 * jnp.log10(jnp.maximum(gp, 1e-300)))


def idler_frequency_dp4wm(
    *,
    pump_frequency_hz: ArrayLike,
    signal_frequency_hz: ArrayLike,
) -> jax.Array:
    """
    Degenerate-pump four-wave-mixing idler frequency:

        f_i = 2 f_p - f_s
    """
    return 2.0 * jnp.asarray(pump_frequency_hz, dtype=jnp.float64) - jnp.asarray(
        signal_frequency_hz,
        dtype=jnp.float64,
    )


def signal_frequency_from_detuning(
    *,
    pump_frequency_hz: ArrayLike,
    detuning_hz: ArrayLike,
    side: Literal["upper", "lower"] = "lower",
) -> jax.Array:
    """
    Build a signal frequency from pump frequency and detuning.

    side="lower":
        f_s = f_p - detuning
        f_i = f_p + detuning

    side="upper":
        f_s = f_p + detuning
        f_i = f_p - detuning
    """
    fp = jnp.asarray(pump_frequency_hz, dtype=jnp.float64)
    df = jnp.asarray(detuning_hz, dtype=jnp.float64)
    if side == "lower":
        return fp - df
    if side == "upper":
        return fp + df
    raise ValueError("side must be 'upper' or 'lower'")


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GainSolveConfig:
    """
    Configuration for one small-signal gain solve.

    Parameters
    ----------
    signal_label:
        Label of the input signal tone in the target FrequencyPlan.
    idler_label:
        Optional label of the idler/conversion tone in the target FrequencyPlan.
    input_node:
        Node where the small-signal current is injected.
    output_node:
        Node where output voltage is measured. If None, uses the final layout
        node.
    signal_current_rms_A:
        Small-signal injected RMS current phasor.
    set_conjugate:
        Whether to populate the negative-frequency conjugate coefficient.
    input_impedance_ohm, output_impedance_ohm:
        Used for matched-power-gain diagnostics.
    """

    signal_label: str = "signal"
    idler_label: str | None = "idler"
    input_node: int = 0
    output_node: int | None = None
    signal_current_rms_A: complex = 1e-12 + 0j
    set_conjugate: bool = True
    input_impedance_ohm: float = 50.0
    output_impedance_ohm: float = 50.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_node", int(self.input_node))
        if self.input_node < 0:
            raise ValueError("input_node must be non-negative")
        if self.output_node is not None:
            object.__setattr__(self, "output_node", int(self.output_node))
            if self.output_node < 0:
                raise ValueError("output_node must be non-negative")
        if self.input_impedance_ohm <= 0.0:
            raise ValueError("input_impedance_ohm must be positive")
        if self.output_impedance_ohm <= 0.0:
            raise ValueError("output_impedance_ohm must be positive")

    def with_updates(self, **kwargs: Any) -> "GainSolveConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_label": self.signal_label,
            "idler_label": self.idler_label,
            "input_node": self.input_node,
            "output_node": self.output_node,
            "signal_current_rms_A": {
                "real": float(jnp.real(self.signal_current_rms_A)),
                "imag": float(jnp.imag(self.signal_current_rms_A)),
                "abs": float(abs(self.signal_current_rms_A)),
            },
            "set_conjugate": self.set_conjugate,
            "input_impedance_ohm": self.input_impedance_ohm,
            "output_impedance_ohm": self.output_impedance_ohm,
        }


@dataclass(frozen=True)
class GainSweepConfig:
    """
    Configuration for a labelled gain sweep.

    Parameters
    ----------
    points:
        Sequence of GainSolveConfig objects. Each point can use a different
        signal/idler label pair if the target plan contains them.
    require_all_converged:
        Whether the sweep validation fails if any point fails.
    min_gain_db:
        Optional lower bound for signal gain in dB.
    max_gain_db:
        Optional upper bound for signal gain in dB.
    """

    points: tuple[GainSolveConfig, ...]
    require_all_converged: bool = True
    min_gain_db: float | None = None
    max_gain_db: float | None = None
    name: str = "gain_sweep"

    def __post_init__(self) -> None:
        if len(self.points) == 0:
            raise ValueError("GainSweepConfig.points may not be empty")
        object.__setattr__(self, "points", tuple(self.points))

    @classmethod
    def from_signal_labels(
        cls,
        signal_labels: Sequence[str],
        *,
        idler_labels: Sequence[str | None] | None = None,
        input_node: int = 0,
        output_node: int | None = None,
        signal_current_rms_A: complex = 1e-12 + 0j,
        set_conjugate: bool = True,
        input_impedance_ohm: float = 50.0,
        output_impedance_ohm: float = 50.0,
        name: str = "gain_sweep",
    ) -> "GainSweepConfig":
        """
        Build a sweep from signal labels and optional idler labels.
        """
        if idler_labels is None:
            idler_labels = [None] * len(signal_labels)
        if len(idler_labels) != len(signal_labels):
            raise ValueError("idler_labels must have same length as signal_labels")

        points = tuple(
            GainSolveConfig(
                signal_label=sig,
                idler_label=idl,
                input_node=input_node,
                output_node=output_node,
                signal_current_rms_A=signal_current_rms_A,
                set_conjugate=set_conjugate,
                input_impedance_ohm=input_impedance_ohm,
                output_impedance_ohm=output_impedance_ohm,
            )
            for sig, idl in zip(signal_labels, idler_labels)
        )

        return cls(points=points, name=name)

    def with_updates(self, **kwargs: Any) -> "GainSweepConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "require_all_converged": self.require_all_converged,
            "min_gain_db": self.min_gain_db,
            "max_gain_db": self.max_gain_db,
            "name": self.name,
        }


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GainPointResult:
    """
    Full result for one gain calculation.

    Attributes
    ----------
    solve:
        Small-signal linear solve result.
    gain:
        Basic gain point from linearization.py.
    config:
        Gain-solve configuration.
    matched_power_gain:
        Matched-power-gain estimate derived from voltage gain.
    matched_power_gain_db:
        dB version of matched_power_gain.
    status:
        PASS/FAIL status based on linear solve success.
    metadata:
        Extra metadata.
    """

    solve: SmallSignalSolveResult
    gain: SmallSignalGainPoint
    config: GainSolveConfig
    matched_power_gain: float
    matched_power_gain_db: float
    status: GainStatus
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.status == GainStatus.PASS

    @property
    def signal_gain_db(self) -> float:
        return self.gain.signal_gain_db

    @property
    def signal_gain_complex(self) -> complex:
        return self.gain.signal_gain_complex

    @property
    def idler_conversion_db(self) -> float | None:
        return self.gain.idler_conversion_db

    @property
    def idler_conversion_complex(self) -> complex | None:
        return self.gain.idler_conversion_complex

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "converged": self.converged,
            "config": self.config.to_dict(),
            "gain": self.gain.to_dict(),
            "matched_power_gain": self.matched_power_gain,
            "matched_power_gain_db": self.matched_power_gain_db,
            "solve": self.solve.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class GainSweepResult:
    """
    Result of a labelled gain sweep.
    """

    points: tuple[GainPointResult, ...]
    config: GainSweepConfig
    linearization: DistributedHBLinearization
    metadata: Mapping[str, Any] | None = None

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_converged(self) -> int:
        return sum(1 for p in self.points if p.converged)

    @property
    def all_converged(self) -> bool:
        return self.n_converged == self.n_points

    @property
    def signal_gain_db_array(self) -> jax.Array:
        return jnp.asarray([p.signal_gain_db for p in self.points], dtype=jnp.float64)

    @property
    def idler_conversion_db_array(self) -> jax.Array:
        values = [
            jnp.nan if p.idler_conversion_db is None else p.idler_conversion_db
            for p in self.points
        ]
        return jnp.asarray(values, dtype=jnp.float64)

    @property
    def passed(self) -> bool:
        if self.config.require_all_converged and not self.all_converged:
            return False
        gains = self.signal_gain_db_array
        if self.config.min_gain_db is not None and bool(jnp.any(gains < self.config.min_gain_db)):
            return False
        if self.config.max_gain_db is not None and bool(jnp.any(gains > self.config.max_gain_db)):
            return False
        return True

    def best_gain_point(self) -> GainPointResult:
        idx = int(jnp.nanargmax(self.signal_gain_db_array))
        return self.points[idx]

    def to_dict(self) -> dict[str, Any]:
        gains = self.signal_gain_db_array
        idlers = self.idler_conversion_db_array
        return {
            "passed": self.passed,
            "n_points": self.n_points,
            "n_converged": self.n_converged,
            "all_converged": self.all_converged,
            "gain_db_min": float(jnp.nanmin(gains)),
            "gain_db_max": float(jnp.nanmax(gains)),
            "gain_db_mean": float(jnp.nanmean(gains)),
            "idler_conversion_db_min": float(jnp.nanmin(idlers)) if bool(jnp.any(jnp.isfinite(idlers))) else None,
            "idler_conversion_db_max": float(jnp.nanmax(idlers)) if bool(jnp.any(jnp.isfinite(idlers))) else None,
            "best_gain_point": self.best_gain_point().gain.to_dict(),
            "config": self.config.to_dict(),
            "linearization": self.linearization.to_dict(include_matrix=False),
            "points": [p.to_dict() for p in self.points],
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class GainOperatingMapPoint:
    """
    One operating-map point: usually one pump setting plus one gain sweep.
    """

    pump_descriptor: Mapping[str, Any]
    sweep: GainSweepResult
    metadata: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.sweep.passed

    @property
    def max_gain_db(self) -> float:
        return float(jnp.nanmax(self.sweep.signal_gain_db_array))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pump_descriptor": dict(self.pump_descriptor),
            "passed": self.passed,
            "max_gain_db": self.max_gain_db,
            "sweep": self.sweep.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class GainOperatingMap:
    """
    Collection of gain sweeps over pump settings.
    """

    points: tuple[GainOperatingMapPoint, ...]
    metadata: Mapping[str, Any] | None = None

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_passed(self) -> int:
        return sum(1 for p in self.points if p.passed)

    def best_point(self) -> GainOperatingMapPoint:
        if not self.points:
            raise ValueError("Operating map has no points")
        idx = int(jnp.nanargmax(jnp.asarray([p.max_gain_db for p in self.points])))
        return self.points[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "n_passed": self.n_passed,
            "best_point": self.best_point().to_dict() if self.points else None,
            "points": [p.to_dict() for p in self.points],
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Linearization construction
# ---------------------------------------------------------------------------

def _unwrap_pump_result(
    pump_result: DistributedHBSolveResult | PumpHBLadderResult,
) -> DistributedHBSolveResult:
    if isinstance(pump_result, PumpHBLadderResult):
        return pump_result.distributed_result
    return pump_result


def build_gain_linearization_from_pump(
    pump_result: DistributedHBSolveResult | PumpHBLadderResult,
    *,
    target_plan: FrequencyPlan | None = None,
    config: SmallSignalLinearizationConfig | None = None,
    operating_injection_A: ArrayLike | None = None,
) -> DistributedHBLinearization:
    """
    Build a gain-ready small-signal linearization from a pump result.

    If target_plan is provided, pump tones are embedded into that richer plan.
    """
    dist = _unwrap_pump_result(pump_result)
    return build_linearization_from_pump_result(
        dist,
        target_plan=target_plan,
        operating_injection_A=operating_injection_A,
        config=config,
    )


# ---------------------------------------------------------------------------
# One-point and sweep solves
# ---------------------------------------------------------------------------

def solve_gain_point(
    linearization: DistributedHBLinearization,
    config: GainSolveConfig | None = None,
) -> GainPointResult:
    """
    Solve one small-signal gain point.

    The target FrequencyPlan inside linearization must contain config.signal_label
    and, if requested, config.idler_label.
    """
    cfg = config or GainSolveConfig()
    layout = linearization.layout

    out_node = layout.n_cells if cfg.output_node is None else cfg.output_node
    if cfg.input_node > layout.n_cells:
        raise ValueError(f"input_node {cfg.input_node} exceeds last node {layout.n_cells}")
    if out_node > layout.n_cells:
        raise ValueError(f"output_node {out_node} exceeds last node {layout.n_cells}")

    source = SmallSignalSource.current_phasor_at_node(
        plan=linearization.plan,
        layout=layout,
        node=cfg.input_node,
        label=cfg.signal_label,
        rms_current_A=cfg.signal_current_rms_A,
        set_conjugate=cfg.set_conjugate,
    )

    solution = solve_linearized_small_signal(
        linearization,
        source,
    )

    gain = compute_gain_point_from_solution(
        solution,
        signal_label=cfg.signal_label,
        idler_label=cfg.idler_label,
        input_node=cfg.input_node,
        output_node=out_node,
    )

    gp = matched_power_gain_from_voltage_gain(
        gain.signal_gain_complex,
        input_impedance_ohm=cfg.input_impedance_ohm,
        output_impedance_ohm=cfg.output_impedance_ohm,
    )
    gp_db = float(10.0 * jnp.log10(jnp.maximum(gp, 1e-300)))

    status = GainStatus.PASS if solution.converged else GainStatus.FAIL

    return GainPointResult(
        solve=solution,
        gain=gain,
        config=cfg,
        matched_power_gain=float(gp),
        matched_power_gain_db=gp_db,
        status=status,
        metadata={
            "source": source.to_dict(),
            "linear_solve": solution.linear_solve.to_dict(),
        },
    )


def solve_gain_sweep(
    linearization: DistributedHBLinearization,
    config: GainSweepConfig,
) -> GainSweepResult:
    """
    Solve multiple labelled signal/idler gain points on the same linearization.
    """
    points: list[GainPointResult] = []

    for point_cfg in config.points:
        points.append(solve_gain_point(linearization, point_cfg))

    return GainSweepResult(
        points=tuple(points),
        config=config,
        linearization=linearization,
        metadata={
            "linearization": linearization.to_dict(include_matrix=False),
        },
    )


def solve_gain_sweep_from_pump(
    pump_result: DistributedHBSolveResult | PumpHBLadderResult,
    *,
    target_plan: FrequencyPlan,
    sweep_config: GainSweepConfig,
    linearization_config: SmallSignalLinearizationConfig | None = None,
) -> GainSweepResult:
    """
    Convenience function:

        pump result -> linearization on target plan -> gain sweep
    """
    lin = build_gain_linearization_from_pump(
        pump_result,
        target_plan=target_plan,
        config=linearization_config,
    )
    return solve_gain_sweep(lin, sweep_config)


# ---------------------------------------------------------------------------
# Frequency-labelled extraction helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelledGainTrace:
    """
    Compact trace arrays extracted from a GainSweepResult.

    This is the object most scripts should serialize to CSV/NPZ.
    """

    signal_labels: tuple[str, ...]
    idler_labels: tuple[str | None, ...]
    signal_frequencies_hz: jax.Array
    idler_frequencies_hz: jax.Array
    signal_gain_db: jax.Array
    matched_power_gain_db: jax.Array
    idler_conversion_db: jax.Array
    converged: jax.Array
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_labels": list(self.signal_labels),
            "idler_labels": list(self.idler_labels),
            "n_points": len(self.signal_labels),
            "signal_frequency_min_hz": float(jnp.nanmin(self.signal_frequencies_hz)),
            "signal_frequency_max_hz": float(jnp.nanmax(self.signal_frequencies_hz)),
            "signal_gain_db_min": float(jnp.nanmin(self.signal_gain_db)),
            "signal_gain_db_max": float(jnp.nanmax(self.signal_gain_db)),
            "matched_power_gain_db_min": float(jnp.nanmin(self.matched_power_gain_db)),
            "matched_power_gain_db_max": float(jnp.nanmax(self.matched_power_gain_db)),
            "idler_conversion_db_min": (
                float(jnp.nanmin(self.idler_conversion_db))
                if bool(jnp.any(jnp.isfinite(self.idler_conversion_db)))
                else None
            ),
            "idler_conversion_db_max": (
                float(jnp.nanmax(self.idler_conversion_db))
                if bool(jnp.any(jnp.isfinite(self.idler_conversion_db)))
                else None
            ),
            "converged_count": int(jnp.sum(self.converged)),
            "metadata": dict(self.metadata or {}),
        }


def _frequency_for_label_or_nan(plan: FrequencyPlan, label: str | None) -> float:
    if label is None:
        return float("nan")
    try:
        idx = plan.position_of_label(label)
        return float(plan.frequencies_hz[idx])
    except Exception:
        return float("nan")


def extract_labelled_gain_trace(sweep: GainSweepResult) -> LabelledGainTrace:
    """
    Extract compact arrays from a gain sweep.
    """
    plan = sweep.linearization.plan

    signal_labels = tuple(p.config.signal_label for p in sweep.points)
    idler_labels = tuple(p.config.idler_label for p in sweep.points)

    fs = jnp.asarray(
        [_frequency_for_label_or_nan(plan, label) for label in signal_labels],
        dtype=jnp.float64,
    )
    fi = jnp.asarray(
        [_frequency_for_label_or_nan(plan, label) for label in idler_labels],
        dtype=jnp.float64,
    )
    gains = jnp.asarray([p.signal_gain_db for p in sweep.points], dtype=jnp.float64)
    power_gains = jnp.asarray([p.matched_power_gain_db for p in sweep.points], dtype=jnp.float64)
    idlers = jnp.asarray(
        [
            jnp.nan if p.idler_conversion_db is None else p.idler_conversion_db
            for p in sweep.points
        ],
        dtype=jnp.float64,
    )
    conv = jnp.asarray([p.converged for p in sweep.points], dtype=bool)

    return LabelledGainTrace(
        signal_labels=signal_labels,
        idler_labels=idler_labels,
        signal_frequencies_hz=fs,
        idler_frequencies_hz=fi,
        signal_gain_db=gains,
        matched_power_gain_db=power_gains,
        idler_conversion_db=idlers,
        converged=conv,
        metadata={
            "sweep": sweep.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Operating-map utilities
# ---------------------------------------------------------------------------

def solve_gain_operating_map_from_pump_results(
    pump_results: Sequence[DistributedHBSolveResult | PumpHBLadderResult],
    *,
    target_plan_factory: Callable[[DistributedHBSolveResult | PumpHBLadderResult], FrequencyPlan],
    sweep_config_factory: Callable[[FrequencyPlan], GainSweepConfig],
    linearization_config: SmallSignalLinearizationConfig | None = None,
) -> GainOperatingMap:
    """
    Build a gain operating map from already-computed pump results.

    Parameters
    ----------
    pump_results:
        Sequence of pump HB results.
    target_plan_factory:
        Callable that creates the small-signal target plan for each pump result.
    sweep_config_factory:
        Callable that creates the gain sweep config for the target plan.
    """
    points: list[GainOperatingMapPoint] = []

    for idx, pump in enumerate(pump_results):
        dist = _unwrap_pump_result(pump)
        target_plan = target_plan_factory(pump)
        sweep_cfg = sweep_config_factory(target_plan)

        sweep = solve_gain_sweep_from_pump(
            pump,
            target_plan=target_plan,
            sweep_config=sweep_cfg,
            linearization_config=linearization_config,
        )

        descriptor: dict[str, Any] = {
            "index": idx,
            "pump_converged": dist.converged,
            "layout_name": dist.layout.name,
            "plan": dist.frequency_plan.to_dict(),
        }

        if isinstance(pump, PumpHBLadderResult):
            descriptor["drive"] = pump.drive.to_dict()
            descriptor["pump_profile"] = pump.profile.to_dict()

        points.append(
            GainOperatingMapPoint(
                pump_descriptor=descriptor,
                sweep=sweep,
                metadata={
                    "target_plan": target_plan.to_dict(),
                },
            )
        )

    return GainOperatingMap(
        points=tuple(points),
        metadata={
            "n_pump_results": len(pump_results),
        },
    )


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GainValidationReport:
    """
    Validation report for gain calculations.
    """

    passed: bool
    messages: list[str]
    n_points: int
    n_converged: int
    gain_db_min: float | None
    gain_db_max: float | None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "messages": list(self.messages),
            "n_points": self.n_points,
            "n_converged": self.n_converged,
            "gain_db_min": self.gain_db_min,
            "gain_db_max": self.gain_db_max,
            "metadata": dict(self.metadata or {}),
        }


def validate_gain_sweep(
    sweep: GainSweepResult,
    *,
    require_all_converged: bool | None = None,
    min_gain_db: float | None = None,
    max_gain_db: float | None = None,
) -> GainValidationReport:
    """
    Validate a GainSweepResult against convergence and optional gain bounds.
    """
    req_all = sweep.config.require_all_converged if require_all_converged is None else require_all_converged
    lower = sweep.config.min_gain_db if min_gain_db is None else min_gain_db
    upper = sweep.config.max_gain_db if max_gain_db is None else max_gain_db

    messages: list[str] = []
    passed = True

    if req_all and not sweep.all_converged:
        passed = False
        messages.append(
            f"FAIL: only {sweep.n_converged}/{sweep.n_points} gain points converged."
        )

    gains = sweep.signal_gain_db_array
    gain_min = float(jnp.nanmin(gains)) if sweep.n_points else None
    gain_max = float(jnp.nanmax(gains)) if sweep.n_points else None

    if lower is not None and gain_min is not None and gain_min < lower:
        passed = False
        messages.append(f"FAIL: minimum gain {gain_min:.3g} dB is below {lower:.3g} dB.")

    if upper is not None and gain_max is not None and gain_max > upper:
        passed = False
        messages.append(f"FAIL: maximum gain {gain_max:.3g} dB is above {upper:.3g} dB.")

    if passed:
        messages.append("PASS: gain sweep validation checks passed.")

    return GainValidationReport(
        passed=passed,
        messages=messages,
        n_points=sweep.n_points,
        n_converged=sweep.n_converged,
        gain_db_min=gain_min,
        gain_db_max=gain_max,
        metadata={
            "sweep": sweep.to_dict(),
        },
    )


def gain_point_table(point: GainPointResult) -> str:
    """
    Markdown table for one gain point.
    """
    lines = [
        "| quantity | value |",
        "|---|---:|",
        f"| status | {point.status.value} |",
        f"| signal label | {point.config.signal_label} |",
        f"| idler label | {point.config.idler_label} |",
        f"| signal voltage gain dB | {point.signal_gain_db:.9g} |",
        f"| matched power gain dB | {point.matched_power_gain_db:.9g} |",
    ]
    if point.idler_conversion_db is not None:
        lines.append(f"| idler conversion dB | {point.idler_conversion_db:.9g} |")
    lines.append(f"| linear solve residual | {point.solve.linear_solve.linear_residual_norm:.9e} |")
    return "\n".join(lines)


def gain_sweep_table(sweep: GainSweepResult) -> str:
    """
    Markdown table for a gain sweep.
    """
    lines = [
        "| idx | signal | idler | status | signal gain dB | matched power gain dB | idler conv dB |",
        "|---:|---|---|---|---:|---:|---:|",
    ]

    for idx, point in enumerate(sweep.points):
        idler = "" if point.config.idler_label is None else point.config.idler_label
        idler_db = "" if point.idler_conversion_db is None else f"{point.idler_conversion_db:.6g}"
        lines.append(
            f"| {idx} | {point.config.signal_label} | {idler} | {point.status.value} | "
            f"{point.signal_gain_db:.6g} | {point.matched_power_gain_db:.6g} | {idler_db} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-check helpers
# ---------------------------------------------------------------------------

def validate_gain_linearization_smoke(
    linearization: DistributedHBLinearization,
    *,
    signal_label: str = "signal",
    idler_label: str | None = None,
    signal_current_rms_A: complex = 1e-12 + 0j,
) -> GainValidationReport:
    """
    Minimal smoke validation for one gain point on an existing linearization.
    """
    cfg = GainSolveConfig(
        signal_label=signal_label,
        idler_label=idler_label,
        signal_current_rms_A=signal_current_rms_A,
    )
    point = solve_gain_point(linearization, cfg)
    sweep = GainSweepResult(
        points=(point,),
        config=GainSweepConfig(points=(cfg,)),
        linearization=linearization,
    )
    return validate_gain_sweep(sweep)


__all__ = [
    "GainQuantity",
    "GainStatus",
    "GainInputKind",
    "complex_abs_db",
    "power_abs_db",
    "matched_power_gain_from_voltage_gain",
    "matched_power_gain_db_from_voltage_gain",
    "idler_frequency_dp4wm",
    "signal_frequency_from_detuning",
    "GainSolveConfig",
    "GainSweepConfig",
    "GainPointResult",
    "GainSweepResult",
    "GainOperatingMapPoint",
    "GainOperatingMap",
    "build_gain_linearization_from_pump",
    "solve_gain_point",
    "solve_gain_sweep",
    "solve_gain_sweep_from_pump",
    "LabelledGainTrace",
    "extract_labelled_gain_trace",
    "solve_gain_operating_map_from_pump_results",
    "GainValidationReport",
    "validate_gain_sweep",
    "gain_point_table",
    "gain_sweep_table",
    "validate_gain_linearization_smoke",
]