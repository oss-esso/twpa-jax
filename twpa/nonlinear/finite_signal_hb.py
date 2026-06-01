"""
twpa.nonlinear.finite_signal_hb
===============================

Finite-signal harmonic-balance utilities for pumped KI-TWPA simulations.

This module extends the pump-only HB workflow to include a finite input signal
inside the nonlinear HB solve itself:

    pump + signal drive
        -> nonlinear distributed HB solve
        -> output signal/idler extraction
        -> gain compression sweeps

This is different from the small-signal linearized gain calculation:

    pump HB
        -> linearization
        -> infinitesimal signal/idler solve

Finite-signal HB is needed for:
    - 1 dB compression,
    - gain saturation,
    - pump depletion,
    - signal-power-dependent idler conversion,
    - large-signal operating maps.

Important scope
---------------
This module uses the existing dense/reference distributed HB backend. It is
correct-first and suitable for reduced layouts. Industrial 20,000-cell finite
signal HB requires the later structured/block-banded backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import inspect
import numpy as np

import jax
import jax.numpy as jnp

from twpa.core import frequency_plan as frequency_plan_module
from twpa.core.frequency_plan import (
    FrequencyPlan,
    FrequencyPlanKind,
    Tone,
    ToneRole,
    dp4wm_lattice_indices_for_plan,
    make_custom_plan,
)
from twpa.core.harmonics import (
    coefficient_power_summary,
    set_single_rms_phasor_by_label,
)
from twpa.core.hb_fft import (
    HBProjectionConfig,
    HBProjectionGrid,
    ProjectionMode,
    make_multi_fundamental_projection_grid,
    make_projection_grid,
)
from twpa.core.layout import LineLayout
from twpa.core.params import NonlinearParams, SolverBackend, SolverConfig
from twpa.nonlinear.distributed_hb import (
    DistributedHBConfig,
    DistributedHBSolveResult,
    DistributedHBState,
    make_distributed_linear_initial_guess,
    make_kinetic_model_from_layout,
    make_node_current_injection_from_rms_phasor,
    solve_distributed_hb,
    zeros_node_injection,
)
from twpa.nonlinear.kinetic_inductance import KineticInductanceModel
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    dbm_to_norton_current_rms,
    norton_current_rms_to_dbm,
)
from twpa.solvers.continuation import ContinuationSolverConfig
from twpa.solvers.hb_solver import DenseNewtonConfig


ArrayLike = Any


class FiniteSignalStatus(str, Enum):
    """Finite-signal HB status."""

    CONVERGED = "converged"
    FAILED = "failed"


class FiniteSignalSweepKind(str, Enum):
    """Supported finite-signal sweep variables."""

    SIGNAL_CURRENT_RMS_A = "signal_current_rms_A"
    SIGNAL_POWER_DBM = "signal_power_dbm"
    SIGNAL_POWER_W = "signal_power_W"


class CompressionMetric(str, Enum):
    """Compression metric for finite-signal sweeps."""

    SIGNAL_GAIN_DB = "signal_gain_db"
    MATCHED_POWER_GAIN_DB = "matched_power_gain_db"


@dataclass(frozen=True)
class SignalDriveConfig:
    """
    Finite input signal drive configuration.

    Parameters
    ----------
    signal_frequency_hz:
        Signal frequency.
    current_rms_A:
        Equivalent Norton RMS current phasor injected at the input node.
    source_impedance_ohm:
        Source impedance used for current/power conversion.
    signal_label:
        Label of the signal tone in the frequency plan.
    idler_label:
        Optional idler label. For degenerate-pump four-wave mixing,
        f_i = 2 f_p - f_s.
    phase_rad:
        Signal current phase.
    input_node:
        Optional input node override.
    """

    signal_frequency_hz: float
    current_rms_A: float
    source_impedance_ohm: float = 50.0
    signal_label: str = "signal"
    idler_label: str | None = "idler"
    phase_rad: float = 0.0
    input_node: int | None = None

    def __post_init__(self) -> None:
        if self.signal_frequency_hz <= 0.0:
            raise ValueError("signal_frequency_hz must be positive")
        if self.current_rms_A < 0.0:
            raise ValueError("current_rms_A must be non-negative")
        if self.source_impedance_ohm <= 0.0:
            raise ValueError("source_impedance_ohm must be positive")
        if self.input_node is not None and int(self.input_node) < 0:
            raise ValueError("input_node must be non-negative if provided")
        if self.input_node is not None:
            object.__setattr__(self, "input_node", int(self.input_node))

    @classmethod
    def from_available_power_dbm(
        cls,
        *,
        signal_frequency_hz: float,
        power_dbm: float,
        source_impedance_ohm: float = 50.0,
        signal_label: str = "signal",
        idler_label: str | None = "idler",
        phase_rad: float = 0.0,
        input_node: int | None = None,
    ) -> "SignalDriveConfig":
        return cls(
            signal_frequency_hz=signal_frequency_hz,
            current_rms_A=float(
                dbm_to_norton_current_rms(
                    power_dbm,
                    source_impedance_ohm=source_impedance_ohm,
                )
            ),
            source_impedance_ohm=source_impedance_ohm,
            signal_label=signal_label,
            idler_label=idler_label,
            phase_rad=phase_rad,
            input_node=input_node,
        )

    @property
    def current_rms_phasor_A(self) -> complex:
        return complex(self.current_rms_A * jnp.exp(1j * self.phase_rad))

    @property
    def available_power_dbm(self) -> float:
        return float(
            norton_current_rms_to_dbm(
                self.current_rms_A,
                source_impedance_ohm=self.source_impedance_ohm,
            )
        )

    @property
    def available_power_W(self) -> float:
        i = self.current_rms_A
        return float(i * i * self.source_impedance_ohm / 4.0)

    def with_current_rms(self, current_rms_A: float) -> "SignalDriveConfig":
        return replace(self, current_rms_A=float(current_rms_A))

    def with_available_power_dbm(self, power_dbm: float) -> "SignalDriveConfig":
        return self.with_current_rms(
            float(
                dbm_to_norton_current_rms(
                    power_dbm,
                    source_impedance_ohm=self.source_impedance_ohm,
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_frequency_hz": self.signal_frequency_hz,
            "signal_frequency_GHz": self.signal_frequency_hz / 1e9,
            "current_rms_A": self.current_rms_A,
            "current_rms_phasor_A": {
                "real": float(jnp.real(self.current_rms_phasor_A)),
                "imag": float(jnp.imag(self.current_rms_phasor_A)),
                "abs": float(abs(self.current_rms_phasor_A)),
            },
            "available_power_W": self.available_power_W,
            "available_power_dbm": self.available_power_dbm,
            "source_impedance_ohm": self.source_impedance_ohm,
            "signal_label": self.signal_label,
            "idler_label": self.idler_label,
            "phase_rad": self.phase_rad,
            "input_node": self.input_node,
        }


@dataclass(frozen=True)
class FiniteSignalHBConfig:
    """
    Finite-signal HB configuration.

    Parameters
    ----------
    n_pump_harmonics:
        Number of pump harmonics in the nonlinear plan.
    include_negative_frequencies:
        Include conjugate tones.
    include_dc:
        Include DC tone.
    include_idler:
        Include degenerate four-wave-mixing idler.
    include_sum_tones:
        Include pump+signal and pump+idler mixing tones.
    include_second_order_sidebands:
        Include 2fp±fs tones in addition to idler.
    distributed:
        Distributed HB boundary/source config.
    projection:
        Time-domain HB projection config.
    solver:
        Dense Newton config.
    """

    n_pump_harmonics: int = 3
    include_negative_frequencies: bool = True
    include_dc: bool = False
    include_idler: bool = True
    include_sum_tones: bool = False
    include_second_order_sidebands: bool = True
    distributed: DistributedHBConfig = DistributedHBConfig()
    projection: HBProjectionConfig = HBProjectionConfig(
        n_time_samples=None,
        oversampling=8,
        force_real_time_signal=True,
        enforce_conjugate_symmetry=True,
    )
    solver: DenseNewtonConfig | SolverConfig = DenseNewtonConfig(
        max_iter=50,
        abs_tol=1e-9,
        rel_tol=1e-9,
        damping_initial=1.0,
        regularization=0.0,
    )
    max_dense_real_unknowns: int = 512
    allow_incommensurate_projection: bool = False
    name: str = "finite_signal_hb"

    def __post_init__(self) -> None:
        if int(self.n_pump_harmonics) <= 0:
            raise ValueError("n_pump_harmonics must be positive")
        object.__setattr__(self, "n_pump_harmonics", int(self.n_pump_harmonics))
        if int(self.max_dense_real_unknowns) <= 0:
            raise ValueError("max_dense_real_unknowns must be positive")
        object.__setattr__(self, "max_dense_real_unknowns", int(self.max_dense_real_unknowns))

    def with_updates(self, **kwargs: Any) -> "FiniteSignalHBConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pump_harmonics": self.n_pump_harmonics,
            "include_negative_frequencies": self.include_negative_frequencies,
            "include_dc": self.include_dc,
            "include_idler": self.include_idler,
            "include_sum_tones": self.include_sum_tones,
            "include_second_order_sidebands": self.include_second_order_sidebands,
            "distributed": self.distributed.to_dict(),
            "projection": self.projection.to_dict(),
            "solver": self.solver.to_dict(),
            "max_dense_real_unknowns": self.max_dense_real_unknowns,
            "allow_incommensurate_projection": self.allow_incommensurate_projection,
            "name": self.name,
        }


@dataclass(frozen=True)
class FiniteSignalObservable:
    """
    Extracted finite-signal observable values.
    """

    input_signal_voltage_V: complex
    output_signal_voltage_V: complex
    output_idler_voltage_V: complex | None
    signal_voltage_gain: complex
    signal_gain_db: float
    matched_power_gain: float
    matched_power_gain_db: float
    idler_conversion: complex | None
    idler_conversion_db: float | None
    pump_output_voltage_V: complex | None
    pump_output_to_input_gain_db: float | None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        def cdict(value: complex | None) -> dict[str, float] | None:
            if value is None:
                return None
            return {
                "real": float(jnp.real(value)),
                "imag": float(jnp.imag(value)),
                "abs": float(abs(value)),
            }

        return {
            "input_signal_voltage_V": cdict(self.input_signal_voltage_V),
            "output_signal_voltage_V": cdict(self.output_signal_voltage_V),
            "output_idler_voltage_V": cdict(self.output_idler_voltage_V),
            "signal_voltage_gain": cdict(self.signal_voltage_gain),
            "signal_gain_db": self.signal_gain_db,
            "matched_power_gain": self.matched_power_gain,
            "matched_power_gain_db": self.matched_power_gain_db,
            "idler_conversion": cdict(self.idler_conversion),
            "idler_conversion_db": self.idler_conversion_db,
            "pump_output_voltage_V": cdict(self.pump_output_voltage_V),
            "pump_output_to_input_gain_db": self.pump_output_to_input_gain_db,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class FiniteSignalHBResult:
    """
    Full finite-signal HB result.
    """

    distributed_result: DistributedHBSolveResult
    pump_drive: PumpDriveConfig
    signal_drive: SignalDriveConfig
    finite_config: FiniteSignalHBConfig
    observables: FiniteSignalObservable
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.distributed_result.converged

    @property
    def status(self) -> FiniteSignalStatus:
        return FiniteSignalStatus.CONVERGED if self.converged else FiniteSignalStatus.FAILED

    @property
    def state(self) -> DistributedHBState:
        return self.distributed_result.state

    @property
    def frequency_plan(self) -> FrequencyPlan:
        return self.distributed_result.frequency_plan

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "converged": self.converged,
            "pump_drive": self.pump_drive.to_dict(),
            "signal_drive": self.signal_drive.to_dict(),
            "finite_config": self.finite_config.to_dict(),
            "observables": self.observables.to_dict(),
            "distributed_result": self.distributed_result.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class CompressionPoint:
    """
    One point in a finite-signal compression sweep.
    """

    signal_power_dbm: float
    signal_current_rms_A: float
    result: FiniteSignalHBResult

    @property
    def converged(self) -> bool:
        return self.result.converged

    @property
    def signal_gain_db(self) -> float:
        return self.result.observables.signal_gain_db

    @property
    def matched_power_gain_db(self) -> float:
        return self.result.observables.matched_power_gain_db

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_power_dbm": self.signal_power_dbm,
            "signal_current_rms_A": self.signal_current_rms_A,
            "converged": self.converged,
            "signal_gain_db": self.signal_gain_db,
            "matched_power_gain_db": self.matched_power_gain_db,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class CompressionSweepResult:
    """
    Finite-signal gain-compression sweep.
    """

    points: tuple[CompressionPoint, ...]
    small_signal_reference_gain_db: float | None = None
    compression_metric: CompressionMetric = CompressionMetric.SIGNAL_GAIN_DB
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

    @property
    def signal_power_dbm_array(self) -> jax.Array:
        return jnp.asarray([p.signal_power_dbm for p in self.points], dtype=jnp.float64)

    @property
    def gain_db_array(self) -> jax.Array:
        if self.compression_metric == CompressionMetric.MATCHED_POWER_GAIN_DB:
            return jnp.asarray([p.matched_power_gain_db for p in self.points], dtype=jnp.float64)
        return jnp.asarray([p.signal_gain_db for p in self.points], dtype=jnp.float64)

    def compression_power_dbm(self, *, compression_db: float = 1.0) -> float | None:
        """
        Estimate input power where gain is compressed by `compression_db`.

        Linear interpolation is used between the first crossing points.
        """
        if self.n_points == 0:
            return None

        gains = np.asarray(self.gain_db_array, dtype=float)
        powers = np.asarray(self.signal_power_dbm_array, dtype=float)

        reference = (
            float(self.small_signal_reference_gain_db)
            if self.small_signal_reference_gain_db is not None
            else float(gains[0])
        )
        target = reference - float(compression_db)
        delta = gains - target

        crossing_indices = np.where(delta <= 0.0)[0]
        if crossing_indices.size == 0:
            return None

        idx = int(crossing_indices[0])
        if idx == 0:
            return float(powers[0])

        x0, x1 = powers[idx - 1], powers[idx]
        y0, y1 = delta[idx - 1], delta[idx]
        if abs(y1 - y0) <= 1e-300:
            return float(x1)

        frac = -y0 / (y1 - y0)
        return float(x0 + frac * (x1 - x0))

    def to_dict(self) -> dict[str, Any]:
        gains = self.gain_db_array
        return {
            "n_points": self.n_points,
            "n_converged": self.n_converged,
            "converged": self.converged,
            "compression_metric": self.compression_metric.value,
            "small_signal_reference_gain_db": self.small_signal_reference_gain_db,
            "gain_db_min": float(jnp.nanmin(gains)) if self.n_points else None,
            "gain_db_max": float(jnp.nanmax(gains)) if self.n_points else None,
            "p1db_input_dbm": self.compression_power_dbm(compression_db=1.0),
            "points": [p.to_dict() for p in self.points],
            "metadata": dict(self.metadata or {}),
        }


def _try_call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**filtered)
    except TypeError:
        return fn(**kwargs)


def make_finite_signal_plan(
    pump_drive: PumpDriveConfig,
    signal_drive: SignalDriveConfig,
    config: FiniteSignalHBConfig,
) -> FrequencyPlan:
    """
    Build a frequency plan for finite-signal HB.

    Native project constructors are tried first. If none are available, a
    generic custom plan is constructed from pump harmonics and signal/idler
    mixing tones.
    """
    fp = pump_drive.pump_frequency_hz
    fs = signal_drive.signal_frequency_hz
    fi = 2.0 * fp - fs

    fpmod = frequency_plan_module

    constructor_names = [
        "make_finite_signal_plan",
        "make_pump_signal_idler_plan",
        "make_signal_idler_plan",
        "make_dp4wm_plan",
        "make_dp4wm_frequency_plan",
        "make_gain_plan",
        "make_small_signal_plan",
    ]

    kwargs = {
        "pump_frequency_hz": fp,
        "signal_frequency_hz": fs,
        "idler_frequency_hz": fi,
        "pump_label": pump_drive.pump_label,
        "signal_label": signal_drive.signal_label,
        "idler_label": signal_drive.idler_label,
        "n_pump_harmonics": config.n_pump_harmonics,
        "n_harmonics": config.n_pump_harmonics,
        "include_negative": config.include_negative_frequencies,
        "include_negative_frequencies": config.include_negative_frequencies,
        "include_dc": config.include_dc,
        "include_idler": config.include_idler,
        "include_sum_tones": config.include_sum_tones,
        "include_second_order_sidebands": config.include_second_order_sidebands,
        "sort": "frequency",
    }

    errors: list[str] = []
    for name in constructor_names:
        fn = getattr(fpmod, name, None)
        if fn is None:
            continue
        try:
            plan = _try_call_with_supported_kwargs(fn, kwargs)
            plan.position_of_label(pump_drive.pump_label)
            plan.position_of_label(signal_drive.signal_label)
            if signal_drive.idler_label is not None and config.include_idler:
                plan.position_of_label(signal_drive.idler_label)
            return plan
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    tones: list[Tone] = []
    used_labels: set[str] = set()
    used_frequencies: list[float] = []

    def add(label: str, freq: float, role: ToneRole, index: int = 0) -> None:
        if label in used_labels:
            return
        if any(abs(float(freq) - existing) <= 1e-6 for existing in used_frequencies):
            return
        tones.append(Tone(label=label, frequency_hz=float(freq), role=role, index=index))
        used_labels.add(label)
        used_frequencies.append(float(freq))

    def add_pair(label: str, freq: float, role: ToneRole, index: int = 0) -> None:
        add(label, freq, role, index)
        if config.include_negative_frequencies and freq != 0.0:
            add(f"neg_{label}", -freq, role, -index)

    if config.include_dc:
        add("dc", 0.0, ToneRole.DC)
    for h in range(1, config.n_pump_harmonics + 1):
        add_pair(
            pump_drive.pump_label if h == 1 else f"{pump_drive.pump_label}_{h}",
            h * fp,
            ToneRole.PUMP if h == 1 else ToneRole.PUMP_HARMONIC,
            h,
        )
    add_pair(signal_drive.signal_label, fs, ToneRole.SIGNAL)
    if config.include_idler and signal_drive.idler_label is not None:
        add_pair(signal_drive.idler_label, fi, ToneRole.IDLER, -1)
    if config.include_second_order_sidebands:
        add_pair(f"2pump_plus_{signal_drive.signal_label}", 2.0 * fp + fs, ToneRole.SIDEBAND, 1)
    if config.include_sum_tones:
        add_pair(f"pump_plus_{signal_drive.signal_label}", fp + fs, ToneRole.SIDEBAND)
        add_pair(f"pump_minus_{signal_drive.signal_label}", fp - fs, ToneRole.SIDEBAND)
        if signal_drive.idler_label is not None:
            add_pair(f"pump_plus_{signal_drive.idler_label}", fp + fi, ToneRole.SIDEBAND)
            add_pair(f"pump_minus_{signal_drive.idler_label}", fp - fi, ToneRole.SIDEBAND)

    return make_custom_plan(
        tones,
        kind=FrequencyPlanKind.FINITE_SIGNAL_HB,
        reference_pump_hz=fp,
        reference_signal_hz=fs,
        metadata={
            "source": "twpa.nonlinear.finite_signal_hb.make_finite_signal_plan",
            "adapter_errors": errors,
        },
        sort="frequency",
    )


def make_pump_signal_injection(
    plan: FrequencyPlan,
    layout: LineLayout,
    pump_drive: PumpDriveConfig,
    signal_drive: SignalDriveConfig,
    distributed_config: DistributedHBConfig,
) -> jax.Array:
    """
    Build combined pump + signal node-current injection.
    """
    injection = zeros_node_injection(plan, layout)

    pump_node = distributed_config.input_node if pump_drive.input_node is None else pump_drive.input_node
    signal_node = distributed_config.input_node if signal_drive.input_node is None else signal_drive.input_node

    injection = injection + make_node_current_injection_from_rms_phasor(
        plan,
        layout,
        node=pump_node,
        label=pump_drive.pump_label,
        rms_current_A=pump_drive.current_rms_phasor_A,
        set_conjugate=True,
    )

    injection = injection + make_node_current_injection_from_rms_phasor(
        plan,
        layout,
        node=signal_node,
        label=signal_drive.signal_label,
        rms_current_A=signal_drive.current_rms_phasor_A,
        set_conjugate=True,
    )

    return injection


def make_finite_signal_projection_grid(
    plan: FrequencyPlan,
    config: FiniteSignalHBConfig,
) -> HBProjectionGrid:
    """Select an exact single-period or 2D DP4WM torus projection grid."""
    single_period_grid = make_projection_grid(
        plan.frequencies_hz,
        fundamental_frequency_hz=None,
        config=config.projection,
    )
    mode = config.projection.mode
    if mode == ProjectionMode.SINGLE_PERIOD:
        return single_period_grid
    if mode == ProjectionMode.AUTO and single_period_grid.is_commensurate:
        assert single_period_grid.integer_indices is not None
        max_index = int(jnp.max(jnp.abs(single_period_grid.integer_indices)))
        if single_period_grid.n_time_samples > 2 * max_index:
            return single_period_grid
    if mode in {ProjectionMode.AUTO, ProjectionMode.MULTI_FUNDAMENTAL}:
        if plan.reference_pump_hz is None or plan.reference_signal_hz is None:
            raise ValueError("Multi-fundamental finite-signal HB requires pump and signal references")
        return make_multi_fundamental_projection_grid(
            plan.frequencies_hz,
            fundamental_frequencies_hz=(
                plan.reference_pump_hz,
                plan.reference_signal_hz,
            ),
            lattice_indices=dp4wm_lattice_indices_for_plan(
                plan,
                atol_hz=config.projection.commensurability_atol_hz,
            ),
            config=config.projection,
        )
    return single_period_grid


def summarize_finite_signal_observables(
    result: DistributedHBSolveResult,
    pump_drive: PumpDriveConfig,
    signal_drive: SignalDriveConfig,
    *,
    input_impedance_ohm: float = 50.0,
    output_impedance_ohm: float = 50.0,
) -> FiniteSignalObservable:
    """
    Extract signal/idler gain and pump diagnostics from a finite-signal HB state.
    """
    plan = result.frequency_plan
    state = result.state

    input_node = result.input_node
    output_node = result.output_node

    sig_idx = plan.position_of_label(signal_drive.signal_label)
    pump_idx = plan.position_of_label(pump_drive.pump_label)

    vin_sig = state.node_voltage_coeffs_V[sig_idx, input_node]
    vout_sig = state.node_voltage_coeffs_V[sig_idx, output_node]

    signal_gain = vout_sig / jnp.where(jnp.abs(vin_sig) > 1e-300, vin_sig, 1e-300 + 0j)
    signal_gain_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(signal_gain), 1e-300))
    matched_power_gain = jnp.abs(signal_gain) ** 2 * input_impedance_ohm / output_impedance_ohm
    matched_power_gain_db = 10.0 * jnp.log10(jnp.maximum(matched_power_gain, 1e-300))

    vout_idler = None
    idler_conversion = None
    idler_conversion_db = None
    idler_idx = None

    if signal_drive.idler_label is not None:
        try:
            idler_idx = plan.position_of_label(signal_drive.idler_label)
            vout_idler_jax = state.node_voltage_coeffs_V[idler_idx, output_node]
            idler_conversion_jax = vout_idler_jax / jnp.where(
                jnp.abs(vin_sig) > 1e-300,
                vin_sig,
                1e-300 + 0j,
            )
            vout_idler = complex(vout_idler_jax)
            idler_conversion = complex(idler_conversion_jax)
            idler_conversion_db = float(
                20.0 * jnp.log10(jnp.maximum(jnp.abs(idler_conversion_jax), 1e-300))
            )
        except Exception:
            pass

    vin_pump = state.node_voltage_coeffs_V[pump_idx, input_node]
    vout_pump = state.node_voltage_coeffs_V[pump_idx, output_node]
    pump_gain = vout_pump / jnp.where(jnp.abs(vin_pump) > 1e-300, vin_pump, 1e-300 + 0j)
    pump_gain_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(pump_gain), 1e-300))

    return FiniteSignalObservable(
        input_signal_voltage_V=complex(vin_sig),
        output_signal_voltage_V=complex(vout_sig),
        output_idler_voltage_V=vout_idler,
        signal_voltage_gain=complex(signal_gain),
        signal_gain_db=float(signal_gain_db),
        matched_power_gain=float(matched_power_gain),
        matched_power_gain_db=float(matched_power_gain_db),
        idler_conversion=idler_conversion,
        idler_conversion_db=idler_conversion_db,
        pump_output_voltage_V=complex(vout_pump),
        pump_output_to_input_gain_db=float(pump_gain_db),
        metadata={
            "signal_tone_index": int(sig_idx),
            "idler_tone_index": None if idler_idx is None else int(idler_idx),
            "pump_tone_index": int(pump_idx),
            "input_node": int(input_node),
            "output_node": int(output_node),
            "input_impedance_ohm": input_impedance_ohm,
            "output_impedance_ohm": output_impedance_ohm,
        },
    )


def solve_finite_signal_hb(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    pump_drive: PumpDriveConfig,
    signal_drive: SignalDriveConfig,
    finite_config: FiniteSignalHBConfig | None = None,
    plan: FrequencyPlan | None = None,
    x0: DistributedHBState | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FiniteSignalHBResult:
    """
    Solve finite-signal distributed HB for pump + signal excitation.
    """
    cfg = finite_config or FiniteSignalHBConfig()

    if plan is None:
        plan = make_finite_signal_plan(pump_drive, signal_drive, cfg)
    n_real_unknowns = 2 * plan.n_tones * (2 * layout.n_cells + 1)
    solver_backend = (
        cfg.solver.backend
        if isinstance(cfg.solver, SolverConfig)
        else SolverBackend.DENSE
    )
    if solver_backend != SolverBackend.NEWTON_KRYLOV and n_real_unknowns > cfg.max_dense_real_unknowns:
        jacobian_gib = (n_real_unknowns**2 * 8) / (1024**3)
        raise RuntimeError(
            "Dense reference finite-signal HB refused unsafe problem size: "
            f"{n_real_unknowns} real unknowns exceed max_dense_real_unknowns="
            f"{cfg.max_dense_real_unknowns}. One float64 Jacobian would require "
            f"about {jacobian_gib:.3f} GiB before solver copies. "
            "Coarsen the layout or use the newton_krylov backend."
        )

    if ki_model is None:
        if nonlinear_params is None:
            raise ValueError("Either nonlinear_params or ki_model must be provided")
        ki_model = make_kinetic_model_from_layout(
            layout,
            nonlinear_params,
            name=f"{layout.name}_finite_signal_ki_model",
        )

    injection = make_pump_signal_injection(
        plan,
        layout,
        pump_drive,
        signal_drive,
        cfg.distributed,
    )

    grid = make_finite_signal_projection_grid(plan, cfg)
    if not grid.is_commensurate and not cfg.allow_incommensurate_projection:
        if grid.mode != ProjectionMode.MULTI_FUNDAMENTAL:
            raise ValueError(
                "Finite-signal HB requires a commensurate single-period tone plan "
                "or multi-fundamental projection."
            )
    if grid.integer_indices is not None:
        max_index = int(jnp.max(jnp.abs(grid.integer_indices)))
        if grid.n_time_samples <= 2 * max_index:
            raise ValueError(
                "Finite-signal HB projection grid is under-resolved: "
                f"n_time_samples={grid.n_time_samples}, max harmonic index={max_index}. "
                "Increase n_time_samples, use a controlled commensurate approximation, "
                "or implement multi-fundamental HB."
            )

    if x0 is None:
        x0 = make_distributed_linear_initial_guess(
            plan,
            layout,
            cfg.distributed,
            injection,
        )

    distributed_result = solve_distributed_hb(
        plan,
        layout,
        cfg.distributed,
        ki_model,
        injection,
        x0=x0,
        projection_grid=grid,
        projection_config=cfg.projection,
        solver_config=cfg.solver,
        metadata={
            "driver": "solve_finite_signal_hb",
            "pump_drive": pump_drive.to_dict(),
            "signal_drive": signal_drive.to_dict(),
            "finite_config": cfg.to_dict(),
            **dict(metadata or {}),
        },
    )

    output_impedance = (
        1.0 / cfg.distributed.load_conductance_S
        if cfg.distributed.load_conductance_S > 0.0
        else signal_drive.source_impedance_ohm
    )

    observables = summarize_finite_signal_observables(
        distributed_result,
        pump_drive,
        signal_drive,
        input_impedance_ohm=signal_drive.source_impedance_ohm,
        output_impedance_ohm=output_impedance,
    )

    return FiniteSignalHBResult(
        distributed_result=distributed_result,
        pump_drive=pump_drive,
        signal_drive=signal_drive,
        finite_config=cfg,
        observables=observables,
        metadata={
            "projection_grid": grid.to_dict(),
            **dict(metadata or {}),
        },
    )


def solve_finite_signal_hb_from_powers_dbm(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    *,
    pump_frequency_hz: float,
    signal_frequency_hz: float,
    pump_power_dbm: float,
    signal_power_dbm: float,
    source_impedance_ohm: float = 50.0,
    finite_config: FiniteSignalHBConfig | None = None,
) -> FiniteSignalHBResult:
    """
    Convenience finite-signal solve from available pump and signal powers.
    """
    pump_drive = PumpDriveConfig.from_available_power_dbm(
        pump_frequency_hz=pump_frequency_hz,
        power_dbm=pump_power_dbm,
        source_impedance_ohm=source_impedance_ohm,
    )
    signal_drive = SignalDriveConfig.from_available_power_dbm(
        signal_frequency_hz=signal_frequency_hz,
        power_dbm=signal_power_dbm,
        source_impedance_ohm=source_impedance_ohm,
    )
    return solve_finite_signal_hb(
        layout,
        nonlinear_params,
        pump_drive=pump_drive,
        signal_drive=signal_drive,
        finite_config=finite_config,
    )


def sweep_signal_power_compression(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    pump_drive: PumpDriveConfig,
    base_signal_drive: SignalDriveConfig,
    signal_power_dbm_values: Sequence[float],
    finite_config: FiniteSignalHBConfig | None = None,
    reuse_previous_solution: bool = True,
    small_signal_reference_gain_db: float | None = None,
    compression_metric: CompressionMetric = CompressionMetric.SIGNAL_GAIN_DB,
) -> CompressionSweepResult:
    """
    Sweep finite signal input power and estimate gain compression.
    """
    cfg = finite_config or FiniteSignalHBConfig()
    points: list[CompressionPoint] = []
    x0: DistributedHBState | None = None

    for p_dbm in signal_power_dbm_values:
        signal_drive = base_signal_drive.with_available_power_dbm(float(p_dbm))

        result = solve_finite_signal_hb(
            layout,
            nonlinear_params,
            ki_model=ki_model,
            pump_drive=pump_drive,
            signal_drive=signal_drive,
            finite_config=cfg,
            x0=x0,
            metadata={
                "sweep": "signal_power_compression",
                "signal_power_dbm": float(p_dbm),
            },
        )

        points.append(
            CompressionPoint(
                signal_power_dbm=float(p_dbm),
                signal_current_rms_A=signal_drive.current_rms_A,
                result=result,
            )
        )

        if reuse_previous_solution and result.converged:
            x0 = result.state

    return CompressionSweepResult(
        points=tuple(points),
        small_signal_reference_gain_db=small_signal_reference_gain_db,
        compression_metric=CompressionMetric(compression_metric),
        metadata={
            "pump_drive": pump_drive.to_dict(),
            "base_signal_drive": base_signal_drive.to_dict(),
            "finite_config": cfg.to_dict(),
        },
    )


def sweep_signal_current_compression(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    pump_drive: PumpDriveConfig,
    base_signal_drive: SignalDriveConfig,
    signal_current_rms_values: Sequence[float],
    finite_config: FiniteSignalHBConfig | None = None,
    reuse_previous_solution: bool = True,
    small_signal_reference_gain_db: float | None = None,
    compression_metric: CompressionMetric = CompressionMetric.SIGNAL_GAIN_DB,
) -> CompressionSweepResult:
    """
    Sweep finite signal input current and estimate gain compression.
    """
    power_values_dbm = [
        float(norton_current_rms_to_dbm(i, source_impedance_ohm=base_signal_drive.source_impedance_ohm))
        for i in signal_current_rms_values
    ]

    cfg = finite_config or FiniteSignalHBConfig()
    points: list[CompressionPoint] = []
    x0: DistributedHBState | None = None

    for i_rms, p_dbm in zip(signal_current_rms_values, power_values_dbm):
        signal_drive = base_signal_drive.with_current_rms(float(i_rms))

        result = solve_finite_signal_hb(
            layout,
            nonlinear_params,
            ki_model=ki_model,
            pump_drive=pump_drive,
            signal_drive=signal_drive,
            finite_config=cfg,
            x0=x0,
            metadata={
                "sweep": "signal_current_compression",
                "signal_current_rms_A": float(i_rms),
            },
        )

        points.append(
            CompressionPoint(
                signal_power_dbm=float(p_dbm),
                signal_current_rms_A=float(i_rms),
                result=result,
            )
        )

        if reuse_previous_solution and result.converged:
            x0 = result.state

    return CompressionSweepResult(
        points=tuple(points),
        small_signal_reference_gain_db=small_signal_reference_gain_db,
        compression_metric=CompressionMetric(compression_metric),
        metadata={
            "pump_drive": pump_drive.to_dict(),
            "base_signal_drive": base_signal_drive.to_dict(),
            "finite_config": cfg.to_dict(),
        },
    )


def finite_signal_result_table(result: FiniteSignalHBResult) -> str:
    """
    Markdown table for one finite-signal HB result.
    """
    obs = result.observables

    lines = [
        "| quantity | value |",
        "|---|---:|",
        f"| converged | {result.converged} |",
        f"| pump frequency GHz | {result.pump_drive.pump_frequency_hz / 1e9:.9g} |",
        f"| signal frequency GHz | {result.signal_drive.signal_frequency_hz / 1e9:.9g} |",
        f"| pump power dBm | {result.pump_drive.available_power_dbm:.9g} |",
        f"| signal power dBm | {result.signal_drive.available_power_dbm:.9g} |",
        f"| signal gain dB | {obs.signal_gain_db:.9g} |",
        f"| matched power gain dB | {obs.matched_power_gain_db:.9g} |",
        f"| residual norm | {result.distributed_result.residual.norm:.9e} |",
    ]

    if obs.idler_conversion_db is not None:
        lines.append(f"| idler conversion dB | {obs.idler_conversion_db:.9g} |")

    if obs.pump_output_to_input_gain_db is not None:
        lines.append(f"| pump output/input gain dB | {obs.pump_output_to_input_gain_db:.9g} |")

    return "\n".join(lines)


def compression_sweep_table(result: CompressionSweepResult) -> str:
    """
    Markdown table for a compression sweep.
    """
    lines = [
        "| idx | signal power dBm | status | signal gain dB | matched power gain dB |",
        "|---:|---:|---|---:|---:|",
    ]

    for idx, point in enumerate(result.points):
        lines.append(
            f"| {idx} | {point.signal_power_dbm:.6g} | "
            f"{point.result.status.value} | "
            f"{point.signal_gain_db:.6g} | "
            f"{point.matched_power_gain_db:.6g} |"
        )

    p1db = result.compression_power_dbm(compression_db=1.0)
    lines += [
        "",
        f"- estimated P1dB input: `{p1db}` dBm",
    ]

    return "\n".join(lines)


__all__ = [
    "FiniteSignalStatus",
    "FiniteSignalSweepKind",
    "CompressionMetric",
    "SignalDriveConfig",
    "FiniteSignalHBConfig",
    "FiniteSignalObservable",
    "FiniteSignalHBResult",
    "CompressionPoint",
    "CompressionSweepResult",
    "make_finite_signal_plan",
    "make_pump_signal_injection",
    "make_finite_signal_projection_grid",
    "summarize_finite_signal_observables",
    "solve_finite_signal_hb",
    "solve_finite_signal_hb_from_powers_dbm",
    "sweep_signal_power_compression",
    "sweep_signal_current_compression",
    "finite_signal_result_table",
    "compression_sweep_table",
]
