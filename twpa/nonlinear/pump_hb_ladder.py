"""
twpa.nonlinear.pump_hb_ladder
=============================

Production-facing pump-only harmonic-balance driver for nonlinear TWPA ladders.

This module wraps the lower-level distributed HB residual into a clean pump
solver API.

Purpose
-------
The industrial simulation flow is:

    1. Build/validate linear layout.
    2. Solve large-signal pump-only HB on a reduced or full backend.
    3. Linearize around the pumped state.
    4. Solve small-signal signal/idler conversion.
    5. Sweep pump power, pump frequency, and fabrication parameters.

This file implements step 2 for the dense/reference backend.

It is not yet the final 20,000-cell block-banded backend. It is the high-level
API and reference implementation that later industrial backends must match.

Drive convention
----------------
The distributed residual uses an injected Norton current at the input node.
For an available pump power P_av into source impedance Zs, the equivalent
Norton RMS current is

    I_N,rms = 2 sqrt(P_av / Zs)

because

    P_av = |V_oc,rms|^2 / (4 Zs)
    I_N,rms = V_oc,rms / Zs.

The Fourier coefficient at +fp is then

    I_{+fp} = I_N,rms / sqrt(2)

with the negative-frequency conjugate populated when the plan is double-sided.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp

from twpa.core.frequency_plan import (
    FrequencyPlan,
    make_pump_only_plan,
)
from twpa.core.harmonics import coefficient_power_summary
from twpa.core.hb_fft import (
    HBProjectionConfig,
    HBProjectionGrid,
    coefficients_to_time,
    make_projection_grid_from_plan,
)
from twpa.core.layout import LineLayout
from twpa.core.params import NonlinearParams, SolverBackend, SolverConfig
from twpa.nonlinear.distributed_hb import (
    DistributedHBConfig,
    DistributedHBResidual,
    DistributedHBSolveResult,
    DistributedHBState,
    evaluate_distributed_hb_residual,
    make_distributed_hb_residual_function,
    make_distributed_linear_initial_guess,
    make_input_pump_current_injection,
    make_kinetic_model_from_layout,
    solve_distributed_hb,
)
from twpa.nonlinear.kinetic_inductance import KineticInductanceModel
from twpa.solvers.continuation import (
    ContinuationResult,
    ContinuationSolverConfig,
    make_continuation_schedule,
    solve_continuation,
)
from twpa.solvers.hb_solver import DenseNewtonConfig, HBSolverResult


ArrayLike = Any
PyTree = Any


# ---------------------------------------------------------------------------
# Enums / scalar conversions
# ---------------------------------------------------------------------------

class PumpDriveKind(str, Enum):
    """Supported pump-drive parameterizations."""

    CURRENT_RMS = "current_rms"
    AVAILABLE_POWER_DBM = "available_power_dbm"
    AVAILABLE_POWER_W = "available_power_w"


class PumpContinuationKind(str, Enum):
    """Supported pump-continuation schedule variables."""

    CURRENT_RMS_A = "current_rms_A"
    AVAILABLE_POWER_DBM = "available_power_dbm"
    AVAILABLE_POWER_W = "available_power_W"


class PumpHBStatus(str, Enum):
    """High-level pump solve status."""

    CONVERGED = "converged"
    FAILED = "failed"


def dbm_to_watt(power_dbm: ArrayLike) -> jax.Array:
    """
    Convert dBm to watts.

        P[W] = 1e-3 * 10^(P_dBm / 10)
    """
    return 1e-3 * 10.0 ** (jnp.asarray(power_dbm, dtype=jnp.float64) / 10.0)


def watt_to_dbm(power_W: ArrayLike, *, floor_W: float = 1e-300) -> jax.Array:
    """
    Convert watts to dBm.
    """
    p = jnp.maximum(jnp.asarray(power_W, dtype=jnp.float64), floor_W)
    return 10.0 * jnp.log10(p / 1e-3)


def norton_current_rms_from_available_power(
    power_W: ArrayLike,
    *,
    source_impedance_ohm: float = 50.0,
) -> jax.Array:
    """
    Equivalent Norton RMS current for available source power.

        I_N,rms = 2 sqrt(P_av / Zs)
    """
    if source_impedance_ohm <= 0.0:
        raise ValueError("source_impedance_ohm must be positive")
    p = jnp.asarray(power_W, dtype=jnp.float64)
    if bool(jnp.any(p < 0.0)):
        raise ValueError("available power must be non-negative")
    return 2.0 * jnp.sqrt(p / source_impedance_ohm)


def available_power_from_norton_current_rms(
    current_rms_A: ArrayLike,
    *,
    source_impedance_ohm: float = 50.0,
) -> jax.Array:
    """
    Available source power from equivalent Norton RMS current.

        P_av = I_N,rms^2 Zs / 4
    """
    if source_impedance_ohm <= 0.0:
        raise ValueError("source_impedance_ohm must be positive")
    i = jnp.asarray(current_rms_A, dtype=jnp.float64)
    return i**2 * source_impedance_ohm / 4.0


def dbm_to_norton_current_rms(
    power_dbm: ArrayLike,
    *,
    source_impedance_ohm: float = 50.0,
) -> jax.Array:
    """dBm -> available power -> Norton RMS current."""
    return norton_current_rms_from_available_power(
        dbm_to_watt(power_dbm),
        source_impedance_ohm=source_impedance_ohm,
    )


def norton_current_rms_to_dbm(
    current_rms_A: ArrayLike,
    *,
    source_impedance_ohm: float = 50.0,
) -> jax.Array:
    """Norton RMS current -> available power -> dBm."""
    return watt_to_dbm(
        available_power_from_norton_current_rms(
            current_rms_A,
            source_impedance_ohm=source_impedance_ohm,
        )
    )


# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PumpDriveConfig:
    """
    Pump-drive configuration.

    Parameters
    ----------
    kind:
        Drive parameterization.
    pump_frequency_hz:
        Pump frequency.
    value:
        Current RMS in A, available power in dBm, or available power in W,
        depending on `kind`.
    source_impedance_ohm:
        Source impedance used for available-power conversion.
    pump_label:
        Label used inside FrequencyPlan.
    phase_rad:
        Pump phase. Applied to the RMS current phasor.
    input_node:
        Optional input node override. If None, DistributedHBConfig.input_node is
        used.
    """

    kind: PumpDriveKind
    pump_frequency_hz: float
    value: float
    source_impedance_ohm: float = 50.0
    pump_label: str = "pump"
    phase_rad: float = 0.0
    input_node: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PumpDriveKind(self.kind))
        if self.pump_frequency_hz <= 0.0:
            raise ValueError("pump_frequency_hz must be positive")
        if self.source_impedance_ohm <= 0.0:
            raise ValueError("source_impedance_ohm must be positive")
        if self.value < 0.0 and self.kind != PumpDriveKind.AVAILABLE_POWER_DBM:
            raise ValueError("drive value must be non-negative for current/power-W kinds")
        if self.input_node is not None and int(self.input_node) < 0:
            raise ValueError("input_node must be non-negative if provided")
        if self.input_node is not None:
            object.__setattr__(self, "input_node", int(self.input_node))

    @classmethod
    def from_current_rms(
        cls,
        *,
        pump_frequency_hz: float,
        current_rms_A: float,
        source_impedance_ohm: float = 50.0,
        pump_label: str = "pump",
        phase_rad: float = 0.0,
        input_node: int | None = None,
    ) -> "PumpDriveConfig":
        return cls(
            kind=PumpDriveKind.CURRENT_RMS,
            pump_frequency_hz=pump_frequency_hz,
            value=current_rms_A,
            source_impedance_ohm=source_impedance_ohm,
            pump_label=pump_label,
            phase_rad=phase_rad,
            input_node=input_node,
        )

    @classmethod
    def from_available_power_dbm(
        cls,
        *,
        pump_frequency_hz: float,
        power_dbm: float,
        source_impedance_ohm: float = 50.0,
        pump_label: str = "pump",
        phase_rad: float = 0.0,
        input_node: int | None = None,
    ) -> "PumpDriveConfig":
        return cls(
            kind=PumpDriveKind.AVAILABLE_POWER_DBM,
            pump_frequency_hz=pump_frequency_hz,
            value=power_dbm,
            source_impedance_ohm=source_impedance_ohm,
            pump_label=pump_label,
            phase_rad=phase_rad,
            input_node=input_node,
        )

    @classmethod
    def from_available_power_watt(
        cls,
        *,
        pump_frequency_hz: float,
        power_W: float,
        source_impedance_ohm: float = 50.0,
        pump_label: str = "pump",
        phase_rad: float = 0.0,
        input_node: int | None = None,
    ) -> "PumpDriveConfig":
        return cls(
            kind=PumpDriveKind.AVAILABLE_POWER_W,
            pump_frequency_hz=pump_frequency_hz,
            value=power_W,
            source_impedance_ohm=source_impedance_ohm,
            pump_label=pump_label,
            phase_rad=phase_rad,
            input_node=input_node,
        )

    @property
    def available_power_W(self) -> float:
        if self.kind == PumpDriveKind.CURRENT_RMS:
            return float(
                available_power_from_norton_current_rms(
                    self.value,
                    source_impedance_ohm=self.source_impedance_ohm,
                )
            )
        if self.kind == PumpDriveKind.AVAILABLE_POWER_DBM:
            return float(dbm_to_watt(self.value))
        if self.kind == PumpDriveKind.AVAILABLE_POWER_W:
            return float(self.value)
        raise ValueError(f"Unsupported drive kind {self.kind}")

    @property
    def available_power_dbm(self) -> float:
        return float(watt_to_dbm(self.available_power_W))

    @property
    def current_rms_A(self) -> float:
        if self.kind == PumpDriveKind.CURRENT_RMS:
            return float(self.value)
        return float(
            norton_current_rms_from_available_power(
                self.available_power_W,
                source_impedance_ohm=self.source_impedance_ohm,
            )
        )

    @property
    def current_rms_phasor_A(self) -> complex:
        return complex(self.current_rms_A * jnp.exp(1j * self.phase_rad))

    def with_value(self, value: float, *, kind: PumpDriveKind | None = None) -> "PumpDriveConfig":
        """
        Return copy with a new drive value and optional kind.
        """
        return replace(self, value=float(value), kind=self.kind if kind is None else PumpDriveKind(kind))

    def with_current_rms(self, current_rms_A: float) -> "PumpDriveConfig":
        return replace(self, kind=PumpDriveKind.CURRENT_RMS, value=float(current_rms_A))

    def with_available_power_dbm(self, power_dbm: float) -> "PumpDriveConfig":
        return replace(self, kind=PumpDriveKind.AVAILABLE_POWER_DBM, value=float(power_dbm))

    def with_available_power_W(self, power_W: float) -> "PumpDriveConfig":
        return replace(self, kind=PumpDriveKind.AVAILABLE_POWER_W, value=float(power_W))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "pump_frequency_hz": self.pump_frequency_hz,
            "pump_frequency_GHz": self.pump_frequency_hz / 1e9,
            "value": self.value,
            "source_impedance_ohm": self.source_impedance_ohm,
            "pump_label": self.pump_label,
            "phase_rad": self.phase_rad,
            "input_node": self.input_node,
            "current_rms_A": self.current_rms_A,
            "available_power_W": self.available_power_W,
            "available_power_dbm": self.available_power_dbm,
        }


@dataclass(frozen=True)
class PumpHBLadderConfig:
    """
    High-level pump HB ladder configuration.

    Parameters
    ----------
    n_pump_harmonics:
        Number of pump harmonics included in the pump-only plan.
    include_negative_frequencies:
        Whether to include negative-frequency conjugate tones.
    include_dc:
        Whether to include DC tone.
    distributed:
        Distributed HB boundary/source/termination configuration.
    projection:
        HB projection configuration.
    solver:
        Dense solver configuration for the reference backend.
    name:
        Human-readable run name.
    """

    n_pump_harmonics: int = 3
    include_negative_frequencies: bool = True
    include_dc: bool = False
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
    name: str = "pump_hb_ladder"

    def __post_init__(self) -> None:
        if int(self.n_pump_harmonics) <= 0:
            raise ValueError("n_pump_harmonics must be positive")
        object.__setattr__(self, "n_pump_harmonics", int(self.n_pump_harmonics))
        if int(self.max_dense_real_unknowns) <= 0:
            raise ValueError("max_dense_real_unknowns must be positive")
        object.__setattr__(self, "max_dense_real_unknowns", int(self.max_dense_real_unknowns))

    def with_updates(self, **kwargs: Any) -> "PumpHBLadderConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pump_harmonics": self.n_pump_harmonics,
            "include_negative_frequencies": self.include_negative_frequencies,
            "include_dc": self.include_dc,
            "distributed": self.distributed.to_dict(),
            "projection": self.projection.to_dict(),
            "solver": self.solver.to_dict(),
            "max_dense_real_unknowns": self.max_dense_real_unknowns,
            "name": self.name,
        }


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PumpProfileSummary:
    """
    Summary of solved pump voltage/current profiles.
    """

    max_node_voltage_abs_V: float
    max_branch_current_abs_A: float
    max_branch_current_peak_time_A: float
    max_pump_current_ratio: float
    output_to_input_voltage_ratio: complex
    output_to_input_voltage_gain_db: float
    input_pump_voltage_abs_V: float
    output_pump_voltage_abs_V: float
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_node_voltage_abs_V": self.max_node_voltage_abs_V,
            "max_branch_current_abs_A": self.max_branch_current_abs_A,
            "max_branch_current_peak_time_A": self.max_branch_current_peak_time_A,
            "max_pump_current_ratio": self.max_pump_current_ratio,
            "output_to_input_voltage_ratio": {
                "real": float(jnp.real(self.output_to_input_voltage_ratio)),
                "imag": float(jnp.imag(self.output_to_input_voltage_ratio)),
                "abs": float(abs(self.output_to_input_voltage_ratio)),
            },
            "output_to_input_voltage_gain_db": self.output_to_input_voltage_gain_db,
            "input_pump_voltage_abs_V": self.input_pump_voltage_abs_V,
            "output_pump_voltage_abs_V": self.output_pump_voltage_abs_V,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class PumpHBLadderResult:
    """
    High-level pump HB result.

    Attributes
    ----------
    distributed_result:
        Underlying distributed HB result.
    drive:
        Pump drive configuration.
    pump_config:
        High-level pump-HB config.
    profile:
        Pump profile summary.
    metadata:
        Extra metadata.
    """

    distributed_result: DistributedHBSolveResult
    drive: PumpDriveConfig
    pump_config: PumpHBLadderConfig
    profile: PumpProfileSummary
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.distributed_result.converged

    @property
    def status(self) -> PumpHBStatus:
        return PumpHBStatus.CONVERGED if self.converged else PumpHBStatus.FAILED

    @property
    def state(self) -> DistributedHBState:
        return self.distributed_result.state

    @property
    def residual(self) -> DistributedHBResidual:
        return self.distributed_result.residual

    @property
    def solver_result(self) -> HBSolverResult:
        return self.distributed_result.solver_result

    @property
    def frequency_plan(self) -> FrequencyPlan:
        return self.distributed_result.frequency_plan

    @property
    def layout(self) -> LineLayout:
        return self.distributed_result.layout

    @property
    def ki_model(self) -> KineticInductanceModel:
        return self.distributed_result.ki_model

    def pump_tone_index(self) -> int:
        return self.frequency_plan.position_of_label(self.drive.pump_label)

    def input_pump_voltage(self) -> complex:
        idx = self.pump_tone_index()
        return complex(self.distributed_result.input_voltage_coeffs_V[idx])

    def output_pump_voltage(self) -> complex:
        idx = self.pump_tone_index()
        return complex(self.distributed_result.output_voltage_coeffs_V[idx])

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "converged": self.converged,
            "drive": self.drive.to_dict(),
            "pump_config": self.pump_config.to_dict(),
            "profile": self.profile.to_dict(),
            "distributed_result": self.distributed_result.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class PumpContinuationResult:
    """
    Pump continuation result.

    The underlying ContinuationResult stores accepted states/residuals. This
    wrapper stores the pump-drive schedule and metadata needed to interpret it.
    """

    continuation: ContinuationResult
    base_drive: PumpDriveConfig
    continuation_kind: PumpContinuationKind
    pump_config: PumpHBLadderConfig
    layout_summary: Mapping[str, Any]
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.continuation.converged

    @property
    def values(self) -> tuple[float, ...]:
        return self.continuation.values

    @property
    def final_state(self) -> DistributedHBState | None:
        return self.continuation.last_solution

    def value_to_drive(self, value: float) -> PumpDriveConfig:
        if self.continuation_kind == PumpContinuationKind.CURRENT_RMS_A:
            return self.base_drive.with_current_rms(value)
        if self.continuation_kind == PumpContinuationKind.AVAILABLE_POWER_DBM:
            return self.base_drive.with_available_power_dbm(value)
        if self.continuation_kind == PumpContinuationKind.AVAILABLE_POWER_W:
            return self.base_drive.with_available_power_W(value)
        raise ValueError(f"Unsupported continuation kind {self.continuation_kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "continuation_kind": self.continuation_kind.value,
            "base_drive": self.base_drive.to_dict(),
            "pump_config": self.pump_config.to_dict(),
            "layout_summary": dict(self.layout_summary),
            "continuation": self.continuation.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Plan/source construction
# ---------------------------------------------------------------------------

def make_pump_plan_from_drive(
    drive: PumpDriveConfig,
    *,
    n_harmonics: int = 3,
    include_negative: bool = True,
    include_dc: bool = False,
) -> FrequencyPlan:
    """
    Build a pump-only FrequencyPlan from PumpDriveConfig.
    """
    return make_pump_only_plan(
        drive.pump_frequency_hz,
        n_harmonics=n_harmonics,
        include_negative=include_negative,
        include_dc=include_dc,
        sort="frequency" if include_negative else "abs_frequency",
    )


def make_pump_plan_from_config(
    drive: PumpDriveConfig,
    config: PumpHBLadderConfig,
) -> FrequencyPlan:
    """
    Build pump-only FrequencyPlan from high-level pump config.
    """
    return make_pump_plan_from_drive(
        drive,
        n_harmonics=config.n_pump_harmonics,
        include_negative=config.include_negative_frequencies,
        include_dc=config.include_dc,
    )


def make_pump_injection(
    plan: FrequencyPlan,
    layout: LineLayout,
    drive: PumpDriveConfig,
    distributed_config: DistributedHBConfig,
) -> jax.Array:
    """
    Build node current injection array for the pump drive.
    """
    input_node = distributed_config.input_node if drive.input_node is None else drive.input_node

    return make_input_current_injection_at_node(
        plan,
        layout,
        node=input_node,
        pump_label=drive.pump_label,
        pump_current_rms_A=drive.current_rms_phasor_A,
    )


def make_input_current_injection_at_node(
    plan: FrequencyPlan,
    layout: LineLayout,
    *,
    node: int,
    pump_label: str,
    pump_current_rms_A: complex,
) -> jax.Array:
    """
    Current injection helper independent of DistributedHBConfig.input_node.
    """
    from twpa.nonlinear.distributed_hb import make_node_current_injection_from_rms_phasor

    return make_node_current_injection_from_rms_phasor(
        plan,
        layout,
        node=node,
        label=pump_label,
        rms_current_A=pump_current_rms_A,
        set_conjugate=True,
    )


# ---------------------------------------------------------------------------
# Profile summaries
# ---------------------------------------------------------------------------

def summarize_pump_profile(
    result: DistributedHBSolveResult,
    drive: PumpDriveConfig,
    *,
    projection_grid: HBProjectionGrid | None = None,
) -> PumpProfileSummary:
    """
    Build a compact physical summary of the pump solution.
    """
    plan = result.frequency_plan
    state = result.state
    idx = plan.position_of_label(drive.pump_label)

    vin = state.node_voltage_coeffs_V[idx, result.input_node]
    vout = state.node_voltage_coeffs_V[idx, result.output_node]

    ratio = vout / jnp.where(jnp.abs(vin) > 1e-300, vin, 1e-300 + 0j)
    gain_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(ratio), 1e-300))

    grid = projection_grid
    if grid is None:
        grid = make_projection_grid_from_plan(
            plan,
            fundamental_frequency_hz=plan.reference_pump_hz,
            config=HBProjectionConfig(
                n_time_samples=None,
                oversampling=8,
                force_real_time_signal=True,
                enforce_conjugate_symmetry=True,
            ),
        )

    branch_current_t = coefficients_to_time(
        state.branch_current_coeffs_A,
        plan.frequencies_hz,
        grid.t_s,
        force_real=True,
    )
    max_current_time = float(jnp.max(jnp.abs(branch_current_t)))

    Istar = jnp.asarray(result.ki_model.I_star_A, dtype=jnp.float64)
    if Istar.ndim == 0:
        Istar = jnp.full((state.branch_current_coeffs_A.shape[1],), Istar)
    ratio_current = jnp.max(
        jnp.abs(branch_current_t) / jnp.maximum(Istar[None, :], 1e-300)
    )

    return PumpProfileSummary(
        max_node_voltage_abs_V=float(jnp.max(jnp.abs(state.node_voltage_coeffs_V))),
        max_branch_current_abs_A=float(jnp.max(jnp.abs(state.branch_current_coeffs_A))),
        max_branch_current_peak_time_A=max_current_time,
        max_pump_current_ratio=float(ratio_current),
        output_to_input_voltage_ratio=complex(ratio),
        output_to_input_voltage_gain_db=float(gain_db),
        input_pump_voltage_abs_V=float(jnp.abs(vin)),
        output_pump_voltage_abs_V=float(jnp.abs(vout)),
        metadata={
            "pump_label": drive.pump_label,
            "pump_tone_index": idx,
            "projection_grid": grid.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Main solve functions
# ---------------------------------------------------------------------------

def solve_pump_hb_ladder(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig | None = None,
    plan: FrequencyPlan | None = None,
    x0: DistributedHBState | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PumpHBLadderResult:
    """
    Solve pump-only distributed HB for a TWPA ladder.

    Either pass nonlinear_params or a prebuilt branch-wise ki_model.
    """
    cfg = pump_config or PumpHBLadderConfig()

    if plan is None:
        plan = make_pump_plan_from_config(drive, cfg)

    n_real_unknowns = 2 * plan.n_tones * (2 * layout.n_cells + 1)
    solver_backend = (
        cfg.solver.backend
        if isinstance(cfg.solver, SolverConfig)
        else SolverBackend.DENSE
    )
    if solver_backend != SolverBackend.NEWTON_KRYLOV and n_real_unknowns > cfg.max_dense_real_unknowns:
        jacobian_gib = (n_real_unknowns**2 * 8) / (1024**3)
        raise RuntimeError(
            "Dense reference pump HB refused unsafe problem size: "
            f"{n_real_unknowns} real unknowns exceed max_dense_real_unknowns="
            f"{cfg.max_dense_real_unknowns}. One float64 Jacobian would require "
            f"about {jacobian_gib:.3f} GiB before solver copies. "
            "Coarsen the layout or implement a structured sparse/matrix-free backend."
        )

    if ki_model is None:
        if nonlinear_params is None:
            raise ValueError("Either nonlinear_params or ki_model must be provided")
        ki_model = make_kinetic_model_from_layout(
            layout,
            nonlinear_params,
            name=f"{layout.name}_pump_ki_model",
        )

    injection = make_pump_injection(
        plan,
        layout,
        drive,
        cfg.distributed,
    )

    grid = make_projection_grid_from_plan(
        plan,
        fundamental_frequency_hz=plan.reference_pump_hz,
        config=cfg.projection,
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
            "driver": "solve_pump_hb_ladder",
            "drive": drive.to_dict(),
            "pump_config": cfg.to_dict(),
            **dict(metadata or {}),
        },
    )

    profile = summarize_pump_profile(
        distributed_result,
        drive,
        projection_grid=grid,
    )

    return PumpHBLadderResult(
        distributed_result=distributed_result,
        drive=drive,
        pump_config=cfg,
        profile=profile,
        metadata={
            "projection_grid": grid.to_dict(),
            **dict(metadata or {}),
        },
    )


def solve_pump_hb_ladder_from_power_dbm(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    pump_frequency_hz: float,
    pump_power_dbm: float,
    source_impedance_ohm: float = 50.0,
    pump_config: PumpHBLadderConfig | None = None,
    plan: FrequencyPlan | None = None,
    x0: DistributedHBState | None = None,
) -> PumpHBLadderResult:
    """
    Convenience pump solve from available pump power in dBm.
    """
    drive = PumpDriveConfig.from_available_power_dbm(
        pump_frequency_hz=pump_frequency_hz,
        power_dbm=pump_power_dbm,
        source_impedance_ohm=source_impedance_ohm,
    )
    return solve_pump_hb_ladder(
        layout,
        nonlinear_params,
        ki_model=ki_model,
        drive=drive,
        pump_config=pump_config,
        plan=plan,
        x0=x0,
    )


def solve_pump_hb_ladder_from_current_rms(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    pump_frequency_hz: float,
    pump_current_rms_A: float,
    source_impedance_ohm: float = 50.0,
    pump_config: PumpHBLadderConfig | None = None,
    plan: FrequencyPlan | None = None,
    x0: DistributedHBState | None = None,
) -> PumpHBLadderResult:
    """
    Convenience pump solve from equivalent Norton RMS current.
    """
    drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=pump_frequency_hz,
        current_rms_A=pump_current_rms_A,
        source_impedance_ohm=source_impedance_ohm,
    )
    return solve_pump_hb_ladder(
        layout,
        nonlinear_params,
        ki_model=ki_model,
        drive=drive,
        pump_config=pump_config,
        plan=plan,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Continuation
# ---------------------------------------------------------------------------

def _drive_from_continuation_value(
    base_drive: PumpDriveConfig,
    value: float,
    kind: PumpContinuationKind,
) -> PumpDriveConfig:
    if kind == PumpContinuationKind.CURRENT_RMS_A:
        return base_drive.with_current_rms(value)
    if kind == PumpContinuationKind.AVAILABLE_POWER_DBM:
        return base_drive.with_available_power_dbm(value)
    if kind == PumpContinuationKind.AVAILABLE_POWER_W:
        return base_drive.with_available_power_W(value)
    raise ValueError(f"Unsupported continuation kind {kind}")


def solve_pump_hb_continuation(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    base_drive: PumpDriveConfig,
    start: float,
    stop: float,
    n_steps: int,
    continuation_kind: PumpContinuationKind = PumpContinuationKind.AVAILABLE_POWER_DBM,
    pump_config: PumpHBLadderConfig | None = None,
    continuation_config: ContinuationSolverConfig | None = None,
    plan: FrequencyPlan | None = None,
    x0: DistributedHBState | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PumpContinuationResult:
    """
    Run pump HB continuation over current, power dBm, or power W.

    The continuation solution state is DistributedHBState.
    """
    cfg = pump_config or PumpHBLadderConfig()
    cont_kind = PumpContinuationKind(continuation_kind)

    if plan is None:
        plan = make_pump_plan_from_config(base_drive, cfg)

    if ki_model is None:
        if nonlinear_params is None:
            raise ValueError("Either nonlinear_params or ki_model must be provided")
        ki_model = make_kinetic_model_from_layout(
            layout,
            nonlinear_params,
            name=f"{layout.name}_pump_continuation_ki_model",
        )

    grid = make_projection_grid_from_plan(
        plan,
        fundamental_frequency_hz=plan.reference_pump_hz,
        config=cfg.projection,
    )

    if x0 is None:
        first_drive = _drive_from_continuation_value(base_drive, start, cont_kind)
        first_injection = make_pump_injection(plan, layout, first_drive, cfg.distributed)
        x0 = make_distributed_linear_initial_guess(
            plan,
            layout,
            cfg.distributed,
            first_injection,
        )

    def residual_factory(
        value: float,
        guess: DistributedHBState,
        context: Mapping[str, Any],
    ) -> Any:
        local_drive = _drive_from_continuation_value(base_drive, value, cont_kind)
        injection = make_pump_injection(plan, layout, local_drive, cfg.distributed)
        return make_distributed_hb_residual_function(
            plan,
            layout,
            cfg.distributed,
            ki_model,
            injection,
            projection_grid=grid,
            projection_config=cfg.projection,
        )

    schedule = make_continuation_schedule(
        start=start,
        stop=stop,
        n_steps=n_steps,
        kind="linear",
    )

    continuation = solve_continuation(
        schedule=schedule,
        residual_factory=residual_factory,
        x0=x0,
        solver_config=cfg.solver,
        continuation_config=continuation_config,
        context={
            "driver": "solve_pump_hb_continuation",
            "continuation_kind": cont_kind.value,
            "layout_name": layout.name,
            "n_cells": layout.n_cells,
            "pump_frequency_hz": base_drive.pump_frequency_hz,
        },
        metadata={
            "base_drive": base_drive.to_dict(),
            "pump_config": cfg.to_dict(),
            "projection_grid": grid.to_dict(),
            **dict(metadata or {}),
        },
    )

    return PumpContinuationResult(
        continuation=continuation,
        base_drive=base_drive,
        continuation_kind=cont_kind,
        pump_config=cfg,
        layout_summary=layout.summary(),
        metadata={
            "projection_grid": grid.to_dict(),
            "plan": plan.to_dict(),
            **dict(metadata or {}),
        },
    )


def solve_pump_power_dbm_continuation(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    pump_frequency_hz: float,
    start_power_dbm: float,
    stop_power_dbm: float,
    n_steps: int = 11,
    source_impedance_ohm: float = 50.0,
    pump_config: PumpHBLadderConfig | None = None,
    continuation_config: ContinuationSolverConfig | None = None,
) -> PumpContinuationResult:
    """
    Convenience continuation over available pump power in dBm.
    """
    base_drive = PumpDriveConfig.from_available_power_dbm(
        pump_frequency_hz=pump_frequency_hz,
        power_dbm=start_power_dbm,
        source_impedance_ohm=source_impedance_ohm,
    )
    return solve_pump_hb_continuation(
        layout,
        nonlinear_params,
        ki_model=ki_model,
        base_drive=base_drive,
        start=start_power_dbm,
        stop=stop_power_dbm,
        n_steps=n_steps,
        continuation_kind=PumpContinuationKind.AVAILABLE_POWER_DBM,
        pump_config=pump_config,
        continuation_config=continuation_config,
    )


def solve_pump_current_continuation(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    pump_frequency_hz: float,
    start_current_rms_A: float,
    stop_current_rms_A: float,
    n_steps: int = 11,
    source_impedance_ohm: float = 50.0,
    pump_config: PumpHBLadderConfig | None = None,
    continuation_config: ContinuationSolverConfig | None = None,
) -> PumpContinuationResult:
    """
    Convenience continuation over equivalent Norton pump RMS current.
    """
    base_drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=pump_frequency_hz,
        current_rms_A=start_current_rms_A,
        source_impedance_ohm=source_impedance_ohm,
    )
    return solve_pump_hb_continuation(
        layout,
        nonlinear_params,
        ki_model=ki_model,
        base_drive=base_drive,
        start=start_current_rms_A,
        stop=stop_current_rms_A,
        n_steps=n_steps,
        continuation_kind=PumpContinuationKind.CURRENT_RMS_A,
        pump_config=pump_config,
        continuation_config=continuation_config,
    )


# ---------------------------------------------------------------------------
# Sweep utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PumpFrequencySweepPoint:
    """One pump-frequency sweep point."""

    pump_frequency_hz: float
    result: PumpHBLadderResult

    @property
    def converged(self) -> bool:
        return self.result.converged

    def to_dict(self) -> dict[str, Any]:
        return {
            "pump_frequency_hz": self.pump_frequency_hz,
            "pump_frequency_GHz": self.pump_frequency_hz / 1e9,
            "converged": self.converged,
            "profile": self.result.profile.to_dict(),
            "solver": self.result.solver_result.report.to_dict(),
        }


@dataclass(frozen=True)
class PumpFrequencySweepResult:
    """Pump-frequency sweep result."""

    points: tuple[PumpFrequencySweepPoint, ...]
    metadata: Mapping[str, Any] | None = None

    @property
    def converged_count(self) -> int:
        return sum(1 for p in self.points if p.converged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": len(self.points),
            "converged_count": self.converged_count,
            "points": [p.to_dict() for p in self.points],
            "metadata": dict(self.metadata or {}),
        }


def sweep_pump_frequency(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    ki_model: KineticInductanceModel | None = None,
    pump_frequencies_hz: Sequence[float],
    pump_power_dbm: float | None = None,
    pump_current_rms_A: float | None = None,
    source_impedance_ohm: float = 50.0,
    pump_config: PumpHBLadderConfig | None = None,
    reuse_previous_solution: bool = True,
) -> PumpFrequencySweepResult:
    """
    Sweep pump frequency at fixed pump power or current.

    Exactly one of pump_power_dbm or pump_current_rms_A must be provided.
    """
    if (pump_power_dbm is None) == (pump_current_rms_A is None):
        raise ValueError("Provide exactly one of pump_power_dbm or pump_current_rms_A")

    cfg = pump_config or PumpHBLadderConfig()
    points: list[PumpFrequencySweepPoint] = []
    x0: DistributedHBState | None = None

    for fp in pump_frequencies_hz:
        if pump_power_dbm is not None:
            drive = PumpDriveConfig.from_available_power_dbm(
                pump_frequency_hz=float(fp),
                power_dbm=float(pump_power_dbm),
                source_impedance_ohm=source_impedance_ohm,
            )
        else:
            assert pump_current_rms_A is not None
            drive = PumpDriveConfig.from_current_rms(
                pump_frequency_hz=float(fp),
                current_rms_A=float(pump_current_rms_A),
                source_impedance_ohm=source_impedance_ohm,
            )

        result = solve_pump_hb_ladder(
            layout,
            nonlinear_params,
            ki_model=ki_model,
            drive=drive,
            pump_config=cfg,
            x0=x0,
            metadata={"sweep": "pump_frequency"},
        )
        points.append(PumpFrequencySweepPoint(pump_frequency_hz=float(fp), result=result))

        if reuse_previous_solution and result.converged:
            x0 = result.state

    return PumpFrequencySweepResult(
        points=tuple(points),
        metadata={
            "layout": layout.summary(),
            "pump_power_dbm": pump_power_dbm,
            "pump_current_rms_A": pump_current_rms_A,
            "source_impedance_ohm": source_impedance_ohm,
            "pump_config": cfg.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Validation / reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PumpHBValidationReport:
    """
    Pump HB validation report.
    """

    passed: bool
    messages: list[str]
    linear_initial_residual_norm: float
    solved_residual_norm: float | None
    solve_converged: bool | None
    profile: Mapping[str, Any] | None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "messages": list(self.messages),
            "linear_initial_residual_norm": self.linear_initial_residual_norm,
            "solved_residual_norm": self.solved_residual_norm,
            "solve_converged": self.solve_converged,
            "profile": None if self.profile is None else dict(self.profile),
            "metadata": dict(self.metadata or {}),
        }


def validate_pump_hb_linear_initial_guess(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    *,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig | None = None,
) -> PumpHBValidationReport:
    """
    Validate that the linear initial guess produces small residual at tiny drive.

    Use this with very small drive values.
    """
    cfg = pump_config or PumpHBLadderConfig()
    plan = make_pump_plan_from_config(drive, cfg)
    model = make_kinetic_model_from_layout(layout, nonlinear_params)
    injection = make_pump_injection(plan, layout, drive, cfg.distributed)

    x0 = make_distributed_linear_initial_guess(
        plan,
        layout,
        cfg.distributed,
        injection,
    )

    grid = make_projection_grid_from_plan(
        plan,
        fundamental_frequency_hz=plan.reference_pump_hz,
        config=cfg.projection,
    )
    residual = evaluate_distributed_hb_residual(
        x0,
        plan,
        layout,
        cfg.distributed,
        model,
        injection,
        projection_grid=grid,
        projection_config=cfg.projection,
    )

    res_norm = residual.norm
    passed = bool(res_norm < 1e-8)

    messages = []
    if passed:
        messages.append("PASS: pump HB linear initial guess residual is small.")
    else:
        messages.append(f"FAIL: pump HB linear initial guess residual is {res_norm:.3e}.")

    return PumpHBValidationReport(
        passed=passed,
        messages=messages,
        linear_initial_residual_norm=res_norm,
        solved_residual_norm=None,
        solve_converged=None,
        profile=None,
        metadata={
            "layout": layout.summary(),
            "drive": drive.to_dict(),
            "pump_config": cfg.to_dict(),
            "x0": x0.summary(),
            "residual": residual.summary(),
        },
    )


def validate_pump_hb_smoke(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    *,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig | None = None,
    residual_tol: float = 1e-6,
) -> PumpHBValidationReport:
    """
    Run a compact pump-HB smoke solve on a small/reduced layout.
    """
    cfg = pump_config or PumpHBLadderConfig()
    result = solve_pump_hb_ladder(
        layout,
        nonlinear_params,
        drive=drive,
        pump_config=cfg,
        metadata={"validation": "pump_hb_smoke"},
    )

    solved_norm = result.residual.norm
    passed = bool(result.converged and solved_norm < residual_tol)

    messages = []
    if passed:
        messages.append("PASS: pump HB smoke solve converged.")
    else:
        messages.append(
            f"FAIL: pump HB smoke solve failed or residual {solved_norm:.3e} "
            f"exceeds {residual_tol:.3e}."
        )

    return PumpHBValidationReport(
        passed=passed,
        messages=messages,
        linear_initial_residual_norm=float("nan"),
        solved_residual_norm=solved_norm,
        solve_converged=result.converged,
        profile=result.profile.to_dict(),
        metadata={
            "result": result.to_dict(),
        },
    )


def run_pump_hb_self_checks(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    *,
    pump_frequency_hz: float,
) -> dict[str, Any]:
    """
    Compact pump-HB self-check suite for a small/reduced layout.
    """
    tiny_drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=pump_frequency_hz,
        current_rms_A=1e-12,
    )
    smoke_drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=pump_frequency_hz,
        current_rms_A=1e-8,
    )

    linear = validate_pump_hb_linear_initial_guess(
        layout,
        nonlinear_params,
        drive=tiny_drive,
    )
    smoke = validate_pump_hb_smoke(
        layout,
        nonlinear_params,
        drive=smoke_drive,
    )

    return {
        "passed": bool(linear.passed and smoke.passed),
        "linear_initial_guess": linear.to_dict(),
        "smoke": smoke.to_dict(),
    }


def pump_solution_table(result: PumpHBLadderResult) -> str:
    """
    Short markdown table for a solved pump result.
    """
    p = result.profile
    d = result.drive
    lines = [
        "| quantity | value |",
        "|---|---:|",
        f"| converged | {result.converged} |",
        f"| pump frequency GHz | {d.pump_frequency_hz / 1e9:.9g} |",
        f"| available power dBm | {d.available_power_dbm:.9g} |",
        f"| Norton current RMS A | {d.current_rms_A:.9e} |",
        f"| max node voltage abs V | {p.max_node_voltage_abs_V:.9e} |",
        f"| max branch current peak A | {p.max_branch_current_peak_time_A:.9e} |",
        f"| max I/Istar | {p.max_pump_current_ratio:.9e} |",
        f"| output/input pump voltage dB | {p.output_to_input_voltage_gain_db:.9g} |",
        f"| final residual norm | {result.residual.norm:.9e} |",
    ]
    return "\n".join(lines)


__all__ = [
    "PumpDriveKind",
    "PumpContinuationKind",
    "PumpHBStatus",
    "dbm_to_watt",
    "watt_to_dbm",
    "norton_current_rms_from_available_power",
    "available_power_from_norton_current_rms",
    "dbm_to_norton_current_rms",
    "norton_current_rms_to_dbm",
    "PumpDriveConfig",
    "PumpHBLadderConfig",
    "PumpProfileSummary",
    "PumpHBLadderResult",
    "PumpContinuationResult",
    "make_pump_plan_from_drive",
    "make_pump_plan_from_config",
    "make_pump_injection",
    "make_input_current_injection_at_node",
    "summarize_pump_profile",
    "solve_pump_hb_ladder",
    "solve_pump_hb_ladder_from_power_dbm",
    "solve_pump_hb_ladder_from_current_rms",
    "solve_pump_hb_continuation",
    "solve_pump_power_dbm_continuation",
    "solve_pump_current_continuation",
    "PumpFrequencySweepPoint",
    "PumpFrequencySweepResult",
    "sweep_pump_frequency",
    "PumpHBValidationReport",
    "validate_pump_hb_linear_initial_guess",
    "validate_pump_hb_smoke",
    "run_pump_hb_self_checks",
    "pump_solution_table",
]
