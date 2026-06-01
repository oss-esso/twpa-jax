"""
twpa.workflows.calibration
==========================

Parameter-extraction and calibration workflow for KI-TWPA simulations.

This module connects measured data to the simulator stack:

    measured pump-off S-parameters
        -> fit linear layout parameters

    measured pump-on gain / conversion
        -> fit nonlinear parameters and pump operating point

The calibration layer is deliberately workflow-oriented. It does not introduce
new circuit physics; instead, it repeatedly calls:

    twpa.linear.cascade
    twpa.nonlinear.pump_hb_ladder
    twpa.nonlinear.gain

and compares simulated observables with experimental data.

Supported calibration targets
-----------------------------
1. Pump-off S-parameter data:
       S11, S21, S12, S22 complex data and/or S21 in dB.

2. Pump-on gain data:
       labelled signal gain and optional idler conversion data.

3. User-defined residual hooks:
       arbitrary residual vectors supplied by the caller.

Optimization philosophy
-----------------------
The first production target is robustness, not cleverness. This file provides:

    - parameter transforms and bounds,
    - weighted residual construction,
    - SciPy least_squares when SciPy is available,
    - a derivative-free coordinate-search fallback,
    - finite-difference Jacobian diagnostics,
    - JSON/NPZ artifact export.

The low-level simulations remain JAX-backed. Calibration itself may use SciPy
or derivative-free Python loops because many workflow calls contain Python
objects, validation, and nonlinear solves that are not always JIT-traceable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import json

import jax
import jax.numpy as jnp
import numpy as np

try:
    from scipy.optimize import least_squares as scipy_least_squares

    SCIPY_AVAILABLE = True
except Exception:
    scipy_least_squares = None
    SCIPY_AVAILABLE = False

from twpa.core.layout import LineLayout, make_layout_from_arrays
from twpa.core.params import NonlinearParams
from twpa.linear.cascade import CascadeConfig, run_linear_scan
from twpa.linear.cells import CellModelConfig
from twpa.nonlinear.gain import (
    GainSweepConfig,
    GainSweepResult,
    extract_labelled_gain_trace,
    solve_gain_sweep_from_pump,
)
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    solve_pump_hb_ladder,
)


ArrayLike = Any
ParameterDict = dict[str, float]
ResidualHook = Callable[[ParameterDict], ArrayLike]
TargetPlanFactory = Callable[[Any], Any]
SweepConfigFactory = Callable[[Any], GainSweepConfig]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _jsonify(obj: Any) -> Any:
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
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
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
# Parameter transforms
# ---------------------------------------------------------------------------

class ParameterTransform(str, Enum):
    """Parameter vector transform."""

    LINEAR = "linear"
    LOG = "log"
    LOG10 = "log10"


@dataclass(frozen=True)
class CalibrationParameterSpec:
    """
    One calibrated scalar parameter.

    Parameters are optimized in transformed coordinates but decoded into
    physical coordinates before simulation.

    Examples
    --------
    L_scale:
        initial=1.0, lower=0.8, upper=1.2, transform="log"

    I_star_scale:
        initial=1.0, lower=0.2, upper=5.0, transform="log"

    pump_power_offset_db:
        initial=0.0, lower=-3.0, upper=3.0, transform="linear"
    """

    name: str
    initial: float
    lower: float
    upper: float
    transform: ParameterTransform = ParameterTransform.LINEAR
    enabled: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", ParameterTransform(self.transform))
        if self.lower > self.upper:
            raise ValueError(f"{self.name}: lower bound exceeds upper bound")
        if not (self.lower <= self.initial <= self.upper):
            raise ValueError(
                f"{self.name}: initial value {self.initial} outside "
                f"[{self.lower}, {self.upper}]"
            )
        if self.transform in {ParameterTransform.LOG, ParameterTransform.LOG10}:
            if self.lower <= 0.0 or self.initial <= 0.0 or self.upper <= 0.0:
                raise ValueError(f"{self.name}: log transforms require positive values")

    def encode(self, value: float) -> float:
        if self.transform == ParameterTransform.LINEAR:
            return float(value)
        if self.transform == ParameterTransform.LOG:
            return float(np.log(value))
        if self.transform == ParameterTransform.LOG10:
            return float(np.log10(value))
        raise ValueError(f"Unsupported transform {self.transform}")

    def decode(self, encoded_value: float) -> float:
        if self.transform == ParameterTransform.LINEAR:
            value = float(encoded_value)
        elif self.transform == ParameterTransform.LOG:
            value = float(np.exp(encoded_value))
        elif self.transform == ParameterTransform.LOG10:
            value = float(10.0 ** encoded_value)
        else:
            raise ValueError(f"Unsupported transform {self.transform}")

        return float(np.clip(value, self.lower, self.upper))

    @property
    def encoded_initial(self) -> float:
        return self.encode(self.initial)

    @property
    def encoded_lower(self) -> float:
        return self.encode(self.lower)

    @property
    def encoded_upper(self) -> float:
        return self.encode(self.upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial": self.initial,
            "lower": self.lower,
            "upper": self.upper,
            "transform": self.transform.value,
            "enabled": self.enabled,
            "description": self.description,
        }


@dataclass(frozen=True)
class CalibrationVectorSpec:
    """
    Ordered collection of calibration parameters.
    """

    parameters: tuple[CalibrationParameterSpec, ...]

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("Calibration parameter names must be unique")

    @property
    def enabled_parameters(self) -> tuple[CalibrationParameterSpec, ...]:
        return tuple(p for p in self.parameters if p.enabled)

    @property
    def enabled_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.enabled_parameters)

    def initial_physical_dict(self) -> ParameterDict:
        return {p.name: p.initial for p in self.parameters}

    def initial_vector(self) -> jax.Array:
        return jnp.asarray([p.encoded_initial for p in self.enabled_parameters], dtype=jnp.float64)

    def lower_vector(self) -> jax.Array:
        return jnp.asarray([p.encoded_lower for p in self.enabled_parameters], dtype=jnp.float64)

    def upper_vector(self) -> jax.Array:
        return jnp.asarray([p.encoded_upper for p in self.enabled_parameters], dtype=jnp.float64)

    def decode_vector(self, vector: ArrayLike) -> ParameterDict:
        v = np.asarray(vector, dtype=float)
        enabled = self.enabled_parameters
        if v.shape != (len(enabled),):
            raise ValueError(f"Expected vector shape {(len(enabled),)}, got {v.shape}")

        out = self.initial_physical_dict()
        for value, spec in zip(v.tolist(), enabled):
            out[spec.name] = spec.decode(value)
        return out

    def encode_dict(self, params: Mapping[str, float]) -> jax.Array:
        values = []
        for p in self.enabled_parameters:
            values.append(p.encode(float(params.get(p.name, p.initial))))
        return jnp.asarray(values, dtype=jnp.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": [p.to_dict() for p in self.parameters],
            "enabled_names": list(self.enabled_names),
        }


def make_default_linear_parameter_spec(
    *,
    include_loss: bool = True,
    include_stub: bool = True,
) -> CalibrationVectorSpec:
    """
    Default parameter set for pump-off S-parameter calibration.
    """
    params: list[CalibrationParameterSpec] = [
        CalibrationParameterSpec(
            "L_scale",
            initial=1.0,
            lower=0.5,
            upper=2.0,
            transform=ParameterTransform.LOG,
            description="Global series inductance scale.",
        ),
        CalibrationParameterSpec(
            "C_scale",
            initial=1.0,
            lower=0.5,
            upper=2.0,
            transform=ParameterTransform.LOG,
            description="Global base shunt capacitance scale.",
        ),
    ]

    if include_stub:
        params.append(
            CalibrationParameterSpec(
                "C_stub_scale",
                initial=1.0,
                lower=0.0,
                upper=5.0,
                transform=ParameterTransform.LINEAR,
                description="Global stub capacitance scale.",
            )
        )

    if include_loss:
        params += [
            CalibrationParameterSpec(
                "R_scale",
                initial=1.0,
                lower=0.0,
                upper=20.0,
                transform=ParameterTransform.LINEAR,
                description="Global series resistance scale.",
            ),
            CalibrationParameterSpec(
                "G_scale",
                initial=1.0,
                lower=0.0,
                upper=20.0,
                transform=ParameterTransform.LINEAR,
                description="Global shunt conductance scale.",
            ),
        ]

    return CalibrationVectorSpec(tuple(params))


def make_default_nonlinear_parameter_spec() -> CalibrationVectorSpec:
    """
    Default parameter set for pump-on nonlinear/gain calibration.
    """
    return CalibrationVectorSpec(
        (
            CalibrationParameterSpec(
                "I_star_scale",
                initial=1.0,
                lower=0.1,
                upper=10.0,
                transform=ParameterTransform.LOG,
                description="Global nonlinear current scale.",
            ),
            CalibrationParameterSpec(
                "beta_nl_scale",
                initial=1.0,
                lower=0.1,
                upper=10.0,
                transform=ParameterTransform.LOG,
                description="Global cubic nonlinearity coefficient scale.",
            ),
            CalibrationParameterSpec(
                "pump_current_scale",
                initial=1.0,
                lower=0.1,
                upper=10.0,
                transform=ParameterTransform.LOG,
                description="Effective pump current calibration scale.",
            ),
        )
    )


# ---------------------------------------------------------------------------
# Calibration data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SParameterCalibrationData:
    """
    Pump-off S-parameter calibration data.

    Parameters
    ----------
    frequency_hz:
        Measurement frequencies.
    s:
        Complex S-parameters with shape (F, 2, 2), optional.
    s21_db:
        Measured S21 magnitude in dB, optional.
    weight_complex:
        Weight for complex S-parameter residuals.
    weight_s21_db:
        Weight for S21 dB residuals.
    use_s11, use_s21, use_s12, use_s22:
        Which complex entries to include when s is provided.
    """

    frequency_hz: jax.Array
    s: jax.Array | None = None
    s21_db: jax.Array | None = None
    weight_complex: float = 1.0
    weight_s21_db: float = 1.0
    use_s11: bool = True
    use_s21: bool = True
    use_s12: bool = False
    use_s22: bool = True
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        f = _as_1d_float("frequency_hz", self.frequency_hz)
        object.__setattr__(self, "frequency_hz", f)

        if self.s is not None:
            s = jnp.asarray(self.s, dtype=jnp.complex128)
            if s.shape != (f.shape[0], 2, 2):
                raise ValueError(f"s must have shape {(f.shape[0], 2, 2)}, got {s.shape}")
            object.__setattr__(self, "s", s)

        if self.s21_db is not None:
            y = _as_1d_float("s21_db", self.s21_db)
            if y.shape[0] != f.shape[0]:
                raise ValueError("s21_db length must match frequency_hz")
            object.__setattr__(self, "s21_db", y)

        if self.s is None and self.s21_db is None:
            raise ValueError("At least one of s or s21_db must be provided")

        if self.weight_complex < 0.0 or self.weight_s21_db < 0.0:
            raise ValueError("weights must be non-negative")

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency_hz": _jsonify(self.frequency_hz),
            "has_complex_s": self.s is not None,
            "has_s21_db": self.s21_db is not None,
            "weight_complex": self.weight_complex,
            "weight_s21_db": self.weight_s21_db,
            "use_s11": self.use_s11,
            "use_s21": self.use_s21,
            "use_s12": self.use_s12,
            "use_s22": self.use_s22,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class GainCalibrationData:
    """
    Pump-on gain calibration data.

    The gain simulation must produce labels matching signal_labels and optional
    idler_labels.
    """

    signal_labels: tuple[str, ...]
    signal_gain_db: jax.Array
    idler_labels: tuple[str | None, ...] | None = None
    idler_conversion_db: jax.Array | None = None
    weight_signal_gain_db: float = 1.0
    weight_idler_conversion_db: float = 1.0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        labels = tuple(self.signal_labels)
        if not labels:
            raise ValueError("signal_labels may not be empty")
        object.__setattr__(self, "signal_labels", labels)

        gain = _as_1d_float("signal_gain_db", self.signal_gain_db)
        if gain.shape[0] != len(labels):
            raise ValueError("signal_gain_db length must match signal_labels")
        object.__setattr__(self, "signal_gain_db", gain)

        if self.idler_labels is not None:
            idlers = tuple(self.idler_labels)
            if len(idlers) != len(labels):
                raise ValueError("idler_labels length must match signal_labels")
            object.__setattr__(self, "idler_labels", idlers)

        if self.idler_conversion_db is not None:
            conv = _as_1d_float("idler_conversion_db", self.idler_conversion_db)
            if conv.shape[0] != len(labels):
                raise ValueError("idler_conversion_db length must match signal_labels")
            object.__setattr__(self, "idler_conversion_db", conv)

        if self.weight_signal_gain_db < 0.0 or self.weight_idler_conversion_db < 0.0:
            raise ValueError("weights must be non-negative")

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_labels": list(self.signal_labels),
            "idler_labels": None if self.idler_labels is None else list(self.idler_labels),
            "has_idler_conversion_db": self.idler_conversion_db is not None,
            "weight_signal_gain_db": self.weight_signal_gain_db,
            "weight_idler_conversion_db": self.weight_idler_conversion_db,
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationTarget:
    """
    Full simulator target used during calibration.

    Parameters
    ----------
    base_layout:
        Nominal layout before calibration scalings.
    base_nonlinear_params:
        Nominal nonlinear parameters before calibration scalings.
    cell_model, cascade:
        Linear simulation configuration.
    pump_drive, pump_config:
        Pump-on simulation configuration.
    target_plan_factory, sweep_config_factory:
        Optional gain-stage factories.
    residual_hooks:
        Optional user-defined residual hooks.
    """

    base_layout: LineLayout
    base_nonlinear_params: NonlinearParams | None = None
    cell_model: CellModelConfig = CellModelConfig()
    cascade: CascadeConfig = CascadeConfig()
    pump_drive: PumpDriveConfig | None = None
    pump_config: PumpHBLadderConfig | None = None
    target_plan_factory: TargetPlanFactory | None = None
    sweep_config_factory: SweepConfigFactory | None = None
    residual_hooks: tuple[ResidualHook, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "residual_hooks", tuple(self.residual_hooks))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def with_updates(self, **kwargs: Any) -> "CalibrationTarget":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_layout": self.base_layout.summary(),
            "base_nonlinear_params": (
                None
                if self.base_nonlinear_params is None
                else self.base_nonlinear_params.to_dict()
            ),
            "cell_model": self.cell_model.to_dict(),
            "cascade": self.cascade.to_dict(),
            "pump_drive": None if self.pump_drive is None else self.pump_drive.to_dict(),
            "pump_config": None if self.pump_config is None else self.pump_config.to_dict(),
            "has_target_plan_factory": self.target_plan_factory is not None,
            "has_sweep_config_factory": self.sweep_config_factory is not None,
            "n_residual_hooks": len(self.residual_hooks),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Applying parameters
# ---------------------------------------------------------------------------

def calibrated_layout_from_parameters(
    base_layout: LineLayout,
    params: Mapping[str, float],
    *,
    name: str | None = None,
) -> LineLayout:
    """
    Apply calibration parameters to a LineLayout.

    Recognized parameters
    ---------------------
    length_scale
    L_scale
    C_scale
    C_stub_scale
    R_scale
    G_scale
    L_res_scale
    C_res_scale
    C_couple_scale
    z0_ohm
    """
    length_scale = float(params.get("length_scale", 1.0))
    L_scale = float(params.get("L_scale", 1.0))
    C_scale = float(params.get("C_scale", 1.0))
    C_stub_scale = float(params.get("C_stub_scale", 1.0))
    R_scale = float(params.get("R_scale", 1.0))
    G_scale = float(params.get("G_scale", 1.0))
    L_res_scale = float(params.get("L_res_scale", 1.0))
    C_res_scale = float(params.get("C_res_scale", 1.0))
    C_couple_scale = float(params.get("C_couple_scale", 1.0))
    z0_ohm = float(params.get("z0_ohm", base_layout.z0_ohm))

    return make_layout_from_arrays(
        length_m=base_layout.length_m * length_scale,
        L_series_H=base_layout.L_series_H * L_scale,
        C_shunt_F=base_layout.C_shunt_F * C_scale,
        R_series_ohm=base_layout.R_series_ohm * R_scale,
        G_shunt_S=base_layout.G_shunt_S * G_scale,
        C_stub_F=base_layout.C_stub_F * C_stub_scale,
        L_res_H=base_layout.L_res_H * L_res_scale,
        C_res_F=base_layout.C_res_F * C_res_scale,
        C_couple_F=base_layout.C_couple_F * C_couple_scale,
        z0_ohm=z0_ohm,
        name=name or f"{base_layout.name}_calibrated",
        metadata={
            **dict(base_layout.metadata or {}),
            "source": "calibrated_layout_from_parameters",
            "calibration_parameters": dict(params),
        },
    )


def calibrated_nonlinear_params_from_parameters(
    base: NonlinearParams,
    params: Mapping[str, float],
) -> NonlinearParams:
    """
    Apply calibration parameters to NonlinearParams.

    Recognized parameters
    ---------------------
    I_star_scale
    I_star_A
    beta_nl_scale
    beta_nl
    quartic_scale
    quartic_coefficient
    dc_bias_A
    """
    I_star = float(params.get("I_star_A", base.I_star_A * float(params.get("I_star_scale", 1.0))))
    beta = float(params.get("beta_nl", base.beta_nl * float(params.get("beta_nl_scale", 1.0))))
    quartic = float(
        params.get(
            "quartic_coefficient",
            base.quartic_coefficient * float(params.get("quartic_scale", 1.0)),
        )
    )
    dc_bias = float(params.get("dc_bias_A", base.dc_bias_A))

    return replace(
        base,
        I_star_A=I_star,
        beta_nl=beta,
        quartic_coefficient=quartic,
        dc_bias_A=dc_bias,
    )


def calibrated_pump_drive_from_parameters(
    base: PumpDriveConfig,
    params: Mapping[str, float],
) -> PumpDriveConfig:
    """
    Apply calibration parameters to PumpDriveConfig.

    Recognized parameters
    ---------------------
    pump_current_scale
    pump_power_offset_db
    pump_frequency_offset_hz
    pump_phase_rad
    """
    drive = base
    fp = base.pump_frequency_hz + float(params.get("pump_frequency_offset_hz", 0.0))
    phase = float(params.get("pump_phase_rad", base.phase_rad))

    if "pump_power_offset_db" in params:
        drive = drive.with_available_power_dbm(base.available_power_dbm + float(params["pump_power_offset_db"]))
    elif "pump_current_scale" in params:
        drive = drive.with_current_rms(base.current_rms_A * float(params["pump_current_scale"]))

    return replace(drive, pump_frequency_hz=fp, phase_rad=phase)


# ---------------------------------------------------------------------------
# Residual construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationObjectiveBreakdown:
    """
    Residual component sizes and scalar losses.
    """

    component_norms: Mapping[str, float]
    component_sizes: Mapping[str, int]
    total_size: int
    total_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_norms": dict(self.component_norms),
            "component_sizes": dict(self.component_sizes),
            "total_size": self.total_size,
            "total_loss": self.total_loss,
        }


@dataclass(frozen=True)
class CalibrationEvaluation:
    """
    One objective evaluation.
    """

    parameters: ParameterDict
    residual: jax.Array
    breakdown: CalibrationObjectiveBreakdown
    simulated_layout_summary: Mapping[str, Any]
    metadata: Mapping[str, Any] | None = None

    @property
    def loss(self) -> float:
        return self.breakdown.total_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": dict(self.parameters),
            "residual_shape": tuple(int(v) for v in self.residual.shape),
            "loss": self.loss,
            "breakdown": self.breakdown.to_dict(),
            "simulated_layout_summary": dict(self.simulated_layout_summary),
            "metadata": dict(self.metadata or {}),
        }


def evaluate_linear_sparameters(
    target: CalibrationTarget,
    params: Mapping[str, float],
    data: SParameterCalibrationData,
) -> jax.Array:
    """
    Return simulated S-parameters on the measurement frequency grid.
    """
    layout = calibrated_layout_from_parameters(target.base_layout, params)
    scan = run_linear_scan(
        data.frequency_hz,
        layout,
        cell_model=target.cell_model,
        cascade_config=target.cascade,
    )
    return scan.s


def sparameter_residual_vector(
    target: CalibrationTarget,
    params: Mapping[str, float],
    data: SParameterCalibrationData,
) -> jax.Array:
    """
    Weighted residual vector for S-parameter data.
    """
    s_model = evaluate_linear_sparameters(target, params, data)
    pieces = []

    if data.s is not None and data.weight_complex > 0.0:
        entries: list[tuple[int, int]] = []
        if data.use_s11:
            entries.append((0, 0))
        if data.use_s12:
            entries.append((0, 1))
        if data.use_s21:
            entries.append((1, 0))
        if data.use_s22:
            entries.append((1, 1))

        for a, b in entries:
            diff = data.weight_complex * (s_model[:, a, b] - data.s[:, a, b])
            pieces.append(jnp.real(diff))
            pieces.append(jnp.imag(diff))

    if data.s21_db is not None and data.weight_s21_db > 0.0:
        s21_model = s_model[:, 1, 0]
        s21_db_model = 20.0 * jnp.log10(jnp.maximum(jnp.abs(s21_model), 1e-300))
        pieces.append(data.weight_s21_db * (s21_db_model - data.s21_db))

    if not pieces:
        return jnp.zeros((0,), dtype=jnp.float64)

    return jnp.concatenate([jnp.ravel(jnp.asarray(p, dtype=jnp.float64)) for p in pieces])


def simulate_gain_sweep_from_parameters(
    target: CalibrationTarget,
    params: Mapping[str, float],
) -> GainSweepResult:
    """
    Simulate pump HB and labelled gain sweep for the current parameters.
    """
    if target.base_nonlinear_params is None:
        raise ValueError("Gain calibration requires target.base_nonlinear_params")
    if target.pump_drive is None:
        raise ValueError("Gain calibration requires target.pump_drive")
    if target.pump_config is None:
        raise ValueError("Gain calibration requires target.pump_config")
    if target.target_plan_factory is None or target.sweep_config_factory is None:
        raise ValueError("Gain calibration requires target_plan_factory and sweep_config_factory")

    layout = calibrated_layout_from_parameters(target.base_layout, params)
    nonlinear = calibrated_nonlinear_params_from_parameters(target.base_nonlinear_params, params)
    drive = calibrated_pump_drive_from_parameters(target.pump_drive, params)

    pump = solve_pump_hb_ladder(
        layout,
        nonlinear,
        drive=drive,
        pump_config=target.pump_config,
        metadata={"calibration": "simulate_gain_sweep_from_parameters"},
    )

    target_plan = target.target_plan_factory(pump)
    sweep_config = target.sweep_config_factory(target_plan)

    return solve_gain_sweep_from_pump(
        pump,
        target_plan=target_plan,
        sweep_config=sweep_config,
    )


def gain_residual_vector(
    target: CalibrationTarget,
    params: Mapping[str, float],
    data: GainCalibrationData,
) -> jax.Array:
    """
    Weighted residual vector for pump-on gain data.
    """
    sweep = simulate_gain_sweep_from_parameters(target, params)
    trace = extract_labelled_gain_trace(sweep)

    label_to_index = {label: i for i, label in enumerate(trace.signal_labels)}

    gain_model = []
    idler_model = []

    for label in data.signal_labels:
        if label not in label_to_index:
            raise ValueError(f"Simulated gain sweep does not contain signal label {label!r}")
        gain_model.append(trace.signal_gain_db[label_to_index[label]])

    pieces = [
        data.weight_signal_gain_db
        * (jnp.asarray(gain_model, dtype=jnp.float64) - data.signal_gain_db)
    ]

    if data.idler_conversion_db is not None:
        if data.idler_labels is None:
            raise ValueError("idler_conversion_db was provided but idler_labels is None")
        idler_label_to_index = {label: i for i, label in enumerate(trace.idler_labels)}
        for label in data.idler_labels:
            if label not in idler_label_to_index:
                raise ValueError(f"Simulated gain sweep does not contain idler label {label!r}")
            idler_model.append(trace.idler_conversion_db[idler_label_to_index[label]])
        pieces.append(
            data.weight_idler_conversion_db
            * (jnp.asarray(idler_model, dtype=jnp.float64) - data.idler_conversion_db)
        )

    return jnp.concatenate([jnp.ravel(jnp.asarray(p, dtype=jnp.float64)) for p in pieces])


def evaluate_calibration_objective(
    target: CalibrationTarget,
    params: Mapping[str, float],
    *,
    sparameter_data: SParameterCalibrationData | None = None,
    gain_data: GainCalibrationData | None = None,
) -> CalibrationEvaluation:
    """
    Evaluate the full calibration residual and objective.
    """
    residual_pieces: list[jax.Array] = []
    norms: dict[str, float] = {}
    sizes: dict[str, int] = {}

    if sparameter_data is not None:
        r_s = sparameter_residual_vector(target, params, sparameter_data)
        residual_pieces.append(r_s)
        norms["sparameters"] = float(jnp.linalg.norm(r_s))
        sizes["sparameters"] = int(r_s.shape[0])

    if gain_data is not None:
        r_g = gain_residual_vector(target, params, gain_data)
        residual_pieces.append(r_g)
        norms["gain"] = float(jnp.linalg.norm(r_g))
        sizes["gain"] = int(r_g.shape[0])

    for idx, hook in enumerate(target.residual_hooks):
        r_h = jnp.ravel(jnp.asarray(hook(dict(params)), dtype=jnp.float64))
        residual_pieces.append(r_h)
        norms[f"hook_{idx}"] = float(jnp.linalg.norm(r_h))
        sizes[f"hook_{idx}"] = int(r_h.shape[0])

    if not residual_pieces:
        raise ValueError("No calibration data or residual hooks were provided")

    residual = jnp.concatenate(residual_pieces)
    loss = 0.5 * float(jnp.sum(residual**2))
    layout = calibrated_layout_from_parameters(target.base_layout, params)

    return CalibrationEvaluation(
        parameters=dict(params),
        residual=residual,
        breakdown=CalibrationObjectiveBreakdown(
            component_norms=norms,
            component_sizes=sizes,
            total_size=int(residual.shape[0]),
            total_loss=loss,
        ),
        simulated_layout_summary=layout.summary(),
        metadata={"target": target.to_dict()},
    )


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

class CalibrationOptimizerMethod(str, Enum):
    """Calibration optimizer method."""

    SCIPY_LEAST_SQUARES = "scipy_least_squares"
    COORDINATE_SEARCH = "coordinate_search"
    AUTO = "auto"


@dataclass(frozen=True)
class CalibrationOptimizerConfig:
    """
    Optimizer configuration.
    """

    method: CalibrationOptimizerMethod = CalibrationOptimizerMethod.AUTO
    max_evaluations: int = 100
    xtol: float = 1e-8
    ftol: float = 1e-8
    gtol: float = 1e-8
    verbose: bool = True
    coordinate_initial_step_fraction: float = 0.10
    coordinate_step_decay: float = 0.5
    coordinate_min_step_fraction: float = 1e-4
    random_seed: int = 1234

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", CalibrationOptimizerMethod(self.method))
        if int(self.max_evaluations) <= 0:
            raise ValueError("max_evaluations must be positive")
        object.__setattr__(self, "max_evaluations", int(self.max_evaluations))
        if self.coordinate_initial_step_fraction <= 0.0:
            raise ValueError("coordinate_initial_step_fraction must be positive")
        if not (0.0 < self.coordinate_step_decay < 1.0):
            raise ValueError("coordinate_step_decay must be in (0, 1)")
        if self.coordinate_min_step_fraction <= 0.0:
            raise ValueError("coordinate_min_step_fraction must be positive")

    def selected_method(self) -> CalibrationOptimizerMethod:
        if self.method == CalibrationOptimizerMethod.AUTO:
            return (
                CalibrationOptimizerMethod.SCIPY_LEAST_SQUARES
                if SCIPY_AVAILABLE
                else CalibrationOptimizerMethod.COORDINATE_SEARCH
            )
        return self.method

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "selected_method": self.selected_method().value,
            "scipy_available": SCIPY_AVAILABLE,
            "max_evaluations": self.max_evaluations,
            "xtol": self.xtol,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "verbose": self.verbose,
            "coordinate_initial_step_fraction": self.coordinate_initial_step_fraction,
            "coordinate_step_decay": self.coordinate_step_decay,
            "coordinate_min_step_fraction": self.coordinate_min_step_fraction,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class CalibrationIterationRecord:
    """
    One calibration optimizer record.
    """

    iteration: int
    encoded_vector: jax.Array
    parameters: ParameterDict
    loss: float
    residual_norm: float
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "encoded_vector": [float(v) for v in np.asarray(self.encoded_vector).tolist()],
            "parameters": dict(self.parameters),
            "loss": self.loss,
            "residual_norm": self.residual_norm,
            "message": self.message,
        }


@dataclass(frozen=True)
class CalibrationResult:
    """
    Final calibration result.
    """

    best_parameters: ParameterDict
    best_encoded_vector: jax.Array
    best_evaluation: CalibrationEvaluation
    parameter_spec: CalibrationVectorSpec
    optimizer_config: CalibrationOptimizerConfig
    records: tuple[CalibrationIterationRecord, ...]
    success: bool
    message: str
    metadata: Mapping[str, Any] | None = None

    @property
    def loss(self) -> float:
        return self.best_evaluation.loss

    @property
    def residual_norm(self) -> float:
        return float(jnp.linalg.norm(self.best_evaluation.residual))

    def calibrated_layout(self, base_layout: LineLayout) -> LineLayout:
        return calibrated_layout_from_parameters(base_layout, self.best_parameters)

    def calibrated_nonlinear_params(self, base: NonlinearParams) -> NonlinearParams:
        return calibrated_nonlinear_params_from_parameters(base, self.best_parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "best_parameters": dict(self.best_parameters),
            "best_encoded_vector": [float(v) for v in np.asarray(self.best_encoded_vector).tolist()],
            "loss": self.loss,
            "residual_norm": self.residual_norm,
            "best_evaluation": self.best_evaluation.to_dict(),
            "parameter_spec": self.parameter_spec.to_dict(),
            "optimizer_config": self.optimizer_config.to_dict(),
            "records": [r.to_dict() for r in self.records],
            "metadata": dict(self.metadata or {}),
        }


def _make_residual_function(
    target: CalibrationTarget,
    parameter_spec: CalibrationVectorSpec,
    *,
    sparameter_data: SParameterCalibrationData | None,
    gain_data: GainCalibrationData | None,
) -> Callable[[np.ndarray], np.ndarray]:
    def residual_fn(encoded_vector: np.ndarray) -> np.ndarray:
        params = parameter_spec.decode_vector(encoded_vector)
        evaluation = evaluate_calibration_objective(
            target,
            params,
            sparameter_data=sparameter_data,
            gain_data=gain_data,
        )
        r = np.asarray(evaluation.residual, dtype=float)
        if not np.all(np.isfinite(r)):
            return np.full_like(r, 1e30)
        return r

    return residual_fn


def calibrate_with_scipy_least_squares(
    target: CalibrationTarget,
    parameter_spec: CalibrationVectorSpec,
    optimizer: CalibrationOptimizerConfig,
    *,
    sparameter_data: SParameterCalibrationData | None = None,
    gain_data: GainCalibrationData | None = None,
) -> CalibrationResult:
    """
    Run SciPy least_squares calibration.
    """
    if not SCIPY_AVAILABLE or scipy_least_squares is None:
        raise RuntimeError("SciPy least_squares is not available")

    residual_fn = _make_residual_function(
        target,
        parameter_spec,
        sparameter_data=sparameter_data,
        gain_data=gain_data,
    )

    x0 = np.asarray(parameter_spec.initial_vector(), dtype=float)
    lower = np.asarray(parameter_spec.lower_vector(), dtype=float)
    upper = np.asarray(parameter_spec.upper_vector(), dtype=float)

    result = scipy_least_squares(
        residual_fn,
        x0,
        bounds=(lower, upper),
        max_nfev=optimizer.max_evaluations,
        xtol=optimizer.xtol,
        ftol=optimizer.ftol,
        gtol=optimizer.gtol,
        verbose=2 if optimizer.verbose else 0,
        loss="soft_l1",
        f_scale=1.0,
    )

    best_params = parameter_spec.decode_vector(result.x)
    best_eval = evaluate_calibration_objective(
        target,
        best_params,
        sparameter_data=sparameter_data,
        gain_data=gain_data,
    )

    record = CalibrationIterationRecord(
        iteration=int(result.nfev),
        encoded_vector=jnp.asarray(result.x),
        parameters=best_params,
        loss=best_eval.loss,
        residual_norm=float(jnp.linalg.norm(best_eval.residual)),
        message=str(result.message),
    )

    return CalibrationResult(
        best_parameters=best_params,
        best_encoded_vector=jnp.asarray(result.x),
        best_evaluation=best_eval,
        parameter_spec=parameter_spec,
        optimizer_config=optimizer,
        records=(record,),
        success=bool(result.success),
        message=str(result.message),
        metadata={
            "optimizer": "scipy_least_squares",
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "status": int(result.status),
        },
    )


def calibrate_with_coordinate_search(
    target: CalibrationTarget,
    parameter_spec: CalibrationVectorSpec,
    optimizer: CalibrationOptimizerConfig,
    *,
    sparameter_data: SParameterCalibrationData | None = None,
    gain_data: GainCalibrationData | None = None,
) -> CalibrationResult:
    """
    Derivative-free bounded coordinate search fallback.

    This is slower than SciPy least_squares but robust for expensive workflow
    residuals and environments without SciPy.
    """
    x = np.asarray(parameter_spec.initial_vector(), dtype=float)
    lower = np.asarray(parameter_spec.lower_vector(), dtype=float)
    upper = np.asarray(parameter_spec.upper_vector(), dtype=float)

    span = np.maximum(upper - lower, 1e-12)
    step = optimizer.coordinate_initial_step_fraction * span
    min_step = optimizer.coordinate_min_step_fraction * span

    def evaluate_x(x_vec: np.ndarray) -> CalibrationEvaluation:
        params = parameter_spec.decode_vector(x_vec)
        return evaluate_calibration_objective(
            target,
            params,
            sparameter_data=sparameter_data,
            gain_data=gain_data,
        )

    best_eval = evaluate_x(x)
    best_loss = best_eval.loss
    records: list[CalibrationIterationRecord] = [
        CalibrationIterationRecord(
            iteration=0,
            encoded_vector=jnp.asarray(x),
            parameters=best_eval.parameters,
            loss=best_loss,
            residual_norm=float(jnp.linalg.norm(best_eval.residual)),
            message="initial",
        )
    ]

    n_eval = 1
    iteration = 0

    while n_eval < optimizer.max_evaluations and np.any(step > min_step):
        iteration += 1
        improved = False

        for dim in range(x.shape[0]):
            for direction in (+1.0, -1.0):
                if n_eval >= optimizer.max_evaluations:
                    break

                candidate = x.copy()
                candidate[dim] = np.clip(candidate[dim] + direction * step[dim], lower[dim], upper[dim])

                if np.allclose(candidate, x):
                    continue

                cand_eval = evaluate_x(candidate)
                n_eval += 1

                if cand_eval.loss < best_loss:
                    x = candidate
                    best_eval = cand_eval
                    best_loss = cand_eval.loss
                    improved = True

                    records.append(
                        CalibrationIterationRecord(
                            iteration=iteration,
                            encoded_vector=jnp.asarray(x),
                            parameters=best_eval.parameters,
                            loss=best_loss,
                            residual_norm=float(jnp.linalg.norm(best_eval.residual)),
                            message=f"accepted dim={dim} direction={direction:+.0f}",
                        )
                    )

        if not improved:
            step *= optimizer.coordinate_step_decay
            records.append(
                CalibrationIterationRecord(
                    iteration=iteration,
                    encoded_vector=jnp.asarray(x),
                    parameters=best_eval.parameters,
                    loss=best_loss,
                    residual_norm=float(jnp.linalg.norm(best_eval.residual)),
                    message="no improvement; step decayed",
                )
            )

    success = bool(np.all(step <= min_step) or n_eval >= optimizer.max_evaluations)

    return CalibrationResult(
        best_parameters=best_eval.parameters,
        best_encoded_vector=jnp.asarray(x),
        best_evaluation=best_eval,
        parameter_spec=parameter_spec,
        optimizer_config=optimizer,
        records=tuple(records),
        success=success,
        message=f"coordinate search finished after {n_eval} evaluations",
        metadata={
            "optimizer": "coordinate_search",
            "n_evaluations": n_eval,
            "final_step": step.tolist(),
            "min_step": min_step.tolist(),
        },
    )


def calibrate(
    target: CalibrationTarget,
    parameter_spec: CalibrationVectorSpec,
    *,
    sparameter_data: SParameterCalibrationData | None = None,
    gain_data: GainCalibrationData | None = None,
    optimizer_config: CalibrationOptimizerConfig | None = None,
) -> CalibrationResult:
    """
    Main calibration dispatcher.
    """
    optimizer = optimizer_config or CalibrationOptimizerConfig()
    method = optimizer.selected_method()

    if method == CalibrationOptimizerMethod.SCIPY_LEAST_SQUARES:
        return calibrate_with_scipy_least_squares(
            target,
            parameter_spec,
            optimizer,
            sparameter_data=sparameter_data,
            gain_data=gain_data,
        )

    if method == CalibrationOptimizerMethod.COORDINATE_SEARCH:
        return calibrate_with_coordinate_search(
            target,
            parameter_spec,
            optimizer,
            sparameter_data=sparameter_data,
            gain_data=gain_data,
        )

    raise ValueError(f"Unsupported calibration optimizer method {method}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def finite_difference_residual_jacobian(
    target: CalibrationTarget,
    parameter_spec: CalibrationVectorSpec,
    encoded_vector: ArrayLike,
    *,
    sparameter_data: SParameterCalibrationData | None = None,
    gain_data: GainCalibrationData | None = None,
    relative_step: float = 1e-5,
) -> dict[str, Any]:
    """
    Finite-difference Jacobian diagnostic for the calibration residual.
    """
    x = np.asarray(encoded_vector, dtype=float)
    residual_fn = _make_residual_function(
        target,
        parameter_spec,
        sparameter_data=sparameter_data,
        gain_data=gain_data,
    )

    r0 = residual_fn(x)
    J = np.zeros((r0.shape[0], x.shape[0]), dtype=float)

    for j in range(x.shape[0]):
        h = relative_step * max(abs(x[j]), 1.0)
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        xp = np.minimum(np.maximum(xp, np.asarray(parameter_spec.lower_vector())), np.asarray(parameter_spec.upper_vector()))
        xm = np.minimum(np.maximum(xm, np.asarray(parameter_spec.lower_vector())), np.asarray(parameter_spec.upper_vector()))
        rp = residual_fn(xp)
        rm = residual_fn(xm)
        denom = xp[j] - xm[j]
        if abs(denom) <= 1e-300:
            J[:, j] = 0.0
        else:
            J[:, j] = (rp - rm) / denom

    singular_values = np.linalg.svd(J, compute_uv=False) if J.size else np.asarray([])
    column_norms = np.linalg.norm(J, axis=0) if J.size else np.zeros((x.shape[0],), dtype=float)
    denom = np.outer(column_norms, column_norms)
    correlation = np.divide(
        J.T @ J,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0.0,
    )
    parameter_names = list(parameter_spec.enabled_names)
    strongly_correlated_pairs = [
        {
            "left": parameter_names[i],
            "right": parameter_names[j],
            "correlation": float(correlation[i, j]),
        }
        for i in range(len(parameter_names))
        for j in range(i + 1, len(parameter_names))
        if abs(float(correlation[i, j])) >= 0.95
    ]

    return {
        "jacobian_shape": tuple(int(v) for v in J.shape),
        "residual_size": int(r0.shape[0]),
        "parameter_size": int(x.shape[0]),
        "singular_values": singular_values.tolist(),
        "condition_number": (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size and singular_values[-1] > 0.0
            else float("inf")
        ),
        "rank_estimate": int(np.linalg.matrix_rank(J)) if J.size else 0,
        "max_abs_jacobian": float(np.max(np.abs(J))) if J.size else 0.0,
        "parameter_names": parameter_names,
        "column_norms": column_norms.tolist(),
        "parameter_correlation": correlation.tolist(),
        "strongly_correlated_pairs": strongly_correlated_pairs,
    }


def calibration_summary_markdown(result: CalibrationResult) -> str:
    """
    Markdown summary of a calibration result.
    """
    lines = [
        "# Calibration summary",
        "",
        f"- success: `{result.success}`",
        f"- message: `{result.message}`",
        f"- loss: `{result.loss:.6e}`",
        f"- residual norm: `{result.residual_norm:.6e}`",
        "",
        "## Best parameters",
        "",
        "| parameter | value |",
        "|---|---:|",
    ]

    for name, value in result.best_parameters.items():
        lines.append(f"| `{name}` | `{value:.12g}` |")

    lines += [
        "",
        "## Objective breakdown",
        "",
        "| component | size | norm |",
        "|---|---:|---:|",
    ]

    for name, size in result.best_evaluation.breakdown.component_sizes.items():
        norm = result.best_evaluation.breakdown.component_norms.get(name, float("nan"))
        lines.append(f"| `{name}` | `{size}` | `{norm:.6e}` |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def export_calibration_artifacts(
    result: CalibrationResult,
    output_dir: str | Path,
    *,
    prefix: str = "calibration",
) -> dict[str, str]:
    """
    Export calibration result JSON and residual arrays.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    paths["summary_json"] = str(write_json(out / f"{prefix}_summary.json", result.to_dict()))
    paths["residual_npz"] = str(
        write_npz(
            out / f"{prefix}_residual.npz",
            residual=result.best_evaluation.residual,
            encoded_vector=result.best_encoded_vector,
        )
    )
    paths["summary_md"] = str(out / f"{prefix}_summary.md")
    Path(paths["summary_md"]).write_text(calibration_summary_markdown(result), encoding="utf-8")

    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_1d_float(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr


def load_sparameter_calibration_npz(
    path: str | Path,
    *,
    frequency_key: str = "frequency_hz",
    s_key: str = "s",
    s21_db_key: str = "s21_db",
    weight_complex: float = 1.0,
    weight_s21_db: float = 1.0,
) -> SParameterCalibrationData:
    """
    Load S-parameter calibration data from an NPZ file.

    Expected optional arrays:
        frequency_hz
        s          shape (F,2,2)
        s21_db     shape (F,)
    """
    data = np.load(path)
    frequency = data[frequency_key]
    s = data[s_key] if s_key in data else None
    s21_db = data[s21_db_key] if s21_db_key in data else None

    return SParameterCalibrationData(
        frequency_hz=jnp.asarray(frequency),
        s=None if s is None else jnp.asarray(s),
        s21_db=None if s21_db is None else jnp.asarray(s21_db),
        weight_complex=weight_complex,
        weight_s21_db=weight_s21_db,
        metadata={"source_path": str(path)},
    )


def make_linear_calibration_target(
    base_layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    cascade: CascadeConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationTarget:
    """
    Convenience constructor for pump-off S-parameter calibration.
    """
    return CalibrationTarget(
        base_layout=base_layout,
        cell_model=cell_model or CellModelConfig(),
        cascade=cascade or CascadeConfig(),
        metadata={
            "target_kind": "linear_sparameter",
            **dict(metadata or {}),
        },
    )


__all__ = [
    "ParameterTransform",
    "CalibrationParameterSpec",
    "CalibrationVectorSpec",
    "make_default_linear_parameter_spec",
    "make_default_nonlinear_parameter_spec",
    "SParameterCalibrationData",
    "GainCalibrationData",
    "CalibrationTarget",
    "calibrated_layout_from_parameters",
    "calibrated_nonlinear_params_from_parameters",
    "calibrated_pump_drive_from_parameters",
    "CalibrationObjectiveBreakdown",
    "CalibrationEvaluation",
    "evaluate_linear_sparameters",
    "sparameter_residual_vector",
    "simulate_gain_sweep_from_parameters",
    "gain_residual_vector",
    "evaluate_calibration_objective",
    "CalibrationOptimizerMethod",
    "CalibrationOptimizerConfig",
    "CalibrationIterationRecord",
    "CalibrationResult",
    "calibrate_with_scipy_least_squares",
    "calibrate_with_coordinate_search",
    "calibrate",
    "finite_difference_residual_jacobian",
    "calibration_summary_markdown",
    "export_calibration_artifacts",
    "load_sparameter_calibration_npz",
    "make_linear_calibration_target",
]
