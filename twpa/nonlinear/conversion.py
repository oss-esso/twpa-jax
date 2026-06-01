"""
twpa.nonlinear.conversion
=========================

Frequency-conversion matrix utilities for pumped TWPA simulations.

This module provides a general small-signal conversion-matrix layer above the
linearized distributed harmonic-balance system.

Given a pumped operating point and a target FrequencyPlan containing signal,
idler, and other mixing sidebands, the linearized equation is solved once per
input tone:

    J dx_k = -r_source,k

The output voltages are collected into a matrix:

    M[out_label, in_label] = V_out(out_label) / I_in(in_label)

and optionally normalized to voltage ratios:

    G[out_label, in_label] = V_out(out_label) / V_in(in_label)

This is the more general object behind scalar signal gain and idler conversion.
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
from twpa.core.frequency_plan import FrequencyPlan
from twpa.nonlinear.distributed_hb import DistributedHBSolveResult
from twpa.nonlinear.linearization import (
    DistributedHBLinearization,
    SmallSignalLinearizationConfig,
    SmallSignalSolveResult,
    SmallSignalSource,
    build_linearization_from_pump_result,
    solve_linearized_small_signal,
)
from twpa.nonlinear.pump_hb_ladder import PumpHBLadderResult


ArrayLike = Any


class ConversionMatrixNormalization(str, Enum):
    """Supported conversion-matrix normalizations."""

    TRANSIMPEDANCE_V_PER_A = "transimpedance_V_per_A"
    VOLTAGE_RATIO = "voltage_ratio"
    MATCHED_POWER_RATIO = "matched_power_ratio"


class ConversionStatus(str, Enum):
    """Status for conversion-matrix calculations."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass(frozen=True)
class ConversionTone:
    """
    One tone used in a conversion calculation.
    """

    label: str
    frequency_hz: float
    node: int | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.frequency_hz == 0.0 and self.label != "dc":
            pass
        if self.node is not None and int(self.node) < 0:
            raise ValueError("node must be non-negative if provided")
        object.__setattr__(self, "frequency_hz", float(self.frequency_hz))
        if self.node is not None:
            object.__setattr__(self, "node", int(self.node))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "frequency_hz": self.frequency_hz,
            "frequency_GHz": self.frequency_hz / 1e9,
            "node": self.node,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class DP4WMToneSet:
    """
    Degenerate-pump four-wave-mixing tone set.

    For a pump frequency fp and signal frequency fs:

        fi = 2 fp - fs
    """

    pump_frequency_hz: float
    signal_frequency_hz: float
    pump_label: str = "pump"
    signal_label: str = "signal"
    idler_label: str = "idler"

    def __post_init__(self) -> None:
        if self.pump_frequency_hz <= 0.0:
            raise ValueError("pump_frequency_hz must be positive")
        if self.signal_frequency_hz <= 0.0:
            raise ValueError("signal_frequency_hz must be positive")

    @property
    def idler_frequency_hz(self) -> float:
        return 2.0 * self.pump_frequency_hz - self.signal_frequency_hz

    def tones(self) -> tuple[ConversionTone, ConversionTone, ConversionTone]:
        return (
            ConversionTone(self.pump_label, self.pump_frequency_hz),
            ConversionTone(self.signal_label, self.signal_frequency_hz),
            ConversionTone(self.idler_label, self.idler_frequency_hz),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pump_frequency_hz": self.pump_frequency_hz,
            "signal_frequency_hz": self.signal_frequency_hz,
            "idler_frequency_hz": self.idler_frequency_hz,
            "pump_label": self.pump_label,
            "signal_label": self.signal_label,
            "idler_label": self.idler_label,
        }


@dataclass(frozen=True)
class ConversionMatrixConfig:
    """
    Configuration for conversion-matrix calculation.
    """

    input_labels: tuple[str, ...]
    output_labels: tuple[str, ...]
    input_node: int = 0
    output_node: int | None = None
    input_current_rms_A: complex = 1e-12 + 0j
    set_conjugate: bool = True
    input_impedance_ohm: float = 50.0
    output_impedance_ohm: float = 50.0
    normalization: ConversionMatrixNormalization = ConversionMatrixNormalization.VOLTAGE_RATIO
    name: str = "conversion_matrix"

    def __post_init__(self) -> None:
        if len(self.input_labels) == 0:
            raise ValueError("input_labels may not be empty")
        if len(self.output_labels) == 0:
            raise ValueError("output_labels may not be empty")
        if int(self.input_node) < 0:
            raise ValueError("input_node must be non-negative")
        if self.output_node is not None and int(self.output_node) < 0:
            raise ValueError("output_node must be non-negative")
        if self.input_impedance_ohm <= 0.0:
            raise ValueError("input_impedance_ohm must be positive")
        if self.output_impedance_ohm <= 0.0:
            raise ValueError("output_impedance_ohm must be positive")
        object.__setattr__(self, "input_labels", tuple(self.input_labels))
        object.__setattr__(self, "output_labels", tuple(self.output_labels))
        object.__setattr__(self, "input_node", int(self.input_node))
        if self.output_node is not None:
            object.__setattr__(self, "output_node", int(self.output_node))
        object.__setattr__(
            self,
            "normalization",
            ConversionMatrixNormalization(self.normalization),
        )

    @classmethod
    def signal_idler(
        cls,
        *,
        signal_label: str = "signal",
        idler_label: str = "idler",
        input_node: int = 0,
        output_node: int | None = None,
        input_current_rms_A: complex = 1e-12 + 0j,
        normalization: ConversionMatrixNormalization = ConversionMatrixNormalization.VOLTAGE_RATIO,
    ) -> "ConversionMatrixConfig":
        return cls(
            input_labels=(signal_label, idler_label),
            output_labels=(signal_label, idler_label),
            input_node=input_node,
            output_node=output_node,
            input_current_rms_A=input_current_rms_A,
            normalization=normalization,
            name="signal_idler_conversion_matrix",
        )

    def with_updates(self, **kwargs: Any) -> "ConversionMatrixConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_labels": list(self.input_labels),
            "output_labels": list(self.output_labels),
            "input_node": self.input_node,
            "output_node": self.output_node,
            "input_current_rms_A": {
                "real": float(jnp.real(self.input_current_rms_A)),
                "imag": float(jnp.imag(self.input_current_rms_A)),
                "abs": float(abs(self.input_current_rms_A)),
            },
            "set_conjugate": self.set_conjugate,
            "input_impedance_ohm": self.input_impedance_ohm,
            "output_impedance_ohm": self.output_impedance_ohm,
            "normalization": self.normalization.value,
            "name": self.name,
        }


@dataclass(frozen=True)
class ConversionColumnResult:
    """
    Result for one injected input tone.
    """

    input_label: str
    source: SmallSignalSource
    solution: SmallSignalSolveResult
    input_voltage_by_label: Mapping[str, complex]
    output_voltage_by_label: Mapping[str, complex]
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.solution.converged

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_label": self.input_label,
            "converged": self.converged,
            "source": self.source.to_dict(),
            "solution": self.solution.to_dict(),
            "input_voltage_by_label": {
                k: {
                    "real": float(jnp.real(v)),
                    "imag": float(jnp.imag(v)),
                    "abs": float(abs(v)),
                }
                for k, v in self.input_voltage_by_label.items()
            },
            "output_voltage_by_label": {
                k: {
                    "real": float(jnp.real(v)),
                    "imag": float(jnp.imag(v)),
                    "abs": float(abs(v)),
                }
                for k, v in self.output_voltage_by_label.items()
            },
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ConversionMatrixResult:
    """
    Full conversion matrix result.

    Matrix axes are:

        rows: output_labels
        cols: input_labels
    """

    config: ConversionMatrixConfig
    linearization: DistributedHBLinearization
    columns: tuple[ConversionColumnResult, ...]
    transimpedance_V_per_A: jax.Array
    voltage_ratio: jax.Array
    matched_power_ratio: jax.Array
    metadata: Mapping[str, Any] | None = None

    @property
    def status(self) -> ConversionStatus:
        n_ok = sum(1 for c in self.columns if c.converged)
        if n_ok == len(self.columns):
            return ConversionStatus.PASS
        if n_ok == 0:
            return ConversionStatus.FAIL
        return ConversionStatus.PARTIAL

    @property
    def converged(self) -> bool:
        return self.status == ConversionStatus.PASS

    @property
    def matrix(self) -> jax.Array:
        if self.config.normalization == ConversionMatrixNormalization.TRANSIMPEDANCE_V_PER_A:
            return self.transimpedance_V_per_A
        if self.config.normalization == ConversionMatrixNormalization.VOLTAGE_RATIO:
            return self.voltage_ratio
        if self.config.normalization == ConversionMatrixNormalization.MATCHED_POWER_RATIO:
            return self.matched_power_ratio
        raise ValueError(f"Unsupported normalization {self.config.normalization}")

    @property
    def matrix_db(self) -> jax.Array:
        if self.config.normalization == ConversionMatrixNormalization.MATCHED_POWER_RATIO:
            return 10.0 * jnp.log10(jnp.maximum(jnp.abs(self.matrix), 1e-300))
        return 20.0 * jnp.log10(jnp.maximum(jnp.abs(self.matrix), 1e-300))

    def element(
        self,
        output_label: str,
        input_label: str,
        *,
        normalization: ConversionMatrixNormalization | None = None,
    ) -> complex:
        row = self.config.output_labels.index(output_label)
        col = self.config.input_labels.index(input_label)
        norm = self.config.normalization if normalization is None else ConversionMatrixNormalization(normalization)

        if norm == ConversionMatrixNormalization.TRANSIMPEDANCE_V_PER_A:
            return complex(self.transimpedance_V_per_A[row, col])
        if norm == ConversionMatrixNormalization.VOLTAGE_RATIO:
            return complex(self.voltage_ratio[row, col])
        if norm == ConversionMatrixNormalization.MATCHED_POWER_RATIO:
            return complex(self.matched_power_ratio[row, col])
        raise ValueError(f"Unsupported normalization {norm}")

    def element_db(
        self,
        output_label: str,
        input_label: str,
        *,
        normalization: ConversionMatrixNormalization | None = None,
    ) -> float:
        value = self.element(output_label, input_label, normalization=normalization)
        norm = self.config.normalization if normalization is None else ConversionMatrixNormalization(normalization)
        if norm == ConversionMatrixNormalization.MATCHED_POWER_RATIO:
            return float(10.0 * jnp.log10(jnp.maximum(abs(value), 1e-300)))
        return float(20.0 * jnp.log10(jnp.maximum(abs(value), 1e-300)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "converged": self.converged,
            "config": self.config.to_dict(),
            "input_labels": list(self.config.input_labels),
            "output_labels": list(self.config.output_labels),
            "matrix_shape": tuple(int(v) for v in self.matrix.shape),
            "matrix_abs_max": float(jnp.max(jnp.abs(self.matrix))),
            "matrix_db_min": float(jnp.nanmin(self.matrix_db)),
            "matrix_db_max": float(jnp.nanmax(self.matrix_db)),
            "transimpedance_abs_max_V_per_A": float(jnp.max(jnp.abs(self.transimpedance_V_per_A))),
            "voltage_ratio_abs_max": float(jnp.max(jnp.abs(self.voltage_ratio))),
            "matched_power_ratio_abs_max": float(jnp.max(jnp.abs(self.matched_power_ratio))),
            "columns": [c.to_dict() for c in self.columns],
            "linearization": self.linearization.to_dict(include_matrix=False),
            "metadata": dict(self.metadata or {}),
        }


def _unwrap_pump_result(
    pump_result: DistributedHBSolveResult | PumpHBLadderResult,
) -> DistributedHBSolveResult:
    if isinstance(pump_result, PumpHBLadderResult):
        return pump_result.distributed_result
    return pump_result


def build_conversion_linearization_from_pump(
    pump_result: DistributedHBSolveResult | PumpHBLadderResult,
    *,
    target_plan: FrequencyPlan,
    config: SmallSignalLinearizationConfig | None = None,
) -> DistributedHBLinearization:
    """
    Build a conversion-ready linearization from a pump result.
    """
    return build_linearization_from_pump_result(
        _unwrap_pump_result(pump_result),
        target_plan=target_plan,
        config=config,
    )


def solve_conversion_matrix(
    linearization: DistributedHBLinearization,
    config: ConversionMatrixConfig,
) -> ConversionMatrixResult:
    """
    Solve the conversion matrix by injecting each input tone independently.
    """
    plan = linearization.plan
    layout = linearization.layout
    out_node = layout.n_cells if config.output_node is None else config.output_node

    if config.input_node > layout.n_cells:
        raise ValueError(f"input_node {config.input_node} exceeds last node {layout.n_cells}")
    if out_node > layout.n_cells:
        raise ValueError(f"output_node {out_node} exceeds last node {layout.n_cells}")

    n_out = len(config.output_labels)
    n_in = len(config.input_labels)

    z = jnp.zeros((n_out, n_in), dtype=jnp.complex128)
    g = jnp.zeros((n_out, n_in), dtype=jnp.complex128)
    p = jnp.zeros((n_out, n_in), dtype=jnp.float64)

    columns: list[ConversionColumnResult] = []

    for col, input_label in enumerate(config.input_labels):
        source = SmallSignalSource.current_phasor_at_node(
            plan=plan,
            layout=layout,
            node=config.input_node,
            label=input_label,
            rms_current_A=config.input_current_rms_A,
            set_conjugate=config.set_conjugate,
        )

        solution = solve_linearized_small_signal(linearization, source)

        input_voltage_by_label: dict[str, complex] = {}
        output_voltage_by_label: dict[str, complex] = {}

        in_idx = plan.position_of_label(input_label)
        vin_reference = solution.node_voltage_coeffs_V[in_idx, config.input_node]
        current_reference = config.input_current_rms_A

        for row, output_label in enumerate(config.output_labels):
            out_idx = plan.position_of_label(output_label)
            vout = solution.node_voltage_coeffs_V[out_idx, out_node]
            output_voltage_by_label[output_label] = complex(vout)

            zin = vout / jnp.where(abs(current_reference) > 1e-300, current_reference, 1e-300 + 0j)
            ratio = vout / jnp.where(jnp.abs(vin_reference) > 1e-300, vin_reference, 1e-300 + 0j)
            power_ratio = (
                jnp.abs(ratio) ** 2
                * config.input_impedance_ohm
                / config.output_impedance_ohm
            )

            z = z.at[row, col].set(zin)
            g = g.at[row, col].set(ratio)
            p = p.at[row, col].set(power_ratio)

        for label in config.input_labels:
            idx = plan.position_of_label(label)
            input_voltage_by_label[label] = complex(
                solution.node_voltage_coeffs_V[idx, config.input_node]
            )

        columns.append(
            ConversionColumnResult(
                input_label=input_label,
                source=source,
                solution=solution,
                input_voltage_by_label=input_voltage_by_label,
                output_voltage_by_label=output_voltage_by_label,
                metadata={
                    "input_tone_index": int(in_idx),
                    "input_node": config.input_node,
                    "output_node": out_node,
                },
            )
        )

    return ConversionMatrixResult(
        config=config,
        linearization=linearization,
        columns=tuple(columns),
        transimpedance_V_per_A=z,
        voltage_ratio=g,
        matched_power_ratio=p,
        metadata={
            "source": "solve_conversion_matrix",
            "output_node": out_node,
        },
    )


def solve_conversion_matrix_from_pump(
    pump_result: DistributedHBSolveResult | PumpHBLadderResult,
    *,
    target_plan: FrequencyPlan,
    config: ConversionMatrixConfig,
    linearization_config: SmallSignalLinearizationConfig | None = None,
) -> ConversionMatrixResult:
    """
    Convenience function:

        pump result -> target-plan linearization -> conversion matrix
    """
    lin = build_conversion_linearization_from_pump(
        pump_result,
        target_plan=target_plan,
        config=linearization_config,
    )
    return solve_conversion_matrix(lin, config)


def solve_signal_idler_conversion_matrix_from_pump(
    pump_result: DistributedHBSolveResult | PumpHBLadderResult,
    *,
    target_plan: FrequencyPlan,
    signal_label: str = "signal",
    idler_label: str = "idler",
    input_node: int = 0,
    output_node: int | None = None,
    input_current_rms_A: complex = 1e-12 + 0j,
    linearization_config: SmallSignalLinearizationConfig | None = None,
) -> ConversionMatrixResult:
    """
    Convenience signal/idler 2x2 conversion matrix.
    """
    cfg = ConversionMatrixConfig.signal_idler(
        signal_label=signal_label,
        idler_label=idler_label,
        input_node=input_node,
        output_node=output_node,
        input_current_rms_A=input_current_rms_A,
    )
    return solve_conversion_matrix_from_pump(
        pump_result,
        target_plan=target_plan,
        config=cfg,
        linearization_config=linearization_config,
    )


def _try_call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**filtered)
    except TypeError:
        return fn(**kwargs)


def make_dp4wm_conversion_plan(
    tone_set: DP4WMToneSet,
    *,
    n_pump_harmonics: int = 3,
    include_negative: bool = True,
    include_dc: bool = False,
) -> FrequencyPlan:
    """
    Build a pump/signal/idler FrequencyPlan for DP4WM conversion.

    This first tries native project constructors and then falls back to generic
    frequency-plan constructors.
    """
    fpmod = frequency_plan_module

    constructor_names = [
        "make_pump_signal_idler_plan",
        "make_signal_idler_plan",
        "make_dp4wm_plan",
        "make_dp4wm_frequency_plan",
        "make_gain_plan",
        "make_small_signal_plan",
    ]

    kwargs = {
        "pump_frequency_hz": tone_set.pump_frequency_hz,
        "signal_frequency_hz": tone_set.signal_frequency_hz,
        "idler_frequency_hz": tone_set.idler_frequency_hz,
        "pump_label": tone_set.pump_label,
        "signal_label": tone_set.signal_label,
        "idler_label": tone_set.idler_label,
        "n_pump_harmonics": n_pump_harmonics,
        "n_harmonics": n_pump_harmonics,
        "include_negative": include_negative,
        "include_negative_frequencies": include_negative,
        "include_dc": include_dc,
        "sort": "frequency",
    }

    errors: list[str] = []
    for name in constructor_names:
        fn = getattr(fpmod, name, None)
        if fn is None:
            continue
        try:
            return _try_call_with_supported_kwargs(fn, kwargs)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    frequencies: list[float] = []
    labels: list[str] = []

    def add(label: str, freq: float) -> None:
        labels.append(label)
        frequencies.append(float(freq))

    if include_dc:
        add("dc", 0.0)

    for h in range(1, n_pump_harmonics + 1):
        pos_label = tone_set.pump_label if h == 1 else f"{h}{tone_set.pump_label}"
        add(pos_label, h * tone_set.pump_frequency_hz)
        if include_negative:
            add(f"-{pos_label}", -h * tone_set.pump_frequency_hz)

    add(tone_set.signal_label, tone_set.signal_frequency_hz)
    add(tone_set.idler_label, tone_set.idler_frequency_hz)

    if include_negative:
        add(f"-{tone_set.signal_label}", -tone_set.signal_frequency_hz)
        add(f"-{tone_set.idler_label}", -tone_set.idler_frequency_hz)

    order = np.argsort(np.asarray(frequencies))
    frequencies_sorted = [frequencies[i] for i in order]
    labels_sorted = [labels[i] for i in order]

    generic_constructors = [
        getattr(fpmod, "make_frequency_plan", None),
        getattr(fpmod, "make_plan_from_frequencies", None),
        getattr(fpmod, "FrequencyPlan", None),
    ]

    candidate_kwargs = [
        {
            "frequencies_hz": jnp.asarray(frequencies_sorted, dtype=jnp.float64),
            "labels": tuple(labels_sorted),
            "reference_pump_hz": tone_set.pump_frequency_hz,
            "kind": "custom",
        },
        {
            "frequencies_hz": jnp.asarray(frequencies_sorted, dtype=jnp.float64),
            "tone_labels": tuple(labels_sorted),
            "reference_pump_hz": tone_set.pump_frequency_hz,
            "kind": "custom",
        },
        {
            "frequency_hz": jnp.asarray(frequencies_sorted, dtype=jnp.float64),
            "labels": tuple(labels_sorted),
            "reference_pump_hz": tone_set.pump_frequency_hz,
        },
    ]

    for ctor in generic_constructors:
        if ctor is None:
            continue
        for kw in candidate_kwargs:
            try:
                return _try_call_with_supported_kwargs(ctor, kw)
            except Exception as exc:
                errors.append(f"{getattr(ctor, '__name__', ctor)}: {exc}")

    raise RuntimeError(
        "Could not construct DP4WM conversion FrequencyPlan. "
        "Add make_pump_signal_idler_plan(...) or make_frequency_plan(...) "
        f"to twpa.core.frequency_plan. Errors: {errors}"
    )


def conversion_matrix_table(result: ConversionMatrixResult) -> str:
    """
    Markdown table of the selected normalized conversion matrix in dB.
    """
    mat_db = np.asarray(result.matrix_db)
    lines = [
        f"# {result.config.name}",
        "",
        f"- status: `{result.status.value}`",
        f"- normalization: `{result.config.normalization.value}`",
        "",
        "| output \\ input | " + " | ".join(f"`{label}`" for label in result.config.input_labels) + " |",
        "|---" + "|---:" * len(result.config.input_labels) + "|",
    ]

    for row, out_label in enumerate(result.config.output_labels):
        values = " | ".join(f"{mat_db[row, col]:.6g}" for col in range(len(result.config.input_labels)))
        lines.append(f"| `{out_label}` | {values} |")

    return "\n".join(lines)


__all__ = [
    "ConversionMatrixNormalization",
    "ConversionStatus",
    "ConversionTone",
    "DP4WMToneSet",
    "ConversionMatrixConfig",
    "ConversionColumnResult",
    "ConversionMatrixResult",
    "build_conversion_linearization_from_pump",
    "solve_conversion_matrix",
    "solve_conversion_matrix_from_pump",
    "solve_signal_idler_conversion_matrix_from_pump",
    "make_dp4wm_conversion_plan",
    "conversion_matrix_table",
]


def _validate_unique_integer_orders(name: str, orders: ArrayLike) -> np.ndarray:
    arr = np.asarray(orders)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    if not np.all(np.equal(arr, arr.astype(int))):
        raise ValueError(f"{name} must contain integer-valued orders")
    arr = arr.astype(int)
    if len(set(arr.tolist())) != arr.size:
        raise ValueError(f"{name} must contain unique orders")
    return arr


def harmonic_multiplication_matrix(
    coeffs: ArrayLike,
    *,
    coeff_orders: ArrayLike,
    input_orders: ArrayLike,
    output_orders: ArrayLike | None = None,
) -> np.ndarray:
    in_orders = _validate_unique_integer_orders("input_orders", input_orders)
    out_orders = in_orders if output_orders is None else _validate_unique_integer_orders("output_orders", output_orders)
    c_orders = _validate_unique_integer_orders("coeff_orders", coeff_orders)
    c = np.asarray(coeffs, dtype=np.complex128)
    if c.ndim != 1 or c.shape[0] != c_orders.size:
        raise ValueError("coeffs length must match coeff_orders")
    cmap = {int(order): value for order, value in zip(c_orders, c)}
    M = np.zeros((out_orders.size, in_orders.size), dtype=np.complex128)
    for row, k in enumerate(out_orders):
        for col, j in enumerate(in_orders):
            M[row, col] = cmap.get(int(k - j), 0.0 + 0.0j)
    return M


def conversion_matrix(
    coeffs: ArrayLike,
    *,
    coeff_orders: ArrayLike,
    input_orders: ArrayLike,
    output_orders: ArrayLike | None = None,
) -> np.ndarray:
    return harmonic_multiplication_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )


build_conversion_matrix = conversion_matrix
mixing_matrix = conversion_matrix
convolution_matrix = conversion_matrix


def apply_conversion_matrix(
    coeffs: ArrayLike,
    x: ArrayLike,
    *,
    coeff_orders: ArrayLike,
    input_orders: ArrayLike,
    output_orders: ArrayLike | None = None,
) -> np.ndarray:
    in_orders = _validate_unique_integer_orders("input_orders", input_orders)
    vec = np.asarray(x, dtype=np.complex128)
    if vec.shape[0] != in_orders.size:
        raise ValueError("x first dimension must match input_orders")
    M = conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=in_orders,
        output_orders=output_orders,
    )
    return M @ vec


def convolve_coefficients(
    coeffs: ArrayLike,
    x: ArrayLike,
    *,
    coeff_orders: ArrayLike,
    input_orders: ArrayLike,
    output_orders: ArrayLike | None = None,
) -> np.ndarray:
    return apply_conversion_matrix(
        coeffs,
        x,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )


@dataclass(frozen=True)
class HarmonicConversionMatrix:
    """Reusable harmonic multiplication matrix wrapper."""

    coeffs: ArrayLike
    coeff_orders: ArrayLike
    input_orders: ArrayLike
    output_orders: ArrayLike | None = None

    @property
    def matrix(self) -> np.ndarray:
        return conversion_matrix(
            self.coeffs,
            coeff_orders=self.coeff_orders,
            input_orders=self.input_orders,
            output_orders=self.output_orders,
        )

    def apply(self, x: ArrayLike) -> np.ndarray:
        return self.matrix @ np.asarray(x, dtype=np.complex128)

    matvec = apply
    multiply = apply
    convolve = apply

    def to_dict(self) -> dict[str, Any]:
        return {
            "coeff_orders": _validate_unique_integer_orders("coeff_orders", self.coeff_orders).tolist(),
            "input_orders": _validate_unique_integer_orders("input_orders", self.input_orders).tolist(),
            "output_orders": _validate_unique_integer_orders(
                "output_orders",
                self.input_orders if self.output_orders is None else self.output_orders,
            ).tolist(),
            "matrix_shape": list(self.matrix.shape),
        }


ConversionMatrix = HarmonicConversionMatrix


def block_diagonal_conversion_matrix(
    coeffs: ArrayLike,
    *,
    coeff_orders: ArrayLike,
    input_orders: ArrayLike,
    output_orders: ArrayLike | None = None,
) -> np.ndarray:
    """Block-diagonal harmonic multiplication matrices for independent branches."""
    kernels = np.asarray(coeffs, dtype=np.complex128)
    if kernels.ndim != 2:
        raise ValueError("coeffs must have shape (n_blocks, n_coeff_orders)")
    blocks = [
        conversion_matrix(
            kernel,
            coeff_orders=coeff_orders,
            input_orders=input_orders,
            output_orders=output_orders,
        )
        for kernel in kernels
    ]
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    out = np.zeros((rows, cols), dtype=np.complex128)
    row = 0
    col = 0
    for block in blocks:
        out[row : row + block.shape[0], col : col + block.shape[1]] = block
        row += block.shape[0]
        col += block.shape[1]
    return out


def signal_idler_conversion_matrix(
    coupling: complex,
    detuning: float = 0.0,
    loss: float = 0.0,
) -> np.ndarray:
    """Reduced DP4WM signal-idler matrix for local coupled-mode diagnostics."""
    if loss < 0.0:
        raise ValueError("loss must be non-negative")
    diagonal = complex(loss, detuning)
    return np.asarray(
        [[diagonal, coupling], [np.conj(coupling), complex(loss, -detuning)]],
        dtype=np.complex128,
    )


__all__.extend(
    [
        "harmonic_multiplication_matrix",
        "conversion_matrix",
        "build_conversion_matrix",
        "mixing_matrix",
        "convolution_matrix",
        "apply_conversion_matrix",
        "convolve_coefficients",
        "HarmonicConversionMatrix",
        "ConversionMatrix",
        "block_diagonal_conversion_matrix",
        "signal_idler_conversion_matrix",
    ]
)
