"""
Shared simulation schema utilities.

This module is the Python-side contract for all Julia/Harmonia simulation runs.
It deliberately contains no HB physics. It defines status semantics, schema
versioning, JSON safety, and small validation helpers used by readers, runners,
registries, campaigns, Bayesian calibration, SBI, and ML dataset builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import json
import math

import numpy as np


SCHEMA_VERSION = "0.1.0"


class SimulationStatus(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


ALLOWED_STATUSES = {status.value for status in SimulationStatus}


class SimulationSchemaError(ValueError):
    """Raised when a simulation artifact violates the shared schema."""


@dataclass(frozen=True)
class TwoPortMetrics:
    frequency_points: int
    frequency_min_hz: float
    frequency_max_hz: float
    s_shape: tuple[int, int, int]
    max_abs_s11: float
    max_abs_s22: float
    max_abs_s21: float
    min_abs_s21: float
    max_abs_s12: float
    min_abs_s12: float
    reciprocal_error_max_abs: float
    passivity_max_singular_value: float
    gain_db_min: float | None
    gain_db_max: float | None
    all_arrays_finite: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency_points": self.frequency_points,
            "frequency_min_hz": self.frequency_min_hz,
            "frequency_max_hz": self.frequency_max_hz,
            "s_shape": list(self.s_shape),
            "max_abs_s11": self.max_abs_s11,
            "max_abs_s22": self.max_abs_s22,
            "max_abs_s21": self.max_abs_s21,
            "min_abs_s21": self.min_abs_s21,
            "max_abs_s12": self.max_abs_s12,
            "min_abs_s12": self.min_abs_s12,
            "reciprocal_error_max_abs": self.reciprocal_error_max_abs,
            "passivity_max_singular_value": self.passivity_max_singular_value,
            "gain_db_min": self.gain_db_min,
            "gain_db_max": self.gain_db_max,
            "all_arrays_finite": self.all_arrays_finite,
        }


def optional_float(value: Any, *, field_name: str = "value") -> float | None:
    if value is None:
        return None

    out = float(value)

    if not math.isfinite(out):
        raise SimulationSchemaError(f"{field_name} must be finite or null, got {out!r}")

    return out


def optional_int(value: Any, *, field_name: str = "value") -> int | None:
    if value is None:
        return None
    return int(value)


def assert_json_serializable(obj: Any, *, context: str = "object") -> None:
    try:
        json.dumps(obj)
    except TypeError as exc:
        raise SimulationSchemaError(f"{context} is not JSON serializable: {exc}") from exc


def write_json(path: str | Path, obj: Mapping[str, Any], *, indent: int = 2) -> None:
    assert_json_serializable(obj, context=str(path))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def read_json_object(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise SimulationSchemaError(f"Expected JSON object in {p}, got {type(obj)!r}")

    return obj


def validate_status_payload(payload: Mapping[str, Any], *, context: str = "status payload") -> dict[str, Any]:
    """
    Validate a status.json-like payload.

    Rules
    -----
    - status must be PASS/PARTIAL/FAIL/UNKNOWN.
    - PASS requires solver_success=true.
    - PASS must not contain failure_reason.
    - PASS/PARTIAL/FAIL/UNKNOWN may use null residuals.
    - If residuals are present, they must be finite.
    - schema_version is warned by exception only if present and wrong.
    """
    raw = dict(payload)

    schema_version = raw.get("schema_version")
    if schema_version is not None and str(schema_version) != SCHEMA_VERSION:
        raise SimulationSchemaError(
            f"{context}: unsupported schema_version={schema_version!r}; expected {SCHEMA_VERSION!r}"
        )

    status = str(raw.get("status", SimulationStatus.UNKNOWN.value))
    if status not in ALLOWED_STATUSES:
        raise SimulationSchemaError(f"{context}: invalid status {status!r}")

    solver_success = bool(raw.get("solver_success", False))

    residual_norm = optional_float(raw.get("residual_norm"), field_name="residual_norm")
    relative_residual_norm = optional_float(
        raw.get("relative_residual_norm"),
        field_name="relative_residual_norm",
    )

    failure_reason = raw.get("failure_reason")

    if status == SimulationStatus.PASS.value:
        if not solver_success:
            raise SimulationSchemaError(f"{context}: PASS requires solver_success=true")

        if failure_reason not in (None, "", "null"):
            raise SimulationSchemaError(f"{context}: PASS cannot have failure_reason={failure_reason!r}")

    raw["status"] = status
    raw["solver_success"] = solver_success
    raw["residual_norm"] = residual_norm
    raw["relative_residual_norm"] = relative_residual_norm

    return raw


def classify_status(
    *,
    solver_success: bool,
    arrays_finite: bool,
    residual_norm: float | None = None,
    residual_tolerance: float | None = None,
    is_fallback: bool = False,
    validation_passed: bool = True,
) -> SimulationStatus:
    """
    Conservative status classifier.

    This is meant for Python-side checks and future Julia status discipline.

    PASS:
      solver success, finite arrays, not fallback, validation passed, and if a
      residual tolerance is supplied the residual is finite and below tolerance.

    PARTIAL:
      solver success but fallback/reduced/insufficient validation.

    FAIL:
      solver failed or arrays are nonfinite.

    UNKNOWN:
      reserved for missing metadata, not normally returned here.
    """
    if not solver_success:
        return SimulationStatus.FAIL

    if not arrays_finite:
        return SimulationStatus.FAIL

    if residual_tolerance is not None:
        if residual_norm is None or not math.isfinite(float(residual_norm)):
            return SimulationStatus.FAIL
        if float(residual_norm) > residual_tolerance:
            return SimulationStatus.FAIL

    if is_fallback or not validation_passed:
        return SimulationStatus.PARTIAL

    return SimulationStatus.PASS


def normalize_s_parameter_shape(
    s_parameters: np.ndarray,
    *,
    n_frequency: int,
    n_ports: int = 2,
) -> np.ndarray:
    """
    Normalize S-parameter arrays to:

        (frequency, port_out, port_in)

    Julia/HDF5 may expose arrays to h5py as either:
      - (frequency, port_out, port_in), or
      - (port_out, port_in, frequency)

    This function returns a normalized array or raises.
    """
    s = np.asarray(s_parameters)

    expected = (n_frequency, n_ports, n_ports)
    julia_seen = (n_ports, n_ports, n_frequency)

    if s.shape == expected:
        return s

    if s.shape == julia_seen:
        return np.transpose(s, (2, 0, 1))

    raise SimulationSchemaError(
        f"Unsupported S-parameter shape {s.shape}; expected {expected} or {julia_seen}"
    )


def compute_two_port_metrics(
    *,
    frequency_hz: np.ndarray,
    s_parameters: np.ndarray,
    gain_db: np.ndarray | None = None,
) -> TwoPortMetrics:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    s = np.asarray(s_parameters, dtype=np.complex128)

    if frequency_hz.ndim != 1:
        raise SimulationSchemaError(f"frequency_hz must be 1D, got {frequency_hz.shape}")

    s = normalize_s_parameter_shape(s, n_frequency=frequency_hz.shape[0], n_ports=2)

    if gain_db is not None:
        gain_db = np.asarray(gain_db, dtype=float)
        if gain_db.shape != frequency_hz.shape:
            raise SimulationSchemaError(
                f"gain_db shape {gain_db.shape} does not match frequency shape {frequency_hz.shape}"
            )

    s11 = s[:, 0, 0]
    s12 = s[:, 0, 1]
    s21 = s[:, 1, 0]
    s22 = s[:, 1, 1]

    singular_values = np.linalg.svd(s, compute_uv=False)

    arrays_finite = (
        np.all(np.isfinite(frequency_hz))
        and np.all(np.isfinite(s.real))
        and np.all(np.isfinite(s.imag))
        and (gain_db is None or np.all(np.isfinite(gain_db)))
    )

    return TwoPortMetrics(
        frequency_points=int(frequency_hz.shape[0]),
        frequency_min_hz=float(np.min(frequency_hz)),
        frequency_max_hz=float(np.max(frequency_hz)),
        s_shape=tuple(int(x) for x in s.shape),
        max_abs_s11=float(np.max(np.abs(s11))),
        max_abs_s22=float(np.max(np.abs(s22))),
        max_abs_s21=float(np.max(np.abs(s21))),
        min_abs_s21=float(np.min(np.abs(s21))),
        max_abs_s12=float(np.max(np.abs(s12))),
        min_abs_s12=float(np.min(np.abs(s12))),
        reciprocal_error_max_abs=float(np.max(np.abs(s21 - s12))),
        passivity_max_singular_value=float(np.max(singular_values)),
        gain_db_min=float(np.min(gain_db)) if gain_db is not None else None,
        gain_db_max=float(np.max(gain_db)) if gain_db is not None else None,
        all_arrays_finite=bool(arrays_finite),
    )