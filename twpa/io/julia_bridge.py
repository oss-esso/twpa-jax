"""
Reader utilities for Julia/Harmonia simulation outputs.

This module reads the versioned output contract produced by:

    Harmonia.jl/scripts/run_simulation.jl

Expected run folder:

    run_dir/
      status.json
      run_manifest.json
      config_resolved.json
      simulation.h5
      stdout.log
      stderr.log

The goal is not to interpret all physics yet. The goal is to enforce the
status/schema contract and expose clean Python objects for later calibration,
dataset building, Bayesian optimization, SBI, and ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import math

import numpy as np

from twpa.io.hdf5_utils import decode_h5_scalar

from twpa.io.simulation_schema import (
    ALLOWED_STATUSES,
    optional_float,
    optional_int,
    read_json_object,
    validate_status_payload,
)

@dataclass(frozen=True)
class JuliaSimulationStatus:
    schema_version: str
    run_id: str
    status: str
    simulation_type: str
    circuit_template: str
    solver_success: bool
    residual_norm: float | None
    relative_residual_norm: float | None
    failure_reason: str | None
    runtime_s: float | None
    random_seed: int | None
    julia_version: str | None
    josephsoncircuits_version: str | None
    harmonia_commit: str | None
    python_commit: str | None
    h5_path: Path | None
    output_dir: Path | None
    raw: dict[str, Any]

    @property
    def is_pass(self) -> bool:
        return self.status == "PASS"

    @property
    def is_failure(self) -> bool:
        return self.status == "FAIL"

    @property
    def has_finite_residual(self) -> bool:
        if self.residual_norm is None:
            return False
        return math.isfinite(self.residual_norm)


@dataclass(frozen=True)
class JuliaSimulationData:
    status: JuliaSimulationStatus
    frequency_hz: np.ndarray | None
    s_parameters: np.ndarray | None
    gain_db: np.ndarray | None
    h5_attrs: dict[str, Any]



def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))



def read_status_json(path_or_run_dir: str | Path) -> JuliaSimulationStatus:
    """
    Read either a status.json path or a run directory containing status.json.
    """
    path = Path(path_or_run_dir)

    if path.is_dir():
        path = path / "status.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing status JSON: {path}")

    raw = validate_status_payload(read_json_object(path), context=str(path))

    status = str(raw.get("status", "UNKNOWN"))
    residual_norm = optional_float(raw.get("residual_norm"), field_name="residual_norm")
    relative_residual_norm = optional_float(
        raw.get("relative_residual_norm"),
        field_name="relative_residual_norm",
    )
    return JuliaSimulationStatus(
        schema_version=str(raw.get("schema_version", "")),
        run_id=str(raw.get("run_id", "")),
        status=status,
        simulation_type=str(raw.get("simulation_type", "UNKNOWN")),
        circuit_template=str(raw.get("circuit_template", "UNKNOWN")),
        solver_success=bool(raw.get("solver_success", False)),
        residual_norm=residual_norm,
        relative_residual_norm=relative_residual_norm,
        failure_reason=raw.get("failure_reason"),
        runtime_s=optional_float(raw.get("runtime_s"), field_name="runtime_s"),
        random_seed=optional_int(raw.get("random_seed"), field_name="random_seed"),
        julia_version=raw.get("julia_version"),
        josephsoncircuits_version=raw.get("josephsoncircuits_version"),
        harmonia_commit=raw.get("harmonia_commit"),
        python_commit=raw.get("python_commit"),
        h5_path=_optional_path(raw.get("h5_path")),
        output_dir=_optional_path(raw.get("output_dir")),
        raw=raw,
    )


def _decode_h5_attr(value: Any) -> Any:
    return decode_h5_scalar(value)


def read_simulation_h5(path: str | Path) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """
    Read core arrays from simulation.h5.

    Returns:
        frequency_hz, s_parameters, gain_db, attrs

    S-parameters are returned as complex array with shape:

        (n_frequency, port_out, port_in)
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read Julia simulation.h5 files") from exc

    h5_path = Path(path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        attrs = {str(k): _decode_h5_attr(v) for k, v in h5.attrs.items()}

        frequency_hz = None
        if "axes" in h5 and "frequency_hz" in h5["axes"]:
            frequency_hz = np.asarray(h5["axes"]["frequency_hz"][...], dtype=float)

        s_parameters = None
        if "results" in h5 and "S" in h5["results"]:
            s_group = h5["results"]["S"]
            if "real" in s_group and "imag" in s_group:
                s_real = np.asarray(s_group["real"][...], dtype=float)
                s_imag = np.asarray(s_group["imag"][...], dtype=float)

                if s_real.shape != s_imag.shape:
                    raise ValueError(
                        f"S real/imag shape mismatch: {s_real.shape} vs {s_imag.shape}"
                    )

                s_parameters = s_real + 1j * s_imag

                # HDF5 written from Julia may appear in Python/h5py with dimensions ordered as
                # (port_out, port_in, frequency), even though the intended contract is
                # (frequency, port_out, port_in). Normalize here so the Python side always sees
                # the industrial contract: frequency first.
                if frequency_hz is not None and s_parameters.ndim == 3:
                    n_freq = int(frequency_hz.shape[0])

                    # Python contract is always:
                    #   (frequency, port_out, port_in)
                    #
                    # Julia/HDF5 may expose:
                    #   (port_out, port_in, frequency)
                    #
                    # Support both 1-port and 2-port, and later N-port.
                    if s_parameters.shape[0] == n_freq:
                        pass
                    elif s_parameters.shape[-1] == n_freq:
                        s_parameters = np.transpose(s_parameters, (2, 0, 1))
                    else:
                        raise ValueError(
                            "Unsupported S-parameter shape "
                            f"{s_parameters.shape}; expected frequency axis first or last "
                            f"with n_frequency={n_freq}."
                        )

                    if s_parameters.shape[1] != s_parameters.shape[2]:
                        raise ValueError(
                            f"S-parameter port dimensions must be square, got {s_parameters.shape}"
                        )

        gain_db = None
        if "results" in h5 and "gain" in h5["results"]:
            gain_group = h5["results"]["gain"]
            if "gain_db" in gain_group:
                gain_db = np.asarray(gain_group["gain_db"][...], dtype=float)

    return frequency_hz, s_parameters, gain_db, attrs


def load_julia_simulation(run_dir: str | Path) -> JuliaSimulationData:
    """
    Load a Julia simulation run folder.

    This function enforces:
      - status.json exists;
      - PASS runs have simulation.h5;
      - HDF5 arrays are finite where present;
      - S-parameters have shape (frequency, port_out, port_in).
    """
    run_path = Path(run_dir)
    status = read_status_json(run_path)

    h5_path = status.h5_path
    if h5_path is None:
        h5_path = run_path / "simulation.h5"

    if status.status == "PASS" and not h5_path.exists():
        raise FileNotFoundError(f"PASS run is missing simulation.h5: {h5_path}")

    if not h5_path.exists():
        return JuliaSimulationData(
            status=status,
            frequency_hz=None,
            s_parameters=None,
            gain_db=None,
            h5_attrs={},
        )

    frequency_hz, s_parameters, gain_db, h5_attrs = read_simulation_h5(h5_path)

    if frequency_hz is not None and not np.all(np.isfinite(frequency_hz)):
        raise ValueError("frequency_hz contains non-finite values")

    if s_parameters is not None:
        if s_parameters.ndim != 3:
            raise ValueError(f"S-parameters must have shape (freq, out, in), got {s_parameters.shape}")
        if not np.all(np.isfinite(s_parameters.real)) or not np.all(np.isfinite(s_parameters.imag)):
            raise ValueError("S-parameters contain non-finite values")

    if gain_db is not None and not np.all(np.isfinite(gain_db)):
        raise ValueError("gain_db contains non-finite values")

    return JuliaSimulationData(
        status=status,
        frequency_hz=frequency_hz,
        s_parameters=s_parameters,
        gain_db=gain_db,
        h5_attrs=h5_attrs,
    )
