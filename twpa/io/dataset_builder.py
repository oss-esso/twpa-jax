"""
Dataset builders for Julia/Harmonia simulation campaigns.

This module turns registered Julia run folders into ML-ready arrays.

For now we support the first real physics dataset:

    simulation_type = linear_sparams

Output NPZ convention:

    parameter_names        (P,)
    parameters             (N, P)
    frequency_hz           (F,)
    s_real                 (N, F, 2, 2)
    s_imag                 (N, F, 2, 2)
    gain_db                (N, F)
    run_ids                (N,)
    output_dirs            (N,)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np

from twpa.io.julia_bridge import load_julia_simulation
from twpa.io.run_registry import read_registry
from twpa.io.simulation_schema import (
    SCHEMA_VERSION,
    compute_two_port_metrics,
    read_json_object,
    write_json,
)


DEFAULT_LINEAR_PARAMETER_NAMES = [
    "z_ref_ohm",
    "z_line_ohm",
    "length_m",
    "phase_velocity_m_per_s",
    "attenuation_np_per_m",
]


@dataclass(frozen=True)
class BuiltDataset:
    dataset_npz: Path
    summary_json: Path
    n_samples: int
    n_frequency: int
    parameter_names: list[str]


def _as_float(value: Any, *, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required parameter {name!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"Parameter {name!r} is non-finite: {out}")
    return out


def read_resolved_config(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "config_resolved.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config_resolved.json in run directory: {run_dir}")
    return read_json_object(path)


def extract_parameter_vector(
    config: dict[str, Any],
    *,
    parameter_names: Iterable[str] = DEFAULT_LINEAR_PARAMETER_NAMES,
) -> np.ndarray:
    params = config.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError("config['parameters'] must be an object")

    values = [
        _as_float(params.get(name), name=name)
        for name in parameter_names
    ]
    return np.asarray(values, dtype=float)


def _require_same_frequency(reference: np.ndarray | None, current: np.ndarray, *, run_dir: Path) -> np.ndarray:
    current = np.asarray(current, dtype=float)

    if reference is None:
        return current

    if current.shape != reference.shape:
        raise ValueError(
            f"Frequency shape mismatch for {run_dir}: {current.shape} vs {reference.shape}"
        )

    if not np.allclose(current, reference, rtol=0.0, atol=0.0):
        raise ValueError(f"Frequency axis mismatch for {run_dir}")

    return reference


def build_linear_sparams_dataset(
    *,
    registry_csv: str | Path,
    output_dir: str | Path,
    parameter_names: Iterable[str] = DEFAULT_LINEAR_PARAMETER_NAMES,
    require_pass: bool = True,
) -> BuiltDataset:
    registry_csv = Path(registry_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_names = list(parameter_names)
    rows = read_registry(registry_csv)

    if not rows:
        raise ValueError(f"Registry is empty: {registry_csv}")

    selected_rows = []
    for row in rows:
        if row.get("simulation_type") != "linear_sparams":
            continue
        if require_pass and row.get("status") != "PASS":
            continue
        selected_rows.append(row)

    if not selected_rows:
        raise ValueError(f"No usable linear_sparams runs found in registry: {registry_csv}")

    run_ids: list[str] = []
    output_dirs: list[str] = []
    parameter_vectors: list[np.ndarray] = []
    s_arrays: list[np.ndarray] = []
    gain_arrays: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    frequency_reference: np.ndarray | None = None

    for row in selected_rows:
        run_dir = Path(row["output_dir"])
        data = load_julia_simulation(run_dir)

        if data.status.status != "PASS" and require_pass:
            raise ValueError(f"Expected PASS run, got {data.status.status}: {run_dir}")
        if data.status.simulation_type != "linear_sparams":
            raise ValueError(f"Expected linear_sparams run, got {data.status.simulation_type}: {run_dir}")
        if data.frequency_hz is None:
            raise ValueError(f"Missing frequency axis: {run_dir}")
        if data.s_parameters is None:
            raise ValueError(f"Missing S-parameters: {run_dir}")
        if data.gain_db is None:
            raise ValueError(f"Missing gain_db: {run_dir}")

        frequency_reference = _require_same_frequency(
            frequency_reference,
            data.frequency_hz,
            run_dir=run_dir,
        )

        config = read_resolved_config(run_dir)
        parameter_vector = extract_parameter_vector(
            config,
            parameter_names=parameter_names,
        )

        two_port_metrics = compute_two_port_metrics(
            frequency_hz=data.frequency_hz,
            s_parameters=data.s_parameters,
            gain_db=data.gain_db,
        ).to_dict()

        run_ids.append(data.status.run_id)
        output_dirs.append(str(run_dir))
        parameter_vectors.append(parameter_vector)
        s_arrays.append(np.asarray(data.s_parameters, dtype=np.complex128))
        gain_arrays.append(np.asarray(data.gain_db, dtype=float))
        metrics.append(two_port_metrics)

    assert frequency_reference is not None

    parameters = np.stack(parameter_vectors, axis=0)
    s_complex = np.stack(s_arrays, axis=0)
    gain_db = np.stack(gain_arrays, axis=0)

    dataset_npz = output_dir / "linear_sparams_dataset.npz"
    summary_json = output_dir / "dataset_summary.json"

    np.savez_compressed(
        dataset_npz,
        schema_version=np.asarray(SCHEMA_VERSION),
        parameter_names=np.asarray(parameter_names),
        parameters=parameters,
        frequency_hz=frequency_reference,
        s_real=s_complex.real,
        s_imag=s_complex.imag,
        gain_db=gain_db,
        run_ids=np.asarray(run_ids),
        output_dirs=np.asarray(output_dirs),
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": "linear_sparams",
        "registry_csv": str(registry_csv),
        "dataset_npz": str(dataset_npz),
        "n_samples": int(parameters.shape[0]),
        "n_parameters": int(parameters.shape[1]),
        "parameter_names": parameter_names,
        "n_frequency": int(frequency_reference.shape[0]),
        "frequency_min_hz": float(np.min(frequency_reference)),
        "frequency_max_hz": float(np.max(frequency_reference)),
        "s_shape": list(s_complex.shape),
        "gain_db_shape": list(gain_db.shape),
        "run_ids": run_ids,
        "output_dirs": output_dirs,
        "metrics": metrics,
    }

    write_json(summary_json, summary)

    return BuiltDataset(
        dataset_npz=dataset_npz,
        summary_json=summary_json,
        n_samples=int(parameters.shape[0]),
        n_frequency=int(frequency_reference.shape[0]),
        parameter_names=parameter_names,
    )


def load_linear_sparams_dataset(dataset_npz: str | Path) -> dict[str, np.ndarray]:
    path = Path(dataset_npz)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset NPZ: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}