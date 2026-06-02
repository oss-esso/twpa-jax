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


DEFAULT_JC_JPA_REFLECTION_PARAMETER_NAMES = [
    "R_ohm",
    "Cc_F",
    "Lj_H",
    "Cj_F",
    "pump_frequency_hz",
    "pump_current_a",
    "n_pump_harmonics",
    "n_modulation_harmonics",
]

DEFAULT_HARMONIA_JTL_LINEAR_PARAMETER_NAMES = [
    "N_cell",
    "Cg_F",
    "Lj_H",
    "Cj_F",
    "port_impedance_ohm",
    "pump_frequency_hz",
    "pump_current_a",
    "n_pump_harmonics",
    "n_modulation_harmonics",
]
DEFAULT_HARMONIA_RF_JTL_LINEAR_PARAMETER_NAMES = DEFAULT_HARMONIA_JTL_LINEAR_PARAMETER_NAMES + ["Lrf_H","Lp_H"]

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


def build_jc_jpa_reflection_dataset(
    *,
    registry_csv: str | Path,
    output_dir: str | Path,
    parameter_names: Iterable[str] = DEFAULT_JC_JPA_REFLECTION_PARAMETER_NAMES,
    require_pass: bool = True,
) -> BuiltDataset:
    """
    Build an ML-ready dataset from a JosephsonCircuits JPA reflection campaign.

    Expected simulation type:
        jc_jpa_reflection_smoke

    Output NPZ convention:
        parameter_names        (P,)
        parameters             (N, P)
        frequency_hz           (F,)
        s11_real               (N, F)
        s11_imag               (N, F)
        reflection_db          (N, F)
        run_ids                (N,)
        output_dirs            (N,)
    """
    registry_csv = Path(registry_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_names = list(parameter_names)
    rows = read_registry(registry_csv)

    if not rows:
        raise ValueError(f"Registry is empty: {registry_csv}")

    selected_rows = []
    for row in rows:
        if row.get("simulation_type") != "jc_jpa_reflection_smoke":
            continue
        if require_pass and row.get("status") != "PASS":
            continue
        selected_rows.append(row)

    if not selected_rows:
        raise ValueError(
            f"No usable jc_jpa_reflection_smoke runs found in registry: {registry_csv}"
        )

    run_ids: list[str] = []
    output_dirs: list[str] = []
    parameter_vectors: list[np.ndarray] = []
    s11_arrays: list[np.ndarray] = []
    reflection_arrays: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    frequency_reference: np.ndarray | None = None

    for row in selected_rows:
        run_dir = Path(row["output_dir"])
        data = load_julia_simulation(run_dir)

        if data.status.status != "PASS" and require_pass:
            raise ValueError(f"Expected PASS run, got {data.status.status}: {run_dir}")
        if data.status.simulation_type != "jc_jpa_reflection_smoke":
            raise ValueError(
                f"Expected jc_jpa_reflection_smoke run, got "
                f"{data.status.simulation_type}: {run_dir}"
            )
        if data.frequency_hz is None:
            raise ValueError(f"Missing frequency axis: {run_dir}")
        if data.s_parameters is None:
            raise ValueError(f"Missing S-parameters: {run_dir}")
        if data.gain_db is None:
            raise ValueError(f"Missing reflection_db/gain_db curve: {run_dir}")

        s = np.asarray(data.s_parameters, dtype=np.complex128)

        if s.ndim != 3 or s.shape[1:] != (1, 1):
            raise ValueError(f"Expected one-port S shape (frequency, 1, 1), got {s.shape}")

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

        s11 = s[:, 0, 0]
        reflection_db = np.asarray(data.gain_db, dtype=float)

        if reflection_db.shape != data.frequency_hz.shape:
            raise ValueError(
                f"reflection_db shape {reflection_db.shape} does not match "
                f"frequency shape {data.frequency_hz.shape}: {run_dir}"
            )

        if not np.all(np.isfinite(s11.real)) or not np.all(np.isfinite(s11.imag)):
            raise ValueError(f"S11 contains non-finite values: {run_dir}")
        if not np.all(np.isfinite(reflection_db)):
            raise ValueError(f"reflection_db contains non-finite values: {run_dir}")

        peak_idx = int(np.argmax(reflection_db))
        dip_idx = int(np.argmin(reflection_db))

        one_port_metrics = {
            "frequency_points": int(data.frequency_hz.shape[0]),
            "frequency_min_hz": float(np.min(data.frequency_hz)),
            "frequency_max_hz": float(np.max(data.frequency_hz)),
            "s_shape": list(s.shape),
            "max_abs_s11": float(np.max(np.abs(s11))),
            "min_abs_s11": float(np.min(np.abs(s11))),
            "reflection_db_min": float(np.min(reflection_db)),
            "reflection_db_max": float(np.max(reflection_db)),
            "reflection_db_peak_to_peak": float(np.max(reflection_db) - np.min(reflection_db)),
            "reflection_peak_frequency_hz": float(data.frequency_hz[peak_idx]),
            "reflection_dip_frequency_hz": float(data.frequency_hz[dip_idx]),
            "all_arrays_finite": bool(
                np.all(np.isfinite(data.frequency_hz))
                and np.all(np.isfinite(s11.real))
                and np.all(np.isfinite(s11.imag))
                and np.all(np.isfinite(reflection_db))
            ),
        }

        run_ids.append(data.status.run_id)
        output_dirs.append(str(run_dir))
        parameter_vectors.append(parameter_vector)
        s11_arrays.append(s11)
        reflection_arrays.append(reflection_db)
        metrics.append(one_port_metrics)

    assert frequency_reference is not None

    parameters = np.stack(parameter_vectors, axis=0)
    s11_complex = np.stack(s11_arrays, axis=0)
    reflection_db = np.stack(reflection_arrays, axis=0)

    dataset_npz = output_dir / "jc_jpa_reflection_dataset.npz"
    summary_json = output_dir / "dataset_summary.json"

    np.savez_compressed(
        dataset_npz,
        schema_version=np.asarray(SCHEMA_VERSION),
        dataset_type=np.asarray("jc_jpa_reflection_smoke"),
        parameter_names=np.asarray(parameter_names),
        parameters=parameters,
        frequency_hz=frequency_reference,
        s11_real=s11_complex.real,
        s11_imag=s11_complex.imag,
        reflection_db=reflection_db,
        run_ids=np.asarray(run_ids),
        output_dirs=np.asarray(output_dirs),
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": "jc_jpa_reflection_smoke",
        "registry_csv": str(registry_csv),
        "dataset_npz": str(dataset_npz),
        "n_samples": int(parameters.shape[0]),
        "n_parameters": int(parameters.shape[1]),
        "parameter_names": parameter_names,
        "n_frequency": int(frequency_reference.shape[0]),
        "frequency_min_hz": float(np.min(frequency_reference)),
        "frequency_max_hz": float(np.max(frequency_reference)),
        "s11_shape": list(s11_complex.shape),
        "reflection_db_shape": list(reflection_db.shape),
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


def build_harmonia_jtl_linear_dataset(
    *,
    registry_csv: str | Path,
    output_dir: str | Path,
    parameter_names: Iterable[str] = DEFAULT_HARMONIA_JTL_LINEAR_PARAMETER_NAMES,
    require_pass: bool = True,
) -> BuiltDataset:
    """
    Build an ML-ready dataset from a Harmonia CircuitIR JTL linear JC campaign.

    Expected simulation type:
        harmonia_jtl_linear_jc_smoke

    Output NPZ convention:
        parameter_names    (P,)
        parameters         (N, P)
        frequency_hz       (F,)
        s_real             (N, F, 2, 2)
        s_imag             (N, F, 2, 2)
        gain_db            (N, F)
        run_ids            (N,)
        output_dirs        (N,)
    """
    registry_csv = Path(registry_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_names = list(parameter_names)
    rows = read_registry(registry_csv)

    if not rows:
        raise ValueError(f"Registry is empty: {registry_csv}")

    selected_rows = []
    for row in rows:
        if row.get("simulation_type") != "harmonia_jtl_linear_jc_smoke":
            continue
        if require_pass and row.get("status") != "PASS":
            continue
        selected_rows.append(row)

    if not selected_rows:
        raise ValueError(
            f"No usable harmonia_jtl_linear_jc_smoke runs found in registry: {registry_csv}"
        )

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

        if data.status.simulation_type != "harmonia_jtl_linear_jc_smoke":
            raise ValueError(
                f"Expected harmonia_jtl_linear_jc_smoke run, got "
                f"{data.status.simulation_type}: {run_dir}"
            )

        if data.frequency_hz is None:
            raise ValueError(f"Missing frequency axis: {run_dir}")
        if data.s_parameters is None:
            raise ValueError(f"Missing S-parameters: {run_dir}")
        if data.gain_db is None:
            raise ValueError(f"Missing gain_db curve: {run_dir}")

        s = np.asarray(data.s_parameters, dtype=np.complex128)
        gain_db = np.asarray(data.gain_db, dtype=float)

        if s.ndim != 3 or s.shape[1:] != (2, 2):
            raise ValueError(f"Expected 2-port S shape (frequency, 2, 2), got {s.shape}")

        if gain_db.shape != data.frequency_hz.shape:
            raise ValueError(
                f"gain_db shape {gain_db.shape} does not match "
                f"frequency shape {data.frequency_hz.shape}: {run_dir}"
            )

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

        if not np.all(np.isfinite(s.real)) or not np.all(np.isfinite(s.imag)):
            raise ValueError(f"S-parameters contain non-finite values: {run_dir}")
        if not np.all(np.isfinite(gain_db)):
            raise ValueError(f"gain_db contains non-finite values: {run_dir}")

        one_run_metrics = compute_two_port_metrics(
            frequency_hz=data.frequency_hz,
            s_parameters=s,
            gain_db=gain_db,
        ).to_dict()

        run_ids.append(data.status.run_id)
        output_dirs.append(str(run_dir))
        parameter_vectors.append(parameter_vector)
        s_arrays.append(s)
        gain_arrays.append(gain_db)
        metrics.append(one_run_metrics)

    assert frequency_reference is not None

    parameters = np.stack(parameter_vectors, axis=0)
    s_complex = np.stack(s_arrays, axis=0)
    gain_db_array = np.stack(gain_arrays, axis=0)

    dataset_npz = output_dir / "harmonia_jtl_linear_dataset.npz"
    summary_json = output_dir / "dataset_summary.json"

    np.savez_compressed(
        dataset_npz,
        schema_version=np.asarray(SCHEMA_VERSION),
        dataset_type=np.asarray("harmonia_jtl_linear_jc_smoke"),
        parameter_names=np.asarray(parameter_names),
        parameters=parameters,
        frequency_hz=frequency_reference,
        s_real=s_complex.real,
        s_imag=s_complex.imag,
        gain_db=gain_db_array,
        run_ids=np.asarray(run_ids),
        output_dirs=np.asarray(output_dirs),
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": "harmonia_jtl_linear_jc_smoke",
        "registry_csv": str(registry_csv),
        "dataset_npz": str(dataset_npz),
        "n_samples": int(parameters.shape[0]),
        "n_parameters": int(parameters.shape[1]),
        "parameter_names": parameter_names,
        "n_frequency": int(frequency_reference.shape[0]),
        "frequency_min_hz": float(np.min(frequency_reference)),
        "frequency_max_hz": float(np.max(frequency_reference)),
        "s_shape": list(s_complex.shape),
        "gain_db_shape": list(gain_db_array.shape),
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

def build_harmonia_rf_jtl_linear_dataset(*, registry_csv, output_dir, parameter_names=DEFAULT_HARMONIA_RF_JTL_LINEAR_PARAMETER_NAMES, require_pass=True):
    return build_two_port_dataset(registry_csv=registry_csv,output_dir=output_dir,simulation_type="harmonia_rf_jtl_linear_jc_smoke",
        dataset_filename="harmonia_rf_jtl_linear_dataset.npz",parameter_names=parameter_names,require_pass=require_pass)

def build_two_port_dataset(*, registry_csv, output_dir, simulation_type, dataset_filename, parameter_names, require_pass=True):
    registry_csv=Path(registry_csv); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    rows=[r for r in read_registry(registry_csv) if r.get("simulation_type")==simulation_type and (not require_pass or r.get("status")=="PASS")]
    if not rows: raise ValueError(f"No usable {simulation_type} runs found in registry: {registry_csv}")
    params=[]; ss=[]; gains=[]; ids=[]; dirs=[]; metrics=[]; freq=None
    for row in rows:
        run=Path(row["output_dir"]); data=load_julia_simulation(run)
        if data.frequency_hz is None or data.s_parameters is None or data.gain_db is None: raise ValueError(f"Missing arrays: {run}")
        freq=_require_same_frequency(freq,data.frequency_hz,run_dir=run); params.append(extract_parameter_vector(read_resolved_config(run),parameter_names=parameter_names))
        ss.append(data.s_parameters); gains.append(data.gain_db); ids.append(data.status.run_id); dirs.append(str(run))
        metrics.append(compute_two_port_metrics(frequency_hz=data.frequency_hz,s_parameters=data.s_parameters,gain_db=data.gain_db).to_dict())
    p=np.stack(params); s=np.stack(ss); g=np.stack(gains); npz=output_dir/dataset_filename; summary=output_dir/"dataset_summary.json"
    np.savez_compressed(npz,parameter_names=np.asarray(list(parameter_names)),parameters=p,frequency_hz=freq,s_real=s.real,s_imag=s.imag,gain_db=g,run_ids=np.asarray(ids),output_dirs=np.asarray(dirs))
    write_json(summary,{"dataset_type":simulation_type,"dataset_npz":str(npz),"n_samples":len(rows),"parameter_names":list(parameter_names),"parameter_shape":list(p.shape),"frequency_shape":list(freq.shape),"s_shape":list(s.shape),"gain_db_shape":list(g.shape),"metrics":metrics})
    return BuiltDataset(npz,summary,len(rows),len(freq),list(parameter_names))


def load_harmonia_jtl_linear_dataset(dataset_npz: str | Path) -> dict[str, np.ndarray]:
    path = Path(dataset_npz)

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset NPZ: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def load_jc_jpa_reflection_dataset(dataset_npz: str | Path) -> dict[str, np.ndarray]:
    path = Path(dataset_npz)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset NPZ: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_linear_sparams_dataset(dataset_npz: str | Path) -> dict[str, np.ndarray]:
    path = Path(dataset_npz)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset NPZ: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
load_harmonia_rf_jtl_linear_dataset = load_harmonia_jtl_linear_dataset
