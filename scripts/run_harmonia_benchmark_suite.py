from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpa.io.julia_bridge import load_julia_simulation
from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.julia_batch_runner import run_harmonia_simulation_batch
from twpa.io.simulation_schema import SCHEMA_VERSION, write_json
from twpa.io.topology_artifacts import load_topology_artifact


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    config_filename: str
    architecture: str
    circuit_family: str
    expected_simulation_type: str
    expected_circuit_template: str
    timeout_s: float
    notes: str


MODERN_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="jtl_linear",
        config_filename="harmonia_jtl_linear_jc_smoke.json",
        architecture="CircuitIR",
        circuit_family="JTL",
        expected_simulation_type="harmonia_jtl_linear_jc_smoke",
        expected_circuit_template="circuit_ir_jtl_chain_linear_jc",
        timeout_s=180.0,
        notes="CircuitIR JTL template, bounded linear two-port solve.",
    ),
    BenchmarkCase(
        name="rf_jtl_linear",
        config_filename="harmonia_rf_jtl_linear_jc_smoke.json",
        architecture="CircuitIR",
        circuit_family="RF-JTL",
        expected_simulation_type="harmonia_rf_jtl_linear_jc_smoke",
        expected_circuit_template="circuit_ir_rf_jtl_chain_linear_jc",
        timeout_s=180.0,
        notes="CircuitIR RF-JTL template, bounded linear two-port solve.",
    ),
    BenchmarkCase(
        name="ethz_jtl_linear",
        config_filename="harmonia_ethz_jtl_linear_jc_smoke.json",
        architecture="CircuitIR",
        circuit_family="ETHZ-JTL",
        expected_simulation_type="harmonia_ethz_jtl_linear_jc_smoke",
        expected_circuit_template="circuit_ir_ethz_jtl_chain_linear_jc",
        timeout_s=240.0,
        notes="CircuitIR ETHZ JTL template, bounded linear two-port solve.",
    ),
    BenchmarkCase(
        name="lumped_jpa_linear",
        config_filename="harmonia_lumped_jpa_linear_jc_smoke.json",
        architecture="CircuitIR",
        circuit_family="JPA",
        expected_simulation_type="harmonia_lumped_jpa_linear_jc_smoke",
        expected_circuit_template="circuit_ir_lumped_jpa_reflection_linear_jc",
        timeout_s=180.0,
        notes="CircuitIR lumped JPA reflection template, bounded one-port solve.",
    ),
    BenchmarkCase(
        name="tiny_nonlinear_hb",
        config_filename="harmonia_tiny_nonlinear_hb_smoke.json",
        architecture="CircuitIR",
        circuit_family="tiny-nonlinear",
        expected_simulation_type="harmonia_tiny_nonlinear_hb_smoke",
        expected_circuit_template="tiny_nonlinear_hb",
        timeout_s=240.0,
        notes="Tiny nonlinear HB smoke, finite bounded output.",
    ),
]

LEGACY_OR_REFERENCE_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="matched_tl_linear_reference",
        config_filename="linear_sparams_matched_tl.json",
        architecture="reference",
        circuit_family="matched-TL",
        expected_simulation_type="linear_sparams",
        expected_circuit_template="matched_transmission_line",
        timeout_s=120.0,
        notes="Simple linear reference case. Useful for runner/schema timing.",
    ),
    BenchmarkCase(
        name="jc_jpa_reflection_legacy",
        config_filename="jc_jpa_reflection_smoke.json",
        architecture="legacy-jc",
        circuit_family="JPA",
        expected_simulation_type="jc_jpa_reflection_smoke",
        expected_circuit_template="one_port_jpa_reflection",
        timeout_s=180.0,
        notes="Legacy/raw JosephsonCircuits JPA smoke retained as a comparison baseline.",
    ),
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def safe_git_status_short(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    return result.stdout


def collect_environment(
    *,
    harmonia_root: Path,
    twpa_root: Path,
    josephsoncircuits_root: Path,
    julia_executable: str,
) -> dict[str, Any]:
    julia_version: str | None = None

    try:
        result = subprocess.run(
            [julia_executable, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        if result.returncode == 0:
            julia_version = result.stdout.strip()
    except Exception:
        julia_version = None

    return {
        "created_utc": utc_timestamp(),
        "schema_version": SCHEMA_VERSION,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "julia_executable": julia_executable,
        "julia_version": julia_version,
        "workspace_root": str(_WORKSPACE_ROOT),
        "twpa_jax_root": str(twpa_root),
        "harmonia_root": str(harmonia_root),
        "josephsoncircuits_root": str(josephsoncircuits_root),
        "git": {
            "twpa_jax_commit": safe_git_commit(twpa_root),
            "harmonia_commit": safe_git_commit(harmonia_root),
            "josephsoncircuits_commit": safe_git_commit(josephsoncircuits_root),
            "twpa_jax_status_short": safe_git_status_short(twpa_root),
            "harmonia_status_short": safe_git_status_short(harmonia_root),
            "josephsoncircuits_status_short": safe_git_status_short(josephsoncircuits_root),
        },
    }


def select_cases(*, suite: str, include_legacy: bool) -> list[BenchmarkCase]:
    if suite == "minimal":
        cases = [
            case
            for case in MODERN_CASES
            if case.name in {"jtl_linear", "lumped_jpa_linear"}
        ]
    elif suite == "modern":
        cases = list(MODERN_CASES)
    elif suite == "all":
        cases = list(MODERN_CASES) + list(LEGACY_OR_REFERENCE_CASES)
    else:
        raise ValueError(f"Unknown suite: {suite}")

    if include_legacy and suite != "all":
        cases += list(LEGACY_OR_REFERENCE_CASES)

    return cases


def _shape_or_none(array: np.ndarray | None) -> list[int] | None:
    if array is None:
        return None
    return [int(x) for x in array.shape]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def collect_run_metrics(
    *,
    case: BenchmarkCase,
    run_dir: Path,
    python_wall_time_s: float,
    repetition: int,
    config_path: Path,
) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    h5_path = run_dir / "simulation.h5"

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_name": case.name,
        "architecture": case.architecture,
        "circuit_family": case.circuit_family,
        "repetition": int(repetition),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "run_dir": str(run_dir),
        "status_json": str(status_path) if status_path.exists() else None,
        "h5_path": str(h5_path) if h5_path.exists() else None,
        "h5_size_bytes": h5_path.stat().st_size if h5_path.exists() else None,
        "python_wall_time_s": float(python_wall_time_s),
        "expected_simulation_type": case.expected_simulation_type,
        "expected_circuit_template": case.expected_circuit_template,
        "case_notes": case.notes,
    }

    try:
        data = load_julia_simulation(run_dir)

        row.update(
            {
                "status": data.status.status,
                "run_id": data.status.run_id,
                "simulation_type": data.status.simulation_type,
                "circuit_template": data.status.circuit_template,
                "solver_success": data.status.solver_success,
                "failure_reason": data.status.failure_reason,
                "julia_status_runtime_s": data.status.runtime_s,
                "residual_norm": data.status.residual_norm,
                "relative_residual_norm": data.status.relative_residual_norm,
                "frequency_shape": _shape_or_none(data.frequency_hz),
                "frequency_points": (
                    int(data.frequency_hz.shape[0])
                    if data.frequency_hz is not None
                    else None
                ),
                "frequency_min_hz": (
                    float(np.min(data.frequency_hz))
                    if data.frequency_hz is not None
                    else None
                ),
                "frequency_max_hz": (
                    float(np.max(data.frequency_hz))
                    if data.frequency_hz is not None
                    else None
                ),
                "s_shape": _shape_or_none(data.s_parameters),
                "gain_db_shape": _shape_or_none(data.gain_db),
                "gain_db_min": (
                    float(np.min(data.gain_db))
                    if data.gain_db is not None
                    else None
                ),
                "gain_db_max": (
                    float(np.max(data.gain_db))
                    if data.gain_db is not None
                    else None
                ),
                "h5_backend": data.h5_attrs.get("backend"),
                "h5_n_ports": data.h5_attrs.get("n_ports"),
                "h5_topology_only": data.h5_attrs.get("topology_only"),
                "matches_expected_simulation_type": (
                    data.status.simulation_type == case.expected_simulation_type
                ),
                "matches_expected_circuit_template": (
                    data.status.circuit_template == case.expected_circuit_template
                ),
            }
        )
            
        timings = read_timings_json(run_dir)

        for timing_key in [
            "topology_build_s",
            "jc_export_s",
            "hbsolve_s",
            "sparameter_extract_s",
            "h5_write_s",
            "total_internal_s",
        ]:
            value = timings.get(timing_key)
            row[timing_key] = None if value is None else float(value)

        if data.s_parameters is not None:
            s = np.asarray(data.s_parameters, dtype=np.complex128)
            row["s_max_abs"] = float(np.max(np.abs(s)))
            row["s_all_finite"] = bool(np.all(np.isfinite(s.real)) and np.all(np.isfinite(s.imag)))
        else:
            row["s_max_abs"] = None
            row["s_all_finite"] = None

        if data.frequency_hz is not None:
            row["frequency_all_finite"] = bool(np.all(np.isfinite(data.frequency_hz)))
        else:
            row["frequency_all_finite"] = None

        if data.gain_db is not None:
            row["gain_db_all_finite"] = bool(np.all(np.isfinite(data.gain_db)))
        else:
            row["gain_db_all_finite"] = None

    except Exception as exc:
        row.update(
            {
                "status": "READ_FAILED",
                "run_id": None,
                "simulation_type": None,
                "circuit_template": None,
                "solver_success": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "julia_status_runtime_s": None,
                "residual_norm": None,
                "relative_residual_norm": None,
                "frequency_shape": None,
                "frequency_points": None,
                "frequency_min_hz": None,
                "frequency_max_hz": None,
                "s_shape": None,
                "gain_db_shape": None,
                "gain_db_min": None,
                "gain_db_max": None,
                "h5_backend": None,
                "h5_n_ports": None,
                "h5_topology_only": None,
                "matches_expected_simulation_type": False,
                "matches_expected_circuit_template": False,
                "s_max_abs": None,
                "s_all_finite": None,
                "frequency_all_finite": None,
                "gain_db_all_finite": None,
            }
        )

    try:
        artifact = load_topology_artifact(run_dir)
        row.update(
            {
                "topology_n_elements": artifact.n_elements,
                "topology_n_nodes": artifact.n_nodes,
                "topology_element_kind_counts_json": json.dumps(
                    artifact.element_kind_counts,
                    sort_keys=True,
                ),
                "topology_jc_tuple_count": artifact.topology.get(
                    "josephsoncircuits_tuple_count"
                ),
                "topology_expected_ir_elements": artifact.topology.get(
                    "expected_ir_elements"
                ),
                "topology_expected_jc_tuples": artifact.topology.get(
                    "expected_jc_tuples"
                ),
                "topology_ir_element_count_match": artifact.topology.get(
                    "ir_element_count_match"
                ),
                "topology_jc_tuple_count_match": artifact.topology.get(
                    "jc_tuple_count_match"
                ),
            }
        )
    except Exception:
        row.update(
            {
                "topology_n_elements": None,
                "topology_n_nodes": None,
                "topology_element_kind_counts_json": None,
                "topology_jc_tuple_count": None,
                "topology_expected_ir_elements": None,
                "topology_expected_jc_tuples": None,
                "topology_ir_element_count_match": None,
                "topology_jc_tuple_count_match": None,
            }
        )

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_timings_json(run_dir: Path) -> dict[str, float | None]:
    h5_path = run_dir / "simulation.h5"

    if not h5_path.exists():
        return {}

    try:
        import h5py

        with h5py.File(h5_path, "r") as h5:
            if "metadata/timings_json" not in h5:
                return {}

            raw = h5["metadata/timings_json"][()]

            if isinstance(raw, bytes):
                text = raw.decode("utf-8")
            elif hasattr(raw, "decode"):
                text = raw.decode("utf-8")
            elif hasattr(raw, "item"):
                item = raw.item()
                text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            else:
                text = str(raw)

            parsed = json.loads(text)

            if not isinstance(parsed, dict):
                return {}

            return parsed
    except Exception:
        return {}

def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_case: dict[str, dict[str, Any]] = {}

    for row in rows:
        status = str(row.get("status"))
        by_status[status] = by_status.get(status, 0) + 1

        case = str(row.get("case_name"))
        case_summary = by_case.setdefault(
            case,
            {
                "n_runs": 0,
                "n_pass": 0,
                "python_wall_time_s": [],
                "julia_status_runtime_s": [],
                "h5_size_bytes": [],
            },
        )

        case_summary["n_runs"] += 1

        if row.get("status") == "PASS":
            case_summary["n_pass"] += 1

        for key in ["python_wall_time_s", "julia_status_runtime_s", "h5_size_bytes"]:
            value = row.get(key)
            if value is not None:
                case_summary[key].append(float(value))

    for case_summary in by_case.values():
        for key in ["python_wall_time_s", "julia_status_runtime_s", "h5_size_bytes"]:
            values = case_summary[key]
            if values:
                case_summary[f"{key}_mean"] = float(np.mean(values))
                case_summary[f"{key}_min"] = float(np.min(values))
                case_summary[f"{key}_max"] = float(np.max(values))
            else:
                case_summary[f"{key}_mean"] = None
                case_summary[f"{key}_min"] = None
                case_summary[f"{key}_max"] = None

            del case_summary[key]

    return {
        "n_rows": len(rows),
        "by_status": by_status,
        "by_case": by_case,
    }



def prepare_benchmark_config(
    *,
    case: BenchmarkCase,
    rep: int,
    source_config_path: Path,
    generated_config_root: Path,
    jtl_linear_backend: str,
    rf_jtl_linear_backend: str,
    ethz_jtl_linear_backend: str,
    enable_jc_setup_cache: bool,
) -> Path:
    solver_overrides: dict[str, Any] = {}

    if case.name == "jtl_linear":
        if jtl_linear_backend != "hbsolve" or enable_jc_setup_cache:
            solver_overrides["jtl_linear_backend"] = jtl_linear_backend
            solver_overrides["enable_jc_setup_cache"] = bool(enable_jc_setup_cache)

    elif case.name == "rf_jtl_linear":
        if rf_jtl_linear_backend != "hbsolve" or enable_jc_setup_cache:
            solver_overrides["rf_jtl_linear_backend"] = rf_jtl_linear_backend
            solver_overrides["enable_jc_setup_cache"] = bool(enable_jc_setup_cache)

    elif case.name == "ethz_jtl_linear":
        if ethz_jtl_linear_backend != "hbsolve" or enable_jc_setup_cache:
            solver_overrides["ethz_jtl_linear_backend"] = ethz_jtl_linear_backend
            solver_overrides["enable_jc_setup_cache"] = bool(enable_jc_setup_cache)

    if not solver_overrides:
        return source_config_path

    config = json.loads(source_config_path.read_text(encoding="utf-8"))
    config["solver"] = solver_overrides

    generated_config_root.mkdir(parents=True, exist_ok=True)
    out = generated_config_root / f"{case.name}__rep{rep:02d}__solver.json"
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out


def run_benchmark_suite(
    *,
    harmonia_root: Path,
    benchmark_dir: Path,
    cases: list[BenchmarkCase],
    repetitions: int,
    julia_executable: str = "julia",
    force: bool = False,
    use_batch_runner: bool = False,
    jtl_linear_backend: str = "hbsolve",
    rf_jtl_linear_backend: str = "hbsolve",
    ethz_jtl_linear_backend: str = "hbsolve",
    enable_jc_setup_cache: bool = False,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    if jtl_linear_backend not in {"hbsolve", "hblinsolve_direct"}:
        raise ValueError("jtl_linear_backend must be 'hbsolve' or 'hblinsolve_direct'")

    if rf_jtl_linear_backend not in {"hbsolve", "hblinsolve_direct"}:
        raise ValueError("rf_jtl_linear_backend must be 'hbsolve' or 'hblinsolve_direct'")

    if ethz_jtl_linear_backend not in {"hbsolve", "hblinsolve_direct"}:
        raise ValueError("ethz_jtl_linear_backend must be 'hbsolve' or 'hblinsolve_direct'")

    if force and benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)

    benchmark_dir.mkdir(parents=True, exist_ok=True)

    config_root = harmonia_root / "examples" / "configs"
    generated_config_root = benchmark_dir / "configs"
    generated_config_root.mkdir(parents=True, exist_ok=True)
    runs_root = benchmark_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    cases_json = [asdict(case) for case in cases]
    write_json(benchmark_dir / "benchmark_cases.json", cases_json)

    environment = collect_environment(
        harmonia_root=harmonia_root,
        twpa_root=_REPO_ROOT,
        josephsoncircuits_root=_WORKSPACE_ROOT / "JosephsonCircuits.jl",
        julia_executable=julia_executable,
    )
    write_json(benchmark_dir / "environment.json", environment)

    prepared: list[tuple[BenchmarkCase, int, Path, Path]] = []

    for case in cases:
        config_path = config_root / case.config_filename

        if not config_path.exists():
            raise FileNotFoundError(f"Missing benchmark config: {config_path}")

        for rep in range(repetitions):
            run_dir = runs_root / f"{case.name}__rep{rep:02d}"
            effective_config_path = prepare_benchmark_config(
                case=case,
                rep=rep,
                source_config_path=config_path,
                generated_config_root=generated_config_root,
                jtl_linear_backend=jtl_linear_backend,
                rf_jtl_linear_backend=rf_jtl_linear_backend,
                ethz_jtl_linear_backend=ethz_jtl_linear_backend,
                enable_jc_setup_cache=enable_jc_setup_cache,
            )
            prepared.append((case, rep, effective_config_path, run_dir))

    if use_batch_runner:
        batch_timeout_s = max(600.0, sum(case.timeout_s for case, _, _, _ in prepared) + 60.0)

        t_batch0 = time.perf_counter()
        batch_result = run_harmonia_simulation_batch(
            items=[(config_path, run_dir) for _, _, config_path, run_dir in prepared],
            harmonia_jl_root=harmonia_root,
            julia_executable=julia_executable,
            timeout_s=batch_timeout_s,
            force=force,
            use_cache=not force,
            batch_work_dir=runs_root / "_julia_batch_runner",
        )
        batch_wall_time_s = time.perf_counter() - t_batch0

        record_by_output = {
            Path(record.output_dir).resolve(): record
            for record in batch_result.records
        }

        fallback_per_run_wall_s = batch_wall_time_s / max(1, len(prepared))

        for case, rep, config_path, run_dir in prepared:
            batch_record = record_by_output.get(run_dir.resolve())
            python_wall_time_s = (
                float(batch_record.runtime_s)
                if batch_record is not None and batch_record.runtime_s is not None
                else fallback_per_run_wall_s
            )

            row = collect_run_metrics(
                case=case,
                run_dir=run_dir,
                python_wall_time_s=python_wall_time_s,
                repetition=rep,
                config_path=config_path,
            )

            row["returncode"] = int(batch_record.returncode) if batch_record is not None else int(batch_result.returncode)
            row["runner_ok"] = bool(batch_record.ok) if batch_record is not None else False
            row["batch_runner"] = True
            row["batch_process_wall_time_s"] = float(batch_wall_time_s)

            rows.append(row)

            write_csv(benchmark_dir / "benchmark_runs.csv", rows)
            write_csv(benchmark_dir / "benchmark_results.csv", rows)
            write_json(
                benchmark_dir / "benchmark_summary.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark_type": "harmonia_josephsoncircuits_baseline",
                    "benchmark_dir": str(benchmark_dir),
                    "created_utc": utc_timestamp(),
                    "repetitions": int(repetitions),
                    "use_batch_runner": bool(use_batch_runner),
                    "jtl_linear_backend": jtl_linear_backend,
                    "rf_jtl_linear_backend": rf_jtl_linear_backend,
                    "ethz_jtl_linear_backend": ethz_jtl_linear_backend,
                    "enable_jc_setup_cache": bool(enable_jc_setup_cache),
                    "cases": cases_json,
                    "environment": environment,
                    "summary": summarize_rows(rows),
                    "rows": rows,
                },
            )

    else:
        for case, rep, config_path, run_dir in prepared:

            t0 = time.perf_counter()
            result = run_harmonia_simulation(
                config_path=config_path,
                output_dir=run_dir,
                harmonia_jl_root=harmonia_root,
                julia_executable=julia_executable,
                timeout_s=case.timeout_s,
                force=force,
                use_cache=not force,
            )
            python_wall_time_s = time.perf_counter() - t0

            row = collect_run_metrics(
                case=case,
                run_dir=run_dir,
                python_wall_time_s=python_wall_time_s,
                repetition=rep,
                config_path=config_path,
            )

            row["returncode"] = int(result.returncode)
            row["runner_ok"] = bool(result.ok)
            row["batch_runner"] = False
            row["batch_process_wall_time_s"] = None

            rows.append(row)

            write_csv(benchmark_dir / "benchmark_runs.csv", rows)
            write_csv(benchmark_dir / "benchmark_results.csv", rows)
            write_json(
                benchmark_dir / "benchmark_summary.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark_type": "harmonia_josephsoncircuits_baseline",
                    "benchmark_dir": str(benchmark_dir),
                    "created_utc": utc_timestamp(),
                    "repetitions": int(repetitions),
                    "use_batch_runner": bool(use_batch_runner),
                    "jtl_linear_backend": jtl_linear_backend,
                    "rf_jtl_linear_backend": rf_jtl_linear_backend,
                    "ethz_jtl_linear_backend": ethz_jtl_linear_backend,
                    "enable_jc_setup_cache": bool(enable_jc_setup_cache),
                    "cases": cases_json,
                    "environment": environment,
                    "summary": summarize_rows(rows),
                    "rows": rows,
                },
            )

    final_summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": "harmonia_josephsoncircuits_baseline",
        "benchmark_dir": str(benchmark_dir),
        "created_utc": utc_timestamp(),
        "repetitions": int(repetitions),
        "use_batch_runner": bool(use_batch_runner),
        "jtl_linear_backend": jtl_linear_backend,
        "rf_jtl_linear_backend": rf_jtl_linear_backend,
        "ethz_jtl_linear_backend": ethz_jtl_linear_backend,
        "enable_jc_setup_cache": bool(enable_jc_setup_cache),
        "cases": cases_json,
        "environment": environment,
        "summary": summarize_rows(rows),
        "rows": rows,
    }

    write_csv(benchmark_dir / "benchmark_runs.csv", rows)
    write_csv(benchmark_dir / "benchmark_results.csv", rows)
    write_json(benchmark_dir / "benchmark_summary.json", final_summary)

    return final_summary


def print_summary(summary: dict[str, Any]) -> None:
    print("Harmonia / JosephsonCircuits benchmark suite")
    print("============================================")
    print(f"benchmark_dir: {summary['benchmark_dir']}")
    print(f"repetitions:   {summary['repetitions']}")
    print(f"rows:          {summary['summary']['n_rows']}")
    print(f"by_status:     {summary['summary']['by_status']}")
    print()

    for case_name, case_summary in summary["summary"]["by_case"].items():
        print(
            f"{case_name}: "
            f"runs={case_summary['n_runs']} "
            f"pass={case_summary['n_pass']} "
            f"python_mean={case_summary['python_wall_time_s_mean']} "
            f"julia_mean={case_summary['julia_status_runtime_s_mean']} "
            f"h5_bytes_mean={case_summary['h5_size_bytes_mean']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harmonia-root",
        type=Path,
        default=_WORKSPACE_ROOT / "Harmonia.jl",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=_WORKSPACE_ROOT / "outputs" / "benchmarks" / f"harmonia_baseline_{utc_timestamp()}",
    )
    parser.add_argument(
        "--suite",
        choices=["minimal", "modern", "all"],
        default="modern",
        help="minimal: two modern cases; modern: CircuitIR cases; all: modern + reference/legacy.",
    )
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-batch-runner", action="store_true", help="Run benchmark cases through one Julia batch process.")
    parser.add_argument("--jtl-linear-backend", choices=["hbsolve", "hblinsolve_direct"], default="hbsolve", help="Backend override for the jtl_linear benchmark case.")
    parser.add_argument("--rf-jtl-linear-backend", choices=["hbsolve", "hblinsolve_direct"], default="hbsolve", help="Backend override for the rf_jtl_linear benchmark case.")
    parser.add_argument("--ethz-jtl-linear-backend", choices=["hbsolve", "hblinsolve_direct"], default="hbsolve", help="Backend override for the ethz_jtl_linear benchmark case.")
    parser.add_argument("--enable-jc-setup-cache", action="store_true", help="Request JC setup-cache telemetry for supported benchmark cases.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = select_cases(suite=args.suite, include_legacy=args.include_legacy)

    summary = run_benchmark_suite(
        harmonia_root=args.harmonia_root,
        benchmark_dir=args.benchmark_dir,
        cases=cases,
        repetitions=args.repetitions,
        julia_executable=args.julia,
        force=args.force,
        use_batch_runner=args.use_batch_runner,
        jtl_linear_backend=args.jtl_linear_backend,
        rf_jtl_linear_backend=args.rf_jtl_linear_backend,
        ethz_jtl_linear_backend=args.ethz_jtl_linear_backend,
        enable_jc_setup_cache=args.enable_jc_setup_cache,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)

    n_rows = int(summary["summary"]["n_rows"])
    n_pass = int(summary["summary"]["by_status"].get("PASS", 0))

    return 0 if n_rows == n_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
