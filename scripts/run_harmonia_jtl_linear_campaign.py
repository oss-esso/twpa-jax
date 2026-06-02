"""
Run a tiny CircuitIR/Harmonia/JosephsonCircuits JTL linear campaign.

This campaign sweeps the Josephson inductance Lj_H for the linearized
two-port JTL chain smoke.

Pipeline:
    Python config generation
      -> Harmonia.jl/scripts/run_simulation.jl
      -> CircuitIR + add_jtl_chain!
      -> JosephsonCircuits.hbsolve / linearized.S
      -> HDF5/status
      -> runs.csv + campaign_summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpa.io.julia_bridge import load_julia_simulation
from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.run_registry import register_run_dir, registry_summary
from twpa.io.simulation_schema import SCHEMA_VERSION, write_json


def make_harmonia_jtl_linear_config(
    *,
    index: int,
    Lj_H: float,
    n_cell: int = 4,
    n_frequency: int = 11,
    f_start_hz: float = 4.0e9,
    f_stop_hz: float = 8.0e9,
) -> dict[str, Any]:
    if index < 0:
        raise ValueError("index must be non-negative")
    if Lj_H <= 0.0:
        raise ValueError("Lj_H must be positive")
    if n_cell <= 0:
        raise ValueError("n_cell must be positive")
    if n_frequency <= 0:
        raise ValueError("n_frequency must be positive")

    return {
        "schema_version": SCHEMA_VERSION,
        "simulation_type": "harmonia_jtl_linear_jc_smoke",
        "circuit_template": "circuit_ir_jtl_chain_linear_jc",
        "seed": 4301 + index,
        "parameters": {
            "N_cell": int(n_cell),
            "prefix": "jtl",
            "start_node": "n1",
            "ground": "0",
            "Cg_F": 50.0e-15,
            "Lj_H": float(Lj_H),
            "Cj_F": 1000.0e-15,
            "port_impedance_ohm": 50.0,
            "pump_frequency_hz": 6.0e9,
            "pump_current_a": 0.0,
            "n_pump_harmonics": 1,
            "n_modulation_harmonics": 1,
            "campaign_index": int(index),
        },
        "axes": {
            "frequency_hz": {
                "start": float(f_start_hz),
                "stop": float(f_stop_hz),
                "points": int(n_frequency),
            }
        },
        "solver": {
            "backend": "Harmonia.CircuitIR + JosephsonCircuits.hbsolve",
            "notes": "Tiny JTL linear campaign over Lj_H.",
        },
    }


def campaign_paths(campaign_dir: Path) -> dict[str, Path]:
    return {
        "configs": campaign_dir / "configs",
        "runs": campaign_dir / "runs",
        "registry": campaign_dir / "runs.csv",
        "summary": campaign_dir / "campaign_summary.json",
    }


def compute_jtl_linear_metrics(run_dir: Path) -> dict[str, Any]:
    data = load_julia_simulation(run_dir)

    if data.frequency_hz is None:
        raise ValueError(f"Missing frequency axis: {run_dir}")
    if data.s_parameters is None:
        raise ValueError(f"Missing S-parameters: {run_dir}")
    if data.gain_db is None:
        raise ValueError(f"Missing gain_db: {run_dir}")

    s = np.asarray(data.s_parameters, dtype=np.complex128)

    if s.ndim != 3 or s.shape[1:] != (2, 2):
        raise ValueError(f"Expected 2-port S shape (frequency, 2, 2), got {s.shape}")

    s11 = s[:, 0, 0]
    s21 = s[:, 1, 0]
    s12 = s[:, 0, 1]
    s22 = s[:, 1, 1]

    gain_db = np.asarray(data.gain_db, dtype=float)

    return {
        "frequency_points": int(data.frequency_hz.shape[0]),
        "frequency_min_hz": float(np.min(data.frequency_hz)),
        "frequency_max_hz": float(np.max(data.frequency_hz)),
        "s_shape": list(s.shape),
        "gain_db_min": float(np.min(gain_db)),
        "gain_db_max": float(np.max(gain_db)),
        "max_abs_s21": float(np.max(np.abs(s21))),
        "min_abs_s21": float(np.min(np.abs(s21))),
        "max_abs_s11": float(np.max(np.abs(s11))),
        "max_abs_s22": float(np.max(np.abs(s22))),
        "reciprocal_error_max_abs": float(np.max(np.abs(s21 - s12))),
        "all_arrays_finite": bool(
            np.all(np.isfinite(data.frequency_hz))
            and np.all(np.isfinite(s.real))
            and np.all(np.isfinite(s.imag))
            and np.all(np.isfinite(gain_db))
        ),
    }


def run_campaign(
    *,
    lj_values_h: list[float],
    harmonia_root: Path,
    campaign_dir: Path,
    julia_executable: str = "julia",
    timeout_s: float = 300.0,
    force: bool = False,
    n_cell: int = 4,
    n_frequency: int = 11,
) -> dict[str, Any]:
    if not lj_values_h:
        raise ValueError("lj_values_h must not be empty")

    if force and campaign_dir.exists():
        shutil.rmtree(campaign_dir)

    paths = campaign_paths(campaign_dir)
    paths["configs"].mkdir(parents=True, exist_ok=True)
    paths["runs"].mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []

    for idx, lj_h in enumerate(lj_values_h):
        run_name = f"Lj_{lj_h:.3e}_H".replace("+", "").replace("-", "m").replace(".", "p")
        config_path = paths["configs"] / f"{run_name}.json"
        output_dir = paths["runs"] / run_name

        config = make_harmonia_jtl_linear_config(
            index=idx,
            Lj_H=float(lj_h),
            n_cell=n_cell,
            n_frequency=n_frequency,
        )
        write_json(config_path, config)

        result = run_harmonia_simulation(
            config_path=config_path,
            output_dir=output_dir,
            harmonia_jl_root=harmonia_root,
            julia_executable=julia_executable,
            timeout_s=timeout_s,
            force=force,
            use_cache=not force,
        )

        run_record: dict[str, Any] = {
            "run_name": run_name,
            "Lj_H": float(lj_h),
            "returncode": result.returncode,
            "ok": result.ok,
            "output_dir": str(output_dir),
            "status": None if result.status is None else result.status.status,
            "run_id": None if result.status is None else result.status.run_id,
        }

        if result.status is not None:
            registered = register_run_dir(paths["registry"], output_dir)
            run_record["registered_status"] = registered.status

        if result.ok:
            run_record["metrics"] = compute_jtl_linear_metrics(output_dir)
        else:
            run_record["metrics"] = None
            run_record["failure_reason"] = None if result.status is None else result.status.failure_reason

        runs.append(run_record)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_type": "harmonia_jtl_linear_lj_sweep",
        "campaign_dir": str(campaign_dir),
        "harmonia_root": str(harmonia_root),
        "lj_values_h": [float(x) for x in lj_values_h],
        "n_cell": int(n_cell),
        "n_frequency": int(n_frequency),
        "n_requested": len(lj_values_h),
        "n_launched": len(runs),
        "registry": registry_summary(paths["registry"]),
        "runs": runs,
    }

    write_json(paths["summary"], summary)
    return summary


def print_human_summary(summary: dict[str, Any]) -> None:
    registry = summary["registry"]

    print("Harmonia JTL linear JosephsonCircuits campaign")
    print("==============================================")
    print(f"campaign_dir: {summary['campaign_dir']}")
    print(f"lj_values_h:  {summary['lj_values_h']}")
    print(f"n_cell:       {summary['n_cell']}")
    print(f"n_frequency:  {summary['n_frequency']}")
    print(f"by_status:    {registry['by_status']}")
    print(f"by_type:      {registry['by_simulation_type']}")
    print()

    for run in summary["runs"]:
        metrics = run.get("metrics") or {}
        print(
            f"{run['run_name']}: "
            f"status={run['status']} "
            f"ok={run['ok']} "
            f"gain_db_min={metrics.get('gain_db_min')} "
            f"gain_db_max={metrics.get('gain_db_max')} "
            f"max_abs_s11={metrics.get('max_abs_s11')} "
            f"max_abs_s21={metrics.get('max_abs_s21')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lj-values-h",
        type=float,
        nargs="+",
        default=[0.8e-9, 1.0e-9, 1.2e-9],
    )
    parser.add_argument(
        "--harmonia-root",
        type=Path,
        default=_WORKSPACE_ROOT / "Harmonia.jl",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=_WORKSPACE_ROOT / "outputs" / "campaigns" / "harmonia_jtl_linear_jc",
    )
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-cell", type=int, default=4)
    parser.add_argument("--n-frequency", type=int, default=11)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_campaign(
        lj_values_h=args.lj_values_h,
        harmonia_root=args.harmonia_root,
        campaign_dir=args.campaign_dir,
        julia_executable=args.julia,
        timeout_s=args.timeout_s,
        force=args.force,
        n_cell=args.n_cell,
        n_frequency=args.n_frequency,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)

    n_pass = summary["registry"]["by_status"].get("PASS", 0)
    return 0 if n_pass >= len(args.lj_values_h) else 1


if __name__ == "__main__":
    raise SystemExit(main())