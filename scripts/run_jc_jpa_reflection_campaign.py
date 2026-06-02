"""
Run a tiny JosephsonCircuits-backed JPA reflection campaign.

This is the first campaign with the real Julia/Harmonia/JosephsonCircuits
solver chain in the loop.

It sweeps pump current for the one-port JPA reflection smoke:

    pump_current_a = [0.0, 2.0e-9, 5.65e-9]

The output is a normal campaign folder:

    outputs/campaigns/jc_jpa_reflection_smoke/
      configs/
      runs/
      runs.csv
      campaign_summary.json

This is still a smoke campaign, not a production TWPA workflow.
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


def make_jc_jpa_reflection_config(
    *,
    index: int,
    pump_current_a: float,
    n_frequency: int = 11,
    f_start_hz: float = 4.5e9,
    f_stop_hz: float = 5.0e9,
    pump_frequency_hz: float = 4.75001e9,
) -> dict[str, Any]:
    if index < 0:
        raise ValueError("index must be non-negative")
    if n_frequency < 1:
        raise ValueError("n_frequency must be >= 1")
    if pump_current_a < 0.0:
        raise ValueError("pump_current_a must be non-negative")

    return {
        "schema_version": SCHEMA_VERSION,
        "simulation_type": "jc_jpa_reflection_smoke",
        "circuit_template": "one_port_jpa_reflection",
        "seed": 3001 + index,
        "parameters": {
            "R_ohm": 50.0,
            "Cc_F": 100.0e-15,
            "Lj_H": 1000.0e-12,
            "Cj_F": 1000.0e-15,
            "pump_frequency_hz": float(pump_frequency_hz),
            "pump_current_a": float(pump_current_a),
            "n_pump_harmonics": 4,
            "n_modulation_harmonics": 4,
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
            "backend": "JosephsonCircuits.hbsolve",
            "notes": (
                "One-port JPA reflection campaign using JosephsonCircuits "
                "hbsolve and linearized.S. This is a smoke campaign."
            ),
        },
    }


def campaign_paths(campaign_dir: Path) -> dict[str, Path]:
    return {
        "configs": campaign_dir / "configs",
        "runs": campaign_dir / "runs",
        "registry": campaign_dir / "runs.csv",
        "summary": campaign_dir / "campaign_summary.json",
    }


def compute_one_port_reflection_metrics(run_dir: Path) -> dict[str, Any]:
    data = load_julia_simulation(run_dir)

    if data.frequency_hz is None:
        raise ValueError(f"Run has no frequency axis: {run_dir}")
    if data.s_parameters is None:
        raise ValueError(f"Run has no S-parameters: {run_dir}")
    if data.gain_db is None:
        raise ValueError(f"Run has no gain/reflection curve: {run_dir}")

    s = np.asarray(data.s_parameters, dtype=np.complex128)

    if s.ndim != 3 or s.shape[1:] != (1, 1):
        raise ValueError(f"Expected one-port S shape (frequency, 1, 1), got {s.shape}")

    s11 = s[:, 0, 0]
    reflection_db = np.asarray(data.gain_db, dtype=float)

    peak_idx = int(np.argmax(reflection_db))
    dip_idx = int(np.argmin(reflection_db))

    return {
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
            and np.all(np.isfinite(s.real))
            and np.all(np.isfinite(s.imag))
            and np.all(np.isfinite(reflection_db))
        ),
    }


def run_campaign(
    *,
    pump_currents_a: list[float],
    harmonia_root: Path,
    campaign_dir: Path,
    julia_executable: str = "julia",
    timeout_s: float = 300.0,
    force: bool = False,
    n_frequency: int = 11,
) -> dict[str, Any]:
    if not pump_currents_a:
        raise ValueError("pump_currents_a must not be empty")

    if force and campaign_dir.exists():
        shutil.rmtree(campaign_dir)

    paths = campaign_paths(campaign_dir)
    paths["configs"].mkdir(parents=True, exist_ok=True)
    paths["runs"].mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []

    for idx, pump_current in enumerate(pump_currents_a):
        run_name = f"pump_{pump_current:.3e}_A".replace("+", "").replace("-", "m").replace(".", "p")
        config_path = paths["configs"] / f"{run_name}.json"
        output_dir = paths["runs"] / run_name

        config = make_jc_jpa_reflection_config(
            index=idx,
            pump_current_a=float(pump_current),
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
            "pump_current_a": float(pump_current),
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
            run_record["metrics"] = compute_one_port_reflection_metrics(output_dir)
        else:
            run_record["metrics"] = None
            run_record["failure_reason"] = None if result.status is None else result.status.failure_reason

        runs.append(run_record)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_type": "jc_jpa_reflection_pump_current_sweep",
        "campaign_dir": str(campaign_dir),
        "harmonia_root": str(harmonia_root),
        "pump_currents_a": [float(x) for x in pump_currents_a],
        "n_requested": len(pump_currents_a),
        "n_launched": len(runs),
        "registry": registry_summary(paths["registry"]),
        "runs": runs,
    }

    write_json(paths["summary"], summary)
    return summary


def print_human_summary(summary: dict[str, Any]) -> None:
    registry = summary["registry"]

    print("JosephsonCircuits JPA reflection campaign")
    print("=========================================")
    print(f"campaign_dir:    {summary['campaign_dir']}")
    print(f"pump_currents_a: {summary['pump_currents_a']}")
    print(f"n_requested:     {summary['n_requested']}")
    print(f"n_launched:      {summary['n_launched']}")
    print(f"by_status:       {registry['by_status']}")
    print(f"by_type:         {registry['by_simulation_type']}")
    print()

    for run in summary["runs"]:
        metrics = run.get("metrics") or {}
        print(
            f"{run['run_name']}: "
            f"status={run['status']} "
            f"ok={run['ok']} "
            f"reflection_db_min={metrics.get('reflection_db_min')} "
            f"reflection_db_max={metrics.get('reflection_db_max')} "
            f"peak_freq={metrics.get('reflection_peak_frequency_hz')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pump-currents-a",
        type=float,
        nargs="+",
        default=[0.0, 2.0e-9, 5.65e-9],
    )
    parser.add_argument(
        "--harmonia-root",
        type=Path,
        default=_WORKSPACE_ROOT / "Harmonia.jl",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=_WORKSPACE_ROOT / "outputs" / "campaigns" / "jc_jpa_reflection_smoke",
    )
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-frequency", type=int, default=11)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_campaign(
        pump_currents_a=args.pump_currents_a,
        harmonia_root=args.harmonia_root,
        campaign_dir=args.campaign_dir,
        julia_executable=args.julia,
        timeout_s=args.timeout_s,
        force=args.force,
        n_frequency=args.n_frequency,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)

    n_pass = summary["registry"]["by_status"].get("PASS", 0)
    return 0 if n_pass >= len(args.pump_currents_a) else 1


if __name__ == "__main__":
    raise SystemExit(main())