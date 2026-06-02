"""
Run a tiny linear S-parameter campaign through the Julia/Harmonia engine.

This is the first physically meaningful campaign after schema-smoke.

It sweeps transmission-line impedance:

    z_line_ohm = 45, 50, 55

Expected behavior:
    - z_line = z_ref gives near-zero reflection.
    - off-match impedances give nonzero reflection.
    - all runs remain reciprocal and passive.
    - gain is ~0 dB for attenuation_np_per_m = 0.

Example
-------
python scripts/run_linear_sparams_campaign.py --force
python scripts/run_linear_sparams_campaign.py --z-lines 40 45 50 55 60 --force
"""

from __future__ import annotations

import argparse
import json
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


SCHEMA_VERSION = "0.1.0"


def assert_json_serializable(obj: Any, *, context: str = "object") -> None:
    try:
        json.dumps(obj)
    except TypeError as exc:
        raise TypeError(f"{context} is not JSON serializable: {exc}") from exc


def write_json(path: Path, obj: dict[str, Any]) -> None:
    assert_json_serializable(obj, context=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_linear_sparams_config(
    *,
    index: int,
    z_line_ohm: float,
    z_ref_ohm: float = 50.0,
    length_m: float = 0.1,
    phase_velocity_m_per_s: float = 1.2e8,
    attenuation_np_per_m: float = 0.0,
    f_start_hz: float = 4.0e9,
    f_stop_hz: float = 8.0e9,
    n_frequency: int = 101,
) -> dict[str, Any]:
    if index < 0:
        raise ValueError("index must be non-negative")
    if z_line_ohm <= 0:
        raise ValueError("z_line_ohm must be positive")
    if z_ref_ohm <= 0:
        raise ValueError("z_ref_ohm must be positive")
    if length_m < 0:
        raise ValueError("length_m must be non-negative")
    if phase_velocity_m_per_s <= 0:
        raise ValueError("phase_velocity_m_per_s must be positive")
    if attenuation_np_per_m < 0:
        raise ValueError("attenuation_np_per_m must be non-negative")
    if n_frequency < 1:
        raise ValueError("n_frequency must be >= 1")

    return {
        "schema_version": SCHEMA_VERSION,
        "simulation_type": "linear_sparams",
        "circuit_template": "matched_transmission_line",
        "seed": 2000 + index,
        "parameters": {
            "z_ref_ohm": float(z_ref_ohm),
            "z_line_ohm": float(z_line_ohm),
            "length_m": float(length_m),
            "phase_velocity_m_per_s": float(phase_velocity_m_per_s),
            "attenuation_np_per_m": float(attenuation_np_per_m),
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
            "backend": "analytic_abcd_transmission_line",
            "notes": (
                "Linear analytic transmission-line campaign. "
                "This is not a JosephsonCircuits HB solve."
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


def compute_2port_metrics(run_dir: Path) -> dict[str, Any]:
    data = load_julia_simulation(run_dir)

    if data.frequency_hz is None:
        raise ValueError(f"Run has no frequency axis: {run_dir}")
    if data.s_parameters is None:
        raise ValueError(f"Run has no S-parameters: {run_dir}")

    s = data.s_parameters

    if s.ndim != 3 or s.shape[1:] != (2, 2):
        raise ValueError(f"Expected S shape (frequency, 2, 2), got {s.shape}")

    s11 = s[:, 0, 0]
    s12 = s[:, 0, 1]
    s21 = s[:, 1, 0]
    s22 = s[:, 1, 1]

    singular_values = np.linalg.svd(s, compute_uv=False)
    max_singular_value = float(np.max(singular_values))

    return {
        "frequency_points": int(data.frequency_hz.shape[0]),
        "frequency_min_hz": float(np.min(data.frequency_hz)),
        "frequency_max_hz": float(np.max(data.frequency_hz)),
        "s_shape": list(s.shape),
        "max_abs_s11": float(np.max(np.abs(s11))),
        "max_abs_s22": float(np.max(np.abs(s22))),
        "max_abs_s21": float(np.max(np.abs(s21))),
        "min_abs_s21": float(np.min(np.abs(s21))),
        "max_abs_s12": float(np.max(np.abs(s12))),
        "min_abs_s12": float(np.min(np.abs(s12))),
        "reciprocal_error_max_abs": float(np.max(np.abs(s21 - s12))),
        "passivity_max_singular_value": max_singular_value,
        "gain_db_min": float(np.min(data.gain_db)) if data.gain_db is not None else None,
        "gain_db_max": float(np.max(data.gain_db)) if data.gain_db is not None else None,
        "all_arrays_finite": bool(
            np.all(np.isfinite(data.frequency_hz))
            and np.all(np.isfinite(s.real))
            and np.all(np.isfinite(s.imag))
            and (data.gain_db is None or np.all(np.isfinite(data.gain_db)))
        ),
    }


def run_campaign(
    *,
    z_lines_ohm: list[float],
    harmonia_root: Path,
    campaign_dir: Path,
    julia_executable: str = "julia",
    timeout_s: float = 300.0,
    force: bool = False,
    n_frequency: int = 101,
) -> dict[str, Any]:
    if not z_lines_ohm:
        raise ValueError("z_lines_ohm must not be empty")

    paths = campaign_paths(campaign_dir)
    paths["configs"].mkdir(parents=True, exist_ok=True)
    paths["runs"].mkdir(parents=True, exist_ok=True)

    runs = []

    for idx, z_line in enumerate(z_lines_ohm):
        run_name = f"zline_{z_line:g}_ohm".replace(".", "p")
        config_path = paths["configs"] / f"{run_name}.json"
        output_dir = paths["runs"] / run_name

        config = make_linear_sparams_config(
            index=idx,
            z_line_ohm=float(z_line),
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
            "z_line_ohm": float(z_line),
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
            run_record["metrics"] = compute_2port_metrics(output_dir)
        else:
            run_record["metrics"] = None
            run_record["failure_reason"] = None if result.status is None else result.status.failure_reason

        runs.append(run_record)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_type": "linear_sparams_zline_sweep",
        "campaign_dir": str(campaign_dir),
        "harmonia_root": str(harmonia_root),
        "z_lines_ohm": [float(z) for z in z_lines_ohm],
        "n_requested": len(z_lines_ohm),
        "n_launched": len(runs),
        "registry": registry_summary(paths["registry"]),
        "runs": runs,
    }

    write_json(paths["summary"], summary)
    return summary


def print_human_summary(summary: dict[str, Any]) -> None:
    registry = summary["registry"]

    print("Linear S-parameter campaign")
    print("===========================")
    print(f"campaign_dir: {summary['campaign_dir']}")
    print(f"z_lines_ohm:  {summary['z_lines_ohm']}")
    print(f"n_requested:  {summary['n_requested']}")
    print(f"n_launched:   {summary['n_launched']}")
    print(f"by_status:    {registry['by_status']}")
    print(f"by_type:      {registry['by_simulation_type']}")
    print()

    for run in summary["runs"]:
        metrics = run.get("metrics") or {}
        print(
            f"{run['run_name']}: "
            f"status={run['status']} "
            f"ok={run['ok']} "
            f"max|S11|={metrics.get('max_abs_s11')} "
            f"recip_err={metrics.get('reciprocal_error_max_abs')} "
            f"passivity={metrics.get('passivity_max_singular_value')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z-lines", type=float, nargs="+", default=[45.0, 50.0, 55.0])
    parser.add_argument(
        "--harmonia-root",
        type=Path,
        default=_WORKSPACE_ROOT / "Harmonia.jl",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=_WORKSPACE_ROOT / "outputs" / "campaigns" / "linear_sparams",
    )
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-frequency", type=int, default=101)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_campaign(
        z_lines_ohm=args.z_lines,
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
    return 0 if n_pass >= len(args.z_lines) else 1


if __name__ == "__main__":
    raise SystemExit(main())