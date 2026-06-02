"""
Inspect a Julia/Harmonia simulation run folder from Python.

Example
-------
python scripts/read_julia_run.py D:/Projects/Thesis/outputs/julia_engine_smoke/schema_smoke_001
python scripts/read_julia_run.py D:/Projects/Thesis/outputs/julia_engine_smoke/schema_smoke_001 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow direct execution:
#   python scripts/read_julia_run.py <run_dir>
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from twpa.io.julia_bridge import load_julia_simulation


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.all(np.isfinite(value.real)) and np.all(np.isfinite(value.imag)))
            if np.iscomplexobj(value)
            else bool(np.all(np.isfinite(value))),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_summary(run_dir: Path) -> dict[str, Any]:
    data = load_julia_simulation(run_dir)
    status = data.status

    s_shape = None
    s21_gain_db = None
    reciprocal_error = None
    match_max_abs = None
    one_port_reflection_db = None

    if data.s_parameters is not None:
        s = data.s_parameters
        s_shape = list(s.shape)

        if s.ndim == 3 and s.shape[1:] == (2, 2):
            s21 = s[:, 1, 0]
            s12 = s[:, 0, 1]
            s11 = s[:, 0, 0]
            s22 = s[:, 1, 1]

            s21_gain_db = (
                20.0 * np.log10(np.maximum(np.abs(s21), 1e-300))
            ).tolist()
            reciprocal_error = float(np.max(np.abs(s21 - s12)))
            match_max_abs = float(max(np.max(np.abs(s11)), np.max(np.abs(s22))))

        elif s.ndim == 3 and s.shape[1:] == (1, 1):
            s11 = s[:, 0, 0]
            one_port_reflection_db = (
                20.0 * np.log10(np.maximum(np.abs(s11), 1e-300))
            ).tolist()

    return {
        "run_dir": str(run_dir),
        "schema_version": status.schema_version,
        "run_id": status.run_id,
        "status": status.status,
        "simulation_type": status.simulation_type,
        "circuit_template": status.circuit_template,
        "solver_success": status.solver_success,
        "residual_norm": status.residual_norm,
        "relative_residual_norm": status.relative_residual_norm,
        "failure_reason": status.failure_reason,
        "runtime_s": status.runtime_s,
        "h5_path": str(status.h5_path) if status.h5_path is not None else None,
        "frequency_shape": list(data.frequency_hz.shape) if data.frequency_hz is not None else None,
        "frequency_min_hz": float(np.min(data.frequency_hz)) if data.frequency_hz is not None else None,
        "frequency_max_hz": float(np.max(data.frequency_hz)) if data.frequency_hz is not None else None,
        "s_parameters_shape": s_shape,
        "gain_db_shape": list(data.gain_db.shape) if data.gain_db is not None else None,
        "gain_db_min": float(np.min(data.gain_db)) if data.gain_db is not None else None,
        "gain_db_max": float(np.max(data.gain_db)) if data.gain_db is not None else None,
        "s21_gain_db": s21_gain_db,
        "one_port_reflection_db": one_port_reflection_db,
        "reciprocal_error_max_abs": reciprocal_error,
        "match_max_abs": match_max_abs,
        "h5_attrs": data.h5_attrs,
    }


def print_human(summary: dict[str, Any]) -> None:
    print("Julia/Harmonia run summary")
    print("===========================")
    print(f"run_dir:          {summary['run_dir']}")
    print(f"status:           {summary['status']}")
    print(f"simulation_type:  {summary['simulation_type']}")
    print(f"template:         {summary['circuit_template']}")
    print(f"solver_success:   {summary['solver_success']}")
    print(f"residual_norm:    {summary['residual_norm']}")
    print(f"runtime_s:        {summary['runtime_s']}")
    print(f"h5_path:          {summary['h5_path']}")
    print()
    print("Arrays")
    print("------")
    print(f"frequency_shape:  {summary['frequency_shape']}")
    print(f"frequency_min_hz: {summary['frequency_min_hz']}")
    print(f"frequency_max_hz: {summary['frequency_max_hz']}")
    print(f"S shape:          {summary['s_parameters_shape']}")
    print(f"gain_db_shape:    {summary['gain_db_shape']}")
    print(f"gain_db_min/max:  {summary['gain_db_min']} / {summary['gain_db_max']}")

    if summary.get("reciprocal_error_max_abs") is not None or summary.get("match_max_abs") is not None:
        print()
        print("2-port sanity")
        print("-------------")
        print(f"reciprocal_error_max_abs: {summary['reciprocal_error_max_abs']}")
        print(f"match_max_abs:            {summary['match_max_abs']}")

    if summary.get("one_port_reflection_db") is not None:
        vals = summary["one_port_reflection_db"]
        print()
        print("1-port sanity")
        print("-------------")
        print(f"reflection_db_min/max: {min(vals)} / {max(vals)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Julia/Harmonia run directory")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = build_summary(args.run_dir)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_ready))
    else:
        print_human(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())