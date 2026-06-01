"""
Launch a Harmonia/JosephsonCircuits Julia simulation from Python.

Example
-------
python scripts/run_julia_simulation.py ^
  --harmonia-root D:/Projects/Thesis/Harmonia.jl ^
  --config D:/Projects/Thesis/Harmonia.jl/examples/configs/schema_smoke.json ^
  --output D:/Projects/Thesis/outputs/python_launched_schema_smoke/run_001 ^
  --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpa.io.julia_runner import run_harmonia_simulation


def _status_summary(result) -> dict[str, Any]:
    status = result.status

    return {
        "returncode": result.returncode,
        "ok": result.ok,
        "output_dir": str(result.output_dir),
        "command": list(result.command),
        "status": None if status is None else status.status,
        "simulation_type": None if status is None else status.simulation_type,
        "solver_success": None if status is None else status.solver_success,
        "residual_norm": None if status is None else status.residual_norm,
        "failure_reason": None if status is None else status.failure_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonia-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_harmonia_simulation(
        config_path=args.config,
        output_dir=args.output,
        harmonia_jl_root=args.harmonia_root,
        julia_executable=args.julia,
        timeout_s=args.timeout_s,
        force=args.force,
        use_cache=not args.no_cache,
    )

    summary = _status_summary(result)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Python-launched Julia simulation")
        print("================================")
        print(f"returncode:       {summary['returncode']}")
        print(f"ok:               {summary['ok']}")
        print(f"output_dir:       {summary['output_dir']}")
        print(f"status:           {summary['status']}")
        print(f"simulation_type:  {summary['simulation_type']}")
        print(f"solver_success:   {summary['solver_success']}")
        print(f"residual_norm:    {summary['residual_norm']}")
        print(f"failure_reason:   {summary['failure_reason']}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())