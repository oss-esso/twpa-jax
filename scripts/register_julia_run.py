"""
Register a Julia/Harmonia run folder in a CSV registry.

Example
-------
python scripts/register_julia_run.py ^
  --run D:/Projects/Thesis/outputs/python_launched_schema_smoke/run_001 ^
  --registry D:/Projects/Thesis/outputs/campaigns/schema_smoke/runs.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpa.io.run_registry import register_run_dir, registry_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Julia/Harmonia run directory")
    parser.add_argument("--registry", type=Path, required=True, help="CSV registry path")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    registered = register_run_dir(args.registry, args.run)
    summary = registry_summary(args.registry)

    if args.json:
        print(
            json.dumps(
                {
                    "registered_run": registered.__dict__,
                    "summary": summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("Registered Julia/Harmonia run")
        print("=============================")
        print(f"registry:        {args.registry}")
        print(f"run_id:          {registered.run_id}")
        print(f"status:          {registered.status}")
        print(f"simulation_type: {registered.simulation_type}")
        print(f"output_dir:      {registered.output_dir}")
        print()
        print("Registry summary")
        print("----------------")
        print(f"n_runs:          {summary['n_runs']}")
        print(f"by_status:       {summary['by_status']}")
        print(f"by_type:         {summary['by_simulation_type']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())