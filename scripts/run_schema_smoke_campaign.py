"""
Run a tiny schema-smoke Julia/Harmonia campaign from Python.

This is the first end-to-end campaign smoke:

    Python creates N configs
    Python launches Julia N times
    Julia writes status.json + simulation.h5
    Python registers every run in runs.csv
    Python prints campaign summary

This is not a physics campaign yet. It validates orchestration, folder layout,
status handling, caching, registry writing, and Python/Julia boundary discipline.

Example
-------
python scripts/run_schema_smoke_campaign.py ^
  --n 3 ^
  --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT =  Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.run_registry import register_run_dir, registry_summary


SCHEMA_VERSION = "0.1.0"


def make_schema_smoke_config(index: int, *, n_frequency: int = 5) -> dict[str, Any]:
    """
    Build a deterministic schema-smoke config.

    We deliberately vary the frequency window slightly with index, so the
    registry/campaign proves it is managing distinct configs and not only
    rerunning the exact same file.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    if n_frequency < 1:
        raise ValueError("n_frequency must be >= 1")

    f_start = 4.0e9 + index * 0.1e9
    f_stop = 8.0e9 + index * 0.1e9

    return {
        "schema_version": SCHEMA_VERSION,
        "simulation_type": "schema_smoke",
        "circuit_template": "matched_through_2port",
        "seed": 1234 + index,
        "parameters": {
            "z0_ohm": 50.0,
            "campaign_index": index,
        },
        "axes": {
            "frequency_hz": {
                "start": f_start,
                "stop": f_stop,
                "points": n_frequency,
            }
        },
        "solver": {
            "backend": "analytic_schema_smoke",
            "notes": (
                "Schema campaign smoke only. "
                "This is not a JosephsonCircuits HB solve."
            ),
        },
    }

def assert_json_serializable(obj: Any, *, context: str = "object") -> None:
    try:
        json.dumps(obj)
    except TypeError as exc:
        raise TypeError(f"{context} is not JSON serializable: {exc}") from exc

def write_json(path: Path, obj: dict[str, Any]) -> None:
    assert_json_serializable(obj, context=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def campaign_paths(campaign_dir: Path) -> dict[str, Path]:
    return {
        'configs': campaign_dir / 'configs',
        "runs": campaign_dir / "runs",
        "registry": campaign_dir / "runs.csv",
        "summary": campaign_dir / "campaign_summary.json", 
    }


def run_campaign(
    *,
    n: int,
    harmonia_root: Path,
    campaign_dir: Path,
    julia_executable: str = "julia",
    timeout_s: float = 300.0,
    force: bool = False,
    n_frequency: int = 5,
) -> dict[str, Any]:
    if n < 1:
        raise ValueError("n must be >= 1")

    paths = campaign_paths(campaign_dir)
    paths["configs"].mkdir(parents=True, exist_ok=True)
    paths["runs"].mkdir(parents=True, exist_ok=True)

    launched = []

    for idx in range(n):
        run_name = f"run_{idx:03d}"
        config_path = paths["configs"] / f"{run_name}.json"
        output_dir = paths["runs"] / run_name

        config = make_schema_smoke_config(idx, n_frequency=n_frequency)
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

        if result.status is None:
            launched.append(
                {
                    "run_name": run_name,
                    "returncode": result.returncode,
                    "status": "MISSING_STATUS",
                    "output_dir": str(output_dir),
                }
            )
            continue

        registered = register_run_dir(paths["registry"], output_dir)

        launched.append(
            {
                "run_name": run_name,
                "returncode": result.returncode,
                "ok": result.ok,
                "run_id": registered.run_id,
                "status": registered.status,
                "simulation_type": registered.simulation_type,
                "output_dir": registered.output_dir,
                "h5_path": registered.h5_path,
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_dir": str(campaign_dir),
        "harmonia_root": str(harmonia_root),
        "n_requested": n,
        "n_launched": len(launched),
        "registry": registry_summary(paths["registry"]),
        "runs": launched,
    }

    write_json(paths["summary"], summary)
    return summary


def print_human_summary(summary: dict[str, Any]) -> None:
    registry = summary["registry"]

    print("Schema-smoke campaign")
    print("=====================")
    print(f"campaign_dir: {summary['campaign_dir']}")
    print(f"n_requested:  {summary['n_requested']}")
    print(f"n_launched:   {summary['n_launched']}")
    print(f"registry_csv: {registry['registry_csv']}")
    print(f"by_status:    {registry['by_status']}")
    print(f"by_type:      {registry['by_simulation_type']}")
    print()

    for run in summary["runs"]:
        print(
            f"{run['run_name']}: "
            f"returncode={run['returncode']} "
            f"status={run['status']} "
            f"ok={run.get('ok')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="Number of tiny runs")
    parser.add_argument(
        "--harmonia-root",
        type=Path,
        default=_WORKSPACE_ROOT / "Harmonia.jl",
        help="Path to Harmonia.jl repo",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=_WORKSPACE_ROOT / "outputs" / "campaigns" / "schema_smoke",
        help="Campaign output directory",
    )
    parser.add_argument("--julia", default="julia", help="Julia executable")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--force", action="store_true", help="Rerun even if outputs exist")
    parser.add_argument("--n-frequency", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_campaign(
        n=args.n,
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

    by_status = summary["registry"]["by_status"]
    n_pass = by_status.get("PASS", 0)

    return 0 if n_pass >= args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())