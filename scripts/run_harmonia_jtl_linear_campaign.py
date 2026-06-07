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
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpa.io.julia_bridge import read_status_json
from twpa.io.campaigns import (
    campaign_paths,
    compute_two_port_run_metrics,
    register_completed_run,
)
from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.julia_batch_runner import run_harmonia_simulation_batch
from twpa.io.run_registry import registry_summary
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


def compute_jtl_linear_metrics(run_dir: Path) -> dict[str, Any]:
    return compute_two_port_run_metrics(run_dir)


def _jtl_run_name(lj_h: float) -> str:
    return f"Lj_{lj_h:.3e}_H".replace("+", "").replace("-", "m").replace(".", "p")


def _result_from_status_path(*, output_dir: Path, returncode: int) -> Any:
    status_path = output_dir / "status.json"
    status = read_status_json(status_path) if status_path.exists() else None
    ok = returncode == 0 and status is not None and status.status == "PASS"
    return SimpleNamespace(
        returncode=returncode,
        ok=ok,
        output_dir=output_dir,
        status=status,
    )


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
    use_batch_runner: bool = False,
    jtl_linear_backend: str = "hbsolve",
    enable_jc_setup_cache: bool = False,
) -> dict[str, Any]:
    if not lj_values_h:
        raise ValueError("lj_values_h must not be empty")

    if jtl_linear_backend not in {"hbsolve", "hblinsolve_direct"}:
        raise ValueError("jtl_linear_backend must be 'hbsolve' or 'hblinsolve_direct'")

    if force and campaign_dir.exists():
        shutil.rmtree(campaign_dir)

    paths = campaign_paths(campaign_dir)
    paths["configs"].mkdir(parents=True, exist_ok=True)
    paths["runs"].mkdir(parents=True, exist_ok=True)

    prepared: list[tuple[float, str, Path, Path]] = []

    for idx, lj_h in enumerate(lj_values_h):
        run_name = _jtl_run_name(float(lj_h))
        config_path = paths["configs"] / f"{run_name}.json"
        output_dir = paths["runs"] / run_name

        config = make_harmonia_jtl_linear_config(
            index=idx,
            Lj_H=float(lj_h),
            n_cell=n_cell,
            n_frequency=n_frequency,
        )
        config["solver"] = {
            "jtl_linear_backend": jtl_linear_backend,
            "enable_jc_setup_cache": bool(enable_jc_setup_cache),
        }
        write_json(config_path, config)
        prepared.append((float(lj_h), run_name, config_path, output_dir))

    runs: list[dict[str, Any]] = []

    if use_batch_runner:
        batch_result = run_harmonia_simulation_batch(
            items=[(config_path, output_dir) for _, _, config_path, output_dir in prepared],
            harmonia_jl_root=harmonia_root,
            julia_executable=julia_executable,
            timeout_s=timeout_s,
            force=force,
            use_cache=not force,
            batch_work_dir=paths["runs"] / "_julia_batch_runner",
        )
        returncode_by_output = {
            Path(record.output_dir).resolve(): int(record.returncode)
            for record in batch_result.records
        }

        for lj_h, run_name, _, output_dir in prepared:
            result = _result_from_status_path(
                output_dir=output_dir,
                returncode=returncode_by_output.get(output_dir.resolve(), int(batch_result.returncode)),
            )
            run_record: dict[str, Any] = {
                "run_name": run_name,
                "Lj_H": float(lj_h),
                "batch_runner": True,
            }
            run_record.update(register_completed_run(
                registry_csv=paths["registry"],
                run_dir=output_dir,
                result=result,
                compute_metrics=compute_jtl_linear_metrics,
            ))
            runs.append(run_record)

    else:
        for lj_h, run_name, config_path, output_dir in prepared:
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
                "batch_runner": False,
            }
            run_record.update(register_completed_run(
                registry_csv=paths["registry"],
                run_dir=output_dir,
                result=result,
                compute_metrics=compute_jtl_linear_metrics,
            ))
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
        "use_batch_runner": bool(use_batch_runner),
        "jtl_linear_backend": jtl_linear_backend,
        "enable_jc_setup_cache": bool(enable_jc_setup_cache),
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
    parser.add_argument("--use-batch-runner", action="store_true", help="Run campaign points through one Julia batch process.")
    parser.add_argument("--jtl-linear-backend", choices=["hbsolve", "hblinsolve_direct"], default="hbsolve", help="Backend for harmonia_jtl_linear_jc_smoke.")
    parser.add_argument("--enable-jc-setup-cache", action="store_true", help="Request JC setup-cache integration telemetry for supported backends.")
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
        use_batch_runner=args.use_batch_runner,
        jtl_linear_backend=args.jtl_linear_backend,
        enable_jc_setup_cache=args.enable_jc_setup_cache,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)

    n_pass = summary["registry"]["by_status"].get("PASS", 0)
    return 0 if n_pass >= len(args.lj_values_h) else 1


if __name__ == "__main__":
    raise SystemExit(main())
