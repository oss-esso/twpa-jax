"""Run the RCSJ damping ladder for the JC JTWPA.

Finite-resistance results are numerical continuation experiments.  The
campaign keeps every setting in a separate circuit and writes a resumable
summary after each setting so a long run cannot lose completed evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import load_circuit, save_circuit, stamp_rcsj_shunt  # noqa: E402


DEVICE_DIR = ROOT / "outputs" / "jc_doc_python_designs" / "jc_jtwpa"
RATIOS = (math.inf, 1.0e6, 1.0e4, 1.0e2, 1.0)
DELTA_EV = 180.0e-6
CJ_F = 55.0e-15
FREQ_GHZ = 7.12


def ratio_slug(ratio: float) -> str:
    return "inf" if math.isinf(ratio) else f"r{ratio:.0e}".replace("+", "")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_ladder_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "resistance_ratio", "rn_ohm_median", "r_ohm_median", "beta_c_median",
        "quality_factor_median", "damping_per_pump_period_median",
        "junction_capacitance_f_median",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return float(ordered[len(ordered) // 2])


def ladder_row(ratio: float, params: Any, variant: Path) -> dict[str, Any]:
    summary = params.summary
    return {
        "resistance_ratio": float(ratio),
        "rn_ohm_median": median(summary["rn_ohm"]),
        "r_ohm_median": median(summary["resistance_ohm"]),
        "beta_c_median": median(summary["beta_c"]),
        "quality_factor_median": median(summary["quality_factor"]),
        "damping_per_pump_period_median": median(summary["damping_per_pump_period"]),
        "junction_capacitance_f_median": median(summary["junction_capacitance_f"]),
        "parameters": summary,
        "variant_dir": str(variant),
    }


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def last_pass_checkpoint(
    hb_dir: Path, *, require_full_residual_gate: bool = False
) -> tuple[Path, float, float] | None:
    report_path = hb_dir / "hb_up_to_failure.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [row for row in report.get("rows", []) if row.get("status") == "PASS"]
    if not rows:
        return None
    if require_full_residual_gate:
        gated = []
        for candidate in rows:
            pump_dir = candidate.get("pump_dir")
            if not pump_dir:
                continue
            candidate_path = Path(str(pump_dir))
            candidate_path = candidate_path if candidate_path.is_absolute() else ROOT / candidate_path
            report_file = candidate_path / "pump_report.json"
            if not report_file.exists():
                continue
            pump_report = json.loads(report_file.read_text(encoding="utf-8"))
            residual = pump_report.get("metadata", {}).get("production_hb_full_residual_rel")
            if pump_report.get("final_status") == "VALID_CONVERGED" and residual is not None and float(residual) <= 1.0e-8:
                gated.append(candidate)
        rows = gated
        if not rows:
            return None
    row = rows[-1]
    pump_dir = row.get("pump_dir")
    if pump_dir:
        candidate = Path(str(pump_dir))
        checkpoint = candidate if candidate.is_absolute() else ROOT / candidate
    else:
        raise FileNotFoundError("HB row has no pump_dir")
    return checkpoint, float(row["pump_power_dbm"]), float(row["pump_current_peak_a"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-dir", type=Path, default=DEVICE_DIR)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "rcsj_jtwpa_campaign")
    parser.add_argument("--run-control", action="store_true")
    parser.add_argument("--run-transient", action="store_true")
    parser.add_argument("--run-monodromy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-transient", action="store_true")
    parser.add_argument("--hold-periods", type=int, default=800)
    args = parser.parse_args(argv)
    device_dir = args.device_dir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    variants_dir = args.outdir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    base = load_circuit(device_dir)
    all_rows: list[dict[str, Any]] = []
    ladder_rows: list[dict[str, Any]] = []
    summary_path = args.outdir / "campaign_summary.json"

    for ratio in RATIOS:
        slug = ratio_slug(ratio)
        variant = variants_dir / slug
        hb_dir = args.outdir / slug / "hb"
        td_dir = args.outdir / slug / "transient"
        floquet_dir = args.outdir / slug / "monodromy"
        if math.isinf(ratio):
            circuit = base
            params = stamp_rcsj_shunt(
                circuit, ratio, junction_capacitance_f=CJ_F,
                delta_ev=DELTA_EV, pump_frequency_hz=FREQ_GHZ * 1e9,
            )[1]
            variant = device_dir
        else:
            if args.force or not (variant / "C.npz").exists():
                damped, params = stamp_rcsj_shunt(
                    base, ratio, junction_capacitance_f=CJ_F,
                    delta_ev=DELTA_EV, pump_frequency_hz=FREQ_GHZ * 1e9,
                )
                save_circuit(damped, variant)
                circuit = damped
            else:
                damped = load_circuit(variant)
                _, params = stamp_rcsj_shunt(
                    base, ratio, junction_capacitance_f=CJ_F,
                    delta_ev=DELTA_EV, pump_frequency_hz=FREQ_GHZ * 1e9,
                )
                circuit = damped
        row = ladder_row(ratio, params, variant)
        row["device_dir"] = str(device_dir)
        row["has_loss"] = bool(circuit.has_loss)
        row["default_loss_model"] = (
            "conductance_abs_omega" if circuit.has_loss else "current_complex_c"
        )

        if args.run_control and math.isinf(ratio):
            hb_dir.mkdir(parents=True, exist_ok=True)
        elif not math.isinf(ratio) and (args.force or not (hb_dir / "hb_up_to_failure.json").exists()):
            command = [
                sys.executable, str(ROOT / "scripts" / "run_hb_column_until_failure.py"),
                "--circuit-dir", str(variant), "--outdir", str(hb_dir),
                "--freq-ghz", str(FREQ_GHZ), "--n-power", "120",
                "--pump-mode-count", "10", "--nt", "40",
                "--power-min-dbm", "-36", "--power-max-dbm", "-24",
            ]
            row["hb_returncode"] = run_command(command, hb_dir / "campaign.log")
        control_hb = ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811" / "jtwpa_ultrafine"
        if math.isinf(ratio) and not (hb_dir / "hb_up_to_failure.json").exists():
            row["hb_artifact"] = str(control_hb)
            hb_dir = control_hb
        hb_checkpoint = last_pass_checkpoint(hb_dir)
        checkpoint = last_pass_checkpoint(hb_dir, require_full_residual_gate=True)
        if hb_checkpoint is not None:
            _hb_checkpoint_dir, hb_power_dbm, _hb_current_a = hb_checkpoint
            row["last_pass_power_dbm"] = hb_power_dbm
            row["last_pass_checkpoint"] = str(_hb_checkpoint_dir)
        if checkpoint is not None:
            checkpoint_dir, power_dbm, current_a = checkpoint
            row["validated_checkpoint_power_dbm"] = power_dbm
            row["validated_checkpoint"] = str(checkpoint_dir)
            if args.run_transient and (args.force or args.force_transient or not (td_dir / "summary.json").exists()):
                td_command = [
                    sys.executable, str(ROOT / "scripts" / "h1_transient_branch_transfer.py"),
                    "--circuit-dir", str(variant), "--checkpoint", str(checkpoint_dir),
                    "--outdir", str(td_dir), "--freq-ghz", str(FREQ_GHZ),
                    "--pump-port", "1", "--out-port", "2", "--target-current-a", str(current_a),
                    "--ramp-periods", "40", "--hold-periods", str(args.hold_periods),
                    "--samples-per-period", "32", "--method", "BDF",
                    "--max-step", str(2.0 * math.pi / 32.0), "--compact-output",
                    "--compact-sample-count", "256", "--compact-history-states", "1024",
                    "--skip-projection",
                ]
                row["transient_returncode"] = run_command(td_command, td_dir / "campaign.log")
            if args.run_monodromy and ratio == 1.0e4 and (args.force or not (floquet_dir / "floquet_results.json").exists()):
                floquet_command = [
                    sys.executable, str(ROOT / "scripts" / "run_floquet_2c.py"),
                    "--circuit-dir", str(variant), "--checkpoints", str(checkpoint_dir),
                    "--output", str(floquet_dir), "--freq-ghz", str(FREQ_GHZ),
                    "--pump-port", "1", "--step-theta", str(2.0 * math.pi / 256.0),
                    "--eigenvalues", "40", "--eigensolver-ncv", "120",
                ]
                row["monodromy_returncode"] = run_command(floquet_command, floquet_dir / "campaign.log")
        if (td_dir / "summary.json").exists():
            row["transient_summary"] = json.loads((td_dir / "summary.json").read_text(encoding="utf-8"))
        if (floquet_dir / "floquet_results.json").exists():
            row["monodromy_summary"] = json.loads((floquet_dir / "floquet_results.json").read_text(encoding="utf-8"))
        all_rows.append(row)
        ladder_rows.append({key: row[key] for key in (
            "resistance_ratio", "rn_ohm_median", "r_ohm_median", "beta_c_median",
            "quality_factor_median", "damping_per_pump_period_median", "junction_capacitance_f_median",
        )})
        payload = {"device_dir": str(device_dir), "rows": all_rows}
        atomic_json(summary_path, payload)
        write_ladder_csv(args.outdir / "rcsj_ladder.csv", ladder_rows)
        print(f"R/Rn={ratio:g} last_pass={row.get('last_pass_power_dbm')} hb={row.get('hb_returncode')}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
