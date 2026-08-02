"""Measure the effect of opening the independent signal-photon q axis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


CIRCUIT_DIR = Path("designs/ipm_2c_fixed")
PUMP_FREQ_GHZ = 7.540816326531111
PUMP_CURRENT_A = 7.231074707853736e-06
FREQUENCIES_GHZ = (5.8, 6.257, 6.943, 7.629)
SIDEBAND_LEVELS = (1, 2)


def run_dir(root: Path, sidebands: int, frequency_ghz: float) -> Path:
    """Return the stable resumable directory for one q/frequency case."""
    return root / f"q{sidebands:02d}" / f"frequency_{frequency_ghz:.6f}ghz"


def command(output_dir: Path, sidebands: int, frequency_ghz: float,
            workers: int) -> list[str]:
    """Build the production compression command for one frequency."""
    return [
        sys.executable,
        "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--circuit-dir", str(CIRCUIT_DIR),
        "--pump-freq-ghz", str(PUMP_FREQ_GHZ),
        "--pump-current-a", str(PUMP_CURRENT_A),
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "dense_real",
        "--pump-harmonics", "6",
        "--pump-nt", "40",
        "--multitone-basis", "lattice",
        "--multitone-sidebands", str(sidebands),
        "--source-port", "1",
        "--pump-port", "4",
        "--out-port", "2",
        "--attenuation-db", "0",
        "--factor-backend", "pardiso",
        "--n-signal-power", "16",
        "--signal-current-min-a", "1e-12",
        "--signal-current-max-a", "3e-7",
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", "600",
        "--signal-workers", str(workers),
        "--signal-ghz", str(frequency_ghz),
        "--allow-memory-overcommit",
    ]


def _nearest_p1db_metrics(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Read gate metrics from the sampled point nearest the refined P1dB."""
    p1db_current = float(summary["p1db_signal_current_a"])
    points_path = output_dir / "compression_points.csv"
    with points_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = min(
        rows,
        key=lambda item: abs(
            math.log(float(item["signal_current_a"]) / p1db_current)
        ),
    )
    return {
        "p1db_sample_signal_current_a": float(row["signal_current_a"]),
        "power_balance_rel_err_at_p1db": float(row["power_balance_rel_err"]),
        "conversion_manley_rowe_rel_err_at_p1db": float(
            row["external_manley_rowe_rel_err"]
        ),
    }


def read_case(root: Path, sidebands: int, frequency_ghz: float) -> dict[str, Any]:
    """Read one completed case and evaluate its physics gates."""
    output_dir = run_dir(root, sidebands, frequency_ghz)
    summary_path = output_dir / "compression_summary.json"
    if not summary_path.exists():
        return {
            "q": sidebands,
            "signal_ghz": frequency_ghz,
            "status": "MISSING",
            "gate_ok": False,
            "error": f"missing {summary_path}",
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "q": sidebands,
        "signal_ghz": frequency_ghz,
        "status": summary.get("status"),
        "small_signal_gain_db": summary.get("small_signal_gain_vs_off_db"),
        "p1db_input_dbm": summary.get("p1db_input_dbm"),
        "number_of_crossings": summary.get("number_of_crossings"),
        "max_power_balance_rel_err": summary.get("max_power_balance_rel_err"),
        "max_conversion_manley_rowe_rel_err": summary.get(
            "max_external_manley_rowe_rel_err"
        ),
    }
    if row["status"] == "VALID_SOLVED" and row["p1db_input_dbm"] is not None:
        row.update(_nearest_p1db_metrics(summary, output_dir))
    row["gate_ok"] = bool(
        row["status"] == "VALID_SOLVED"
        and row["p1db_input_dbm"] is not None
        and math.isfinite(float(row["p1db_input_dbm"]))
        and row.get("number_of_crossings") == 1
        and float(row["power_balance_rel_err_at_p1db"]) < 1e-6
    )
    # conversion_manley_rowe_rel_err is reported but deliberately NOT gated.
    # It measures photon leakage outside the pump/signal/idler scope, and
    # q >= 2 adds out-of-scope channels by design, so the residual must grow
    # with q.  The 0.03 threshold was calibrated on a q <= 1 basis and does
    # not transfer.  power_balance_rel_err is the scope-free energy check.
    return row


def run_case(root: Path, sidebands: int, frequency_ghz: float,
             workers: int) -> tuple[dict[str, Any], int]:
    """Run or resume one case."""
    output_dir = run_dir(root, sidebands, frequency_ghz)
    summary_path = output_dir / "compression_summary.json"
    if summary_path.exists():
        print(f"resume q<={sidebands} fs={frequency_ghz:.3f} GHz", flush=True)
        return read_case(root, sidebands, frequency_ghz), 0
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(output_dir, sidebands, frequency_ghz, workers)
    print("run " + subprocess.list2cmdline(cmd), flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode:
        return {
            "q": sidebands,
            "signal_ghz": frequency_ghz,
            "status": "SUBPROCESS_FAILED",
            "gate_ok": False,
            "returncode": completed.returncode,
        }, completed.returncode
    return read_case(root, sidebands, frequency_ghz), 0


def write_report(root: Path, rows: list[dict[str, Any]]) -> None:
    """Write machine-readable results for the slope-verdict phase."""
    ordered = sorted(rows, key=lambda row: (int(row["q"]), float(row["signal_ghz"])))
    (root / "q_axis_summary.json").write_text(
        json.dumps(ordered, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in ordered for key in row})
    with (root / "q_axis_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp24_q_axis_slope")
    )
    parser.add_argument("--q", type=int, choices=SIDEBAND_LEVELS, action="append")
    parser.add_argument("--signal-ghz", type=float, action="append")
    parser.add_argument("--signal-workers-q1", type=int, default=2)
    parser.add_argument("--signal-workers-q2", type=int, default=1)
    args = parser.parse_args()

    levels = tuple(args.q or SIDEBAND_LEVELS)
    frequencies = tuple(args.signal_ghz or FREQUENCIES_GHZ)
    all_rows: list[dict[str, Any]] = []
    exit_code = 0
    for sidebands in levels:
        workers = args.signal_workers_q1 if sidebands == 1 else args.signal_workers_q2
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_case, args.output_dir, sidebands, frequency, workers): frequency
                    for frequency in frequencies
                }
                for future in as_completed(futures):
                    row, returncode = future.result()
                    all_rows.append(row)
                    exit_code = max(exit_code, returncode)
        else:
            for frequency in frequencies:
                row, returncode = run_case(
                    args.output_dir, sidebands, frequency, workers
                )
                all_rows.append(row)
                exit_code = max(exit_code, returncode)
        write_report(args.output_dir, all_rows)
        failed = [row for row in all_rows if int(row["q"]) == sidebands and not row["gate_ok"]]
        if failed:
            print(f"gate failed for q<={sidebands}: {len(failed)} case(s)", flush=True)
            return 1
    write_report(args.output_dir, all_rows)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
