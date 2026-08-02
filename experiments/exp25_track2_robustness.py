"""Re-run and diagnose the two exp24b q<=2 robustness failures."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DEFAULT = Path("outputs/exp25_track2_robustness")
CASES = (5.8, 6.03)


def command(output_dir: Path, frequency_ghz: float, include_debug_flag: bool) -> list[str]:
    """Build the corrected q<=2 command, optionally with the requested flag."""
    command = [
        sys.executable,
        "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--circuit-dir", "designs/ipm_2c_fixed",
        "--pump-freq-ghz", "7.540816326531111",
        "--pump-current-a", "7.231074707853736e-06",
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "dense_real",
        "--pump-harmonics", "6",
        "--pump-nt", "40",
        "--multitone-basis", "lattice",
        "--multitone-sidebands", "2",
        "--source-port", "1",
        "--pump-port", "4",
        "--out-port", "2",
        "--attenuation-db", "0",
        "--factor-backend", "pardiso",
        "--n-signal-power", "16",
        "--signal-current-min-a", "1e-10",
        "--signal-current-max-a", "1e-6",
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", "600",
        "--signal-workers", "1",
        "--signal-ghz", str(frequency_ghz),
        "--allow-memory-overcommit",
    ]
    if include_debug_flag:
        command.extend(("--log-level", "DEBUG"))
    return command


def run_logged(command_line: list[str], log_path: Path) -> int:
    """Run a command and persist both streams for post-mortem inspection."""
    result = subprocess.run(command_line, capture_output=True, text=True, check=False)
    log_path.write_text(
        "COMMAND: " + subprocess.list2cmdline(command_line) + "\n\n"
        + result.stdout + "\n--- STDERR ---\n" + result.stderr,
        encoding="utf-8",
    )
    return int(result.returncode)


def case_report(case_dir: Path, frequency_ghz: float, flag_returncode: int,
                solver_returncode: int) -> dict[str, Any]:
    """Summarize solver statuses, gaps, and crossing calculations."""
    points_path = case_dir / "compression_points.csv"
    summary_path = case_dir / "compression_summary.json"
    report: dict[str, Any] = {
        "signal_ghz": frequency_ghz,
        "log_level_flag_returncode": flag_returncode,
        "solver_returncode": solver_returncode,
        "summary_exists": summary_path.exists(),
    }
    if not points_path.exists():
        return report
    rows = list(csv.DictReader(points_path.open(newline="", encoding="utf-8")))
    valid = [row for row in rows if row["status"] == "VALID_SOLVED"]
    missing = [row for row in rows if row["status"] != "VALID_SOLVED"]
    compression_crossings_raw = [
        index for index, (left, right) in enumerate(zip(rows, rows[1:]))
        if left["status"] == right["status"] == "VALID_SOLVED"
        and float(left["compression_db"]) < 1.0
        and float(right["compression_db"]) >= 1.0
    ]
    compression_crossings_valid = [
        index for index, (left, right) in enumerate(zip(valid, valid[1:]))
        if float(left["compression_db"]) < 1.0
        and float(right["compression_db"]) >= 1.0
    ]
    report.update({
        "n_rows": len(rows),
        "n_valid": len(valid),
        "missing_rows": [
            {
                "index": rows.index(row),
                "signal_power_dbm": row["signal_power_dbm"],
                "status": row["status"],
                "compression_db": row["compression_db"],
            }
            for row in missing
        ],
        "raw_adjacent_crossing_indices": compression_crossings_raw,
        "valid_sequence_crossing_indices": compression_crossings_valid,
    })
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report.update({
            "status": summary.get("status"),
            "p1db_input_dbm": summary.get("p1db_input_dbm"),
            "p1db_interpolated_dbm": summary.get("p1db_interpolated_dbm"),
            "number_of_crossings": summary.get("number_of_crossings"),
        })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for frequency in CASES:
        case_dir = args.output_dir / f"frequency_{frequency:.6f}ghz"
        case_dir.mkdir(parents=True, exist_ok=True)
        debug_code = run_logged(
            command(case_dir, frequency, True), case_dir / "log_level_attempt.log"
        )
        summary = case_dir / "compression_summary.json"
        if summary.exists():
            solver_code = 0
        else:
            solver_code = run_logged(
                command(case_dir, frequency, False), case_dir / "solver.log"
            )
        reports.append(case_report(case_dir, frequency, debug_code, solver_code))
        (args.output_dir / "robustness_report.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
