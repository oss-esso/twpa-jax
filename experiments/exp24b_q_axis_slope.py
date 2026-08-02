"""Run the corrected q-axis compression matrix and q<=3 spot checks."""

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
MATRIX_FREQUENCIES_GHZ = (5.8, 6.03, 6.257, 6.49, 6.71, 6.943, 7.29, 7.629)
SPOT_FREQUENCIES_GHZ = (6.943, 7.629)


def run_dir(root: Path, sidebands: int, frequency_ghz: float) -> Path:
    """Return the stable resumable directory for one case."""
    return root / f"q{sidebands:02d}" / f"frequency_{frequency_ghz:.6f}ghz"


def command(
    output_dir: Path,
    sidebands: int,
    frequency_ghz: float,
    workers: int,
) -> list[str]:
    """Build the fixed compression command for one frequency."""
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
        "--signal-current-min-a", "1e-10",
        "--signal-current-max-a", "1e-6",
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", "600",
        "--signal-workers", str(workers),
        "--signal-ghz", str(frequency_ghz),
        "--allow-memory-overcommit",
    ]


def _nearest_p1db_row(summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Return the sampled row nearest the refined P1dB current."""
    p1db_current = float(summary["p1db_signal_current_a"])
    with (output_dir / "compression_points.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return min(
        rows,
        key=lambda row: abs(
            math.log(float(row["signal_current_a"]) / p1db_current)
        ),
    )


def read_case(root: Path, sidebands: int, frequency_ghz: float) -> dict[str, Any]:
    """Read one case and apply the corrected scope-free gate."""
    output_dir = run_dir(root, sidebands, frequency_ghz)
    summary_path = output_dir / "compression_summary.json"
    if not summary_path.exists():
        return {
            "q": sidebands,
            "signal_ghz": frequency_ghz,
            "status": "MISSING",
            "gate_ok": False,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    p1db = summary.get("p1db_input_dbm")
    row: dict[str, Any] = {
        "q": sidebands,
        "signal_ghz": frequency_ghz,
        "status": summary.get("status"),
        "small_signal_gain_db": summary.get("small_signal_gain_vs_off_db"),
        "p1db_input_dbm": p1db,
        "number_of_crossings": summary.get("number_of_crossings"),
        "max_power_balance_rel_err": summary.get("max_power_balance_rel_err"),
        "max_conversion_manley_rowe_rel_err": summary.get(
            "max_external_manley_rowe_rel_err"
        ),
    }
    if p1db is not None and summary.get("p1db_signal_current_a") is not None:
        nearest = _nearest_p1db_row(summary, output_dir)
        row.update(
            {
                "power_balance_rel_err_at_p1db": float(
                    nearest["power_balance_rel_err"]
                ),
                "conversion_manley_rowe_rel_err_at_p1db": float(
                    nearest["external_manley_rowe_rel_err"]
                ),
                "p1db_sample_signal_current_a": float(
                    nearest["signal_current_a"]
                ),
            }
        )
    row["gate_ok"] = bool(
        row["status"] == "VALID_SOLVED"
        and p1db is not None
        and math.isfinite(float(p1db))
        and float(row["max_power_balance_rel_err"]) < 1e-6
    )
    return row


def run_case(
    root: Path,
    sidebands: int,
    frequency_ghz: float,
    workers: int,
) -> tuple[dict[str, Any], int]:
    """Run or resume one frequency case."""
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


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    """Write the combined machine-readable matrix summary."""
    root.mkdir(parents=True, exist_ok=True)
    existing_path = root / "q_axis_summary.json"
    existing = (
        json.loads(existing_path.read_text(encoding="utf-8"))
        if existing_path.exists()
        else []
    )
    merged = {
        (int(row["q"]), float(row["signal_ghz"])): row
        for row in existing + rows
    }
    ordered = sorted(
        merged.values(), key=lambda row: (int(row["q"]), float(row["signal_ghz"]))
    )
    (root / "q_axis_summary.json").write_text(
        json.dumps(ordered, indent=2), encoding="utf-8"
    )
    fields = sorted({key for row in ordered for key in row})
    with (root / "q_axis_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def run_matrix(root: Path, frequencies: tuple[float, ...]) -> list[dict[str, Any]]:
    """Run q<=1 concurrently, then q<=2 sequentially."""
    rows: list[dict[str, Any]] = []
    for sidebands, workers in ((1, 2), (2, 1)):
        if sidebands == 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(run_case, root, sidebands, frequency, workers)
                    for frequency in frequencies
                ]
                results = [future.result() for future in as_completed(futures)]
        else:
            results = [
                run_case(root, sidebands, frequency, workers)
                for frequency in frequencies
            ]
        rows.extend(result[0] for result in results)
        write_summary(root, rows)
    return rows


def run_spot_checks(root: Path, frequencies: tuple[float, ...]) -> list[dict[str, Any]]:
    """Run q<=3 and compare finite P1dB values with q<=2."""
    q3_rows = [
        run_case(root, 3, frequency, 1)[0] for frequency in frequencies
    ]
    summary_path = root / "q_axis_summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8"))
    write_summary(root, existing + q3_rows)
    checks = []
    for q3 in q3_rows:
        q2 = read_case(root, 2, float(q3["signal_ghz"]))
        p3 = q3.get("p1db_input_dbm")
        p2 = q2.get("p1db_input_dbm")
        delta = None if p3 is None or p2 is None else float(p3) - float(p2)
        checks.append(
            {
                "signal_ghz": q3["signal_ghz"],
                "p1db_q2_dbm": p2,
                "p1db_q3_dbm": p3,
                "delta_q3_minus_q2_db": delta,
                "gate_ok": delta is not None and abs(delta) < 0.5,
            }
        )
    (root / "q3_spot_check.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    with (root / "q3_spot_check.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp24b_q_axis_slope")
    )
    parser.add_argument("--spot-check", action="store_true")
    parser.add_argument("--signal-ghz", type=float, action="append")
    args = parser.parse_args()
    if args.spot_check:
        checks = run_spot_checks(args.output_dir, SPOT_FREQUENCIES_GHZ)
        print(json.dumps(checks, indent=2), flush=True)
        return 0
    frequencies = tuple(args.signal_ghz or MATRIX_FREQUENCIES_GHZ)
    rows = run_matrix(args.output_dir, frequencies)
    missing = [row for row in rows if not row["gate_ok"]]
    print(f"matrix_cases={len(rows)} gate_failures={len(missing)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
