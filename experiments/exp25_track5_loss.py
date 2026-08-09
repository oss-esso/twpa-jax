"""Build real-G lossy 2c variants and measure q<=1 P1dB slopes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


TAN_DELTAS = (1e-4, 1e-3, 1e-2)
FREQUENCIES = (5.8, 6.943, 7.629)
PUMP_GHZ = 7.540816326531111


def variant_name(tan_delta: float) -> str:
    return f"ipm_python_design_tand{tan_delta:g}"


def build_variant(source: Path, destination: Path, tan_delta: float) -> None:
    """Copy a circuit and add diagonal internal-node conductance."""
    shutil.copytree(source, destination, dirs_exist_ok=True)
    capacitance = sp.load_npz(source / "C.npz").tocsr()
    conductance = sp.load_npz(source / "G.npz").tocsr()
    port_nodes = {int(row) for row, _ in zip(*conductance.nonzero())}
    internal = np.ones(capacitance.shape[0], dtype=bool)
    internal[list(port_nodes)] = False
    diagonal = np.zeros(capacitance.shape[0], dtype=float)
    diagonal[internal] = (
        2.0 * np.pi * PUMP_GHZ * 1e9 * capacitance.diagonal()[internal] * tan_delta
    )
    updated = conductance + sp.diags(diagonal, format="csr")
    sp.save_npz(destination / "G.npz", updated)
    (destination / "loss_variant.json").write_text(
        json.dumps({
            "tan_delta": tan_delta,
            "pump_omega_rad_per_s": 2.0 * np.pi * PUMP_GHZ * 1e9,
            "port_nodes_untouched": sorted(port_nodes),
            "internal_diagonal_nonzero": int(np.count_nonzero(diagonal)),
            "g_added_min_s": float(np.min(diagonal[internal])),
            "g_added_max_s": float(np.max(diagonal[internal])),
        }, indent=2), encoding="utf-8"
    )


def command(output_dir: Path, circuit_dir: Path, frequency: float) -> list[str]:
    """Build the exp24b q<=1 command for one lossy circuit."""
    return [
        sys.executable, "scripts/run_compression.py",
        "--output-dir", str(output_dir), "--circuit-dir", str(circuit_dir),
        "--pump-freq-ghz", str(PUMP_GHZ),
        "--pump-current-a", "7.231074707853736e-06",
        "--pump-current-jc-scale", "1.0", "--pump-mode-policy", "dense_real",
        "--pump-harmonics", "6", "--pump-nt", "40",
        "--multitone-basis", "lattice", "--multitone-sidebands", "1",
        "--source-port", "1", "--pump-port", "4", "--out-port", "2",
        "--attenuation-db", "0", "--factor-backend", "pardiso",
        "--n-signal-power", "16", "--signal-current-min-a", "1e-10",
        "--signal-current-max-a", "1e-6", "--recovery", "ladder",
        "--signal-continuation-deadline-s", "600", "--signal-workers", "1",
        "--signal-ghz", str(frequency), "--allow-memory-overcommit",
    ]


def read_result(output_dir: Path, tan_delta: float, frequency: float,
                returncode: int) -> dict[str, object]:
    """Read a completed summary or retain a subprocess failure row."""
    summary_path = output_dir / "compression_summary.json"
    row: dict[str, object] = {
        "tan_delta": tan_delta, "signal_ghz": frequency,
        "returncode": returncode,
    }
    if not summary_path.exists():
        row["status"] = "SUBPROCESS_FAILED"
        return row
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row.update({
        "status": summary.get("status"),
        "small_signal_gain_db": summary.get("small_signal_gain_vs_off_db"),
        "p1db_input_dbm": summary.get("p1db_input_dbm"),
        "number_of_crossings": summary.get("number_of_crossings"),
        "max_power_balance_rel_err": summary.get("max_power_balance_rel_err"),
    })
    return row


def slope_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Fit P1dB against gain for each loss tangent with finite crossings."""
    output: list[dict[str, object]] = []
    for tan_delta in TAN_DELTAS:
        finite = [
            row for row in rows
            if float(row["tan_delta"]) == tan_delta
            and row.get("p1db_input_dbm") is not None
        ]
        gain = np.asarray([float(row["small_signal_gain_db"]) for row in finite])
        p1db = np.asarray([float(row["p1db_input_dbm"]) for row in finite])
        result: dict[str, object] = {"tan_delta": tan_delta, "n": int(gain.size)}
        if gain.size >= 2 and np.ptp(gain) > 0.0:
            slope, intercept = np.polyfit(gain, p1db, 1)
            residual = p1db - (slope * gain + intercept)
            dof = max(gain.size - 2, 1)
            standard_error = math.sqrt(float(np.sum(residual**2)) / dof / float(np.sum((gain - gain.mean()) ** 2)))
            result.update({
                "slope_db_per_db": float(slope),
                "slope_se_db_per_db": standard_error,
                "intercept_dbm": float(intercept),
                "fit_rms_db": float(np.sqrt(np.mean(residual**2))),
            })
        else:
            result.update({
                "slope_db_per_db": None,
                "slope_se_db_per_db": None,
                "intercept_dbm": None,
                "fit_rms_db": None,
            })
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("designs/ipm_2c_fixed"))
    parser.add_argument("--circuit-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp25_track5_loss"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for tan_delta in TAN_DELTAS:
        circuit_dir = args.circuit_root / variant_name(tan_delta)
        build_variant(args.source, circuit_dir, tan_delta)
        for frequency in FREQUENCIES:
            case_dir = args.output_dir / variant_name(tan_delta) / f"frequency_{frequency:.6f}ghz"
            case_dir.mkdir(parents=True, exist_ok=True)
            summary_path = case_dir / "compression_summary.json"
            if summary_path.exists():
                returncode = 0
            else:
                returncode = subprocess.run(
                    command(case_dir, circuit_dir, frequency), check=False
                ).returncode
            rows.append(read_result(case_dir, tan_delta, frequency, returncode))
            (args.output_dir / "loss_results.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
    with (args.output_dir / "loss_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    slopes = slope_rows(rows)
    (args.output_dir / "loss_slopes.json").write_text(json.dumps(slopes, indent=2), encoding="utf-8")
    with (args.output_dir / "loss_slopes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slopes[0]))
        writer.writeheader()
        writer.writerows(slopes)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
