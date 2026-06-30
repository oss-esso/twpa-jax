"""Run an exp08/exp09 IPM gain map over pump power and pump frequency.

The map axes are available/delivered pump power in dBm and pump frequency in GHz.
Power is converted to the peak sinusoidal current expected by exp08:

    P_W = 1e-3 * 10**(dBm / 10)
    I_peak = sqrt(2 * P_W / Z0)

Each point runs:
  1. experiments/exp08_full_ipm_pump_solve.py
  2. experiments/exp09_full_ipm_gain_from_pump.py

The script is deliberately thin: it orchestrates the current experiment scripts
and aggregates their JSON/CSV outputs without changing solver internals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def dbm_to_peak_current_a(power_dbm: float, z0_ohm: float) -> float:
    power_w = 1e-3 * 10.0 ** (float(power_dbm) / 10.0)
    return math.sqrt(2.0 * power_w / float(z0_ohm))


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> tuple[int, float, str]:
    start = time.perf_counter()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=out,
                stderr=err,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        return completed.returncode, time.perf_counter() - start, ""
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        with stderr_path.open("a", encoding="utf-8") as err:
            err.write(f"\nTIMEOUT after {timeout_s:.3f} s\n")
        return 124, elapsed, f"timeout after {timeout_s:.3f} s"


def point_name(index: int, power_dbm: float, pump_freq_ghz: float) -> str:
    p = f"{power_dbm:.6g}".replace("-", "m").replace(".", "p")
    f = f"{pump_freq_ghz:.6g}".replace(".", "p")
    return f"point_{index:04d}_p_{p}dbm_fp_{f}ghz"


def extract_pump_status(report: dict[str, Any] | None, returncode: int) -> str:
    if report is None:
        return "MISSING" if returncode == 0 else "ERROR"
    status = str(report.get("final_status", ""))
    if returncode != 0:
        return "ERROR"
    return status or "UNKNOWN"


def extract_gain_status(report: dict[str, Any] | None, returncode: int) -> str:
    if report is None:
        return "MISSING" if returncode == 0 else "ERROR"
    rows = report.get("results", [])
    if returncode != 0:
        return "ERROR"
    if rows and all(r.get("status") == "VALID_SOLVED" for r in rows):
        return "VALID_SOLVED"
    return "UNKNOWN"


def extract_gain_metrics(report: dict[str, Any] | None) -> dict[str, float | None]:
    if report is None:
        return {
            "gain_db": None,
            "gain_vs_off_db": None,
            "gain_vs_pumpdiag_db": None,
            "signal_ghz": None,
            "linear_rel_residual": None,
            "gain_factor_solve_runtime_s": None,
            "gain_total_runtime_s": None,
        }
    results = report.get("results", [])
    valid = [r for r in results if finite_or_none(r.get("gain_db")) is not None]
    best = max(valid, key=lambda r: float(r["gain_db"])) if valid else {}
    return {
        "gain_db": finite_or_none(best.get("gain_db")),
        "gain_vs_off_db": finite_or_none(best.get("gain_vs_off_db")),
        "gain_vs_pumpdiag_db": finite_or_none(best.get("gain_vs_pumpdiag_db")),
        "signal_ghz": finite_or_none(best.get("signal_ghz")),
        "linear_rel_residual": finite_or_none(best.get("linear_rel_residual")),
        "gain_factor_solve_runtime_s": sum(
            finite_or_none(r.get("factor_solve_runtime_s")) or 0.0 for r in results
        )
        if results
        else None,
        "gain_total_runtime_s": finite_or_none(report.get("metadata", {}).get("total_runtime_s")),
    }


def extract_pump_metrics(report: dict[str, Any] | None) -> dict[str, float | None]:
    if report is None:
        return {
            "pump_runtime_s": None,
            "pump_factor_runtime_s": None,
            "pump_current_over_ic_median": None,
            "pump_coeff_rel": None,
            "pump_time_rel": None,
        }
    reports = report.get("reports", [])
    final = reports[-1] if reports else {}
    return {
        "pump_runtime_s": sum(finite_or_none(r.get("runtime_s")) or 0.0 for r in reports)
        if reports
        else None,
        "pump_factor_runtime_s": sum(
            finite_or_none(r.get("factor_runtime_s")) or 0.0 for r in reports
        )
        if reports
        else None,
        "pump_current_over_ic_median": finite_or_none(
            report.get("metadata", {}).get("pump_current_ratio_ic_median")
        ),
        "pump_coeff_rel": finite_or_none(final.get("coeff_rel")),
        "pump_time_rel": finite_or_none(final.get("time_rel")),
    }


def write_points_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "point_index",
        "pump_power_dbm",
        "pump_freq_ghz",
        "pump_current_peak_a",
        "status",
        "pump_status",
        "gain_status",
        "gain_db",
        "gain_vs_off_db",
        "gain_vs_pumpdiag_db",
        "signal_ghz",
        "linear_rel_residual",
        "pump_runtime_s",
        "pump_factor_runtime_s",
        "gain_total_runtime_s",
        "gain_factor_solve_runtime_s",
        "elapsed_s",
        "pump_current_over_ic_median",
        "pump_coeff_rel",
        "pump_time_rel",
        "point_dir",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_arrays(path: Path, rows: list[dict[str, Any]], powers: np.ndarray, freqs: np.ndarray) -> None:
    gain = np.full((len(powers), len(freqs)), np.nan, dtype=float)
    status = np.empty((len(powers), len(freqs)), dtype=object)
    status[:, :] = ""
    index_by_power = {float(v): i for i, v in enumerate(powers)}
    index_by_freq = {float(v): j for j, v in enumerate(freqs)}
    for row in rows:
        i = index_by_power[float(row["pump_power_dbm"])]
        j = index_by_freq[float(row["pump_freq_ghz"])]
        value = finite_or_none(row.get("gain_db"))
        if value is not None:
            gain[i, j] = value
        status[i, j] = str(row.get("status", ""))
    np.savez(
        path,
        pump_power_dbm=powers,
        pump_freq_ghz=freqs,
        gain_db=gain,
        status=status,
    )


def write_summary(
    path_json: Path,
    path_md: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    powers: np.ndarray,
    freqs: np.ndarray,
    elapsed_s: float,
) -> None:
    valid = [r for r in rows if finite_or_none(r.get("gain_db")) is not None and r["status"] == "PASS"]
    best = max(valid, key=lambda r: float(r["gain_db"])) if valid else None
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1

    summary = {
        "output_dir": str(args.output_dir),
        "n_power": int(len(powers)),
        "n_frequency": int(len(freqs)),
        "n_points": int(len(rows)),
        "status_counts": status_counts,
        "all_pass": len(valid) == len(rows),
        "elapsed_s": elapsed_s,
        "pump_power_dbm_min": float(np.min(powers)),
        "pump_power_dbm_max": float(np.max(powers)),
        "pump_freq_ghz_min": float(np.min(freqs)),
        "pump_freq_ghz_max": float(np.max(freqs)),
        "z0_ohm": float(args.z0_ohm),
        "current_convention": "I_peak = sqrt(2 * P_W / Z0)",
        "signal_ghz": float(args.signal_ghz),
        "best": best,
    }
    with path_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# IPM exp08/exp09 Gain Map",
        "",
        f"- grid: `{len(powers)} x {len(freqs)}`",
        f"- pump power: `{float(np.min(powers))}` to `{float(np.max(powers))}` dBm",
        f"- pump frequency: `{float(np.min(freqs))}` to `{float(np.max(freqs))}` GHz",
        f"- signal frequency: `{float(args.signal_ghz)}` GHz",
        f"- current convention: `{summary['current_convention']}`",
        f"- elapsed: `{elapsed_s:.3f}` s",
        f"- status counts: `{status_counts}`",
    ]
    if best is not None:
        lines.extend(
            [
                "",
                "## Best Point",
                "",
                f"- gain: `{float(best['gain_db']):.6g}` dB",
                f"- pump power: `{float(best['pump_power_dbm']):.6g}` dBm",
                f"- pump frequency: `{float(best['pump_freq_ghz']):.6g}` GHz",
                f"- pump current peak: `{float(best['pump_current_peak_a']):.6g}` A",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `gain_map_points.csv`",
            "- `gain_map_arrays.npz`",
            "- `gain_map_summary.json`",
        ]
    )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ipm_exp08_exp09_gain_map_5x5"))
    parser.add_argument("--ipm-dir", type=Path, default=Path("outputs/ipm_python_design"))
    parser.add_argument("--n-power", type=int, default=5)
    parser.add_argument("--n-frequency", type=int, default=5)
    parser.add_argument("--pump-power-start-dbm", type=float, default=-30.0)
    parser.add_argument("--pump-power-stop-dbm", type=float, default=-20.0)
    parser.add_argument("--pump-frequency-start-ghz", type=float, default=6.0)
    parser.add_argument("--pump-frequency-stop-ghz", type=float, default=8.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--signal-ghz", type=float, default=6.0)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--source-port", type=int, default=1)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--continuation-steps", type=int, default=20)
    parser.add_argument("--sidebands", type=int, default=2)
    parser.add_argument("--gamma-nt", type=int, default=128)
    parser.add_argument("--pump-timeout-s", type=float, default=600.0)
    parser.add_argument("--gain-timeout-s", type=float, default=300.0)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--warm-start-by-frequency",
        action="store_true",
        help="Warm-start each higher-power point from the previous power at the same pump frequency.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    output_dir = args.output_dir

    if output_dir.exists() and args.overwrite and not args.resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    powers = np.linspace(args.pump_power_start_dbm, args.pump_power_stop_dbm, args.n_power)
    freqs = np.linspace(args.pump_frequency_start_ghz, args.pump_frequency_stop_ghz, args.n_frequency)

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    point_index = 0

    for pump_freq in freqs:
        previous_pump_dir: Path | None = None
        for power_dbm in powers:
            point_start = time.perf_counter()
            current_peak_a = dbm_to_peak_current_a(float(power_dbm), args.z0_ohm)
            pdir = output_dir / "points" / point_name(point_index, float(power_dbm), float(pump_freq))
            pump_dir = pdir / "pump"
            gain_dir = pdir / "gain"
            pdir.mkdir(parents=True, exist_ok=True)

            pump_report_path = pump_dir / "pump_report.json"
            gain_report_path = gain_dir / "gain_report.json"
            message = ""

            if args.resume and pump_report_path.exists() and gain_report_path.exists():
                pump_returncode = 0
                gain_returncode = 0
                pump_elapsed = 0.0
                gain_elapsed = 0.0
            else:
                pump_cmd = [
                    args.python_executable,
                    "experiments/exp08_full_ipm_pump_solve.py",
                    "--ipm-dir",
                    str(args.ipm_dir),
                    "--outdir",
                    str(pump_dir),
                    "--pump-port",
                    str(args.pump_port),
                    "--pump-freq-ghz",
                    f"{float(pump_freq):.12g}",
                    "--pump-current-a",
                    f"{current_peak_a:.17g}",
                    "--harmonics",
                    str(args.harmonics),
                    "--nt",
                    str(args.nt),
                    "--continuation-steps",
                    str(args.continuation_steps),
                    "--quiet",
                ]
                if args.warm_start_by_frequency and previous_pump_dir is not None:
                    pump_cmd.extend(["--promote-from-pump-dir", str(previous_pump_dir)])

                pump_returncode, pump_elapsed, pump_message = run_command(
                    pump_cmd,
                    cwd=repo,
                    stdout_path=pdir / "pump_stdout.txt",
                    stderr_path=pdir / "pump_stderr.txt",
                    timeout_s=args.pump_timeout_s,
                )
                if pump_message:
                    message = pump_message

                if pump_returncode == 0 and pump_report_path.exists():
                    gain_cmd = [
                        args.python_executable,
                        "experiments/exp09_full_ipm_gain_from_pump.py",
                        "--ipm-dir",
                        str(args.ipm_dir),
                        "--pump-dir",
                        str(pump_dir),
                        "--outdir",
                        str(gain_dir),
                        "--source-port",
                        str(args.source_port),
                        "--out-port",
                        str(args.out_port),
                        "--signal-ghz",
                        f"{float(args.signal_ghz):.12g}",
                        "--sidebands",
                        str(args.sidebands),
                        "--gamma-nt",
                        str(args.gamma_nt),
                    ]
                    gain_returncode, gain_elapsed, gain_message = run_command(
                        gain_cmd,
                        cwd=repo,
                        stdout_path=pdir / "gain_stdout.txt",
                        stderr_path=pdir / "gain_stderr.txt",
                        timeout_s=args.gain_timeout_s,
                    )
                    if gain_message:
                        message = f"{message}; {gain_message}".strip("; ")
                else:
                    gain_returncode = -1
                    gain_elapsed = 0.0

            pump_report = read_json(pump_report_path)
            gain_report = read_json(gain_report_path)
            pump_status = extract_pump_status(pump_report, pump_returncode)
            gain_status = extract_gain_status(gain_report, gain_returncode)
            status = "PASS" if pump_status == "VALID_CONVERGED" and gain_status == "VALID_SOLVED" else "ERROR"

            row: dict[str, Any] = {
                "point_index": point_index,
                "pump_power_dbm": float(power_dbm),
                "pump_freq_ghz": float(pump_freq),
                "pump_current_peak_a": current_peak_a,
                "status": status,
                "pump_status": pump_status,
                "gain_status": gain_status,
                "elapsed_s": time.perf_counter() - point_start,
                "point_dir": str(pdir),
                "message": message,
            }
            row.update(extract_pump_metrics(pump_report))
            row.update(extract_gain_metrics(gain_report))
            rows.append(row)

            print(
                f"[{point_index + 1}/{len(powers) * len(freqs)}] "
                f"P={float(power_dbm):.6g} dBm fp={float(pump_freq):.6g} GHz "
                f"Ipk={current_peak_a:.6g} A status={status} gain={row.get('gain_db')}"
            )

            write_points_csv(output_dir / "gain_map_points.csv", rows)
            write_arrays(output_dir / "gain_map_arrays.npz", rows, powers, freqs)
            write_summary(
                output_dir / "gain_map_summary.json",
                output_dir / "gain_map_summary.md",
                args=args,
                rows=rows,
                powers=powers,
                freqs=freqs,
                elapsed_s=time.perf_counter() - start,
            )

            if status == "PASS":
                previous_pump_dir = pump_dir
            if status != "PASS" and args.fail_fast:
                return 1

            point_index += 1

    write_summary(
        output_dir / "gain_map_summary.json",
        output_dir / "gain_map_summary.md",
        args=args,
        rows=rows,
        powers=powers,
        freqs=freqs,
        elapsed_s=time.perf_counter() - start,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

