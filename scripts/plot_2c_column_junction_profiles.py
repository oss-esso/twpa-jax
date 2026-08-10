"""Plot the strongest validated HB pump state in every frequency column.

The map is read from its authoritative ``map_points.csv``.  For each column,
the highest-power row with ``status=PASS`` is selected and independently
reconstructed by :mod:`plot_junction_current_profile`.  The resulting plots
show peak junction current, utilization ``|I|/Ic``, and biased junction phase
along the complete 2c array.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = Path(__file__).resolve().with_name("plot_junction_current_profile.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def _resolve_repo_path(value: str, run_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = ROOT / path
    if direct.exists():
        return direct
    return run_dir / path


def _read_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "map_points.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_highest_validated_hb(rows: list[dict[str, str]], run_dir: Path) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "PASS":
            continue
        grouped.setdefault(row["pump_freq_ghz"], []).append(row)

    selected: list[dict[str, str]] = []
    for _, candidates in grouped.items():
        candidates.sort(
            key=lambda row: (
                float(row.get("pump_power_dbm", "-inf")),
                int(row.get("point_index", "-1")),
            )
        )
        row = candidates[-1]
        pump_dir = _resolve_repo_path(row["pump_dir"], run_dir)
        if not (pump_dir / "pump_solution.npz").exists():
            raise FileNotFoundError(f"selected pump solution is missing: {pump_dir}")
        if not (pump_dir / "pump_report.json").exists():
            raise FileNotFoundError(f"selected pump report is missing: {pump_dir}")
        selected.append(row)
    selected.sort(key=lambda row: float(row["pump_freq_ghz"]))
    return selected


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    circuit_dir = args.circuit_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    selected = _select_highest_validated_hb(_read_rows(run_dir), run_dir)
    if not selected:
        raise RuntimeError("no PASS rows with validated pump states were found")

    index_rows: list[dict[str, object]] = []
    for column_index, row in enumerate(selected):
        frequency = float(row["pump_freq_ghz"])
        power = float(row["pump_power_dbm"])
        column_dir = outdir / f"f_{column_index:03d}_{frequency:.6f}ghz"
        pump_dir = _resolve_repo_path(row["pump_dir"], run_dir)
        command = [
            sys.executable,
            str(PROFILE_SCRIPT),
            "--pump-dir", str(pump_dir),
            "--circuit-dir", str(circuit_dir),
            "--outdir", str(column_dir),
        ]
        print(
            f"[{column_index + 1}/{len(selected)}] "
            f"fp={frequency:.9f} GHz, Pp={power:.3f} dBm"
        )
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"junction profile failed for fp={frequency:.9f} GHz "
                f"(return code {completed.returncode})"
            )
        summary_path = column_dir / "junction_current_profile.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        index_rows.append({
            "column_index": column_index,
            "point_index": int(row["point_index"]),
            "pump_freq_ghz": frequency,
            "pump_power_dbm": power,
            "pump_dir": str(pump_dir),
            "profile_dir": str(column_dir),
            "max_peak_ratio_ic": summary["max_peak_ratio_ic"],
            "max_peak_abs_phase_rad": summary["max_peak_abs_phase_rad"],
            "csv": summary["csv"],
            "plot": summary["plot"],
        })

    index_csv = outdir / "column_profile_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    index_json = outdir / "column_profile_index.json"
    index_json.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(index_rows)} column profiles to {outdir}")
    print(f"Index: {index_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
