"""Compare Python old-Julia parity maps against old Julia reference maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    old_root = Path(args.old_root)
    python_root = Path(args.python_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)

    old_rows = _read_csv(old_root / "report_old_ipm_power_map_rows.csv")
    python_rows = _read_csv(python_root / "rows.csv")
    comparison_rows, summary = compare_rows(old_rows, python_rows)
    _write_rows(outdir / "parity_rows.csv", comparison_rows)
    _write_grids(outdir, comparison_rows)
    _write_summary(outdir / "parity_summary.md", summary)
    _write_plots(outdir / "plots", comparison_rows)


def compare_rows(
    old_rows: list[dict[str, str]],
    python_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_lookup = {_key(row): row for row in old_rows}
    python_lookup = {_key(row): row for row in python_rows}
    keys = sorted(set(old_lookup) | set(python_lookup))
    rows: list[dict[str, Any]] = []
    for key in keys:
        old = old_lookup.get(key)
        py = python_lookup.get(key)
        old_gain = _float_or_nan(old, "gain_db_max")
        py_gain = _float_or_nan(py, "signal_gain_db")
        old_valid = _old_valid(old)
        py_valid = _python_valid(py)
        rows.append(
            {
                "pump_frequency_ghz": key[0],
                "external_power_dbm": key[1],
                "old_present": old is not None,
                "python_present": py is not None,
                "old_gain_db": old_gain,
                "python_gain_db": py_gain,
                "gain_difference_db": py_gain - old_gain
                if np.isfinite(old_gain) and np.isfinite(py_gain)
                else np.nan,
                "old_valid": old_valid,
                "python_valid": py_valid,
                "valid_mask_difference": int(old_valid) - int(py_valid),
                "old_status": "" if old is None else old.get("status", ""),
                "python_status": "" if py is None else py.get("status", ""),
                "old_source_power_dbm": _float_or_nan(old, "source_power_dbm"),
                "python_source_power_dbm": _float_or_nan(py, "source_power_dbm"),
                "source_power_mismatch_db": _float_or_nan(py, "source_power_dbm")
                - _float_or_nan(old, "source_power_dbm"),
                "old_pump_current_a": _float_or_nan(old, "pump_current_ua") * 1e-6,
                "python_pump_current_a": _float_or_nan(py, "pump_current_a"),
                "pump_current_mismatch_a": _float_or_nan(py, "pump_current_a")
                - _float_or_nan(old, "pump_current_ua") * 1e-6,
            }
        )
    summary = _summary(rows)
    return rows, summary


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aligned = [row for row in rows if row["old_present"] and row["python_present"]]
    mutually_valid = [row for row in aligned if row["old_valid"] and row["python_valid"]]
    diffs = [
        abs(row["gain_difference_db"])
        for row in mutually_valid
        if np.isfinite(row["gain_difference_db"])
    ]
    mask_disagreements = [
        row for row in aligned if bool(row["old_valid"]) != bool(row["python_valid"])
    ]
    old_valid = [row for row in aligned if row["old_valid"] and np.isfinite(row["old_gain_db"])]
    py_valid = [
        row
        for row in aligned
        if row["python_valid"] and np.isfinite(row["python_gain_db"])
    ]
    return {
        "aligned_cells": len(aligned),
        "missing_old_cells": sum(not row["old_present"] for row in rows),
        "missing_python_cells": sum(not row["python_present"] for row in rows),
        "mean_abs_gain_difference_valid_db": float(np.mean(diffs)) if diffs else np.nan,
        "max_abs_gain_difference_valid_db": float(np.max(diffs)) if diffs else np.nan,
        "valid_mask_agreement_count": len(aligned) - len(mask_disagreements),
        "valid_mask_disagreement_count": len(mask_disagreements),
        "best_old_valid_point": _best_point(old_valid, "old_gain_db"),
        "best_python_valid_point": _best_point(py_valid, "python_gain_db"),
        "source_power_mismatch_max_db": _nanmax_abs(
            [row["source_power_mismatch_db"] for row in aligned]
        ),
        "pump_current_mismatch_max_a": _nanmax_abs(
            [row["pump_current_mismatch_a"] for row in aligned]
        ),
    }


def _best_point(rows: list[dict[str, Any]], field: str) -> str:
    if not rows:
        return ""
    best = max(rows, key=lambda row: row[field])
    return (
        f"fp={best['pump_frequency_ghz']}GHz, "
        f"Pext={best['external_power_dbm']}dBm, {field}={best[field]}"
    )


def _nanmax_abs(values: list[float]) -> float:
    finite = [abs(value) for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else np.nan


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Python vs Old Julia Parity Comparison", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "Interpretation: matching source power/current does not imply matching gain.",
            "Gain differences must be read with the geometry/coupler mismatch documented in the docs.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_grids(outdir: Path, rows: list[dict[str, Any]]) -> None:
    for field, filename in [
        ("gain_difference_db", "gain_difference_grid.csv"),
        ("valid_mask_difference", "valid_mask_difference_grid.csv"),
    ]:
        np.savetxt(outdir / filename, _grid(rows, field), delimiter=",")


def _write_plots(plot_dir: Path, rows: list[dict[str, Any]]) -> None:
    for field, filename, title in [
        ("old_gain_db", "old_julia_gain.png", "Old Julia gain max (dB)"),
        ("python_gain_db", "python_parity_gain.png", "Python parity signal gain (dB)"),
        ("gain_difference_db", "gain_difference.png", "Python - old gain (dB)"),
        ("old_valid", "old_julia_valid_mask.png", "Old Julia valid mask"),
        ("python_valid", "python_valid_mask.png", "Python valid mask"),
        ("valid_mask_difference", "mask_difference.png", "Valid mask difference"),
    ]:
        fig, ax = plt.subplots(figsize=(6.0, 4.8), constrained_layout=True)
        image = ax.imshow(_grid(rows, field).T, origin="lower", aspect="auto")
        ax.set_xlabel("pump frequency index")
        ax.set_ylabel("external power index")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)


def _grid(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    freqs = sorted({row["pump_frequency_ghz"] for row in rows})
    powers = sorted({row["external_power_dbm"] for row in rows})
    lookup = {(row["pump_frequency_ghz"], row["external_power_dbm"]): row for row in rows}
    grid = np.full((len(freqs), len(powers)), np.nan)
    for i, freq in enumerate(freqs):
        for j, power in enumerate(powers):
            value = lookup[(freq, power)][field]
            if isinstance(value, bool):
                value = int(value)
            grid[i, j] = float(value)
    return grid


def _key(row: dict[str, str]) -> tuple[float, float]:
    return (
        round(float(row["pump_frequency_ghz"]), 9),
        round(float(row.get("external_power_dbm", row.get("pump_power_dbm", "nan"))), 9),
    )


def _old_valid(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    status = row.get("status", "").lower()
    return status in {"valid_converged", "converged"} or (
        row.get("solver_converged", "").lower() == "true"
        and row.get("finite_gain", "").lower() == "true"
    )


def _python_valid(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    return row.get("success", "").lower() == "true" or row.get("status", "").lower() == "converged"


def _float_or_nan(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return np.nan
    value = row.get(key, "")
    if value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--python-root", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
