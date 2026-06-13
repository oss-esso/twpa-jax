"""Compare backend-substitution 5x5 runs to the stored JosephsonCircuits reference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


BACKEND_DIRS = {
    "josephsoncircuits": "josephsoncircuits",
    "scipy-least-squares": "scipy_least_squares",
    "scipy-least-squares-real": "scipy_least_squares_real",
    "scipy-root": "scipy_root",
    "scipy-newton-krylov": "scipy_newton_krylov",
    "jax-dense-newton": "jax_dense_newton",
    "jax-newton-krylov": "jax_newton_krylov",
    "pseudo-transient": "pseudo_transient",
}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    reference_root = Path(args.reference_root)
    backend_root = Path(args.backend_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ref_rows = _read_rows(reference_root / "report_old_ipm_power_map_rows.csv")
    ref_lookup = _lookup(ref_rows)

    comparison_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for backend, dirname in BACKEND_DIRS.items():
        root = backend_root / dirname
        rows_path = root / "report_old_ipm_power_map_rows.csv"
        if not rows_path.exists():
            summaries.append(_missing_summary(backend, "FAILED_INTERFACE", f"missing {rows_path}"))
            continue
        rows = _read_rows(rows_path)
        lookup = _lookup(rows)
        gain_diffs: list[float] = []
        valid_gain_diffs: list[float] = []
        status_matches = 0
        missing = 0
        not_impl = 0
        failed_interface = 0
        gain_count = 0
        for key, ref in ref_lookup.items():
            row = lookup.get(key)
            if row is None:
                missing += 1
                continue
            if row.get("status") == ref.get("status"):
                status_matches += 1
            if row.get("status") == "BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN":
                not_impl += 1
            if row.get("status") == "FAILED_INTERFACE":
                failed_interface += 1
            ref_gain = _float_or_none(ref.get("gain_db_max"))
            gain = _float_or_none(row.get("gain_db_max"))
            if ref_gain is not None and gain is not None:
                gain_count += 1
                diff = abs(gain - ref_gain)
                gain_diffs.append(diff)
                if ref.get("status") == "VALID_CONVERGED" and row.get("status") == "VALID_CONVERGED":
                    valid_gain_diffs.append(diff)
            comparison_rows.append(
                {
                    "backend": backend,
                    "pump_frequency_ghz": key[0],
                    "external_power_dbm": key[1],
                    "reference_status": ref.get("status", ""),
                    "backend_status": row.get("status", ""),
                    "reference_gain_db_max": ref.get("gain_db_max", ""),
                    "backend_gain_db_max": row.get("gain_db_max", ""),
                    "abs_gain_diff_db": "" if ref_gain is None or gain is None else abs(gain - ref_gain),
                }
            )
        classification = _classify(
            backend=backend,
            total=len(ref_lookup),
            missing=missing,
            not_impl=not_impl,
            failed_interface=failed_interface,
            gain_count=gain_count,
            gain_diffs=gain_diffs,
            valid_gain_diffs=valid_gain_diffs,
            status_matches=status_matches,
        )
        summaries.append(
            {
                "backend": backend,
                "classification": classification,
                "rows": len(rows),
                "missing_reference_cells": missing,
                "status_matches": status_matches,
                "not_implemented_rows": not_impl,
                "failed_interface_rows": failed_interface,
                "gain_cells_compared": gain_count,
                "mean_abs_gain_diff_db": "" if not gain_diffs else mean(gain_diffs),
                "max_abs_gain_diff_db": "" if not gain_diffs else max(gain_diffs),
                "valid_converged_gain_cells_compared": len(valid_gain_diffs),
                "valid_converged_mean_abs_gain_diff_db": "" if not valid_gain_diffs else mean(valid_gain_diffs),
                "valid_converged_max_abs_gain_diff_db": "" if not valid_gain_diffs else max(valid_gain_diffs),
            }
        )

    _write_csv(outdir / "backend_comparison_rows.csv", comparison_rows)
    _write_csv(outdir / "backend_status_matrix.csv", summaries)
    _write_csv(
        outdir / "backend_gain_difference_summary.csv",
        [
            {
                "backend": s["backend"],
                "classification": s["classification"],
                "gain_cells_compared": s["gain_cells_compared"],
                "mean_abs_gain_diff_db": s["mean_abs_gain_diff_db"],
                "max_abs_gain_diff_db": s["max_abs_gain_diff_db"],
                "valid_converged_gain_cells_compared": s.get("valid_converged_gain_cells_compared", ""),
                "valid_converged_mean_abs_gain_diff_db": s.get("valid_converged_mean_abs_gain_diff_db", ""),
                "valid_converged_max_abs_gain_diff_db": s.get("valid_converged_max_abs_gain_diff_db", ""),
            }
            for s in summaries
        ],
    )
    _write_summary(outdir / "backend_comparison_summary.md", reference_root, backend_root, summaries)
    print(outdir)


def _classify(
    *,
    backend: str,
    total: int,
    missing: int,
    not_impl: int,
    failed_interface: int,
    gain_count: int,
    gain_diffs: list[float],
    valid_gain_diffs: list[float],
    status_matches: int,
) -> str:
    if missing == total or failed_interface > 0:
        return "FAILED_INTERFACE"
    if not_impl == total:
        return "NOT_IMPLEMENTED"
    if (
        backend == "josephsoncircuits"
        and gain_count == total
        and status_matches == total
        and max(valid_gain_diffs or [0.0]) < 1e-6
    ):
        return "REFERENCE_REPRODUCED"
    if gain_count > 0 and max(gain_diffs or [0.0]) < 1e-3:
        return "GAIN_MATCH"
    if gain_count == 0 and status_matches > 0:
        return "STATUS_ONLY"
    return "FAILED_NUMERICALLY"


def _missing_summary(backend: str, classification: str, message: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "classification": classification,
        "rows": 0,
        "missing_reference_cells": "",
        "status_matches": 0,
        "not_implemented_rows": 0,
        "failed_interface_rows": "",
        "gain_cells_compared": 0,
        "mean_abs_gain_diff_db": "",
        "max_abs_gain_diff_db": "",
        "valid_converged_gain_cells_compared": "",
        "valid_converged_mean_abs_gain_diff_db": "",
        "valid_converged_max_abs_gain_diff_db": "",
        "message": message,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _lookup(rows: list[dict[str, str]]) -> dict[tuple[float, float], dict[str, str]]:
    return {
        (round(float(row["pump_frequency_ghz"]), 9), round(float(row["external_power_dbm"]), 9)): row
        for row in rows
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, reference_root: Path, backend_root: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Backend 5x5 Comparison To JC Reference",
        "",
        f"- reference_root: `{reference_root}`",
        f"- backend_root: `{backend_root}`",
        "",
        "| backend | classification | rows | gain cells | trusted gain max diff dB | all-cell mean abs gain diff dB | notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        notes = s.get("message", "")
        lines.append(
            f"| `{s['backend']}` | `{s['classification']}` | {s['rows']} | "
            f"{s['gain_cells_compared']} | {s.get('valid_converged_max_abs_gain_diff_db', '')} | "
            f"{s['mean_abs_gain_diff_db']} | {notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--backend-root", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
