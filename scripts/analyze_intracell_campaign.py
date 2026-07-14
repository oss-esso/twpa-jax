"""Compare intra-cell continuation maps and locate their first pump errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", default="m31_fixed")
    return parser.parse_args()


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    points = pd.read_csv(run_dir / "map_points.csv")
    points = points[points["pass"] == "warm"].copy()
    summary = json.loads((run_dir / "map_summary.json").read_text(encoding="utf-8"))
    return points, summary


def first_error_rows(run_id: str, points: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for freq, column in points.groupby("pump_freq_ghz", sort=True):
        column = column.sort_values("i_power")
        attempted = column[column["status"] != "SKIP_PAST_FOLD"]
        errors = attempted[attempted["pump_status"] != "VALID_CONVERGED"]
        if errors.empty:
            continue
        first = errors.iloc[0]
        later = column[column["i_power"] > first["i_power"]]
        earlier_pass = column[
            (column["i_power"] < first["i_power"]) & (column["status"] == "PASS")
        ]
        previous = earlier_pass.iloc[-1] if not earlier_pass.empty else None
        rows.append(
            {
                "run_id": run_id,
                "pump_freq_ghz": freq,
                "first_error_i_power": int(first["i_power"]),
                "first_error_power_dbm": first["pump_power_dbm"],
                "previous_pass_power_dbm": (
                    previous["pump_power_dbm"] if previous is not None else np.nan
                ),
                "previous_pass_gain_db": (
                    previous["gain_db"] if previous is not None else np.nan
                ),
                "failure_reason": first.get("pump_failure_reason"),
                "coeff_rel": first.get("pump_coeff_rel"),
                "newton_total": first.get("pump_newton_total"),
                "gmres_total": first.get("pump_gmres_total"),
                "solve_wall_s": first.get("pump_solve_wall_runtime_s"),
                "later_pass_count": int((later["status"] == "PASS").sum()),
                "later_error_count": int((later["status"] == "ERROR").sum()),
                "later_skip_count": int((later["status"] == "SKIP_PAST_FOLD").sum()),
                "false_skip_evidence": bool((later["status"] == "PASS").any()),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    runs: dict[str, tuple[pd.DataFrame, dict]] = {}
    for path in sorted(args.root.iterdir() if args.root.exists() else []):
        if (path / "map_points.csv").exists() and (path / "map_summary.json").exists():
            runs[path.name] = load_run(path)
    if not runs:
        raise SystemExit(f"no completed maps under {args.root}")

    reference = runs.get(args.reference)
    reference_points = None if reference is None else reference[0]
    summary_rows: list[dict] = []
    errors: list[dict] = []
    for run_id, (points, summary) in runs.items():
        counts = points["status"].value_counts()
        common_count = 0
        status_mismatches = 0
        max_gain_delta = np.nan
        if reference_points is not None:
            merged = points.merge(
                reference_points,
                on=["i_power", "j_freq"],
                suffixes=("", "_reference"),
            )
            common = merged[
                (merged["status"] == "PASS")
                & (merged["status_reference"] == "PASS")
            ]
            common_count = len(common)
            status_mismatches = int(
                (merged["status"] != merged["status_reference"]).sum()
            )
            if common_count:
                max_gain_delta = float(
                    np.nanmax(np.abs(common["gain_db"] - common["gain_db_reference"]))
                )
        summary_rows.append(
            {
                "run_id": run_id,
                "pass": int(counts.get("PASS", 0)),
                "error": int(counts.get("ERROR", 0)),
                "skip": int(counts.get("SKIP_PAST_FOLD", 0)),
                "elapsed_s": summary.get("elapsed_s"),
                "pump_runtime_s": summary.get("warm_pump_runtime_s"),
                "status_mismatches_vs_reference": status_mismatches,
                "common_pass_count": common_count,
                "max_gain_delta_db_vs_reference": max_gain_delta,
            }
        )
        errors.extend(first_error_rows(run_id, points))

    summary_df = pd.DataFrame(summary_rows).sort_values("run_id")
    errors_df = pd.DataFrame(errors)
    summary_df.to_csv(args.root / "campaign_summary.csv", index=False)
    errors_df.to_csv(args.root / "first_pump_errors.csv", index=False)

    lines = [
        "# Intra-cell continuation comparison",
        "",
        f"Reference: `{args.reference}`",
        "",
        summary_df.to_markdown(index=False),
        "",
        "`false_skip_evidence` means a higher-power cell in the same column passed "
        "after the first attempted pump error.",
    ]
    (args.root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"wrote {args.root / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
