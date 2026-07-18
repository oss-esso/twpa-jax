"""Phase A1 (docs/multistability_and_cli_trim_plan.md): read-only scan of a
finished map's map_points.csv for cells where a cold reseed (warm_retry_reseed
=True, run_gain_map.py:562,1661,2068) coincides with a discontinuous jump in
gain_db relative to the column's local smooth trend -- a signature consistent
with the pump solve landing on a different solution branch than its warm
neighbors.

Per docs/convergence_investigation_log.md terminology rule: this only reports
"discontinuity," not "fold" or "branch switch" -- confirming multistability is
Phase A2 (direct seed-vs-warm comparison), not this scanner.

Usage:
    python scripts/debug_branch_discontinuity_scan.py
    python scripts/debug_branch_discontinuity_scan.py --map-dir outputs/foo --sigma 4
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAP_DIRS = [
    ROOT / "outputs" / "measurement_match_debug_01" / "column_debug_col3_trim",
    ROOT / "outputs" / "campaign_continuation_methods" / "c04_baseline_prod",
]


def _parse_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _parse_bool(value: str) -> bool:
    return value == "True"


def load_map_points(csv_path: Path) -> list[dict[str, Any]]:
    with open(csv_path, newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    for r in raw_rows:
        rows.append({
            "i_power": int(r["i_power"]),
            "j_freq": int(r["j_freq"]),
            "pump_power_dbm": _parse_float(r["pump_power_dbm"]),
            "pump_freq_ghz": _parse_float(r["pump_freq_ghz"]),
            "gain_db": _parse_float(r.get("gain_db", "")),
            "pump_status": r.get("pump_status", ""),
            "warm_started": _parse_bool(r.get("warm_started", "")),
            "warm_retry_reseed": _parse_bool(r.get("warm_retry_reseed", "")),
        })
    return rows


def group_by_column(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    columns: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        columns.setdefault(row["j_freq"], []).append(row)
    for col_rows in columns.values():
        col_rows.sort(key=lambda r: r["i_power"])
    return columns


def scan_column(col_rows: list[dict[str, Any]], sigma_n: float) -> list[dict[str, Any]]:
    """Flag reseeded cells whose gain_db second-difference exceeds
    sigma_n * column_std of all finite second-differences in the column."""
    gains = [r["gain_db"] for r in col_rows]
    second_diffs: list[float] = []
    for i in range(1, len(col_rows) - 1):
        g0, g1, g2 = gains[i - 1], gains[i], gains[i + 1]
        if math.isfinite(g0) and math.isfinite(g1) and math.isfinite(g2):
            second_diffs.append(g0 - 2 * g1 + g2)

    if len(second_diffs) < 3:
        return []

    mean = sum(second_diffs) / len(second_diffs)
    variance = sum((d - mean) ** 2 for d in second_diffs) / len(second_diffs)
    column_std = math.sqrt(variance)
    threshold = sigma_n * column_std

    flagged = []
    for i in range(1, len(col_rows) - 1):
        row = col_rows[i]
        if not row["warm_retry_reseed"]:
            continue
        g0, g1, g2 = gains[i - 1], gains[i], gains[i + 1]
        if not (math.isfinite(g0) and math.isfinite(g1) and math.isfinite(g2)):
            continue
        d2 = g0 - 2 * g1 + g2
        if column_std > 0 and abs(d2) > threshold:
            flagged.append({
                **row,
                "second_diff": d2,
                "column_std": column_std,
                "threshold": threshold,
            })
    return flagged


def scan_map_dir(map_dir: Path, sigma_n: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    csv_path = map_dir / "map_points.csv"
    rows = load_map_points(csv_path)
    columns = group_by_column(rows)
    flagged: list[dict[str, Any]] = []
    for col_rows in columns.values():
        flagged.extend(scan_column(col_rows, sigma_n))
    return rows, flagged


def write_flagged_csv(out_path: Path, all_flagged: list[dict[str, Any]]) -> None:
    cols = [
        "map_dir", "i_power", "j_freq", "pump_power_dbm", "pump_freq_ghz",
        "gain_db", "pump_status", "warm_started", "warm_retry_reseed",
        "second_diff", "column_std", "threshold",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in all_flagged:
            writer.writerow({c: row.get(c) for c in cols})
    print(f"wrote {out_path} ({len(all_flagged)} flagged rows)")


def plot_map_dir(map_dir: Path, rows: list[dict[str, Any]], flagged: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flagged_keys = {(r["i_power"], r["j_freq"]) for r in flagged}
    reseed_keys = {(r["i_power"], r["j_freq"]) for r in rows if r["warm_retry_reseed"]}

    fig, ax = plt.subplots(figsize=(9, 6))
    xs_all = [r["pump_freq_ghz"] for r in rows]
    ys_all = [r["pump_power_dbm"] for r in rows]
    ax.scatter(xs_all, ys_all, c="lightgray", s=8, zorder=1, label="all cells")

    reseed_only = [r for r in rows if (r["i_power"], r["j_freq"]) in reseed_keys
                   and (r["i_power"], r["j_freq"]) not in flagged_keys]
    if reseed_only:
        ax.scatter(
            [r["pump_freq_ghz"] for r in reseed_only],
            [r["pump_power_dbm"] for r in reseed_only],
            c="tab:orange", s=20, zorder=2, label="reseed (not flagged)",
        )
    if flagged:
        ax.scatter(
            [r["pump_freq_ghz"] for r in flagged],
            [r["pump_power_dbm"] for r in flagged],
            c="tab:red", s=40, marker="x", zorder=3, label="reseed + discontinuity",
        )

    ax.set_xlabel("pump_freq_ghz")
    ax.set_ylabel("pump_power_dbm")
    ax.set_title(f"branch discontinuity scan: {map_dir.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-dir", action="append", type=Path, default=None,
        help="Directory containing map_points.csv; repeatable. "
             "Default: measurement_match_debug_01/column_debug_col3_trim + "
             "campaign_continuation_methods/c04_baseline_prod.",
    )
    parser.add_argument(
        "--sigma", type=float, default=3.0,
        help="N-sigma threshold on the column's local gain_db second-difference "
             "std for flagging a reseeded cell as discontinuous.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "branch_discontinuity_scan",
        help="Output directory for the flagged CSV and per-map plots.",
    )
    args = parser.parse_args()

    map_dirs = args.map_dir if args.map_dir else DEFAULT_MAP_DIRS

    all_flagged: list[dict[str, Any]] = []
    for map_dir in map_dirs:
        csv_path = map_dir / "map_points.csv"
        if not csv_path.exists():
            print(f"SKIP {map_dir}: no map_points.csv")
            continue
        rows, flagged = scan_map_dir(map_dir, args.sigma)
        n_reseed = sum(1 for r in rows if r["warm_retry_reseed"])
        print(f"{map_dir}: {len(rows)} rows, {n_reseed} reseeded, {len(flagged)} flagged")
        for row in flagged:
            row = dict(row)
            row["map_dir"] = str(map_dir)
            all_flagged.append(row)
        plot_map_dir(map_dir, rows, flagged, args.out_dir / f"{map_dir.name}_discontinuity.png")

    write_flagged_csv(args.out_dir / "flagged_discontinuities.csv", all_flagged)


if __name__ == "__main__":
    main()
