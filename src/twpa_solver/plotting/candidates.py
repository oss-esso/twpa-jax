"""Candidate selection for fitted gain-map metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twpa_solver.plotting.data import MapData, spectrum_for_point
from twpa_solver.plotting.metrics import compute_fit_metrics

PASS_STATUSES = {"PASS", "VALID_SOLVED", "VALID_CONVERGED"}


class PlotConfig:
    """Duck-typed plotting config used by the CLI and tests."""

    operation_drop_db: float
    n_dense: int
    window_frac: float
    polyorder: int


def _is_converged_status(status: Any) -> bool:
    text = str(status).upper()
    return text in PASS_STATUSES or text.startswith("PASS")


def compute_all_fit_metrics(data: MapData, config: PlotConfig) -> pd.DataFrame:
    """Compute fitted metrics for every point in a saved map."""
    rows: list[dict[str, Any]] = []
    diagnostic_columns = ("pump_status", "gain_status", "pump_failure_reason", "gain_failure_reason")
    for _, point in data.points.iterrows():
        metadata = {
            "point_index": int(point["point_index"]),
            "pump_power_dbm": float(point["pump_power_dbm"]),
            "pump_freq_ghz": float(point["pump_freq_ghz"]),
            "status": str(point.get("status", "UNKNOWN")),
        }
        for column in diagnostic_columns:
            if column in point:
                metadata[column] = point.get(column)
        if not _is_converged_status(metadata["status"]):
            row = dict(metadata)
            for name in (
                "peak_gain_db_fit",
                "peak_signal_freq_ghz_fit",
                "band_left_ghz_fit",
                "band_right_ghz_fit",
                "bandwidth_ghz_fit",
                "gbp_ghz_fit",
                "gbp_dbghz_fit",
                "ripple_db_fit",
                "smoothness_rms_curvature_fit",
                "smoothness_norm_fit",
                "mean_gain_db_fit",
                "median_gain_db_fit",
                "min_gain_db_fit",
                "score_fit",
            ):
                row[name] = np.nan
            row["valid_fit"] = False
            rows.append(row)
            continue

        try:
            freq, gain = spectrum_for_point(data, metadata["point_index"])
            fit = compute_fit_metrics(
                freq,
                gain,
                metadata,
                drop_db=float(config.operation_drop_db),
                n_dense=int(config.n_dense),
                window_frac=float(config.window_frac),
                polyorder=int(config.polyorder),
            )
            row = fit.metrics.to_dict()
            for column in diagnostic_columns:
                if column in metadata:
                    row[column] = metadata[column]
            row["valid_fit"] = bool(np.isfinite(row["score_fit"]))
        except (KeyError, ValueError) as exc:
            row = dict(metadata)
            row["status"] = f"INVALID_GAIN:{exc}"
            for name in (
                "peak_gain_db_fit",
                "peak_signal_freq_ghz_fit",
                "band_left_ghz_fit",
                "band_right_ghz_fit",
                "bandwidth_ghz_fit",
                "gbp_ghz_fit",
                "gbp_dbghz_fit",
                "ripple_db_fit",
                "smoothness_rms_curvature_fit",
                "smoothness_norm_fit",
                "mean_gain_db_fit",
                "median_gain_db_fit",
                "min_gain_db_fit",
                "score_fit",
            ):
                row[name] = np.nan
            row["valid_fit"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def gain_ranked_candidates(
    points: pd.DataFrame,
    *,
    top_k: int = 5,
    min_gain_db: float = 10.0,
) -> pd.DataFrame:
    """Top-K PASS cells by trailing ``gain_db`` (no spectrum fit needed).

    Single source of truth for the candidate cells whose ``pump_solution.npz``
    the plotter re-sweeps for S21 (``plot_gain_map.fit_gain_candidates``). Cells
    at or above ``min_gain_db`` are preferred; if none clear the bar, the plain
    gain ranking is used. Consumed by the S21 re-sweep and by the pump-solution
    prune (``scripts/prune_map_solutions.py``) so both agree on what to keep.
    """
    df = points.copy()
    df["gain_db_num"] = pd.to_numeric(df.get("gain_db"), errors="coerce")
    passed = df["status"].map(lambda s: str(s).upper().startswith("PASS"))
    ranked = df[passed & np.isfinite(df["gain_db_num"])].sort_values(
        "gain_db_num", ascending=False
    )
    strong = ranked[ranked["gain_db_num"] >= float(min_gain_db)]
    chosen = strong if not strong.empty else ranked
    return chosen.head(int(top_k))


def _best_row(df: pd.DataFrame, column: str, *, largest: bool) -> pd.Series | None:
    usable = df[df["valid_fit"] & np.isfinite(df[column])]
    if usable.empty:
        return None
    idx = usable[column].idxmax() if largest else usable[column].idxmin()
    return usable.loc[idx]


def select_candidates(
    metrics_df: pd.DataFrame,
    *,
    min_gain_db: float = 10.0,
    top_k: int = 5,
) -> dict[str, pd.Series]:
    """Select canonical candidates from a fitted metrics table."""
    candidates: dict[str, pd.Series] = {}
    best_peak = _best_row(metrics_df, "peak_gain_db_fit", largest=True)
    best_gbp = _best_row(metrics_df, "gbp_ghz_fit", largest=True)
    best_score = _best_row(metrics_df, "score_fit", largest=True)
    if best_peak is not None:
        candidates["best_peak_gain"] = best_peak
    if best_gbp is not None:
        candidates["best_gbp"] = best_gbp
    constrained = metrics_df[
        metrics_df["valid_fit"]
        & (metrics_df["peak_gain_db_fit"] >= float(min_gain_db))
    ]
    best_ripple = _best_row(constrained, "ripple_db_fit", largest=False)
    best_smooth = _best_row(constrained, "smoothness_norm_fit", largest=False)
    if best_ripple is not None:
        candidates["best_ripple"] = best_ripple
    if best_smooth is not None:
        candidates["best_smoothness"] = best_smooth
    if best_score is not None:
        candidates["best_score"] = best_score

    ranked = metrics_df[metrics_df["valid_fit"]].sort_values(
        "score_fit",
        ascending=False,
    )
    for rank, (_, row) in enumerate(ranked.head(int(top_k)).iterrows(), start=1):
        candidates[f"rank_{rank:03d}"] = row
    return candidates


def write_candidate_tables(
    candidates: dict[str, pd.Series],
    metrics_df: pd.DataFrame,
    outdir: Path | str,
) -> pd.DataFrame:
    """Write fitted metric and selected candidate tables."""
    root = Path(outdir)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(table_dir / "point_fit_metrics.csv", index=False)
    metrics_df.to_json(table_dir / "point_fit_metrics.json", orient="records", indent=2)

    rows: list[dict[str, Any]] = []
    for label, row in candidates.items():
        record = row.to_dict()
        record["candidate"] = label
        rows.append(record)
    selected = pd.DataFrame(rows)
    selected.to_csv(table_dir / "selected_candidates.csv", index=False)
    selected.to_json(table_dir / "selected_candidates.json", orient="records", indent=2)
    return selected
