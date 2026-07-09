"""Standard fitted gain-map plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from twpa_solver.plotting.style import THESIS_FIGSIZE_MAP, save_figure

STATUS_LABELS = (
    "PASS",
    "PUMP_FAILED",
    "GAIN_FAILED",
    "INVALID_GAIN",
    "FOLD_SKIPPED",
    "TIMEOUT",
    "UNKNOWN",
)


def _edges(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 1:
        step = 0.5
        return np.asarray([vals[0] - step, vals[0] + step])
    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - (mids[0] - vals[0])
    last = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _metric_grid(metrics_df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = metrics_df.copy()
    valid.loc[~valid.get("valid_fit", False), column] = np.nan
    pivot = valid.pivot_table(
        index="pump_power_dbm",
        columns="pump_freq_ghz",
        values=column,
        aggfunc="first",
    ).sort_index().sort_index(axis=1)
    return (
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def _plot_metric_map(
    metrics_df: pd.DataFrame,
    column: str,
    outpath: Path | str,
    *,
    title: str,
    colorbar_label: str,
    candidates: dict[str, pd.Series] | None = None,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    x, y, z = _metric_grid(metrics_df, column)
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_MAP)
    mesh = ax.pcolormesh(_edges(x), _edges(y), z, shading="auto")
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    ax.set_xlabel("Pump frequency fp / GHz")
    ax.set_ylabel("Pump power Pp / dBm")
    ax.set_title(title)
    if candidates:
        _overlay_candidate_markers(ax, candidates)
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)


def _overlay_candidate_markers(
    ax: plt.Axes,
    candidates: dict[str, pd.Series],
) -> None:
    marker_by_key = {
        "best_score": ("*", 180),
        "best_peak_gain": ("o", 70),
        "best_gbp": ("s", 70),
        "best_ripple": ("^", 80),
        "best_smoothness": ("D", 65),
    }
    for label, row in candidates.items():
        marker, size = marker_by_key.get(label, ("x", 55))
        if marker == "x":
            ax.scatter(
                float(row["pump_freq_ghz"]),
                float(row["pump_power_dbm"]),
                marker=marker,
                s=size,
                c="black",
                linewidths=1.0,
                label=label,
            )
        else:
            ax.scatter(
                float(row["pump_freq_ghz"]),
                float(row["pump_power_dbm"]),
                marker=marker,
                s=size,
                c="none" if marker != "*" else "white",
                edgecolors="black",
                linewidths=1.0,
                label=label,
            )


def _group_candidate_rows(
    candidates: dict[str, pd.Series],
) -> list[tuple[list[str], pd.Series]]:
    grouped: dict[int, tuple[list[str], pd.Series]] = {}
    for label, row in candidates.items():
        point_index = int(row["point_index"])
        if point_index not in grouped:
            grouped[point_index] = ([label], row)
        else:
            grouped[point_index][0].append(label)
    return list(grouped.values())


def plot_peak_gain_fit_map(
    metrics_df: pd.DataFrame,
    outpath: Path | str,
    candidates: dict[str, pd.Series] | None = None,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    _plot_metric_map(
        metrics_df,
        "peak_gain_db_fit",
        outpath,
        title="Peak fitted gain",
        colorbar_label="Peak fitted gain (dB)",
        candidates=candidates,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )


def plot_gbp_fit_map(
    metrics_df: pd.DataFrame,
    outpath: Path | str,
    candidates: dict[str, pd.Series] | None = None,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    _plot_metric_map(
        metrics_df,
        "gbp_ghz_fit",
        outpath,
        title="Gain-bandwidth product",
        colorbar_label="Gain-bandwidth product (GHz)",
        candidates=candidates,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )


def plot_ripple_fit_map(
    metrics_df: pd.DataFrame,
    outpath: Path | str,
    candidates: dict[str, pd.Series] | None = None,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    _plot_metric_map(
        metrics_df,
        "ripple_db_fit",
        outpath,
        title="Fitted ripple",
        colorbar_label="Fitted ripple in operation band (dB)",
        candidates=candidates,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )


def plot_smoothness_fit_map(
    metrics_df: pd.DataFrame,
    outpath: Path | str,
    candidates: dict[str, pd.Series] | None = None,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    _plot_metric_map(
        metrics_df,
        "smoothness_norm_fit",
        outpath,
        title="Fitted smoothness",
        colorbar_label="Normalized fitted curvature",
        candidates=candidates,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )


def plot_status_map(
    metrics_df: pd.DataFrame,
    outpath: Path | str,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    labels = list(STATUS_LABELS)
    label_to_code = {label: i for i, label in enumerate(labels)}

    df = metrics_df.copy()
    df["status_code"] = df.apply(lambda row: label_to_code[status_label_for_row(row)], axis=1)
    pivot = df.pivot_table(
        index="pump_power_dbm",
        columns="pump_freq_ghz",
        values="status_code",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    cmap = ListedColormap(["#2ca25f", "#de2d26", "#fb6a4a", "#756bb1", "#fdae6b", "#636363", "#bdbdbd"])
    norm = BoundaryNorm(np.arange(len(labels) + 1) - 0.5, cmap.N)
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_MAP)
    mesh = ax.pcolormesh(_edges(x), _edges(y), z, cmap=cmap, norm=norm, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(len(labels)))
    cbar.ax.set_yticklabels(labels)
    ax.set_xlabel("Pump frequency fp / GHz")
    ax.set_ylabel("Pump power Pp / dBm")
    ax.set_title("Point status")
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)


def plot_simple_gain_map(
    points_df: pd.DataFrame,
    outpath: Path | str,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    """Plot the trailing-signal gain grid straight from the point table.

    Used for maps run without a per-cell signal spectrum: there is no fitted
    peak gain, so fall back to the single trailing gain (``gain_db``) each cell
    already carries. Non-converged cells are left blank.
    """
    df = points_df.copy()
    df["gain_value"] = pd.to_numeric(df.get("gain_db"), errors="coerce")
    if "status" in df.columns:
        passed = df["status"].map(lambda s: str(s).upper().startswith("PASS"))
        df.loc[~passed, "gain_value"] = np.nan
    pivot = df.pivot_table(
        index="pump_power_dbm",
        columns="pump_freq_ghz",
        values="gain_value",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_MAP)
    mesh = ax.pcolormesh(_edges(x), _edges(y), z, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Trailing-signal gain (dB)")
    ax.set_xlabel("Pump frequency fp / GHz")
    ax.set_ylabel("Pump power Pp / dBm")
    ax.set_title("Trailing-signal gain")
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)


def plot_runtime_map(
    points_df: pd.DataFrame,
    outpath: Path | str,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    """Plot per-cell runtime with total and average runtime annotations."""
    df = points_df.copy()
    runtime = _runtime_seconds(df)
    df["runtime_s"] = runtime
    pivot = df.pivot_table(
        index="pump_power_dbm",
        columns="pump_freq_ghz",
        values="runtime_s",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    finite = np.isfinite(z)
    total_s = float(np.nansum(z))
    average_s = float(np.nanmean(z)) if np.any(finite) else float("nan")
    solved = int(np.count_nonzero(finite))

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.25], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    mesh = ax.pcolormesh(_edges(x), _edges(y), z, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Runtime per cell (s)")
    ax.set_xlabel("Pump frequency fp / GHz")
    ax.set_ylabel("Pump power Pp / dBm")
    ax.set_title("Runtime per cell")

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "Runtime summary\n"
        f"Total = {_format_seconds(total_s)}\n"
        f"Average cell = {average_s:.3f} s\n"
        f"Cells = {solved}",
        va="top",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.7"},
    )
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)


def _runtime_seconds(df: pd.DataFrame) -> pd.Series:
    if "elapsed_s" in df.columns:
        return pd.to_numeric(df["elapsed_s"], errors="coerce")
    total = pd.Series(np.nan, index=df.index, dtype=float)
    parts = []
    for column in ("pump_wall_runtime_s", "gain_wall_runtime_s"):
        if column in df.columns:
            parts.append(pd.to_numeric(df[column], errors="coerce").fillna(0.0))
    if parts:
        total = sum(parts)
    return total


def _format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "n/a"
    if seconds < 120.0:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 120.0:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.2f} h"


def status_label_for_row(row: pd.Series | dict[str, object]) -> str:
    """Map solver status columns to the categorical status-map label."""
    status = _upper_value(row.get("status", "UNKNOWN"))
    pump_status = _upper_value(row.get("pump_status", ""))
    gain_status = _upper_value(row.get("gain_status", ""))
    combined = " ".join(part for part in (status, pump_status, gain_status) if part)
    if status.startswith("PASS"):
        return "PASS"
    if "INVALID_GAIN" in combined:
        return "INVALID_GAIN"
    if "FOLD" in combined or "SKIP" in combined:
        return "FOLD_SKIPPED"
    if "TIMEOUT" in combined:
        return "TIMEOUT"
    if pump_status in {"FAIL", "ERROR", "MISSING"}:
        return "PUMP_FAILED"
    if gain_status in {"FAIL", "ERROR", "MISSING", "UNKNOWN"} and pump_status in {
        "",
        "VALID_CONVERGED",
        "PASS",
    }:
        return "GAIN_FAILED"
    if "PUMP" in combined and ("FAIL" in combined or "ERROR" in combined):
        return "PUMP_FAILED"
    if "GAIN" in combined and ("FAIL" in combined or "ERROR" in combined):
        return "GAIN_FAILED"
    if status == "ERROR":
        return "GAIN_FAILED" if pump_status == "VALID_CONVERGED" else "PUMP_FAILED"
    return "UNKNOWN"


def _upper_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).upper()


def plot_selected_candidate_map(
    metrics_df: pd.DataFrame,
    candidates: dict[str, pd.Series],
    outpath: Path | str,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    x, y, z = _metric_grid(metrics_df, "peak_gain_db_fit")
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.45], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    mesh = ax.pcolormesh(_edges(x), _edges(y), z, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Peak fitted gain (dB)")
    _overlay_candidate_markers(ax, candidates)
    ax_text.axis("off")
    blocks: list[str] = []
    for labels, row in _group_candidate_rows(candidates):
        label_text = ", ".join(labels)
        blocks.append(
            f"{label_text}\n"
            f"Pp = {float(row['pump_power_dbm']):.3f} dBm\n"
            f"fp = {float(row['pump_freq_ghz']):.3f} GHz\n"
            f"Gmax = {float(row['peak_gain_db_fit']):.2f} dB\n"
            f"GBP = {float(row['gbp_ghz_fit']):.3g} GHz\n"
            f"Ripple = {float(row['ripple_db_fit']):.2f} dB\n"
            f"Smoothness = {float(row['smoothness_norm_fit']):.3g}"
        )
    ax_text.text(
        0.0,
        1.0,
        "\n\n".join(blocks),
        va="top",
        ha="left",
        fontsize=8,
        transform=ax_text.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.7"},
    )
    ax.set_xlabel("Pump frequency fp / GHz")
    ax.set_ylabel("Pump power Pp / dBm")
    ax.set_title("Selected candidates")
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)
