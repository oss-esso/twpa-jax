"""Plot canonical all-backend old-IPM map outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    root = Path(args.root)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        (root / "plots_unavailable.txt").write_text(
            f"matplotlib unavailable: {exc!r}\n", encoding="utf-8"
        )
        return

    backends_dir = root / "backends"
    for backend_dir in sorted(p for p in backends_dir.iterdir() if p.is_dir()):
        rows_path = backend_dir / "rows.csv"
        if not rows_path.exists():
            continue
        rows = _read_rows(rows_path)
        plots_dir = backend_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_heatmap(plt, rows, "gain_db_max", plots_dir / "gain_marked.png", "gain dB")
        _plot_heatmap(
            plt,
            [r for r in rows if r.get("status") == "VALID_CONVERGED"],
            "gain_db_max",
            plots_dir / "gain_converged_only.png",
            "converged gain dB",
        )
        _plot_heatmap(plt, rows, "residual_norm", plots_dir / "residual_norm.png", "residual norm")
        _plot_heatmap(plt, rows, "infinity_norm", plots_dir / "infinity_norm.png", "infinity norm")
        _plot_heatmap(plt, rows, "point_runtime_s", plots_dir / "point_runtime.png", "runtime s")
        _plot_status_counts(plt, rows, plots_dir / "status_counts.png")

    comp_dir = root / "comparison"
    comp_plot_dir = comp_dir / "plots"
    comp_plot_dir.mkdir(parents=True, exist_ok=True)
    summary_path = root / "all_backend_summary.csv"
    if summary_path.exists():
        summary_rows = _read_rows(summary_path)
        _plot_summary_bar(
            plt,
            summary_rows,
            "median_point_runtime_s",
            comp_plot_dir / "runtime_comparison.png",
            "median point runtime s",
        )
        _plot_summary_bar(
            plt,
            summary_rows,
            "valid_converged",
            comp_plot_dir / "convergence_rate_by_backend.png",
            "valid converged cells",
        )
        _plot_summary_bar(
            plt,
            summary_rows,
            "gain_cells",
            comp_plot_dir / "gain_comparison_panel.png",
            "finite gain cells",
        )
        _plot_summary_bar(
            plt,
            summary_rows,
            "residual_reduced_not_converged",
            comp_plot_dir / "residual_by_backend.png",
            "residual-reduced cells",
        )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_nan(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _grid(rows: list[dict[str, str]], key: str) -> tuple[list[float], list[float], np.ndarray]:
    freqs = sorted({_float_or_nan(r.get("pump_frequency_ghz")) for r in rows})
    powers = sorted({_float_or_nan(r.get("external_power_dbm")) for r in rows})
    freqs = [x for x in freqs if np.isfinite(x)]
    powers = [x for x in powers if np.isfinite(x)]
    values = np.full((len(powers), len(freqs)), np.nan)
    f_index = {v: i for i, v in enumerate(freqs)}
    p_index = {v: i for i, v in enumerate(powers)}
    for row in rows:
        fp = _float_or_nan(row.get("pump_frequency_ghz"))
        pext = _float_or_nan(row.get("external_power_dbm"))
        if fp in f_index and pext in p_index:
            values[p_index[pext], f_index[fp]] = _float_or_nan(row.get(key))
    return freqs, powers, values


def _plot_heatmap(plt, rows: list[dict[str, str]], key: str, path: Path, title: str) -> None:
    freqs, powers, values = _grid(rows, key)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    if values.size and np.isfinite(values).any():
        im = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=[min(freqs), max(freqs), min(powers), max(powers)],
        )
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("pump frequency GHz")
    ax.set_ylabel("external pump power dBm")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_status_counts(plt, rows: list[dict[str, str]], path: Path) -> None:
    counts = Counter(row.get("status", "") for row in rows)
    labels = list(counts)
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("count")
    ax.set_title("status counts")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_summary_bar(plt, rows: list[dict[str, str]], key: str, path: Path, title: str) -> None:
    labels = [row.get("backend", "") for row in rows]
    values = [_float_or_nan(row.get(key)) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_title(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
