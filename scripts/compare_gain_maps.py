"""Plot headline gain error maps between saved gain-map runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twpa_solver.plotting.maps import _edges
from twpa_solver.plotting.style import THESIS_FIGSIZE_MAP, save_figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--save-svg", action="store_true")
    return parser


def load_gain_grid(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays_path = run_dir / "map_arrays.npz"
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=True) as data:
            gain_name = "gain_db_warm" if "gain_db_warm" in data.files else "peak_gain_db"
            if gain_name in data.files:
                power = np.asarray(data["pump_power_dbm"], dtype=float)
                freq = np.asarray(data["pump_frequency_ghz"], dtype=float)
                gain = np.asarray(data[gain_name], dtype=float)
                return power, freq, gain

    points_path = run_dir / "map_points.csv"
    if not points_path.exists():
        raise FileNotFoundError(f"missing map_arrays.npz and map_points.csv in {run_dir}")
    points = pd.read_csv(points_path)
    if "pass" in points.columns:
        points = points[points["pass"].astype(str).eq("warm")]
    pivot = points.pivot_table(
        index="pump_power_dbm",
        columns="pump_freq_ghz",
        values="gain_db",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)
    return pivot.index.to_numpy(dtype=float), pivot.columns.to_numpy(dtype=float), pivot.to_numpy(dtype=float)


def plot_gain_error_map(
    reference_run: Path,
    run_dir: Path,
    outpath: Path,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    ref_power, ref_freq, ref_gain = load_gain_grid(reference_run)
    power, freq, gain = load_gain_grid(run_dir)
    if ref_gain.shape != gain.shape or not np.allclose(ref_power, power) or not np.allclose(ref_freq, freq):
        raise ValueError(f"{run_dir} grid does not match reference grid")

    diff = gain - ref_gain
    finite = np.isfinite(diff)
    max_abs = float(np.nanmax(np.abs(diff))) if np.any(finite) else float("nan")
    rms = float(np.sqrt(np.nanmean(diff[finite] ** 2))) if np.any(finite) else float("nan")
    mean_abs = float(np.nanmean(np.abs(diff[finite]))) if np.any(finite) else float("nan")
    count = int(np.count_nonzero(finite))
    vlim = max(max_abs, 1e-9) if np.isfinite(max_abs) else 1.0

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.25], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    mesh = ax.pcolormesh(
        _edges(freq),
        _edges(power),
        diff,
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax, label="Gain error: run - reference (dB)")
    ax.set_xlabel("Pump frequency fp / GHz")
    ax.set_ylabel("Pump power Pp / dBm")
    ax.set_title(f"Gain error: {run_dir.name}")
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        f"Reference\n{reference_run.name}\n\n"
        f"Run\n{run_dir.name}\n\n"
        f"RMS = {rms:.4g} dB\n"
        f"Mean |err| = {mean_abs:.4g} dB\n"
        f"Max |err| = {max_abs:.4g} dB\n"
        f"Compared cells = {count}",
        va="top",
        ha="left",
        fontsize=9,
        transform=ax_text.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.7"},
    )
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def main() -> int:
    args = build_parser().parse_args()
    outdir = args.outdir or (args.reference_run / "plots" / "comparisons")
    outdir.mkdir(parents=True, exist_ok=True)
    for run_dir in args.run:
        outpath = outdir / f"gain_error_{_slug(run_dir.name)}.png"
        plot_gain_error_map(
            args.reference_run,
            run_dir,
            outpath,
            save_pdf=args.save_pdf,
            save_svg=args.save_svg,
        )
    print(f"Wrote comparison maps to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
