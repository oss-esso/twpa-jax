"""Plot IPM gain-map artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_gain_map(root: str | Path) -> None:
    root = Path(root)
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    signal = np.loadtxt(root / "gain_signal_db_grid.csv", delimiter=",")
    idler = np.loadtxt(root / "idler_gain_db_grid.csv", delimiter=",")
    mask = np.loadtxt(root / "convergence_mask_grid.csv", delimiter=",")
    residual = np.loadtxt(root / "residual_norm_grid.csv", delimiter=",")
    runtime = np.loadtxt(root / "runtime_grid.csv", delimiter=",")
    _heatmap(signal, plots / "signal_gain_unmarked.png", "Signal gain (dB)")
    marked = signal.copy()
    marked[mask < 0.5] = np.nan
    _heatmap(marked, plots / "signal_gain_marked_by_convergence.png", "Signal gain, converged cells")
    _heatmap(marked, plots / "signal_gain_converged_only.png", "Signal gain, converged only")
    _heatmap(idler, plots / "idler_gain_unmarked.png", "Idler gain (dB)")
    _heatmap(np.log10(np.maximum(residual, 1e-300)), plots / "residual_norm_heatmap.png", "log10 residual inf")
    _heatmap(runtime, plots / "runtime_heatmap.png", "Runtime (s)")


def _heatmap(values: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.8), constrained_layout=True)
    image = ax.imshow(values.T, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Pump frequency index")
    ax.set_ylabel("Pump power index")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    plot_gain_map(args.root)


if __name__ == "__main__":
    main()
