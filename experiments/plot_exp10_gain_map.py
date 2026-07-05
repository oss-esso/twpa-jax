"""Plot exp10 warm-start gain maps (gain in dB over pump power x pump frequency).

Reads ``map_arrays.npz`` from an exp10 output dir and renders a heatmap per
available gain grid (`gain_db_warm`, and for ``both`` runs `gain_db_cold` and
`gain_drift_db`). NaN holes (non-converged points) are drawn in grey.

Usage:
    python experiments/plot_exp10_gain_map.py outputs/exp10_pump_map_warmstart_5x5
    python experiments/plot_exp10_gain_map.py <dir> --signal-ghz 7.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        c = centers[0]
        return np.array([c - 0.5, c + 0.5])
    mid = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def signal_label(map_dir: Path, override_ghz: float | None) -> str:
    """Describe the readout signal for the plot title.

    A map sweeps the *pump* frequency; the signal is either a fixed absolute
    frequency (``--signal-ghz`` / summary ``signal_ghz``) or, by default, a
    trailing tone at ``ws = fp - detuning`` that tracks the pump per column. A
    single "signal X GHz" label is wrong for the trailing case, so describe the
    convention instead.
    """
    if override_ghz is not None:
        return f"signal {override_ghz:g} GHz"
    summ = map_dir / "map_summary.json"
    if not summ.exists():
        return "signal ?"
    try:
        meta = json.loads(summ.read_text())
    except (ValueError, OSError):
        return "signal ?"
    fixed = meta.get("signal_ghz")
    if fixed is not None:
        return f"signal {float(fixed):g} GHz"
    det = meta.get("signal_detuning_mhz")
    if det is not None:
        return f"trailing signal (ws = fp - {float(det):g} MHz)"
    conv = meta.get("signal_convention")
    if isinstance(conv, str) and conv.lower().startswith("ws ="):
        return f"trailing {conv.replace('wp', 'fp')}"
    return "signal ?"


def plot_grid(
    grid: np.ndarray,
    powers: np.ndarray,
    freqs: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
    diverging: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    masked = np.ma.masked_invalid(grid)
    x_edges = edges_from_centers(freqs)
    y_edges = edges_from_centers(powers)

    if diverging:
        vmax = float(np.nanmax(np.abs(grid))) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.cm.RdBu_r.copy()
    else:
        finite = grid[np.isfinite(grid)]
        norm = None
        cmap = plt.cm.viridis.copy()
        if finite.size:
            ax_vmin, ax_vmax = float(np.min(finite)), float(np.max(finite))
            norm = plt.Normalize(vmin=ax_vmin, vmax=ax_vmax)
    cmap.set_bad("0.6")  # grey for NaN holes

    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, norm=norm, shading="flat")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel("pump frequency (GHz)")
    ax.set_ylabel("pump power (dBm, external)")
    ax.set_title(title)

    # Mark the peak (finite) gain cell.
    if not diverging and np.any(np.isfinite(grid)):
        ip, jf = np.unravel_index(np.nanargmax(grid), grid.shape)
        ax.plot(freqs[jf], powers[ip], "r*", markersize=14, markeredgecolor="white")
        ax.annotate(
            f"max {grid[ip, jf]:.2f} dB",
            (freqs[jf], powers[ip]),
            color="white",
            fontsize=8,
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("--signal-ghz", type=float, default=None,
                    help="Override the signal frequency shown in titles.")
    args = ap.parse_args()

    map_dir = args.map_dir
    arrays = np.load(map_dir / "map_arrays.npz", allow_pickle=True)
    powers = arrays["pump_power_dbm"]
    freqs = arrays["pump_frequency_ghz"]
    sig = signal_label(map_dir, args.signal_ghz)
    shape = f"{len(powers)}x{len(freqs)}"

    if "gain_db_warm" in arrays.files:
        g = arrays["gain_db_warm"]
        n_hole = int(np.sum(~np.isfinite(g)))
        plot_grid(
            g, powers, freqs,
            title=f"IPM JTWPA gain (warm-start) {shape}, {sig}"
            + (f" — {n_hole} holes" if n_hole else ""),
            cbar_label="gain S21 (dB)",
            out_path=map_dir / "gain_map_warm.png",
        )

    if "gain_db_cold" in arrays.files:
        plot_grid(
            arrays["gain_db_cold"], powers, freqs,
            title=f"IPM JTWPA gain (cold reference) {shape}, {sig}",
            cbar_label="gain S21 (dB)",
            out_path=map_dir / "gain_map_cold.png",
        )

    if "gain_drift_db" in arrays.files:
        plot_grid(
            arrays["gain_drift_db"], powers, freqs,
            title=f"warm - cold gain drift {shape}, {sig}",
            cbar_label="|gain drift| (dB)",
            out_path=map_dir / "gain_map_drift.png",
        )


if __name__ == "__main__":
    main()
