"""Plot exp10 warm-start gain maps (gain in dB over pump power x pump frequency).

Reads ``map_arrays.npz`` from an exp10 output dir and renders heatmaps for the
available base gain grids. With ``--metric spectrum_sg_peak`` it reads
``map_spectrum.npz``, smooths each signal spectrum with Savitzky-Golay,
interpolates along signal offset, and plots the per-cell peak gain.

Usage:
    python experiments/plot_exp10_gain_map.py outputs/exp10_pump_map_warmstart_5x5
    python experiments/plot_exp10_gain_map.py <dir> --signal-ghz 7.5
    python experiments/plot_exp10_gain_map.py <dir> --metric spectrum_sg_peak
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
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


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
    """Describe the readout signal for the plot title."""
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


def _valid_savgol_window(n: int, requested: int, polyorder: int) -> int | None:
    """Largest usable odd Savitzky-Golay window, or None if too few points."""
    if n <= polyorder:
        return None
    w = min(int(requested), n)
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        w = polyorder + 1
        if w % 2 == 0:
            w += 1
    if w > n:
        return None
    return w


def spectrum_peak_grid(
    map_dir: Path,
    *,
    metric: str,
    sg_window: int,
    sg_polyorder: int,
    interp_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gain/signal/offset peak grids from ``map_spectrum.npz``."""
    spec_path = map_dir / "map_spectrum.npz"
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} missing; rerun map with --signal-spectrum")

    spec = np.load(spec_path)
    offsets = np.asarray(spec["signal_offset_mhz"], dtype=float)
    cube = np.asarray(spec["gain_spectrum_db"], dtype=float)
    signal_ghz = np.asarray(spec["signal_ghz"], dtype=float)

    if offsets.ndim != 1 or cube.ndim != 3 or cube.shape[2] != offsets.size:
        raise ValueError("map_spectrum.npz has unexpected spectrum shapes")
    if signal_ghz.shape != (offsets.size, cube.shape[1]):
        raise ValueError("signal_ghz shape must be (n_offset, n_frequency)")
    if interp_factor < 1:
        raise ValueError("--interp-factor must be >= 1")

    dense_offsets = np.linspace(
        float(offsets.min()),
        float(offsets.max()),
        (offsets.size - 1) * int(interp_factor) + 1,
    )
    peak_gain = np.full(cube.shape[:2], np.nan, dtype=float)
    peak_signal = np.full(cube.shape[:2], np.nan, dtype=float)
    peak_offset = np.full(cube.shape[:2], np.nan, dtype=float)

    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            y = cube[i, j, :]
            mask = np.isfinite(y)
            if not np.any(mask):
                continue

            x = offsets[mask]
            yy = y[mask]
            if metric == "spectrum_sg_peak":
                w = _valid_savgol_window(len(yy), sg_window, sg_polyorder)
                if w is not None:
                    yy = savgol_filter(
                        yy,
                        window_length=w,
                        polyorder=min(sg_polyorder, w - 1),
                        mode="interp",
                    )

            if len(yy) >= 2:
                if metric == "spectrum_raw_peak":
                    k = int(np.nanargmax(yy))
                    off = float(x[k])
                    val = float(yy[k])
                else:
                    dense_mask = (dense_offsets >= x.min()) & (dense_offsets <= x.max())
                    xd = dense_offsets[dense_mask]
                    if xd.size:
                        yd = PchipInterpolator(x, yy, extrapolate=False)(xd)
                        k = int(np.nanargmax(yd))
                        off = float(xd[k])
                        val = float(yd[k])
                    else:
                        k = int(np.nanargmax(yy))
                        off = float(x[k])
                        val = float(yy[k])
            else:
                off = float(x[0])
                val = float(yy[0])

            peak_gain[i, j] = val
            peak_offset[i, j] = off
            peak_signal[i, j] = float(np.interp(off, offsets, signal_ghz[:, j]))

    return peak_gain, peak_signal, peak_offset


def write_spectrum_peak_cache(
    map_dir: Path,
    *,
    metric: str,
    peak_gain: np.ndarray,
    peak_signal: np.ndarray,
    peak_offset: np.ndarray,
    sg_window: int,
    sg_polyorder: int,
    interp_factor: int,
) -> None:
    out = map_dir / f"{metric}_arrays.npz"
    np.savez(
        out,
        peak_gain_db=peak_gain,
        peak_signal_ghz=peak_signal,
        peak_offset_mhz=peak_offset,
        sg_window=np.array([sg_window], dtype=np.int64),
        sg_polyorder=np.array([sg_polyorder], dtype=np.int64),
        interp_factor=np.array([interp_factor], dtype=np.int64),
    )
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir", type=Path)
    ap.add_argument(
        "--signal-ghz",
        type=float,
        default=None,
        help="Override the signal frequency shown in titles.",
    )
    ap.add_argument(
        "--metric",
        choices=["warm", "cold", "drift", "spectrum_raw_peak", "spectrum_sg_peak"],
        default=None,
        help="Plot one metric. Omit to keep legacy behavior and plot all base grids.",
    )
    ap.add_argument("--sg-window", type=int, default=5)
    ap.add_argument("--sg-polyorder", type=int, default=2)
    ap.add_argument("--interp-factor", type=int, default=25)
    args = ap.parse_args()

    map_dir = args.map_dir
    arrays = np.load(map_dir / "map_arrays.npz", allow_pickle=True)
    powers = arrays["pump_power_dbm"]
    freqs = arrays["pump_frequency_ghz"]
    sig = signal_label(map_dir, args.signal_ghz)
    shape = f"{len(powers)}x{len(freqs)}"

    if args.metric in ("spectrum_raw_peak", "spectrum_sg_peak"):
        g, peak_signal, peak_offset = spectrum_peak_grid(
            map_dir,
            metric=args.metric,
            sg_window=args.sg_window,
            sg_polyorder=args.sg_polyorder,
            interp_factor=args.interp_factor,
        )
        write_spectrum_peak_cache(
            map_dir,
            metric=args.metric,
            peak_gain=g,
            peak_signal=peak_signal,
            peak_offset=peak_offset,
            sg_window=args.sg_window,
            sg_polyorder=args.sg_polyorder,
            interp_factor=args.interp_factor,
        )
        n_hole = int(np.sum(~np.isfinite(g)))
        label = "raw spectrum peak" if args.metric == "spectrum_raw_peak" else "SG/interpolated spectrum peak"
        plot_grid(
            g,
            powers,
            freqs,
            title=f"IPM JTWPA gain ({label}) {shape}, {sig}"
            + (f" - {n_hole} holes" if n_hole else ""),
            cbar_label="peak gain S21 (dB)",
            out_path=map_dir / f"gain_map_{args.metric}.png",
        )
        return

    if args.metric in (None, "warm") and "gain_db_warm" in arrays.files:
        g = arrays["gain_db_warm"]
        n_hole = int(np.sum(~np.isfinite(g)))
        plot_grid(
            g,
            powers,
            freqs,
            title=f"IPM JTWPA gain (warm-start) {shape}, {sig}"
            + (f" - {n_hole} holes" if n_hole else ""),
            cbar_label="gain S21 (dB)",
            out_path=map_dir / "gain_map_warm.png",
        )

    if args.metric in (None, "cold") and "gain_db_cold" in arrays.files:
        plot_grid(
            arrays["gain_db_cold"],
            powers,
            freqs,
            title=f"IPM JTWPA gain (cold reference) {shape}, {sig}",
            cbar_label="gain S21 (dB)",
            out_path=map_dir / "gain_map_cold.png",
        )

    if args.metric in (None, "drift") and "gain_drift_db" in arrays.files:
        plot_grid(
            arrays["gain_drift_db"],
            powers,
            freqs,
            title=f"warm - cold gain drift {shape}, {sig}",
            cbar_label="|gain drift| (dB)",
            out_path=map_dir / "gain_map_drift.png",
        )


if __name__ == "__main__":
    main()
