#!/usr/bin/env python3
"""Compute and plot pump-off S-parameters for an IPM design.

Examples
--------
    python scripts/plot_passive_s21_s24.py \
        --ipm-dir outputs/ipm_python_design \
        --outdir outputs/passive_2c_s21_s24

The port convention is the IPM convention: S21 is signal input port 1 to
signal output port 2, and S24 is pump-side port 4 to signal output port 2.
Use ``--plot-set directional`` for one four-panel figure containing
S11/S21/S31/S41.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twpa_solver.signal.passive import db20, passive_s_matrix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ipm-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--start-ghz", type=float, default=4.0)
    p.add_argument("--stop-ghz", type=float, default=11.0)
    p.add_argument("--points", type=int, default=1401)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument(
        "--plot-set",
        choices=("s21_s24", "directional"),
        default="s21_s24",
        help="Default two-panel S21/S24 plot, or two three-panel directional plots.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.points < 2:
        raise SystemExit("--points must be at least 2")
    if args.stop_ghz <= args.start_ghz:
        raise SystemExit("--stop-ghz must exceed --start-ghz")

    args.outdir.mkdir(parents=True, exist_ok=True)
    freq_ghz = np.linspace(args.start_ghz, args.stop_ghz, args.points)
    t0 = time.perf_counter()
    s = passive_s_matrix(
        args.ipm_dir,
        freq_ghz * 1e9,
        ports=(1, 2, 3, 4),
        z0_ohm=args.z0_ohm,
    )
    names = ("s11", "s21", "s31", "s41", "s14", "s24", "s34")
    port_pairs = ((1, 1), (2, 1), (3, 1), (4, 1), (1, 4), (2, 4), (3, 4))
    traces = {
        name: db20(s[:, out - 1, source - 1])
        for name, (out, source) in zip(names, port_pairs)
    }
    s21_db = traces["s21"]
    s24_db = traces["s24"]

    np.savez_compressed(
        args.outdir / "passive_s21_s24.npz",
        freq_ghz=freq_ghz,
        **{f"{name}_db": values for name, values in traces.items()},
    )

    def save_figure(names_to_plot: tuple[str, ...], stem: str, title: str) -> None:
        fig, axes = plt.subplots(
            len(names_to_plot), 1,
            figsize=(11, 3.2 * len(names_to_plot)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names_to_plot):
            ax.plot(freq_ghz, traces[name], lw=1.2)
            ax.set_ylabel(f"|{name.upper()}| (dB)")
            ax.grid(True, alpha=0.3)
            ax.minorticks_on()
        axes[0].set_title(title)
        axes[-1].set_xlabel("Signal frequency (GHz)")
        fig.tight_layout()
        for suffix, kwargs in (("png", {"dpi": 200}), ("pdf", {}), ("svg", {})):
            fig.savefig(args.outdir / f"{stem}.{suffix}", **kwargs)
        plt.close(fig)

    if args.plot_set == "directional":
        save_figure(
            ("s11", "s21", "s31", "s41"),
            "passive_s11_s21_s31_s41",
            f"Pump-off response from port 1: {args.ipm_dir.name}",
        )
    else:
        save_figure(
            ("s21", "s24"),
            "passive_s21_s24",
            f"Pump-off response: {args.ipm_dir.name}",
        )

    elapsed = time.perf_counter() - t0
    print(
        f"wrote={args.outdir} points={args.points} elapsed_s={elapsed:.3f} "
        f"s21_db=[{s21_db.min():.3f},{s21_db.max():.3f}] "
        f"s24_db=[{s24_db.min():.3f},{s24_db.max():.3f}]."
    )


if __name__ == "__main__":
    main()
