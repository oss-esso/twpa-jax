"""Overlay pump-off S21 vs signal frequency for a grid of (Lj, Cg) designs.

Usage:
    python scripts/plot_s21_lj_cg_grid.py \
        --design outputs/lj_periodicity_designs/ipm_lj123p9_cg66 "Lj=123.9 Cg=66" \
        --design outputs/periodicity_campaign_designs/lj79_cg66 "Lj=79 Cg=66" \
        ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import ripple_common as rc  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", nargs=2, action="append", required=True,
                    metavar=("DIR", "LABEL"))
    p.add_argument("--start-ghz", type=float, default=6.0)
    p.add_argument("--stop-ghz", type=float, default=9.0)
    p.add_argument("--points", type=int, default=601)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument("--out", type=Path, default=Path("plots/s21_lj_cg_grid.png"))
    p.add_argument("--separate", action="store_true",
                    help="One subplot per design instead of one overlay.")
    args = p.parse_args()

    freq_ghz = np.linspace(args.start_ghz, args.stop_ghz, args.points)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = []
    for design_dir, label in args.design:
        s = rc.passive_s_matrix(Path(design_dir), freq_ghz * 1e9, ports=(1, 2), z0_ohm=args.z0_ohm)
        s21_db = rc.db20(s[:, 1, 0])
        curves.append((label, s21_db))
        print(f"{label:20s} S21 range [{s21_db.min():.3f}, {s21_db.max():.3f}] dB")

    if args.separate:
        n = len(curves)
        ncols = 2
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (label, s21_db) in enumerate(curves):
            ax = axes[i]
            ax.plot(freq_ghz, s21_db, lw=1.2, color=colors[i % len(colors)])
            ax.set_title(label, fontsize=10)
            ax.grid(True, which="both", alpha=0.3)
            ax.minorticks_on()
        for ax in axes[len(curves):]:
            ax.axis("off")
        for ax in axes[-ncols:]:
            ax.set_xlabel("signal frequency (GHz)")
        for row_start in range(0, len(curves), ncols):
            axes[row_start].set_ylabel("|S21| (dB)")
        fig.suptitle("pump-off S21 vs signal frequency")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        for label, s21_db in curves:
            ax.plot(freq_ghz, s21_db, lw=1.3, label=label)
        ax.set_xlabel("signal frequency (GHz)")
        ax.set_ylabel("|S21| (dB)")
        ax.set_title("pump-off S21 vs signal frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
