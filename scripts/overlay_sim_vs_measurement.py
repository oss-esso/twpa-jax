"""Superimpose a run_gain_map sim map on the Themis measurement, same axes.

Unlike align_map_to_measurement.py's 4-panel comparison (measured / aligned
sim / residual / loss surface as separate side-by-side panels), this draws
BOTH on the same (pump freq, pump power) axes so misalignment/agreement is
visible by eye directly, with no shift fit applied. Meant for a map already
built on the measurement's own grid (same n-power/n-frequency and range), so
the two datasets share axes with no interpolation needed.

  * Panel 1: measurement heatmap, sim drawn on top as gain-level contour lines.
  * Panel 2: sim heatmap, measurement drawn on top as gain-level contour lines.
  * Panel 3: measurement (background, muted) with sim as a semi-transparent
    color overlay -- the closest thing to literal superposition.

Usage:
    python scripts/overlay_sim_vs_measurement.py \\
        --map-dir outputs/lj84_cg62p7_themis_grid_map \\
        --measurement-dir docs/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK \\
        --out outputs/lj84_cg62p7_themis_grid_map/overlay.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import align_map_to_measurement as amm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map-dir", required=True, type=Path)
    p.add_argument("--measurement-dir", required=True, type=Path)
    p.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    p.add_argument("--contour-levels", type=float, nargs="+", default=[3.0, 6.0, 10.0, 15.0, 20.0])
    p.add_argument("--vmax", type=float, default=None, help="Colorbar max gain (dB); default auto.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    meas = amm.load_measurement_map(args.measurement_dir, tuple(args.signal_band_ghz))
    sim = amm.load_sim_map(args.map_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mf, mp, mg = meas["pump_freq_ghz"], meas["pump_power_dbm"], meas["peak_gain_db"].T
    sf, sp, sg = sim["pump_freq_ghz"], sim["pump_power_dbm"], sim["gain_db"].T

    vmax = args.vmax if args.vmax is not None else max(20.0, float(np.nanmax(mg)))
    levels = sorted(args.contour_levels)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6), sharex=True, sharey=True)

    pcm1 = ax1.pcolormesh(mf, mp, mg, shading="nearest", cmap="magma", vmin=0.0, vmax=vmax)
    cs1 = ax1.contour(sf, sp, sg, levels=levels, colors="cyan", linewidths=1.2)
    ax1.clabel(cs1, inline=True, fontsize=7, fmt="%.0f dB")
    ax1.set_title("measurement (color) + sim (cyan contours)")
    fig.colorbar(pcm1, ax=ax1, label="measured gain (dB)")

    pcm2 = ax2.pcolormesh(sf, sp, sg, shading="nearest", cmap="magma", vmin=0.0, vmax=vmax)
    cs2 = ax2.contour(mf, mp, mg, levels=levels, colors="lime", linewidths=1.2)
    ax2.clabel(cs2, inline=True, fontsize=7, fmt="%.0f dB")
    ax2.set_title("sim (color) + measurement (lime contours)")
    fig.colorbar(pcm2, ax=ax2, label="sim gain (dB)")

    ax3.pcolormesh(mf, mp, mg, shading="nearest", cmap="Greys", vmin=0.0, vmax=vmax)
    sg_masked = np.ma.masked_invalid(sg)
    pcm3 = ax3.pcolormesh(sf, sp, sg_masked, shading="nearest", cmap="magma",
                          vmin=0.0, vmax=vmax, alpha=0.65)
    ax3.set_title("measurement (grey bg) + sim (color, alpha=0.65)")
    fig.colorbar(pcm3, ax=ax3, label="sim gain (dB)")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("pump frequency (GHz)")
    ax1.set_ylabel("pump power (dBm)")

    fig.suptitle(f"{args.map_dir.name} vs {args.measurement_dir.name}")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
