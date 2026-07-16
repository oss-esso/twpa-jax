"""Plot a Themis gain-map measurement and align a 2c simulation to it by shifts.

Two jobs:

  1. Reduce the measurement cube (per-pump-frequency ``105C5_*GHz.npy`` files,
     each a transmission/gain response over pump power x signal frequency) to a
     2-D peak-gain map ``G_meas[pump_freq, pump_power]`` and plot it. This is the
     "plot the measurement map" step -- the measurement ships as raw data only.

  2. Treat the measurement<->simulation calibration offsets as nuisance
     parameters and fit them directly, rather than hand-tuning them. The model:

         G_meas(f, P) ~= G_sim(f - df, P - dP) + dG

     * df : pump-frequency translation (GHz)
     * dP : pump-power translation (dBm)
     * dG : additive gain-reference offset (dB) -- a z-offset of the map value

     For a weighted least-squares fit dG is analytic for any (df, dP):

         dG*(df, dP) = sum_i w_i [G_meas_i - G_sim_i(df, dP)] / sum_i w_i

     so only a 2-D search over (df, dP) remains. We do a coarse grid then a fine
     refine, sampling the simulation with ``RegularGridInterpolator`` at the
     shifted measurement coordinates, masking non-overlap and failed (NaN) sim
     cells, and down-weighting the flat low-gain background so the amplified
     ridge -- not the zero-gain sea -- drives the fit. The full (df, dP) loss
     surface is returned and plotted: a diagonal gain ridge makes df and dP
     partly correlated, and the width/orientation of the minimum shows how well
     the offsets are identified.

Writes a JSON summary and, if matplotlib is present, two PNGs: the measurement
map on its own, and a four-panel comparison (measured map, aligned simulation,
residual, loss surface).

This is a calibration layer: absorb measurement-axis and gain-reference offsets
first, before fitting physical circuit parameters (Ic, Lj, coupling, loss).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

FREQ_RE = re.compile(r"105C5_([0-9.]+)GHz\.npy$")


def load_measurement_map(
    meas_dir: Path, signal_band_ghz: tuple[float, float]
) -> dict[str, np.ndarray]:
    """Reduce the measurement cube to a peak-gain map over a signal band.

    Returns ``pump_freq_ghz`` (n_freq,), ``pump_power_dbm`` (n_power,) and
    ``peak_gain_db`` (n_freq, n_power) = max transmission over the signal band.
    """
    files = sorted(meas_dir.glob("105C5_*GHz.npy"))
    if not files:
        raise FileNotFoundError(f"no measurement npy files in {meas_dir}")
    lo, hi = signal_band_ghz
    freqs: list[float] = []
    rows: list[np.ndarray] = []
    powers_ref: np.ndarray | None = None
    for f in files:
        m = FREQ_RE.search(f.name)
        if not m:
            continue
        d = np.load(f, allow_pickle=True).item()
        sig = np.asarray(d["Frequency"], dtype=float) / 1e9  # Hz -> GHz
        resp = np.asarray(d["Response"], dtype=float)  # (n_power, n_sig)
        powers = np.asarray(d["PumpPower"], dtype=float)  # (n_power,)
        band = (sig >= lo) & (sig <= hi)
        if not band.any():
            raise ValueError(f"signal band {signal_band_ghz} GHz empty for {f.name}")
        freqs.append(float(m.group(1)))
        rows.append(resp[:, band].max(axis=1))  # peak over signal
        powers_ref = powers if powers_ref is None else powers_ref
    order = np.argsort(freqs)
    return {
        "pump_freq_ghz": np.asarray(freqs)[order],
        "pump_power_dbm": np.asarray(powers_ref, dtype=float),
        "peak_gain_db": np.asarray(rows)[order],  # (n_freq, n_power)
    }


def load_sim_map(map_dir: Path) -> dict[str, np.ndarray]:
    """Load a run_gain_map output as a peak-gain map ``gain[freq, power]``.

    ``map_arrays.npz`` stores ``gain_db_warm[power, freq]`` with NaN in failed
    cells; we transpose to (freq, power) to match the measurement layout.
    """
    arr = np.load(map_dir / "map_arrays.npz")
    gain = arr["gain_db_warm"].astype(float)  # (n_power, n_freq)
    return {
        "pump_freq_ghz": arr["pump_frequency_ghz"].astype(float),
        "pump_power_dbm": arr["pump_power_dbm"].astype(float),
        "gain_db": gain.T,  # (n_freq, n_power)
    }


def _roi_weights(
    peak_gain_db: np.ndarray,
    threshold_db: float,
    floor: float,
    freq: np.ndarray,
    power: np.ndarray,
    fit_freq_range: tuple[float, float] | None,
    fit_power_range: tuple[float, float] | None,
) -> np.ndarray:
    """Region-of-interest weights: 1 on the amplified ridge, ``floor`` elsewhere.

    Keeps the flat low-gain background from dominating a map-wide least-squares.
    ``fit_freq_range`` / ``fit_power_range`` (measurement axes) hard-mask cells
    outside a window to fit only one section of the map (weight 0 outside).
    """
    w = np.full(peak_gain_db.shape, floor, dtype=float)
    w[np.isfinite(peak_gain_db) & (peak_gain_db > threshold_db)] = 1.0
    w[~np.isfinite(peak_gain_db)] = 0.0
    if fit_freq_range is not None:
        lo, hi = fit_freq_range
        w[(freq < lo) | (freq > hi), :] = 0.0
    if fit_power_range is not None:
        lo, hi = fit_power_range
        w[:, (power < lo) | (power > hi)] = 0.0
    return w


def _huber_rho(residual: np.ndarray, delta: float) -> np.ndarray:
    """Huber loss rho(r): quadratic within +/-delta, linear outside."""
    a = np.abs(residual)
    quad = 0.5 * residual**2
    lin = delta * (a - 0.5 * delta)
    return np.where(a <= delta, quad, lin)


def _score_shift(
    df: float,
    dP: float,
    meas_ff: np.ndarray,
    meas_pp: np.ndarray,
    meas_g: np.ndarray,
    weights: np.ndarray,
    sim_interp: RegularGridInterpolator,
    n_ref: float,
    loss: str,
    huber_delta: float,
    min_overlap_frac: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Score one (df, dP) shift. Returns (score, dG, sim_on_meas, valid_mask).

    ``sim_on_meas`` is the simulation resampled onto the measurement grid (before
    adding dG); ``score`` is normalised by overlap so shrinking the overlap is
    not rewarded. A shift whose overlap covers less than ``min_overlap_frac`` of
    the weighted fit region is rejected (a soft 1/overlap penalty alone still
    lets a tiny-overlap corner with near-zero local residual win).
    """
    pts = np.column_stack([(meas_ff - df).ravel(), (meas_pp - dP).ravel()])
    sim_on_meas = sim_interp(pts).reshape(meas_g.shape)
    valid = np.isfinite(sim_on_meas) & np.isfinite(meas_g) & (weights > 0)
    if valid.sum() < 8 or valid.sum() < min_overlap_frac * n_ref:
        return np.inf, np.nan, sim_on_meas, valid
    w = weights[valid]
    resid0 = meas_g[valid] - sim_on_meas[valid]
    dG = float(np.average(resid0, weights=w))
    resid = resid0 - dG
    if loss == "huber":
        cost = float(np.average(_huber_rho(resid, huber_delta), weights=w))
    else:
        cost = float(np.average(resid**2, weights=w))
    overlap = valid.sum() / n_ref
    score = cost / overlap
    return score, dG, sim_on_meas, valid


def align_maps(
    meas: dict[str, np.ndarray],
    sim: dict[str, np.ndarray],
    *,
    freq_shift_bounds: tuple[float, float] = (-1.5, 1.5),
    power_shift_bounds: tuple[float, float] = (-6.0, 6.0),
    coarse_freq_step: float = 0.05,
    coarse_power_step: float = 0.25,
    roi_threshold_db: float = 3.0,
    roi_floor: float = 0.05,
    loss: str = "l2",
    huber_delta: float = 2.0,
    fit_freq_range: tuple[float, float] | None = None,
    fit_power_range: tuple[float, float] | None = None,
    min_overlap_frac: float = 0.25,
) -> dict[str, Any]:
    """Fit (df, dP, dG) aligning sim to measurement; return fit + loss surface.

    Coarse grid over (df, dP) with analytic dG, then a fine refine around the
    coarse minimum. dG stays analytic (weighted mean) even for the Huber loss.
    ``fit_freq_range`` / ``fit_power_range`` restrict the fit to one section of
    the measurement map (both measurement-axis windows, inclusive).
    """
    meas_f = meas["pump_freq_ghz"]
    meas_p = meas["pump_power_dbm"]
    meas_g = meas["peak_gain_db"]  # (n_freq, n_power)
    meas_ff, meas_pp = np.meshgrid(meas_f, meas_p, indexing="ij")

    weights = _roi_weights(meas_g, roi_threshold_db, roi_floor, meas_f, meas_p,
                           fit_freq_range, fit_power_range)
    if (weights > 0).sum() < 8:
        raise ValueError(
            "fewer than 8 weighted measurement cells in the fit window "
            f"(freq {fit_freq_range}, power {fit_power_range}); widen it"
        )
    n_ref = float(max((weights > 0).sum(), 1))

    sim_interp = RegularGridInterpolator(
        (sim["pump_freq_ghz"], sim["pump_power_dbm"]),
        sim["gain_db"],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    def scan(df_vals: np.ndarray, dP_vals: np.ndarray
             ) -> tuple[np.ndarray, tuple[int, int]]:
        surf = np.full((df_vals.size, dP_vals.size), np.inf)
        best = (0, 0)
        best_score = np.inf
        for a, df in enumerate(df_vals):
            for b, dP in enumerate(dP_vals):
                s, _, _, _ = _score_shift(
                    df, dP, meas_ff, meas_pp, meas_g, weights,
                    sim_interp, n_ref, loss, huber_delta, min_overlap_frac,
                )
                surf[a, b] = s
                if s < best_score:
                    best_score = s
                    best = (a, b)
        return surf, best

    df_coarse = np.arange(freq_shift_bounds[0], freq_shift_bounds[1] + 1e-9,
                          coarse_freq_step)
    dP_coarse = np.arange(power_shift_bounds[0], power_shift_bounds[1] + 1e-9,
                          coarse_power_step)
    surf_coarse, (ia, ib) = scan(df_coarse, dP_coarse)
    df0, dP0 = float(df_coarse[ia]), float(dP_coarse[ib])

    # Fine refine within +/- 2 coarse steps of the coarse optimum.
    df_fine = np.linspace(df0 - 2 * coarse_freq_step, df0 + 2 * coarse_freq_step, 21)
    dP_fine = np.linspace(dP0 - 2 * coarse_power_step, dP0 + 2 * coarse_power_step, 21)
    surf_fine, (ja, jb) = scan(df_fine, dP_fine)
    df_best, dP_best = float(df_fine[ja]), float(dP_fine[jb])

    score, dG, sim_on_meas, valid = _score_shift(
        df_best, dP_best, meas_ff, meas_pp, meas_g, weights,
        sim_interp, n_ref, loss, huber_delta, min_overlap_frac,
    )
    aligned_sim = sim_on_meas + dG
    resid = np.where(valid, meas_g - aligned_sim, np.nan)
    rms = float(np.sqrt(np.nanmean(resid[valid] ** 2))) if valid.any() else np.nan

    return {
        "freq_shift_ghz": df_best,
        "power_shift_db": dP_best,
        "gain_offset_db": dG,
        "score": float(score),
        "rmse_db": rms,
        "overlap_cells": int(valid.sum()),
        "loss": loss,
        "aligned_sim_db": aligned_sim,  # (n_freq, n_power) on measurement grid
        "residual_db": resid,
        "coarse": {
            "freq_shift_ghz": df_coarse,
            "power_shift_db": dP_coarse,
            "loss_surface": surf_coarse,
        },
        "fine": {
            "freq_shift_ghz": df_fine,
            "power_shift_db": dP_fine,
            "loss_surface": surf_fine,
        },
    }


def plot_measurement_map(meas: dict[str, np.ndarray], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipped measurement plot", flush=True)
        return
    f = meas["pump_freq_ghz"]
    p = meas["pump_power_dbm"]
    g = meas["peak_gain_db"].T  # (n_power, n_freq) for imshow rows=power
    fig, ax = plt.subplots(figsize=(9, 5))
    pcm = ax.pcolormesh(f, p, g, shading="nearest", cmap="magma",
                        vmin=0.0, vmax=max(20.0, float(np.nanmax(g))))
    ax.set_xlabel("pump frequency (GHz)")
    ax.set_ylabel("pump power (dBm)")
    ax.set_title("measured peak gain map")
    fig.colorbar(pcm, ax=ax, label="peak gain (dB)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def plot_comparison(
    meas: dict[str, np.ndarray],
    fit: dict[str, Any],
    out_png: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipped comparison plot", flush=True)
        return
    f = meas["pump_freq_ghz"]
    p = meas["pump_power_dbm"]
    meas_g = meas["peak_gain_db"].T
    sim_g = fit["aligned_sim_db"].T
    resid = fit["residual_db"].T
    gmax = max(20.0, float(np.nanmax(meas_g)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_m, ax_s, ax_r, ax_l = axes.ravel()

    for ax, data, title in (
        (ax_m, meas_g, "measured peak gain"),
        (ax_s, sim_g, "aligned simulation"),
    ):
        pcm = ax.pcolormesh(f, p, data, shading="nearest", cmap="magma",
                            vmin=0.0, vmax=gmax)
        ax.set_xlabel("pump frequency (GHz)")
        ax.set_ylabel("pump power (dBm)")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="gain (dB)")

    rmax = float(np.nanmax(np.abs(resid))) if np.isfinite(resid).any() else 1.0
    pcm = ax_r.pcolormesh(f, p, resid, shading="nearest", cmap="coolwarm",
                          vmin=-rmax, vmax=rmax)
    ax_r.set_xlabel("pump frequency (GHz)")
    ax_r.set_ylabel("pump power (dBm)")
    ax_r.set_title(f"residual (meas - aligned sim), RMS {fit['rmse_db']:.2f} dB")
    fig.colorbar(pcm, ax=ax_r, label="dB")

    cf = fit["coarse"]
    surf = np.where(np.isfinite(cf["loss_surface"]), cf["loss_surface"], np.nan)
    # Non-overlap edge cells have huge loss and would flatten the interior
    # colour scale; clip to a robust range so the real minimum basin is visible.
    finite = surf[np.isfinite(surf)]
    vmin = float(np.nanmin(surf)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 80)) if finite.size else 1.0
    pcm = ax_l.pcolormesh(cf["power_shift_db"], cf["freq_shift_ghz"], surf,
                          shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_l.plot(fit["power_shift_db"], fit["freq_shift_ghz"], "r*", ms=14)
    ax_l.set_xlabel("power shift dP (dB)")
    ax_l.set_ylabel("freq shift df (GHz)")
    ax_l.set_title(
        f"loss surface  (df={fit['freq_shift_ghz']:.3f} GHz, "
        f"dP={fit['power_shift_db']:.2f} dB, dG={fit['gain_offset_db']:.2f} dB)"
    )
    fig.colorbar(pcm, ax=ax_l, label="weighted loss")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map-dir", required=True, type=Path)
    p.add_argument("--measurement-dir", required=True, type=Path)
    p.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    p.add_argument("--freq-shift-bounds", type=float, nargs=2, default=(-1.5, 1.5))
    p.add_argument("--power-shift-bounds", type=float, nargs=2, default=(-6.0, 6.0))
    p.add_argument("--roi-threshold-db", type=float, default=3.0,
                   help="Gain above which a cell gets full fit weight.")
    p.add_argument("--roi-floor", type=float, default=0.05,
                   help="Weight on background (<= threshold) cells.")
    p.add_argument("--loss", choices=("l2", "huber"), default="l2")
    p.add_argument("--huber-delta", type=float, default=2.0)
    p.add_argument("--fit-freq-ghz", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Fit only measurement pump freqs in [LO,HI] GHz.")
    p.add_argument("--fit-power-dbm", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Fit only measurement pump powers in [LO,HI] dBm.")
    p.add_argument("--min-overlap-frac", type=float, default=0.25,
                   help="Reject shifts covering < this fraction of the fit "
                        "region (guards against tiny-overlap corner fits).")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--measurement-plot", type=Path, default=None)
    p.add_argument("--comparison-plot", type=Path, default=None)
    args = p.parse_args(argv)

    meas = load_measurement_map(args.measurement_dir, tuple(args.signal_band_ghz))
    sim = load_sim_map(args.map_dir)

    fit = align_maps(
        meas, sim,
        freq_shift_bounds=tuple(args.freq_shift_bounds),
        power_shift_bounds=tuple(args.power_shift_bounds),
        roi_threshold_db=args.roi_threshold_db,
        roi_floor=args.roi_floor,
        loss=args.loss,
        huber_delta=args.huber_delta,
        fit_freq_range=tuple(args.fit_freq_ghz) if args.fit_freq_ghz else None,
        fit_power_range=tuple(args.fit_power_dbm) if args.fit_power_dbm else None,
        min_overlap_frac=args.min_overlap_frac,
    )

    summary = {
        "map_dir": str(args.map_dir),
        "measurement_dir": str(args.measurement_dir),
        "signal_band_ghz": list(args.signal_band_ghz),
        "fit_freq_ghz": list(args.fit_freq_ghz) if args.fit_freq_ghz else None,
        "fit_power_dbm": list(args.fit_power_dbm) if args.fit_power_dbm else None,
        "loss": args.loss,
        "freq_shift_ghz": fit["freq_shift_ghz"],
        "power_shift_db": fit["power_shift_db"],
        "gain_offset_db": fit["gain_offset_db"],
        "rmse_db": fit["rmse_db"],
        "score": fit["score"],
        "overlap_cells": fit["overlap_cells"],
        "measured_peak_gain_db": float(np.nanmax(meas["peak_gain_db"])),
        "sim_peak_gain_db": float(np.nanmax(sim["gain_db"])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}", flush=True)

    if args.measurement_plot is not None:
        plot_measurement_map(meas, args.measurement_plot)
    if args.comparison_plot is not None:
        plot_comparison(meas, fit, args.comparison_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
