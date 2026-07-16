"""Compare a simulated 2c gain map to the Themis 15 mK transmission measurement.

The measurement (docs/17.03.10_Themis_SetupAug25_noVTS_transmission_15mK) is a
cube of transmission (dB) over 51 pump frequencies x 31 pump powers x 2001
signal frequencies. The 2c model reproduces it only after a known calibration
offset: pump frequency shifted by ~+0.99 GHz and pump power by a few dB (see
diagnostics/2c_measurement_comparison). This script does NOT re-fit that
offset; it applies the offsets given on the CLI and overlays two reduced views
so the qualitative agreement (peak-gain wave, high-power collapse envelope) can
be read directly:

  1. peak gain vs pump frequency (measured vs simulated), and
  2. the high-power boundary vs pump frequency -- the measured gain-collapse
     power and the simulated last-converged power -- which is where the model's
     numerical/fold boundary should line up with the physical collapse.

Simulated gains come from a run_gain_map output dir (map_arrays.npz:
gain_db_warm[power, freq]); NaN entries are failed cells and set the sim
boundary. Writes a JSON summary and, if matplotlib is present, a two-panel PNG.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

FREQ_RE = re.compile(r"105C5_([0-9.]+)GHz\.npy$")


def load_measurement(meas_dir: Path, signal_band_ghz: tuple[float, float]
                     ) -> dict[str, Any]:
    """Reduce the measurement cube to peak gain[freq, power] over a signal band."""
    files = sorted(meas_dir.glob("105C5_*GHz.npy"))
    if not files:
        raise FileNotFoundError(f"no measurement npy files in {meas_dir}")
    freqs: list[float] = []
    peak_rows: list[np.ndarray] = []
    powers_ref: np.ndarray | None = None
    lo, hi = signal_band_ghz
    for f in files:
        m = FREQ_RE.search(f.name)
        if not m:
            continue
        d = np.load(f, allow_pickle=True).item()
        sig = np.asarray(d["Frequency"], dtype=float) / 1e9  # Hz -> GHz
        resp = np.asarray(d["Response"], dtype=float)         # (n_power, n_sig)
        powers = np.asarray(d["PumpPower"], dtype=float)      # (n_power,)
        band = (sig >= lo) & (sig <= hi)
        peak = resp[:, band].max(axis=1)                      # peak over signal
        freqs.append(float(m.group(1)))
        peak_rows.append(peak)
        powers_ref = powers if powers_ref is None else powers_ref
    order = np.argsort(freqs)
    return {
        "pump_freq_ghz": np.asarray(freqs)[order],
        "pump_power_dbm": np.asarray(powers_ref, dtype=float),
        "peak_gain_db": np.asarray(peak_rows)[order],  # (n_freq, n_power)
    }


def collapse_power(peak_gain_db: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Per-frequency pump power just before the largest adjacent gain drop."""
    out = np.full(peak_gain_db.shape[0], np.nan)
    for i in range(peak_gain_db.shape[0]):
        g = peak_gain_db[i]
        d = np.diff(g)
        if d.size:
            out[i] = powers[int(np.argmin(d))]  # power at the steepest drop
    return out


def load_sim(map_dir: Path) -> dict[str, Any]:
    arr = np.load(map_dir / "map_arrays.npz")
    gain = arr["gain_db_warm"]          # (n_power, n_freq), NaN where failed
    return {
        "pump_power_dbm": arr["pump_power_dbm"].astype(float),
        "pump_freq_ghz": arr["pump_frequency_ghz"].astype(float),
        "gain_db": gain.astype(float),
    }


def sim_boundary_power(gain_db: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Per-frequency highest pump power with a converged (non-NaN) gain cell."""
    out = np.full(gain_db.shape[1], np.nan)
    for j in range(gain_db.shape[1]):
        ok = np.where(np.isfinite(gain_db[:, j]))[0]
        if ok.size:
            out[j] = powers[ok.max()]
    return out


def sim_peak_gain(gain_db: np.ndarray) -> np.ndarray:
    """Per-frequency peak converged gain over the power axis."""
    out = np.full(gain_db.shape[1], np.nan)
    for j in range(gain_db.shape[1]):
        col = gain_db[:, j]
        if np.isfinite(col).any():
            out[j] = np.nanmax(col)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-dir", required=True, type=Path)
    p.add_argument("--measurement-dir", required=True, type=Path)
    p.add_argument("--freq-shift-ghz", type=float, default=0.99,
                   help="Added to measured pump freq to compare with sim.")
    p.add_argument("--power-shift-db", type=float, default=-4.3,
                   help="Added to measured pump power to compare with sim "
                        "(sim on-chip power is below measured external).")
    p.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--plot", type=Path, default=None)
    args = p.parse_args(argv)

    meas = load_measurement(args.measurement_dir, tuple(args.signal_band_ghz))
    sim = load_sim(args.map_dir)

    meas_freq_aligned = meas["pump_freq_ghz"] + args.freq_shift_ghz
    meas_collapse = collapse_power(meas["peak_gain_db"], meas["pump_power_dbm"])
    meas_collapse_aligned = meas_collapse + args.power_shift_db
    meas_peak = meas["peak_gain_db"].max(axis=1)

    sim_boundary = sim_boundary_power(sim["gain_db"], sim["pump_power_dbm"])
    sim_peak = sim_peak_gain(sim["gain_db"])
    n_cells = sim["gain_db"].size
    n_ok = int(np.isfinite(sim["gain_db"]).sum())

    summary = {
        "map_dir": str(args.map_dir),
        "measurement_dir": str(args.measurement_dir),
        "freq_shift_ghz": args.freq_shift_ghz,
        "power_shift_db": args.power_shift_db,
        "signal_band_ghz": list(args.signal_band_ghz),
        "sim_coverage_frac": n_ok / n_cells,
        "sim_converged_cells": n_ok,
        "sim_total_cells": n_cells,
        "measured": {
            "pump_freq_ghz": meas["pump_freq_ghz"].tolist(),
            "pump_freq_ghz_aligned": meas_freq_aligned.tolist(),
            "peak_gain_db": meas_peak.tolist(),
            "collapse_power_dbm": meas_collapse.tolist(),
            "collapse_power_dbm_aligned": meas_collapse_aligned.tolist(),
        },
        "simulated": {
            "pump_freq_ghz": sim["pump_freq_ghz"].tolist(),
            "peak_gain_db": [None if not np.isfinite(x) else float(x) for x in sim_peak],
            "boundary_power_dbm": [None if not np.isfinite(x) else float(x) for x in sim_boundary],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"sim coverage {n_ok}/{n_cells} = {n_ok / n_cells:.1%}")
    print(f"measured peak gain: {np.nanmin(meas_peak):.1f}..{np.nanmax(meas_peak):.1f} dB")
    if np.isfinite(sim_peak).any():
        print(f"sim peak gain:      {np.nanmin(sim_peak):.1f}..{np.nanmax(sim_peak):.1f} dB")
    print(f"wrote {args.out}")

    if args.plot is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipped plot")
            return 0
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        ax1.plot(meas_freq_aligned, meas_peak, "o-", ms=3, label="measured (aligned)")
        ax1.plot(sim["pump_freq_ghz"], sim_peak, "s-", ms=3, label="simulated")
        ax1.set_ylabel("peak gain (dB)")
        ax1.legend()
        ax1.set_title(f"peak gain vs pump freq  (freq+{args.freq_shift_ghz} GHz, "
                      f"pow{args.power_shift_db:+g} dB)")
        ax2.plot(meas_freq_aligned, meas_collapse_aligned, "o-", ms=3,
                 label="measured collapse power (aligned)")
        ax2.plot(sim["pump_freq_ghz"], sim_boundary, "s-", ms=3,
                 label="sim last-converged power")
        ax2.set_ylabel("pump power (dBm)")
        ax2.set_xlabel("pump frequency (GHz, sim axis)")
        ax2.legend()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
