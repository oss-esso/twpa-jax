"""Quantify estimator sensitivity of the measured 2c P1dB slope."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


MEASUREMENT = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
INPUT_LOSS_DB = 72.5
PUMP_GHZ = 7.256
PUMP_GUARD_GHZ = 0.15


@dataclass(frozen=True)
class Variant:
    name: str
    g0_estimator: str = "median_plateau"
    window: int = 11
    crossing: str = "current_last"
    band: str = "full"
    gain_floor: float = 8.0


def variants() -> list[Variant]:
    base = Variant("baseline")
    return [
        base,
        Variant("g0_lowest_point", g0_estimator="lowest_point"),
        Variant("g0_lowest_five_mean", g0_estimator="lowest_five_mean"),
        Variant("window_7", window=7),
        Variant("window_21", window=21),
        Variant("first_downward", crossing="first_downward"),
        Variant("band_below_pump", band="below_pump"),
        Variant("band_above_pump", band="above_pump"),
        Variant("gain_floor_4db", gain_floor=4.0),
        Variant("gain_floor_12db", gain_floor=12.0),
    ]


def fit(gain: np.ndarray, p1db: np.ndarray) -> tuple[float, float, float, float]:
    """Return slope, intercept, RMS residual, and slope standard error."""
    slope, intercept = np.polyfit(gain, p1db, 1)
    residual = p1db - (slope * gain + intercept)
    rms = float(np.sqrt(np.mean(residual**2)))
    dof = max(gain.size - 2, 1)
    sxx = float(np.sum((gain - np.mean(gain)) ** 2))
    standard_error = float(np.sqrt(np.sum(residual**2) / dof / sxx))
    return float(slope), float(intercept), rms, standard_error


def crossing(pin: np.ndarray, signal: np.ndarray, target: float,
             rule: str) -> float | None:
    """Interpolate a threshold crossing in ascending input-power order."""
    delta = signal - target
    downward = np.flatnonzero((delta[:-1] >= 0.0) & (delta[1:] < 0.0))
    upward = np.flatnonzero((delta[:-1] < 0.0) & (delta[1:] >= 0.0))
    if rule == "first_downward":
        indices = downward
    elif rule == "current_last":
        # The historical estimator selected the final row above threshold
        # and interpolated to the following row. In the measured power order
        # this is the last downward threshold transition.
        indices = downward
    else:
        raise ValueError(f"unknown crossing rule: {rule}")
    if indices.size == 0:
        return None
    index = int(indices[0] if rule == "first_downward" else indices[-1])
    if pin[index + 1] == pin[index] or signal[index + 1] == signal[index]:
        return None
    return float(
        pin[index]
        + (target - signal[index])
        * (pin[index + 1] - pin[index])
        / (signal[index + 1] - signal[index])
    )


def g0_value(signal: np.ndarray, pin: np.ndarray, estimator: str) -> float:
    """Estimate the small-signal plateau gain for one frequency column."""
    plateau = signal[pin < -120.0]
    if estimator == "median_plateau":
        return float(np.median(plateau))
    if estimator == "lowest_point":
        return float(signal[0])
    if estimator == "lowest_five_mean":
        return float(np.mean(signal[:5]))
    raise ValueError(f"unknown gain estimator: {estimator}")


def band_mask(freq: np.ndarray, band: str) -> np.ndarray:
    """Select the requested 5.34--8.95 GHz measurement band."""
    selected = (freq >= 5.34) & (freq <= 8.95)
    selected &= np.abs(freq - PUMP_GHZ) >= PUMP_GUARD_GHZ
    if band == "below_pump":
        selected &= freq < PUMP_GHZ - PUMP_GUARD_GHZ
    elif band == "above_pump":
        selected &= freq > PUMP_GHZ + PUMP_GUARD_GHZ
    elif band != "full":
        raise ValueError(f"unknown frequency band: {band}")
    return selected


def analyze_variant(data: dict[str, np.ndarray], variant: Variant) -> dict[str, object]:
    """Compute one slope sensitivity row."""
    freq = np.asarray(data["Frequency"], dtype=float) / 1e9
    pin = np.asarray(data["SignalPower"], dtype=float) - INPUT_LOSS_DB
    response = np.asarray(data["Response"], dtype=float)
    gains: list[float] = []
    p1dbs: list[float] = []
    selected = band_mask(freq, variant.band)
    for index in np.flatnonzero(selected):
        smooth = savgol_filter(response[:, index], variant.window, 2)
        g0 = g0_value(smooth, pin, variant.g0_estimator)
        if g0 <= variant.gain_floor:
            continue
        p1db = crossing(pin, smooth, g0 - 1.0, variant.crossing)
        if p1db is not None and math.isfinite(p1db):
            gains.append(g0)
            p1dbs.append(p1db)
    gain_array = np.asarray(gains)
    p1db_array = np.asarray(p1dbs)
    slope, intercept, rms, standard_error = fit(gain_array, p1db_array)
    return {
        "variant": variant.name,
        "g0_estimator": variant.g0_estimator,
        "window": variant.window,
        "crossing": variant.crossing,
        "band": variant.band,
        "gain_floor_db": variant.gain_floor,
        "n": int(gain_array.size),
        "slope_db_per_db": slope,
        "slope_se_db_per_db": standard_error,
        "intercept_dbm": intercept,
        "fit_rms_db": rms,
        "gain_min_db": float(np.min(gain_array)),
        "gain_max_db": float(np.max(gain_array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, default=MEASUREMENT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp25_track1_measured_slope")
    )
    args = parser.parse_args()
    data = np.load(args.measurement, allow_pickle=True).item()
    rows = [analyze_variant(data, variant) for variant in variants()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "slope_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "slope_sensitivity.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    slopes = np.asarray([float(row["slope_db_per_db"]) for row in rows])
    print(f"n variants={len(rows)} min={slopes.min():.6f} max={slopes.max():.6f} "
          f"spread={slopes.max() - slopes.min():.6f}")
    for row in rows:
        print(
            f"{row['variant']:>20s} n={row['n']:3d} "
            f"slope={row['slope_db_per_db']:+.6f} "
            f"se={row['slope_se_db_per_db']:.6f} rms={row['fit_rms_db']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
