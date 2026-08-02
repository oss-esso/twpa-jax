"""Quantify shared-G0 noise effects in the measured P1dB slope."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


MEASUREMENT = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
PUMP_GHZ = 7.256
INPUT_LOSS_DB = 72.5
N_REPLICATES = 200


def fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Fit slope, intercept, RMS, and slope standard error."""
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    sxx = float(np.sum((x - x.mean()) ** 2))
    return {
        "n": int(x.size),
        "slope_db_per_db": float(slope),
        "slope_se_db_per_db": math.sqrt(float(np.sum(residual**2)) / max(x.size - 2, 1) / sxx),
        "intercept_dbm": float(intercept),
        "fit_rms_db": float(np.sqrt(np.mean(residual**2))),
    }


def crossing(power: np.ndarray, smooth: np.ndarray, g0: float) -> float | None:
    """Use the historical last-above-threshold crossing rule."""
    above = np.flatnonzero(smooth >= g0 - 1.0)
    if above.size == 0 or int(above[-1]) >= smooth.size - 1:
        return None
    index = int(above[-1])
    if smooth[index + 1] == smooth[index]:
        return None
    return float(
        power[index]
        + (g0 - 1.0 - smooth[index])
        * (power[index + 1] - power[index])
        / (smooth[index + 1] - smooth[index])
    )


def load_curves(path: Path) -> dict[str, np.ndarray]:
    """Load and smooth all non-pump measurement columns."""
    data = np.load(path, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - INPUT_LOSS_DB
    response = np.asarray(data["Response"], dtype=float)
    selected = np.abs(frequency - PUMP_GHZ) >= 0.15
    frequencies: list[float] = []
    gains: list[float] = []
    p1dbs: list[float] = []
    lowest: list[float] = []
    plateau_sigma: list[float] = []
    smoothed_curves: list[np.ndarray] = []
    for index in np.flatnonzero(selected):
        smooth = savgol_filter(response[:, index], 11, 2)
        plateau = smooth[power < -120.0]
        g0 = float(np.median(plateau))
        p1db = crossing(power, smooth, g0)
        if p1db is None or g0 <= 8.0:
            continue
        frequencies.append(float(frequency[index]))
        gains.append(g0)
        p1dbs.append(p1db)
        lowest.append(float(smooth[0]))
        raw_plateau = response[power < -120.0, index]
        plateau_sigma.append(float(np.std(raw_plateau - smooth[power < -120.0])))
        smoothed_curves.append(smooth)
    return {
        "frequency_ghz": np.asarray(frequencies),
        "power_dbm": power,
        "g0": np.asarray(gains),
        "p1db": np.asarray(p1dbs),
        "lowest": np.asarray(lowest),
        "plateau_sigma": np.asarray(plateau_sigma),
        "smooth": np.asarray(smoothed_curves),
    }


def split_sample(curves: dict[str, np.ndarray]) -> dict[str, object]:
    """Estimate G0 on even powers and crossing on independent odd powers."""
    data = curves
    power = data["power_dbm"]
    even_power = power[::2]
    odd_power = power[1::2]
    even_gains: list[float] = []
    odd_p1db: list[float] = []
    frequencies: list[float] = []
    for frequency, curve in zip(data["frequency_ghz"], data["smooth"]):
        even_smooth = savgol_filter(curve[::2], 11, 2)
        odd_smooth = savgol_filter(curve[1::2], 11, 2)
        g0 = float(np.median(even_smooth[even_power < -120.0]))
        p1db = crossing(odd_power, odd_smooth, g0)
        if p1db is not None and g0 > 8.0:
            frequencies.append(float(frequency))
            even_gains.append(g0)
            odd_p1db.append(p1db)
    result = fit(np.asarray(even_gains), np.asarray(odd_p1db))
    result["frequencies_ghz"] = frequencies
    return result


def synthetic_injection(curves: dict[str, np.ndarray], seed: int) -> dict[str, float]:
    """Inject plateau-sized noise into G0 while holding crossings fixed."""
    rng = np.random.default_rng(seed)
    noisy_g0 = curves["g0"] + rng.normal(0.0, curves["plateau_sigma"])
    return fit(noisy_g0, curves["p1db"])


def independent_axes(curves: dict[str, np.ndarray]) -> dict[str, object]:
    """Fit with target and gain axes taken from different estimators."""
    p1_plateau_target = curves["p1db"]
    axis_lowest = fit(curves["lowest"], p1_plateau_target)
    lowest_target: list[float] = []
    for smooth in curves["smooth"]:
        p1 = crossing(curves["power_dbm"], smooth, float(smooth[0]))
        lowest_target.append(float("nan") if p1 is None else p1)
    lowest_target_array = np.asarray(lowest_target)
    finite = np.isfinite(lowest_target_array)
    plateau_axis = fit(curves["g0"][finite], lowest_target_array[finite])
    return {
        "axis_lowest_target_plateau": axis_lowest,
        "axis_plateau_target_lowest": plateau_axis,
        "n_target_lowest": int(finite.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, default=MEASUREMENT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp26_track2_shared_noise"))
    args = parser.parse_args()
    curves = load_curves(args.measurement)
    baseline = fit(curves["g0"], curves["p1db"])
    split = split_sample(curves)
    injection = [synthetic_injection(curves, seed) for seed in range(N_REPLICATES)]
    injection_slopes = np.asarray([float(row["slope_db_per_db"]) for row in injection])
    independent = independent_axes(curves)
    report = {
        "baseline": baseline,
        "split_sample_even_G0_odd_crossing": split,
        "synthetic_injection": {
            "replicates": N_REPLICATES,
            "plateau_sigma_median_db": float(np.median(curves["plateau_sigma"])),
            "plateau_sigma_mean_db": float(np.mean(curves["plateau_sigma"])),
            "slope_mean_db_per_db": float(injection_slopes.mean()),
            "slope_std_db_per_db": float(injection_slopes.std(ddof=1)),
            "induced_slope_change_mean_db_per_db": float(injection_slopes.mean() - baseline["slope_db_per_db"]),
            "induced_slope_change_std_db_per_db": float(injection_slopes.std(ddof=1)),
        },
        "independent_gain_axes": independent,
        "window": "full 4-12 GHz measurement, excluding pump guard, G0 > 8 dB",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shared_noise_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [
        {"variant": "baseline", **baseline},
        {"variant": "split_sample", **split},
        {"variant": "independent_axis_lowest_target_plateau", **independent["axis_lowest_target_plateau"]},
        {"variant": "independent_axis_plateau_target_lowest", **independent["axis_plateau_target_lowest"]},
    ]
    with (args.output_dir / "shared_noise_fits.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
