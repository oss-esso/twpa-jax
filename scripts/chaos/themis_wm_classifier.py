#!/usr/bin/env python3
"""Calibration-free Wiesenfeld--McNamara analysis of Themis response cubes."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ResponseCube:
    path: str
    frequency_hz: np.ndarray
    response: np.ndarray
    pump_power_dbm: np.ndarray
    signal_power_dbm: float


@dataclass(frozen=True)
class PeakTrack:
    pump_frequency_hz: float
    collapse_index: int
    power_indices: tuple[int, ...]
    peak_frequency_hz: tuple[float, ...]
    peak_response_db: tuple[float, ...]
    collapse_power_dbm: float


@dataclass(frozen=True)
class DivergenceFit:
    exponent: float
    r_squared: float
    fitted_threshold_dbm: float
    points: int
    status: str


def _scalar(value: object) -> float:
    array = np.asarray(value).reshape(-1)
    return float(array[0])


def load_cube(path: str | Path) -> ResponseCube:
    """Load one Themis ``.npy`` object while preserving its stated axes."""
    source = Path(path)
    payload = np.load(source, allow_pickle=True).item()
    required = {"Frequency", "Response", "PumpPower", "SignalPower"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{source}: missing keys {sorted(missing)}")
    frequency = np.asarray(payload["Frequency"], dtype=float).reshape(-1)
    response = np.asarray(payload["Response"], dtype=float)
    pump_power = np.asarray(payload["PumpPower"], dtype=float).reshape(-1)
    if response.shape != (pump_power.size, frequency.size):
        raise ValueError(f"{source}: Response shape {response.shape} does not match axes")
    if frequency.size < 2 or pump_power.size < 2:
        raise ValueError(f"{source}: cube axes are too short")
    return ResponseCube(str(source), frequency, response, pump_power, _scalar(payload["SignalPower"]))


def load_cubes(directory: str | Path) -> list[ResponseCube]:
    return [load_cube(path) for path in sorted(Path(directory).glob("*.npy"))]


def peak_track(cube: ResponseCube, *, min_gain_db: float = 8.0,
               collapse_drop_db: float = 3.0) -> PeakTrack:
    """Track the pre-collapse resonance at each amplifying power point."""
    response = np.asarray(cube.response, dtype=float)
    row_peak = np.max(response, axis=1)
    amplifying = np.flatnonzero(row_peak >= min_gain_db)
    if amplifying.size == 0:
        amplifying = np.flatnonzero(np.isfinite(row_peak))
    collapse = int(amplifying[-1]) if amplifying.size else int(np.argmax(row_peak))
    # If a later point drops sharply, the last pre-collapse point is the one
    # immediately before that drop; this avoids calling a flat gain floor a
    # resonance.
    drops = np.flatnonzero(row_peak[1:] < row_peak[:-1] - collapse_drop_db)
    if drops.size:
        candidate = int(drops[0])
        before = amplifying[amplifying <= candidate]
        if before.size:
            collapse = int(before[-1])
    indices = amplifying[amplifying <= collapse]
    peaks = np.argmax(response[indices], axis=1)
    collapse_power_index = min(collapse + 1, cube.pump_power_dbm.size - 1)
    return PeakTrack(
        pump_frequency_hz=_pump_frequency(cube.path),
        collapse_index=collapse, power_indices=tuple(int(x) for x in indices),
        peak_frequency_hz=tuple(float(cube.frequency_hz[x]) for x in peaks),
        peak_response_db=tuple(float(response[i, x]) for i, x in zip(indices, peaks)),
        collapse_power_dbm=float(cube.pump_power_dbm[collapse_power_index]),
    )


def _pump_frequency(path: str) -> float:
    """Extract the leading pump frequency from common Themis filenames."""
    import re
    matches = re.findall(r"(?<![0-9])([0-9]+(?:\.[0-9]+)?)\s*GHz", Path(path).stem, re.I)
    return float(matches[0]) * 1e9 if matches else float("nan")


def classify_track(track: PeakTrack, *, measured_low_hz: float = 4e9,
                   measured_high_hz: float = 12e9, tolerance: float = 0.05) -> str:
    """Apply the band-aware W-M frequency predictions to a peak track."""
    if len(track.peak_frequency_hz) < 2:
        return "UNDETERMINED: insufficient amplifying points"
    pump = track.pump_frequency_hz
    if not np.isfinite(pump) or pump <= 0.0:
        return "UNDETERMINED: pump frequency not encoded in input"
    peaks = np.asarray(track.peak_frequency_hz)
    in_band_half = [k * pump / 2.0 for k in (1, 3, 5) if measured_low_hz <= k * pump / 2.0 <= measured_high_hz]
    if in_band_half and np.min([np.min(np.abs(peaks - target)) for target in in_band_half]) <= tolerance * pump:
        return "PERIOD_DOUBLING"
    # A drifting peak not tied to an integer or half harmonic is the measured
    # signature of a sideband set by an imaginary critical exponent.
    integer_or_half = np.concatenate((np.arange(1, 8) * pump, np.arange(1, 8) * pump / 2.0))
    residual = np.min(np.abs(peaks[:, None] - integer_or_half[None, :]), axis=1) / pump
    if float(np.ptp(peaks)) > 0.01 * pump and float(np.mean(residual)) > tolerance:
        return "NEIMARK_SACKER"
    return "UNDETERMINED: measured band does not resolve a Wiesenfeld-McNamara signature"


def fit_gain_divergence(cube: ResponseCube, track: PeakTrack | None = None,
                        *, min_gain_db: float = 8.0) -> DivergenceFit:
    """Fit ``G ~ (P_c-P)^(-alpha)`` on the amplifying pre-collapse points.

    The fit is a diagnostic secondary to the frequency classifier.  It uses
    linear power gain (not dB), excludes the first post-collapse point, and
    reports the threshold obtained independently from a linear fit of
    ``1/sqrt(G)``.  A coarse 0.335 dB grid is explicitly represented by the
    point count and can legitimately return ``UNDETERMINED``.
    """
    track = track or peak_track(cube, min_gain_db=min_gain_db)
    row_peak_db = np.max(cube.response, axis=1)
    stop = min(track.collapse_index + 1, row_peak_db.size)
    indices = np.arange(stop, dtype=int)
    indices = indices[row_peak_db[:stop] >= min_gain_db]
    if indices.size < 3:
        return DivergenceFit(float("nan"), float("nan"), float("nan"), int(indices.size),
                             "UNDETERMINED: fewer than three amplifying points")
    powers = cube.pump_power_dbm[indices]
    gains = np.power(10.0, row_peak_db[indices] / 10.0)
    positive = np.isfinite(powers) & np.isfinite(gains) & (gains > 1.0)
    powers, gains = powers[positive], gains[positive]
    if powers.size < 3:
        return DivergenceFit(float("nan"), float("nan"), float("nan"), int(powers.size),
                             "UNDETERMINED: invalid gain samples")
    # The first point after the last pre-collapse point is the measured upper
    # bound.  Fit its threshold freely as a consistency check.
    x0 = float(track.collapse_power_dbm)
    distance = x0 - powers
    keep = distance > 0.0
    if np.count_nonzero(keep) < 3:
        return DivergenceFit(float("nan"), float("nan"), float("nan"), int(np.count_nonzero(keep)),
                             "UNDETERMINED: collapse grid has no positive distances")
    log_distance = np.log(distance[keep])
    log_gain = np.log(gains[keep])
    slope, intercept = np.polyfit(log_distance, log_gain, 1)
    fitted = slope * log_distance + intercept
    residual = float(np.sum((log_gain - fitted) ** 2))
    total = float(np.sum((log_gain - np.mean(log_gain)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    inv_sqrt = 1.0 / np.sqrt(gains[keep])
    threshold_slope, threshold_intercept = np.polyfit(powers[keep], inv_sqrt, 1)
    threshold = float(-threshold_intercept / threshold_slope) if threshold_slope != 0.0 else float("nan")
    return DivergenceFit(float(-slope), r_squared, threshold, int(log_gain.size), "OK")


def analyze_cubes(cubes: Iterable[ResponseCube]) -> dict:
    records = []
    verdicts = []
    for cube in cubes:
        track = peak_track(cube)
        pump = _pump_frequency(cube.path)
        track = PeakTrack(pump, track.collapse_index, track.power_indices,
                          track.peak_frequency_hz, track.peak_response_db,
                          track.collapse_power_dbm)
        verdict = classify_track(track)
        divergence = fit_gain_divergence(cube, track)
        records.append({"cube": cube.path, "pump_frequency_hz": pump,
                        "track": asdict(track), "divergence_fit": asdict(divergence),
                        "verdict": verdict})
        verdicts.append(verdict.split(":", 1)[0])
    return {"records": records, "aggregate_verdict": (
        verdicts[0] if verdicts and len(set(verdicts)) == 1 else "UNDETERMINED: cube verdicts disagree"
    )}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = analyze_cubes(load_cubes(args.directory))
    encoded = json.dumps(result, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
