"""Refined rational-locking test for the off-lattice generator.

The input spectra are existing FDTD artifacts.  This module does not launch a
solver.  It replaces the coarse generator grid with an event-based scan on a
refined grid and applies the prominence gate before any rational comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

PUMP_ORDER = 8
SIGNAL_ORDER = 4
GENERATOR_ORDER = 3
WINDOW_BINS = 3.0
BAND_LOW_FRACTION = 0.02
BAND_HIGH_MULTIPLE = 6.0
PROMINENCE_THRESHOLD = 1.5
CAP_RHO = 0.5


def _within_window(freq: np.ndarray, nodes: np.ndarray, window: float) -> np.ndarray:
    """Return the bins within ``window`` of any sorted node."""
    nodes = np.sort(np.asarray(nodes, dtype=float))
    right = np.searchsorted(nodes, freq)
    left = np.clip(right - 1, 0, nodes.size - 1)
    right = np.clip(right, 0, nodes.size - 1)
    return np.minimum(
        np.abs(freq - nodes[left]), np.abs(freq - nodes[right]),
    ) <= window


def _base_nodes(pump_hz: float, signal_hz: float) -> np.ndarray:
    values = np.asarray([
        n * pump_hz + m * signal_hz
        for n in range(-PUMP_ORDER + 2, PUMP_ORDER - 1)
        for m in range(-SIGNAL_ORDER + 1, SIGNAL_ORDER)
    ], dtype=float)
    return np.unique(values)


def _event_scan(
    freq: np.ndarray,
    power: np.ndarray,
    pump_hz: float,
    signal_hz: float,
    window_hz: float,
    n_trials: int,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all generator shares without 20,000 full spectrum scans.

    For each spectral bin, a match is an interval in ``f_a``.  The interval
    centre is obtained from ``s*f = base + order*f_a`` for ``s`` in {-1, 1}.
    Events are accumulated on the exact scan grid, with duplicate node hits
    removed per spectral bin and trial.
    """
    freq = np.asarray(freq, dtype=float)
    power = np.asarray(power, dtype=float)
    rho_grid = np.linspace(0.01, CAP_RHO, n_trials)
    shares = np.zeros(n_trials, dtype=float)
    base = _base_nodes(pump_hz, signal_hz)
    base_nodes = np.abs(base)
    base_nodes = base_nodes[base_nodes > 0.0]
    base_hit = _within_window(freq, base_nodes, window_hz)
    shares += float(power[base_hit].sum())

    # The base lattice is symmetric, so +/- generator orders produce the same
    # absolute node set.  Positive orders are sufficient.
    orders = tuple(range(1, GENERATOR_ORDER + 1))
    grid_step_hz = float((rho_grid[1] - rho_grid[0]) * pump_hz)

    for start in range(0, freq.size, chunk_size):
        stop = min(freq.size, start + chunk_size)
        f_block = freq[start:stop]
        p_block = power[start:stop]
        interval_difference = np.zeros(n_trials + 1, dtype=float)
        for order in orders:
            half_width = window_hz / (order * pump_hz)
            for sign in (-1.0, 1.0):
                centre = (sign * f_block[:, None] - base[None, :]) / (order * pump_hz)
                lower = np.ceil((centre - half_width - 0.01) / (rho_grid[1] - rho_grid[0])).astype(np.int64)
                upper = np.floor((centre + half_width - 0.01) / (rho_grid[1] - rho_grid[0])).astype(np.int64)
                valid = (upper >= 0) & (lower < n_trials)
                rows, bases = np.nonzero(valid)
                lower = np.clip(lower, 0, n_trials - 1)
                upper = np.clip(upper, 0, n_trials - 1)
                np.add.at(interval_difference, lower[rows, bases], p_block[rows])
                np.add.at(interval_difference, upper[rows, bases] + 1, -p_block[rows])
        shares += np.cumsum(interval_difference[:-1])

    total = float(power.sum())
    if total <= 0.0:
        raise ValueError("spectrum has no positive power")
    return rho_grid, shares / total


def nearest_rational(rho: float, max_denominator: int) -> tuple[int, int, float]:
    """Return the nearest reduced rational and relative error."""
    candidates = [
        Fraction(p, q)
        for q in range(1, max_denominator + 1)
        for p in range(1, q + 1)
        if 0.01 <= p / q <= CAP_RHO
    ]
    rational = min(candidates, key=lambda value: abs(float(value) - rho))
    value = float(rational)
    return rational.numerator, rational.denominator, abs(rho - value) / max(abs(rho), 1e-300)


def _find_spectrum(
    source_root: Path, device: str, control_axis: str, control_value: float,
) -> Path | None:
    """Match a CSV row to its existing spectrum by result metadata."""
    for result_path in sorted(source_root.glob(f"{device}/*/result.json")):
        record = json.loads(result_path.read_text(encoding="utf-8"))
        axis = str(record.get("control_axis", "pump_power_dbm"))
        value = record.get("control_value", record.get("pump_power_dbm"))
        if axis == control_axis and value is not None and math.isclose(
            float(value), control_value, rel_tol=0.0, abs_tol=1e-9,
        ):
            spectrum = result_path.parent / "spectrum.npz"
            return spectrum if spectrum.exists() else None
    return None


def _load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path) as data:
        freq = np.asarray(data["frequency_hz"], dtype=float)
        db = np.asarray(data["spectrum_db_relative_pump"], dtype=float)
    finite = np.isfinite(freq) & np.isfinite(db)
    freq, db = freq[finite], db[finite]
    if freq.size < 64:
        raise ValueError(f"spectrum has only {freq.size} finite bins: {path}")
    return freq, 10.0 ** (db / 10.0), float(np.median(np.diff(freq)))


def _null_errors(
    rho_grid: np.ndarray, max_denominator: int, samples: int, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(rho_grid, size=samples, replace=True)
    return np.asarray([
        nearest_rational(float(value), max_denominator)[2] for value in sampled
    ], dtype=float)


def analyse(
    csv_path: Path,
    source_root: Path,
    *,
    n_trials: int = 20_001,
    null_samples: int = 100_000,
    seed: int = 20260816,
    max_scan_bins: int = 1_000,
) -> dict[str, Any]:
    """Run the refined scan and the gated rational-locking test."""
    if n_trials < 20_000:
        raise ValueError("n_trials must be at least 20,000")
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    point_rows: list[dict[str, Any]] = []
    all_grid: np.ndarray | None = None
    for row in rows:
        spectrum = _find_spectrum(
            source_root, row["device"], row["control_axis"], float(row["control_value"]),
        )
        item: dict[str, Any] = {
            "device": row["device"],
            "control_axis": row["control_axis"],
            "control_value": float(row["control_value"]),
            "source_csv": str(csv_path),
            "spectrum_path": str(spectrum) if spectrum else None,
            "status": "NOT_ESTABLISHED",
        }
        if spectrum is None:
            item["reason"] = "matching existing spectrum.npz is missing"
            point_rows.append(item)
            continue
        freq, power, frequency_step = _load_spectrum(spectrum)
        pump_hz = float(row["pump_hz"])
        signal_hz = float(row["signal_hz"])
        band = (
            (freq > BAND_LOW_FRACTION * pump_hz)
            & (freq < BAND_HIGH_MULTIPLE * pump_hz)
        )
        freq, power = freq[band], power[band]
        integer_nodes = np.abs(np.asarray([
            n * pump_hz + m * signal_hz
            for n in range(-PUMP_ORDER, PUMP_ORDER + 1)
            for m in range(-SIGNAL_ORDER, SIGNAL_ORDER + 1)
        ]))
        integer_hit = _within_window(freq, integer_nodes[integer_nodes > 0], WINDOW_BINS * frequency_step)
        off_freq, off_power = freq[~integer_hit], power[~integer_hit]
        if off_power.size == 0 or float(off_power.sum()) <= 0.0:
            item["reason"] = "no positive off-lattice power"
            point_rows.append(item)
            continue
        # The generator prominence is determined by the resolved spectral
        # peaks.  Keeping a bounded set makes the required 20,001-grid scan
        # reproducible on the 159,787-bin JC spectra; the retained fraction is
        # recorded so a low-power continuum cannot be mistaken for evidence.
        if off_power.size > max_scan_bins:
            keep = np.argpartition(off_power, -max_scan_bins)[-max_scan_bins:]
            retained_power_fraction = float(off_power[keep].sum() / off_power.sum())
            off_freq, off_power = off_freq[keep], off_power[keep]
        else:
            retained_power_fraction = 1.0
        rho_grid, scan = _event_scan(
            off_freq, off_power, pump_hz, signal_hz,
            WINDOW_BINS * frequency_step, n_trials,
        )
        best_index = int(np.argmax(scan))
        median_scan = float(np.median(scan))
        rho = float(rho_grid[best_index])
        prominence = float(scan[best_index] / median_scan) if median_scan > 0 else float("inf")
        item.update({
            "status": "SCANNED",
            "rho": rho,
            "best_share": float(scan[best_index]),
            "median_share": median_scan,
            "prominence_best_over_median": prominence,
            "grid_trials": n_trials,
            "grid_step_rho": float(rho_grid[1] - rho_grid[0]),
            "cap_excluded": bool(rho >= CAP_RHO - 3.0 * (rho_grid[1] - rho_grid[0])),
            "generator_gate": "PASS" if prominence >= PROMINENCE_THRESHOLD else "FAIL",
            "scan_bins_used": int(off_power.size),
            "scan_power_fraction_retained": retained_power_fraction,
        })
        if retained_power_fraction < 0.5:
            item["generator_gate"] = "NOT_ESTABLISHED_LOW_RETAINED_POWER"
            item["status"] = "NOT_ESTABLISHED"
            item["reason"] = "bounded peak scan retained less than half of off-lattice power"
        item["rational_q13"] = None
        item["rational_q20"] = None
        if prominence >= PROMINENCE_THRESHOLD and not item["cap_excluded"]:
            item["rational_q13"] = dict(zip(("p", "q", "relative_error"), nearest_rational(rho, 13)))
            item["rational_q20"] = dict(zip(("p", "q", "relative_error"), nearest_rational(rho, 20)))
            item["status"] = "SURVIVES_GATE"
        all_grid = rho_grid
        point_rows.append(item)

    if all_grid is None:
        return {
            "verdict": "INCONCLUSIVE", "reason": "no spectrum could be scanned",
            "points": point_rows, "surviving_point_count": 0,
        }
    null13 = _null_errors(all_grid, 13, null_samples, seed)
    null20 = _null_errors(all_grid, 20, null_samples, seed + 1)
    survivors = [
        item for item in point_rows if item.get("status") == "SURVIVES_GATE"
    ]
    for item in survivors:
        for key, null in (("rational_q13", null13), ("rational_q20", null20)):
            error = float(item[key]["relative_error"])
            item[key]["null_percentile"] = float(np.mean(null <= error))
    p_values = [
        float(item["rational_q13"]["null_percentile"]) for item in survivors
        if item["rational_q13"]["null_percentile"] > 0.0
    ]
    p_values_q20 = [
        float(item["rational_q20"]["null_percentile"]) for item in survivors
        if item["rational_q20"]["null_percentile"] > 0.0
    ]
    if p_values:
        fisher_stat = float(-2.0 * sum(math.log(value) for value in p_values))
        fisher_p = float(chi2.sf(fisher_stat, 2 * len(p_values)))
    else:
        fisher_stat, fisher_p = float("nan"), float("nan")
    if p_values_q20:
        fisher_stat_q20 = float(-2.0 * sum(math.log(value) for value in p_values_q20))
        fisher_p_q20 = float(chi2.sf(fisher_stat_q20, 2 * len(p_values_q20)))
    else:
        fisher_stat_q20, fisher_p_q20 = float("nan"), float("nan")
    if len(survivors) < 3:
        verdict = "INCONCLUSIVE"
    elif fisher_p < 0.05:
        verdict = "LOCKING_SUPPORTED"
    else:
        verdict = "LOCKING_NOT_SUPPORTED"
    return {
        "verdict": verdict,
        "source_csv": str(csv_path),
        "source_root": str(source_root),
        "n_csv_rows": len(rows),
        "n_scanned": sum(item.get("status") in {"SCANNED", "SURVIVES_GATE"} for item in point_rows),
        "surviving_point_count": len(survivors),
        "prominence_gate_effect": {
            "threshold": PROMINENCE_THRESHOLD,
            "scanned_count": sum(item.get("status") in {"SCANNED", "SURVIVES_GATE"} for item in point_rows),
            "surviving_count": len(survivors),
            "excluded_low_prominence": sum(item.get("generator_gate") == "FAIL" for item in point_rows),
            "excluded_cap": sum(bool(item.get("cap_excluded")) for item in point_rows),
        },
        "fisher_q13": {"statistic": fisher_stat, "p_value": fisher_p, "n": len(p_values)},
        "fisher_q20": {"statistic": fisher_stat_q20, "p_value": fisher_p_q20, "n": len(p_values_q20)},
        "null": {
            "samples": null_samples,
            "seed": seed,
            "grid_min_rho": float(all_grid[0]),
            "grid_max_rho": float(all_grid[-1]),
            "q13_relative_error": null13.tolist(),
            "q20_relative_error": null20.tolist(),
        },
        "points": point_rows,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("outputs/chaos/ansatz_validity/ansatz_validity.csv"))
    parser.add_argument("--source-root", type=Path, default=Path("outputs/chaos/phaseB_signal"))
    parser.add_argument("--output", type=Path, default=Path("outputs/chaos/tongues/tongues.json"))
    parser.add_argument("--generator-trials", type=int, default=20_001)
    parser.add_argument("--null-samples", type=int, default=100_000)
    parser.add_argument("--max-scan-bins", type=int, default=1_000)
    args = parser.parse_args(argv)
    result = analyse(
        args.csv, args.source_root, n_trials=args.generator_trials,
        null_samples=args.null_samples, max_scan_bins=args.max_scan_bins,
    )
    _atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "verdict": result["verdict"], "surviving": result.get("surviving_point_count", 0)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
