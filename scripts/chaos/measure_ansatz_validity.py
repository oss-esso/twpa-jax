"""Measure how much of an FDTD waveform a harmonic-balance ansatz can represent.

Harmonic balance solves for amplitudes on a fixed frequency lattice.  Whatever
power the true waveform carries away from that lattice is invisible to it, no
matter how well the Newton solve converges, so the fraction of spectral power
sitting on the lattice is an upper bound on any harmonic-balance result at that
operating point.  The time-domain kernel assumes no lattice at all, which is
what makes it able to measure this.

Three questions are answered per operating point:

``on_lattice``
    Share of in-band power within one analysis window of a node
    ``n*f_p + m*f_s``.  This is the ceiling for the production multitone basis.

``on_half`` / ``on_third``
    The same with half-integer and third-integer pump nodes admitted.  A
    period-doubled state would lift ``on_half`` close to one; if it does not,
    a period-2 basis cannot recover the missing power and should not be built.

``top20`` / ``generator_share``
    Structure of what is left over.  Discrete off-lattice content concentrates
    into few bins and fits a single extra generator ``f_a``, which is the
    signature of a torus and is representable by a three-frequency balance.
    A raised continuum does neither and is representable by no balance at all.

Usage::

    python -m scripts.chaos.measure_ansatz_validity outputs/chaos/phaseB_signal
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# Devices whose campaign records predate the recorded ``signal_hz`` field.
# The values are ``derive_device_spec``'s, in GHz.
LEGACY_SIGNAL_GHZ = {"jc_jtwpa": 6.62, "jc_fqjtwpa": 7.40, "ipm_2c_fixed": 7.40}

# Lattice extent.  The pump harmonic range covers the analysis band six times
# over; the signal range covers the sideband orders the production basis keeps.
PUMP_ORDER = 8
SIGNAL_ORDER = 4
GENERATOR_ORDER = 3
WINDOW_BINS = 3.0
BAND_LOW_FRACTION = 0.02
BAND_HIGH_MULTIPLE = 6.0


def _nodes(pump_hz: float, signal_hz: float, pump_divisor: int) -> np.ndarray:
    """Positive lattice frequencies for ``n/pump_divisor * f_p + m * f_s``."""
    orders = range(-PUMP_ORDER * pump_divisor, PUMP_ORDER * pump_divisor + 1)
    grid = np.array([
        (n / pump_divisor) * pump_hz + m * signal_hz
        for n in orders
        for m in range(-SIGNAL_ORDER, SIGNAL_ORDER + 1)
    ])
    grid = np.abs(grid)
    return np.unique(grid[grid > 0.0])


def _within_window(freq: np.ndarray, nodes: np.ndarray, window_hz: float) -> np.ndarray:
    """Boolean mask of the frequencies lying within ``window_hz`` of a node."""
    sorted_nodes = np.sort(nodes)
    right = np.searchsorted(sorted_nodes, freq)
    left = np.clip(right - 1, 0, sorted_nodes.size - 1)
    right = np.clip(right, 0, sorted_nodes.size - 1)
    distance = np.minimum(np.abs(freq - sorted_nodes[left]),
                          np.abs(freq - sorted_nodes[right]))
    return distance <= window_hz


def _best_generator(
    freq: np.ndarray, power: np.ndarray, pump_hz: float, signal_hz: float,
    window_hz: float, n_trials: int,
) -> tuple[float, float]:
    """Scan one extra generator and return its best (share, f_a/f_p).

    The scan is over ``f_a`` alone rather than a joint fit because the question
    is only whether *some* single extra frequency organizes the residue; the
    precise value matters less than whether a value exists at all.
    """
    base = np.array([
        n * pump_hz + m * signal_hz
        for n in range(-PUMP_ORDER + 2, PUMP_ORDER - 1)
        for m in range(-SIGNAL_ORDER + 1, SIGNAL_ORDER)
    ])
    total = float(power.sum())
    if total <= 0.0 or freq.size == 0:
        return 0.0, float("nan")
    orders = np.arange(-GENERATOR_ORDER, GENERATOR_ORDER + 1)
    best_share, best_fa = 0.0, float("nan")
    for f_a in np.linspace(0.01 * pump_hz, 0.5 * pump_hz, n_trials):
        nodes = np.abs((base[:, None] + f_a * orders[None, :]).ravel())
        nodes = nodes[nodes > 0.0]
        share = float(power[_within_window(freq, nodes, window_hz)].sum() / total)
        if share > best_share:
            best_share, best_fa = share, f_a / pump_hz
    return best_share, best_fa


def analyse_point(point_dir: Path, n_generator_trials: int) -> dict[str, Any] | None:
    """Return the ansatz-validity statistics for one campaign point."""
    result_path = point_dir / "result.json"
    spectrum_path = point_dir / "spectrum.npz"
    if not result_path.exists() or not spectrum_path.exists():
        return None
    record = json.loads(result_path.read_text(encoding="utf-8"))
    pump_hz = float(record.get("pump_hz") or 0.0)
    signal_hz = float(record.get("signal_hz") or 0.0)
    if signal_hz <= 0.0:
        legacy = LEGACY_SIGNAL_GHZ.get(str(record.get("device", "")))
        signal_hz = legacy * 1e9 if legacy else 0.0
    if pump_hz <= 0.0 or signal_hz <= 0.0:
        return None

    data = np.load(spectrum_path)
    freq = data["frequency_hz"]
    magnitude_db = data["spectrum_db_relative_pump"]
    band = (
        (freq > BAND_LOW_FRACTION * pump_hz)
        & (freq < BAND_HIGH_MULTIPLE * pump_hz)
        & np.isfinite(magnitude_db)
    )
    freq, magnitude_db = freq[band], magnitude_db[band]
    if freq.size < 64:
        return None
    power = 10.0 ** (magnitude_db / 10.0)
    total = float(power.sum())
    window_hz = WINDOW_BINS * float(np.median(np.diff(freq)))

    integer_mask = _within_window(freq, _nodes(pump_hz, signal_hz, 1), window_hz)
    on_lattice = float(power[integer_mask].sum() / total)
    on_half = float(
        power[_within_window(freq, _nodes(pump_hz, signal_hz, 2), window_hz)].sum()
        / total
    )
    on_third = float(
        power[_within_window(freq, _nodes(pump_hz, signal_hz, 3), window_hz)].sum()
        / total
    )

    off_power = power[~integer_mask]
    off_total = float(off_power.sum())
    top20 = (float(np.sort(off_power)[::-1][:20].sum() / off_total)
             if off_total > 0.0 else float("nan"))
    generator_share, generator_ratio = _best_generator(
        freq[~integer_mask], off_power, pump_hz, signal_hz, window_hz,
        n_generator_trials,
    )

    return {
        "device": record.get("device"),
        "control_axis": record.get("control_axis"),
        "control_value": record.get("control_value"),
        "gain_vs_off_db": record.get("gain_vs_off_db"),
        "branch_current_max_over_ic": record.get("r_j"),
        "min_cos_phi": record.get("min_cos_phi"),
        "pump_hz": pump_hz,
        "signal_hz": signal_hz,
        "on_lattice": on_lattice,
        "on_half": on_half,
        "on_third": on_third,
        "off_lattice": 1.0 - on_lattice,
        "top20_of_off_lattice": top20,
        "generator_share": generator_share,
        "generator_over_pump": generator_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        help="campaign directory holding <device>/<point>/ dirs")
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the full table here as well as printing it")
    parser.add_argument("--generator-trials", type=int, default=400,
                        help="number of f_a values in the extra-generator scan")
    args = parser.parse_args()

    rows = [
        row
        for point in sorted(args.root.glob("*/*/"))
        if (row := analyse_point(point, args.generator_trials)) is not None
    ]
    if not rows:
        print(f"no reducible points under {args.root}")
        return 1
    rows.sort(key=lambda r: (str(r["device"]), float(r["control_value"] or 0.0)))

    device = None
    for row in rows:
        if row["device"] != device:
            device = row["device"]
            print("=" * 100)
            print(f"{device}   ({row['control_axis']})")
            print(f"{'control':>10} {'gain dB':>8} {'I/Ic':>7} {'mincos':>7} "
                  f"{'lattice':>8} {'+half':>8} {'+third':>8} "
                  f"{'top20':>7} {'gen':>7} {'fa/fp':>7}")
        print(
            f"{row['control_value']:>10.3f} "
            f"{_fmt(row['gain_vs_off_db']):>8} {_fmt(row['branch_current_max_over_ic'], 4):>7} "
            f"{_fmt(row['min_cos_phi'], 3):>7} "
            f"{row['on_lattice']:>8.4f} {row['on_half']:>8.4f} {row['on_third']:>8.4f} "
            f"{row['top20_of_off_lattice']:>7.3f} {row['generator_share']:>7.3f} "
            f"{row['generator_over_pump']:>7.4f}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


def _fmt(value: Any, digits: int = 2) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
