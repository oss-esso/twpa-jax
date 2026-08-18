"""Quantitative return-map diagnostics for the TWPA chaos campaign.

The campaign stores only ``t`` and ``v_out``.  This module therefore uses a
fixed scalar delay pair instead of a state-space reconstruction.  The primary
coordinate is the within-period pair ``(v(t_n), v(t_n + Delta))``.  The
strobe-to-strobe pair is retained as a secondary return map.

Examples
--------
    python scripts/chaos/return_map.py --validate
    python scripts/chaos/return_map.py --device jc_jtwpa
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lyapunov_kantz import mutual_information_delay  # noqa: E402
from nonlinear_diagnostics import stroboscopic_section  # noqa: E402
from nonlinear_diagnostics import correlation_dimension as nd_correlation_dimension  # noqa: E402


ANALYSIS_VERSION = "return-map-v6"
Q_MAX = 32
# Above this correlation the two section coordinates span a line rather than a
# plane; see the measurement note in ``choose_delay``.
MAX_SECTION_CORRELATION = 0.99
# Relative section radius below which the strobe is a point at the numerical
# floor rather than a resolved attractor; justified against measurement in
# ``describe_section``.
STROBE_FLOOR_RELATIVE = 1.0e-3
# D2 must agree across embedding dimension to count as saturated, and sit near
# one to count as a closed curve; see ``section_dimension``.  Both are set in
# measured gaps rather than assumed.  Golden-mean circle map: D2 = 1.032 with
# spread 0.0077 across m = 2, 3, 4.  Henon: D2 = 1.208 with spread 0.0993
# (literature 1.22, so the estimator is sound).  Saturation SPREAD is the
# stronger of the two discriminators -- a factor of 13 rather than 6 -- because
# an invariant circle has an exact integer dimension and must therefore be
# m-independent, while a strange attractor's estimate drifts upward with m.
D2_SATURATION_SPREAD = 0.05
D2_CIRCLE_TOLERANCE = 0.12
RATIONAL_Q_MAX = 12
CAMPAIGNS: dict[str, Path] = {
    "jc_jtwpa": ROOT / "outputs/chaos/phaseB_jtwpa_2100/jc_jtwpa",
    "ipm_2c_fixed": ROOT / "outputs/chaos/phaseB_2c_gap/ipm_2c_fixed",
    "guarcello": ROOT / "outputs/chaos/phaseB_guarcello_s4/guarcello",
}


@dataclass(frozen=True)
class DelaySelection:
    device: str
    delay_samples: int
    samples_per_period: float
    delta_fraction_of_period: float
    selection_point: str
    criterion: str
    section_correlation: float = float("nan")


@dataclass(frozen=True)
class ClassificationThresholds:
    circle_ratio_threshold: float
    source: str
    period_gap_ratio_threshold: float = 4.0


def _atomic_json(path: Path, value: Any) -> None:
    """Write JSON through a same-directory temporary file and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _point_dirs(root: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir()
            and (p / "result.json").exists()
            and (p / "trace.npz").exists()
        ],
        key=lambda p: float(
            json.loads((p / "result.json").read_text())["control_value"]
        ),
    )


def _load_trace(point_dir: Path) -> tuple[dict[str, Any], np.ndarray]:
    result = json.loads((point_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(point_dir / "trace.npz") as data:
        values = np.asarray(data["v_out"], dtype=np.float64)
    start = int(result.get("steady_state_start_index", 0))
    start = min(max(start, 0), max(values.size - 1, 0))
    return result, values[start:]


def _samples_per_period(result: dict[str, Any]) -> float:
    return 1.0 / (
        float(result["pump_hz"])
        * float(result["dt_s"])
        * float(result.get("record_stride", 1))
    )


def _section_correlation(
    values: np.ndarray, result: dict[str, Any], delay_samples: int
) -> float:
    """Pearson correlation of the two section coordinates at one delay.

    A magnitude near one means the pair spans a line rather than a plane, so
    every geometric descriptor built on it is meaningless.
    """
    _, section, _ = section_coordinates(values, result, delay_samples)
    if section.shape[0] < 3:
        return float("nan")
    z1, z2 = section[:, 0], section[:, 1]
    if np.std(z1) <= 0.0 or np.std(z2) <= 0.0:
        return float("nan")
    return float(np.corrcoef(z1, z2)[0, 1])


def choose_delay(device: str, point_dirs: Iterable[Path]) -> DelaySelection:
    """Choose one AMI delay from the median-control trace for one device."""
    paths = list(point_dirs)
    if not paths:
        raise ValueError(f"no trace points for {device}")
    point = paths[len(paths) // 2]
    result, values = _load_trace(point)
    samples_per_period = _samples_per_period(result)
    max_lag = max(4, min(int(samples_per_period), 512))
    delay = mutual_information_delay(
        values, max_lag=max_lag, bins=32, subsample=40_000
    )
    delay = max(1, min(int(delay), max(1, int(samples_per_period) - 1)))
    criterion = (
        "first local minimum of mutual_information_delay "
        "on median-control trace"
    )

    # The AMI minimum is a statement about the CONTINUOUS trace and carries no
    # guarantee that the resulting SECTION pair spans a plane.  Measured
    # 2026-08-17: on guarcello, whose trace is oversampled at 622.5 samples per
    # pump period, AMI returned 7 samples -- 1.1 percent of a period -- and
    # ``y(t_n)`` and ``y(t_n + Delta)`` came out with |corr| = 0.99985 (45 of 77
    # points above 0.999).  The section was a line, not a plane, so the radius,
    # rotation number, circle test and locking verdict were all computed on a
    # degenerate coordinate.  jc_jtwpa (0.197) and ipm_2c_fixed (0.281) landed
    # near quadrature on their own and are unaffected.
    #
    # Fall back to the quarter period, which is the ``(phi, phi_dot)`` analogue
    # this coordinate is standing in for, whenever AMI's answer degenerates.
    correlation = _section_correlation(values, result, delay)
    if not math.isfinite(correlation) or abs(correlation) > MAX_SECTION_CORRELATION:
        delay = max(1, int(round(0.25 * samples_per_period)))
        fallback = _section_correlation(values, result, delay)
        criterion = (
            f"{criterion}; rejected at |corr| = {correlation:.5f} > "
            f"{MAX_SECTION_CORRELATION}, fell back to the quarter period "
            f"(|corr| = {fallback:.5f})"
        )
        correlation = fallback
    return DelaySelection(
        device=device,
        delay_samples=delay,
        samples_per_period=float(samples_per_period),
        delta_fraction_of_period=float(delay / samples_per_period),
        selection_point=str(point),
        criterion=criterion,
        section_correlation=float(correlation),
    )


def _strobe_indices(n: int, samples_per_period: float) -> np.ndarray:
    n_periods = int((n - 1) / samples_per_period)
    if n_periods < 8:
        return np.empty(0, dtype=np.float64)
    return samples_per_period * np.arange(n_periods, dtype=np.float64)


def section_coordinates(
    values: np.ndarray,
    result: dict[str, Any],
    delay_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strobe values, fixed within-period coordinates, and return map."""
    values = np.asarray(values, dtype=np.float64)
    samples_per_period = _samples_per_period(result)
    indices = _strobe_indices(values.size, samples_per_period)
    if indices.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty((0, 2), dtype=np.float64), np.empty((0, 2))
    axis = np.arange(values.size, dtype=np.float64)
    y0 = np.interp(indices, axis, values)
    y1 = np.interp(indices + float(delay_samples), axis, values)
    section = np.column_stack((y0, y1))
    return y0, section, np.column_stack((y0[:-1], y0[1:]))


def _empirical_floor(values: np.ndarray) -> tuple[float, list[int], str, float]:
    """Find a recurrence floor from the largest measured log-scale gap."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values) & (values >= 0.0)
    clean = values[finite]
    if clean.size == 0:
        return float("nan"), [], "no finite recurrence values", float("nan")
    positive = clean[clean > 0.0]
    if positive.size == 0:
        floor = 0.0
        return floor, list(range(clean.size)), "all values are exactly zero", float("inf")
    tiny = np.finfo(float).tiny
    ordered = np.sort(np.maximum(clean, tiny))
    if ordered.size == 1:
        return float(ordered[0]), [], "single recurrence value", float("nan")
    log_gaps = np.diff(np.log(ordered))
    split = int(np.argmax(log_gaps))
    gap_ratio = float(ordered[split + 1] / ordered[split])
    floor = float(math.sqrt(ordered[split] * ordered[split + 1]))
    floor_indices = np.flatnonzero(values <= floor).astype(int).tolist()
    return (
        floor,
        floor_indices,
        "geometric midpoint at the largest measured log gap",
        gap_ratio,
    )


def _period_candidate(
    d_q: np.ndarray,
    floor: float,
    floor_indices: list[int],
    gap_ratio: float,
    gap_ratio_threshold: float,
) -> int | None:
    if (
        not np.isfinite(floor)
        or not floor_indices
        or gap_ratio < gap_ratio_threshold
    ):
        return None
    floor_set = {q + 1 for q in floor_indices}
    q_min = int(np.argmin(d_q)) + 1
    if q_min not in floor_set:
        q_min = min(floor_set)
    for q in sorted(floor_set):
        multiples = {m for m in floor_set if m > q and m % q == 0}
        if q == 1:
            has_repeat = 2 in floor_set
        else:
            has_repeat = bool(multiples)
        if not has_repeat:
            continue
        prior_nondivisors = [
            d_q[k - 1]
            for k in range(1, q)
            if q % k != 0
        ]
        if prior_nondivisors and max(prior_nondivisors) <= floor:
            continue
        return q
    return None


def _nearest_rational(rho: float) -> tuple[int, int, float]:
    fraction = Fraction(float(rho)).limit_denominator(RATIONAL_Q_MAX)
    return (
        int(fraction.numerator),
        int(fraction.denominator),
        abs(float(rho) - float(fraction)),
    )


def successive_maxima(
    values: np.ndarray, *, max_stored: int = 4000
) -> dict[str, Any]:
    """Lorenz-style return map built from successive local maxima.

    This is an alternative section that does not use the pump phase at all: it
    samples the trajectory wherever the output peaks rather than once per pump
    period.  Because the pump imposes a natural stroboscopic section, this is a
    secondary instrument here, but it is independent of the strobe timebase and
    therefore of every timebase defect that has already bitten this campaign.

    A period-1 orbit gives one repeated amplitude, an invariant circle gives a
    closed curve, and a chaotic set gives structure with a visible fold.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 8:
        return {"status": "NO_DATA", "n_maxima": 0}
    indices, _ = find_peaks(values)
    if indices.size < 4:
        return {"status": "TOO_FEW_MAXIMA", "n_maxima": int(indices.size)}
    amplitudes = values[indices]
    mean = float(np.mean(amplitudes))
    spread = float(np.std(amplitudes))
    # Normalise by half the peak-to-peak range, NOT by the mean.  These are
    # local maxima of an oscillating waveform, so they include negative peaks
    # and the mean sits near zero; dividing by it produced 2.00 / 1.66 / 1.77
    # for period-1 / torus / chaos on jc_jtwpa, i.e. no discrimination at all
    # and a number dominated by how close the mean came to zero.
    span = float(np.ptp(amplitudes))
    scale = 0.5 * span if span > 0.0 else 1.0
    stored = amplitudes
    if stored.size > max_stored:
        stored = stored[:max_stored]
    return {
        "status": "OK",
        "n_maxima": int(amplitudes.size),
        "mean": mean,
        "std": spread,
        "relative_spread": float(spread / scale),
        "amplitudes": [float(v) for v in stored],
    }


def section_dimension(strobe: np.ndarray) -> dict[str, Any]:
    """Test the section for closedness by D2 saturation on the strobe series.

    The roundness ratio ``r_std / r_mean`` answers the wrong question.  It is
    calibrated on a round circle, but a torus section reconstructed from a
    scalar delay pair is a closed CURVE -- generally an ellipse or a Lissajous
    loop -- so a modest eccentricity fails a roundness test even though the set
    is a perfectly good invariant circle.  Measured 2026-08-17 on jc_jtwpa, the
    ratio runs 0.58 -> 1.16 -> 0.16 smoothly across the torus window, which is a
    shape changing rather than a regime changing.

    D2 asks the right question: a fixed point gives 0, an invariant circle 1, a
    chaotic set more.  Saturation across embedding dimension is what separates a
    genuine low-dimensional set from noise, which fills whatever dimension it is
    given -- and at the strobe floor the section IS noise, so a growing D2 there
    is correct rather than a failure.

    Only about 1049 strobes are available per point, so the Eckmann-Ruelle bound
    ``N > 10**D2`` supports D2 up to roughly 3.  Above that the estimator
    correctly returns nothing, which is reported as insufficient data and never
    as a dimension.
    """
    values = np.asarray(strobe, dtype=np.float64)
    per_dimension: dict[str, float | None] = {}
    estimates: list[float] = []
    for dimension in (2, 3, 4):
        measured = nd_correlation_dimension(
            values, dimension, 1, theiler_window=2
        )
        per_dimension[f"m{dimension}"] = (
            None if measured.d2 is None else float(measured.d2)
        )
        if measured.d2 is not None:
            estimates.append(float(measured.d2))

    if len(estimates) < 2:
        return {
            "per_dimension": per_dimension,
            "d2_saturated": None,
            "spread": None,
            "verdict": "INSUFFICIENT_DATA",
        }
    spread = float(max(estimates) - min(estimates))
    mean = float(np.mean(estimates))
    saturated = spread <= D2_SATURATION_SPREAD
    if saturated and abs(mean - 1.0) <= D2_CIRCLE_TOLERANCE:
        verdict = "CLOSED_CURVE"
    elif saturated:
        verdict = "SATURATED_ABOVE_CIRCLE"
    else:
        verdict = "NOT_SATURATED"
    return {
        "per_dimension": per_dimension,
        "d2_saturated": mean if saturated else None,
        "spread": spread,
        "verdict": verdict,
    }


def describe_section(
    section: np.ndarray,
    pump_hz: float,
    *,
    thresholds: ClassificationThresholds,
    strobe: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute recurrence, geometry, rotation, and classification descriptors."""
    x = np.asarray(section, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("section must have shape (n, 2)")
    n = x.shape[0]
    if n == 0:
        return {
            "status": "NO_SECTION",
            "classification": "NOT_ESTABLISHED",
            "d_q": [],
            "q_min": None,
        }
    centroid = np.mean(x, axis=0)
    centered = x - centroid
    radii = np.linalg.norm(centered, axis=1)
    r_rms = float(np.sqrt(np.mean(radii**2)))
    r_mean = float(np.mean(radii))
    r_std = float(np.std(radii))
    spread = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    d_q = np.array(
        [
            float(np.median(np.linalg.norm(x[q:] - x[:-q], axis=1)) / spread)
            if q < n and spread > 0.0
            else 0.0
            for q in range(1, Q_MAX + 1)
        ],
        dtype=np.float64,
    )
    floor, floor_indices, floor_method, floor_gap_ratio = _empirical_floor(d_q)
    q_min = int(np.argmin(d_q)) + 1
    period = _period_candidate(
        d_q,
        floor,
        floor_indices,
        floor_gap_ratio,
        thresholds.period_gap_ratio_threshold,
    )

    # The radius floor must be RELATIVE to the section's own amplitude, not to
    # machine epsilon.  ``eps * max|x|`` is about 1e-20 here, twelve orders of
    # magnitude below the floor the strobing actually has, so every quiescent
    # point passed the gate and -- because ``d_q`` is normalised by the
    # section's own spread, which makes floor noise indistinguishable from a
    # chaotic scatter -- was classified "chaos".  Measured 2026-08-17, that put
    # jc_jtwpa at -30.5 dBm, ipm_2c_fixed at 0.5750 and guarcello at -70 dBm
    # (all period-1 to machine precision by on_comb, K and D2) in the chaotic
    # class, i.e. 129 of 133 points across the three devices.
    #
    # The floor is a property of the strobe interpolation, not of a device: the
    # quiescent relative radius measures 2.9e-4..7.0e-4 on jc_jtwpa,
    # 2.2e-4..3.2e-4 on ipm_2c_fixed and 1.8e-4..2.2e-4 on guarcello, while the
    # first point past each device's independently established onset jumps to
    # 1.4e-3 (jc_jtwpa, -29.45 dBm) and 1.9e-3 (ipm_2c_fixed, 0.5850).  The gate
    # goes in that gap.
    section_scale = float(np.sqrt(np.mean(np.sum(x**2, axis=1))))
    relative_radius = r_rms / section_scale if section_scale > 0.0 else 0.0
    radius_floor = STROBE_FLOOR_RELATIVE * section_scale
    radius_gate = bool(relative_radius > STROBE_FLOOR_RELATIVE)
    rho: float | None = None
    rho_stderr: float | None = None
    fa_hz: float | None = None
    if radius_gate and n > 2:
        angles = np.unwrap(np.arctan2(centered[:, 1], centered[:, 0]))
        increments = np.diff(angles)
        rho_unwrapped = float(np.mean(increments) / (2.0 * math.pi))
        rho = float(rho_unwrapped % 1.0)
        rho_stderr = float(
            np.std(increments, ddof=1)
            / math.sqrt(increments.size)
            / (2.0 * math.pi)
        )
        fa_hz = float(rho * pump_hz)

    rational: dict[str, Any] | None = None
    locking = "NOT_MEANINGFUL"
    if rho is not None:
        numerator, denominator, distance = _nearest_rational(rho)
        collapsed = denominator <= Q_MAX and denominator in {
            q + 1 for q in floor_indices
        }
        rational = {
            "p": numerator,
            "q": denominator,
            "value": float(numerator / denominator),
            "continued_fraction_distance": distance,
            "d_q_collapsed": bool(collapsed),
        }
        distance_limit = max(0.01, 3.0 * float(rho_stderr or 0.0))
        locking = "LOCKED" if distance <= distance_limit and collapsed else "UNLOCKED"

    dimension = (
        section_dimension(strobe)
        if strobe is not None and radius_gate
        else {
            "per_dimension": {},
            "d2_saturated": None,
            "spread": None,
            "verdict": "NOT_EVALUATED",
        }
    )
    round_enough = bool(
        r_mean > 0.0 and r_std / r_mean <= thresholds.circle_ratio_threshold
    )

    if not radius_gate:
        # The section is a point at the strobe floor: a period-1 orbit.  This
        # must be tested before the recurrence spectrum, because d_q is
        # normalised by the section's own spread and therefore reports O(1) for
        # every q on pure floor noise.
        classification = "period-1"
    elif period is not None:
        classification = "period-1" if period == 1 else "period-q"
    elif dimension["verdict"] == "CLOSED_CURVE":
        # D2 saturating near one is the closedness test; roundness is not.
        classification = "torus"
    elif (
        dimension["verdict"] in {"INSUFFICIENT_DATA", "NOT_EVALUATED"}
        and round_enough
    ):
        # Fall back to the roundness ratio only where D2 could not be measured,
        # so a torus is never claimed on a criterion known to be too strict.
        classification = "torus"
    else:
        classification = "chaos"

    return {
        "status": "OK",
        "classification": classification,
        "d_q": [float(v) for v in d_q],
        "d_1": float(d_q[0]),
        "q_min": q_min,
        "period_q": period,
        "recurrence_floor": float(floor),
        "recurrence_floor_indices": [int(q + 1) for q in floor_indices],
        "recurrence_floor_method": floor_method,
        "recurrence_floor_gap_ratio": float(floor_gap_ratio),
        "period_gap_ratio_threshold": thresholds.period_gap_ratio_threshold,
        "centroid": [float(v) for v in centroid],
        "r_RMS": r_rms,
        "r_mean": r_mean,
        "r_std": r_std,
        "r_std_over_r_mean": float(r_std / r_mean) if r_mean > 0.0 else None,
        "radius_floor": float(radius_floor),
        "rotation_gate": "ON" if radius_gate else "OFF",
        "rho": rho,
        "rho_stderr": rho_stderr,
        "f_a_hz": fa_hz,
        "nearest_rational": rational,
        "locking_verdict": locking,
        "circle_ratio_threshold": thresholds.circle_ratio_threshold,
        "circle_threshold_source": thresholds.source,
        "section_dimension": dimension,
        "relative_radius": float(relative_radius),
        "roundness_passes": round_enough,
    }


def _circle_map(rho: float, n: int = 5000) -> np.ndarray:
    indices = np.arange(n, dtype=np.float64)
    return np.column_stack(
        (np.cos(2.0 * math.pi * rho * indices),
         np.sin(2.0 * math.pi * rho * indices))
    )


def _locked_circle(rho: float, repeats: int = 1000) -> np.ndarray:
    """Return an exact finite cycle for the locked-map validation case."""
    return np.tile(_circle_map(rho, 5), (repeats, 1))


def _henon(n: int = 5000) -> np.ndarray:
    x, y = 0.1, 0.1
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n + 100):
        x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
        if i >= 100:
            result[i - 100] = (x, y)
    return result


def run_validation() -> dict[str, Any]:
    """Run the reference suite before any campaign point is measured."""
    preliminary = ClassificationThresholds(1.0, "temporary validation threshold")
    golden = describe_section(
        _circle_map((math.sqrt(5.0) - 1.0) / 2.0),
        1.0,
        thresholds=preliminary,
    )
    henon = describe_section(_henon(), 1.0, thresholds=preliminary)
    circle_threshold = float(
        0.5 * (golden["r_std_over_r_mean"] + henon["r_std_over_r_mean"])
    )
    period_gap_threshold = 2.0 * max(
        float(golden["recurrence_floor_gap_ratio"]),
        float(henon["recurrence_floor_gap_ratio"]),
    )
    thresholds = ClassificationThresholds(
        circle_threshold,
        "midpoint of measured golden-mean circle and Hénon radius ratios",
        period_gap_threshold,
    )
    cases: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        "fixed_point": (np.zeros((1000, 2)), {"q": 1, "rho_off": True}),
        "period_2": (np.tile([[1.0, 0.0], [-1.0, 0.0]], (500, 1)), {"q": 2}),
        "period_5": (np.tile(_circle_map(1.0 / 5.0, 5), (200, 1)), {"q": 5}),
        "circle_map_golden_mean": (
            _circle_map((math.sqrt(5.0) - 1.0) / 2.0),
            {"rho": (math.sqrt(5.0) - 1.0) / 2.0},
        ),
        "circle_map_locked_2_over_5": (
            _locked_circle(2.0 / 5.0),
            {"locked": True},
        ),
        "henon": (_henon(), {"no_period": True, "chaotic": True}),
    }
    results: dict[str, Any] = {}
    for name, (sequence, expectation) in cases.items():
        # Drive the D2 closedness test from the reference's own first
        # coordinate, so the references gate that estimator too rather than
        # leaving it to run for the first time on device data.
        measured = describe_section(
            sequence, 1.0, thresholds=thresholds, strobe=sequence[:, 0]
        )
        passed = True
        if "q" in expectation:
            passed &= measured["q_min"] == expectation["q"]
        if expectation.get("rho_off"):
            passed &= measured["rotation_gate"] == "OFF"
        if "rho" in expectation:
            passed &= abs(float(measured["rho"]) - expectation["rho"]) < 0.01
            passed &= measured["classification"] == "torus"
        if expectation.get("locked"):
            passed &= measured["locking_verdict"] == "LOCKED"
            passed &= measured["q_min"] == 5
        if expectation.get("no_period"):
            passed &= measured["period_q"] is None
        if expectation.get("chaotic"):
            # Checking only "no period found" is too weak: it let Henon be
            # labelled a torus once the D2 closedness test was added, because
            # D2(Henon) = 1.21 is close enough to one to pass a loose
            # tolerance.  Assert the label itself.
            passed &= measured["classification"] == "chaos"
        results[name] = {
            "passed": bool(passed),
            "expected": expectation,
            "measured": measured,
        }
    return {
        "status": "PASS" if all(v["passed"] for v in results.values()) else "FAIL",
        "thresholds": asdict(thresholds),
        "cases": results,
    }


def _lyapunov_rows(device: str) -> list[dict[str, Any]]:
    path = ROOT / "outputs/chaos/lyapunov_kantz" / f"{device}.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else value.get("points", [])


def _cross_check(device: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point = {Path(str(r["point_dir"])).name: r for r in records}
    rows: list[dict[str, Any]] = []
    for spectral in _lyapunov_rows(device):
        generator = spectral.get("second_generator", {})
        if generator.get("verdict") != "TORUS":
            continue
        point = by_point.get(Path(str(spectral.get("point_dir", ""))).name)
        if point is None:
            rows.append(
                {
                    "spectral_point": spectral.get("point_dir"),
                    "status": "NO_MATCH",
                }
            )
            continue
        rho = point.get("descriptors", {}).get("rho")
        spectral_rho = generator.get("generator_ratio_to_pump")
        if rho is None or spectral_rho is None:
            rows.append({"control_value": point["control_value"], "status": "NO_RHO"})
            continue
        candidates = {
            "rho": float(rho),
            "1-rho": float(1.0 - rho),
            "rho+1": float(rho + 1.0),
            "rho-1": float(rho - 1.0),
        }
        branch, candidate = min(
            candidates.items(), key=lambda item: abs(item[1] - spectral_rho)
        )
        rows.append(
            {
                "control_value": point["control_value"],
                "point_dir": point["point_dir"],
                "geometric_rho": float(rho),
                "spectral_rho": float(spectral_rho),
                "matched_branch": branch,
                "absolute_difference": float(abs(candidate - spectral_rho)),
                "status": "MATCHED",
            }
        )
    return rows


def analyse_device(
    device: str,
    *,
    output_root: Path = ROOT / "outputs/chaos/return_map",
    force: bool = False,
) -> dict[str, Any]:
    """Analyse one device and checkpoint every point atomically."""
    if device not in CAMPAIGNS:
        raise ValueError(f"unsupported device {device!r}")
    point_dirs = _point_dirs(CAMPAIGNS[device])
    delay = choose_delay(device, point_dirs)
    validation = run_validation()
    if validation["status"] != "PASS":
        raise RuntimeError("validation gate failed; no device point was analysed")
    thresholds = ClassificationThresholds(**validation["thresholds"])
    checkpoint_root = output_root / "_points" / device
    records: list[dict[str, Any]] = []
    for point_dir in point_dirs:
        checkpoint = checkpoint_root / f"{point_dir.name}.json"
        if checkpoint.exists() and not force:
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                record.get("analysis_version") == ANALYSIS_VERSION
                and record.get("delay_samples") == delay.delay_samples
            ):
                records.append(record)
                continue
        result, values = _load_trace(point_dir)
        strobe, section, return_map = section_coordinates(
            values, result, delay.delay_samples
        )
        descriptor = describe_section(
            section, float(result["pump_hz"]), thresholds=thresholds, strobe=strobe
        )
        maxima = successive_maxima(values)
        record = {
            "analysis_version": ANALYSIS_VERSION,
            "device": device,
            "point_dir": str(point_dir),
            "control_value": result.get("control_value"),
            "control_axis": result.get("control_axis"),
            "pump_hz": result.get("pump_hz"),
            "steady_state_start_index": result.get("steady_state_start_index"),
            "delay_samples": delay.delay_samples,
            "delta_fraction_of_period": delay.delta_fraction_of_period,
            "n_strobe": int(strobe.size),
            "strobe_values": [float(v) for v in strobe],
            "return_map_strobe_to_strobe": return_map.tolist(),
            "section_coordinate": "within_period_pair",
            "section_z1_z2": section.tolist(),
            "descriptors": descriptor,
            "successive_maxima": maxima,
        }
        _atomic_json(checkpoint, record)
        records.append(record)
    records.sort(key=lambda r: float(r["control_value"]))
    output = {
        "analysis_version": ANALYSIS_VERSION,
        "device": device,
        "source_root": str(CAMPAIGNS[device]),
        "source_trace_fields": ["t", "v_out"],
        "delay_selection": asdict(delay),
        "validation": validation,
        "points": records,
        "spectral_cross_check": _cross_check(device, records),
        "not_computable_from_stored_data": {
            "global_state_space_pca": (
                "requires an FDTD rerun recording the full state once per pump period"
            ),
            "effective_linear_dimension_N99": "requires the same full-state FDTD rerun",
            "per_node_sigma": "requires the same full-state FDTD rerun",
            "participation_ratio": "requires the same full-state FDTD rerun",
            "tangent_space_largest_lyapunov_exponent": (
                "requires the integrator kernel and state; do not infer it "
                "from trace.npz"
            ),
        },
    }
    _atomic_json(output_root / f"{device}.json", output)
    return output


def _print_validation(result: dict[str, Any]) -> None:
    print(f"validation: {result['status']}")
    for name, case in result["cases"].items():
        measured = case["measured"]
        print(
            f"{name}: {'PASS' if case['passed'] else 'FAIL'} "
            f"classification={measured.get('classification')} "
            f"q_min={measured.get('q_min')} rho={measured.get('rho')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=sorted(CAMPAIGNS))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/chaos/return_map"
    )
    args = parser.parse_args(argv)
    if args.validate:
        validation = run_validation()
        _print_validation(validation)
        if not args.device:
            return 0 if validation["status"] == "PASS" else 1
    devices = [args.device] if args.device else list(CAMPAIGNS)
    for device in devices:
        result = analyse_device(device, output_root=args.output_root, force=args.force)
        print(
            f"{device}: {len(result['points'])} points; "
            f"cross-check={len(result['spectral_cross_check'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
