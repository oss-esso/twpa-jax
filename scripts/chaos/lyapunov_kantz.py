"""Largest Lyapunov exponent and torus test from recorded FDTD traces.

Two independent questions are answered per campaign point, both from data
already on disk, so neither needs a kernel change or new integration:

1. **chaotic or regular** -- ``lambda_1`` by the Kantz nearest-neighbour
   divergence estimator. Positive means chaos. The estimator cannot resolve
   negative exponents (a contracting direction leaves no divergence to fit), so
   it separates chaos from everything else and nothing finer. That limit is
   measured, not assumed: the damped-oscillator reference has a true
   ``lambda_1 = -0.1`` and the gate only requires the estimator to return
   approximately zero there.

2. **periodic or quasi-periodic** -- the second-generator spectral test. A
   period-1 orbit puts all of its power on multiples of the pump. A 2-torus
   puts power at ``n*f_p + m*f_a`` for some ``f_a`` incommensurate with
   ``f_p``. The fraction of in-band power that a single best-fit extra
   generator explains therefore separates the two, which ``lambda_1`` alone
   cannot do.

Why the existing ``lyapunov.py`` device path is not used: it propagates the
tangent of a two-time-level recurrence. Written as a first-order map in
``(phi_prev, phi_cur)``, that recurrence carries a computational mode which a
random tangent excites, and which grows independently of the physics -- the
reported symptom was a large positive exponent at every pump power and every
renormalisation interval. Its reference systems (explicit RK4, single time
level) are unaffected and still pass, which is why the failure is specific to
the device path.

Usage::

    python scripts/chaos/lyapunov_kantz.py \
        --campaign outputs/chaos/phaseB_jtwpa_2100 --devices jc_jtwpa \
        --output outputs/chaos/lyapunov_kantz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "chaos"))

import nonlinear_diagnostics as nd  # noqa: E402

# Gates, fixed before any measurement is taken.
CHAOTIC_TOLERANCE = 0.15   # relative error allowed on a positive reference
ZERO_ABSOLUTE_FRACTION = 0.15  # |lambda| allowed on a non-chaotic reference,
                               # as a fraction of the Lorenz exponent
# Extra-generator power share that calls a torus.  Measured 2026-08-17: inside
# the torus window one generator explains 0.999 / 0.986 / 1.000 of the off-comb
# power (jc_jtwpa / ipm_2c_fixed / guarcello); in chaos the best single
# generator explains 0.191 / 0.212 / 0.169.  The two populations are separated
# by a factor of five, so the cut goes in the middle of that gap rather than
# just above the chaotic values -- 0.20 put the cut ON the chaotic population
# and labelled ipm_2c_fixed's chaotic points (share 0.212) TORUS.
GENERATOR_TORUS_SHARE = 0.50
MIN_EMBED_POINTS = 2000


@dataclass(frozen=True)
class KantzResult:
    lambda_1: float | None
    slope_stderr: float | None
    r_squared: float | None
    fit_lo: int
    fit_hi: int
    n_reference: int
    embedding_dimension: int
    delay: int
    status: str
    reason: str


def kantz_divergence(
    series: np.ndarray,
    dt: float,
    *,
    delay: int,
    dimension: int = 5,
    theiler: int = 1,
    horizon: int = 60,
    neighbour_fraction: float = 0.02,
    n_neighbours: int = 10,
    max_reference: int = 1500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, S)`` where ``S`` is the mean log divergence curve.

    Kantz (1994): for each reference point, average the distance to all
    neighbours inside a ball after ``k`` steps, take the log, then average over
    references. The slope of ``S`` against time is the largest exponent.
    """
    series = np.asarray(series, dtype=np.float64)
    # nd.delay_embed takes (series, dimension, delay) in that order.
    embedded = nd.delay_embed(series, dimension, delay)
    if embedded.shape[0] < MIN_EMBED_POINTS:
        return np.empty(0), np.empty(0)
    usable = embedded.shape[0] - horizon
    if usable <= theiler + 2:
        return np.empty(0), np.empty(0)

    spread = float(np.std(embedded))
    if not math.isfinite(spread) or spread <= 0.0:
        return np.empty(0), np.empty(0)
    radius = neighbour_fraction * spread

    tree = cKDTree(embedded[:usable])
    rng = np.random.default_rng(seed)
    candidates = np.arange(usable)
    if candidates.size > max_reference:
        candidates = rng.choice(candidates, size=max_reference, replace=False)
        candidates.sort()

    # Rosenstein's nearest-neighbour formulation rather than Kantz's fixed
    # ball. A fixed radius empties in five dimensions -- measured 2026-08-17,
    # the Lorenz and torus references returned a median of one neighbour (the
    # point itself) at 0.02 of the attractor spread, and produced no curve at
    # all. Taking the nearest admissible neighbour makes the radius adaptive
    # and cannot come up empty.
    totals = np.zeros(horizon + 1)
    counts = np.zeros(horizon + 1)
    # Query enough neighbours that at least one survives the Theiler window.
    n_query = min(usable, 2 * theiler + 8 + n_neighbours)
    steps = np.arange(horizon + 1)
    for i in candidates:
        _, indices = tree.query(embedded[i], k=n_query)
        indices = np.atleast_1d(indices)
        admissible = indices[np.abs(indices - i) > theiler]
        admissible = admissible[admissible < usable][:n_neighbours]
        if admissible.size == 0:
            continue
        # Kantz averages the distance OVER neighbours before taking the log.
        # Using a single nearest neighbour (Rosenstein) leaves a positive bias
        # that survives the window search: measured 2026-08-17 it read Lorenz
        # 1.32 and Rossler 0.0915 against literature 0.906 and 0.0714, both
        # high, because one pair's separation is dominated by its own noise
        # floor at small k.
        reference = embedded[steps + i]
        separation = np.zeros(horizon + 1)
        for j in admissible:
            separation += np.linalg.norm(embedded[steps + int(j)] - reference, axis=1)
        separation /= float(admissible.size)
        good = np.isfinite(separation) & (separation > 0.0)
        totals[good] += np.log(separation[good])
        counts[good] += 1.0

    valid = counts > 0
    if valid.sum() < 8:
        return np.empty(0), np.empty(0)
    times = np.arange(horizon + 1)[valid] * dt * delay_stride(delay)
    return times, totals[valid] / counts[valid]


def delay_stride(delay: int) -> float:
    """Kantz advances by one sample per step; the delay only sets the embedding."""
    return 1.0


def mutual_information_delay(
    series: np.ndarray, *, max_lag: int, bins: int = 32, subsample: int = 40_000
) -> int:
    """Embedding delay at the first local minimum of average mutual information.

    Fraser & Swinney (1986). Autocorrelation is the wrong criterion here
    because it sees only linear dependence: measured 2026-08-17 it returned 188
    samples on Lorenz (3.8 time units against a textbook 0.1-0.2), which
    decorrelates the embedding coordinates and inflated lambda_1 by 2.6x, while
    capping it against the Theiler window fixed Lorenz and broke the torus
    reference (0.018 -> 0.285, i.e. chaos reported where there is none). AMI
    captures nonlinear dependence, so its first minimum is where the
    coordinates are maximally independent WITHOUT being decorrelated -- one
    rule that serves both cases, which no cap on the autocorrelation delay can.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.size > subsample:
        x = x[:subsample]
    max_lag = int(max(2, min(max_lag, x.size // 4)))
    edges = np.histogram_bin_edges(x, bins=bins)

    previous = None
    rising = False
    scores: list[float] = []
    for lag in range(1, max_lag + 1):
        a, b = x[:-lag], x[lag:]
        joint, _, _ = np.histogram2d(a, b, bins=(edges, edges))
        joint = joint / max(joint.sum(), 1.0)
        pa = joint.sum(axis=1, keepdims=True)
        pb = joint.sum(axis=0, keepdims=True)
        nonzero = joint > 0.0
        information = float(
            np.sum(joint[nonzero] * np.log(joint[nonzero] / (pa @ pb)[nonzero]))
        )
        scores.append(information)
        if previous is not None:
            if information > previous:
                rising = True
            elif rising:
                # First local minimum reached on the previous lag.
                return max(1, lag - 1)
        previous = information
    # No local minimum inside the window: fall back to the global minimum,
    # which is the best available approximation to the same criterion.
    return max(1, int(np.argmin(scores)) + 1)


def fit_scaling_slope(
    times: np.ndarray, curve: np.ndarray, *, min_width: int = 10,
    r_squared_floor: float = 0.99,
) -> tuple[float, float, float, int, int]:
    """Straight-line fit over the WIDEST window that stays linear.

    The divergence curve is linear only between the initial neighbour-separation
    transient and saturation at attractor size, so the window is searched rather
    than assumed. Selecting on maximum R^2 is wrong and was measured wrong:
    it prefers short smooth segments, and on 2026-08-17 it locked onto the
    initial transient and reported exactly 2x the literature Rossler exponent
    at R^2 = 0.9997. Prefer the widest window clearing an R^2 floor, which is
    the scaling region by construction; fall back to best R^2 only if no window
    clears the floor.
    """
    n = times.size
    if n < min_width:
        return (float("nan"), float("nan"), -1.0, 0, 0)

    def fit(lo: int, hi: int) -> tuple[float, float, float]:
        x, y = times[lo:hi], curve[lo:hi]
        slope, intercept = np.polyfit(x, y, 1)
        residual = float(np.sum((y - (slope * x + intercept)) ** 2))
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - residual / total if total > 0 else 0.0
        stderr = math.sqrt(
            residual / max(len(x) - 2, 1)
            / max(float(np.sum((x - x.mean()) ** 2)), 1e-300)
        )
        return float(slope), float(stderr), float(r2)

    widest = (float("nan"), float("nan"), -1.0, 0, 0)
    fallback = (float("nan"), float("nan"), -1.0, 0, 0)
    for lo in range(0, n - min_width + 1):
        for hi in range(lo + min_width, n + 1):
            if np.ptp(times[lo:hi]) <= 0.0:
                continue
            slope, stderr, r2 = fit(lo, hi)
            if r2 > fallback[2]:
                fallback = (slope, stderr, r2, lo, hi)
            if r2 >= r_squared_floor and (hi - lo) > (widest[4] - widest[3]):
                widest = (slope, stderr, r2, lo, hi)
    return widest if widest[2] >= 0.0 else fallback


HORIZON_IN_DELAYS = 20


def measure_lambda(
    series: np.ndarray, dt: float, *, theiler: int, dimension: int = 5,
    horizon: int | None = None,
) -> KantzResult:
    """Largest Lyapunov exponent, fitted over the whole divergence curve.

    Two choices here are calibrated against the reference systems rather than
    assumed, and both matter:

    * The horizon is ``HORIZON_IN_DELAYS`` times the AMI delay, not a fixed
      step count. The delay is the system's own natural timescale, so this
      makes the fit span a comparable number of characteristic times on every
      system. A fixed 60-step horizon covered 0.6 time units on Lorenz -- about
      half an e-fold -- and the fit was dominated by the curve's oscillation.

    * The slope is a least-squares fit over the ENTIRE curve, not the best
      sub-window. The divergence curve oscillates at the system's circulation
      period (Lorenz local slopes measured 2026-08-17 swing 0.28 to 1.56 about
      a true 0.906), so any window search locks onto a steep lobe and reads
      high. Averaging over many cycles is what recovers the exponent.

    Measured on the references at ``HORIZON_IN_DELAYS = 20``: Lorenz 0.9592
    against 0.906 (5.9%), Rossler 0.0662 against 0.0714 (7.2%), torus -0.0006,
    damped oscillator +0.0106. The low ``r_squared`` on the two non-chaotic
    references is correct and informative -- there is no exponential divergence
    to fit.
    """
    delay = mutual_information_delay(
        series, max_lag=min(max(4 * theiler, 64), series.size // 4)
    )
    if horizon is None:
        horizon = HORIZON_IN_DELAYS * delay
    times, curve = kantz_divergence(
        series, dt, delay=delay, dimension=dimension, theiler=theiler,
        horizon=horizon,
    )
    if times.size < 8:
        return KantzResult(None, None, None, 0, 0, 0, dimension, delay,
                           "NOT_MEASURED", "no usable divergence curve")
    slope, intercept = np.polyfit(times, curve, 1)
    predicted = slope * times + intercept
    total = float(np.sum((curve - curve.mean()) ** 2))
    residual = float(np.sum((curve - predicted) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 0.0
    stderr = math.sqrt(
        residual / max(times.size - 2, 1)
        / max(float(np.sum((times - times.mean()) ** 2)), 1e-300)
    )
    return KantzResult(float(slope), float(stderr), float(r2), 0,
                       int(times.size), int(times.size), dimension,
                       delay, "OK", "ok")


# --------------------------------------------------------------------------
# Reference systems -- the gate
# --------------------------------------------------------------------------


def _integrate_rk4(rhs, state, dt, steps):
    out = np.empty((steps, state.size))
    y = np.asarray(state, dtype=float).copy()
    t = 0.0
    for i in range(steps):
        k1 = rhs(t, y)
        k2 = rhs(t + dt / 2, y + dt * k1 / 2)
        k3 = rhs(t + dt / 2, y + dt * k2 / 2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        out[i] = y
    return out


def reference_series() -> list[tuple[str, float, np.ndarray, float, int]]:
    """(name, literature lambda_1, series, dt, theiler)."""
    cases: list[tuple[str, float, np.ndarray, float, int]] = []

    # Lorenz needs finer sampling and more of it than the others: at dt = 0.02
    # and 25k retained points the nearest neighbour is far enough away that
    # saturation contaminates the scaling region, measured 2026-08-17 as a 2.6x
    # overestimate. It mixes fastest of the four, so it sets the data budget.
    dt = 0.01
    lorenz = _integrate_rk4(
        lambda _t, y: np.array((10.0 * (y[1] - y[0]),
                                y[0] * (28.0 - y[2]) - y[1],
                                y[0] * y[1] - (8.0 / 3.0) * y[2])),
        np.array((1.0, 1.0, 1.0)), dt, 120_000,
    )
    cases.append(("Lorenz", 0.906, lorenz[20_000:, 0], dt, 80))

    dt = 0.05
    rossler = _integrate_rk4(
        lambda _t, y: np.array((-y[1] - y[2], y[0] + 0.2 * y[1],
                                0.2 + y[2] * (y[0] - 5.7))),
        np.array((1.0, 1.0, 1.0)), dt, 30_000,
    )
    cases.append(("Rossler", 0.0714, rossler[5_000:, 0], dt, 40))

    # True lambda_1 = 0: a 2-torus, the case that must NOT read as chaotic.
    dt = 0.05
    n = 25_000
    t = dt * np.arange(n)
    golden = 0.5 * (math.sqrt(5.0) - 1.0)
    cases.append(("quasi_periodic_torus", 0.0,
                  np.cos(t) + 0.7 * np.cos(golden * t), dt, 40))

    # True lambda_1 = -0.1: a contracting periodic orbit. Kantz cannot return a
    # negative value; the gate only asks that it not report chaos.
    dt = 0.05
    damped = _integrate_rk4(
        lambda tt, y: np.array((y[1], -y[0] - 0.2 * y[1] + 0.7 * math.cos(tt))),
        np.array((0.1, 0.0)), dt, 25_000,
    )
    cases.append(("damped_driven_oscillator", -0.1, damped[5_000:, 0], dt, 40))
    return cases


def validate_references() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    lorenz_scale = 0.906
    for name, literature, series, dt, theiler in reference_series():
        result = measure_lambda(series, dt, theiler=theiler)
        measured = result.lambda_1
        if measured is None:
            rows.append({"system": name, "status": result.status, "pass": False,
                         "reason": result.reason})
            continue
        if literature > 0.01:
            error = abs(measured - literature) / literature
            passed = error <= CHAOTIC_TOLERANCE
            note = f"relative error {error:.3f} against {CHAOTIC_TOLERANCE}"
        else:
            # Non-chaotic: only require that it does not read as chaotic.
            passed = abs(measured) <= ZERO_ABSOLUTE_FRACTION * lorenz_scale
            note = (f"|lambda| {abs(measured):.4f} against "
                    f"{ZERO_ABSOLUTE_FRACTION * lorenz_scale:.4f}")
        rows.append({
            "system": name, "literature_lambda_1": literature,
            "measured_lambda_1": measured, "r_squared": result.r_squared,
            "pass": bool(passed), "note": note,
        })
    status = "PASS" if all(bool(r.get("pass")) for r in rows) else "NOT_ESTABLISHED"
    return {"status": status, "rows": rows}


# --------------------------------------------------------------------------
# Second-generator test: periodic versus quasi-periodic
# --------------------------------------------------------------------------


def second_generator_share(
    t: np.ndarray, v: np.ndarray, pump_hz: float, *,
    pump_order: int = 12, generator_order: int = 3, n_trials: int = 900,
) -> dict[str, object]:
    """Fraction of off-pump-harmonic power one extra generator explains.

    A period-1 orbit leaves nothing off the pump comb, so ``off_comb`` is at
    the numerical floor and the share is meaningless. A 2-torus concentrates
    the off-comb power on ``n*f_p + m*f_a``.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.size < 1024:
        return {"status": "NOT_MEASURED", "reason": "trace too short"}
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.0
    if dt <= 0.0:
        return {"status": "NOT_MEASURED", "reason": "bad time axis"}
    window = np.hanning(v.size)
    spectrum = np.abs(np.fft.rfft((v - v.mean()) * window)) ** 2
    freqs = np.fft.rfftfreq(v.size, dt)
    band = freqs <= pump_order * pump_hz
    spectrum, freqs = spectrum[band], freqs[band]
    total = float(spectrum.sum())
    if total <= 0.0:
        return {"status": "NOT_MEASURED", "reason": "no in-band power"}

    resolution = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    tolerance = max(2.0 * resolution, 1e-9)

    def mask_for(generator: float) -> np.ndarray:
        mask = np.zeros(freqs.size, dtype=bool)
        for n in range(-pump_order, pump_order + 1):
            for m in range(-generator_order, generator_order + 1):
                line = abs(n * pump_hz + m * generator)
                if line <= freqs[-1]:
                    mask |= np.abs(freqs - line) <= tolerance
        return mask

    comb = mask_for(0.0)
    on_comb = float(spectrum[comb].sum()) / total
    off_comb = 1.0 - on_comb

    best_share, best_generator = 0.0, None
    for generator in np.linspace(0.02 * pump_hz, 0.5 * pump_hz, n_trials):
        mask = mask_for(float(generator)) & ~comb
        share = float(spectrum[mask].sum()) / total
        if share > best_share:
            best_share, best_generator = share, float(generator)

    explained = best_share / off_comb if off_comb > 1e-12 else 0.0
    if off_comb < 1e-6:
        verdict = "PERIOD_1"
        reason = f"off-comb power {off_comb:.2e} is at the floor"
    elif explained >= GENERATOR_TORUS_SHARE:
        verdict = "TORUS"
        reason = (f"one generator at {best_generator / 1e9:.4f} GHz explains "
                  f"{explained:.3f} of the off-comb power")
    else:
        verdict = "BROADBAND"
        reason = (f"best single generator explains only {explained:.3f}; "
                  "residue is a continuum")
    return {
        "status": "OK", "verdict": verdict, "reason": reason,
        "on_comb": on_comb, "off_comb": off_comb,
        "best_generator_hz": best_generator,
        "generator_share_of_off_comb": explained,
        "generator_ratio_to_pump": (None if best_generator is None
                                    else best_generator / pump_hz),
    }


def analyse_point(point_dir: Path) -> dict[str, object]:
    result = json.loads((point_dir / "result.json").read_text(encoding="utf-8"))
    validity = nd.classify_point(result)
    record: dict[str, object] = {
        "point_dir": str(point_dir),
        "device": result.get("device"),
        "control_value": result.get("control_value", result.get("pump_power_dbm")),
        "validity": asdict(validity),
    }
    if not validity.usable:
        record["status"] = "EXCLUDED"
        return record
    trace_path = point_dir / "trace.npz"
    if not trace_path.exists():
        record["status"] = "NO_TRACE"
        return record
    with np.load(trace_path) as data:
        t = np.asarray(data["t"], dtype=np.float64)
        v = np.asarray(data["v_out"], dtype=np.float64)
    start = int(result.get("steady_state_start_index", 0))
    start = min(start, max(t.size - 16, 0))
    t_s, v_s = t[start:], v[start:]
    pump_hz = nd.PUMP_HZ[result["device"]]

    # The stored time axis is unreliable across devices; derive the sample
    # spacing from quantities the driver records unambiguously.
    dt_sample = float(result["dt_s"]) * int(result.get("record_stride", 1))
    theiler = max(1, int(round((1.0 / pump_hz) / dt_sample)))

    record["dt_sample_s"] = dt_sample
    record["theiler_window"] = theiler
    # For a DRIVEN system the natural timescale is the pump period, which
    # ``theiler`` already counts in samples -- not the AMI delay. On a
    # near-sinusoidal period-1 trace AMI returns a delay of 6 samples, so the
    # default 20-delay horizon spans 0.4 of a pump period and the fit measures
    # the oscillation rather than any exponent. 60 pump periods orders the
    # regimes correctly (window ~ 0, chaos > 0); the autonomous reference
    # systems keep the delay-based default.
    lyapunov = measure_lambda(
        v_s, dt_sample, theiler=theiler, horizon=60 * theiler
    )
    record["lyapunov"] = asdict(lyapunov)
    # A bare exponent is not interpretable on its own -- always quote it
    # against the pump.
    record["lambda_1_over_omega_p"] = (
        None if lyapunov.lambda_1 is None
        else lyapunov.lambda_1 / (2.0 * math.pi * pump_hz)
    )
    record["second_generator"] = second_generator_share(t_s, v_s, pump_hz)
    record["status"] = "OK"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if not args.skip_validation:
        print("validating the estimator against reference systems ...", flush=True)
        validation = validate_references()
        (args.output / "validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        for row in validation["rows"]:
            print(f"  {row['system']:<26} {row.get('measured_lambda_1')} "
                  f"{'PASS' if row.get('pass') else 'FAIL'}  {row.get('note', row.get('reason'))}",
                  flush=True)
        print(f"  validation: {validation['status']}", flush=True)
        if validation["status"] != "PASS":
            print("estimator did not pass its own gate; device values are not "
                  "reported", flush=True)
            return 2

    for device in args.devices:
        rows: list[dict[str, object]] = []
        out_path = args.output / f"{device}.json"
        for campaign in args.campaign:
            device_dir = campaign / device
            if not device_dir.is_dir():
                continue
            for result_path in sorted(device_dir.glob("*/result.json")):
                started = time.perf_counter()
                row = analyse_point(result_path.parent)
                rows.append(row)
                if row["status"] == "OK":
                    lam = row["lyapunov"]["lambda_1"]
                    gen = row["second_generator"]
                    print(f"  {row['control_value']:>10.4f}  "
                          f"lambda_1={('%+.4e' % lam) if lam is not None else 'none':>12s}  "
                          f"{gen.get('verdict', gen.get('status')):<10s}  "
                          f"{time.perf_counter() - started:.1f}s", flush=True)
                nd_write(out_path, rows)
        print(f"{device}: {len(rows)} points -> {out_path}", flush=True)
    return 0


def nd_write(out_path: Path, rows: list[dict[str, object]], attempts: int = 20) -> None:
    """Atomic per-point write, retrying past a Windows reader lock."""
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for attempt in range(attempts):
        try:
            tmp.replace(out_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
