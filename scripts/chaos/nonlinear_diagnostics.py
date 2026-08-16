"""Literature-standard nonlinear-dynamics diagnostics for the chaos campaigns.

Three independent measurements, chosen because each rests on different
mathematics and each replaces a field this project has already found broken:

1. ``correlation_dimension`` -- Grassberger & Procaccia (1983) D2 from the
   correlation integral of a delay embedding.  Replaces ``poincare_clusters``,
   which is inverted on ``jc_fqjtwpa`` and non-monotone on ``jc_jtwpa``.

2. ``zero_one_test`` -- Gottwald & Melbourne (2004, 2009) K statistic.  Needs
   no embedding dimension and no scaling-region judgement, so its failure modes
   do not overlap with D2's.

3. ``fit_normal_form_exponent`` -- amplitude scaling of the invariant set
   against the control parameter.  A supercritical Neimark-Sacker bifurcation
   grows its invariant circle as ``(mu - mu_c)**0.5``; a hard (subcritical)
   transition jumps with no scaling region at all.

Every routine returns its own quality gate.  A diagnostic that cannot justify
itself returns ``None`` for the estimate and a reason, rather than a number.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Validity gating
# --------------------------------------------------------------------------

#: Pump periods a device must integrate before its trace is analysable.  The
#: near-lossless devices reach their off-lattice floor at about 1050 periods
#: (measured 2026-08-14); anything shorter analyses residual ringing, which is
#: non-recurrent and therefore inflates BOTH D2 and K toward "chaotic".
MIN_SETTLED_PERIODS: dict[str, float] = {
    "guarcello": 600.0,
    "ipm_2c_fixed": 1050.0,
    "jc_jtwpa": 1050.0,
    "jc_fqjtwpa": 1050.0,
    "rf_squid_2393_3wm": 1050.0,
}

PUMP_HZ: dict[str, float] = {
    "guarcello": 7.0e9,
    "ipm_2c_fixed": 7.9e9,
    "jc_jtwpa": 7.12e9,
    "jc_fqjtwpa": 7.9e9,
}

#: Fields this project has measured to be unreliable.  Listed so that a reader
#: of the output knows they were excluded on purpose and not overlooked.
EXCLUDED_FIELDS: tuple[str, ...] = (
    "poincare_clusters",
    "verdict",
    "verdict_reason",
    "period_multiple",
    "winning_n",
    "d1",
    "tau_periods",
)


@dataclass
class PointValidity:
    """Whether one campaign point may be used for attractor classification."""

    usable: bool
    reason: str
    periods: float
    analysed_periods: float
    signal_installed: bool


def classify_point(result: dict[str, Any]) -> PointValidity:
    """Gate one ``result.json`` for dynamical analysis.

    Two disqualifiers, both structural rather than statistical:

    * A signal tone makes the forcing quasi-periodic by construction, so the
      attractor is a 2-torus a priori and D2 >= 2 carries no information about
      the pump-only dynamics.
    * Fewer settled periods than the device's measured settling floor means the
      analysed window still contains the ramp transient.
    """
    device = result.get("device", "")
    if result.get("signal_installed"):
        return PointValidity(
            False,
            "signal installed: forcing is quasi-periodic by construction",
            math.nan,
            math.nan,
            True,
        )

    pump_hz = PUMP_HZ.get(device)
    if pump_hz is None:
        return PointValidity(False, f"no pump frequency known for {device!r}", math.nan, math.nan, False)

    periods = result["n_steps"] * result["dt_s"] * pump_hz
    start = int(result.get("steady_state_start_index", 0))
    stride = int(result.get("record_stride", 1))
    analysed = periods - (start * stride * result["dt_s"] * pump_hz)

    floor = MIN_SETTLED_PERIODS.get(device, 1050.0)
    if periods < floor:
        return PointValidity(
            False,
            f"integrated {periods:.0f} periods against a {floor:.0f}-period settling floor",
            periods,
            analysed,
            False,
        )
    return PointValidity(True, "ok", periods, analysed, False)


# --------------------------------------------------------------------------
# Delay embedding
# --------------------------------------------------------------------------


def autocorrelation_delay(series: np.ndarray, max_lag: int | None = None) -> int:
    """First zero crossing of the autocorrelation, the standard delay choice.

    Falls back to the first ``1/e`` crossing when the autocorrelation never
    changes sign inside ``max_lag`` (a nearly periodic signal can do this).
    """
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    if max_lag is None:
        max_lag = min(len(x) // 2, 20000)
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 1
    acf = np.empty(max_lag, dtype=np.float64)
    for lag in range(max_lag):
        acf[lag] = float(np.dot(x[: len(x) - lag], x[lag:])) / denom
    sign_change = np.nonzero(acf <= 0.0)[0]
    if sign_change.size:
        return max(int(sign_change[0]), 1)
    below = np.nonzero(acf <= 1.0 / math.e)[0]
    return max(int(below[0]), 1) if below.size else 1


def delay_embed(series: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Return the ``(N, dimension)`` Takens delay-coordinate matrix."""
    x = np.asarray(series, dtype=np.float64)
    span = (dimension - 1) * delay
    rows = len(x) - span
    if rows <= 0:
        raise ValueError(f"series of {len(x)} too short for m={dimension}, tau={delay}")
    return np.column_stack([x[i * delay : i * delay + rows] for i in range(dimension)])


# --------------------------------------------------------------------------
# 1. Grassberger-Procaccia correlation dimension
# --------------------------------------------------------------------------


@dataclass
class CorrelationDimension:
    d2: float | None
    r_squared: float | None
    scaling_low: float | None
    scaling_high: float | None
    dimension: int
    delay: int
    n_points: int
    theiler_window: int
    reason: str
    log_r: list[float] = field(default_factory=list)
    log_c: list[float] = field(default_factory=list)


def correlation_dimension(
    series: np.ndarray,
    dimension: int,
    delay: int,
    *,
    theiler_window: int,
    max_points: int = 4000,
    n_radii: int = 40,
    min_scaling_decades: float = 0.75,
    rng: np.random.Generator | None = None,
) -> CorrelationDimension:
    """Estimate D2 from the correlation integral of a delay embedding.

    The Theiler window is not optional here.  Without it, temporally adjacent
    samples -- which are close purely because the trajectory is continuous, not
    because the attractor is dense there -- dominate the small-``r`` counts and
    bias D2 downward toward 1.  Set it to at least one pump period.

    The Eckmann-Ruelle bound ``N > 10**D2`` is checked against the estimate and
    reported as a failure rather than silently returning an unsupportable
    dimension.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    try:
        embedded = delay_embed(series, dimension, delay)
    except ValueError as error:
        return CorrelationDimension(
            None, None, None, None, dimension, delay, 0, theiler_window, str(error)
        )

    n_total = embedded.shape[0]
    if n_total > max_points:
        # Uniform decimation preserves the attractor's measure; random sampling
        # would too, but uniform keeps the index arithmetic of the Theiler
        # window exact.
        step = int(math.ceil(n_total / max_points))
        embedded = embedded[::step]
        theiler = max(1, theiler_window // step)
    else:
        step = 1
        theiler = max(1, theiler_window)

    n = embedded.shape[0]
    if n < 200:
        return CorrelationDimension(
            None, None, None, None, dimension, delay, n, theiler,
            f"only {n} embedded points after decimation",
        )

    # ``pdist`` returns the condensed upper triangle in C, which avoids the
    # (n, n, m) broadcast an earlier version built -- that allocated 576 MB at
    # n = 3000, m = 8 and made a single campaign point take minutes.
    from scipy.spatial.distance import pdist

    row_index, col_index = np.triu_indices(n, k=1)
    valid = pdist(embedded)[(col_index - row_index) > theiler]
    del row_index, col_index
    if valid.size == 0:
        return CorrelationDimension(
            None, None, None, None, dimension, delay, n, theiler,
            "Theiler window removed every pair",
        )

    positive = valid[valid > 0.0]
    if positive.size == 0:
        return CorrelationDimension(
            None, None, None, None, dimension, delay, n, theiler, "all distances zero"
        )

    r_lo = float(np.percentile(positive, 0.5))
    r_hi = float(np.percentile(positive, 60.0))
    if not (r_hi > r_lo > 0.0):
        return CorrelationDimension(
            None, None, None, None, dimension, delay, n, theiler, "degenerate radius range"
        )

    radii = np.logspace(math.log10(r_lo), math.log10(r_hi), n_radii)
    # One sort plus a binary search beats 40 full passes over several million
    # distances.
    valid.sort()
    counts = np.searchsorted(valid, radii, side="left").astype(np.float64)
    total_pairs = float(valid.size)
    with np.errstate(divide="ignore"):
        log_c = np.log10(counts / total_pairs)
    log_r = np.log10(radii)
    finite = np.isfinite(log_c)
    log_r, log_c = log_r[finite], log_c[finite]
    if log_r.size < 8:
        return CorrelationDimension(
            None, None, None, None, dimension, delay, n, theiler,
            "fewer than 8 usable radii", log_r.tolist(), log_c.tolist(),
        )

    # Sliding-window search for the widest span whose local slope is most
    # nearly constant -- the "scaling region".  Reported with its own R^2 so a
    # curved log-log plot cannot masquerade as a dimension.
    best: tuple[float, float, float, float, float] | None = None
    # The usable radius count shrinks as ``m`` grows, because distances scale
    # with sqrt(m) and the smallest bins empty out.  A fixed decade-width
    # requirement therefore makes the search range go empty at high ``m`` and
    # silently returns "no scaling window" for every embedding above 3 -- which
    # is what an earlier version did on the Lorenz control.  Cap the demand at
    # half the grid actually available.
    spacing = float(log_r[1] - log_r[0])
    min_width = max(4, min(int(min_scaling_decades / spacing), log_r.size // 2))
    for start in range(0, log_r.size - min_width):
        for stop in range(start + min_width, log_r.size + 1):
            xs, ys = log_r[start:stop], log_c[start:stop]
            slope, intercept = np.polyfit(xs, ys, 1)
            residual = ys - (slope * xs + intercept)
            ss_res = float(np.dot(residual, residual))
            ss_tot = float(np.dot(ys - ys.mean(), ys - ys.mean()))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            width = float(xs[-1] - xs[0])
            score = r2 * width
            if best is None or score > best[0]:
                best = (score, float(slope), r2, float(xs[0]), float(xs[-1]))

    if best is None:
        return CorrelationDimension(
            None, None, None, None, dimension, delay, n, theiler,
            "no scaling window found", log_r.tolist(), log_c.tolist(),
        )

    _, slope, r2, lo, hi = best
    d2 = float(slope)
    if r2 < 0.99:
        return CorrelationDimension(
            None, r2, lo, hi, dimension, delay, n, theiler,
            f"no linear scaling region (best R^2 {r2:.4f})",
            log_r.tolist(), log_c.tolist(),
        )
    if d2 > math.log10(n):
        return CorrelationDimension(
            None, r2, lo, hi, dimension, delay, n, theiler,
            f"D2 {d2:.2f} violates the Eckmann-Ruelle bound at N={n}",
            log_r.tolist(), log_c.tolist(),
        )
    return CorrelationDimension(
        d2, r2, lo, hi, dimension, delay, n, theiler, "ok",
        log_r.tolist(), log_c.tolist(),
    )


def d2_saturation(
    series: np.ndarray,
    delay: int,
    *,
    theiler_window: int,
    dimensions: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    plateau_tolerance: float = 0.10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run D2 across embedding dimensions and take the FIRST plateau.

    At finite ``N`` the correlation integral over-counts sparse neighbourhoods,
    so D2 climbs without bound as ``m`` grows past the attractor's own
    dimension.  Measured on the Henon map (literature D2 = 1.220) with
    N = 3000: m = 2..6 gives 1.184, 1.230, 1.289, 1.344, 1.438.  Averaging the
    top three dimensions -- an earlier version of this function -- therefore
    reported the most over-embedded estimate available and was biased high by
    about 15 percent.

    The estimate is taken at the smallest ``m`` whose forward difference falls
    below ``plateau_tolerance``, which is the standard unfolding criterion:
    once the attractor is embedded, adding dimensions stops changing D2 until
    finite-``N`` bias takes over.
    """
    runs = [
        correlation_dimension(series, m, delay, theiler_window=theiler_window, **kwargs)
        for m in dimensions
    ]
    good = [(m, r.d2) for m, r in zip(dimensions, runs) if r.d2 is not None]
    plateau: float | None = None
    plateau_dimension: int | None = None
    for (m_a, d_a), (_, d_b) in zip(good, good[1:]):
        if abs(d_b - d_a) < plateau_tolerance:
            plateau, plateau_dimension = float(d_a), int(m_a)
            break
    return {
        "per_dimension": [asdict(r) for r in runs],
        "d2_curve": {str(m): d for m, d in good},
        "saturated": plateau is not None,
        "d2_plateau": plateau,
        "plateau_dimension": plateau_dimension,
    }


# --------------------------------------------------------------------------
# 2. Gottwald-Melbourne 0-1 test for chaos
# --------------------------------------------------------------------------


@dataclass
class ZeroOneTest:
    k_median: float | None
    k_values: list[float]
    n_samples: int
    n_c: int
    reason: str


def zero_one_test(
    series: np.ndarray,
    *,
    n_c: int = 100,
    rng: np.random.Generator | None = None,
    min_samples: int = 500,
) -> ZeroOneTest:
    """Gottwald-Melbourne K statistic; 0 = regular, 1 = chaotic.

    Uses the 2009 *modified* mean-square displacement, which subtracts the
    oscillatory term that otherwise contaminates ``M_c(n)`` and lets a merely
    quasi-periodic orbit read as chaotic.

    ``c`` is drawn from ``(pi/5, 4pi/5)`` rather than ``(0, pi)``: the endpoints
    and ``pi/2`` are resonant with the drive and produce spurious K near 1.

    The input must be sampled so the drive is not aliased -- for a periodically
    forced system, pass the stroboscopic section, not the raw trace.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    phi = np.asarray(series, dtype=np.float64)
    phi = phi[np.isfinite(phi)]
    n = phi.size
    if n < min_samples:
        return ZeroOneTest(None, [], n, 0, f"only {n} samples, need {min_samples}")
    if float(np.std(phi)) == 0.0:
        return ZeroOneTest(0.0, [], n, 0, "constant series: regular by inspection")

    n_max = n // 10
    if n_max < 20:
        return ZeroOneTest(None, [], n, 0, "series too short for a displacement window")

    j = np.arange(1, n + 1, dtype=np.float64)
    mean_phi = float(phi.mean())
    lags = np.arange(1, n_max + 1)
    k_values: list[float] = []

    for _ in range(n_c):
        c = float(rng.uniform(math.pi / 5.0, 4.0 * math.pi / 5.0))
        p = np.cumsum(phi * np.cos(j * c))
        q = np.cumsum(phi * np.sin(j * c))
        m = np.empty(n_max, dtype=np.float64)
        limit = n - n_max
        for idx, lag in enumerate(lags):
            dp = p[lag : lag + limit] - p[:limit]
            dq = q[lag : lag + limit] - q[:limit]
            m[idx] = float(np.mean(dp * dp + dq * dq))
        # 2009 modification: remove the bounded oscillatory contribution.
        denom = 1.0 - math.cos(c)
        oscillatory = (mean_phi ** 2) * (1.0 - np.cos(lags * c)) / denom
        d = m - oscillatory
        # Correlating D against n is the 2009 regression method; it is far less
        # sensitive to the choice of n_max than the asymptotic-growth-rate form.
        if float(np.std(d)) == 0.0:
            k_values.append(0.0)
            continue
        k_values.append(float(np.corrcoef(lags.astype(np.float64), d)[0, 1]))

    return ZeroOneTest(float(np.median(k_values)), k_values, n, n_c, "ok")


# --------------------------------------------------------------------------
# 3. Normal-form amplitude scaling
# --------------------------------------------------------------------------


@dataclass
class NormalFormFit:
    exponent: float | None
    exponent_stderr: float | None
    critical_control: float | None
    amplitude: float | None
    r_squared: float | None
    n_points: int
    verdict: str
    reason: str


def fit_normal_form_exponent(
    control: Sequence[float],
    amplitude: Sequence[float],
    *,
    min_points: int = 5,
    grid: int = 400,
) -> NormalFormFit:
    """Fit ``A(mu - mu_c)**beta`` to the invariant-set amplitude above onset.

    ``beta = 1/2`` is the supercritical Neimark-Sacker (and supercritical Hopf)
    normal-form prediction: the invariant circle's radius grows as the square
    root of the distance past onset.  A hard/subcritical transition has no
    scaling region -- the amplitude jumps -- and shows up here as a fit that
    either fails outright or returns an exponent far below 1/2 with a critical
    control value pinned against the lowest supercritical sample.

    ``mu_c`` is not free-floated with ``A`` and ``beta`` simultaneously; that
    problem is badly conditioned.  It is scanned on a grid, and for each trial
    value the remaining two parameters are obtained by linear least squares in
    log-log space, which is exact.
    """
    mu = np.asarray(control, dtype=np.float64)
    amp = np.asarray(amplitude, dtype=np.float64)
    order = np.argsort(mu)
    mu, amp = mu[order], amp[order]
    finite = np.isfinite(mu) & np.isfinite(amp) & (amp > 0.0)
    mu, amp = mu[finite], amp[finite]
    if mu.size < min_points:
        return NormalFormFit(
            None, None, None, None, None, int(mu.size), "INSUFFICIENT_DATA",
            f"{mu.size} usable points, need {min_points}",
        )

    # mu_c must lie below the smallest control value carrying amplitude, and
    # cannot be so far below that every point sits deep in the scaling tail.
    span = float(mu[-1] - mu[0])
    lo = float(mu[0]) - span
    hi = float(mu[0]) - 1e-9 * max(abs(float(mu[0])), 1.0)
    best: tuple[float, float, float, float, float] | None = None
    for mu_c in np.linspace(lo, hi, grid):
        x = np.log(mu - mu_c)
        y = np.log(amp)
        if not np.all(np.isfinite(x)):
            continue
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        ss_res = float(np.dot(residual, residual))
        ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if best is None or r2 > best[0]:
            n = x.size
            dof = max(n - 2, 1)
            var_x = float(np.dot(x - x.mean(), x - x.mean()))
            stderr = math.sqrt(ss_res / dof / var_x) if var_x > 0 else math.nan
            best = (r2, float(slope), float(mu_c), float(math.exp(intercept)), stderr)

    if best is None:
        return NormalFormFit(
            None, None, None, None, None, int(mu.size), "FIT_FAILED",
            "no admissible critical control value",
        )

    r2, beta, mu_c, amp0, stderr = best
    if r2 < 0.90:
        verdict = "NO_SCALING_REGION"
        reason = f"best R^2 {r2:.4f} below 0.90; consistent with a hard transition"
    elif abs(beta - 0.5) <= max(3.0 * stderr, 0.05):
        # ``stderr`` is the linear-regression error at FIXED ``mu_c``; it does
        # not carry the uncertainty of the scanned critical value, so a pure
        # 2-sigma band is too tight.  Measured on a synthetic supercritical
        # control (beta = 0.5 exactly, 2 percent noise, 14 points) the fit
        # returns 0.4872 +/- 0.0050 -- correct to 2.6 percent, but 2.6 sigma
        # out.  The 0.05 absolute floor admits that case while still excluding
        # the beta = 1 control, which lands 0.51 away.
        verdict = "SUPERCRITICAL_CONSISTENT"
        reason = f"beta {beta:.4f} +/- {stderr:.4f} consistent with 1/2"
    else:
        verdict = "SCALING_BUT_NOT_ONE_HALF"
        reason = f"beta {beta:.4f} +/- {stderr:.4f} excludes 1/2"
    return NormalFormFit(beta, stderr, mu_c, amp0, r2, int(mu.size), verdict, reason)


# --------------------------------------------------------------------------
# Campaign driver
# --------------------------------------------------------------------------


def stroboscopic_section(
    v: np.ndarray, pump_hz: float, dt_s: float, record_stride: int
) -> np.ndarray:
    """Sample ``v`` once per pump period, indexing in samples not in ``t``.

    The stored time axis is not trustworthy across devices: measured
    2026-08-16, ``trace.npz['t']`` for ``guarcello`` advances by ``dt_s`` per
    stored sample while ``ipm_2c_fixed`` advances by ``record_stride * dt_s``.
    Trusting ``t`` therefore strobed guarcello once every 20 pump periods
    instead of once per period.  The sample spacing is instead derived from
    quantities the driver records unambiguously.

    Returns an empty array when the record is too coarsely strided to place a
    strobe accurately -- fewer than eight stored samples per pump period.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.size < 16:
        return np.empty(0)
    samples_per_period = 1.0 / (pump_hz * dt_s * float(record_stride))
    if samples_per_period < 8.0:
        return np.empty(0)
    n_periods = int((v.size - 1) / samples_per_period)
    if n_periods < 8:
        return np.empty(0)
    index = np.arange(v.size, dtype=np.float64)
    strobe_index = samples_per_period * np.arange(n_periods)
    return np.interp(strobe_index, index, v)


def analyse_point(point_dir: Path) -> dict[str, Any]:
    """Run the full diagnostic set on one campaign point directory."""
    result = json.loads((point_dir / "result.json").read_text(encoding="utf-8"))
    validity = classify_point(result)
    record: dict[str, Any] = {
        "point_dir": str(point_dir),
        "device": result.get("device"),
        "control_value": result.get("control_value", result.get("pump_power_dbm")),
        "control_axis": result.get("control_axis", "pump_power_dbm"),
        "pump_power_dbm": result.get("pump_power_dbm"),
        "sigma_vprime_ps": result.get("sigma_vprime_ps"),
        "gain_db": result.get("gain_db"),
        "validity": asdict(validity),
        "excluded_fields": list(EXCLUDED_FIELDS),
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
    pump_hz = PUMP_HZ[result["device"]]

    # One pump period in stored samples sets the Theiler window: pairs closer
    # than that in time are correlated by the drive, not by the attractor.
    dt_stored = float(np.median(np.diff(t_s))) if t_s.size > 2 else 0.0
    theiler = max(1, int(round((1.0 / pump_hz) / dt_stored))) if dt_stored > 0 else 1

    delay = autocorrelation_delay(v_s, max_lag=min(v_s.size // 4, 5000))
    record["delay"] = delay
    record["theiler_window"] = theiler
    record["n_analysed_samples"] = int(v_s.size)

    record["correlation_dimension"] = d2_saturation(
        v_s, delay, theiler_window=theiler
    )

    strobe = stroboscopic_section(
        v_s, pump_hz, float(result["dt_s"]), int(result.get("record_stride", 1))
    )
    record["strobe_points"] = int(strobe.size)
    if strobe.size:
        record["zero_one_test"] = asdict(zero_one_test(strobe))
        record["strobe_std"] = float(np.std(strobe))
    else:
        branches = point_dir / "poincare_branches.npz"
        if branches.exists():
            with np.load(branches) as data:
                section = np.asarray(data["upward"], dtype=np.float64)
            record["zero_one_test"] = asdict(zero_one_test(section, min_samples=250))
            record["zero_one_input"] = "stored_poincare_upward"
            record["strobe_std"] = float(np.std(section))
        else:
            record["zero_one_test"] = None
            record["strobe_std"] = None
    record["status"] = "OK"
    return record


def analyse_campaign(root: Path, devices: Sequence[str] | None = None) -> dict[str, Any]:
    """Analyse every point under ``root`` and fit the per-device scaling law."""
    records: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("*/*/result.json")):
        device = json.loads(result_path.read_text(encoding="utf-8")).get("device")
        if devices and device not in devices:
            continue
        records.append(analyse_point(result_path.parent))

    by_device: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_device.setdefault(record["device"], []).append(record)

    fits: dict[str, Any] = {}
    for device, rows in by_device.items():
        usable = [r for r in rows if r.get("status") == "OK" and r.get("strobe_std")]
        usable.sort(key=lambda r: r["control_value"])
        fits[device] = asdict(
            fit_normal_form_exponent(
                [r["control_value"] for r in usable],
                [r["strobe_std"] for r in usable],
            )
        )
    return {"points": records, "normal_form_fits": fits}
