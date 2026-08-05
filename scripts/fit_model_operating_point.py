"""Fit the model's pump operating point (f_p, I_p) against measured G0(f).

G0(f) -- small-signal gain vs frequency -- is a transmission RATIO
(pump-on over pump-off), the only observable in this pipeline free of every
line-loss / port-power calibration constant (see
docs/development/psat_comparison_fix_plan.md Phase 1-3: Norton port power,
loss_B1/loss_A10 line calibration, self-consistent P_sat -- none of that
touches a ratio). Because real-device nonidealities make the device's actual
on-chip pump current differ from the model's, BOTH pump frequency and pump
current are fit jointly against the measured band SHAPE, rather than reading
one map cell's S21 at a fixed detuning against the measured band peak (the
approach this replaces, off by +2.63 dB / +351 MHz per that plan's Phase 4).

Method: coarse-then-fine 2-D grid search over (pump_freq_ghz, pump_current_a)
-- log-spaced in current, mirroring scripts/align_map_to_measurement.py's
(df, dP) search.

Two pieces are reused from elsewhere in this repo rather than reimplemented,
per the 2026-08-05 finding below:

1. PUMP SOLVE ROBUSTNESS: each candidate's pump harmonic-balance problem is
   solved via ``scripts/run_gain_map.py``'s ``InProcessEngine`` (cold solve
   via its adaptive-secant continuation, warm single-Newton solve from the
   previous lower-current candidate, and -- when that warm solve fails --
   ``InProcessEngine.solve_power_substep``'s geometric current ladder, the
   same "Power substep" continuation production gain maps use to cross a
   cold-start convergence wall). See ``solve_pump_point_robust``.

2. GAIN AT A FINITE, MEASUREMENT-MATCHED SIGNAL POWER: rather than the
   small-signal LINEAR Floquet limit (``twpa_solver.signal.solve_gain_one``,
   which approximates the reference power as zero), gain is evaluated with
   the SATURATION (multitone) model at the on-chip current corresponding to
   the SMALLEST signal power actually present in the measured cube (its
   lowest instrument-power row, converted on-chip via loss_B1). This mirrors
   scripts/run_compression.py's finite-signal machinery
   (FullMultiToneProblem, AffineSourcePath, solve_signal_power_point) for a
   single reference power point per signal frequency, instead of a full
   power sweep. See ``solve_finite_signal_gain_db``.

Both are materially more expensive than the linear-Floquet approach this
replaces: a 2c pump solve is O(seconds), a power-substep bridge across a
~2x current gap took 54.7s (14 substeps) in the measurement below, and each
signal-frequency point is now a full nonlinear solve rather than a shared
linear-system reuse. Keep grids small; see the CLI's coarse/fine point
defaults and this module's own test for cost-scoped examples.

The measured envelope (already ripple-averaged and pump-gap excised in
scripts/measured_psat_pipeline.compute_smoothed_g0) is sub-sampled onto the
model's own frequency grid -- never the reverse, since the model grid is
coarser (>= a few tens of MHz) than the measured 4 MHz grid and interpolating
the model onto the finer grid would invent ripple structure the model never
resolved.

Usage:

    python scripts/fit_model_operating_point.py \\
        --circuit-dir designs/ipm_2c_fixed --outdir outputs/fit_op_point

2026-08-05 finding that shaped this design: a single warm Newton solve from
a converged lower-current candidate could NOT bridge the jump from 5e-6 A
(converges) to designs/ipm_2c_fixed's physically-relevant current range
(~1.1e-5 to 3.5e-5 A) -- it failed the same way a cold solve does. Bridging
that same gap with ``InProcessEngine.solve_power_substep``'s geometric
current ladder (grow x1.5 / shrink x0.5 on failure) DID reach it: 5e-6 A ->
1.077e-5 A converged in 14 substeps / 54.7s. That is why pump solving here
goes through the engine rather than a simpler fixed/adaptive ladder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from twpa_solver.core import CircuitMatrices, load_circuit
from twpa_solver.loss import signal_line_loss_model
from twpa_solver.multitone.basis import build_sideband_matched_basis
from twpa_solver.multitone.compression import solve_signal_power_point
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.seed import promote_pump_solution
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver, NewtonKrylovSettings, resolve_pump_basis
from twpa_solver.ports import port_available_power_w, port_current_from_power_a

ROOT = Path(__file__).resolve().parents[1]
# Allows both `python -m scripts.fit_model_operating_point` and running this
# file directly (`python scripts/fit_model_operating_point.py`) -- the latter
# puts only scripts/ on sys.path, not its parent, so `import scripts.*` would
# otherwise fail outside pytest (whose rootdir config adds ROOT for us).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_gain_map as rgm  # noqa: E402
from scripts.measured_psat_pipeline import (  # noqa: E402
    DEFAULT_CUBE_PATH,
    G0_SMOOTH_POLYORDER,
    G0_SMOOTH_WINDOW_FRAC,
    MEAS_PUMP_GHZ,
    PUMP_EXCLUSION_GHZ,
    compute_smoothed_g0,
    load_cube,
)

DEFAULT_CIRCUIT_DIR = ROOT / "designs" / "ipm_2c_fixed"
DEFAULT_OUTDIR = ROOT / "outputs" / "fit_model_operating_point"

# A model grid coarser than this aliases its own 3-4 dB ripple against the
# measured ripple of similar period -- see this module's docstring and
# docs/development/psat_comparison_fix_plan.md Phase 4.
MAX_MODEL_FREQ_STEP_MHZ = 25.0

ROI_WEIGHT_FLOOR_DB = 0.1

# Multitone solve settings for the finite-signal gain point (mirrors
# scripts/run_compression.py's _solve_compression settings block).
_MULTITONE_PRECONDITIONER = "real_coupled_fast"
_MULTITONE_SETTINGS_KWARGS = dict(
    newton_tol=1e-10, max_newton=20, gmres_rtol=1e-8, gmres_atol=0.0,
    gmres_restart=20, gmres_maxiter=40, min_alpha=1.0 / 1024.0,
    preconditioner=_MULTITONE_PRECONDITIONER, compute_time_residual=False, verbose=False,
    continuation_predictor="none", jvp_mode="aft",
    precond_reuse=1, precond_reuse_refresh_gmres=0,
)


def build_pump_engine(
    circuit_dir: Path,
    *,
    pump_port: int,
    source_port: int,
    out_port: int,
    pump_mode_policy: str,
    pump_mode_count: int,
    pump_harmonics: int,
    pump_nt: int,
    outdir: Path,
) -> "rgm.InProcessEngine":
    """Construct a run_gain_map.py InProcessEngine for robust pump solving.

    Only the pump-solve machinery is used (``solve_point``,
    ``solve_power_substep``); the linear gain fields ``solve_point`` also
    computes are unused here and ignored (see ``solve_pump_point_robust``).
    ``--inproc-pump-backend full`` keeps the returned pump state full-node
    (not Schur-retained), so it is directly usable with
    ``promote_pump_solution`` without a reconstruction step.
    """
    argv = [
        "--circuit-dir", str(circuit_dir),
        "--pump-port", str(pump_port), "--source-port", str(source_port),
        "--out-port", str(out_port),
        "--pump-mode-policy", pump_mode_policy,
        "--pump-mode-count", str(pump_mode_count),
        "--harmonics", str(pump_harmonics), "--nt", str(pump_nt),
        "--inproc-pump-backend", "full", "--inproc-preconditioner", "real_coupled",
        "--no-signal-spectrum", "--sidebands", "2", "--gamma-nt", "32",
        "--attenuation-db", "0", "--outdir", str(outdir),
    ]
    args = rgm.parse_args(argv)
    return rgm.InProcessEngine(args)


def solve_pump_point_robust(
    engine: "rgm.InProcessEngine",
    pass_dir: Path,
    index: int,
    pump_freq_ghz: float,
    pump_current_a: float,
    *,
    warm_state: np.ndarray | None,
    warm_current_a: float | None,
    power_substep_init_db: float = 0.5,
    power_substep_min_db: float = 0.02,
    power_substep_deadline_s: float = 120.0,
):
    """Robust pump HB solve at one (pump_freq_ghz, pump_current_a).

    No warm state: cold solve via the engine's default continuation
    (adaptive-secant). With a warm state at a LOWER current: a cheap warm
    single-Newton solve first; if that fails, bridge the gap with
    ``engine.solve_power_substep``'s geometric current ladder and retry warm
    from the bridged state (see this module's top-level docstring for why
    the ladder is necessary, not optional).

    Returns ``(X_full, pump_basis)`` or ``(None, pump_basis)``.
    """
    point = rgm.GridPoint(
        index=index, i_power=0, j_freq=0, power_dbm=0.0,
        pump_freq_ghz=pump_freq_ghz, current_a=pump_current_a,
    )
    if warm_state is None or warm_current_a is None:
        row, X = engine.solve_point(point, pass_dir, mode="cold", warm_X=None)
    else:
        row, X = engine.solve_point(point, pass_dir, mode="warm", warm_X=warm_state)
        if row["pump_status"] != "VALID_CONVERGED" and pump_current_a > warm_current_a:
            bridged_X, _info = engine.solve_power_substep(
                pump_freq_ghz, warm_state, warm_current_a, pump_current_a,
                init_db=power_substep_init_db, min_db=power_substep_min_db,
                deadline_s=power_substep_deadline_s,
            )
            if bridged_X is not None:
                row, X = engine.solve_point(point, pass_dir, mode="warm", warm_X=bridged_X)

    omega_p = 2.0 * math.pi * pump_freq_ghz * 1e9
    pump_basis = resolve_pump_basis(
        policy=engine.args.pump_mode_policy, omega_p=omega_p,
        harmonics=engine.args.harmonics, mode_count=engine.args.pump_mode_count,
        explicit_modes=None, design_meta=engine.ipm08.metadata,
    )
    if row["pump_status"] != "VALID_CONVERGED" or X is None:
        return None, pump_basis
    return X, pump_basis


def reference_signal_current_a(
    smallest_instrument_dbm: float, signal_ghz: float, z0_ohm: float = 50.0
) -> float:
    """On-chip current at the measurement's SMALLEST available signal power.

    ``smallest_instrument_dbm`` is the cube's lowest SignalPower row
    (constant across frequency -- the instrument power axis is shared by
    every column); the on-chip value still varies with frequency through
    loss_B1's attenuation. This replaces the idealized zero-signal linear
    limit with the actual smallest power the measurement resolves.
    """
    onchip_dbm = smallest_instrument_dbm - float(
        signal_line_loss_model().attenuation_db(signal_ghz)
    )
    power_w = 1.0e-3 * 10.0 ** (onchip_dbm / 10.0)
    return port_current_from_power_a(power_w, z0_ohm, convention="norton")


def solve_finite_signal_gain_db(
    circuit: CircuitMatrices,
    pump_X: np.ndarray,
    pump_basis,
    pump_current_a: float,
    omega_p: float,
    signal_ghz: float,
    signal_current_a: float,
    *,
    pump_port: int,
    source_port: int,
    out_port: int,
    z0_ohm: float,
    multitone_sidebands: int,
    problem_cache: dict[object, object] | None = None,
) -> float | None:
    """gain_vs_off_db at ONE finite signal power (not the linear limit).

    Mirrors scripts/run_compression.py's ``_solve_compression`` pump-off /
    pump-only / signal-turn-on sequence, restricted to a single reference
    current instead of a full power sweep. Returns None if any stage fails
    to converge, the multitone basis would silently truncate a pump harmonic
    (see ``_build_multitone_basis`` in run_compression.py), or ``signal_ghz``
    lands exactly on a degenerate zero-frequency tone combination (delta==0
    when a grid search's candidate pump frequency happens to equal one of
    its own swept signal frequencies is the common case -- a real, not
    merely synthetic, occurrence).
    """
    delta = omega_p - 2.0 * math.pi * signal_ghz * 1e9
    omega_max = omega_p * (max(pump_basis.modes) + float(multitone_sidebands) + 1.0)
    try:
        basis = build_sideband_matched_basis(
            pump_basis.modes, multitone_sidebands, omega_p, delta, omega_max
        )
    except ValueError:
        return None
    represented_modes = {tone.h for tone in basis.tones if tone.q == 0}
    if set(pump_basis.modes) - represented_modes:
        return None

    pump_seed = promote_pump_solution(pump_X, pump_basis, basis)
    pump_source = MultiToneDrive(
        basis.pump_tone, circuit.port_to_index[pump_port], pump_current_a
    ).to_coeffs(basis, circuit.node_count)
    signal_unit = MultiToneDrive(
        basis.signal_tone, circuit.port_to_index[source_port], 1.0
    ).to_coeffs(basis, circuit.node_count)

    cache = problem_cache if problem_cache is not None else {}

    def make_problem(path: AffineSourcePath) -> FullMultiToneProblem:
        return FullMultiToneProblem(
            circuit, basis, path, preconditioner=_MULTITONE_PRECONDITIONER, cache=cache,
        )

    settings = NewtonKrylovSettings(**_MULTITONE_SETTINGS_KWARGS)
    solver = HarmonicNewtonKrylovSolver(settings)
    signal_row = basis.index_of(basis.signal_tone)
    output_node = circuit.port_to_index[out_port]

    pump_off_path = AffineSourcePath.signal_turn_on(
        np.zeros_like(pump_source), signal_unit * signal_current_a,
    )
    pump_off_problem = make_problem(pump_off_path)
    pump_off_state, pump_off_report = solver.solve_one(
        pump_off_problem, pump_off_problem.zeros(), 1.0
    )
    if not pump_off_report.converged:
        return None
    pump_off_voltage = (
        1j * basis.omegas[signal_row] * pump_off_state[signal_row, output_node]
    )

    pump_only_problem = make_problem(AffineSourcePath.pump_turn_on(pump_source))
    pump_only_state, pump_only_report = solver.solve_one(pump_only_problem, pump_seed, 1.0)
    if not pump_only_report.converged:
        pump_only_state, reports, trace = solver.solve_adaptive_continuation(
            pump_only_problem, None,
            initial_step=0.25, min_step=0.01, growth=1.5, shrink=0.5,
            fallback_fixed_steps=20,
        )
        if (
            not reports
            or not reports[-1].converged
            or not trace.accepted_lambdas
            or trace.accepted_lambdas[-1] < 1.0 - 1e-12
        ):
            return None

    solved = solve_signal_power_point(
        pump_only_problem, pump_only_state, None, signal_current_a,
        pump_source=pump_source, signal_source=signal_unit, solver=solver,
        signal_current_prev_a=0.0, recovery="ladder", pump_seed=pump_only_state,
    )
    if solved.status != "VALID_SOLVED":
        return None
    signal_voltage = 1j * basis.omegas[signal_row] * solved.state[signal_row, output_node]
    return float(
        20.0 * np.log10(max(abs(signal_voltage / pump_off_voltage), 1e-300))
    )


def sweep_model_g0_db(
    circuit: CircuitMatrices,
    engine: "rgm.InProcessEngine",
    pass_dir: Path,
    index: int,
    pump_freq_ghz: float,
    pump_current_a: float,
    signal_freqs_ghz: np.ndarray,
    reference_currents_a: np.ndarray,
    *,
    pump_port: int,
    source_port: int,
    out_port: int,
    multitone_sidebands: int,
    z0_ohm: float,
    warm_state: np.ndarray | None = None,
    warm_current_a: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Finite-signal gain_vs_off_db(f) at one (pump_freq_ghz, pump_current_a).

    One robust pump HB solve (``solve_pump_point_robust``), then one
    finite-signal multitone solve per entry in ``signal_freqs_ghz`` at the
    matching ``reference_currents_a`` (see ``reference_signal_current_a``).
    Serial, not threaded: each point shares a mutable ``problem_cache`` that
    is not thread-safe (unlike the retired linear sweep, which had no shared
    mutable state to race on).

    Returns ``(gains, pump_state)``: ``gains`` is None if the pump did not
    converge (individual signal points that fail are NaN otherwise);
    ``pump_state`` is the converged X (for the caller to warm-start the next
    candidate with), or None alongside a failed pump solve.
    """
    X, pump_basis = solve_pump_point_robust(
        engine, pass_dir, index, pump_freq_ghz, pump_current_a,
        warm_state=warm_state, warm_current_a=warm_current_a,
    )
    if X is None:
        return None, None

    omega_p = 2.0 * math.pi * pump_freq_ghz * 1e9
    problem_cache: dict[object, object] = {}
    gains = [
        solve_finite_signal_gain_db(
            circuit, X, pump_basis, pump_current_a, omega_p,
            float(signal_ghz), float(ref_current_a),
            pump_port=pump_port, source_port=source_port, out_port=out_port,
            z0_ohm=z0_ohm, multitone_sidebands=multitone_sidebands,
            problem_cache=problem_cache,
        )
        for signal_ghz, ref_current_a in zip(signal_freqs_ghz, reference_currents_a)
    ]
    gains = [g if g is not None else float("nan") for g in gains]
    return np.asarray(gains, dtype=float), X


def _ripple_weights(target_g0_db: np.ndarray, floor: float = ROI_WEIGHT_FLOOR_DB) -> np.ndarray:
    """Weight proportional to measured G0, floored so the background is not
    literally zero-weighted (the amplified ridge should drive the fit, per
    this module's docstring, not the whole flat/negative-gain background)."""
    return np.maximum(np.clip(target_g0_db, 0.0, None), floor)


def score_operating_point(
    pump_freq_ghz: float,
    pump_current_a: float,
    circuit: CircuitMatrices,
    engine: "rgm.InProcessEngine",
    pass_dir: Path,
    index: int,
    *,
    model_freq_ghz: np.ndarray,
    target_g0_db: np.ndarray,
    weight: np.ndarray,
    reference_currents_a: np.ndarray,
    pump_port: int,
    source_port: int,
    out_port: int,
    multitone_sidebands: int,
    z0_ohm: float,
    min_valid_points: int = 8,
    warm_state: np.ndarray | None = None,
    warm_current_a: float | None = None,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """One (f_p, I_p) candidate's cost (inf if unusable), its G0(f) curve, and
    its converged pump state (for the caller to warm-start the next
    candidate with -- see ``solve_pump_point_robust``).

    Weighted least squares, normalised by the fraction of the fit window that
    actually converged -- mirrors align_map_to_measurement.py's overlap
    normalisation, so a candidate that only converges at a few favourable
    points is not rewarded for the shrunk window.
    """
    model_g0, pump_state = sweep_model_g0_db(
        circuit, engine, pass_dir, index, pump_freq_ghz, pump_current_a,
        model_freq_ghz, reference_currents_a,
        pump_port=pump_port, source_port=source_port, out_port=out_port,
        multitone_sidebands=multitone_sidebands, z0_ohm=z0_ohm,
        warm_state=warm_state, warm_current_a=warm_current_a,
    )
    if model_g0 is None:
        return math.inf, None, None
    valid = np.isfinite(model_g0) & (weight > 0.0)
    if valid.sum() < min_valid_points:
        return math.inf, model_g0, pump_state
    resid = target_g0_db[valid] - model_g0[valid]
    cost = float(np.average(resid**2, weights=weight[valid]))
    overlap = valid.sum() / max(int((weight > 0.0).sum()), 1)
    return cost / overlap, model_g0, pump_state


@dataclass
class BandSummary:
    peak_gain_db: float
    peak_freq_ghz: float
    bandwidth_3db_ghz: float


def summarize_band(freq_ghz: np.ndarray, gain_db: np.ndarray) -> BandSummary:
    valid = np.isfinite(gain_db)
    if not valid.any():
        return BandSummary(float("nan"), float("nan"), float("nan"))
    idx = int(np.nanargmax(np.where(valid, gain_db, -np.inf)))
    peak_gain = float(gain_db[idx])
    peak_freq = float(freq_ghz[idx])
    above = valid & (gain_db >= peak_gain - 3.0)
    bandwidth = float(freq_ghz[above].max() - freq_ghz[above].min()) if above.any() else float("nan")
    return BandSummary(peak_gain, peak_freq, bandwidth)


def fit_operating_point(
    circuit: CircuitMatrices,
    circuit_dir: Path,
    measured_freq_ghz: np.ndarray,
    measured_g0_db: np.ndarray,
    smallest_instrument_dbm: float,
    *,
    freq_bounds_ghz: tuple[float, float],
    current_bounds_a: tuple[float, float],
    fit_freq_bounds_ghz: tuple[float, float],
    coarse_freq_points: int = 6,
    coarse_current_points: int = 6,
    fine_points: int = 5,
    signal_freq_step_mhz: float = MAX_MODEL_FREQ_STEP_MHZ,
    pump_port: int = 4,
    source_port: int = 1,
    out_port: int = 2,
    pump_mode_policy: str = "positive_odd_jc",
    pump_mode_count: int = 10,
    pump_harmonics: int = 10,
    pump_nt: int = 40,
    multitone_sidebands: int = 2,
    z0_ohm: float = 50.0,
    outdir: Path | None = None,
    progress: bool = True,
) -> dict[str, object]:
    """Coarse-then-fine 2-D grid search for (pump_freq_ghz, pump_current_a).

    ``measured_freq_ghz``/``measured_g0_db`` should already be ripple-averaged
    and pump-gap-excised (scripts.measured_psat_pipeline.compute_smoothed_g0).
    ``smallest_instrument_dbm`` is the measured cube's lowest SignalPower row
    (see ``reference_signal_current_a``).
    """
    if signal_freq_step_mhz > MAX_MODEL_FREQ_STEP_MHZ:
        raise ValueError(
            f"signal_freq_step_mhz={signal_freq_step_mhz} exceeds "
            f"{MAX_MODEL_FREQ_STEP_MHZ} MHz -- the model grid would alias its "
            "own ripple against the measured ripple (see module docstring)."
        )
    scratch_dir = Path(outdir) if outdir is not None else DEFAULT_OUTDIR
    engine = build_pump_engine(
        circuit_dir, pump_port=pump_port, source_port=source_port, out_port=out_port,
        pump_mode_policy=pump_mode_policy, pump_mode_count=pump_mode_count,
        pump_harmonics=pump_harmonics, pump_nt=pump_nt,
        outdir=scratch_dir / "_engine_scratch",
    )

    lo, hi = fit_freq_bounds_ghz
    model_freq_ghz = np.arange(lo, hi + 1e-9, signal_freq_step_mhz / 1000.0)
    target_g0_db = np.interp(model_freq_ghz, measured_freq_ghz, measured_g0_db)
    coverage = (model_freq_ghz >= measured_freq_ghz.min()) & (
        model_freq_ghz <= measured_freq_ghz.max()
    )
    weight = np.where(coverage, _ripple_weights(target_g0_db), 0.0)
    if (weight > 0.0).sum() < 8:
        raise ValueError(
            "fewer than 8 weighted points in the fit window "
            f"({fit_freq_bounds_ghz}); widen it or check measured coverage"
        )
    reference_currents_a = np.array([
        reference_signal_current_a(smallest_instrument_dbm, float(f), z0_ohm)
        for f in model_freq_ghz
    ])

    point_index = [0]

    def scan(
        freq_vals: np.ndarray, log_current_vals: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int], np.ndarray | None]:
        """log_current_vals is traversed low-to-high (np.linspace is always
        ascending here), warm-starting each candidate from the previous
        (lower-current) converged pump state at the SAME frequency --
        ``solve_pump_point_robust`` bridges a failed warm attempt with the
        engine's power-substep ladder rather than giving up (see this
        module's top-level docstring). Resets at each new frequency row
        since the state was converged at the previous row's pump frequency,
        not this one.
        """
        surf = np.full((freq_vals.size, log_current_vals.size), np.inf)
        best_index = (0, 0)
        best_score = math.inf
        best_curve: np.ndarray | None = None
        for a, fp in enumerate(freq_vals):
            warm_state: np.ndarray | None = None
            warm_current_a: float | None = None
            for b, log_i in enumerate(log_current_vals):
                current_a = 10.0**log_i
                t0 = time.perf_counter()
                point_index[0] += 1
                score, curve, pump_state = score_operating_point(
                    float(fp), current_a, circuit, engine,
                    scratch_dir / "_engine_scratch", point_index[0],
                    model_freq_ghz=model_freq_ghz, target_g0_db=target_g0_db,
                    weight=weight, reference_currents_a=reference_currents_a,
                    pump_port=pump_port, source_port=source_port, out_port=out_port,
                    multitone_sidebands=multitone_sidebands, z0_ohm=z0_ohm,
                    warm_state=warm_state, warm_current_a=warm_current_a,
                )
                if pump_state is not None:
                    warm_state, warm_current_a = pump_state, current_a
                surf[a, b] = score
                if progress:
                    print(
                        f"  fp={fp:.4f} GHz  I={current_a:.4g} A  "
                        f"score={score:.4f}  ({time.perf_counter() - t0:.1f}s)",
                        flush=True,
                    )
                if score < best_score:
                    best_score = score
                    best_index = (a, b)
                    best_curve = curve
        return surf, best_index, best_curve

    freq_coarse = np.linspace(freq_bounds_ghz[0], freq_bounds_ghz[1], coarse_freq_points)
    log_current_coarse = np.linspace(
        math.log10(current_bounds_a[0]), math.log10(current_bounds_a[1]),
        coarse_current_points,
    )
    if progress:
        print(f"coarse scan: {freq_coarse.size}x{log_current_coarse.size} candidates", flush=True)
    surf_coarse, (ia, ib), _ = scan(freq_coarse, log_current_coarse)
    fp0, logi0 = float(freq_coarse[ia]), float(log_current_coarse[ib])

    freq_step = float(freq_coarse[1] - freq_coarse[0]) if freq_coarse.size > 1 else 0.05
    logi_step = (
        float(log_current_coarse[1] - log_current_coarse[0])
        if log_current_coarse.size > 1
        else 0.2
    )
    freq_fine = np.linspace(fp0 - 2 * freq_step, fp0 + 2 * freq_step, fine_points)
    logi_fine = np.linspace(logi0 - 2 * logi_step, logi0 + 2 * logi_step, fine_points)
    if progress:
        print(f"fine scan: {freq_fine.size}x{logi_fine.size} candidates", flush=True)
    surf_fine, (ja, jb), best_curve = scan(freq_fine, logi_fine)
    fp_best = float(freq_fine[ja])
    current_best = float(10.0 ** logi_fine[jb])

    if best_curve is None:
        raise RuntimeError("no candidate in the fine grid produced a usable G0(f) curve")

    on_chip_pump_dbm = 10.0 * math.log10(
        port_available_power_w(current_best, z0_ohm, convention="norton") / 1.0e-3
    )
    model_summary = summarize_band(model_freq_ghz, best_curve)
    measured_summary = summarize_band(model_freq_ghz, target_g0_db)

    return {
        "pump_freq_ghz": fp_best,
        "pump_current_a": current_best,
        "on_chip_pump_dbm_norton": on_chip_pump_dbm,
        "model_freq_ghz": model_freq_ghz,
        "target_g0_db": target_g0_db,
        "model_g0_db": best_curve,
        "weight": weight,
        "reference_currents_a": reference_currents_a,
        "model_band": model_summary,
        "measured_band": measured_summary,
        "coarse_freq_ghz": freq_coarse,
        "coarse_log_current": log_current_coarse,
        "coarse_surface": surf_coarse,
        "fine_freq_ghz": freq_fine,
        "fine_log_current": logi_fine,
        "fine_surface": surf_fine,
        "fine_best_index": (ja, jb),
    }


def _write_overlay_and_surface_plot(result: dict[str, object], outdir: Path) -> Path:
    figure, (axis_curve, axis_surface) = plt.subplots(1, 2, figsize=(16, 6))

    freq = result["model_freq_ghz"]
    axis_curve.plot(freq, result["target_g0_db"], color="#d1495b", linewidth=1.6,
                     label="measured (ripple-averaged, on model grid)")
    axis_curve.plot(freq, result["model_g0_db"], color="#8a4fc9", linewidth=1.6,
                     label="model (fitted operating point)")
    axis_curve.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis_curve.set_xlabel("signal frequency (GHz)")
    axis_curve.set_ylabel("G0 (dB)")
    axis_curve.set_title(
        f"fp={result['pump_freq_ghz']:.4f} GHz  I_p={result['pump_current_a']:.4g} A  "
        f"({result['on_chip_pump_dbm_norton']:.2f} dBm on-chip, Norton)"
    )
    axis_curve.legend(fontsize=8.5)
    axis_curve.grid(alpha=0.25, linestyle=":")

    surf = result["fine_surface"]
    finite = surf[np.isfinite(surf)]
    vmax = float(np.percentile(finite, 90)) if finite.size else 1.0
    im = axis_surface.imshow(
        surf.T, origin="lower", aspect="auto", cmap="viridis_r", vmax=vmax,
        extent=[
            result["fine_freq_ghz"].min(), result["fine_freq_ghz"].max(),
            result["fine_log_current"].min(), result["fine_log_current"].max(),
        ],
    )
    figure.colorbar(im, ax=axis_surface, label="cost (clipped at p90)")
    ja, jb = result["fine_best_index"]
    axis_surface.plot(
        result["fine_freq_ghz"][ja], result["fine_log_current"][jb], "r*", markersize=14,
    )
    axis_surface.set_xlabel("pump frequency (GHz)")
    axis_surface.set_ylabel("log10(pump current / A)")
    axis_surface.set_title("fine-grid loss surface")

    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "fit_operating_point.png"
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    plt.close(figure)
    return png


def _json_safe(result: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, BandSummary):
            out[key] = {
                "peak_gain_db": value.peak_gain_db,
                "peak_freq_ghz": value.peak_freq_ghz,
                "bandwidth_3db_ghz": value.bandwidth_3db_ghz,
            }
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE_PATH)
    parser.add_argument("--circuit-dir", type=Path, default=DEFAULT_CIRCUIT_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--source-port", type=int, default=1)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--pump-mode-policy", default="positive_odd_jc")
    parser.add_argument("--pump-mode-count", type=int, default=10)
    parser.add_argument("--pump-harmonics", type=int, default=10)
    parser.add_argument("--pump-nt", type=int, default=40)
    parser.add_argument(
        "--multitone-sidebands", type=int, default=2,
        help="Sideband count for the finite-signal gain solve (matches "
             "run_compression.py's default; keep small, each point is now a "
             "genuine nonlinear solve, not a shared linear-system reuse.",
    )
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--pump-freq-min-ghz", type=float, default=7.6)
    parser.add_argument("--pump-freq-max-ghz", type=float, default=7.85)
    parser.add_argument(
        "--pump-current-min-a", type=float, default=5.0e-6,
        help=(
            "Default bounds span the on-chip current corresponding to the "
            "standard gain-map's -26..-16 dBm EXTERNAL pump power range "
            "(port_current_from_power_a at on-chip -61..-51 dBm after "
            "loss_A10 at ~7.7 GHz), widened a bit either side for real-device "
            "nonidealities. This is a raw on-chip current -- no line loss or "
            "power-convention math applies to it a second time here."
        ),
    )
    parser.add_argument("--pump-current-max-a", type=float, default=5.0e-5)
    parser.add_argument("--fit-freq-min-ghz", type=float, default=6.0)
    parser.add_argument("--fit-freq-max-ghz", type=float, default=8.5)
    parser.add_argument("--coarse-freq-points", type=int, default=6)
    parser.add_argument("--coarse-current-points", type=int, default=6)
    parser.add_argument("--fine-points", type=int, default=5)
    parser.add_argument("--signal-freq-step-mhz", type=float, default=MAX_MODEL_FREQ_STEP_MHZ)
    args = parser.parse_args()

    cube = load_cube(args.cube_path)
    g0 = compute_smoothed_g0(cube)
    measured_freq_ghz = cube.frequency_ghz[g0.keep]
    measured_g0_db = g0.values[g0.keep]
    smallest_instrument_dbm = float(cube.instrument_power_dbm.min())
    print(
        f"measured target: n={measured_freq_ghz.size} points, "
        f"{measured_freq_ghz.min():.3f}-{measured_freq_ghz.max():.3f} GHz, "
        f"savgol window={g0.window} (frac={G0_SMOOTH_WINDOW_FRAC}, "
        f"order={G0_SMOOTH_POLYORDER}); smallest instrument signal power = "
        f"{smallest_instrument_dbm:.1f} dBm"
    )

    circuit = load_circuit(args.circuit_dir)

    t0 = time.perf_counter()
    result = fit_operating_point(
        circuit, args.circuit_dir, measured_freq_ghz, measured_g0_db,
        smallest_instrument_dbm,
        freq_bounds_ghz=(args.pump_freq_min_ghz, args.pump_freq_max_ghz),
        current_bounds_a=(args.pump_current_min_a, args.pump_current_max_a),
        fit_freq_bounds_ghz=(args.fit_freq_min_ghz, args.fit_freq_max_ghz),
        coarse_freq_points=args.coarse_freq_points,
        coarse_current_points=args.coarse_current_points,
        fine_points=args.fine_points,
        signal_freq_step_mhz=args.signal_freq_step_mhz,
        pump_port=args.pump_port, source_port=args.source_port, out_port=args.out_port,
        pump_mode_policy=args.pump_mode_policy, pump_mode_count=args.pump_mode_count,
        pump_harmonics=args.pump_harmonics, pump_nt=args.pump_nt,
        multitone_sidebands=args.multitone_sidebands,
        z0_ohm=args.z0_ohm, outdir=args.outdir,
    )
    elapsed_s = time.perf_counter() - t0

    print(
        f"fitted: fp={result['pump_freq_ghz']:.4f} GHz  "
        f"I_p={result['pump_current_a']:.4g} A  "
        f"on-chip pump={result['on_chip_pump_dbm_norton']:.2f} dBm (Norton)  "
        f"[{elapsed_s:.1f}s]"
    )
    model_band: BandSummary = result["model_band"]
    measured_band: BandSummary = result["measured_band"]
    print(
        f"model band:    peak={model_band.peak_gain_db:.2f} dB @ "
        f"{model_band.peak_freq_ghz:.4f} GHz  bw3db={model_band.bandwidth_3db_ghz:.3f} GHz"
    )
    print(
        f"measured band: peak={measured_band.peak_gain_db:.2f} dB @ "
        f"{measured_band.peak_freq_ghz:.4f} GHz  bw3db={measured_band.bandwidth_3db_ghz:.3f} GHz"
    )
    print(
        f"peak delta: {model_band.peak_gain_db - measured_band.peak_gain_db:+.2f} dB, "
        f"{1000.0 * (model_band.peak_freq_ghz - measured_band.peak_freq_ghz):+.0f} MHz"
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = _write_overlay_and_surface_plot(result, args.outdir)
    print(f"wrote {png}")

    summary_json = args.outdir / "fit_operating_point.json"
    payload = _json_safe(result)
    payload["elapsed_s"] = elapsed_s
    payload["args"] = vars(args) | {
        "cube_path": str(args.cube_path), "circuit_dir": str(args.circuit_dir),
        "outdir": str(args.outdir),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
