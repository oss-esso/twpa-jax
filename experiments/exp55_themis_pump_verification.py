"""Compare simulation against the Themis Jan28 saturation dataset at its own
literal calibrated pump setting.

The Jan28 cube (10.15.34_Themis_SetupJan28_VTS_transmission_15mK) is the one
Themis dataset with a genuine signal-power sweep -- pump held fixed at
PumpFrequency=7.256 GHz, PumpPower=-21 dBm (both read directly from the
dataset, not fitted). Every prior comparison this session used a pump
current that was fit to reproduce measured gain, never the literal
calibrated dBm. This script uses --pump-power-dbm (added for exactly this)
to derive on-chip pump current straight from -21 dBm via
pump_line_loss_model() at 7.256 GHz, run the compression curve at one
measured signal frequency, and plot both curves on the same axis.

The fixed 4-step pump continuation ladder inside run_compression.py stalls
at this pump current (harder operating point than the historically-fit
one), so the pump is solved here with an escalating-ladder + adaptive
fallback (same pattern as exp51/exp52) and handed to run_compression.py via
--pump-solution-dir.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_compression as rc  # noqa: E402

from twpa_solver.core import default_loss_model_for  # noqa: E402
from twpa_solver.core.kinetic import kinetic_dc_branch_flux  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.pump import HarmonicGrid, HarmonicNewtonKrylovSolver, NewtonKrylovSettings  # noqa: E402
from twpa_solver.pump.basis import resolve_pump_basis  # noqa: E402
from twpa_solver.pump.problem import FullPumpProblem  # noqa: E402
from twpa_solver.pump.io import summarize_solution, write_results  # noqa: E402

CUBE = ROOT / (
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
PUMP_FREQ_GHZ = 7.256
PUMP_POWER_DBM = -21.0
SIGNAL_GHZ = 7.052
SIGNAL_POWER_MIN_DBM = -60.0
SIGNAL_POWER_MAX_DBM = 6.0
N_SIGNAL_POWER = 16
OUTPUT = ROOT / "outputs" / "exp55_themis_pump_verification"


def _solve_pump_robust(args, circuit, metadata, pump_port, pump_current, omega_p):
    """Escalating fixed-step ladder, then adaptive fallback -- same pattern
    as exp51/exp52's ``_solve_pump_robust``, needed because -21 dBm's
    on-chip current is a harder pump operating point than the historically
    fit 7.2311e-6 A that run_compression.py's fixed 4-step ladder was tuned
    against.
    """
    pump_basis = resolve_pump_basis(
        policy=args.pump_mode_policy, omega_p=omega_p, harmonics=args.pump_harmonics,
        mode_count=args.pump_mode_count, explicit_modes=args.pump_modes,
        design_meta=metadata,
    )
    pump_problem = FullPumpProblem(
        C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
        branch=make_branch_law(circuit),
        grid=HarmonicGrid(np.asarray(pump_basis.modes), nt=args.pump_nt, omega=omega_p),
        pump_node_index=circuit.port_to_index[pump_port], pump_current_a=pump_current,
        dc_branch_flux=kinetic_dc_branch_flux(circuit, args.dc_current_a),
        loss_model=default_loss_model_for(circuit),
    )
    pump_settings = NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=20, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=20, gmres_maxiter=40, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled",
        compute_time_residual=False, verbose=False,
        continuation_predictor="none", jvp_mode="aft",
        precond_reuse=1, precond_reuse_refresh_gmres=0,
    )
    solver = HarmonicNewtonKrylovSolver(pump_settings)
    for continuation_steps in (4, 20, 60):
        pump_state, pump_reports = solver.solve_continuation(
            pump_problem, continuation_steps=continuation_steps
        )
        if pump_reports[-1].converged:
            print(f"[exp55] pump solve converged with continuation_steps={continuation_steps}")
            return pump_state, pump_basis, pump_reports, pump_problem
        print(f"[exp55] continuation_steps={continuation_steps} stalled: "
              f"coeff_rel={pump_reports[-1].coeff_rel:.6g}, "
              f"reason={pump_reports[-1].failure_reason}")
    pump_state, pump_reports, trace = solver.solve_adaptive_continuation(
        pump_problem, None, initial_step=0.1, min_step=0.002, growth=1.3,
        shrink=0.5, fallback_fixed_steps=40, max_wall_s=180.0,
    )
    if (
        not pump_reports or not pump_reports[-1].converged
        or not trace.accepted_lambdas or trace.accepted_lambdas[-1] < 1.0 - 1e-12
    ):
        reason = pump_reports[-1].failure_reason if pump_reports else trace.failure_reason
        raise RuntimeError(f"pump solve did not converge even with adaptive fallback: {reason}")
    print("[exp55] pump solve converged via adaptive fallback")
    return pump_state, pump_basis, pump_reports, pump_problem


def solve_pump_and_save(pump_dir: Path, convention: str) -> None:
    pump_dir.mkdir(parents=True, exist_ok=True)
    parser = rc.build_parser()
    args = parser.parse_args([
        "--output-dir", "unused",
        "--circuit-dir", "designs/ipm_2c_fixed",
        "--pump-freq-ghz", str(PUMP_FREQ_GHZ),
        "--pump-power-dbm", str(PUMP_POWER_DBM),
        "--signal-ghz", str(SIGNAL_GHZ),
        "--power-convention", convention,
    ])
    circuit, metadata, _ = rc._load_source(args)
    source_port = int(args.source_port or 1)
    pump_port = rc._resolve_pump_port(args, source_port)
    pump_current, source = rc._resolve_pump_current_a(args, None)
    print(f"[exp55] pump_current={pump_current:.6e} A (source={source})")
    omega_p = 2.0 * math.pi * PUMP_FREQ_GHZ * 1e9
    pump_state, pump_basis, pump_reports, pump_problem = _solve_pump_robust(
        args, circuit, metadata, pump_port, pump_current, omega_p
    )
    solution_summary = summarize_solution(pump_problem, pump_state)
    write_results(
        pump_dir, pump_state, pump_reports, solution_summary,
        {
            **pump_basis.to_metadata(),
            "pump_freq_ghz": PUMP_FREQ_GHZ,
            "pump_current_a": pump_current,
            "nt": args.pump_nt,
        },
    )
    print(f"[exp55] wrote pump solution to {pump_dir}")


def run_signal_sweep(pump_dir: Path, sweep_dir: Path, convention: str) -> dict:
    command = [
        sys.executable, "-u", "scripts/run_compression.py",
        "--output-dir", str(sweep_dir),
        "--circuit-dir", "designs/ipm_2c_fixed",
        "--pump-freq-ghz", str(PUMP_FREQ_GHZ),
        "--pump-power-dbm", str(PUMP_POWER_DBM),
        "--pump-solution-dir", str(pump_dir),
        "--signal-ghz", str(SIGNAL_GHZ),
        "--signal-power-min-dbm", str(SIGNAL_POWER_MIN_DBM),
        "--signal-power-max-dbm", str(SIGNAL_POWER_MAX_DBM),
        "--n-signal-power", str(N_SIGNAL_POWER),
        "--multitone-basis", "matched", "--multitone-sidebands", "10",
        "--recovery", "ladder", "--factor-backend", "pardiso",
        "--allow-memory-overcommit",
        "--signal-continuation-deadline-s", "240",
        "--power-convention", convention,
    ]
    print("[exp55] running:", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (sweep_dir).mkdir(parents=True, exist_ok=True)
    (sweep_dir / "run_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (sweep_dir / "run_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary_path = sweep_dir / "compression_summary.json"
    if not summary_path.exists():
        tail = completed.stderr.strip().splitlines()[-15:]
        raise RuntimeError("signal sweep failed:\n" + "\n".join(tail))
    return json.loads(summary_path.read_text(encoding="utf-8"))


def measured_curve(target_ghz: float) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.load(CUBE, allow_pickle=True).item()
    freq = np.asarray(data["Frequency"], dtype=float) / 1e9
    sp = np.asarray(data["SignalPower"], dtype=float)
    resp = np.asarray(data["Response"], dtype=float)
    idx = int(np.argmin(np.abs(freq - target_ghz)))
    actual_freq = float(freq[idx])
    return sp, resp[:, idx], actual_freq


def measured_curve_gain_matched(
    target_gain_db: float, pump_exclusion_ghz: float = 0.15, pump_ghz: float = PUMP_FREQ_GHZ
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pick the measured frequency column whose own small-signal gain (median
    of the 10 lowest-power points) is closest to ``target_gain_db``, instead
    of matching by frequency. Same "compare at equal gain, not equal
    frequency" methodology as exp53 -- this device's gain ripples with
    frequency, so a frequency match confounds the compression comparison
    with wherever that frequency happens to sit in the ripple.
    """
    data = np.load(CUBE, allow_pickle=True).item()
    freq = np.asarray(data["Frequency"], dtype=float) / 1e9
    sp = np.asarray(data["SignalPower"], dtype=float)
    resp = np.asarray(data["Response"], dtype=float)
    g0 = np.median(resp[:10, :], axis=0)
    usable = np.abs(freq - pump_ghz) > pump_exclusion_ghz
    diff = np.abs(g0 - target_gain_db)
    diff[~usable] = np.inf
    idx = int(np.argmin(diff))
    return sp, resp[:, idx], float(freq[idx]), float(g0[idx])


def run_convention(convention: str) -> dict:
    """Solve pump+sweep for one power convention, reusing a prior run's
    output dir if it already has a finished sweep (so re-running to add a
    second convention doesn't re-solve the first one)."""
    conv_dir = OUTPUT / f"conv_{convention}"
    pump_dir = conv_dir / "pump"
    sweep_dir = conv_dir / "sweep"
    summary_path = sweep_dir / "compression_summary.json"
    if summary_path.exists():
        print(f"[exp55] reusing existing {convention} sweep: {sweep_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        solve_pump_and_save(pump_dir, convention)
        summary = run_signal_sweep(pump_dir, sweep_dir, convention)
    import csv
    rows = list(csv.DictReader((sweep_dir / "compression_points.csv").open(encoding="utf-8")))
    return {
        "convention": convention,
        "summary": summary,
        "power_dbm": np.array([float(r["signal_power_dbm"]) for r in rows]),
        "gain_db": np.array([float(r["gain_vs_off_db"]) for r in rows]),
        "status": [r["status"] for r in rows],
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # legacy_traveling_wave first: this session's earlier run already lives
    # at OUTPUT/pump + OUTPUT/sweep (not conv_-prefixed) -- point the legacy
    # run there directly so it's reused instead of re-solved.
    legacy_sweep_dir = OUTPUT / "sweep"
    if (legacy_sweep_dir / "compression_summary.json").exists():
        print(f"[exp55] reusing existing legacy_traveling_wave sweep: {legacy_sweep_dir}")
        import csv
        rows = list(csv.DictReader((legacy_sweep_dir / "compression_points.csv").open(encoding="utf-8")))
        legacy = {
            "convention": "legacy_traveling_wave",
            "summary": json.loads((legacy_sweep_dir / "compression_summary.json").read_text(encoding="utf-8")),
            "power_dbm": np.array([float(r["signal_power_dbm"]) for r in rows]),
            "gain_db": np.array([float(r["gain_vs_off_db"]) for r in rows]),
            "status": [r["status"] for r in rows],
        }
    else:
        legacy = run_convention("legacy_traveling_wave")
    # Norton is NOT re-attempted here: at this exact pump setting its 2x
    # current pushes the hottest junction's peak branch current to ~1.006x Ic
    # (linear-scaled from the legacy solution's measured 0.503x Ic), which is
    # why its pump continuation stalls near lambda=0.6 rather than a plain
    # numerical convergence issue -- see the report.json from that run and
    # [[2c-norton-pump-exceeds-critical-current]] for the check.
    sim_gain_db = float(legacy["summary"]["small_signal_gain_vs_off_db"])

    meas_power_dbm, meas_gain_db, actual_meas_freq = measured_curve(SIGNAL_GHZ)
    gm_power_dbm, gm_gain_db, gm_freq, gm_g0 = measured_curve_gain_matched(sim_gain_db)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(meas_power_dbm, meas_gain_db, "-", color="tab:orange", lw=1.2,
              label=f"measured, same freq ({actual_meas_freq:.3f} GHz, G0={np.median(meas_gain_db[:10]):.2f} dB)")
    axis.plot(gm_power_dbm, gm_gain_db, "-", color="tab:purple", lw=1.2,
              label=f"measured, gain-matched ({gm_freq:.3f} GHz, G0={gm_g0:.2f} dB)")
    power_dbm, gain_db, status = legacy["power_dbm"], legacy["gain_db"], legacy["status"]
    valid = [s == "VALID_SOLVED" for s in status]
    axis.plot(power_dbm[valid], gain_db[valid], "o-", color="tab:blue", lw=1.5, ms=6,
              label=f"simulation ({SIGNAL_GHZ} GHz, pump {PUMP_POWER_DBM} dBm @ {PUMP_FREQ_GHZ} GHz, G0={sim_gain_db:.2f} dB)")
    invalid = [not v for v in valid]
    if any(invalid):
        axis.plot(power_dbm[invalid], gain_db[invalid], "x", color="tab:blue", ms=8,
                  label="simulation (not converged)")
    axis.set_xlabel("source-referred signal power (dBm)")
    axis.set_ylabel("gain vs pump-off (dB)")
    axis.set_title("2c: simulation vs. Themis Jan28, same-frequency and gain-matched measured curves")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    figure.savefig(OUTPUT / "pump_calibration_verification.png", dpi=150)
    plt.close(figure)

    report = {
        "pump_freq_ghz": PUMP_FREQ_GHZ,
        "pump_power_dbm": PUMP_POWER_DBM,
        "signal_ghz_requested": SIGNAL_GHZ,
        "signal_ghz_measured_actual": actual_meas_freq,
        "meas_small_signal_gain_db_same_freq": float(np.median(meas_gain_db[:10])),
        "gain_matched_freq_ghz": gm_freq,
        "gain_matched_g0_db": gm_g0,
        "sim_small_signal_gain_vs_off_db": sim_gain_db,
        "sim_status": legacy["summary"].get("status"),
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUTPUT / 'pump_calibration_verification.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
