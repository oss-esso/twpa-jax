"""Distributed (branch-resolved) pump/signal profile on the live 2c circuit.

Follow-up to [[2c-reference-plane-hypothesis-overshoots]]: a single
lumped-port pump-coupling correction to the depletion-only P1dB bound
overshoots rather than resolving the "model compresses early" gap. The next
candidate explanation is that the pump field is spatially inhomogeneous along
the JTL chain (consistent with [[2c-standing-wave-not-traveling-wave]]'s
resonant, non-traveling-wave finding), so a device-averaged depletion
fraction (~1% at P1dB) can coexist with much larger LOCAL depletion at a
concentrated subset of branches -- those hot branches, not the port-averaged
total, would be what actually limits gain.

Solves two signal-power points at the exp31/[[2c-live-pump-port-split-measured]]
operating point (fp=7.100 GHz, I_p=7.2311e-6 A) on ``designs/ipm_2c_fixed``:
a deep small-signal reference (I=1e-12 A) and the on-chip P1dB current at
fs=7.2 GHz read from ``outputs/exp45_2c_p1db_vs_frequency`` (on-chip
P1dB=-86.08 dBm at this frequency, already Norton/loss_B1-corrected). Computes
``spatial_profiles`` at both points and:

  1. the distributed depletion-only gain estimate (``spatial_depletion_null``)
     against the actual nonlinear-solved gain -- does a per-branch model
     reproduce compression that the lumped port model could not?
  2. the spatial concentration of pump intensity (``spatial_profile_summary``)
     -- is the pump amplitude uniform along the chain or concentrated?
  3. a per-branch local depletion fraction from the pump-flux power ratio
     between the two states, to find where along the chain depletion is
     actually large.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_compression as rc  # noqa: E402

from twpa_solver.core import default_loss_model_for  # noqa: E402
from twpa_solver.core.kinetic import kinetic_dc_branch_flux  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.multitone.compression import solve_signal_power_point  # noqa: E402
from twpa_solver.multitone.observables import (  # noqa: E402
    power_balance,
    spatial_depletion_null,
    spatial_profile_summary,
    spatial_profiles,
    tone_s21,
)
from twpa_solver.multitone.preconditioners import (  # noqa: E402
    resolve_multitone_preconditioner,
)
from twpa_solver.multitone.problem import FullMultiToneProblem  # noqa: E402
from twpa_solver.multitone.schur import build_multitone_schur_problem  # noqa: E402
from twpa_solver.multitone.seed import promote_pump_solution  # noqa: E402
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive  # noqa: E402
from twpa_solver.ports import port_current_from_power_a  # noqa: E402
from twpa_solver.pump import (  # noqa: E402
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import resolve_pump_basis  # noqa: E402

OUTPUT = ROOT / "outputs" / "exp51_spatial_depletion_profile"


def _solve_pump_robust(args, circuit, metadata, pump_port, pump_current, omega_p):
    """Like ``rc._solve_pump_from_scratch`` but falls back past a fixed
    4-step ladder when disorder (Lj/Cj scatter) shifts the pump fold away
    from the nominal circuit's operating point, instead of just raising.
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
            print(f"  pump solve converged with continuation_steps={continuation_steps}")
            return pump_state, pump_basis, pump_reports
        print(f"  continuation_steps={continuation_steps} stalled: "
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
    print("  pump solve converged via adaptive continuation fallback")
    return pump_state, pump_basis, pump_reports

PUMP_FREQ_GHZ = 7.100
PUMP_CURRENT_A = 7.231074707853736e-06
SIGNAL_GHZ = 7.2
# On-chip (Norton, loss_B1-corrected) P1dB at fs=7.2 GHz, op7.100, read from
# outputs/exp45_2c_p1db_vs_frequency (see the an6.py scratch analysis in the
# port-split session): -86.08 dBm.
P1DB_ON_CHIP_DBM = -86.08
REFERENCE_CURRENT_A = 1e-12
Z0_OHM = 50.0


def main(circuit_dir: str = "designs/ipm_2c_fixed", label: str | None = None) -> dict:
    label = label or Path(circuit_dir).name
    outdir = OUTPUT / label
    outdir.mkdir(parents=True, exist_ok=True)
    parser = rc.build_parser()
    args = parser.parse_args([
        "--output-dir", str(outdir / "_unused"),
        "--circuit-dir", circuit_dir,
        "--pump-freq-ghz", str(PUMP_FREQ_GHZ),
        "--pump-current-a", str(PUMP_CURRENT_A),
        "--signal-ghz", str(SIGNAL_GHZ),
        "--multitone-basis", "matched",
        "--multitone-sidebands", "2",
    ])

    circuit, metadata, _ = rc._load_source(args)
    dc_branch_flux = kinetic_dc_branch_flux(circuit, args.dc_current_a)
    source_port = int(args.source_port or 1)
    pump_port = rc._resolve_pump_port(args, source_port)
    out_port = int(args.out_port or 2)
    pump_current = float(args.pump_current_a) * float(args.pump_current_jc_scale)
    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9

    pump_state, pump_basis, _pump_reports = _solve_pump_robust(
        args, circuit, metadata, pump_port, pump_current, omega_p
    )

    omega_s = 2.0 * math.pi * args.signal_ghz * 1e9
    delta = omega_p - omega_s
    basis = rc._build_multitone_basis(args, pump_basis.modes, omega_p, delta)
    pump_seed = promote_pump_solution(pump_state, pump_basis, basis)
    pump_source = MultiToneDrive(
        basis.pump_tone, circuit.port_to_index[pump_port], pump_current
    ).to_coeffs(basis, circuit.node_count)
    signal_unit = MultiToneDrive(
        basis.signal_tone, circuit.port_to_index[source_port], 1.0
    ).to_coeffs(basis, circuit.node_count)

    selected_preconditioner = resolve_multitone_preconditioner(
        args.multitone_preconditioner
    )
    solver_preconditioner = (
        "real_coupled_fast"
        if selected_preconditioner == "floquet_sector"
        else selected_preconditioner
    )
    settings = NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=20, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=20, gmres_maxiter=40, min_alpha=1.0 / 1024.0,
        preconditioner=solver_preconditioner,
        compute_time_residual=False, verbose=False,
        continuation_predictor="none", jvp_mode="aft",
        precond_reuse=1, precond_reuse_refresh_gmres=0,
    )

    problem_cache: dict[object, object] = {}
    schur_partition = None

    def make_problem(path: AffineSourcePath):
        nonlocal schur_partition
        full = FullMultiToneProblem(
            circuit, basis, path, preconditioner=selected_preconditioner,
            cache=problem_cache, dc_branch_flux=dc_branch_flux,
        )
        if schur_partition is None:
            reduced = build_multitone_schur_problem(
                full, list(circuit.port_to_index.values()),
                preconditioner=selected_preconditioner,
            )
            schur_partition = reduced.partition
            return reduced
        from twpa_solver.multitone.schur import SchurMultiToneProblem
        return SchurMultiToneProblem(full, schur_partition, preconditioner=selected_preconditioner)

    def observable_state(problem, state: np.ndarray) -> np.ndarray:
        return problem.reconstruct_full(state)

    # First call to make_problem() initializes schur_partition; the returned
    # problem is reused as the seed's reduction target. solve_signal_power_point
    # runs its own adaptive continuation from this seed for both currents below
    # (matching run_compression.py's per-point loop) -- no separate pump-only
    # Newton solve is needed since the reference current (1e-12 A) is already
    # deep in the small-signal regime.
    _ = make_problem(AffineSourcePath.pump_turn_on(pump_source))
    pump_seed_solve = pump_seed[:, schur_partition.retained]

    p1db_current_a = port_current_from_power_a(
        1.0e-3 * 10.0 ** (P1DB_ON_CHIP_DBM / 10.0), Z0_OHM, convention="norton"
    )
    print(f"target on-chip P1dB current at fs={SIGNAL_GHZ} GHz: {p1db_current_a:.6e} A "
          f"({P1DB_ON_CHIP_DBM} dBm)")

    solver = HarmonicNewtonKrylovSolver(settings)
    states_full: dict[str, np.ndarray] = {}
    gains: dict[str, float] = {}
    previous = pump_seed_solve
    previous_previous = None
    previous_current = 0.0
    previous_previous_current = 0.0
    for point_label, current in (("small_signal", REFERENCE_CURRENT_A), ("p1db", p1db_current_a)):
        base_problem = make_problem(AffineSourcePath.pump_turn_on(pump_source))
        solved = solve_signal_power_point(
            base_problem, previous, previous_previous, float(current),
            pump_source=pump_source, signal_source=signal_unit, solver=solver,
            signal_current_prev_a=previous_current,
            signal_current_prevprev_a=previous_previous_current,
            recovery="ladder", pump_seed=pump_seed_solve,
            signal_substep_init_db=0.5, signal_substep_min_db=0.01,
            continuation_deadline_s=0.0, arclength_recovery=False,
        )
        if solved.status != "VALID_SOLVED":
            raise RuntimeError(f"{point_label} point failed to solve: status={solved.status}")
        state_full = observable_state(base_problem, solved.state)
        states_full[point_label] = state_full
        signal_s21 = tone_s21(
            state_full, basis, circuit, signal_tone=basis.signal_tone,
            source_port=source_port, out_port=out_port,
            source_current_a=float(current), z0_ohm=Z0_OHM,
        )
        gains[point_label] = float(20.0 * np.log10(max(abs(signal_s21), 1e-300)))
        previous_previous, previous_previous_current = previous, previous_current
        previous, previous_current = solved.state, float(current)
        print(f"solved {point_label}: I={current:.4e} A, gain_db(paper-normalized)={gains[point_label]:.4f}")

    actual_compression_db = gains["small_signal"] - gains["p1db"]
    print(f"\nactual solved compression at P1dB current: {actual_compression_db:.4f} dB "
          f"(should be ~1.0 dB by construction of the target current)")

    rows_small = spatial_profiles(states_full["small_signal"], basis, circuit)
    rows_p1db = spatial_profiles(states_full["p1db"], basis, circuit)

    summary_small = spatial_profile_summary(rows_small)
    summary_p1db = spatial_profile_summary(rows_p1db)
    bound_db = spatial_depletion_null(rows_p1db, rows_small)

    print(f"\nspatial_profile_summary (small-signal reference): {summary_small}")
    print(f"spatial_profile_summary (P1dB operating point):    {summary_p1db}")
    print(f"\nspatial_depletion_null bound (distributed depletion-only gain estimate): "
          f"{bound_db:.4f} dB")
    print(f"reference (small-signal) end-to-end gain baked into that estimate: "
          f"{20.0 * math.log10(abs(rows_small[-1]['signal_flux_abs']) / abs(rows_small[0]['signal_flux_abs'])):.4f} dB")

    # Per-branch local pump depletion (power ratio between the two states).
    pump_small = np.asarray([r["pump_flux_abs"] for r in rows_small])
    pump_p1db = np.asarray([r["pump_flux_abs"] for r in rows_p1db])
    signal_small = np.asarray([r["signal_flux_abs"] for r in rows_small])
    local_depletion_db = 20.0 * np.log10(
        np.clip(pump_p1db, 1e-300, None) / np.clip(pump_small, 1e-300, None)
    )
    worst_branch = int(np.argmin(local_depletion_db))
    print(f"\nlocal pump depletion (dB, P1dB vs small-signal), per branch: "
          f"min={local_depletion_db.min():.4f} at branch {worst_branch}, "
          f"max={local_depletion_db.max():.4f}, median={np.median(local_depletion_db):.4f}")

    lumped_balance = power_balance(
        states_full["p1db"], basis, circuit,
        reference_X_full=states_full["small_signal"], z0_ohm=Z0_OHM,
    )
    lumped_depletion_db = lumped_balance["pump_depletion_all_port_db"]
    print(f"lumped (all-port) pump depletion at this point: {lumped_depletion_db:.4f} dB "
          f"-- vs local median {np.median(local_depletion_db):.4f} dB, "
          f"local worst-case {local_depletion_db.min():.4f} dB")

    # Where along the chain is the pump concentrated, and does the worst-depleted
    # branch coincide with the highest-pump-intensity region?
    pump_intensity_small = np.asarray([r["pump_intensity_normalized"] for r in rows_small])
    hot_branches = np.argsort(pump_intensity_small)[::-1][:10]
    print(f"\ntop-10 highest small-signal pump-intensity branches: {hot_branches.tolist()}")
    print(f"local depletion (dB) at those branches: "
          f"{[round(float(local_depletion_db[b]), 4) for b in hot_branches]}")

    report = {
        "circuit_dir": circuit_dir,
        "label": label,
        "pump_freq_ghz": PUMP_FREQ_GHZ,
        "pump_current_a": PUMP_CURRENT_A,
        "signal_ghz": SIGNAL_GHZ,
        "p1db_on_chip_dbm_target": P1DB_ON_CHIP_DBM,
        "p1db_current_a": p1db_current_a,
        "gains_db": gains,
        "actual_compression_db": actual_compression_db,
        "spatial_profile_summary_small_signal": summary_small,
        "spatial_profile_summary_p1db": summary_p1db,
        "spatial_depletion_null_bound_db": bound_db,
        "lumped_all_port_pump_depletion_db": lumped_depletion_db,
        "local_depletion_db_min": float(local_depletion_db.min()),
        "local_depletion_db_max": float(local_depletion_db.max()),
        "local_depletion_db_median": float(np.median(local_depletion_db)),
        "worst_branch": worst_branch,
        "hot_branches_top10": hot_branches.tolist(),
        "n_branches": circuit.branch_count,
    }
    (outdir / "spatial_profile_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez(
        outdir / "spatial_profile_arrays.npz",
        pump_small=pump_small, pump_p1db=pump_p1db, signal_small=signal_small,
        local_depletion_db=local_depletion_db,
        pump_intensity_small=pump_intensity_small,
    )
    print(f"\nwrote {outdir / 'spatial_profile_report.json'}")
    return report


if __name__ == "__main__":
    main()
