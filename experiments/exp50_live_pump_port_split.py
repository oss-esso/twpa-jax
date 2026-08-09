"""Measure the pump-tone power split across all 4 ports on the LIVE 2c circuit.

Answers open question #1 from the 2026-08-05 saturation-gap research: prior
port-split numbers ([[2c-pump-depletion-measured-at-wrong-port]], 9.1% port 2
/ 90.2% port 3) were measured on the stale ``outputs/ipm_python_design``
circuit. This re-measures on the live ``designs/ipm_2c_fixed`` device at the
exp31 matched operating point (7.100 GHz, 7.2311074707853736e-06 A) with a
pump-only (signal-off) multitone solve, so the result is a linear property of
the coupler geometry, not a saturation artifact.

Reuses the exact pump-solve and multitone-basis machinery from
``scripts/run_compression.py`` (imported as a module) so the port split is
measured on the identical solve path production compression sweeps use, then
reads all 4 ports' pump-tone power waves via ``extract_port_waves``.
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

from twpa_solver.core.kinetic import kinetic_dc_branch_flux  # noqa: E402
from twpa_solver.multitone.observables import (  # noqa: E402
    extract_port_waves,
    power_balance,
)
from twpa_solver.multitone.preconditioners import (  # noqa: E402
    resolve_multitone_preconditioner,
)
from twpa_solver.multitone.problem import FullMultiToneProblem  # noqa: E402
from twpa_solver.multitone.schur import build_multitone_schur_problem  # noqa: E402
from twpa_solver.multitone.seed import promote_pump_solution  # noqa: E402
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive  # noqa: E402
from twpa_solver.pump import HarmonicNewtonKrylovSolver, NewtonKrylovSettings  # noqa: E402

OUTPUT = ROOT / "outputs" / "exp50_live_pump_port_split"

# exp31 matched operating point on designs/ipm_2c_fixed (see
# [[real-designs-live-in-designs-not-outputs]]): fp=7.100 GHz,
# I=7.2311074707853736e-06 A, rms 1.247 dB against measured G0.
PUMP_FREQ_GHZ = 7.100
PUMP_CURRENT_A = 7.231074707853736e-06
Z0_OHM = 50.0


def solve_pump_only(args) -> tuple:
    """Return (basis, circuit, pump_only_state_full, pump_tone)."""
    circuit, metadata, _ = rc._load_source(args)
    dc_branch_flux = kinetic_dc_branch_flux(circuit, args.dc_current_a)
    source_port = int(args.source_port or 1)
    pump_port = rc._resolve_pump_port(args, source_port)
    pump_current = float(args.pump_current_a) * float(args.pump_current_jc_scale)
    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9

    pump_state, pump_basis, _pump_reports, _pump_problem = rc._solve_pump_from_scratch(
        args, circuit, metadata, pump_port, pump_current, omega_p
    )

    # Signal frequency only sets the (h, q=-1) sideband spacing; it does not
    # affect the pump-only (q=0) solve, so any in-band value works here.
    omega_s = 2.0 * math.pi * args.signal_ghz * 1e9
    delta = omega_p - omega_s
    basis = rc._build_multitone_basis(args, pump_basis.modes, omega_p, delta)
    pump_seed = promote_pump_solution(pump_state, pump_basis, basis)
    pump_source = MultiToneDrive(
        basis.pump_tone, circuit.port_to_index[pump_port], pump_current
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

    full = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(pump_source),
        preconditioner=selected_preconditioner, cache={}, dc_branch_flux=dc_branch_flux,
    )
    reduced = build_multitone_schur_problem(
        full, list(circuit.port_to_index.values()), preconditioner=selected_preconditioner,
    )
    pump_seed_solve = pump_seed[:, reduced.partition.retained]

    pump_only_state, pump_only_report = HarmonicNewtonKrylovSolver(settings).solve_one(
        reduced, pump_seed_solve, 1.0,
    )
    if not pump_only_report.converged:
        pump_only_state, pump_only_reports, pump_only_trace = (
            HarmonicNewtonKrylovSolver(settings).solve_adaptive_continuation(
                reduced, None, initial_step=0.25, min_step=0.01, growth=1.5,
                shrink=0.5, fallback_fixed_steps=20, max_wall_s=120.0,
            )
        )
        if (
            not pump_only_reports or not pump_only_reports[-1].converged
            or not pump_only_trace.accepted_lambdas
            or pump_only_trace.accepted_lambdas[-1] < 1.0 - 1e-12
        ):
            reason = (
                pump_only_reports[-1].failure_reason
                if pump_only_reports else pump_only_trace.failure_reason
            )
            raise RuntimeError(f"pump-only solve did not converge: {reason}")

    pump_only_state_full = reduced.reconstruct_full(pump_only_state)
    return basis, circuit, pump_only_state_full, pump_port, source_port, dc_branch_flux


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parser = rc.build_parser()
    args = parser.parse_args([
        "--output-dir", str(OUTPUT / "_unused"),
        "--circuit-dir", "designs/ipm_2c_fixed",
        "--pump-freq-ghz", str(PUMP_FREQ_GHZ),
        "--pump-current-a", str(PUMP_CURRENT_A),
        "--signal-ghz", "7.4",
        "--multitone-basis", "matched",
        "--multitone-sidebands", "2",
    ])

    basis, circuit, state_full, pump_port, source_port, dc_branch_flux = solve_pump_only(args)
    pump_tone = basis.pump_tone

    waves = extract_port_waves(
        state_full, basis, circuit, tuple(circuit.port_to_index), z0_ohm=Z0_OHM,
        dc_branch_flux=dc_branch_flux,
    )
    balance = power_balance(state_full, basis, circuit, z0_ohm=Z0_OHM)

    per_port = {}
    total_outgoing_w = 0.0
    total_incoming_w = 0.0
    for port in sorted(circuit.port_to_index):
        outgoing_w = 0.5 * waves["b_power"][(pump_tone, port)]
        incoming_w = 0.5 * waves["a_power"][(pump_tone, port)]
        total_outgoing_w += outgoing_w
        total_incoming_w += incoming_w
        per_port[port] = {"outgoing_w": outgoing_w, "incoming_w": incoming_w}

    for port in per_port:
        per_port[port]["outgoing_fraction"] = (
            per_port[port]["outgoing_w"] / total_outgoing_w if total_outgoing_w > 0 else float("nan")
        )
        per_port[port]["outgoing_dbm"] = 10.0 * math.log10(
            max(per_port[port]["outgoing_w"], 1e-300) / 1e-3
        )

    cross_check_rel_err = (
        abs(total_outgoing_w - balance["pump_outgoing_power_w"])
        / max(total_outgoing_w, 1e-300)
    )

    report = {
        "circuit_dir": "designs/ipm_2c_fixed",
        "pump_freq_ghz": PUMP_FREQ_GHZ,
        "pump_current_a": PUMP_CURRENT_A,
        "pump_port": pump_port,
        "source_port": source_port,
        "multitone_basis": "matched",
        "multitone_sidebands": 2,
        "z0_ohm": Z0_OHM,
        "per_port": {str(k): v for k, v in per_port.items()},
        "total_outgoing_pump_power_w": total_outgoing_w,
        "power_balance_pump_outgoing_power_w": balance["pump_outgoing_power_w"],
        "cross_check_rel_err": cross_check_rel_err,
        "power_balance_rel_err": balance["power_balance_rel_err"],
    }
    (OUTPUT / "port_split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"pump-only solve on designs/ipm_2c_fixed, fp={PUMP_FREQ_GHZ} GHz, "
          f"I={PUMP_CURRENT_A:.6e} A, pump_port={pump_port}")
    print(f"power_balance_rel_err={balance['power_balance_rel_err']:.3e} "
          f"(sanity: internal supplied vs dissipated)")
    print(f"cross-check: sum(per-port outgoing) vs power_balance's own "
          f"pump_outgoing_power_w, rel_err={cross_check_rel_err:.3e}")
    print()
    print(f"{'port':>5} {'outgoing_dBm':>13} {'outgoing_frac':>14} {'incoming_w':>14} {'outgoing_w':>14}")
    for port in sorted(per_port):
        p = per_port[port]
        print(f"{port:5d} {p['outgoing_dbm']:13.3f} {100 * p['outgoing_fraction']:13.4f}% "
              f"{p['incoming_w']:14.6e} {p['outgoing_w']:14.6e}")
    print()
    print(f"wrote {OUTPUT / 'port_split_report.json'}")


if __name__ == "__main__":
    main()
