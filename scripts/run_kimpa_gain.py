"""Run one-port KIMPA pump/Floquet reflection gain points."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from twpa_solver.builders.kimpa import KIMPA_FIXTURES, build_kimpa
from twpa_solver.core import PortEnvironment, kinetic_dc_branch_flux
from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.ports import port_available_power_w, port_current_from_power_a
from twpa_solver.core.linear import port_s_from_unit_current_response
from twpa_solver.pump import FullPumpProblem, HarmonicGrid, HarmonicNewtonKrylovSolver, NewtonKrylovSettings
from twpa_solver.pump.basis import resolve_pump_basis
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.gain import GainResult
from twpa_solver.signal.floquet import solve_gain_one
from twpa_solver.signal.io import PumpSolution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=tuple(KIMPA_FIXTURES), default="kimpa_fabricated_nominal")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pump-dbm", type=float, default=-29.6)
    parser.add_argument(
        "--pump-attenuation-db", type=float, default=0.0,
        help="Pump-line loss to subtract before driving the circuit; paper runs use 0 dB.",
    )
    parser.add_argument("--pump-ghz", type=float, default=16.94)
    parser.add_argument("--signal-ghz", type=float, default=8.47)
    parser.add_argument("--dc-current-a", type=float, default=550e-6)
    parser.add_argument("--sidebands", type=int, default=5)
    parser.add_argument("--max-ell", type=int, default=6)
    parser.add_argument("--pump-nt", type=int, default=32)
    parser.add_argument("--pump-mode-policy", choices=("dense_real", "positive_odd_jc"), default="dense_real")
    parser.add_argument("--pump-mode-count", type=int, default=3)
    parser.add_argument(
        "--environment", choices=("ideal", "paper_standing_wave"), default="ideal",
        help="Use the ideal 50-ohm termination or the quoted paper standing-wave environment.",
    )
    parser.add_argument("--no-waveforms", action="store_true", help="Do not save pump/current waveform arrays.")
    parser.add_argument("--no-solve", action="store_true", help="Only validate CLI/configuration and write metadata.")
    return parser


def _settings() -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=20, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=30, gmres_maxiter=80, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=False, verbose=False,
        continuation_predictor="none", jvp_mode="aft",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    circuit = build_kimpa(args.fixture)
    if args.dc_current_a != 0.0 and args.pump_mode_policy == "positive_odd_jc":
        raise ValueError("odd-only pump modes are invalid with non-zero KI DC bias; use --pump-mode-policy dense_real")
    omega_p = 2.0 * math.pi * args.pump_ghz * 1e9
    source_dbm = args.pump_dbm - args.pump_attenuation_db
    pump_power_w = 1.0e-3 * 10.0 ** (source_dbm / 10.0)
    pump_current = port_current_from_power_a(pump_power_w, 50.0, convention="legacy_traveling_wave")
    dc_flux = kinetic_dc_branch_flux(circuit, args.dc_current_a)
    environment = PortEnvironment() if args.environment == "paper_standing_wave" else None
    basis = resolve_pump_basis(
        policy=args.pump_mode_policy, omega_p=omega_p, harmonics=args.pump_mode_count,
        mode_count=args.pump_mode_count, explicit_modes=None, design_meta=circuit.metadata,
    )
    output: dict[str, object] = {
        "fixture": args.fixture, "pump_dbm_external": args.pump_dbm,
        "pump_attenuation_db": args.pump_attenuation_db,
        "pump_dbm_on_chip": 10.0 * math.log10(port_available_power_w(pump_current, 50.0) / 1e-3),
        "pump_current_a": pump_current, "pump_ghz": args.pump_ghz,
        "signal_ghz": args.signal_ghz, "dc_current_a": args.dc_current_a,
        "Lk_h": circuit.metadata.get("Lk_h"),
        "Lg_h": circuit.metadata.get("Lg_h"),
        "Ic_a": float(circuit.Ic[-1]),
        "Istar2_a": circuit.metadata.get("Istar2_a"),
        "Istar4_a": circuit.metadata.get("Istar4_a"),
        "dc_branch_flux": dc_flux.tolist(), "pump_modes": basis.modes,
        "norton_power_note": "I^2 Z/2 is 6.0206 dB above Norton available power I^2 Z/8.",
        "environment": args.environment,
    }
    if args.no_solve:
        return output
    if args.pump_nt < 2 * max(basis.modes) + 1 or args.pump_nt % 2:
        raise ValueError("--pump-nt must be even and at least 2*max(pump mode)+1")
    problem = FullPumpProblem(
        C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
        branch=make_branch_law(circuit),
        grid=HarmonicGrid(np.asarray(basis.modes), nt=args.pump_nt, omega=omega_p),
        pump_node_index=circuit.port_to_index[1], pump_current_a=pump_current,
        dc_branch_flux=dc_flux,
    )
    state, reports = HarmonicNewtonKrylovSolver(_settings()).solve_continuation(problem, continuation_steps=4)
    final = reports[-1]
    pump = PumpSolution(
        X=state, omega_p=omega_p, pump_freq_ghz=args.pump_ghz,
        harmonics=state.shape[0], nt_original=args.pump_nt, metadata={},
        modes=basis.modes, basis=basis,
    )
    psi = problem.branch_flux_time(state)
    total_current = problem.branch.current(psi + dc_flux[None, :])
    ratio = np.max(np.abs(total_current), axis=0) / circuit.Ic
    output.update({
        "pump_converged": final.converged, "pump_coeff_rel": final.coeff_rel,
        "max_current_over_ic": float(np.max(ratio)),
        "kinetic_status": "SUPERCONDUCTING" if np.all(ratio < 1.0) else "THRESHOLD_CROSSED",
        "pump_reports": [asdict(report) for report in reports],
    })
    gamma_hat = compute_gamma_hat(circuit, pump, args.max_ell, args.pump_nt, dc_flux)
    khat = build_khat(circuit.Bphi, gamma_hat, 1e-30)
    gamma_off = make_branch_law(circuit).tangent(dc_flux[None, :])[0]
    khat_off = (circuit.Bphi @ sp.diags(gamma_off) @ circuit.Bphi.T).astype(np.complex128).tocsr()
    gain: GainResult = solve_gain_one(
        circuit, khat, khat_off, omega_p, args.signal_ghz, args.sidebands,
        signal_m=0, idler_m=-1, source_index=circuit.port_to_index[1],
        out_index=circuit.port_to_index[1], source_current_a=1.0,
        source_port=1, out_port=1, z0_ohm=50.0,
        environment=environment,
    )
    output.update({
        "gain_status": gain.status, "gain_db": gain.gain_db,
        "gain_vs_off_db": gain.gain_vs_off_db,
        "s11_real": float(np.real(port_s_from_unit_current_response(gain.vout_on, source_port=1, out_port=1, z0_ohm=50.0))),
        "s11_imag": float(np.imag(port_s_from_unit_current_response(gain.vout_on, source_port=1, out_port=1, z0_ohm=50.0))),
        "s11_phase_deg": float(np.angle(port_s_from_unit_current_response(gain.vout_on, source_port=1, out_port=1, z0_ohm=50.0), deg=True)),
        "idler_ghz": float(args.pump_ghz - args.signal_ghz),
        "idler_m": gain.idler_m,
        "idler_power_rel_to_signal_off_db": gain.idler_power_rel_to_signal_off_db,
        "linear_rel_residual": gain.linear_rel_residual,
        "gamma_hat_odd_1_max": float(np.max(np.abs(gamma_hat.get(1, np.zeros(circuit.branch_count))))),
        "kinetic_out_of_domain_samples": int(getattr(circuit.branch_law, "out_of_domain_samples", 0)),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_waveforms:
        waveform_path = args.output_dir / "kimpa_gain_waveforms.npz"
        np.savez(
            waveform_path,
            pump_state=state,
            branch_flux_time=psi,
            branch_current_time=total_current,
            branch_current_spectrum=np.fft.rfft(total_current, axis=0),
            dc_branch_flux=dc_flux,
        )
        output["waveform_file"] = waveform_path.name
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    path = args.output_dir / "kimpa_gain.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
