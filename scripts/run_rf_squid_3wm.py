"""Run the staged rf-SQUID 3WM validation design."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

from twpa_solver.core import PortEnvironment
from twpa_solver.core.circuit import load_circuit
from twpa_solver.core.linear import port_s_from_unit_current_response
from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.design import load_design
from twpa_solver.design.__main__ import main as compile_design_main
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import PumpBasis, resolve_pump_basis
from twpa_solver.ports import port_current_from_power_a
from twpa_solver.signal.floquet import solve_gain_one
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.io import PumpSolution
from twpa_solver.signal.passive import db20, passive_s_matrix


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--phi-ext-over-phi0", type=float, default=0.33)
    parser.add_argument("--pump-ghz", type=float, default=12.08)
    parser.add_argument("--pump-dbm", type=float, default=-56.0)
    parser.add_argument("--signal-ghz", type=float, default=6.04)
    parser.add_argument("--signal-dbm", type=float, default=-126.0)
    parser.add_argument("--passive-start-ghz", type=float, default=4.0)
    parser.add_argument("--passive-stop-ghz", type=float, default=25.0)
    parser.add_argument("--passive-points", type=int, default=1401)
    parser.add_argument("--pump-harmonics", type=int, default=3)
    parser.add_argument("--sidebands", type=int, default=3)
    parser.add_argument("--pump-nt", type=int, default=32)
    parser.add_argument("--max-ell", type=int, default=6)
    parser.add_argument("--no-pump", action="store_true")
    parser.add_argument("--environment", choices=("ideal", "paper_standing_wave"),
                        default="ideal")
    return parser


def _dc_flux(branch_count: int, fraction: float, *, ic: float, lm: float) -> tuple[np.ndarray, float, float]:
    phi0 = 2.067833848e-15 / (2.0 * math.pi)
    external_phase = fraction * 2.0 * math.pi
    beta_l = lm * ic / phi0
    phi_dc = brentq(
        lambda phase: phase - external_phase + beta_l * math.sin(phase),
        external_phase - beta_l - 0.5,
        external_phase + beta_l + 0.5,
    )
    return np.full(branch_count, phi_dc * phi0), phi_dc, beta_l


def _write_passive(args: argparse.Namespace, circuit: object, dc: np.ndarray) -> None:
    frequencies = np.linspace(args.passive_start_ghz, args.passive_stop_ghz,
                              args.passive_points)
    scattering = passive_s_matrix(
        args.outdir, frequencies * 1e9, ports=(1, 2), z0_ohm=50.0,
        dc_branch_flux=dc,
    )
    np.savez_compressed(
        args.outdir / "passive_sparameters.npz",
        freq_ghz=frequencies,
        s11_db=db20(scattering[:, 0, 0]),
        s21_db=db20(scattering[:, 1, 0]),
        s12_db=db20(scattering[:, 0, 1]),
        s22_db=db20(scattering[:, 1, 1]),
    )


def _settings() -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=20, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=30, gmres_maxiter=80, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
    )


def _run_pump(args: argparse.Namespace, circuit: object, dc: np.ndarray) -> dict[str, object]:
    omega_p = 2.0 * math.pi * args.pump_ghz * 1e9
    basis = resolve_pump_basis(
        policy="dense_real", omega_p=omega_p, harmonics=args.pump_harmonics,
        mode_count=args.pump_harmonics, explicit_modes=None,
        design_meta={"features": {"dc_bias": True, "three_wave_mixing": True}},
    )
    pump_modes = np.concatenate((np.array([0], dtype=int), np.asarray(basis.modes, dtype=int)))
    pump_basis = PumpBasis(
        modes=pump_modes.tolist(), policy=basis.policy, omega_p=omega_p,
        source_mode=1,
    )
    power_w = 1e-3 * 10.0 ** (args.pump_dbm / 10.0)
    pump_current = port_current_from_power_a(power_w, 50.0,
                                              convention="legacy_traveling_wave")
    environment = PortEnvironment() if args.environment == "paper_standing_wave" else None
    problem = FullPumpProblem(
        C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
        branch=make_branch_law(circuit),
        grid=HarmonicGrid(pump_modes, nt=args.pump_nt, omega=omega_p),
        pump_node_index=circuit.port_to_index[1], pump_current_a=pump_current,
        dc_branch_flux=dc, environment=environment,
    )
    state, reports = HarmonicNewtonKrylovSolver(_settings()).solve_continuation(
        problem, continuation_steps=4)
    pump = PumpSolution(
        X=state, omega_p=omega_p, pump_freq_ghz=args.pump_ghz,
        harmonics=state.shape[0], nt_original=args.pump_nt,
        metadata={}, modes=pump_modes, basis=pump_basis,
    )
    gamma_hat = compute_gamma_hat(circuit, pump, args.max_ell, args.pump_nt, dc)
    khat = build_khat(circuit.Bphi, gamma_hat, 1e-30)
    gamma_off = make_branch_law(circuit).tangent(dc[None, :])[0]
    khat_off = (circuit.Bphi @ sp.diags(gamma_off) @ circuit.Bphi.T).astype(
        np.complex128).tocsr()
    gain = solve_gain_one(
        circuit, khat, khat_off, omega_p, args.signal_ghz, args.sidebands,
        signal_m=0, idler_m=-1, source_index=circuit.port_to_index[1],
        out_index=circuit.port_to_index[2], source_current_a=1.0,
        source_port=1, out_port=2, z0_ohm=50.0, environment=environment,
    )
    return {
        "pump_ghz": args.pump_ghz, "pump_dbm": args.pump_dbm,
        "signal_ghz": args.signal_ghz, "signal_dbm": args.signal_dbm,
        "pump_current_a": pump_current, "pump_modes": pump_modes,
        "pump_converged": reports[-1].converged,
        "pump_reports": [asdict(report) for report in reports],
        "signal_idler_relation": "f_idler = f_pump - f_signal",
        "idler_m": gain.idler_m, "gain_status": gain.status,
        "gain_db": gain.gain_db,
        "gain_vs_off_db": gain.gain_vs_off_db,
        "s21": abs(port_s_from_unit_current_response(
            gain.vout_on, source_port=1, out_port=2, z0_ohm=50.0)),
        "idler_conversion_db": gain.idler_power_rel_to_signal_off_db,
        "linear_rel_residual": gain.linear_rel_residual,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    compile_design_main([
        "--design", str(args.design), "--outdir", str(args.outdir),
        "--write-matrices", "--strict", "--overwrite",
    ])
    source = load_design(args.design)
    circuit = load_circuit(args.outdir)
    parameters = source["parameters"]
    dc, phi_dc, beta_l = _dc_flux(
        circuit.branch_count, args.phi_ext_over_phi0,
        ic=float(parameters["Ic"]), lm=float(parameters["Lm"]),
    )
    metadata: dict[str, object] = {
        "design": source["name"], "cells": int(source["parameters"]["cells"]),
        "phi_ext_over_phi0": args.phi_ext_over_phi0,
        "pump_ghz": args.pump_ghz, "pump_dbm": args.pump_dbm,
        "signal_ghz": args.signal_ghz, "signal_dbm": args.signal_dbm,
        "dc_branch_flux_wb": float(dc[0]),
        "phi_dc_rad": phi_dc,
        "beta_L": beta_l,
        "three_wave_mixing": True,
        "sideband_spacing": "pump_frequency",
    }
    _write_passive(args, circuit, dc)
    if not args.no_pump:
        metadata["pump_result"] = _run_pump(args, circuit, dc)
    (args.outdir / "rf_squid_3wm_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote={args.outdir / 'rf_squid_3wm_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
