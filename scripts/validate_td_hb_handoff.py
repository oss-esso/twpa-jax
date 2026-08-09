"""Minimal fixed-drive TD -> production HB handoff validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.h1_transient_branch_transfer import (  # noqa: E402
    build_system,
    checkpoint_dc_flux,
    dc_flux_from_external_fraction,
    implicit_trapezoid_ramp,
    project_periodic_state,
)
from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.pump.validation import validate_production_hb_state  # noqa: E402


def run(args: argparse.Namespace) -> dict[str, object]:
    restart = np.load(args.restart)
    y0 = np.asarray(restart["y"], dtype=float)
    theta0 = float(restart["theta"])
    current = float(restart["target_current"])
    report = json.loads((args.hb_checkpoint / "pump_report.json").read_text())
    data = np.load(args.hb_checkpoint / "pump_solution.npz")
    modes = np.asarray(report["metadata"].get("pump_modes", data["pump_modes"]), dtype=int)
    circuit = load_circuit(args.circuit_dir)
    fallback_dc = dc_flux_from_external_fraction(
        args.circuit_dir, float(getattr(args, "dc_flux_over_phi0", 0.0)), circuit.phi0
    )
    dc_flux = checkpoint_dc_flux(report, circuit, fallback_dc, args.hb_checkpoint)
    system = build_system(args.circuit_dir, args.freq_ghz, args.pump_port, dc_flux)

    total_theta = theta0 + 2.0 * np.pi * args.periods
    theta, states, integrator = implicit_trapezoid_ramp(
        system, y0, current, current, total_theta, 0.0,
        args.step_theta, args.newton_tol, args.max_newton,
        initial_theta=theta0,
    )
    if not integrator["success"]:
        return {"status": "TRANSIENT_NUMERICAL_FAILURE", "integrator": integrator}

    dense = lambda query: np.vstack([
        np.interp(query, theta, row) for row in states
    ])
    transfer = project_periodic_state(system, dense, float(theta[-1]), modes, current)
    hb_state = np.asarray(transfer.pop("hb_state"))
    validation = validate_production_hb_state(
        system.circuit, system.branch, frequency_hz=args.freq_ghz * 1e9,
        pump_port=args.pump_port, pump_current_a=current, modes=modes,
        state=hb_state, nt=max(2 * int(modes.max()) + 1, 40),
        metadata=report.get("metadata", {}),
        dc_branch_flux=dc_flux,
    )
    result = {
        "status": (
            "TD_TO_HB_RESTART_VALIDATED"
            if validation["checkpoint_validated"] else "HB_RESTART_VALIDATION_FAILED"
        ),
        "restart_checkpoint": str(args.restart),
        "hb_checkpoint": str(args.hb_checkpoint),
        "frequency_ghz": args.freq_ghz,
        "drive_current_a": current,
        "periods": args.periods,
        "d1": None,
        "integrator": integrator,
        "projected": {
            "projection_error_rms": transfer["projection_error_rms"],
            "projected_hb_coeff_rel": transfer["projected_hb_coeff_rel"],
            "projected_hb_time_rel": transfer["projected_hb_time_rel"],
        },
        "production_restart": {
            "hb_coeff_rel": transfer["hb_coeff_rel"],
            "hb_time_rel": transfer["hb_time_rel"],
            "newton_iterations": transfer["hb_newton_iterations"],
            "runtime_s": transfer["hb_runtime_s"],
            "validation": validation,
        },
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--hb-checkpoint", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--dc-flux-over-phi0", type=float, default=0.0)
    parser.add_argument("--periods", type=int, default=5)
    parser.add_argument("--step-theta", type=float, default=0.5)
    parser.add_argument("--newton-tol", type=float, default=1e-6)
    parser.add_argument("--max-newton", type=int, default=12)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
