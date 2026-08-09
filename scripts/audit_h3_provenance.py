"""Audit and, if needed, production-correct the H3 7.9 GHz starting state."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import default_loss_model_for, load_circuit  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.pump.hb import FullPumpProblem, HarmonicGrid  # noqa: E402
from twpa_solver.pump.solver import (  # noqa: E402
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
)
from twpa_solver.pump.validation import validate_production_hb_state  # noqa: E402


def main() -> int:
    source = ROOT / "g1_current_79" / "pass" / "points" / "point_0012_p_m19p6842dbm_fp_7p9ghz" / "pump"
    outdir = ROOT / "outputs" / "h3_provenance_79"
    checkpoint = source / "pump_solution.npz"
    report = json.loads((source / "pump_report.json").read_text(encoding="utf-8"))
    data = np.load(checkpoint)
    modes = np.asarray(data["pump_modes"], dtype=int)
    state = np.asarray(data["X_real"], dtype=float) + 1j * np.asarray(data["X_imag"], dtype=float)
    circuit = load_circuit(ROOT / "designs" / "ipm_2c_fixed")
    branch = make_branch_law(circuit)
    frequency_hz = 7.9e9
    current = float(report["metadata"]["pump_current_a"])
    omega_metadata = float(report["metadata"]["omega_p"])
    omega_expected = 2.0 * math.pi * frequency_hz
    before = validate_production_hb_state(
        circuit, branch, frequency_hz=frequency_hz, pump_port=4,
        pump_current_a=current, modes=modes, state=state, nt=40,
        metadata=report["metadata"],
    )

    grid = HarmonicGrid(modes=modes, nt=40, omega=omega_expected)
    problem = FullPumpProblem(
        circuit.C, circuit.G, circuit.K, circuit.Bphi, branch, grid,
        circuit.port_to_index[4], current,
        loss_model=default_loss_model_for(circuit),
    )
    solver = HarmonicNewtonKrylovSolver(NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=15, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=60, gmres_maxiter=100, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=True,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
        stall_ratio=0.8, stall_patience=4, solve_deadline_s=180.0,
    ))
    corrected, solve_report = solver.solve_one(problem, state, 1.0)
    after = validate_production_hb_state(
        circuit, branch, frequency_hz=frequency_hz, pump_port=4,
        pump_current_a=current, modes=modes, state=corrected, nt=40,
        metadata=report["metadata"],
    )
    out_checkpoint = outdir / "hb_checkpoint"
    out_checkpoint.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_checkpoint / "pump_solution.npz",
        X_real=corrected.real, X_imag=corrected.imag, pump_modes=modes,
    )
    provenance = {
        "hb_solver_family": "production_pump",
        "hb_solver_module": "twpa_solver.pump.solver",
        "hb_entrypoint": "HarmonicNewtonKrylovSolver.solve_one",
        "production_problem": "twpa_solver.pump.hb.FullPumpProblem",
        "production_grid": "twpa_solver.pump.problem.HarmonicGrid",
        "pump_frequency_hz": frequency_hz,
        "omega_p": omega_expected,
        "pump_current_a": current,
        "pump_port": 4,
        "pump_modes": modes.tolist(),
        "nt": 40,
        "loss_model": default_loss_model_for(circuit),
        "source_checkpoint": str(source),
        "source_metadata_omega_matches": bool(abs(omega_metadata - omega_expected) < 1e-6),
        "source_checkpoint_validation": before,
        "correction_report": solve_report.__dict__,
        "checkpoint_validation": after,
        "h3_path": {
            "transient_entrypoint": "scripts.h1_transient_branch_transfer.run_experiment",
            "transient_system": "scripts.h1_transient_branch_transfer.build_system",
            "angular_frequency": "2*pi*freq_ghz*1e9",
            "complexity_ladder_imported": False,
        },
        "H3_PROVENANCE": "VERIFIED_UNAFFECTED"
        if abs(omega_metadata - omega_expected) < 1e-6 and after["checkpoint_validated"]
        else "AFFECTED_OR_UNCERTAIN",
    }
    (out_checkpoint / "pump_report.json").write_text(json.dumps({
        "final_status": "VALID_CONVERGED",
        "metadata": provenance,
    }, indent=2), encoding="utf-8")
    (outdir / "audit.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({
        "H3_PROVENANCE": provenance["H3_PROVENANCE"],
        "source_residual": before.get("production_hb_residual_rel"),
        "corrected_residual": after.get("production_hb_residual_rel"),
        "omega_metadata": omega_metadata,
        "omega_expected": omega_expected,
        "solve_converged": solve_report.converged,
    }, indent=2))
    return 0 if provenance["H3_PROVENANCE"] == "VERIFIED_UNAFFECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
