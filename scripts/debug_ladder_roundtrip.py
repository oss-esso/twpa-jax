"""Direct HB-versus-transient check for one saved complexity-ladder rung."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from h1_transient_branch_transfer import (
    build_system,
    implicit_trapezoid_ramp,
    load_hb_initial,
)
from twpa_solver.core.constants import PHI0_REDUCED
from twpa_solver.pump.validation import validate_production_hb_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--periods", type=int, default=10)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()
    checkpoint = args.run / "hb_checkpoint"
    system = build_system(args.run / "circuit", 7.9, 1)
    data = np.load(checkpoint / "pump_solution.npz")
    X = np.asarray(data["X_real"], dtype=float) + 1j * np.asarray(data["X_imag"], dtype=float)
    x0, w0, current, report = load_hb_initial(checkpoint, system.circuit, system.omega)
    modes = np.asarray(data["pump_modes"], dtype=int)
    validation = validate_production_hb_state(
        system.circuit, system.branch, frequency_hz=7.9e9, pump_port=1,
        pump_current_a=current, modes=modes, state=X, nt=max(2 * int(modes.max()) + 1, 40),
        metadata=report.get("metadata", {}),
    )
    if not validation["checkpoint_validated"]:
        print(json.dumps({
            "classification": "INVALID_HB_FIXTURE",
            "hb_validation": validation,
        }, indent=2))
        return 2
    y0 = system.pack(x0 / PHI0_REDUCED, w0 / PHI0_REDUCED)
    theta, states, integration = implicit_trapezoid_ramp(
        system, y0, current, current, 2.0 * math.pi * args.periods, 0.0,
        args.step, newton_tol=1e-6, max_newton=12,
    )
    basis = np.exp(1j * theta[:, None] * modes[None, :])
    x_hb = 2.0 * np.real(basis @ X)
    x_td = PHI0_REDUCED * np.asarray([system.unpack(states[:, i])[0] for i in range(states.shape[1])])
    e_x = np.linalg.norm(x_td - x_hb, axis=1) / max(np.linalg.norm(x_hb) / math.sqrt(x_hb.size), 1e-300)
    strobe = np.arange(args.periods + 1) * 2.0 * math.pi
    selected = [int(np.argmin(np.abs(theta - value))) for value in strobe]
    d1 = np.linalg.norm(np.diff(x_td[selected], axis=0), axis=1) / max(np.linalg.norm(x_hb[0]), 1e-300)
    print(json.dumps({
        "integration": integration, "current_a": current,
        "e_x_rms": float(np.sqrt(np.mean(e_x ** 2))), "e_x_max": float(np.max(e_x)),
        "strobe_max": float(np.max(d1)), "strobe_tail": d1[-max(1, len(d1)//2):].tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
