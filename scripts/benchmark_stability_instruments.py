"""Benchmark the Phase 4 stability instruments on a damped linear reference.

The production selection is the branch-tracked Hill scan.  The matrix-free
monodromy implementation is retained as a control and is benchmarked against
the analytic damped oscillator used by the Floquet unit tests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.h1_transient_branch_transfer import TransientSystem
from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.core.nonlinear import JosephsonBranchLaw
from twpa_solver.stability import build_hb_periodic_orbit, build_monodromy_operator


def _reference_operator(steps: int):
    circuit = CircuitMatrices(
        C=sp.diags([1.0], format="csr"),
        G=sp.csr_matrix([[0.2]]),
        K=sp.csr_matrix([[2.0]]),
        Bphi=sp.csr_matrix((1, 1)),
        Ic=np.zeros(1),
        phi0=1.0,
    )
    system = TransientSystem(
        circuit=circuit,
        branch=JosephsonBranchLaw(circuit.Ic, circuit.phi0),
        omega=3.0,
        pump_node=0,
        differential=np.array([0]),
        algebraic=np.array([], dtype=int),
        c_factor=spla.splu(circuit.C.tocsc()),
        g_alg_factor=None,
    )
    orbit = build_hb_periodic_orbit(
        np.zeros((1, 1), dtype=complex), [1], system.omega,
        system.phi0, steps_per_period=steps,
    )
    return build_monodromy_operator(
        system, orbit, max_step_theta=2.0 * math.pi / steps
    )


def damped_reference(steps: int) -> dict[str, float]:
    started = time.perf_counter()
    operator = _reference_operator(steps)
    runtime_s = time.perf_counter() - started
    theta_matrix = np.array([[0.0, 1.0], [-2.0 / 9.0, -0.2 / 3.0]])
    exact = np.linalg.eigvals(la.expm(theta_matrix * 2.0 * math.pi))
    actual = np.linalg.eigvals(operator.as_linear_operator().matmat(np.eye(2)))
    exact_radius = float(np.max(np.abs(exact)))
    actual_radius = float(np.max(np.abs(actual)))
    return {
        "steps_per_period": steps,
        "runtime_s": runtime_s,
        "exact_spectral_radius": exact_radius,
        "monodromy_spectral_radius": actual_radius,
        "absolute_radius_error": abs(actual_radius - exact_radius),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hill-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    hill = json.loads(args.hill_json.read_text(encoding="utf-8"))
    target = hill.get("target", hill)
    if isinstance(target, list):
        target = target[0]
    result = {
        "selected_instrument": "branch_tracked_hill",
        "hill_operating_point": {
            "pump_dir": target.get("pump_dir"),
            "runtime_s": target.get("runtime_s"),
            "n_points": len(target.get("signal_ghz", [])),
            "sidebands": hill.get("metadata", {}).get("sidebands"),
        },
        "analytic_monodromy_control": damped_reference(96),
        "alternatives": {
            "koopman_hill_projection": "not implemented; not selected",
            "lossy_time_domain_monodromy": (
                "not applicable to complex-C dielectric representation"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
