"""Demonstrate the port-shunt power-balance defect and its repair.

The old external balance removed V/Z0 from the port-wave current but retained
the same shunt loss in the dissipation term.  This script prints the old
arithmetic and the repaired arithmetic on the deterministic two-port fixture.
It is intentionally standalone and does not invoke pytest.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.observables import power_balance
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver, NewtonKrylovSettings


def make_state() -> tuple[np.ndarray, object, CircuitMatrices]:
    circuit = CircuitMatrices(
        C=sp.eye(2, format="csr") * 1e-15,
        G=sp.eye(2, format="csr") * (1.0 / 50.0 + 1e-3),
        K=sp.csr_matrix([[2e9, -1e9], [-1e9, 2e9]]),
        Bphi=sp.csr_matrix([[1.0], [-1.0]]),
        Ic=np.array([1e-6]),
        port_to_index={1: 0, 2: 1},
    )
    basis = build_three_tone_basis(2.0e10, 1.0e9)
    source = MultiToneDrive(
        basis.pump_tone, 0, 1e-8
    ).to_coeffs(basis, circuit.C.shape[0])
    source += MultiToneDrive(
        basis.signal_tone, 0, 1e-10
    ).to_coeffs(basis, circuit.C.shape[0])
    problem = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(source)
    )
    settings = NewtonKrylovSettings(
        newton_tol=1e-10,
        max_newton=20,
        gmres_rtol=1e-8,
        gmres_atol=0.0,
        gmres_restart=20,
        gmres_maxiter=40,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )
    state, reports = HarmonicNewtonKrylovSolver(settings).solve_continuation(
        problem, continuation_steps=4
    )
    if not reports[-1].converged:
        raise RuntimeError("fixture solve did not converge")
    return state, basis, circuit


def main() -> None:
    state, basis, circuit = make_state()
    result = power_balance(state, basis, circuit)
    old_dissipation = float(result["external_dissipated_power"])
    shunt = float(result["port_resistor_dissipated_power"])
    supplied = float(result["external_supplied_power"])
    old_error = abs(supplied - (old_dissipation + shunt)) / max(
        abs(supplied), abs(old_dissipation + shunt)
    )
    repaired_error = float(result["external_power_balance_rel_err"])
    print("before repair: port shunt is counted in dissipation but excluded from wave current")
    print(f"  external supplied power = {supplied:.6e} W")
    print(f"  dissipation + omitted shunt = {old_dissipation + shunt:.6e} W")
    print(f"  relative error = {old_error:.6e}")
    print("after repair: subtract the same shunt loss from the dissipation side")
    print(f"  shunt dissipation = {shunt:.6e} W")
    print(f"  relative error = {repaired_error:.6e}")
    if not np.isfinite(repaired_error):
        raise RuntimeError("repaired balance is not finite")


if __name__ == "__main__":
    main()
