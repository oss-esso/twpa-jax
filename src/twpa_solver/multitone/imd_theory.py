"""Pump-dressed perturbative intermodulation coefficients.

This module expands the exact multitone HB residual in the signal amplitude
about a converged pump-only state.  The pump is retained non-perturbatively;
only the injected signal source is expanded.  The implementation is limited
to Josephson branch laws because the second and third derivatives are used
explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex
from twpa_solver.multitone.observables import extract_port_waves
from twpa_solver.pump.problem import pack_complex, unpack_complex


@dataclass(frozen=True)
class PumpDressedIMDResult:
    """Unit-signal perturbation coefficients through fifth order."""

    first_order: np.ndarray
    second_order: np.ndarray
    third_order: np.ndarray
    fourth_order: np.ndarray
    fifth_order: np.ndarray
    linear_residual: float
    second_order_residual: float
    third_order_residual: float
    fourth_order_residual: float
    fifth_order_residual: float


def _require_josephson_branch(problem) -> tuple[np.ndarray, float]:
    branch = problem.branch
    if not hasattr(branch, "critical_current") or not hasattr(branch, "phi0"):
        raise TypeError(
            "pump-dressed IMD theory currently requires a JosephsonBranchLaw"
        )
    return np.asarray(branch.critical_current, dtype=float), float(branch.phi0)


def _solve_linearized(problem, tangent, rhs: np.ndarray, *, rtol: float) -> tuple[np.ndarray, float]:
    """Solve the real-linear pumped Jacobian for one coefficient array."""
    shape = rhs.shape
    rhs_real = pack_complex(rhs)
    dimension = rhs_real.size

    def matvec(value: np.ndarray) -> np.ndarray:
        perturbation = unpack_complex(value, shape)
        return pack_complex(problem.jvp_coeffs_with_tangent(perturbation, tangent))

    operator = spla.LinearOperator(
        (dimension, dimension), matvec=matvec, dtype=np.float64
    )
    factor = problem.assemble_real_coupled_fast(tangent)
    preconditioner = spla.LinearOperator(
        (dimension, dimension),
        matvec=lambda value: factor.solve(np.asarray(value, dtype=float)),
        dtype=np.float64,
    )
    solution_real, info = spla.gmres(
        operator,
        rhs_real,
        M=preconditioner,
        rtol=float(rtol),
        atol=0.0,
        restart=40,
        maxiter=200,
    )
    if info != 0:
        raise RuntimeError(f"pump-dressed IMD linear solve did not converge: info={info}")
    solution = unpack_complex(solution_real, shape)
    residual = matvec(solution_real) - rhs_real
    return solution, float(np.linalg.norm(residual) / max(np.linalg.norm(rhs_real), 1e-300))


def _branch_phase(problem, coefficients: np.ndarray, phi0: float) -> np.ndarray:
    waveform = problem.grid.synthesize(coefficients)
    return (problem.BphiT @ waveform.T).T / phi0


def _project_branch_force(problem, branch_force: np.ndarray) -> np.ndarray:
    nodal = (problem.Bphi @ branch_force.T).T
    return problem.grid.project(nodal)


def _second_derivative_force(problem, pump_state: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    ic, phi0 = _require_josephson_branch(problem)
    dc = np.asarray(problem.dc_branch_flux, dtype=float)
    pump_phase = _branch_phase(problem, pump_state, phi0) + dc[None, :] / phi0
    left_phase = _branch_phase(problem, left, phi0)
    right_phase = _branch_phase(problem, right, phi0)
    branch_force = -ic[None, :] * np.sin(pump_phase) * left_phase * right_phase
    return _project_branch_force(problem, branch_force)


def _third_derivative_force(
    problem,
    pump_state: np.ndarray,
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    ic, phi0 = _require_josephson_branch(problem)
    dc = np.asarray(problem.dc_branch_flux, dtype=float)
    pump_phase = _branch_phase(problem, pump_state, phi0) + dc[None, :] / phi0
    left_phase = _branch_phase(problem, left, phi0)
    middle_phase = _branch_phase(problem, middle, phi0)
    right_phase = _branch_phase(problem, right, phi0)
    branch_force = -ic[None, :] * np.cos(pump_phase) * left_phase * middle_phase * right_phase
    return _project_branch_force(problem, branch_force)


def _fourth_derivative_force(
    problem,
    pump_state: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
) -> np.ndarray:
    ic, phi0 = _require_josephson_branch(problem)
    dc = np.asarray(problem.dc_branch_flux, dtype=float)
    pump_phase = _branch_phase(problem, pump_state, phi0) + dc[None, :] / phi0
    phases = [_branch_phase(problem, value, phi0) for value in (first, second, third, fourth)]
    branch_force = ic[None, :] * np.sin(pump_phase)
    for phase in phases:
        branch_force = branch_force * phase
    return _project_branch_force(problem, branch_force)


def _fifth_derivative_force(
    problem, pump_state: np.ndarray, first: np.ndarray
) -> np.ndarray:
    ic, phi0 = _require_josephson_branch(problem)
    dc = np.asarray(problem.dc_branch_flux, dtype=float)
    pump_phase = _branch_phase(problem, pump_state, phi0) + dc[None, :] / phi0
    first_phase = _branch_phase(problem, first, phi0)
    branch_force = ic[None, :] * np.cos(pump_phase) * first_phase**5
    return _project_branch_force(problem, branch_force)


def solve_pump_dressed_im3(
    problem,
    pump_state: np.ndarray,
    signal_source_unit: np.ndarray,
    *,
    rtol: float = 1e-10,
) -> PumpDressedIMDResult:
    """Solve the exact residual expansion through fifth signal order.

    ``signal_source_unit`` contains the source coefficients for unit current
    in each injected tone.  The returned coefficient arrays therefore scale
    as I, I**2, and I**3 for a physical tone-1 current ``I``.
    """
    if signal_source_unit.shape != pump_state.shape:
        raise ValueError("pump and signal coefficient arrays must have equal shapes")
    tangent = problem.tangent_state(pump_state)
    first, first_residual = _solve_linearized(
        problem, tangent, signal_source_unit, rtol=rtol
    )
    second_rhs = -0.5 * _second_derivative_force(problem, pump_state, first, first)
    second, second_residual = _solve_linearized(
        problem, tangent, second_rhs, rtol=rtol
    )
    third_rhs = -(
        _second_derivative_force(problem, pump_state, first, second)
        + (1.0 / 6.0)
        * _third_derivative_force(problem, pump_state, first, first, first)
    )
    third, third_residual = _solve_linearized(
        problem, tangent, third_rhs, rtol=rtol
    )
    fourth_rhs = -(
        _second_derivative_force(problem, pump_state, first, third)
        + 0.5 * _second_derivative_force(problem, pump_state, second, second)
        + 0.5
        * _third_derivative_force(problem, pump_state, first, first, second)
        + (1.0 / 24.0)
        * _fourth_derivative_force(problem, pump_state, first, first, first, first)
    )
    fourth, fourth_residual = _solve_linearized(
        problem, tangent, fourth_rhs, rtol=rtol
    )
    fifth_rhs = -(
        _second_derivative_force(problem, pump_state, first, fourth)
        + _second_derivative_force(problem, pump_state, second, third)
        + 0.5
        * _third_derivative_force(problem, pump_state, first, first, third)
        + 0.5
        * _third_derivative_force(problem, pump_state, first, second, second)
        + (1.0 / 6.0)
        * _fourth_derivative_force(problem, pump_state, first, first, first, second)
        + (1.0 / 120.0) * _fifth_derivative_force(problem, pump_state, first)
    )
    fifth, fifth_residual = _solve_linearized(
        problem, tangent, fifth_rhs, rtol=rtol
    )
    return PumpDressedIMDResult(
        first_order=first,
        second_order=second,
        third_order=third,
        fourth_order=fourth,
        fifth_order=fifth,
        linear_residual=first_residual,
        second_order_residual=second_residual,
        third_order_residual=third_residual,
        fourth_order_residual=fourth_residual,
        fifth_order_residual=fifth_residual,
    )


def perturbative_imd_dbc(
    result: PumpDressedIMDResult,
    basis: MultiToneBasis,
    circuit,
    target_tone: ToneIndex,
    *,
    order: int,
    reference_tone: ToneIndex,
    signal_current_a: float,
    out_port: int,
    z0_ohm: float = 50.0,
    dc_branch_flux: np.ndarray | None = None,
) -> float:
    """Return the order-matched perturbative target power in dBc."""
    if signal_current_a <= 0.0:
        return float("nan")
    if order == 3:
        coefficient = result.third_order
    elif order == 5:
        coefficient = result.fifth_order
    else:
        return float("nan")
    target_waves = extract_port_waves(
        coefficient,
        basis,
        circuit,
        ports=[int(out_port)],
        z0_ohm=z0_ohm,
        dc_branch_flux=dc_branch_flux,
    )
    first_waves = extract_port_waves(
        result.first_order,
        basis,
        circuit,
        ports=[int(out_port)],
        z0_ohm=z0_ohm,
        dc_branch_flux=dc_branch_flux,
    )
    target_power = target_waves["b_power"].get((target_tone, int(out_port)))
    reference_power = first_waves["b_power"].get((reference_tone, int(out_port)))
    if target_power is None or reference_power is None or target_power <= 0.0 or reference_power <= 0.0:
        return float("nan")
    amplitude_power_exponent = 2 * int(order) - 2
    ratio = (
        float(target_power / reference_power)
        * float(signal_current_a) ** amplitude_power_exponent
    )
    return float(10.0 * math.log10(max(ratio, 1e-300)))
