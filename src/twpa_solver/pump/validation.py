"""Validation contract for persisted production pump-HB states.

This module intentionally delegates residual evaluation to ``FullPumpProblem``.
It is a provenance gate, not a second HB implementation.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np

from twpa_solver.pump.problem import FullPumpProblem, HarmonicGrid
from twpa_solver.core.linear import default_loss_model_for


PRODUCTION_SOLVER_FAMILY = "production_pump"
PRODUCTION_SOLVER_MODULE = "twpa_solver.pump.solver"
PRODUCTION_SOLVER_ENTRYPOINT = "HarmonicNewtonKrylovSolver.solve_one"


def production_solver_provenance() -> dict[str, str]:
    """Return stable identifiers for the production HB implementation."""
    solver_module = importlib.import_module(PRODUCTION_SOLVER_MODULE)
    return {
        "hb_solver_family": PRODUCTION_SOLVER_FAMILY,
        "hb_solver_module": solver_module.__name__,
        "hb_entrypoint": PRODUCTION_SOLVER_ENTRYPOINT,
    }


def validate_production_hb_state(
    circuit: Any,
    branch: Any,
    *,
    frequency_hz: float,
    pump_port: int,
    pump_current_a: float,
    modes: np.ndarray,
    state: np.ndarray,
    nt: int,
    residual_threshold: float = 1e-8,
    metadata: dict[str, Any] | None = None,
    dc_branch_flux: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate a state using the actual production pump residual.

    The returned record is JSON-serializable and is suitable for checkpoint
    provenance.  It never labels an invalid state as physical dynamics.
    """
    modes = np.asarray(modes, dtype=int).reshape(-1)
    state = np.asarray(state, dtype=np.complex128)
    if modes.size == 0 or np.any(modes < 0) or np.unique(modes).size != modes.size:
        raise ValueError("invalid non-negative harmonic metadata")
    expected_shape = (modes.size, circuit.node_count)
    shape_ok = state.shape == expected_shape
    finite_ok = bool(np.all(np.isfinite(state)))
    result: dict[str, Any] = {
        **production_solver_provenance(),
        "pump_frequency_hz": float(frequency_hz),
        "pump_current_a": float(pump_current_a),
        "pump_port": int(pump_port),
        "harmonic_modes": modes.tolist(),
        "nt": int(nt),
        "state_shape": list(state.shape),
        "expected_state_shape": list(expected_shape),
        "state_finite": finite_ok,
        "state_shape_ok": shape_ok,
        "circuit_nodes": int(circuit.node_count),
        "circuit_junctions": int(circuit.branch_count),
        "checkpoint_validated": False,
        "loss_model": default_loss_model_for(circuit),
    }
    if metadata:
        result["metadata_match_context"] = metadata
    if not shape_ok or not finite_ok:
        result["invalid_reason"] = "state shape or finite-value check failed"
        return result

    omega = 2.0 * math.pi * float(frequency_hz)
    grid = HarmonicGrid(modes=modes, nt=int(nt), omega=omega)
    problem = FullPumpProblem(
        circuit.C, circuit.G, circuit.K, circuit.Bphi, branch, grid,
        circuit.port_to_index[int(pump_port)], float(pump_current_a),
        dc_branch_flux=dc_branch_flux,
        loss_model=default_loss_model_for(circuit),
    )
    norms = problem.norms(state, 1.0, True)
    result.update({
        "production_hb_coeff_rel": float(norms["coeff_rel"]),
        "production_hb_time_rel": float(norms["time_rel"]),
        "production_hb_residual_rel": float(norms["coeff_rel"]),
        "production_hb_residual_threshold": float(residual_threshold),
    })
    # ``coeff_rel`` is the production solver's convergence residual.  The
    # synthesized time residual may contain intentionally omitted harmonics;
    # retain it as diagnostic evidence but do not silently replace the
    # production convergence contract with a different norm.
    result["checkpoint_validated"] = bool(
        float(norms["coeff_rel"]) <= residual_threshold
    )
    if not result["checkpoint_validated"]:
        result["invalid_reason"] = "production HB residual gate failed"
    return result
