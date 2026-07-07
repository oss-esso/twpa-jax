"""General harmonic-balance solver for Josephson circuits."""

from twpa_solver.core import (
    CircuitMatrices,
    load_circuit,
    save_circuit,
    solve_linear_scattering,
)
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
    PumpBasis,
    resolve_pump_basis,
)
from twpa_solver.signal import (
    GainResult,
    build_khat,
    compute_gamma_hat,
    solve_gain_one,
    solve_gain_one_schur,
)

__all__ = [
    "CircuitMatrices",
    "load_circuit",
    "save_circuit",
    "solve_linear_scattering",
    "PumpBasis",
    "resolve_pump_basis",
    "FullPumpProblem",
    "HarmonicGrid",
    "JosephsonBranchArray",
    "HarmonicNewtonKrylovSolver",
    "NewtonKrylovSettings",
    "GainResult",
    "compute_gamma_hat",
    "build_khat",
    "solve_gain_one",
    "solve_gain_one_schur",
]
