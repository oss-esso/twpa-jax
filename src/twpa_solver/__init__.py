"""General harmonic-balance solver for Josephson circuits."""

from twpa_solver.core import (
    CircuitMatrices,
    default_loss_model_for,
    load_circuit,
    save_circuit,
    solve_linear_scattering,
    KI_MODEL_PRESETS,
    KineticInductorBranchLaw,
    KineticInductorAltBranchLaw,
    CompositeBranchLaw,
    kinetic_validity,
    kinetic_dc_branch_flux,
    resolve_ki_model,
    PortEnvironment,
)
from twpa_solver.loss import (
    InsertionLossModel,
    default_loss_model,
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
    "default_loss_model_for",
    "load_circuit",
    "save_circuit",
    "solve_linear_scattering",
    "KI_MODEL_PRESETS",
    "KineticInductorBranchLaw",
    "KineticInductorAltBranchLaw",
    "CompositeBranchLaw",
    "kinetic_validity",
    "kinetic_dc_branch_flux",
    "resolve_ki_model",
    "PortEnvironment",
    "InsertionLossModel",
    "default_loss_model",
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
