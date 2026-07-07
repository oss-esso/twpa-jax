from __future__ import annotations

# Compatibility aggregation module.
# New code should import from problem/solver/seeds/io directly.
# Old backend code can temporarily do:
#     import twpa_solver.pump.hb as exp08

from twpa_solver.pump.problem import (
    FullPumpProblem,
    FullIPMPumpProblem,
    HarmonicGrid,
    JosephsonBranchArray,
    SpectralTangentState,
    TangentState,
    pack_complex,
    unpack_complex,
)
from twpa_solver.pump.solver import (
    ContinuationTrace,
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
    StepReport,
    empty_continuation_trace,
    gmres_call,
)
from twpa_solver.pump.seeds import (
    build_linear_phasor_seed,
    load_dc_solution,
)
from twpa_solver.pump.io import (
    summarize_solution,
    write_results,
)

__all__ = [
    "FullPumpProblem",
    "FullIPMPumpProblem",
    "HarmonicGrid",
    "JosephsonBranchArray",
    "SpectralTangentState",
    "TangentState",
    "pack_complex",
    "unpack_complex",
    "ContinuationTrace",
    "HarmonicNewtonKrylovSolver",
    "NewtonKrylovSettings",
    "StepReport",
    "empty_continuation_trace",
    "gmres_call",
    "build_linear_phasor_seed",
    "load_dc_solution",
    "summarize_solution",
    "write_results",
]
