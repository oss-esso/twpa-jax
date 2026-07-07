from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

def db20(x: float) -> float:
    return 20.0 * math.log10(max(float(abs(x)), 1e-300))


def db10(x: float) -> float:
    return 10.0 * math.log10(max(float(x), 1e-300))


def gain_db_from_s(s: complex) -> float:
    return 20.0 * np.log10(max(abs(s), 1e-300))


def complex_to_pair(z: complex) -> tuple[float, float]:
    return float(np.real(z)), float(np.imag(z))


@dataclass
class GainResult:
    status: str
    signal_ghz: float
    signal_m: int
    idler_m: int
    sidebands: int
    conversion_unknowns: int
    matrix_nnz: int
    assemble_runtime_s: float
    factor_solve_runtime_s: float
    baseline_off_runtime_s: float
    baseline_pumpdiag_runtime_s: float
    linear_abs_residual: float
    linear_rel_residual: float

    vout_on: complex
    vout_off: complex
    vout_pumpdiag: complex
    vout_idler: complex | None

    gain_vs_off: float
    s_param_abs: float
    gain_db: float
    gain_vs_off_db: float
    gain_vs_pumpdiag: float
    gain_vs_pumpdiag_db: float
    idler_power_rel_to_signal_off: float | None
    idler_power_rel_to_signal_off_db: float | None
