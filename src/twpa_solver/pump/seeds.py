from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.pump.problem import FullPumpProblem, pack_complex

def load_dc_solution(dc_solution: str | Path | None, circuit: CircuitMatrices) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a static DC operating point for shifted pump solves."""
    if dc_solution is None:
        return None, None

    p = Path(dc_solution)
    if p.is_dir():
        p = p / "dc_solution.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing dc solution: {p}")

    sol = np.load(p)
    x_dc = None
    psi_dc = None

    if "x_dc" in sol.files:
        x_dc = np.asarray(sol["x_dc"], dtype=np.float64).reshape(-1)
        if x_dc.size != circuit.C.shape[0]:
            raise ValueError(f"x_dc length {x_dc.size} != node count {circuit.C.shape[0]}")
        psi_dc = np.asarray(circuit.Bphi.T @ x_dc, dtype=np.float64).reshape(-1)

    if "psi_dc" in sol.files:
        psi_dc_file = np.asarray(sol["psi_dc"], dtype=np.float64).reshape(-1)
        if psi_dc_file.size != circuit.Bphi.shape[1]:
            raise ValueError(f"psi_dc length {psi_dc_file.size} != branch count {circuit.Bphi.shape[1]}")
        psi_dc = psi_dc_file

    if psi_dc is None:
        raise ValueError(f"dc solution {p} must contain x_dc or psi_dc")

    return x_dc, psi_dc


def build_linear_phasor_seed(
    problem: FullPumpProblem,
    *,
    source_scale: float = 1.0,
    method: str = "gmres",
    rtol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 60,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Build a first-harmonic seed from D(omega_p) X1 = S1.

    This intentionally uses the same linear block and positive-frequency
    source coefficient convention as the residual: x(t) = 2 Re X1 exp(i wt),
    source(t) = I_p cos(wt), so S1 = 0.5 I_p at the pump node.
    """
    X = problem.zeros()
    S = problem.source_coeffs(source_scale)
    source_from_time = problem.grid.project_positive(problem.source_time(source_scale))
    source_error = S - source_from_time
    source_error_abs = float(np.linalg.norm(pack_complex(source_error)))
    source_norm = float(np.linalg.norm(pack_complex(S)))
    source_error_rel = source_error_abs / max(source_norm, 1e-300)

    row = problem.source_row
    A = problem._linear_blocks[row]
    b = S[row]
    gmres_iterations = 0
    info = 0
    t0 = time.perf_counter()

    if method == "direct":
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            x1 = spla.spsolve(A, b)
    elif method == "gmres":
        def cb(_pr_norm: float) -> None:
            nonlocal gmres_iterations
            gmres_iterations += 1

        try:
            x1, info = spla.gmres(
                A,
                b,
                rtol=rtol,
                atol=0.0,
                restart=restart,
                maxiter=maxiter,
                callback=cb,
                callback_type="pr_norm",
            )
        except TypeError:
            x1, info = spla.gmres(
                A,
                b,
                tol=rtol,
                restart=restart,
                maxiter=maxiter,
                callback=cb,
            )
    else:
        raise ValueError(f"unknown linear seed method {method!r}")

    runtime_s = time.perf_counter() - t0

    if not np.all(np.isfinite(x1)):
        raise RuntimeError("linear_phasor seed solve produced non-finite entries")

    X[row] = np.asarray(x1, dtype=np.complex128)
    linear_residual_abs = float(np.linalg.norm(A @ X[row] - b))
    linear_residual_rel = linear_residual_abs / max(float(np.linalg.norm(b)), 1e-300)

    return X, {
        "initial_guess": "linear_phasor",
        "linear_seed_source_scale": float(source_scale),
        "linear_seed_mode": str(int(round(problem.grid.k[row]))),
        "linear_seed_method": method,
        "linear_seed_solve_runtime_s": float(runtime_s),
        "linear_seed_solver_info": int(info),
        "linear_seed_gmres_iterations": int(gmres_iterations),
        "linear_seed_linear_residual_abs": linear_residual_abs,
        "linear_seed_linear_residual_rel": linear_residual_rel,
        "linear_seed_norm": float(np.linalg.norm(X[row])),
        "linear_seed_source_error_abs": source_error_abs,
        "linear_seed_source_error_rel": source_error_rel,
    }
