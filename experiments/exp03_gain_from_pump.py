# experiments/exp03_gain_from_pump.py
"""
Experiment 03: gain/conversion solve from a converged pump.

Pipeline:
    1. Use Exp02 harmonic-coefficient pump solver.
    2. Compute gamma(t) = d i_nl / d psi on the converged pump.
    3. Fourier-project gamma(t) to gamma_hat_ell.
    4. Build sparse sideband conversion matrix:

        A_mq =
            [K - Omega_m^2 C + i Omega_m G] delta_mq
            + B diag(gamma_hat_{m-q}) B^T

    5. Solve A y = r for a toy unit source at sideband m=0.
    6. Extract an S21-like scalar:

        S21_like = 2 Y0 i omega_s y_{m=0, output_node}

Important:
    This is not yet the final physical power-wave model.
    It is the first fast end-to-end test of:

        nonlinear pump -> linearized periodic stiffness -> sideband conversion solve.

Default gain backend:
    Sparse assembled direct solve.

If sparse direct becomes slow at real size, the next experiment should replace
this with matrix-free/preconditioned GMRES.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# Allow importing exp02 from the same experiments directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from exp02_harmonic_coeff_pump import (  # noqa: E402
    HarmonicPumpProblem,
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
    SolveReport,
    make_tiny_ladder_problem,
)


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class PumpStageResult:
    problem: HarmonicPumpProblem
    Xpump: np.ndarray
    reports: list[SolveReport]
    total_runtime_s: float
    converged: bool


@dataclass
class GainResult:
    signal_omega: float
    sidebands: int
    conversion_unknowns: int
    assemble_runtime_s: float
    factor_runtime_s: float
    solve_runtime_s: float
    total_runtime_s: float
    linear_abs_residual: float
    linear_rel_residual: float
    s21_like: complex
    gain_linear: float
    gain_db: float
    status: str


# =============================================================================
# Pump stage
# =============================================================================

def solve_pump_stage(args: argparse.Namespace) -> PumpStageResult:
    problem = make_tiny_ladder_problem(
        n_nodes=args.nodes,
        harmonics=args.pump_harmonics,
        nt=args.pump_nt,
        omega_p=args.omega_p,
        pump_amp=args.pump_amp,
        damping=args.damping,
        c_value=args.capacitance,
        k_ground=args.k_ground,
        ic=args.ic,
        phi0=args.phi0,
    )

    settings = NewtonKrylovSettings(
        newton_tol=args.pump_newton_tol,
        max_newton=args.pump_max_newton,
        gmres_rtol=args.pump_gmres_rtol,
        gmres_restart=args.pump_gmres_restart,
        gmres_maxiter=args.pump_gmres_maxiter,
        verbose=not args.quiet,
    )

    solver = HarmonicNewtonKrylovSolver(settings)

    t0 = time.perf_counter()
    Xpump, reports = solver.solve_continuation(
        problem,
        continuation_steps=args.continuation_steps,
    )
    total_runtime_s = time.perf_counter() - t0

    final = reports[-1] if reports else None
    converged = bool(
        final is not None
        and final.converged
        and abs(final.source_scale - 1.0) < 1e-12
    )

    return PumpStageResult(
        problem=problem,
        Xpump=Xpump,
        reports=reports,
        total_runtime_s=total_runtime_s,
        converged=converged,
    )


# =============================================================================
# Conversion matrix construction
# =============================================================================

def fourier_coeff_real_time(
    values_t: np.ndarray,
    t: np.ndarray,
    omega: float,
    ell: int,
) -> np.ndarray:
    """
    Fourier coefficient of a real periodic sampled signal:

        f_hat_ell = mean_j f(t_j) exp(-i ell omega t_j)

    values_t:
        shape (nt, n_values)

    returns:
        shape (n_values,), complex
    """
    phase = np.exp(-1j * ell * omega * t)
    return np.mean(phase[:, None] * values_t, axis=0)


def compute_gamma_hat_blocks(
    problem: HarmonicPumpProblem,
    Xpump: np.ndarray,
    max_ell: int,
) -> dict[int, sp.csc_matrix]:
    """
    Compute Khat_ell = B diag(gamma_hat_ell) B^T for ell in [-max_ell, max_ell].

    gamma(t) is the differential inverse inductance evaluated on the pump:

        gamma(t) = d i_nl / d psi |_{psi = B^T x_p(t)}

    Returns sparse node-space matrices.
    """
    psi_t = problem.branch_flux_time(Xpump)
    gamma_t = problem.branch.gamma(psi_t)

    blocks: dict[int, sp.csc_matrix] = {}

    for ell in range(-max_ell, max_ell + 1):
        gamma_hat = fourier_coeff_real_time(
            gamma_t,
            problem.grid.t,
            problem.grid.omega,
            ell,
        )

        Khat = (
            problem.Bphi
            @ sp.diags(gamma_hat, offsets=0, format="csr")
            @ problem.Bphi.T
        ).astype(np.complex128)

        blocks[ell] = Khat.tocsc()

    return blocks


def linear_dynamic_block(
    problem: HarmonicPumpProblem,
    omega: float,
) -> sp.csc_matrix:
    """
    D(omega) = K - omega^2 C + i omega G.
    """
    D = problem.K.astype(np.complex128)
    D = D + (-omega**2) * problem.C.astype(np.complex128)
    D = D + (1j * omega) * problem.G.astype(np.complex128)
    return D.tocsc()


def assemble_conversion_matrix(
    problem: HarmonicPumpProblem,
    Xpump: np.ndarray,
    signal_omega: float,
    sidebands: int,
) -> tuple[sp.csc_matrix, list[int], dict[int, sp.csc_matrix]]:
    """
    Assemble full sparse sideband conversion matrix.

    Sideband indices:
        m = -M, ..., M

    Frequencies:
        Omega_m = omega_s + m omega_p

    Matrix:
        A_mq = D(Omega_m) delta_mq + Khat_{m-q}
    """
    M = sidebands
    ms = list(range(-M, M + 1))
    nsb = len(ms)
    n = problem.n

    max_ell = 2 * M
    Khat = compute_gamma_hat_blocks(problem, Xpump, max_ell=max_ell)

    blocks: list[list[sp.csc_matrix | None]] = []

    for m in ms:
        row_blocks: list[sp.csc_matrix | None] = []
        omega_m = signal_omega + m * problem.grid.omega
        Dm = linear_dynamic_block(problem, omega_m)

        for q in ms:
            ell = m - q

            if m == q:
                block = Dm + Khat[ell]
            else:
                block = Khat[ell]

            row_blocks.append(block.tocsc())

        blocks.append(row_blocks)

    A = sp.bmat(blocks, format="csc")
    return A, ms, Khat


def build_gain_rhs(
    n: int,
    sidebands: int,
    input_node: int,
) -> np.ndarray:
    """
    Mathematical RHS r = E_0 u_input.

    We leave the 2 sqrt(Y0) source normalization outside and use the compact
    formula S21_like = 2 Y0 i omega_s ell^T A^{-1} r.
    """
    M = sidebands
    nsb = 2 * M + 1
    rhs = np.zeros(nsb * n, dtype=np.complex128)

    m0_index = M
    rhs[m0_index * n + input_node] = 1.0

    return rhs


def sideband_block_view(
    y: np.ndarray,
    n: int,
    sidebands: int,
) -> np.ndarray:
    """
    Flat sideband vector -> shape (2M+1, n).
    """
    return y.reshape((2 * sidebands + 1, n))


# =============================================================================
# Gain solve
# =============================================================================

def solve_gain_at_signal(
    problem: HarmonicPumpProblem,
    Xpump: np.ndarray,
    signal_omega: float,
    sidebands: int,
    input_node: int,
    output_node: int,
    y0: float,
    linear_tol: float,
) -> GainResult:
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    A, ms, _Khat = assemble_conversion_matrix(
        problem=problem,
        Xpump=Xpump,
        signal_omega=signal_omega,
        sidebands=sidebands,
    )
    assemble_runtime_s = time.perf_counter() - t0

    n = problem.n
    rhs = build_gain_rhs(
        n=n,
        sidebands=sidebands,
        input_node=input_node,
    )

    t0 = time.perf_counter()
    lu = spla.splu(A)
    factor_runtime_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y = lu.solve(rhs)
    solve_runtime_s = time.perf_counter() - t0

    residual = A @ y - rhs
    linear_abs_residual = float(np.linalg.norm(residual))
    linear_rel_residual = float(
        linear_abs_residual / max(np.linalg.norm(rhs), 1e-30)
    )

    Y = sideband_block_view(y, n=n, sidebands=sidebands)
    m0_index = sidebands

    # Compact toy S21-like extraction:
    #
    #   S21_like = 2 Y0 i omega_s u_out^T y_{m=0}
    #
    # This matches the mathematical compact expression but is not yet a full
    # physical port de-embedding model.
    s21_like = 2.0 * y0 * 1j * signal_omega * Y[m0_index, output_node]

    gain_linear = float(abs(s21_like) ** 2)
    gain_db = float(10.0 * np.log10(max(gain_linear, 1e-300)))

    total_runtime_s = time.perf_counter() - t_total0

    status = "VALID_CONVERGED" if linear_rel_residual < linear_tol else "FINITE_NONCONVERGED"

    return GainResult(
        signal_omega=signal_omega,
        sidebands=sidebands,
        conversion_unknowns=A.shape[0],
        assemble_runtime_s=assemble_runtime_s,
        factor_runtime_s=factor_runtime_s,
        solve_runtime_s=solve_runtime_s,
        total_runtime_s=total_runtime_s,
        linear_abs_residual=linear_abs_residual,
        linear_rel_residual=linear_rel_residual,
        s21_like=complex(s21_like),
        gain_linear=gain_linear,
        gain_db=gain_db,
        status=status,
    )


def signal_grid_from_args(args: argparse.Namespace) -> list[float]:
    if args.signal_points <= 1:
        return [args.signal_omega]

    return list(
        np.linspace(
            args.signal_start,
            args.signal_stop,
            args.signal_points,
        )
    )


# =============================================================================
# Printing
# =============================================================================

def print_pump_summary(pump: PumpStageResult) -> None:
    final = pump.reports[-1] if pump.reports else None

    print("\n=== pump summary ===")
    print(f"pump_status={'VALID_CONVERGED' if pump.converged else 'FAIL'}")
    print(f"pump_runtime_s={pump.total_runtime_s:.6f}")
    print(f"nodes={pump.problem.n}")
    print(f"branches={pump.problem.Bphi.shape[1]}")
    print(f"pump_harmonics={pump.problem.H}")
    print(f"pump_nt={pump.problem.grid.nt}")
    print(f"pump_real_unknowns={2 * pump.problem.H * pump.problem.n}")

    if final is not None:
        print(f"pump_final_coeff_rel={final.coeff_rel:.6e}")
        print(f"pump_final_time_rel={final.time_rel:.6e}")
        print(f"pump_final_newton_last={final.newton_iterations}")
        print(f"pump_final_gmres_last={final.gmres_iterations_total}")
        print(f"pump_failure_reason={final.failure_reason}")


def print_gain_result(result: GainResult) -> None:
    print("\n=== gain result ===")
    print(f"gain_status={result.status}")
    print(f"signal_omega={result.signal_omega:.9g}")
    print(f"sidebands={result.sidebands}")
    print(f"conversion_unknowns={result.conversion_unknowns}")
    print(f"assemble_runtime_s={result.assemble_runtime_s:.6f}")
    print(f"factor_runtime_s={result.factor_runtime_s:.6f}")
    print(f"solve_runtime_s={result.solve_runtime_s:.6f}")
    print(f"gain_total_runtime_s={result.total_runtime_s:.6f}")
    print(f"linear_abs_residual={result.linear_abs_residual:.6e}")
    print(f"linear_rel_residual={result.linear_rel_residual:.6e}")
    print(f"s21_like_real={result.s21_like.real:.12e}")
    print(f"s21_like_imag={result.s21_like.imag:.12e}")
    print(f"gain_linear={result.gain_linear:.12e}")
    print(f"gain_db={result.gain_db:.6f}")


def print_sweep_header() -> None:
    print("\n=== gain sweep ===")
    print(
        "signal_omega,status,gain_db,gain_linear,linear_rel_residual,"
        "assemble_runtime_s,factor_runtime_s,solve_runtime_s,total_runtime_s"
    )


def print_sweep_row(result: GainResult) -> None:
    print(
        f"{result.signal_omega:.12g},"
        f"{result.status},"
        f"{result.gain_db:.9g},"
        f"{result.gain_linear:.12e},"
        f"{result.linear_rel_residual:.6e},"
        f"{result.assemble_runtime_s:.6f},"
        f"{result.factor_runtime_s:.6f},"
        f"{result.solve_runtime_s:.6f},"
        f"{result.total_runtime_s:.6f}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(conflict_handler="resolve")

    # Fixture / pump
    p.add_argument("--nodes", type=int, default=64)
    p.add_argument("--pump-harmonics", type=int, default=3)
    p.add_argument("--pump-nt", type=int, default=32)
    p.add_argument("--omega-p", type=float, default=1.0)
    p.add_argument("--pump-amp", type=float, default=0.12)
    p.add_argument("--continuation-steps", type=int, default=12)

    # Toy circuit parameters
    p.add_argument("--damping", type=float, default=0.15)
    p.add_argument("--capacitance", type=float, default=1.0)
    p.add_argument("--k-ground", type=float, default=0.10)
    p.add_argument("--ic", type=float, default=1.0)
    p.add_argument("--phi0", type=float, default=1.0)

    # Pump solve controls
    p.add_argument("--pump-newton-tol", type=float, default=1e-9)
    p.add_argument("--pump-max-newton", type=int, default=16)
    p.add_argument("--pump-gmres-rtol", type=float, default=1e-7)
    p.add_argument("--pump-gmres-restart", type=int, default=40)
    p.add_argument("--pump-gmres-maxiter", type=int, default=40)

    # Gain/conversion controls
    p.add_argument("--sidebands", type=int, default=2)
    p.add_argument("--signal-omega", type=float, default=0.6)
    p.add_argument("--signal-start", type=float, default=0.2)
    p.add_argument("--signal-stop", type=float, default=1.8)
    p.add_argument("--signal-points", type=int, default=1)
    p.add_argument("--linear-tol", type=float, default=1e-9)

    # Toy input/output ports
    p.add_argument("--input-node", type=int, default=0)
    p.add_argument("--output-node", type=int, default=-1)
    p.add_argument("--y0", type=float, default=1.0)

    # Output
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.nodes <= 1:
        raise ValueError("--nodes must be > 1")

    input_node = args.input_node
    output_node = args.output_node if args.output_node >= 0 else args.nodes - 1

    if not (0 <= input_node < args.nodes):
        raise ValueError("--input-node out of range")
    if not (0 <= output_node < args.nodes):
        raise ValueError("--output-node out of range")
    if args.sidebands < 0:
        raise ValueError("--sidebands must be >= 0")

    print("=== experiment 03: gain from converged pump ===")
    print(f"nodes={args.nodes}")
    print(f"pump_harmonics={args.pump_harmonics}")
    print(f"pump_nt={args.pump_nt}")
    print(f"sidebands={args.sidebands}")
    print(f"pump_amp={args.pump_amp}")
    print(f"omega_p={args.omega_p}")
    print(f"input_node={input_node}")
    print(f"output_node={output_node}")

    pump = solve_pump_stage(args)
    print_pump_summary(pump)

    if not pump.converged:
        print("\nABORT: pump did not converge, so gain is not a clean object.")
        raise SystemExit(2)

    signal_omegas = signal_grid_from_args(args)

    if len(signal_omegas) == 1:
        result = solve_gain_at_signal(
            problem=pump.problem,
            Xpump=pump.Xpump,
            signal_omega=signal_omegas[0],
            sidebands=args.sidebands,
            input_node=input_node,
            output_node=output_node,
            y0=args.y0,
            linear_tol=args.linear_tol,
        )
        print_gain_result(result)
    else:
        print_sweep_header()
        for omega_s in signal_omegas:
            result = solve_gain_at_signal(
                problem=pump.problem,
                Xpump=pump.Xpump,
                signal_omega=float(omega_s),
                sidebands=args.sidebands,
                input_node=input_node,
                output_node=output_node,
                y0=args.y0,
                linear_tol=args.linear_tol,
            )
            print_sweep_row(result)


if __name__ == "__main__":
    main()