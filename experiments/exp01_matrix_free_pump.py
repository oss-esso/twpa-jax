# experiments/exp01_matrix_free_pump.py
"""
Experiment 01: matrix-free nonlinear periodic pump solve.

Goal:
    Test the core pump-solve skeleton quickly:
        F[x] = C xddot + G xdot + K x + B i(B^T x) - source = 0

Representation:
    Unknown x is stored as real time samples over one pump period.
    Derivatives are pseudospectral using FFT.
    The nonlinear branch law is evaluated in time.
    Newton corrections are solved by GMRES using a matrix-free JVP.

This is deliberately experimental:
    - no package ceremony
    - no gain solve yet
    - no dense Jacobian
    - prints decision-relevant diagnostics
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------
# Fourier pseudospectral grid
# ---------------------------------------------------------------------

@dataclass
class FourierTimeGrid:
    nt: int
    omega: float

    def __post_init__(self) -> None:
        if self.nt < 8:
            raise ValueError("nt should be at least 8 for this experiment.")
        if self.nt % 2 != 0:
            raise ValueError("Use an even nt for now.")
        self.period = 2.0 * np.pi / self.omega
        self.t = np.arange(self.nt, dtype=float) * self.period / self.nt

        # Integer Fourier modes corresponding to np.fft ordering:
        # [0, 1, 2, ..., nt/2-1, -nt/2, ..., -1]
        self.k = np.fft.fftfreq(self.nt, d=1.0 / self.nt)

    def derivative(self, x_t: np.ndarray, order: int) -> np.ndarray:
        """
        Spectral derivative along axis 0.

        x_t shape:
            (nt, n)

        Returns:
            real array of shape (nt, n)
        """
        if order == 0:
            return x_t

        X = np.fft.fft(x_t, axis=0)
        multiplier = (1j * self.k * self.omega) ** order
        out = np.fft.ifft(multiplier[:, None] * X, axis=0)

        # Physical solutions are real. Numerical imaginary part is roundoff.
        return out.real


# ---------------------------------------------------------------------
# Branch laws
# ---------------------------------------------------------------------

@dataclass
class JosephsonBranch:
    ic: float = 1.0
    phi0: float = 1.0

    def current(self, psi_t: np.ndarray) -> np.ndarray:
        return self.ic * np.sin(psi_t / self.phi0)

    def gamma(self, psi_t: np.ndarray) -> np.ndarray:
        """
        Differential inverse inductance:
            d i / d psi
        """
        return (self.ic / self.phi0) * np.cos(psi_t / self.phi0)


# ---------------------------------------------------------------------
# Pump problem
# ---------------------------------------------------------------------

@dataclass
class PumpProblem:
    C: sp.csr_matrix
    G: sp.csr_matrix
    K: sp.csr_matrix
    Bphi: sp.csr_matrix
    branch: JosephsonBranch
    grid: FourierTimeGrid
    pump_node: int
    pump_amp: float

    def __post_init__(self) -> None:
        self.C = self.C.tocsr()
        self.G = self.G.tocsr()
        self.K = self.K.tocsr()
        self.Bphi = self.Bphi.tocsr()

        self.n = self.C.shape[0]
        if self.C.shape != (self.n, self.n):
            raise ValueError("C must be square.")
        if self.G.shape != (self.n, self.n):
            raise ValueError("G must match C.")
        if self.K.shape != (self.n, self.n):
            raise ValueError("K must match C.")
        if self.Bphi.shape[0] != self.n:
            raise ValueError("Bphi must have n rows.")

        self.source_unit = np.zeros((self.grid.nt, self.n), dtype=float)
        self.source_unit[:, self.pump_node] = np.cos(self.grid.omega * self.grid.t)

    def zeros(self) -> np.ndarray:
        return np.zeros((self.grid.nt, self.n), dtype=float)

    def source(self, source_scale: float) -> np.ndarray:
        return source_scale * self.pump_amp * self.source_unit

    def _apply_mat_time(self, A: sp.csr_matrix, x_t: np.ndarray) -> np.ndarray:
        """
        Apply sparse matrix A to every time sample.

        A:   (n, n)
        x_t: (nt, n)

        returns:
             (nt, n)
        """
        return (A @ x_t.T).T

    def branch_flux(self, x_t: np.ndarray) -> np.ndarray:
        """
        psi(t) = Bphi^T x(t)

        returns:
            (nt, n_branches)
        """
        return (self.Bphi.T @ x_t.T).T

    def nonlinear_current_nodes(self, x_t: np.ndarray) -> np.ndarray:
        """
        Bphi * i(Bphi^T x)
        """
        psi_t = self.branch_flux(x_t)
        i_t = self.branch.current(psi_t)
        return (self.Bphi @ i_t.T).T

    def residual(self, x_t: np.ndarray, source_scale: float) -> np.ndarray:
        """
        Time-domain residual:
            C xddot + G xdot + K x + B i(B^T x) - source
        """
        dx_t = self.grid.derivative(x_t, order=1)
        ddx_t = self.grid.derivative(x_t, order=2)

        r = self._apply_mat_time(self.C, ddx_t)
        r += self._apply_mat_time(self.G, dx_t)
        r += self._apply_mat_time(self.K, x_t)
        r += self.nonlinear_current_nodes(x_t)
        r -= self.source(source_scale)
        return r

    def jvp(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """
        Jacobian-vector product:
            DF[x] v =
                C vddot + G vdot + K v
                + B Gamma(B^T x) B^T v
        """
        dv_t = self.grid.derivative(v_t, order=1)
        ddv_t = self.grid.derivative(v_t, order=2)

        out = self._apply_mat_time(self.C, ddv_t)
        out += self._apply_mat_time(self.G, dv_t)
        out += self._apply_mat_time(self.K, v_t)

        psi_t = self.branch_flux(x_t)
        dpsi_t = self.branch_flux(v_t)
        gamma_t = self.branch.gamma(psi_t)

        # Branch tangent current:
        #     di = gamma * dpsi
        di_t = gamma_t * dpsi_t
        out += (self.Bphi @ di_t.T).T
        return out

    def residual_norms(self, x_t: np.ndarray, source_scale: float) -> tuple[float, float]:
        r = self.residual(x_t, source_scale)
        abs_rms = np.linalg.norm(r.ravel()) / np.sqrt(r.size)

        src = self.source(source_scale)
        src_rms = np.linalg.norm(src.ravel()) / max(np.sqrt(src.size), 1.0)
        rel = abs_rms / max(src_rms, 1e-14)

        return abs_rms, rel


# ---------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------

@dataclass
class NewtonKrylovSettings:
    newton_tol: float = 1e-9
    max_newton: int = 20
    gmres_rtol: float = 1e-5
    gmres_atol: float = 0.0
    gmres_restart: int = 80
    gmres_maxiter: int = 200
    min_alpha: float = 1.0 / 1024.0
    verbose: bool = True


@dataclass
class SolveReport:
    converged: bool
    source_scale: float
    abs_residual: float
    rel_residual: float
    newton_iterations: int
    gmres_iterations_total: int
    runtime_s: float
    failure_reason: str


class NewtonKrylovPumpSolver:
    def __init__(self, settings: NewtonKrylovSettings):
        self.settings = settings

    @staticmethod
    def _pack(x_t: np.ndarray) -> np.ndarray:
        return np.asarray(x_t, dtype=float).ravel()

    @staticmethod
    def _unpack(z: np.ndarray, nt: int, n: int) -> np.ndarray:
        return np.asarray(z, dtype=float).reshape((nt, n))

    def solve_one(
        self,
        problem: PumpProblem,
        x0_t: np.ndarray,
        source_scale: float,
    ) -> tuple[np.ndarray, SolveReport]:
        s = self.settings
        t0 = time.perf_counter()

        nt, n = problem.grid.nt, problem.n
        x_t = np.array(x0_t, dtype=float, copy=True)

        abs_r, rel_r = problem.residual_norms(x_t, source_scale)
        if s.verbose:
            print(
                f"  init: lambda={source_scale:.5f} "
                f"abs_rms={abs_r:.3e} rel={rel_r:.3e}"
            )

        if rel_r < s.newton_tol:
            return x_t, SolveReport(
                converged=True,
                source_scale=source_scale,
                abs_residual=abs_r,
                rel_residual=rel_r,
                newton_iterations=0,
                gmres_iterations_total=0,
                runtime_s=time.perf_counter() - t0,
                failure_reason="",
            )

        gmres_iterations_total = 0
        failure_reason = ""

        for it in range(1, s.max_newton + 1):
            r_t = problem.residual(x_t, source_scale)
            rhs = -self._pack(r_t)

            def matvec(v_flat: np.ndarray) -> np.ndarray:
                v_t = self._unpack(v_flat, nt, n)
                return self._pack(problem.jvp(x_t, v_t))

            Aop = spla.LinearOperator(
                shape=(nt * n, nt * n),
                matvec=matvec,
                dtype=np.float64,
            )

            gmres_counter = {"n": 0}

            def cb(_pr_norm: float) -> None:
                gmres_counter["n"] += 1

            delta_flat, info = spla.gmres(
                Aop,
                rhs,
                rtol=s.gmres_rtol,
                atol=s.gmres_atol,
                restart=s.gmres_restart,
                maxiter=s.gmres_maxiter,
                callback=cb,
                callback_type="pr_norm",
            )
            gmres_iterations_total += gmres_counter["n"]

            if info != 0:
                failure_reason = f"GMRES did not fully converge, info={info}"
                # We still try the step; sometimes an inexact Newton step is enough.

            delta_t = self._unpack(delta_flat, nt, n)

            # Backtracking line search on residual norm
            accepted = False
            alpha = 1.0
            best_abs, best_rel = abs_r, rel_r
            best_x = x_t

            while alpha >= s.min_alpha:
                trial_x = x_t + alpha * delta_t
                trial_abs, trial_rel = problem.residual_norms(trial_x, source_scale)

                if trial_rel < rel_r:
                    accepted = True
                    best_x = trial_x
                    best_abs = trial_abs
                    best_rel = trial_rel
                    break

                alpha *= 0.5

            if not accepted:
                failure_reason = (
                    failure_reason
                    or f"line search failed at Newton iteration {it}"
                )
                runtime = time.perf_counter() - t0
                return x_t, SolveReport(
                    converged=False,
                    source_scale=source_scale,
                    abs_residual=abs_r,
                    rel_residual=rel_r,
                    newton_iterations=it,
                    gmres_iterations_total=gmres_iterations_total,
                    runtime_s=runtime,
                    failure_reason=failure_reason,
                )

            x_t = best_x
            abs_r, rel_r = best_abs, best_rel

            if s.verbose:
                print(
                    f"  newton={it:02d} alpha={alpha:.3e} "
                    f"gmres_it={gmres_counter['n']:03d} "
                    f"abs_rms={abs_r:.3e} rel={rel_r:.3e}"
                )

            if rel_r < s.newton_tol:
                runtime = time.perf_counter() - t0
                return x_t, SolveReport(
                    converged=True,
                    source_scale=source_scale,
                    abs_residual=abs_r,
                    rel_residual=rel_r,
                    newton_iterations=it,
                    gmres_iterations_total=gmres_iterations_total,
                    runtime_s=runtime,
                    failure_reason="",
                )

        failure_reason = failure_reason or "maximum Newton iterations reached"
        runtime = time.perf_counter() - t0
        return x_t, SolveReport(
            converged=False,
            source_scale=source_scale,
            abs_residual=abs_r,
            rel_residual=rel_r,
            newton_iterations=s.max_newton,
            gmres_iterations_total=gmres_iterations_total,
            runtime_s=runtime,
            failure_reason=failure_reason,
        )

    def solve_continuation(
        self,
        problem: PumpProblem,
        continuation_steps: int,
    ) -> tuple[np.ndarray, list[SolveReport]]:
        x_t = problem.zeros()
        reports: list[SolveReport] = []

        lambdas = np.linspace(1.0 / continuation_steps, 1.0, continuation_steps)

        for lam in lambdas:
            print(f"\n=== continuation lambda={lam:.5f} ===")
            x_t, report = self.solve_one(problem, x_t, source_scale=float(lam))
            reports.append(report)

            status = "VALID_CONVERGED" if report.converged else "FAIL"
            print(
                f"step_status={status} "
                f"rel={report.rel_residual:.3e} "
                f"newton={report.newton_iterations} "
                f"gmres_total={report.gmres_iterations_total} "
                f"runtime_s={report.runtime_s:.3f} "
                f"reason={report.failure_reason}"
            )

            if not report.converged:
                break

        return x_t, reports


# ---------------------------------------------------------------------
# Tiny fixture
# ---------------------------------------------------------------------

def make_chain_incidence(n_nodes: int) -> sp.csr_matrix:
    """
    Series branches between neighboring nodes:
        branch j connects node j -> node j+1.

    B has shape (n_nodes, n_nodes-1).
    """
    rows = []
    cols = []
    data = []

    for j in range(n_nodes - 1):
        rows.extend([j, j + 1])
        cols.extend([j, j])
        data.extend([1.0, -1.0])

    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_nodes, n_nodes - 1),
    ).tocsr()


def make_tiny_ladder_problem(
    n_nodes: int,
    nt: int,
    omega_p: float,
    pump_amp: float,
    damping: float,
    c_value: float,
    k_ground: float,
    ic: float,
    phi0: float,
) -> PumpProblem:
    """
    Dimensionless tiny nonlinear ladder.

    This is not yet a calibrated physical TWPA.
    It is the smallest useful nonlinear periodic test fixture.

    C = c I
    G = damping I
    K = k_ground I
    nonlinear branches = Josephson branches between neighboring nodes
    source = pump_amp cos(omega_p t) at node 0
    """
    C = sp.eye(n_nodes, format="csr") * c_value
    G = sp.eye(n_nodes, format="csr") * damping
    K = sp.eye(n_nodes, format="csr") * k_ground

    Bphi = make_chain_incidence(n_nodes)
    branch = JosephsonBranch(ic=ic, phi0=phi0)
    grid = FourierTimeGrid(nt=nt, omega=omega_p)

    return PumpProblem(
        C=C,
        G=G,
        K=K,
        Bphi=Bphi,
        branch=branch,
        grid=grid,
        pump_node=0,
        pump_amp=pump_amp,
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def summarize_solution(problem: PumpProblem, x_t: np.ndarray) -> None:
    psi_t = problem.branch_flux(x_t)
    i_t = problem.branch.current(psi_t)

    print("\n=== solution summary ===")
    print(f"x_rms={np.sqrt(np.mean(x_t**2)):.6e}")
    print(f"x_max_abs={np.max(np.abs(x_t)):.6e}")
    print(f"branch_psi_rms={np.sqrt(np.mean(psi_t**2)):.6e}")
    print(f"branch_psi_max_abs={np.max(np.abs(psi_t)):.6e}")
    print(f"branch_i_rms={np.sqrt(np.mean(i_t**2)):.6e}")
    print(f"branch_i_max_abs={np.max(np.abs(i_t)):.6e}")


def print_final_report(
    problem: PumpProblem,
    reports: list[SolveReport],
    total_runtime_s: float,
) -> None:
    final = reports[-1] if reports else None
    converged = bool(final and final.converged and abs(final.source_scale - 1.0) < 1e-12)

    print("\n=== final report ===")
    print(f"status={'VALID_CONVERGED' if converged else 'FAIL'}")
    print(f"nodes={problem.n}")
    print(f"time_samples={problem.grid.nt}")
    print(f"unknowns={problem.n * problem.grid.nt}")
    print(f"branches={problem.Bphi.shape[1]}")
    print(f"omega_p={problem.grid.omega}")
    print(f"pump_amp={problem.pump_amp}")
    print(f"continuation_steps_completed={len(reports)}")
    print(f"total_runtime_s={total_runtime_s:.3f}")

    if final:
        print(f"final_lambda={final.source_scale}")
        print(f"final_abs_residual={final.abs_residual:.6e}")
        print(f"final_rel_residual={final.rel_residual:.6e}")
        print(f"final_newton_iterations_last_step={final.newton_iterations}")
        print(f"final_gmres_iterations_last_step={final.gmres_iterations_total}")
        print(f"failure_reason={final.failure_reason}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--nt", type=int, default=32)
    p.add_argument("--omega-p", type=float, default=1.0)
    p.add_argument("--pump-amp", type=float, default=0.12)
    p.add_argument("--continuation-steps", type=int, default=12)

    p.add_argument("--damping", type=float, default=0.15)
    p.add_argument("--capacitance", type=float, default=1.0)
    p.add_argument("--k-ground", type=float, default=0.10)
    p.add_argument("--ic", type=float, default=1.0)
    p.add_argument("--phi0", type=float, default=1.0)

    p.add_argument("--newton-tol", type=float, default=1e-9)
    p.add_argument("--max-newton", type=int, default=18)
    p.add_argument("--gmres-rtol", type=float, default=1e-5)
    p.add_argument("--gmres-maxiter", type=int, default=200)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    problem = make_tiny_ladder_problem(
        n_nodes=args.nodes,
        nt=args.nt,
        omega_p=args.omega_p,
        pump_amp=args.pump_amp,
        damping=args.damping,
        c_value=args.capacitance,
        k_ground=args.k_ground,
        ic=args.ic,
        phi0=args.phi0,
    )

    settings = NewtonKrylovSettings(
        newton_tol=args.newton_tol,
        max_newton=args.max_newton,
        gmres_rtol=args.gmres_rtol,
        gmres_maxiter=args.gmres_maxiter,
        verbose=not args.quiet,
    )
    solver = NewtonKrylovPumpSolver(settings=settings)

    t0 = time.perf_counter()
    x_t, reports = solver.solve_continuation(
        problem,
        continuation_steps=args.continuation_steps,
    )
    total_runtime_s = time.perf_counter() - t0

    summarize_solution(problem, x_t)
    print_final_report(problem, reports, total_runtime_s)


if __name__ == "__main__":
    main()