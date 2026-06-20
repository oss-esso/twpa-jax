# experiments/exp02_harmonic_coeff_pump.py
"""
Experiment 02: harmonic-coefficient nonlinear pump solve.

Why this exists:
    Exp01 used all time samples as Newton unknowns:
        unknowns = nt * n_nodes
    That was already slow at 8 nodes because GMRES stagnated.

    This experiment uses Fourier coefficients as unknowns:
        x(t) = 2 Re sum_{k=1..H} X_k exp(i k omega_p t)

    Unknown count:
        2 * H * n_nodes real unknowns

    Nonlinear current is still evaluated in time by AFT:
        coefficients -> time samples -> nonlinear current -> Fourier coefficients

Default method:
    Newton-Krylov with matrix-free JVP and sparse block preconditioner.

This is an experimental prototype, not polished package code.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# Harmonic grid
# =============================================================================

@dataclass
class HarmonicGrid:
    harmonics: int
    nt: int
    omega: float

    def __post_init__(self) -> None:
        if self.harmonics < 1:
            raise ValueError("harmonics must be >= 1")
        if self.nt < 2 * self.harmonics + 1:
            raise ValueError("nt must be >= 2*harmonics+1")
        if self.nt % 2 != 0:
            raise ValueError("Use even nt for now.")

        self.k = np.arange(1, self.harmonics + 1, dtype=float)
        self.period = 2.0 * np.pi / self.omega
        self.t = np.arange(self.nt, dtype=float) * self.period / self.nt

        # E[j,k] = exp(i k omega t_j), k starts at 1
        self.E = np.exp(1j * self.omega * self.t[:, None] * self.k[None, :])
        self.E_conj_T_over_nt = self.E.conj().T / self.nt

    def synthesize(self, X: np.ndarray) -> np.ndarray:
        """
        X shape:
            (H, n), complex positive-harmonic coefficients

        Returns:
            x_t shape (nt, n), real
        """
        return (2.0 * np.real(self.E @ X)).astype(float, copy=False)

    def synthesize_derivative(self, X: np.ndarray, order: int) -> np.ndarray:
        """
        Spectral derivative in time, returned as real samples.
        """
        if order == 0:
            return self.synthesize(X)

        multiplier = (1j * self.k * self.omega) ** order
        Xd = multiplier[:, None] * X
        return (2.0 * np.real(self.E @ Xd)).astype(float, copy=False)

    def project_positive(self, y_t: np.ndarray) -> np.ndarray:
        """
        Project real time samples onto positive complex harmonics.

        y_t shape:
            (nt, n)

        Returns:
            Y shape (H, n), complex, where
                y(t) ~= sum_k Y_k e^{ikwt} + conj(Y_k)e^{-ikwt}
        """
        return self.E_conj_T_over_nt @ y_t


# =============================================================================
# Branch laws
# =============================================================================

@dataclass
class JosephsonBranch:
    ic: float = 1.0
    phi0: float = 1.0

    def current(self, psi_t: np.ndarray) -> np.ndarray:
        return self.ic * np.sin(psi_t / self.phi0)

    def gamma(self, psi_t: np.ndarray) -> np.ndarray:
        return (self.ic / self.phi0) * np.cos(psi_t / self.phi0)


# =============================================================================
# Pump problem in harmonic coefficient form
# =============================================================================

@dataclass
class HarmonicPumpProblem:
    C: sp.csr_matrix
    G: sp.csr_matrix
    K: sp.csr_matrix
    Bphi: sp.csr_matrix
    branch: JosephsonBranch
    grid: HarmonicGrid
    pump_node: int
    pump_amp: float

    def __post_init__(self) -> None:
        self.C = self.C.tocsr()
        self.G = self.G.tocsr()
        self.K = self.K.tocsr()
        self.Bphi = self.Bphi.tocsr()

        self.n = self.C.shape[0]
        self.H = self.grid.harmonics

        if self.C.shape != (self.n, self.n):
            raise ValueError("C must be square")
        if self.G.shape != (self.n, self.n):
            raise ValueError("G shape mismatch")
        if self.K.shape != (self.n, self.n):
            raise ValueError("K shape mismatch")
        if self.Bphi.shape[0] != self.n:
            raise ValueError("Bphi row count mismatch")
        if not (0 <= self.pump_node < self.n):
            raise ValueError("pump_node out of range")

        self._linear_blocks = self._build_linear_blocks()

    def zeros(self) -> np.ndarray:
        return np.zeros((self.H, self.n), dtype=np.complex128)

    def _build_linear_blocks(self) -> list[sp.csc_matrix]:
        """
        D_k = K - (k omega)^2 C + i k omega G
        """
        blocks: list[sp.csc_matrix] = []
        for k in range(1, self.H + 1):
            omega_k = k * self.grid.omega
            Dk = self.K.astype(np.complex128)
            Dk = Dk + (-omega_k**2) * self.C.astype(np.complex128)
            Dk = Dk + (1j * omega_k) * self.G.astype(np.complex128)
            blocks.append(Dk.tocsc())
        return blocks

    def source_coeffs(self, source_scale: float) -> np.ndarray:
        """
        Source b(t) = lambda * pump_amp * cos(omega t) e_pump

        Positive harmonic coefficient at k=1 is amplitude/2.
        """
        S = np.zeros((self.H, self.n), dtype=np.complex128)
        S[0, self.pump_node] = 0.5 * source_scale * self.pump_amp
        return S

    def source_time(self, source_scale: float) -> np.ndarray:
        src = np.zeros((self.grid.nt, self.n), dtype=float)
        src[:, self.pump_node] = (
            source_scale
            * self.pump_amp
            * np.cos(self.grid.omega * self.grid.t)
        )
        return src

    def branch_flux_time(self, X: np.ndarray) -> np.ndarray:
        x_t = self.grid.synthesize(X)
        return (self.Bphi.T @ x_t.T).T

    def nonlinear_current_time(self, X: np.ndarray) -> np.ndarray:
        psi_t = self.branch_flux_time(X)
        i_t = self.branch.current(psi_t)
        return (self.Bphi @ i_t.T).T

    def nonlinear_current_coeffs(self, X: np.ndarray) -> np.ndarray:
        nl_t = self.nonlinear_current_time(X)
        return self.grid.project_positive(nl_t)

    def residual_coeffs(self, X: np.ndarray, source_scale: float) -> np.ndarray:
        """
        R_k = D_k X_k + N_k(X) - S_k
        """
        R = np.empty_like(X)
        Ncoeff = self.nonlinear_current_coeffs(X)
        S = self.source_coeffs(source_scale)

        for h in range(self.H):
            R[h] = self._linear_blocks[h] @ X[h] + Ncoeff[h] - S[h]

        return R

    def jvp_coeffs(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Matrix-free JVP:
            DF[X] V = D_k V_k + projection of B Gamma(x(t)) B^T v(t)
        """
        Vlin = np.empty_like(V)
        for h in range(self.H):
            Vlin[h] = self._linear_blocks[h] @ V[h]

        x_t = self.grid.synthesize(X)
        v_t = self.grid.synthesize(V)

        psi_t = (self.Bphi.T @ x_t.T).T
        dpsi_t = (self.Bphi.T @ v_t.T).T
        gamma_t = self.branch.gamma(psi_t)

        di_t = gamma_t * dpsi_t
        dn_t = (self.Bphi @ di_t.T).T
        DN = self.grid.project_positive(dn_t)

        return Vlin + DN

    def time_residual(self, X: np.ndarray, source_scale: float) -> np.ndarray:
        """
        Full time-domain residual, including harmonics outside the retained basis.
        This is a truncation diagnostic, not the Newton residual.
        """
        x_t = self.grid.synthesize(X)
        dx_t = self.grid.synthesize_derivative(X, order=1)
        ddx_t = self.grid.synthesize_derivative(X, order=2)

        r = (self.C @ ddx_t.T).T
        r = r + (self.G @ dx_t.T).T
        r = r + (self.K @ x_t.T).T
        r = r + self.nonlinear_current_time(X)
        r = r - self.source_time(source_scale)
        return np.asarray(r, dtype=float)

    def norms(self, X: np.ndarray, source_scale: float) -> dict[str, float]:
        R = self.residual_coeffs(X, source_scale)
        R_flat = pack_complex(R)
        coeff_abs = np.linalg.norm(R_flat) / np.sqrt(R_flat.size)

        S = self.source_coeffs(source_scale)
        S_flat = pack_complex(S)
        source_abs = np.linalg.norm(S_flat) / max(np.sqrt(S_flat.size), 1.0)

        coeff_rel = coeff_abs / max(source_abs, 1e-14)

        Rt = self.time_residual(X, source_scale)
        time_abs = np.linalg.norm(Rt.ravel()) / np.sqrt(Rt.size)

        St = self.source_time(source_scale)
        src_t_abs = np.linalg.norm(St.ravel()) / max(np.sqrt(St.size), 1.0)
        time_rel = time_abs / max(src_t_abs, 1e-14)

        return {
            "coeff_abs": float(coeff_abs),
            "coeff_rel": float(coeff_rel),
            "time_abs": float(time_abs),
            "time_rel": float(time_rel),
        }

    def build_preconditioner_factors(self, X: np.ndarray) -> list[spla.SuperLU]:
        """
        Build block diagonal preconditioner:
            P_k = D_k + B diag(mean_t gamma(t)) B^T

        This approximates the diagonal part of the current Jacobian.
        It ignores harmonic mixing in gamma(t), which is fine for preconditioning.
        """
        psi_t = self.branch_flux_time(X)
        gamma_mean = np.mean(self.branch.gamma(psi_t), axis=0)

        Ktan = (
            self.Bphi
            @ sp.diags(gamma_mean, offsets=0, format="csr")
            @ self.Bphi.T
        ).astype(np.complex128).tocsc()

        factors: list[spla.SuperLU] = []
        for h in range(self.H):
            Pk = (self._linear_blocks[h] + Ktan).tocsc()
            factors.append(spla.splu(Pk))
        return factors


# =============================================================================
# Packing utilities
# =============================================================================

def pack_complex(X: np.ndarray) -> np.ndarray:
    """
    Complex array -> real vector [real, imag].
    """
    z = np.asarray(X, dtype=np.complex128).ravel()
    return np.concatenate([z.real, z.imag])


def unpack_complex(v: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Real vector [real, imag] -> complex array of shape.
    """
    v = np.asarray(v, dtype=float)
    size = shape[0] * shape[1]
    return (v[:size] + 1j * v[size:2 * size]).reshape(shape)


# =============================================================================
# Solver
# =============================================================================

@dataclass
class NewtonKrylovSettings:
    newton_tol: float = 1e-9
    max_newton: int = 20
    gmres_rtol: float = 1e-7
    gmres_atol: float = 0.0
    gmres_restart: int = 40
    gmres_maxiter: int = 40
    min_alpha: float = 1.0 / 1024.0
    verbose: bool = True


@dataclass
class SolveReport:
    converged: bool
    source_scale: float
    coeff_abs: float
    coeff_rel: float
    time_abs: float
    time_rel: float
    newton_iterations: int
    gmres_iterations_total: int
    runtime_s: float
    failure_reason: str


class HarmonicNewtonKrylovSolver:
    def __init__(self, settings: NewtonKrylovSettings):
        self.settings = settings

    def solve_one(
        self,
        problem: HarmonicPumpProblem,
        X0: np.ndarray,
        source_scale: float,
    ) -> tuple[np.ndarray, SolveReport]:
        s = self.settings
        t0 = time.perf_counter()
        X = np.array(X0, dtype=np.complex128, copy=True)

        nrm = problem.norms(X, source_scale)
        if s.verbose:
            print(
                f"  init: lambda={source_scale:.5f} "
                f"coeff_rel={nrm['coeff_rel']:.3e} "
                f"time_rel={nrm['time_rel']:.3e}"
            )

        if nrm["coeff_rel"] < s.newton_tol:
            return X, self._make_report(
                True, source_scale, nrm, 0, 0, t0, ""
            )

        gmres_total = 0
        failure_reason = ""

        shape = X.shape
        dim_real = 2 * shape[0] * shape[1]

        for it in range(1, s.max_newton + 1):
            R = problem.residual_coeffs(X, source_scale)
            rhs = -pack_complex(R)

            factors = problem.build_preconditioner_factors(X)

            def matvec(v_real: np.ndarray) -> np.ndarray:
                V = unpack_complex(v_real, shape)
                JV = problem.jvp_coeffs(X, V)
                return pack_complex(JV)

            def psolve(v_real: np.ndarray) -> np.ndarray:
                V = unpack_complex(v_real, shape)
                Z = np.empty_like(V)
                for h in range(problem.H):
                    Z[h] = factors[h].solve(V[h])
                return pack_complex(Z)

            Aop = spla.LinearOperator(
                shape=(dim_real, dim_real),
                matvec=matvec,
                dtype=np.float64,
            )
            Mop = spla.LinearOperator(
                shape=(dim_real, dim_real),
                matvec=psolve,
                dtype=np.float64,
            )

            gmres_counter = {"n": 0}

            def cb(_pr_norm: float) -> None:
                gmres_counter["n"] += 1

            delta_real, info = spla.gmres(
                Aop,
                rhs,
                M=Mop,
                rtol=s.gmres_rtol,
                atol=s.gmres_atol,
                restart=s.gmres_restart,
                maxiter=s.gmres_maxiter,
                callback=cb,
                callback_type="pr_norm",
            )
            gmres_total += gmres_counter["n"]

            if info != 0:
                failure_reason = f"GMRES did not fully converge, info={info}"

            Delta = unpack_complex(delta_real, shape)

            # Backtracking line search on projected coefficient residual.
            accepted = False
            alpha = 1.0
            best_X = X
            best_nrm = nrm

            while alpha >= s.min_alpha:
                Xtrial = X + alpha * Delta
                trial_nrm = problem.norms(Xtrial, source_scale)

                if trial_nrm["coeff_rel"] < nrm["coeff_rel"]:
                    accepted = True
                    best_X = Xtrial
                    best_nrm = trial_nrm
                    break

                alpha *= 0.5

            if not accepted:
                failure_reason = failure_reason or f"line search failed at Newton {it}"
                return X, self._make_report(
                    False, source_scale, nrm, it, gmres_total, t0, failure_reason
                )

            X = best_X
            nrm = best_nrm

            if s.verbose:
                print(
                    f"  newton={it:02d} alpha={alpha:.3e} "
                    f"gmres_it={gmres_counter['n']:04d} "
                    f"coeff_rel={nrm['coeff_rel']:.3e} "
                    f"time_rel={nrm['time_rel']:.3e}"
                )

            if nrm["coeff_rel"] < s.newton_tol:
                return X, self._make_report(
                    True, source_scale, nrm, it, gmres_total, t0, ""
                )

        failure_reason = failure_reason or "maximum Newton iterations reached"
        return X, self._make_report(
            False, source_scale, nrm, s.max_newton, gmres_total, t0, failure_reason
        )

    def solve_continuation(
        self,
        problem: HarmonicPumpProblem,
        continuation_steps: int,
    ) -> tuple[np.ndarray, list[SolveReport]]:
        X = problem.zeros()
        reports: list[SolveReport] = []

        lambdas = np.linspace(1.0 / continuation_steps, 1.0, continuation_steps)

        for lam in lambdas:
            print(f"\n=== continuation lambda={lam:.5f} ===")
            X, report = self.solve_one(problem, X, float(lam))
            reports.append(report)

            status = "VALID_CONVERGED" if report.converged else "FAIL"
            print(
                f"step_status={status} "
                f"coeff_rel={report.coeff_rel:.3e} "
                f"time_rel={report.time_rel:.3e} "
                f"newton={report.newton_iterations} "
                f"gmres_total={report.gmres_iterations_total} "
                f"runtime_s={report.runtime_s:.3f} "
                f"reason={report.failure_reason}"
            )

            if not report.converged:
                break

        return X, reports

    @staticmethod
    def _make_report(
        converged: bool,
        source_scale: float,
        nrm: dict[str, float],
        newton_iterations: int,
        gmres_iterations_total: int,
        t0: float,
        failure_reason: str,
    ) -> SolveReport:
        return SolveReport(
            converged=converged,
            source_scale=source_scale,
            coeff_abs=nrm["coeff_abs"],
            coeff_rel=nrm["coeff_rel"],
            time_abs=nrm["time_abs"],
            time_rel=nrm["time_rel"],
            newton_iterations=newton_iterations,
            gmres_iterations_total=gmres_iterations_total,
            runtime_s=time.perf_counter() - t0,
            failure_reason=failure_reason,
        )


# =============================================================================
# Fixture
# =============================================================================

def make_chain_incidence(n_nodes: int) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

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
    harmonics: int,
    nt: int,
    omega_p: float,
    pump_amp: float,
    damping: float,
    c_value: float,
    k_ground: float,
    ic: float,
    phi0: float,
) -> HarmonicPumpProblem:
    """
    Dimensionless tiny nonlinear ladder.

    Still not a physical TWPA.
    It is a fast nonlinear periodic test fixture.

    C = c I
    G = damping I
    K = k_ground I
    nonlinear Josephson branches between neighboring nodes
    pump source at node 0
    """
    C = sp.eye(n_nodes, format="csr") * c_value
    G = sp.eye(n_nodes, format="csr") * damping
    K = sp.eye(n_nodes, format="csr") * k_ground

    Bphi = make_chain_incidence(n_nodes)
    branch = JosephsonBranch(ic=ic, phi0=phi0)
    grid = HarmonicGrid(harmonics=harmonics, nt=nt, omega=omega_p)

    return HarmonicPumpProblem(
        C=C,
        G=G,
        K=K,
        Bphi=Bphi,
        branch=branch,
        grid=grid,
        pump_node=0,
        pump_amp=pump_amp,
    )


# =============================================================================
# Diagnostics
# =============================================================================

def summarize_solution(problem: HarmonicPumpProblem, X: np.ndarray) -> None:
    x_t = problem.grid.synthesize(X)
    psi_t = problem.branch_flux_time(X)
    i_t = problem.branch.current(psi_t)

    print("\n=== solution summary ===")
    print(f"x_rms={np.sqrt(np.mean(x_t**2)):.6e}")
    print(f"x_max_abs={np.max(np.abs(x_t)):.6e}")
    print(f"branch_psi_rms={np.sqrt(np.mean(psi_t**2)):.6e}")
    print(f"branch_psi_max_abs={np.max(np.abs(psi_t)):.6e}")
    print(f"branch_i_rms={np.sqrt(np.mean(i_t**2)):.6e}")
    print(f"branch_i_max_abs={np.max(np.abs(i_t)):.6e}")

    for h in range(problem.H):
        print(f"X_h{h+1}_norm={np.linalg.norm(X[h]):.6e}")


def print_final_report(
    problem: HarmonicPumpProblem,
    reports: list[SolveReport],
    total_runtime_s: float,
) -> None:
    final = reports[-1] if reports else None
    converged = bool(final and final.converged and abs(final.source_scale - 1.0) < 1e-12)

    print("\n=== final report ===")
    print(f"status={'VALID_CONVERGED' if converged else 'FAIL'}")
    print(f"nodes={problem.n}")
    print(f"branches={problem.Bphi.shape[1]}")
    print(f"harmonics={problem.H}")
    print(f"time_samples_for_aft={problem.grid.nt}")
    print(f"real_unknowns={2 * problem.H * problem.n}")
    print(f"omega_p={problem.grid.omega}")
    print(f"pump_amp={problem.pump_amp}")
    print(f"continuation_steps_completed={len(reports)}")
    print(f"total_runtime_s={total_runtime_s:.3f}")

    if final:
        print(f"final_lambda={final.source_scale}")
        print(f"final_coeff_abs={final.coeff_abs:.6e}")
        print(f"final_coeff_rel={final.coeff_rel:.6e}")
        print(f"final_time_abs={final.time_abs:.6e}")
        print(f"final_time_rel={final.time_rel:.6e}")
        print(f"final_newton_iterations_last_step={final.newton_iterations}")
        print(f"final_gmres_iterations_last_step={final.gmres_iterations_total}")
        print(f"failure_reason={final.failure_reason}")

    print("\ninterpretation:")
    print("  coeff_rel = residual in retained harmonic basis; solver convergence.")
    print("  time_rel  = full time residual; harmonic truncation diagnostic.")
    print("  If coeff_rel is tiny but time_rel is not, increase --harmonics and --nt.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(conflict_handler="resolve")

    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--harmonics", type=int, default=1)
    p.add_argument("--nt", type=int, default=16)
    p.add_argument("--omega-p", type=float, default=1.0)
    p.add_argument("--pump-amp", type=float, default=0.12)
    p.add_argument("--continuation-steps", type=int, default=12)

    p.add_argument("--damping", type=float, default=0.15)
    p.add_argument("--capacitance", type=float, default=1.0)
    p.add_argument("--k-ground", type=float, default=0.10)
    p.add_argument("--ic", type=float, default=1.0)
    p.add_argument("--phi0", type=float, default=1.0)
    p.add_argument("--phi0", type=float, default=1.0)

    p.add_argument("--newton-tol", type=float, default=1e-9)
    p.add_argument("--max-newton", type=int, default=16)
    p.add_argument("--gmres-rtol", type=float, default=1e-7)
    p.add_argument("--gmres-restart", type=int, default=40)
    p.add_argument("--gmres-maxiter", type=int, default=40)
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    problem = make_tiny_ladder_problem(
        n_nodes=args.nodes,
        harmonics=args.harmonics,
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
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
        verbose=not args.quiet,
    )

    solver = HarmonicNewtonKrylovSolver(settings)

    print("=== experiment 02: harmonic-coefficient pump solve ===")
    print(f"nodes={problem.n}")
    print(f"harmonics={problem.H}")
    print(f"nt={problem.grid.nt}")
    print(f"real_unknowns={2 * problem.H * problem.n}")
    print(f"pump_amp={problem.pump_amp}")
    print(f"continuation_steps={args.continuation_steps}")

    t0 = time.perf_counter()
    X, reports = solver.solve_continuation(
        problem,
        continuation_steps=args.continuation_steps,
    )
    total_runtime_s = time.perf_counter() - t0

    summarize_solution(problem, X)
    print_final_report(problem, reports, total_runtime_s)


if __name__ == "__main__":
    main()
