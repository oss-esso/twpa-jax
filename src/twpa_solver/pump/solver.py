from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from twpa_solver.pump.problem import (
    FullPumpProblem,
    pack_complex,
    unpack_complex,
)

@dataclass
class NewtonKrylovSettings:
    newton_tol: float
    max_newton: int
    gmres_rtol: float
    gmres_atol: float
    gmres_restart: int
    gmres_maxiter: int
    min_alpha: float
    preconditioner: str
    compute_time_residual: bool
    verbose: bool
    continuation_predictor: str
    jvp_mode: str
    # Stall detection: abort a Newton solve that is making negligible progress
    # (residual reduction ratio above stall_ratio for stall_patience consecutive
    # accepted steps while still far from tolerance). This stops the solver from
    # grinding all max_newton iterations on an over-fold / unsolvable operating
    # point, which is the dominant cost when sweeping past the pump-power fold.
    # stall_patience <= 0 disables it.
    stall_ratio: float = 0.8
    stall_patience: int = 4
    # Hard wall-time budget per solve_one (seconds). 0 disables. Aborts a solve
    # that exceeds the budget (over-fold points near the harmonic-balance fold
    # are stiff: each Newton step runs GMRES to maxiter, so they grind without a
    # clean stall signal). Convergent warm-start points finish well under it.
    solve_deadline_s: float = 0.0
    # Preconditioner factor reuse (modified-Newton preconditioning). The
    # preconditioner is only a GMRES accelerator -- the Newton update is always
    # computed against the true Jacobian via the matvec -- so reusing one stale
    # factor across several Newton steps does not change the converged solution,
    # it only trades a few extra GMRES iterations for skipping the (expensive,
    # exact) LU rebuild. Dominant win for ``real_coupled`` near the fold, where
    # the full-Jacobian LU is re-factored every step but barely changes.
    # precond_reuse = 1 refactors every step (legacy). N>1 reuses for up to N
    # consecutive steps. precond_reuse_refresh_gmres > 0 forces an early refresh
    # whenever the previous step's GMRES iteration count crossed the threshold
    # (staleness guard: the factor has drifted too far to precondition well).
    precond_reuse: int = 1
    precond_reuse_refresh_gmres: int = 0


@dataclass
class StepReport:
    converged: bool
    source_scale: float
    coeff_abs: float
    coeff_rel: float
    time_abs: float | None
    time_rel: float | None
    newton_iterations: int
    gmres_iterations_total: int
    factor_runtime_s: float
    runtime_s: float
    failure_reason: str
    preconditioner_assembly_runtime_s: float = 0.0
    preconditioner_numeric_factor_runtime_s: float = 0.0


@dataclass
class ContinuationTrace:
    mode: str
    attempted_lambdas: list[float]
    accepted_lambdas: list[float]
    failed_attempts: int
    accepted_steps: int
    fallback_used: bool
    failure_reason: str


def empty_continuation_trace(mode: str) -> ContinuationTrace:
    return ContinuationTrace(
        mode=mode,
        attempted_lambdas=[],
        accepted_lambdas=[],
        failed_attempts=0,
        accepted_steps=0,
        fallback_used=False,
        failure_reason="",
    )


class HarmonicNewtonKrylovSolver:
    def __init__(self, settings: NewtonKrylovSettings):
        self.settings = settings

    def solve_one(
        self,
        problem: FullPumpProblem,
        X0: np.ndarray,
        source_scale: float,
    ) -> tuple[np.ndarray, StepReport]:
        s = self.settings
        t0 = time.perf_counter()
        X = np.array(X0, dtype=np.complex128, copy=True)

        nrm = problem.norms(X, source_scale, s.compute_time_residual)

        if s.verbose:
            msg = (
                f"  init: lambda={source_scale:.6f} "
                f"coeff_rel={nrm['coeff_rel']:.3e}"
            )
            if nrm["time_rel"] is not None:
                msg += f" time_rel={nrm['time_rel']:.3e}"
            print(msg)

        if nrm["coeff_rel"] < s.newton_tol:
            return X, self._make_report(
                True, source_scale, nrm, 0, 0, 0.0, t0, ""
            )

        shape = X.shape
        dim_real = 2 * shape[0] * shape[1]
        gmres_total = 0
        factor_total = 0.0
        precond_assembly_total = 0.0
        precond_numeric_factor_total = 0.0
        failure_reason = ""
        stall_count = 0

        # Cached preconditioner factor (modified-Newton reuse, see settings).
        cached_real: spla.SuperLU | None = None
        cached_coupled: spla.SuperLU | None = None
        cached_factors: list[spla.SuperLU] | None = None
        steps_since_factor = 0
        last_gmres = 0

        for it in range(1, s.max_newton + 1):
            if s.solve_deadline_s > 0.0 and (time.perf_counter() - t0) > s.solve_deadline_s:
                failure_reason = f"solve exceeded {s.solve_deadline_s:.1f}s budget at Newton {it}"
                return X, self._make_report(
                    False, source_scale, nrm, it - 1, gmres_total,
                    factor_total, t0, failure_reason,
                    preconditioner_assembly_runtime_s=precond_assembly_total,
                    preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
                )
            pre_coeff = nrm["coeff_rel"]
            R = problem.residual_coeffs(X, source_scale)
            rhs = -pack_complex(R)

            tangent = problem.tangent_state(X)

            # Decide whether to (re)build the preconditioner factor or reuse the
            # cached one from an earlier Newton step (modified-Newton reuse).
            have_cache = (cached_real is not None or cached_coupled is not None
                          or cached_factors is not None)
            refresh = (
                not have_cache
                or s.precond_reuse <= 1
                or steps_since_factor >= s.precond_reuse
                or (s.precond_reuse_refresh_gmres > 0
                    and last_gmres >= s.precond_reuse_refresh_gmres)
            )

            spectral_tangent = None
            if refresh and (s.jvp_mode == "spectral"
                            or s.preconditioner in ("spectral_coupled", "real_coupled")):
                spectral_tangent = problem.spectral_tangent_state(tangent)
            elif s.jvp_mode == "spectral":
                spectral_tangent = problem.spectral_tangent_state(tangent)

            tf = time.perf_counter()
            if refresh:
                if s.preconditioner == "real_coupled":
                    cached_real = problem.assemble_real_coupled_preconditioner(spectral_tangent)
                    cached_coupled = cached_factors = None
                elif s.preconditioner == "real_coupled_fast":
                    # Exact real-coupled with assembly + symbolic-factorization
                    # reuse. Implemented ONLY for the Schur-reduced problem
                    # (exp10 in-process backend); the direct exp08 solver runs
                    # the full problem, which has no such method.
                    if not hasattr(problem, "assemble_real_coupled_fast"):
                        raise NotImplementedError(
                            "real_coupled_fast is available only for the "
                            "Schur-reduced backend (exp10 --inproc-preconditioner "
                            "real_coupled_fast). For the direct exp08 solver use "
                            "--preconditioner real_coupled."
                        )
                    cached_real = problem.assemble_real_coupled_fast(tangent)
                    cached_coupled = cached_factors = None
                elif s.preconditioner == "spectral_coupled":
                    cached_coupled = problem.assemble_coupled_preconditioner(spectral_tangent)
                    cached_real = cached_factors = None
                else:
                    cached_factors = problem.build_preconditioner_factors(
                        X, s.preconditioner, tangent=tangent)
                    cached_real = cached_coupled = None
                steps_since_factor = 0
            else:
                steps_since_factor += 1
            real_factor = cached_real
            coupled_factor = cached_coupled
            factors = cached_factors
            factor_s = time.perf_counter() - tf
            factor_total += factor_s
            if refresh and s.preconditioner == "real_coupled_fast" and cached_real is not None:
                precond_assembly_total += float(getattr(cached_real, "last_assembly_runtime_s", 0.0))
                precond_numeric_factor_total += float(getattr(cached_real, "last_factor_runtime_s", 0.0))

            def matvec(v_real: np.ndarray) -> np.ndarray:
                V = unpack_complex(v_real, shape)
                JV = problem.jvp_coeffs_with_tangent(V, tangent)
                return pack_complex(JV)

            Aop = spla.LinearOperator(
                shape=(dim_real, dim_real),
                matvec=matvec,
                dtype=np.float64,
            )

            Mop = None
            if real_factor is not None:
                def psolve_real(v_real: np.ndarray) -> np.ndarray:
                    return real_factor.solve(v_real)

                Mop = spla.LinearOperator(
                    shape=(dim_real, dim_real),
                    matvec=psolve_real,
                    dtype=np.float64,
                )
            elif coupled_factor is not None:
                def psolve_coupled(v_real: np.ndarray) -> np.ndarray:
                    V = unpack_complex(v_real, shape)
                    z = coupled_factor.solve(V.reshape(-1))
                    return pack_complex(z.reshape(shape))

                Mop = spla.LinearOperator(
                    shape=(dim_real, dim_real),
                    matvec=psolve_coupled,
                    dtype=np.float64,
                )
            elif factors is not None:
                def psolve(v_real: np.ndarray) -> np.ndarray:
                    V = unpack_complex(v_real, shape)
                    Z = np.empty_like(V)
                    for h in range(problem.H):
                        Z[h] = factors[h].solve(V[h])
                    return pack_complex(Z)

                Mop = spla.LinearOperator(
                    shape=(dim_real, dim_real),
                    matvec=psolve,
                    dtype=np.float64,
                )

            gmres_counter = {"n": 0}

            def cb(_pr_norm: float) -> None:
                gmres_counter["n"] += 1

            delta_real, info = gmres_call(
                A=Aop,
                b=rhs,
                M=Mop,
                rtol=s.gmres_rtol,
                atol=s.gmres_atol,
                restart=s.gmres_restart,
                maxiter=s.gmres_maxiter,
                callback=cb,
            )
            gmres_total += gmres_counter["n"]
            last_gmres = gmres_counter["n"]

            if info != 0:
                failure_reason = f"GMRES did not fully converge, info={info}"

            Delta = unpack_complex(delta_real, shape)

            accepted = False
            alpha = 1.0
            best_X = X
            best_nrm = nrm

            while alpha >= s.min_alpha:
                Xtrial = X + alpha * Delta
                trial_nrm = problem.norms(
                    Xtrial,
                    source_scale,
                    s.compute_time_residual,
                )

                if trial_nrm["coeff_rel"] < nrm["coeff_rel"]:
                    accepted = True
                    best_X = Xtrial
                    best_nrm = trial_nrm
                    break

                alpha *= 0.5

            if not accepted:
                failure_reason = failure_reason or f"line search failed at Newton {it}"
                return X, self._make_report(
                    False,
                    source_scale,
                    nrm,
                    it,
                    gmres_total,
                    factor_total,
                    t0,
                    failure_reason,
                    preconditioner_assembly_runtime_s=precond_assembly_total,
                    preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
                )

            X = best_X
            nrm = best_nrm

            # Stall detection: if accepted steps keep barely reducing the
            # residual while still far from tolerance, the operating point is
            # over the fold / unsolvable. Bail instead of grinding to max_newton.
            if s.stall_patience > 0 and nrm["coeff_rel"] > 100.0 * s.newton_tol:
                ratio = nrm["coeff_rel"] / max(pre_coeff, 1e-300)
                stall_count = stall_count + 1 if ratio > s.stall_ratio else 0
                if stall_count >= s.stall_patience:
                    failure_reason = (
                        f"stalled at Newton {it} (reduction ratio {ratio:.3f})"
                    )
                    if s.verbose:
                        print(f"  newton={it:02d} STALL ratio={ratio:.3f} coeff_rel={nrm['coeff_rel']:.3e}")
                    return X, self._make_report(
                        False, source_scale, nrm, it, gmres_total,
                        factor_total, t0, failure_reason,
                        preconditioner_assembly_runtime_s=precond_assembly_total,
                        preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
                    )

            if s.verbose:
                msg = (
                    f"  newton={it:02d} alpha={alpha:.3e} "
                    f"gmres_it={gmres_counter['n']:04d} "
                    f"factor_s={factor_s:.3f} "
                    f"coeff_rel={nrm['coeff_rel']:.3e}"
                )
                if nrm["time_rel"] is not None:
                    msg += f" time_rel={nrm['time_rel']:.3e}"
                if info != 0:
                    msg += f" gmres_info={info}"
                print(msg)

            if nrm["coeff_rel"] < s.newton_tol:
                return X, self._make_report(
                    True,
                    source_scale,
                    nrm,
                    it,
                    gmres_total,
                    factor_total,
                    t0,
                    "",
                    preconditioner_assembly_runtime_s=precond_assembly_total,
                    preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
                )

        failure_reason = failure_reason or "maximum Newton iterations reached"
        return X, self._make_report(
            False,
            source_scale,
            nrm,
            s.max_newton,
            gmres_total,
            factor_total,
            t0,
            failure_reason,
            preconditioner_assembly_runtime_s=precond_assembly_total,
            preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
        )

    def solve_direct(
        self,
        problem: FullPumpProblem,
        x_init: np.ndarray,
    ) -> tuple[np.ndarray, list[StepReport]]:
        """Single Newton-Krylov solve at full pump scale from a warm start."""
        print("=== warm-start direct solve at lambda=1.0 ===")
        X_new, report = self.solve_one(problem, x_init, 1.0)
        status = "VALID_CONVERGED" if report.converged else "FAIL"
        print(
            f"step_status={status} "
            f"coeff_rel={report.coeff_rel:.3e} "
            f"newton={report.newton_iterations} "
            f"gmres_total={report.gmres_iterations_total} "
            f"runtime_s={report.runtime_s:.3f} "
            f"reason={report.failure_reason}"
        )
        return X_new, [report]

    def solve_continuation(
        self,
        problem: FullPumpProblem,
        continuation_steps: int,
        x_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[StepReport]]:
        reports: list[StepReport] = []

        lambdas = np.linspace(1.0 / continuation_steps, 1.0, continuation_steps)

        X_prevprev: np.ndarray | None = None
        X_prev = problem.zeros() if x_init is None else np.array(x_init, dtype=np.complex128, copy=True)
        lam_prevprev: float | None = None
        lam_prev: float | None = None

        for lam_raw in lambdas:
            lam = float(lam_raw)

            X_guess = X_prev

            if (
                self.settings.continuation_predictor == "secant"
                and X_prevprev is not None
                and lam_prev is not None
                and lam_prevprev is not None
                and abs(lam_prev - lam_prevprev) > 0.0
            ):
                beta = (lam - lam_prev) / (lam_prev - lam_prevprev)
                X_guess = X_prev + beta * (X_prev - X_prevprev)

            print(f"=== continuation lambda={lam:.6f} ===")
            X_new, report = self.solve_one(problem, X_guess, lam)
            reports.append(report)

            status = "VALID_CONVERGED" if report.converged else "FAIL"
            msg = (
                f"step_status={status} "
                f"coeff_rel={report.coeff_rel:.3e} "
                f"newton={report.newton_iterations} "
                f"gmres_total={report.gmres_iterations_total} "
                f"factor_s={report.factor_runtime_s:.3f} "
                f"runtime_s={report.runtime_s:.3f} "
                f"reason={report.failure_reason}"
            )
            if report.time_rel is not None:
                msg = msg.replace(
                    f"newton={report.newton_iterations}",
                    f"time_rel={report.time_rel:.3e} newton={report.newton_iterations}",
                )
            print(msg)

            if not report.converged:
                return X_new, reports

            X_prevprev = X_prev
            lam_prevprev = lam_prev
            X_prev = X_new
            lam_prev = lam

        return X_prev, reports

    def solve_adaptive_continuation(
        self,
        problem: FullPumpProblem,
        x_init: np.ndarray | None,
        *,
        initial_step: float,
        min_step: float,
        growth: float,
        shrink: float,
        fallback_fixed_steps: int,
    ) -> tuple[np.ndarray, list[StepReport], ContinuationTrace]:
        reports: list[StepReport] = []
        trace = empty_continuation_trace("adaptive")

        if initial_step <= 0.0 or initial_step > 1.0:
            raise ValueError("--adaptive-initial-step must be in (0, 1]")
        if min_step <= 0.0 or min_step > 1.0:
            raise ValueError("--adaptive-min-step must be in (0, 1]")
        if growth < 1.0:
            raise ValueError("--adaptive-growth must be >= 1")
        if shrink <= 0.0 or shrink >= 1.0:
            raise ValueError("--adaptive-shrink must be in (0, 1)")

        X_current = problem.zeros() if x_init is None else np.array(x_init, dtype=np.complex128, copy=True)
        lambda_current = 0.0
        step = float(initial_step)

        X_prevprev: np.ndarray | None = None
        X_prev = X_current
        lam_prevprev: float | None = None
        lam_prev: float | None = 0.0

        while lambda_current < 1.0 - 1e-12:
            step = min(step, 1.0 - lambda_current)
            target = float(min(1.0, lambda_current + step))
            trace.attempted_lambdas.append(target)

            X_guess = X_current
            if (
                self.settings.continuation_predictor == "secant"
                and X_prevprev is not None
                and lam_prev is not None
                and lam_prevprev is not None
                and abs(lam_prev - lam_prevprev) > 0.0
            ):
                beta = (target - lam_prev) / (lam_prev - lam_prevprev)
                X_guess = X_prev + beta * (X_prev - X_prevprev)

            print(f"=== adaptive continuation lambda={target:.6f} step={step:.6f} ===")
            X_new, report = self.solve_one(problem, X_guess, target)
            reports.append(report)

            status = "VALID_CONVERGED" if report.converged else "FAIL"
            msg = (
                f"step_status={status} "
                f"coeff_rel={report.coeff_rel:.3e} "
                f"newton={report.newton_iterations} "
                f"gmres_total={report.gmres_iterations_total} "
                f"factor_s={report.factor_runtime_s:.3f} "
                f"runtime_s={report.runtime_s:.3f} "
                f"reason={report.failure_reason}"
            )
            if report.time_rel is not None:
                msg = msg.replace(
                    f"newton={report.newton_iterations}",
                    f"time_rel={report.time_rel:.3e} newton={report.newton_iterations}",
                )
            print(msg)

            if report.converged:
                X_prevprev = X_current
                lam_prevprev = lambda_current
                X_current = X_new
                X_prev = X_new
                lambda_current = target
                lam_prev = target
                trace.accepted_lambdas.append(target)
                step = min(1.0, max(min_step, step * growth))
                continue

            trace.failed_attempts += 1
            trace.failure_reason = report.failure_reason
            next_step = step * shrink
            if next_step < min_step:
                trace.fallback_used = True
                print(
                    "adaptive continuation failed below minimum step; "
                    f"falling back to fixed {fallback_fixed_steps} steps"
                )
                X_fallback, fallback_reports = self.solve_continuation(
                    problem,
                    continuation_steps=fallback_fixed_steps,
                    x_init=x_init,
                )
                reports.extend(fallback_reports)
                trace.accepted_steps = len(trace.accepted_lambdas)
                return X_fallback, reports, trace

            step = next_step

        trace.accepted_steps = len(trace.accepted_lambdas)
        return X_current, reports, trace

    @staticmethod
    def _make_report(
        converged: bool,
        source_scale: float,
        nrm: dict[str, float | None],
        newton_iterations: int,
        gmres_iterations_total: int,
        factor_runtime_s: float,
        t0: float,
        failure_reason: str,
        *,
        preconditioner_assembly_runtime_s: float = 0.0,
        preconditioner_numeric_factor_runtime_s: float = 0.0,
    ) -> StepReport:
        return StepReport(
            converged=converged,
            source_scale=source_scale,
            coeff_abs=float(nrm["coeff_abs"]),
            coeff_rel=float(nrm["coeff_rel"]),
            time_abs=None if nrm["time_abs"] is None else float(nrm["time_abs"]),
            time_rel=None if nrm["time_rel"] is None else float(nrm["time_rel"]),
            newton_iterations=newton_iterations,
            gmres_iterations_total=gmres_iterations_total,
            factor_runtime_s=factor_runtime_s,
            runtime_s=time.perf_counter() - t0,
            failure_reason=failure_reason,
            preconditioner_assembly_runtime_s=float(preconditioner_assembly_runtime_s),
            preconditioner_numeric_factor_runtime_s=float(preconditioner_numeric_factor_runtime_s),
        )


def gmres_call(
    A: spla.LinearOperator,
    b: np.ndarray,
    M: spla.LinearOperator | None,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    callback,
) -> tuple[np.ndarray, int]:
    try:
        return spla.gmres(
            A,
            b,
            M=M,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            callback=callback,
            callback_type="pr_norm",
        )
    except TypeError:
        return spla.gmres(
            A,
            b,
            M=M,
            tol=rtol,
            restart=restart,
            maxiter=maxiter,
            callback=callback,
        )
