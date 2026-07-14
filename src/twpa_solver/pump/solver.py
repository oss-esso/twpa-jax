from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla


def _real_dot(A: np.ndarray, B: np.ndarray) -> float:
    """Real Euclidean inner product on complex coefficient arrays.

    Uses ``Re(sum(conj(A) * B))`` = the dot product of the real-packed vectors,
    so arclength tangent/constraint algebra stays in real arithmetic.
    """
    return float(np.real(np.vdot(A, B)))


def _finite_state(X: np.ndarray) -> bool:
    """Return whether a predictor is safe to pass into sparse linear algebra."""
    return bool(np.all(np.isfinite(X)))


def _finite_residual(norms: dict[str, float | None]) -> bool:
    """Check the coefficient residual before constructing a preconditioner."""
    value = norms.get("coeff_rel")
    return value is not None and math.isfinite(float(value))

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

        # A bad secant/tangent predictor can overflow the Josephson evaluation.
        # Let the map record a controlled pump failure instead of handing NaN or
        # Inf values to SuperLU/PARDISO, which may terminate the native process.
        if not _finite_state(X) or not _finite_residual(nrm):
            return X, self._make_report(
                False, source_scale, nrm, 0, 0, 0.0, t0,
                "non-finite initial state or residual",
            )

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
            if not _finite_state(R):
                return X, self._make_report(
                    False, source_scale, nrm, it - 1, gmres_total,
                    factor_total, t0,
                    "non-finite residual before linear solve",
                    preconditioner_assembly_runtime_s=precond_assembly_total,
                    preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
                )
            rhs = -pack_complex(R)

            tangent = problem.tangent_state(X)
            if not _finite_state(tangent.gamma_t) or not _finite_state(tangent.gamma_mean):
                return X, self._make_report(
                    False, source_scale, nrm, it - 1, gmres_total,
                    factor_total, t0,
                    "non-finite tangent before linear solve",
                    preconditioner_assembly_runtime_s=precond_assembly_total,
                    preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
                )

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
        controller: str = "fixed",
        max_wall_s: float = 0.0,
    ) -> tuple[np.ndarray, list[StepReport], ContinuationTrace]:
        reports: list[StepReport] = []
        trace = empty_continuation_trace("adaptive")
        continuation_t0 = time.perf_counter()

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
            if max_wall_s > 0.0 and time.perf_counter() - continuation_t0 > max_wall_s:
                trace.failure_reason = (
                    f"continuation exceeded {max_wall_s:.1f}s budget"
                )
                trace.fallback_used = True
                return X_current, reports, trace
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
            elif (
                self.settings.continuation_predictor == "tangent"
                and lam_prev is not None
                and lambda_current > 0.0
            ):
                # Exact lambda-tangent Euler predictor from the current point.
                X_guess = self.tangent_predictor(problem, X_current, target - lambda_current)

            if X_guess is not X_current:
                # Reject predictors that are finite but grossly worse than the
                # copy predictor. This is especially important after a failed
                # corrector near a fold: the next secant/tangent solve can
                # otherwise produce a pathological sparse factorization input.
                use_copy = not _finite_state(X_guess)
                if not use_copy:
                    try:
                        copy_rel = problem.norms(X_current, target, False)["coeff_rel"]
                        guess_rel = problem.norms(X_guess, target, False)["coeff_rel"]
                        use_copy = (
                            not math.isfinite(float(guess_rel))
                            or (
                                math.isfinite(float(copy_rel))
                                and float(guess_rel) > max(100.0 * float(copy_rel), 1e6)
                            )
                        )
                    except (FloatingPointError, ValueError, OverflowError):
                        use_copy = True
                if use_copy:
                    X_guess = X_current
                    if self.settings.verbose:
                        print("  predictor rejected; using copy predictor")

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
                if controller == "newton_contraction":
                    target_iterations = 4.0
                    observed = max(1.0, float(report.newton_iterations))
                    factor = math.sqrt(target_iterations / observed)
                    factor = min(2.0, max(0.5, factor))
                    step = min(1.0, max(min_step, step * factor))
                else:
                    step = min(1.0, max(min_step, step * growth))
                continue

            trace.failed_attempts += 1
            trace.failure_reason = report.failure_reason
            next_step = step * shrink
            if next_step < min_step:
                trace.fallback_used = True
                if (
                    max_wall_s > 0.0
                    and time.perf_counter() - continuation_t0 > max_wall_s
                ):
                    trace.failure_reason = (
                        f"continuation exceeded {max_wall_s:.1f}s budget"
                    )
                    trace.accepted_steps = len(trace.accepted_lambdas)
                    return X_current, reports, trace
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

    def solve_affine_continuation(
        self,
        problem: FullPumpProblem,
        x_init: np.ndarray | None,
        *,
        initial_step: float,
        min_step: float,
        fallback_fixed_steps: int,
        max_wall_s: float = 0.0,
    ) -> tuple[np.ndarray, list[StepReport], ContinuationTrace]:
        """Adaptive continuation controlled by observed corrector contraction.

        Newton iteration count is used as the available affine-invariant
        contraction monitor: the next step targets four corrector iterations,
        with the scale change bounded to [0.5, 2.0]. Failed correctors retain
        the established half-step recovery and deterministic fixed fallback.
        """
        return self.solve_adaptive_continuation(
            problem,
            x_init,
            initial_step=initial_step,
            min_step=min_step,
            growth=1.5,
            shrink=0.5,
            fallback_fixed_steps=fallback_fixed_steps,
            controller="newton_contraction",
            max_wall_s=max_wall_s,
        )

    # -----------------------------------------------------------------
    # Advanced continuation (report methods): tangent predictor, affine
    # step control, pseudo-transient, pseudo-arclength + fold locator.
    # These reuse solve_one's matvec/preconditioner machinery through the
    # matrix-free _solve_linear helper below.
    # -----------------------------------------------------------------

    def _linear_solver(
        self,
        problem: FullPumpProblem,
        X: np.ndarray,
        *,
        shift: float = 0.0,
    ):
        """Build a reusable ``J(X)+shift*I`` GMRES solver at the point ``X``.

        Returns a closure ``solve(rhs_coeffs) -> delta`` sharing one tangent
        state and one mean-tangent preconditioner factorization across multiple
        right-hand sides. Reusing this is the key cost saver in the arclength
        corrector, which solves two systems (``J a = -R`` and ``J b = S``) per
        Newton iteration with the same operator.
        """
        s = self.settings
        shape = X.shape
        dim_real = 2 * shape[0] * shape[1]
        tangent = problem.tangent_state(X)
        # Use the same cached exact real-coupled factor as ordinary Newton when
        # the Schur backend provides it. The old path built a fresh SuperLU
        # factor for every tangent call, which made recovery at the fold much
        # more memory-intensive than the normal solve.
        if hasattr(problem, "assemble_real_coupled_fast"):
            real_factor = problem.assemble_real_coupled_fast(tangent)
        else:
            spectral = problem.spectral_tangent_state(tangent)
            real_factor = problem.assemble_real_coupled_preconditioner(spectral)

        def matvec(v_real: np.ndarray) -> np.ndarray:
            V = unpack_complex(v_real, shape)
            JV = problem.jvp_coeffs_with_tangent(V, tangent)
            if shift != 0.0:
                JV = JV + shift * V
            return pack_complex(JV)

        Aop = spla.LinearOperator((dim_real, dim_real), matvec=matvec, dtype=np.float64)

        def psolve(v_real: np.ndarray) -> np.ndarray:
            return real_factor.solve(v_real)

        Mop = spla.LinearOperator((dim_real, dim_real), matvec=psolve, dtype=np.float64)

        def solve(rhs_coeffs: np.ndarray) -> np.ndarray:
            delta_real, _info = gmres_call(
                A=Aop, b=pack_complex(rhs_coeffs), M=Mop,
                rtol=s.gmres_rtol, atol=s.gmres_atol, restart=s.gmres_restart,
                maxiter=s.gmres_maxiter, callback=lambda _n: None,
            )
            return unpack_complex(delta_real, shape)

        return solve

    def _solve_linear(
        self,
        problem: FullPumpProblem,
        X: np.ndarray,
        rhs_coeffs: np.ndarray,
        *,
        shift: float = 0.0,
    ) -> np.ndarray:
        """Solve ``(J(X)+shift*I) delta = rhs_coeffs`` matrix-free by GMRES."""
        return self._linear_solver(problem, X, shift=shift)(rhs_coeffs)

    def tangent_predictor(
        self, problem: FullPumpProblem, X: np.ndarray, d_lambda: float,
    ) -> np.ndarray:
        """Euler predictor along the branch: ``X + d_lambda * dX/dlambda``.

        ``dR/dlambda = -S`` (the source is linear in lambda), so the branch
        tangent solves ``J * Xdot = S`` where ``S = source_coeffs(1)``. More
        robust than the secant when the previous two points are far apart or the
        branch curves sharply (near the fold).
        """
        S = problem.source_coeffs(1.0)
        if not _finite_state(X):
            return np.array(X, copy=True)
        try:
            tangent = problem.tangent_state(X)
            if not _finite_state(tangent.gamma_t) or not _finite_state(tangent.gamma_mean):
                return np.array(X, copy=True)
            Xdot = self._solve_linear(problem, X, S)
        except (FloatingPointError, RuntimeError, ValueError, OverflowError):
            return np.array(X, copy=True)
        if not _finite_state(Xdot):
            return np.array(X, copy=True)
        return X + d_lambda * Xdot

    def solve_pseudo_transient(
        self,
        problem: FullPumpProblem,
        X0: np.ndarray | None,
        *,
        delta0: float = 1.0,
        max_iter: int | None = None,
    ) -> tuple[np.ndarray, list[StepReport]]:
        """Pseudo-transient continuation (Psi-tc) to full drive (lambda=1).

        Embeds the root-find in a fake time march
        ``(X^{n+1}-X^n)/delta + R(X^{n+1}) = 0`` and grows the pseudo-timestep
        ``delta`` by the SER rule as the residual falls, so early steps are
        globally stable and late steps recover Newton's rate. Degrades
        gracefully near the fold (slows rather than diverges).
        """
        s = self.settings
        t0 = time.perf_counter()
        X = problem.zeros() if X0 is None else np.array(X0, dtype=np.complex128, copy=True)
        delta = float(delta0)
        max_it = int(max_iter if max_iter is not None else s.max_newton * 4)
        nrm = problem.norms(X, 1.0, s.compute_time_residual)
        prev_res = nrm["coeff_rel"]
        gmres_total = 0
        for it in range(1, max_it + 1):
            if (
                s.solve_deadline_s > 0.0
                and time.perf_counter() - t0 > s.solve_deadline_s
            ):
                return X, [self._make_report(
                    False,
                    1.0,
                    nrm,
                    it - 1,
                    gmres_total,
                    0.0,
                    t0,
                    f"ptc exceeded {s.solve_deadline_s:.1f}s budget",
                )]
            if nrm["coeff_rel"] < s.newton_tol:
                return X, [self._make_report(True, 1.0, nrm, it - 1, gmres_total, 0.0, t0, "")]
            R = problem.residual_coeffs(X, 1.0)
            # Modified Newton step with the 1/delta pseudo-time shift.
            delta_X = self._solve_linear(problem, X, -R, shift=1.0 / delta)
            # Damped acceptance on the true residual.
            alpha = 1.0
            accepted = False
            while alpha >= s.min_alpha:
                Xtrial = X + alpha * delta_X
                trial = problem.norms(Xtrial, 1.0, s.compute_time_residual)
                if trial["coeff_rel"] < nrm["coeff_rel"]:
                    X, nrm, accepted = Xtrial, trial, True
                    break
                alpha *= 0.5
            if not accepted:
                return X, [self._make_report(
                    False, 1.0, nrm, it, gmres_total, 0.0, t0,
                    f"ptc line search failed at iter {it}")]
            # SER: grow the pseudo-timestep inversely with the residual drop.
            ratio = prev_res / max(nrm["coeff_rel"], 1e-300)
            delta = max(delta, delta * ratio)
            prev_res = nrm["coeff_rel"]
        return X, [self._make_report(
            False, 1.0, nrm, max_it, gmres_total, 0.0, t0, "ptc max iterations reached")]

    def solve_arclength(
        self,
        problem: FullPumpProblem,
        X0: np.ndarray | None,
        lam0: float,
        *,
        ds: float = 0.1,
        max_steps: int = 200,
        target_lam: float = 1.0,
        newton_max: int = 12,
        max_wall_s: float = 0.0,
    ) -> tuple[np.ndarray, float, dict]:
        """Pseudo-arclength continuation in the source scale ``lambda``.

        Treats ``(X, lambda)`` as joint unknowns and advances a pseudo-arclength
        ``s`` with the tangent-plane constraint, so the augmented system stays
        non-singular through the fold (Keller 1977). Each corrector Newton step
        uses the bordering algorithm: two solves with ``J`` (``J a = -R`` and
        ``J b = S``) combined by the scalar constraint.

        Returns ``(X, lambda, info)`` where ``info`` records whether ``target_lam``
        was reached and the fold ``lambda`` if a turning point (sign change of
        ``lambda_dot``) was crossed first.
        """
        X = problem.zeros() if X0 is None else np.array(X0, dtype=np.complex128, copy=True)
        lam = float(lam0)
        S = problem.source_coeffs(1.0)
        t0 = time.perf_counter()
        info: dict = {
            "reached_target": False,
            "fold_lambda": None,
            "steps": 0,
            "terminal_reason": "max_steps",
        }

        # Initial tangent: J Xdot = S, then normalise (Xdot, lam_dot).
        Xdot = self._solve_linear(problem, X, S)
        lam_dot = 1.0
        norm = math.sqrt(_real_dot(Xdot, Xdot) + lam_dot * lam_dot)
        Xdot, lam_dot = Xdot / norm, lam_dot / norm

        tol = max(self.settings.newton_tol * 100.0, 1e-7)  # relative coeff tol
        for step in range(1, max_steps + 1):
            info["steps"] = step
            if max_wall_s > 0.0 and time.perf_counter() - t0 > max_wall_s:
                info["terminal_reason"] = "deadline"
                return X, lam, info
            X_pred = X + ds * Xdot
            lam_pred = lam + ds * lam_dot
            Xc, lamc = np.array(X_pred, copy=True), float(lam_pred)
            # Modified Newton: one factorization at the predictor point, reused
            # across the inner corrector iterations (the coupled factor barely
            # changes over a corrector and is the dominant cost).
            lin = self._linear_solver(problem, X_pred)
            b = lin(S)  # J b = S (dX/dlam) -- constant RHS, factor once
            converged = False
            for _ in range(newton_max):
                R = problem.residual_coeffs(Xc, lamc)
                n = _real_dot(Xdot, Xc - X) + lam_dot * (lamc - lam) - ds
                if problem.norms(Xc, lamc, False)["coeff_rel"] < tol and abs(n) < tol * max(ds, 1.0):
                    converged = True
                    break
                a = lin(-R)  # J a = -R (modified Newton: reuse predictor factor)
                denom = _real_dot(Xdot, b) + lam_dot
                if abs(denom) < 1e-300:
                    break
                d_lam = (-n - _real_dot(Xdot, a)) / denom
                d_X = a + d_lam * b
                Xc = Xc + d_X
                lamc = lamc + d_lam
            if not converged:
                ds *= 0.5
                if ds < 1e-4:
                    info["terminal_reason"] = "minimum_step"
                    return X, lam, info
                continue
            # New tangent (keep continuation direction via sign).
            Xdot_new = lin(S)
            lam_dot_new = 1.0
            nrm = math.sqrt(_real_dot(Xdot_new, Xdot_new) + lam_dot_new ** 2)
            Xdot_new, lam_dot_new = Xdot_new / nrm, lam_dot_new / nrm
            if _real_dot(Xdot_new, Xdot) + lam_dot_new * lam_dot < 0.0:
                Xdot_new, lam_dot_new = -Xdot_new, -lam_dot_new
            # Fold = sign change of lam_dot.
            if lam_dot_new * lam_dot < 0.0 and info["fold_lambda"] is None:
                info["fold_lambda"] = float(lamc)
            # Target crossing (interpolate to target_lam).
            if (lam - target_lam) * (lamc - target_lam) <= 0.0 and lamc != lam:
                theta = (target_lam - lam) / (lamc - lam)
                X = X + theta * (Xc - X)
                info["reached_target"] = True
                info["terminal_reason"] = "target"
                return X, target_lam, info
            X, lam, Xdot, lam_dot = Xc, lamc, Xdot_new, lam_dot_new
        return X, lam, info

    def trace_arclength_from_two_points(
        self,
        problem: FullPumpProblem,
        X0: np.ndarray,
        lam0: float,
        X1: np.ndarray,
        lam1: float,
        *,
        ds: float = 0.02,
        max_steps: int = 160,
        newton_max: int = 10,
    ) -> tuple[list[tuple[np.ndarray, float]], dict]:
        """Trace a branch from two converged points with a scaled secant metric.

        Pump coefficients and source scale have very different physical units.
        The state scale inferred from the initial secant makes both components
        contribute comparably to the arclength constraint. This is intended for
        map recovery, where two adjacent power solutions are already available.
        """
        X_prev = np.array(X0, dtype=np.complex128, copy=True)
        X = np.array(X1, dtype=np.complex128, copy=True)
        lam_prev, lam = float(lam0), float(lam1)
        dX0 = X - X_prev
        dlam0 = lam - lam_prev
        if abs(dlam0) < 1e-14:
            raise ValueError("arclength seed points must have distinct source scales")
        state_scale = math.sqrt(_real_dot(dX0, dX0)) / abs(dlam0)
        state_scale = max(state_scale, math.sqrt(_real_dot(X, X)) * 1e-6, 1e-300)

        def metric_x(a: np.ndarray, b: np.ndarray) -> float:
            return _real_dot(a, b) / (state_scale * state_scale)

        def normalized_tangent(
            Xa: np.ndarray, la: float, Xb: np.ndarray, lb: float,
        ) -> tuple[np.ndarray, float]:
            tx, tl = Xb - Xa, lb - la
            nrm = math.sqrt(metric_x(tx, tx) + tl * tl)
            return tx / nrm, tl / nrm

        tx, tl = normalized_tangent(X_prev, lam_prev, X, lam)
        points = [(X_prev, lam_prev), (X, lam)]
        info = {
            "steps": 0,
            "fold_lambdas": [],
            "failed_steps": 0,
            "state_scale": state_scale,
            "terminal_reason": "max_steps",
        }
        step_size = float(ds)
        min_ds = max(1e-5, float(ds) / 128.0)
        tol = max(self.settings.newton_tol * 10.0, 1e-8)
        S = problem.source_coeffs(1.0)

        for step in range(1, int(max_steps) + 1):
            info["steps"] = step
            X_pred = X + step_size * tx
            lam_pred = lam + step_size * tl
            Xc, lamc = np.array(X_pred, copy=True), float(lam_pred)
            converged = False
            used_newton = newton_max
            for it in range(1, int(newton_max) + 1):
                R = problem.residual_coeffs(Xc, lamc)
                constraint = (
                    metric_x(tx, Xc - X)
                    + tl * (lamc - lam)
                    - step_size
                )
                rel = problem.norms(Xc, lamc, False)["coeff_rel"]
                if rel < tol and abs(constraint) < tol:
                    converged, used_newton = True, it - 1
                    break
                lin = self._linear_solver(problem, Xc)
                a = lin(-R)
                b = lin(S)
                denom = metric_x(tx, b) + tl
                if not math.isfinite(denom) or abs(denom) < 1e-14:
                    break
                dlam = (-constraint - metric_x(tx, a)) / denom
                Xc = Xc + a + dlam * b
                lamc = lamc + dlam
            if not converged:
                info["failed_steps"] += 1
                step_size *= 0.5
                if step_size < min_ds:
                    info["terminal_reason"] = "minimum_step"
                    break
                continue

            tx_new, tl_new = normalized_tangent(X, lam, Xc, lamc)
            if metric_x(tx_new, tx) + tl_new * tl < 0.0:
                tx_new, tl_new = -tx_new, -tl_new
            if tl_new * tl < 0.0:
                info["fold_lambdas"].append(float(lamc))
            X_prev, lam_prev = X, lam
            X, lam = Xc, float(lamc)
            tx, tl = tx_new, float(tl_new)
            points.append((X, lam))
            if used_newton <= 3:
                step_size = min(float(ds), step_size * 1.25)

        return points, info

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


def fold_power(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    *,
    ds: float = 0.1,
    max_steps: int = 300,
) -> float | None:
    """Locate the harmonic-balance fold in the source scale ``lambda``.

    ``problem`` is built at a reference pump current; ``lambda`` scales that
    current linearly, so the fold ``lambda`` maps directly to a fold current
    (``lambda_fold * reference_current``) and thus a fold power. Runs
    pseudo-arclength from ``lambda=0`` and returns the ``lambda`` at the first
    turning point (sign change of ``lambda_dot``), or ``None`` if none is found
    within ``max_steps`` (branch has no fold in range).
    """
    _X, _lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=ds, max_steps=max_steps,
        target_lam=float("inf"),  # never "reach" target -> run until fold/steps
    )
    fold = info.get("fold_lambda")
    return float(fold) if fold is not None else None
