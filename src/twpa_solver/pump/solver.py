from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


def _real_dot(A: np.ndarray, B: np.ndarray) -> float:
    """Real Euclidean inner product on complex coefficient arrays.

    Uses ``Re(sum(conj(A) * B))`` = the dot product of the real-packed vectors,
    so arclength tangent/constraint algebra stays in real arithmetic.
    """
    return float(np.real(np.vdot(A, B)))


def _bordered_block_step(
    linsolve: Callable[[np.ndarray], np.ndarray],
    rhs_a: np.ndarray,
    target: float,
    S: np.ndarray,
    c_dot: Callable[[np.ndarray], float],
    lam_dot: float,
    b: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """One block-elimination pass of the arclength bordered system.

    Solves ``J d_X = rhs_a + d_lam * S`` combined with the scalar constraint
    ``c_dot(d_X) + lam_dot * d_lam = target`` via Keller's bordering
    algorithm: two applications of ``linsolve`` (an approximate ``J^-1``)
    combined through the scalar constraint. Returns ``None`` if the scalar
    denominator degenerates. ``J b = S`` is constant across both the
    corrector Newton loop and the refinement pass in
    :func:`bordered_solve_refined`; pass it in to skip its solve.
    """
    if b is None:
        b = linsolve(S)
    a = linsolve(rhs_a)
    denom = c_dot(b) + lam_dot
    if not math.isfinite(denom) or abs(denom) < 1e-300:
        return None
    d_lam = (target - c_dot(a)) / denom
    d_X = a + d_lam * b
    return d_X, d_lam, b


def bordered_solve_refined(
    matvec: Callable[[np.ndarray], np.ndarray],
    linsolve: Callable[[np.ndarray], np.ndarray],
    R: np.ndarray,
    n: float,
    S: np.ndarray,
    c_dot: Callable[[np.ndarray], float],
    lam_dot: float,
) -> tuple[np.ndarray, float] | None:
    """Bordered block elimination with one Govaerts-Pryce refinement pass.

    Plain block elimination solves the augmented arclength system through
    two applications of an approximate ``J^-1``; near a fold ``J`` is
    (near-)singular, so that approximation's error is amplified by ``J``'s
    huge condition number, even though the *bordered* system itself stays
    well-posed. One extra pass -- computing the bordered system's own
    residual and eliminating again with the same ``linsolve``/``b`` -- costs
    one more linear solve and restores accuracy through the singularity
    (Govaerts & Pryce, "Block elimination with one refinement solves
    bordered linear systems accurately", BIT 30, 1990). Returns ``None`` if
    the first pass degenerates.
    """
    first = _bordered_block_step(linsolve, -R, -n, S, c_dot, lam_dot)
    if first is None:
        return None
    d_X, d_lam, b = first
    r1 = -R - (matvec(d_X) - S * d_lam)
    r2 = -n - (c_dot(d_X) + lam_dot * d_lam)
    second = _bordered_block_step(linsolve, r1, r2, S, c_dot, lam_dot, b=b)
    if second is None:
        return d_X, d_lam
    dd_X, dd_lam, _b = second
    return d_X + dd_X, d_lam + dd_lam


def _finite_state(X: np.ndarray) -> bool:
    """Return whether a predictor is safe to pass into sparse linear algebra."""
    return bool(np.all(np.isfinite(X)))


def _finite_residual(norms: dict[str, float | None]) -> bool:
    """Check the coefficient residual before constructing a preconditioner."""
    value = norms.get("coeff_rel")
    return value is not None and math.isfinite(float(value))


def _rescale_arclength_tangent(
    Xdot: np.ndarray,
    lam_dot: float,
    new_state_scale: float,
    step_size: float,
) -> tuple[np.ndarray, float, float, float] | None:
    """Change the state metric while preserving the physical predictor.

    ``Xdot`` and ``lam_dot`` are normalized in the old metric.  The new
    metric sees the same physical tangent with norm ``q``; dividing the
    tangent by ``q`` and multiplying ``step_size`` by ``q`` therefore keeps
    ``step_size * (Xdot, lam_dot)`` unchanged.
    """
    if (
        not math.isfinite(new_state_scale)
        or new_state_scale <= 1e-300
        or not math.isfinite(lam_dot)
        or not math.isfinite(step_size)
    ):
        return None
    q_squared = _real_dot(Xdot, Xdot) / (new_state_scale * new_state_scale)
    q_squared += lam_dot * lam_dot
    if not math.isfinite(q_squared) or q_squared <= 1e-600:
        return None
    q = math.sqrt(q_squared)
    if not math.isfinite(q) or q <= 1e-300:
        return None
    return Xdot / q, lam_dot / q, step_size * q, q


def _adaptive_step_size(
    step_size: float,
    used_newton: int,
    theta_k: float,
    *,
    newton_target: int,
    growth_exponent: float,
    growth_clamp: tuple[float, float],
    ds_max: float,
    ds_min: float,
) -> float:
    """Newton-effort + curvature step control (fold_plan.md Sections 10-11).

    ``q_N = (newton_target / max(used_newton, 1)) ** growth_exponent``,
    clamped to ``growth_clamp`` -- an easy corrector grows the step modestly,
    a hard one shrinks it, never by a huge jump in one step. The tangent
    angle ``theta_k`` (radians, between the predicting tangent and the newly
    accepted point's own tangent) then further caps growth as the branch
    curves: a fold naturally produces increasing curvature and therefore a
    shrinking step with no fold-specific special case. Table per plan
    Section 11 (heuristic thresholds, not physics constants).
    """
    q_n = (newton_target / max(used_newton, 1)) ** growth_exponent
    q_n = max(growth_clamp[0], min(growth_clamp[1], q_n))
    ds_trial = step_size * q_n
    theta_deg = math.degrees(theta_k)
    if theta_deg < 5.0:
        pass  # allow full Newton-based growth
    elif theta_deg < 15.0:
        pass  # no special action beyond the Newton-effort factor
    elif theta_deg < 30.0:
        ds_trial = min(ds_trial, step_size)  # prevent growth
    elif theta_deg < 45.0:
        ds_trial = min(ds_trial, step_size * 0.75)  # shrink moderately
    else:
        ds_trial = min(ds_trial, step_size * 0.5)  # shrink aggressively
    return max(ds_min, min(ds_max, ds_trial))


class GmresDeadlineExceeded(RuntimeError):
    """Raised from the GMRES callback when a wall-time budget is exceeded.

    Subclasses RuntimeError so call sites that already catch
    ``(FloatingPointError, RuntimeError, ValueError, OverflowError)`` around a
    linear solve (e.g. ``tangent_predictor``) absorb it without change.
    """

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
        logger.debug(
            "solve_one_entry X0_shape=%s source_scale=%s max_newton=%d newton_tol=%s",
            X0.shape, source_scale, s.max_newton, s.newton_tol,
        )

        nrm = problem.norms(X, source_scale, s.compute_time_residual)
        logger.debug("solve_one_initial_norms coeff_rel=%s coeff_abs=%s", nrm.get("coeff_rel"), nrm.get("coeff_abs"))

        # A bad secant/tangent predictor can overflow the Josephson evaluation.
        # Let the map record a controlled pump failure instead of handing NaN or
        # Inf values to SuperLU/PARDISO, which may terminate the native process.
        if not _finite_state(X) or not _finite_residual(nrm):
            logger.debug(
                "solve_one_fail reason=%s finite_state=%s finite_residual=%s",
                "non-finite initial state or residual", _finite_state(X), _finite_residual(nrm),
            )
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
            logger.debug(
                "solve_one_converged newton_iterations=0 coeff_rel=%s",
                nrm["coeff_rel"],
            )
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
            logger.debug(
                "newton_iter_start it=%d coeff_rel=%s elapsed_s=%.3f",
                it, nrm["coeff_rel"], time.perf_counter() - t0,
            )
            if s.solve_deadline_s > 0.0 and (time.perf_counter() - t0) > s.solve_deadline_s:
                failure_reason = f"solve exceeded {s.solve_deadline_s:.1f}s budget at Newton {it}"
                logger.debug("deadline_exceeded phase=newton_top it=%d reason=%s", it, failure_reason)
                return X, self._make_report(
                    False, source_scale, nrm, it - 1, gmres_total,
                    factor_total, t0, failure_reason,
                    preconditioner_assembly_runtime_s=precond_assembly_total,
                    preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
                )
            pre_coeff = nrm["coeff_rel"]
            R = problem.residual_coeffs(X, source_scale)
            if not _finite_state(R):
                logger.debug("solve_one_fail it=%d reason=%s", it, "non-finite residual before linear solve")
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
                logger.debug("solve_one_fail it=%d reason=%s", it, "non-finite tangent before linear solve")
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
            logger.debug(
                "preconditioner_reuse_decision it=%d refresh=%s have_cache=%s "
                "steps_since_factor=%d precond_reuse=%d last_gmres=%d "
                "refresh_gmres_threshold=%d preconditioner=%s",
                it, refresh, have_cache, steps_since_factor, s.precond_reuse,
                last_gmres, s.precond_reuse_refresh_gmres, s.preconditioner,
            )

            spectral_tangent = None
            if refresh and (s.jvp_mode == "spectral"
                            or s.preconditioner in ("spectral_coupled", "real_coupled")):
                spectral_tangent = problem.spectral_tangent_state(tangent)
            elif s.jvp_mode == "spectral":
                spectral_tangent = problem.spectral_tangent_state(tangent)

            tf = time.perf_counter()
            if refresh:
                logger.debug("preconditioner_rebuild it=%d kind=%s", it, s.preconditioner)
                if s.preconditioner == "real_coupled":
                    cached_real = problem.assemble_real_coupled_preconditioner(spectral_tangent)
                    cached_coupled = cached_factors = None
                elif s.preconditioner == "real_coupled_fast":
                    # Exact real-coupled with assembly + symbolic-factorization
                    # reuse. Implemented ONLY for the Schur-reduced problem
                    # (in-process Schur backend); the full solver runs
                    # the full problem, which has no such method.
                    if not hasattr(problem, "assemble_real_coupled_fast"):
                        logger.debug(
                            "preconditioner_unavailable it=%d kind=real_coupled_fast "
                            "reason=%s", it, "problem lacks assemble_real_coupled_fast",
                        )
                        raise NotImplementedError(
                            "real_coupled_fast is available only for the "
                            "Schur-reduced backend (in-process preconditioner "
                            "real_coupled_fast). For the full solver use "
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
                logger.debug(
                    "preconditioner_reused it=%d steps_since_factor=%d",
                    it, steps_since_factor,
                )
            real_factor = cached_real
            coupled_factor = cached_coupled
            factors = cached_factors
            factor_s = time.perf_counter() - tf
            factor_total += factor_s
            logger.debug("preconditioner_factor_timing it=%d factor_s=%.6f factor_total=%.6f", it, factor_s, factor_total)
            if refresh and s.preconditioner == "real_coupled_fast" and cached_real is not None:
                assembly_rt = float(getattr(cached_real, "last_assembly_runtime_s", 0.0))
                factor_rt = float(getattr(cached_real, "last_factor_runtime_s", 0.0))
                precond_assembly_total += assembly_rt
                precond_numeric_factor_total += factor_rt
                logger.debug(
                    "preconditioner_backend_timing it=%d assembly_runtime_s=%.6f "
                    "factor_runtime_s=%.6f backend=%s",
                    it, assembly_rt, factor_rt,
                    getattr(cached_real, "last_factor_backend", None),
                )

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

            logger.debug(
                "gmres_call_begin it=%d rtol=%s atol=%s restart=%d maxiter=%d",
                it, s.gmres_rtol, s.gmres_atol, s.gmres_restart, s.gmres_maxiter,
            )
            try:
                delta_real, info = gmres_call(
                    A=Aop,
                    b=rhs,
                    M=Mop,
                    rtol=s.gmres_rtol,
                    atol=s.gmres_atol,
                    restart=s.gmres_restart,
                    maxiter=s.gmres_maxiter,
                    callback=cb,
                    deadline_s=s.solve_deadline_s,
                    t0=t0,
                )
            except GmresDeadlineExceeded:
                gmres_total += gmres_counter["n"]
                failure_reason = (
                    f"solve exceeded {s.solve_deadline_s:.1f}s budget mid-GMRES "
                    f"at Newton {it}"
                )
                logger.debug(
                    "deadline_exceeded phase=mid_gmres it=%d gmres_iters=%d reason=%s",
                    it, gmres_counter["n"], failure_reason,
                )
                return X, self._make_report(
                    False, source_scale, nrm, it - 1, gmres_total,
                    factor_total, t0, failure_reason,
                    preconditioner_assembly_runtime_s=precond_assembly_total,
                    preconditioner_numeric_factor_runtime_s=precond_numeric_factor_total,
                )
            gmres_total += gmres_counter["n"]
            last_gmres = gmres_counter["n"]
            logger.debug(
                "gmres_call_end it=%d gmres_iters=%d info=%d gmres_total=%d",
                it, gmres_counter["n"], info, gmres_total,
            )

            if info != 0:
                failure_reason = f"GMRES did not fully converge, info={info}"
                logger.debug("gmres_not_converged it=%d info=%d", it, info)

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

                decreased = trial_nrm["coeff_rel"] < nrm["coeff_rel"]
                logger.debug(
                    "line_search_step it=%d alpha=%.6e trial_coeff_rel=%s decreased=%s",
                    it, alpha, trial_nrm["coeff_rel"], decreased,
                )
                if decreased:
                    accepted = True
                    best_X = Xtrial
                    best_nrm = trial_nrm
                    break

                alpha *= 0.5

            if not accepted:
                failure_reason = failure_reason or f"line search failed at Newton {it}"
                logger.debug("line_search_failed it=%d min_alpha=%s reason=%s", it, s.min_alpha, failure_reason)
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
            logger.debug(
                "line_search_accepted it=%d alpha=%.6e coeff_rel=%s",
                it, alpha, nrm["coeff_rel"],
            )

            # Stall detection: if accepted steps keep barely reducing the
            # residual while still far from tolerance, the operating point is
            # over the fold / unsolvable. Bail instead of grinding to max_newton.
            if s.stall_patience > 0 and nrm["coeff_rel"] > 100.0 * s.newton_tol:
                ratio = nrm["coeff_rel"] / max(pre_coeff, 1e-300)
                stall_count = stall_count + 1 if ratio > s.stall_ratio else 0
                logger.debug(
                    "stall_check it=%d ratio=%.6f stall_count=%d stall_patience=%d",
                    it, ratio, stall_count, s.stall_patience,
                )
                if stall_count >= s.stall_patience:
                    failure_reason = (
                        f"stalled at Newton {it} (reduction ratio {ratio:.3f})"
                    )
                    logger.debug(
                        "stall_detected it=%d ratio=%.6f coeff_rel=%s reason=%s",
                        it, ratio, nrm["coeff_rel"], failure_reason,
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
                logger.debug(
                    "newton_converged it=%d coeff_rel=%s gmres_total=%d",
                    it, nrm["coeff_rel"], gmres_total,
                )
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
        logger.debug(
            "newton_loop_exhausted max_newton=%d reason=%s coeff_rel=%s",
            s.max_newton, failure_reason, nrm["coeff_rel"],
        )
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
        max_wall_s: float = 0.0,
        lambda_start: float = 0.0,
    ) -> tuple[np.ndarray, list[StepReport]]:
        """Fixed-step source-scale ladder from ``lambda_start`` to 1.0.

        ``lambda_start`` lets a caller resume a ladder mid-span (e.g. the
        adaptive-continuation fallback below, which restarts from the best
        state already reached instead of re-deriving the cheap low-lambda
        region from scratch).
        """
        reports: list[StepReport] = []
        continuation_t0 = time.perf_counter()

        lambdas = np.linspace(lambda_start, 1.0, continuation_steps + 1)[1:]

        X_prevprev: np.ndarray | None = None
        X_prev = problem.zeros() if x_init is None else np.array(x_init, dtype=np.complex128, copy=True)
        lam_prevprev: float | None = None
        lam_prev: float | None = None

        for lam_raw in lambdas:
            if max_wall_s > 0.0 and time.perf_counter() - continuation_t0 > max_wall_s:
                return X_prev, reports
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
                # Resume the fixed ladder from X_current/lambda_current (the
                # best state the adaptive phase already reached), not from
                # x_init/lambda=0. Restarting from the original seed threw
                # away all adaptive progress and re-derived the cheap
                # low-lambda region from scratch, burning the wall-time
                # budget before the ladder could get back near the fold with
                # finer steps (see docs/control_flow debug trace, point 8 of
                # column_debug_col3_trim: adaptive reached lambda=0.9375
                # converged, then the old fallback restarted at lambda=0.05
                # and only got back to lambda=0.75 before the deadline hit).
                # Steps sized to the original 1/fallback_fixed_steps
                # granularity, applied to the remaining span only.
                remaining = 1.0 - lambda_current
                step_size = 1.0 / fallback_fixed_steps
                fallback_steps = max(1, math.ceil(remaining / step_size - 1e-9))
                logger.debug(
                    "adaptive_continuation_fallback lambda_current=%.6f "
                    "remaining=%.6f fallback_steps=%d coeff_rel=%s",
                    lambda_current, remaining, fallback_steps,
                    reports[-1].coeff_rel if reports else None,
                )
                print(
                    "adaptive continuation failed below minimum step; "
                    f"falling back to fixed {fallback_steps} steps from "
                    f"lambda={lambda_current:.6f} (resuming, not restarting)"
                )
                X_fallback, fallback_reports = self.solve_continuation(
                    problem,
                    continuation_steps=fallback_steps,
                    x_init=X_current,
                    lambda_start=lambda_current,
                    max_wall_s=max(
                        0.0,
                        max_wall_s - (time.perf_counter() - continuation_t0),
                    ) if max_wall_s > 0.0 else 0.0,
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
        deadline_s: float = 0.0,
        t0: float | None = None,
        gmres_iter_sink: list[int] | None = None,
    ):
        """Build a reusable ``J(X)+shift*I`` GMRES solver at the point ``X``.

        Returns a closure ``solve(rhs_coeffs) -> delta`` sharing one tangent
        state and one mean-tangent preconditioner factorization across multiple
        right-hand sides. Reusing this is the key cost saver in the arclength
        corrector, which solves two systems (``J a = -R`` and ``J b = S``) per
        Newton iteration with the same operator.

        ``deadline_s``/``t0`` (if ``deadline_s > 0``) bound each individual
        GMRES call so a single stiff solve cannot silently outrun the caller's
        wall-time budget; on expiry the returned closure raises
        ``GmresDeadlineExceeded``.

        ``gmres_iter_sink`` (default ``None``, no behavior change): if given,
        every ``solve()`` call appends one entry per inner GMRES iteration
        (``callback_type="pr_norm"`` fires once per iteration) -- callers
        that want per-corrector-step GMRES iteration counts for telemetry
        read ``len(sink)`` after the calls they care about, instead of the
        solver tracking this unconditionally.
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

        if gmres_iter_sink is not None:
            def _count_iter(_n: float) -> None:
                gmres_iter_sink.append(1)
        else:
            def _count_iter(_n: float) -> None:
                pass

        def solve(rhs_coeffs: np.ndarray) -> np.ndarray:
            delta_real, _info = gmres_call(
                A=Aop, b=pack_complex(rhs_coeffs), M=Mop,
                rtol=s.gmres_rtol, atol=s.gmres_atol, restart=s.gmres_restart,
                maxiter=s.gmres_maxiter, callback=_count_iter,
                deadline_s=deadline_s, t0=t0,
            )
            return unpack_complex(delta_real, shape)

        return solve

    def _matvec_fn(
        self, problem: FullPumpProblem, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Build a JVP-only closure (no factorization) at ``X``.

        Used by :func:`bordered_solve_refined` to compute the bordered
        system's own residual -- a separate, cheap operation from the
        factorized ``linsolve`` closure ``_linear_solver`` builds.
        """
        tangent = problem.tangent_state(X)

        def matvec(V: np.ndarray) -> np.ndarray:
            return problem.jvp_coeffs_with_tangent(V, tangent)

        return matvec

    def _solve_linear(
        self,
        problem: FullPumpProblem,
        X: np.ndarray,
        rhs_coeffs: np.ndarray,
        *,
        shift: float = 0.0,
        deadline_s: float = 0.0,
        t0: float | None = None,
    ) -> np.ndarray:
        """Solve ``(J(X)+shift*I) delta = rhs_coeffs`` matrix-free by GMRES."""
        return self._linear_solver(
            problem, X, shift=shift, deadline_s=deadline_s, t0=t0,
        )(rhs_coeffs)

    def tangent_predictor(
        self, problem: FullPumpProblem, X: np.ndarray, d_lambda: float,
    ) -> np.ndarray:
        """Euler predictor along the branch: ``X + d_lambda * dX/dlambda``.

        ``dR/dlambda = -S`` (the source is linear in lambda), so the branch
        tangent solves ``J * Xdot = S`` where ``S = source_coeffs(1)``. More
        robust than the secant when the previous two points are far apart or the
        branch curves sharply (near the fold).
        """
        S = problem.source_coeffs(1.0) - problem.source_coeffs(0.0)
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
            try:
                delta_X = self._solve_linear(
                    problem, X, -R, shift=1.0 / delta,
                    deadline_s=s.solve_deadline_s, t0=t0,
                )
            except GmresDeadlineExceeded:
                return X, [self._make_report(
                    False, 1.0, nrm, it - 1, gmres_total, 0.0, t0,
                    f"ptc exceeded {s.solve_deadline_s:.1f}s budget mid-GMRES",
                )]
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

    def _corrector_step(
        self,
        problem: FullPumpProblem,
        X_anchor: np.ndarray,
        lam_anchor: float,
        Xdot: np.ndarray,
        lam_dot: float,
        step_size: float,
        metric_x: Callable[[np.ndarray, np.ndarray], float],
        S: np.ndarray,
        newton_max: int,
        max_wall_s: float,
        t0: float,
        tol: float,
        gmres_iter_sink: list[int] | None = None,
    ) -> tuple[bool, np.ndarray, float, int, Callable[[np.ndarray], np.ndarray] | None]:
        """One PALC predictor + bordering-algorithm Newton corrector step.

        Shared by :meth:`solve_arclength`'s main loop and
        :meth:`_refine_fold`'s secant sub-steps -- a pure extraction, no
        behavior change. Returns ``(converged, Xc, lamc, used_newton, lin)``
        where ``lin`` is the last factored linear solve at ``Xc`` (``None``
        if the corrector converged on the first check, before any factor was
        built) so callers can reuse it for a tangent solve without a
        redundant refactor. Raises ``GmresDeadlineExceeded`` or one of
        ``(RuntimeError, FloatingPointError, ValueError, OverflowError)`` on
        a genuine linear-solve failure -- callers are responsible for
        catching these, exactly as before this was extracted.

        ``gmres_iter_sink`` (default ``None``, no behavior change): forwarded
        to every ``_linear_solver`` call made during this corrector step, so
        a caller can read ``len(sink)`` afterward for a telemetry GMRES
        iteration count.
        """
        X_pred = X_anchor + step_size * Xdot
        lam_pred = lam_anchor + step_size * lam_dot
        Xc, lamc = np.array(X_pred, copy=True), float(lam_pred)
        converged = False
        used_newton = newton_max
        lin: Callable[[np.ndarray], np.ndarray] | None = None
        for it in range(1, newton_max + 1):
            R = problem.residual_coeffs(Xc, lamc)
            n = metric_x(Xdot, Xc - X_anchor) + lam_dot * (lamc - lam_anchor) - step_size
            rel = problem.norms(Xc, lamc, False)["coeff_rel"]
            if rel < tol and abs(n) < tol * max(step_size, 1.0):
                converged, used_newton = True, it - 1
                break
            lin = self._linear_solver(
                problem, Xc, deadline_s=max_wall_s, t0=t0,
                gmres_iter_sink=gmres_iter_sink,
            )
            matvec = self._matvec_fn(problem, Xc)
            step_result = bordered_solve_refined(
                matvec, lin, R, n, S, lambda v: metric_x(Xdot, v), lam_dot,
            )
            if step_result is None:
                break
            d_X, d_lam = step_result
            Xc = Xc + d_X
            lamc = lamc + d_lam
        return converged, Xc, lamc, used_newton, lin

    def _refine_fold(
        self,
        problem: FullPumpProblem,
        X_anchor: np.ndarray,
        lam_anchor: float,
        Xdot_anchor: np.ndarray,
        lam_dot_anchor: float,
        lam_bracket_b: float,
        t_bracket_b: float,
        ds_ab: float,
        metric_x: Callable[[np.ndarray, np.ndarray], float],
        S: np.ndarray,
        newton_max: int,
        max_wall_s: float,
        t0: float,
        tol: float,
        *,
        t_tol: float,
        lam_tol: float,
        max_iter: int,
    ) -> dict:
        """Secant-refine a fold bracketed between the anchor and point B.

        fold_plan.md Section 16: repeatedly predicts from the FIXED anchor
        point (the last accepted point before the sign flip, with its own
        tangent) at a secant-estimated arclength offset toward the zero of
        the tangent's lambda-component, using the same bordering corrector as
        the main loop (:meth:`_corrector_step`) -- no minimally augmented
        fold system (fold_plan.md Section 17's transversality test is
        Milestone F, not this one). Stops when ``|lambda_dot| < t_tol`` or the
        lambda-bracket has collapsed below ``lam_tol``, whichever comes
        first. A corrector or linear-solve failure mid-refinement stops the
        refinement (not the caller's continuation run) and reports
        ``converged=False`` with whatever the last successful iterate was.
        """
        sa, va, lam_a = 0.0, lam_dot_anchor, lam_anchor
        sb, vb, lam_b = ds_ab, t_bracket_b, lam_bracket_b
        X_best: np.ndarray | None = None
        lam_best: float | None = None
        t_best: float | None = None
        converged = False
        iterations = 0
        for _ in range(max_iter):
            iterations += 1
            if abs(sb - sa) < 1e-300 or vb == va:
                break
            s_star = sa - va * (sb - sa) / (vb - va)
            lo, hi = (sa, sb) if sa < sb else (sb, sa)
            # Only fall back to bisection on genuine extrapolation (s_star
            # outside the bracket, e.g. from a near-degenerate va/vb) --
            # secant legitimately lands close to whichever edge the root is
            # nearer to as the bracket narrows, and rejecting that in favor
            # of a flat interior margin (the original 10%-of-width guard)
            # forces slow bisection that never recovers secant's rate.
            margin = 1e-9 * abs(sb - sa)
            if not (lo + margin <= s_star <= hi - margin):
                s_star = 0.5 * (sa + sb)
            step_size_star = s_star - sa
            try:
                conv, Xc, lamc, _used, lin = self._corrector_step(
                    problem, X_anchor, lam_anchor, Xdot_anchor, lam_dot_anchor,
                    step_size_star, metric_x, S, newton_max, max_wall_s, t0, tol,
                )
            except (GmresDeadlineExceeded, RuntimeError, FloatingPointError, ValueError, OverflowError):
                break
            if not conv:
                break
            try:
                if lin is None:
                    lin = self._linear_solver(problem, Xc, deadline_s=max_wall_s, t0=t0)
                Xdot_c = lin(S)
            except (GmresDeadlineExceeded, RuntimeError, FloatingPointError, ValueError, OverflowError):
                break
            lam_dot_c = 1.0
            nrm = math.sqrt(metric_x(Xdot_c, Xdot_c) + lam_dot_c * lam_dot_c)
            if not math.isfinite(nrm) or nrm <= 1e-300:
                break
            Xdot_c, lam_dot_c = Xdot_c / nrm, lam_dot_c / nrm
            if metric_x(Xdot_c, Xdot_anchor) + lam_dot_c * lam_dot_anchor < 0.0:
                Xdot_c, lam_dot_c = -Xdot_c, -lam_dot_c
            t_star = lam_dot_c
            X_best, lam_best, t_best = Xc, lamc, t_star
            if t_star * va >= 0.0:
                sa, va, lam_a = s_star, t_star, lamc
            else:
                sb, vb, lam_b = s_star, t_star, lamc
            if abs(t_star) < t_tol or abs(lam_b - lam_a) < lam_tol:
                converged = True
                break
        return {
            "converged": bool(converged),
            "lam": lam_best,
            "X": X_best,
            "lam_dot": t_best,
            "bracket_width": abs(lam_b - lam_a),
            "iterations": iterations,
        }

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
        on_step: Callable[[np.ndarray, float, float, np.ndarray, float, float], None] | None = None,
        rescale_every: int = 0,
        max_steps_after_fold: int | None = None,
        step_control: str = "legacy",
        newton_target: int = 4,
        step_growth_exponent: float = 0.5,
        step_growth_clamp: tuple[float, float] = (0.5, 1.5),
        ds_max: float | None = None,
        refine_fold: bool = False,
        fold_t_tol: float = 1e-6,
        fold_lambda_tol: float | None = None,
        fold_refine_max_iter: int = 30,
        step_telemetry: Callable[[dict], None] | None = None,
    ) -> tuple[np.ndarray, float, dict]:
        """Pseudo-arclength continuation in the source scale ``lambda``.

        Treats ``(X, lambda)`` as joint unknowns and advances a pseudo-arclength
        ``s`` with the tangent-plane constraint, so the augmented system stays
        non-singular through the fold (Keller 1977). Each corrector Newton step
        uses the bordering algorithm: two solves with ``J`` (``J a = -R`` and
        ``J b = S``) combined by the scalar constraint.

        Returns ``(X, lambda, info)`` where ``info`` records whether ``target_lam``
        was reached and the fold ``lambda`` if a turning point (sign change of
        ``lambda_dot``) was crossed first. ``info["state_scale"]`` is the
        scale factor applied to the state contribution of the arclength
        metric (``metric_x(a,b) = Re<a,b> / state_scale**2``), derived from
        the initial tangent so the state and lambda terms of the arclength
        constraint are comparable regardless of the state's physical units.

        ``on_step``, if given, is called
        ``on_step(Xc, lamc, step_size, Xdot, lam_dot, state_scale)`` once per
        *accepted* corrector step (never on a halved/rejected attempt), at
        the point the solver actually visited, with the tangent that
        predicted it (not yet updated to the new point's own tangent) and the
        metric's current ``state_scale`` -- callers that need a singularity
        measurement along the branch (not just at the final endpoint) hook in
        here.

        ``rescale_every`` (default ``0``, disabled -- identical behavior to
        before this parameter existed): every ``N`` accepted steps, update
        ``state_scale`` from the robust median of recent accepted state
        amplitudes. The proposed scale is independent of tangent orientation;
        in particular it never contains ``1 / abs(lambda_dot)`` and therefore
        cannot erase a fold by forcing the lambda tangent component to a fixed
        value. When the metric changes, the tangent and next step are
        transformed together so the physical predictor ``ds * (Xdot, lam_dot)``
        is preserved to floating-point precision. ``info["rescale_count"]``
        records how many times this fired and ``info["rescale_events"]`` stores
        the corresponding predictor-invariance diagnostics.

        ``max_steps_after_fold`` (default ``None`` -- no change from plain
        ``max_steps``): once a fold is first detected (``lambda_dot`` sign
        change), the step budget is extended to at least
        ``step_at_fold + max_steps_after_fold`` if that is larger than the
        original ``max_steps``. A fold consumes an a-priori-unknown number of
        steps just to reach; without this, a run that spends most of
        ``max_steps`` getting to the fold has too little budget left to round
        it and find a returning branch, and reports ``terminal_reason=
        "max_steps"`` even though a returning branch may exist just past the
        exhausted budget.

        ``step_control`` (default ``"legacy"``, byte-identical to before this
        parameter existed): ``"adaptive"`` replaces the crude "grow 1.25x on
        an easy corrector, cap at ``ds_initial``" rule with the Newton-effort
        + curvature step control of fold_plan.md Sections 10-11
        (:func:`_adaptive_step_size`) -- ``newton_target``/
        ``step_growth_exponent``/``step_growth_clamp`` tune the Newton-effort
        factor, ``ds_max`` (default ``None`` -> ``ds``, matching the legacy
        cap) raises the ceiling so a well-behaved branch can accelerate past
        its initial step. The halving-on-failure/``step_floor`` behavior
        (Section 12) is unchanged in both modes.

        ``refine_fold`` (default ``False``): once the first turning point is
        detected, run a bounded secant refinement
        (:meth:`_refine_fold`, fold_plan.md Section 16) instead of accepting
        the bracketing corrector point as the fold location. Stores
        ``info["fold_refined"]`` (``None`` if not requested or if the
        refinement made zero successful iterations) with keys
        ``converged``, ``lam``, ``X``, ``lam_dot``, ``bracket_width``,
        ``iterations``. ``fold_lambda_tol`` (default ``None`` -> scales off
        the arclength step actually taken at the fold, ``max(1e-8, step_size
        * 1e-4)`` -- NOT off ``|lambda_fold - lambda_prev|``, which is itself
        degenerate near a fold since ``lambda_dot -> 0`` there) is the
        lambda-bracket stopping width; ``fold_t_tol`` is the ``|lambda_dot|``
        stopping tolerance.
        """
        X = problem.zeros() if X0 is None else np.array(X0, dtype=np.complex128, copy=True)
        lam = float(lam0)
        S = problem.source_coeffs(1.0)
        t0 = time.perf_counter()
        info: dict = {
            "reached_target": False,
            "fold_lambda": None,
            "fold_refined": None,
            "steps": 0,
            "terminal_reason": "max_steps",
            "state_scale": None,
            "rescale_count": 0,
            "rescale_events": [],
            "rejected_steps": 0,
        }

        # Initial tangent: J Xdot = S (unnormalised).
        try:
            Xdot = self._solve_linear(problem, X, S, deadline_s=max_wall_s, t0=t0)
        except GmresDeadlineExceeded:
            info["terminal_reason"] = "deadline"
            return X, lam, info
        except (RuntimeError, FloatingPointError, ValueError, OverflowError) as exc:
            info["terminal_reason"] = "singular_jacobian"
            info["error"] = repr(exc)
            return X, lam, info
        lam_dot = 1.0

        # State-vs-lambda have very different physical units (node flux in
        # webers vs a dimensionless source scale); an unscaled Euclidean
        # metric makes the lambda term dominate by ~1e26 on a real device,
        # collapsing this into natural-parameter continuation (fold detection
        # structurally impossible, corrector never converges past a turn).
        # Derive the state scale from the initial tangent, same construction
        # as the seeded ``trace_arclength_from_two_points`` (:1173-1177), so
        # both contributions to the arclength constraint are O(1) together.
        raw_state_norm = math.sqrt(_real_dot(Xdot, Xdot))
        x_norm = math.sqrt(_real_dot(X, X))
        if not math.isfinite(raw_state_norm) or not math.isfinite(x_norm):
            info["terminal_reason"] = "degenerate_tangent"
            return X, lam, info
        state_scale = max(raw_state_norm / abs(lam_dot), x_norm * 1e-6, 1e-300)
        if state_scale <= 1e-300:
            info["terminal_reason"] = "degenerate_tangent"
            return X, lam, info
        info["state_scale"] = state_scale
        recent_state_norms: list[float] = []
        if x_norm > 1e-300:
            recent_state_norms.append(x_norm)

        def metric_x(a: np.ndarray, b: np.ndarray) -> float:
            return _real_dot(a, b) / (state_scale * state_scale)

        norm = math.sqrt(metric_x(Xdot, Xdot) + lam_dot * lam_dot)
        Xdot, lam_dot = Xdot / norm, lam_dot / norm

        tol = max(self.settings.newton_tol * 10.0, 1e-8)  # relative coeff tol
        ds_initial = float(ds)
        step_size = float(ds)
        step_floor = ds_initial * 1e-6
        info["endpoint_above_newton_tol"] = None
        accepted_steps = 0
        effective_max_steps = max_steps
        pending_rescale_event: dict[str, float] | None = None
        step = 0
        while step < effective_max_steps:
            step += 1
            info["steps"] = step
            if max_wall_s > 0.0 and time.perf_counter() - t0 > max_wall_s:
                info["terminal_reason"] = "deadline"
                return X, lam, info
            # Full Newton: refactor at Xc every corrector iteration (a frozen
            # predictor-point factor is worst exactly where this method is
            # supposed to work -- near a fold, where the Jacobian moves fastest).
            # Each GMRES call is itself bounded by max_wall_s so one stiff
            # corrector step cannot silently outrun the budget between the
            # outer checks above.
            gmres_iter_sink: list[int] | None = [] if step_telemetry is not None else None
            try:
                converged, Xc, lamc, used_newton, lin = self._corrector_step(
                    problem, X, lam, Xdot, lam_dot, step_size,
                    metric_x, S, newton_max, max_wall_s, t0, tol,
                    gmres_iter_sink=gmres_iter_sink,
                )
            except GmresDeadlineExceeded:
                info["terminal_reason"] = "deadline"
                return X, lam, info
            except (RuntimeError, FloatingPointError, ValueError, OverflowError) as exc:
                info["terminal_reason"] = "singular_jacobian"
                info["error"] = repr(exc)
                return X, lam, info
            if not converged:
                info["rejected_steps"] += 1
                step_size *= 0.5
                if step_size < step_floor:
                    info["terminal_reason"] = "minimum_step"
                    return X, lam, info
                continue
            if on_step is not None:
                on_step(Xc, lamc, step_size, Xdot, lam_dot, state_scale)
            # New tangent from the corrected point's own factor (a stale
            # predictor-point factor would be systematically O(step_size) wrong
            # for the tangent direction). Converging on the very first check
            # (no correction needed) leaves ``lin`` unbuilt -- build it once
            # at the accepted Xc in that case.
            try:
                if lin is None:
                    lin = self._linear_solver(
                        problem, Xc, deadline_s=max_wall_s, t0=t0,
                    )
                Xdot_new = lin(S)
            except GmresDeadlineExceeded:
                info["terminal_reason"] = "deadline"
                return X, lam, info
            except (RuntimeError, FloatingPointError, ValueError, OverflowError) as exc:
                info["terminal_reason"] = "singular_jacobian"
                info["error"] = repr(exc)
                return X, lam, info
            lam_dot_new = 1.0
            nrm = math.sqrt(metric_x(Xdot_new, Xdot_new) + lam_dot_new ** 2)
            Xdot_new, lam_dot_new = Xdot_new / nrm, lam_dot_new / nrm
            if metric_x(Xdot_new, Xdot) + lam_dot_new * lam_dot < 0.0:
                Xdot_new, lam_dot_new = -Xdot_new, -lam_dot_new
            # Tangent angle between the predicting tangent and the newly
            # accepted point's own tangent (fold_plan.md Section 11) -- cheap
            # (metric_x is already O(n) elsewhere this iteration), computed
            # unconditionally so it is available as telemetry regardless of
            # step_control mode.
            cos_theta = metric_x(Xdot, Xdot_new) + lam_dot * lam_dot_new
            theta_k = math.acos(max(-1.0, min(1.0, cos_theta)))
            info["theta_last_deg"] = math.degrees(theta_k)
            if step_telemetry is not None:
                # Recomputed here (not read out of `_corrector_step`, which
                # only returns the boolean convergence flag) -- both are
                # cheap function evaluations (no new factorization) relative
                # to the corrector's own linear solves.
                hb_residual_rel = problem.norms(Xc, lamc, False)["coeff_rel"]
                palc_residual = (
                    metric_x(Xdot, Xc - X) + lam_dot * (lamc - lam) - step_size
                )
                step_telemetry({
                    "step": step, "lam": lamc, "step_size": step_size,
                    "used_newton": used_newton, "theta_deg": math.degrees(theta_k),
                    "state_scale": state_scale, "rejected_steps": info["rejected_steps"],
                    "hb_residual_rel": float(hb_residual_rel),
                    "palc_residual": float(palc_residual),
                    "gmres_iterations": (
                        len(gmres_iter_sink) if gmres_iter_sink is not None else None
                    ),
                    # t_lam_pred: the tangent that PREDICTED this accepted
                    # point (same convention as BranchPoint.t_mu). t_lam_own:
                    # this point's own freshly computed tangent -- the two
                    # agree away from a fold and both are provided since a
                    # traversal analysis cares about the tangent's sign AT
                    # each point, not just the one that led into it.
                    "t_lam_pred": float(lam_dot),
                    "t_lam_own": float(lam_dot_new),
                    "rescale_event": float(pending_rescale_event is not None),
                    "rescale_norm": (
                        pending_rescale_event["rescale_norm"]
                        if pending_rescale_event is not None else 1.0
                    ),
                    "rescale_step_size_before": (
                        pending_rescale_event["step_size_before"]
                        if pending_rescale_event is not None else step_size
                    ),
                    "rescale_step_size_after": (
                        pending_rescale_event["step_size_after"]
                        if pending_rescale_event is not None else step_size
                    ),
                    "rescale_predictor_error": (
                        pending_rescale_event["predictor_error"]
                        if pending_rescale_event is not None else 0.0
                    ),
                })
                pending_rescale_event = None
            # Fold = sign change of lam_dot.
            if lam_dot_new * lam_dot < 0.0 and info["fold_lambda"] is None:
                info["fold_lambda"] = float(lamc)
                if max_steps_after_fold is not None:
                    effective_max_steps = max(
                        effective_max_steps, step + max_steps_after_fold,
                    )
                if refine_fold:
                    # Default scales off the arclength step actually taken
                    # (well-conditioned, ~ds), not off |lamc - lam| -- that
                    # lambda displacement is exactly what goes to zero AT a
                    # fold (lambda_dot -> 0 there), so scaling off it makes
                    # the default tolerance degenerate near the very point
                    # being refined.
                    lam_tol_eff = (
                        fold_lambda_tol if fold_lambda_tol is not None
                        else max(1e-8, step_size * 1e-4)
                    )
                    try:
                        info["fold_refined"] = self._refine_fold(
                            problem, X, lam, Xdot, lam_dot, lamc, lam_dot_new,
                            step_size, metric_x, S, newton_max, max_wall_s, t0, tol,
                            t_tol=fold_t_tol, lam_tol=lam_tol_eff,
                            max_iter=fold_refine_max_iter,
                        )
                    except (
                        GmresDeadlineExceeded, RuntimeError,
                        FloatingPointError, ValueError, OverflowError,
                    ) as exc:
                        info["fold_refined"] = {
                            "converged": False, "error": repr(exc),
                        }
            # Target crossing (interpolate to target_lam). The interpolated
            # endpoint is consumed downstream as a warm guess for a final
            # target solve, not a converged point -- flag when it sits above
            # production newton_tol so a caller can tell a loose endpoint from
            # a tight one instead of silently wasting a solve on it.
            if (lam - target_lam) * (lamc - target_lam) <= 0.0 and lamc != lam:
                theta = (target_lam - lam) / (lamc - lam)
                X = X + theta * (Xc - X)
                endpoint_rel = problem.norms(X, target_lam, False)["coeff_rel"]
                info["endpoint_above_newton_tol"] = bool(
                    endpoint_rel > self.settings.newton_tol
                )
                info["reached_target"] = True
                info["terminal_reason"] = "target"
                return X, target_lam, info
            X, lam, Xdot, lam_dot = Xc, lamc, Xdot_new, lam_dot_new
            accepted_steps += 1
            new_x_norm = math.sqrt(_real_dot(X, X))
            if math.isfinite(new_x_norm) and new_x_norm > 1e-300:
                recent_state_norms.append(new_x_norm)
                del recent_state_norms[:-9]
            rescale_norm: float | None = None
            rescale_scale_before: float | None = None
            rescale_scale_after: float | None = None
            if rescale_every > 0 and accepted_steps % rescale_every == 0:
                if recent_state_norms:
                    proposed_state_scale = float(np.median(recent_state_norms))
                    # Let the metric follow changing state amplitudes without
                    # making one update itself a geometric kink. The bounds
                    # depend only on the previous metric, never on tangent
                    # orientation or ``lam_dot``.
                    new_state_scale = max(
                        state_scale * 0.99,
                        min(state_scale * 1.01, proposed_state_scale),
                    )
                    transformed = _rescale_arclength_tangent(
                        Xdot, lam_dot, new_state_scale, step_size,
                    )
                    if transformed is not None:
                        Xdot, lam_dot, _unused_step_size, rescale_norm = transformed
                        rescale_scale_before = state_scale
                        state_scale = new_state_scale
                        rescale_scale_after = state_scale
                        info["state_scale"] = state_scale
                        info["rescale_count"] += 1
            # Adjust the step for the next iteration; halving/floor on a
            # failed corrector (above) is unchanged in both modes.
            next_step_size = step_size
            if step_control == "adaptive":
                next_step_size = _adaptive_step_size(
                    step_size, used_newton, theta_k,
                    newton_target=newton_target,
                    growth_exponent=step_growth_exponent,
                    growth_clamp=step_growth_clamp,
                    ds_max=ds_max if ds_max is not None else ds_initial,
                    ds_min=step_floor,
                )
            elif used_newton <= 3:
                next_step_size = min(ds_initial, step_size * 1.25)
            if rescale_norm is not None:
                step_size_before_rescale = next_step_size
                next_step_size *= rescale_norm
                predictor_error = abs(
                    next_step_size / (step_size_before_rescale * rescale_norm) - 1.0
                )
                rescale_event = {
                    "step": step,
                    "state_scale_before": rescale_scale_before,
                    "state_scale_after": rescale_scale_after,
                    "rescale_norm": rescale_norm,
                    "step_size_before": step_size_before_rescale,
                    "step_size_after": next_step_size,
                    "predictor_error": predictor_error,
                }
                info["rescale_events"].append(rescale_event)
                pending_rescale_event = rescale_event
            step_size = next_step_size
        return X, lam, info

    def solve_arclength_mu(
        self,
        problem: FullPumpProblem,
        X0: np.ndarray | None,
        mu0: float,
        *,
        i_ref: float,
        target_mu: float = 1.0,
        ds: float = 0.1,
        max_steps: int = 200,
        newton_max: int = 12,
        max_wall_s: float = 0.0,
        on_step: Callable[[np.ndarray, float, float, np.ndarray, float, float], None] | None = None,
        rescale_every: int = 0,
        max_steps_after_fold: int | None = None,
        step_control: str = "legacy",
        newton_target: int = 4,
        step_growth_exponent: float = 0.5,
        step_growth_clamp: tuple[float, float] = (0.5, 1.5),
        ds_max: float | None = None,
        refine_fold: bool = False,
        fold_t_tol: float = 1e-6,
        fold_mu_tol: float | None = None,
        fold_refine_max_iter: int = 30,
        step_telemetry: Callable[[dict], None] | None = None,
    ) -> tuple[np.ndarray, float, dict]:
        """``solve_arclength`` reparameterized around a fixed physical drive.

        ``solve_arclength``'s ``lambda`` is target-normalized: ``lambda=1``
        means "``problem.pump_current_a``", a value baked into ``problem`` at
        construction and, in production, different for every requested map
        power (``run_gain_map.py::_build_problem`` is called once per grid
        cell with that cell's own target current). Two map points at the same
        frequency therefore each define a *different* arclength problem for
        the same physical branch (docs/development/fold_plan.md Section 1).

        This wrapper introduces ``mu = I_physical / i_ref``, a coordinate
        whose meaning is fixed by the caller's choice of ``i_ref`` and does
        not depend on ``problem.pump_current_a`` at all -- the same traced
        branch (and the same ``i_ref``) can be reused to bracket any number
        of physical target currents (see :func:`trace_branch`). It changes
        no physics: ``R(X, mu) := R(X, mu * k)`` where ``k = i_ref /
        problem.pump_current_a`` is a fixed scalar, computed once and applied
        only to the corrector's ``lambda``-valued inputs/outputs (``mu0``,
        ``target_mu``, ``ds``, and ``fold_lambda``/``on_step``'s ``lamc``).
        The underlying ``solve_arclength`` -- metric, bordered corrector,
        fold detection, rescale, post-fold budget -- runs completely
        unmodified. ``k=1`` (``i_ref == problem.pump_current_a``) reduces to
        calling ``solve_arclength`` directly, bit-for-bit.

        ``step_control``/``newton_target``/``step_growth_exponent``/
        ``step_growth_clamp``/``ds_max`` and ``refine_fold``/``fold_t_tol``/
        ``fold_mu_tol``/``fold_refine_max_iter`` forward to the same-named
        (Milestone C/D) ``solve_arclength`` parameters -- ``ds_max`` and
        ``fold_mu_tol`` are given in ``mu`` units and converted by ``k`` at
        the boundary like every other lambda-valued input here.

        ``step_telemetry`` (default ``None``): forwarded to
        ``solve_arclength``'s same-named parameter with its ``lam``/
        ``step_size`` entries converted to ``mu`` units (divided by ``k``,
        same convention as ``on_step`` above); every other entry (residuals,
        Newton/GMRES counts, tangent components) is dimensionless or already
        metric-normalized and passes through unchanged.
        """
        k = float(i_ref) / float(problem.pump_current_a)

        def _lambda_on_step(
            Xc: np.ndarray, lamc: float, step_size: float,
            Xdot: np.ndarray, lam_dot: float, state_scale: float,
        ) -> None:
            on_step(Xc, lamc / k, step_size / k, Xdot, lam_dot, state_scale)

        def _lambda_step_telemetry(payload: dict) -> None:
            converted = dict(payload)
            converted["mu"] = converted.pop("lam") / k
            converted["step_size"] = converted["step_size"] / k
            step_telemetry(converted)

        X, lam, info = self.solve_arclength(
            problem, X0, mu0 * k,
            ds=ds * k, max_steps=max_steps, target_lam=target_mu * k,
            newton_max=newton_max, max_wall_s=max_wall_s,
            on_step=_lambda_on_step if on_step is not None else None,
            rescale_every=rescale_every,
            max_steps_after_fold=max_steps_after_fold,
            step_control=step_control,
            newton_target=newton_target,
            step_growth_exponent=step_growth_exponent,
            step_growth_clamp=step_growth_clamp,
            ds_max=ds_max * k if ds_max is not None else None,
            refine_fold=refine_fold,
            fold_t_tol=fold_t_tol,
            fold_lambda_tol=fold_mu_tol * k if fold_mu_tol is not None else None,
            fold_refine_max_iter=fold_refine_max_iter,
            step_telemetry=_lambda_step_telemetry if step_telemetry is not None else None,
        )
        mu = lam / k
        info = dict(info)
        info["mu_ref_current_a"] = i_ref
        fold_lambda = info.get("fold_lambda")
        info["fold_mu"] = float(fold_lambda) / k if fold_lambda is not None else None
        fold_refined = info.get("fold_refined")
        if fold_refined is not None and fold_refined.get("lam") is not None:
            fr = dict(fold_refined)
            fr["mu"] = fr.pop("lam") / k
            fr["bracket_width"] = fr["bracket_width"] / k
            info["fold_refined"] = fr
        return X, mu, info

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
        max_wall_s: float = 0.0,
    ) -> tuple[list[tuple[np.ndarray, float]], dict]:
        """Trace a branch from two converged points with a scaled secant metric.

        Pump coefficients and source scale have very different physical units.
        The state scale inferred from the initial secant makes both components
        contribute comparably to the arclength constraint. This is intended for
        map recovery, where two adjacent power solutions are already available.

        ``max_wall_s`` (0 disables) bounds total trace time: checked once per
        outer step and, per GMRES call, inside each corrector solve -- this
        trace previously had no wall-clock cap at all, so a stiff corrector
        near a fold could run unbounded.
        """
        t0 = time.perf_counter()
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
            if max_wall_s > 0.0 and time.perf_counter() - t0 > max_wall_s:
                info["terminal_reason"] = "deadline"
                break
            X_pred = X + step_size * tx
            lam_pred = lam + step_size * tl
            Xc, lamc = np.array(X_pred, copy=True), float(lam_pred)
            converged = False
            used_newton = newton_max
            try:
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
                    lin = self._linear_solver(
                        problem, Xc, deadline_s=max_wall_s, t0=t0,
                    )
                    a = lin(-R)
                    b = lin(S)
                    denom = metric_x(tx, b) + tl
                    if not math.isfinite(denom) or abs(denom) < 1e-14:
                        break
                    dlam = (-constraint - metric_x(tx, a)) / denom
                    Xc = Xc + a + dlam * b
                    lamc = lamc + dlam
            except GmresDeadlineExceeded:
                info["terminal_reason"] = "deadline"
                break
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
        report = StepReport(
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
        logger.debug(
            "step_report converged=%s source_scale=%s coeff_rel=%s newton_iterations=%d "
            "gmres_iterations_total=%d factor_runtime_s=%.6f runtime_s=%.3f failure_reason=%r",
            report.converged, report.source_scale, report.coeff_rel,
            report.newton_iterations, report.gmres_iterations_total,
            report.factor_runtime_s, report.runtime_s, report.failure_reason,
        )
        return report


def gmres_call(
    A: spla.LinearOperator,
    b: np.ndarray,
    M: spla.LinearOperator | None,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    callback,
    *,
    deadline_s: float = 0.0,
    t0: float | None = None,
) -> tuple[np.ndarray, int]:
    """Run GMRES, aborting mid-solve if a wall-time budget is exceeded.

    ``deadline_s > 0`` checks the elapsed time (from ``t0``) on every GMRES
    iteration (``callback_type="pr_norm"`` fires per-iteration, not just per
    restart cycle) and raises ``GmresDeadlineExceeded`` from inside the
    callback. scipy does not catch callback exceptions, so this aborts the
    single GMRES call immediately instead of letting it grind to
    ``maxiter`` -- the previous "check only between Newton iterations" scheme
    let one pathological GMRES call run ~200s past a 14s budget.
    """
    def wrapped(pr_norm: float) -> None:
        if deadline_s > 0.0 and t0 is not None and time.perf_counter() - t0 > deadline_s:
            raise GmresDeadlineExceeded(
                f"GMRES exceeded {deadline_s:.1f}s budget mid-solve"
            )
        callback(pr_norm)

    try:
        return spla.gmres(
            A,
            b,
            M=M,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            callback=wrapped,
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
            callback=wrapped,
        )


def fold_power(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    *,
    ds: float = 0.1,
    max_steps: int = 300,
    rescale_every: int = 0,
) -> float | None:
    """Locate the harmonic-balance fold in the source scale ``lambda``.

    ``problem`` is built at a reference pump current; ``lambda`` scales that
    current linearly, so the fold ``lambda`` maps directly to a fold current
    (``lambda_fold * reference_current``) and thus a fold power. Runs
    pseudo-arclength from ``lambda=0`` and returns the ``lambda`` at the first
    turning point (sign change of ``lambda_dot``), or ``None`` if none is found
    within ``max_steps`` (branch has no fold in range).

    ``rescale_every`` (default ``0``, disabled) forwards to
    ``solve_arclength``'s periodic metric rescale
    (docs/development/arclength_fold_resolution_plan.md Phase 1) -- a
    mistuned metric can make the corrector die (``minimum_step``) before
    ever reaching a real fold, which reads identically to "no fold in range"
    without this.
    """
    _X, _lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=ds, max_steps=max_steps,
        target_lam=float("inf"),  # never "reach" target -> run until fold/steps
        rescale_every=rescale_every,
    )
    fold = info.get("fold_lambda")
    return float(fold) if fold is not None else None


@dataclass
class BranchPoint:
    """One accepted pseudo-arclength point on a physical-mu branch.

    ``s`` is a step-index proxy (cumulative ``step_size``), not a
    metric-exact arclength -- it only orders points within one trace and is
    not comparable across a metric rescale event (see
    docs/development/fold_plan.md Section 4; a fully metric-consistent ``s``
    is deferred to Milestone C's adaptive-step work). ``t_mu`` is the raw
    ``lam_dot`` component ``solve_arclength_mu`` forwards from its inner
    ``solve_arclength`` call, in the inner lambda-parametrization -- its
    *sign* is exactly the mu-tangent's sign (the wrapper's ``k`` is a
    positive constant), which is all fold detection needs; its magnitude is
    not separately mu-normalized.
    """

    s: float
    mu: float
    X: np.ndarray
    t_mu: float
    step_size: float


@dataclass
class ContinuationBranch:
    """A traced solution branch in a fixed physical-current coordinate ``mu``.

    ``points`` are accepted continuation states in traversal order -- not
    necessarily monotonic in ``mu`` (a folded branch reverses direction
    partway through). ``i_ref`` is the physical reference current that
    defines this branch's ``mu = I_physical / i_ref``; it is fixed for the
    whole branch and independent of any particular map target (plan Section
    1/23) -- the same branch can be queried for any number of target
    currents via :func:`resolve_target_on_branch` instead of each target
    re-running its own ``lambda: 0 -> 1`` continuation problem.
    """

    i_ref: float
    points: list[BranchPoint]
    info: dict


def trace_branch(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    *,
    i_ref: float,
    X0: np.ndarray | None = None,
    mu0: float = 0.0,
    mu_max: float = 1.0,
    ds: float = 0.1,
    max_steps: int = 300,
    newton_max: int = 12,
    max_wall_s: float = 0.0,
    rescale_every: int = 0,
    max_steps_after_fold: int | None = None,
    step_control: str = "legacy",
    newton_target: int = 4,
    step_growth_exponent: float = 0.5,
    step_growth_clamp: tuple[float, float] = (0.5, 1.5),
    ds_max: float | None = None,
    refine_fold: bool = False,
    fold_t_tol: float = 1e-6,
    fold_mu_tol: float | None = None,
    fold_refine_max_iter: int = 30,
    step_telemetry: Callable[[dict], None] | None = None,
) -> ContinuationBranch:
    """Trace one physical-``mu`` branch, storing every accepted point.

    Milestone B (docs/development/fold_plan.md Section 20/23): the branch
    becomes the primary numerical object. This calls ``solve_arclength_mu``
    exactly once (target ``mu_max``, or the branch runs until a fold/step
    budget if ``mu_max`` is never reached) and collects each accepted point
    via the existing ``on_step`` hook -- no change to the corrector itself.

    ``step_control``/``refine_fold`` and their tuning parameters (Milestone
    C/D) forward to ``solve_arclength_mu`` unchanged; both default to the
    pre-Milestone-C/D behavior (``"legacy"`` step growth, no fold
    refinement), so an un-opted-in caller sees no change.

    ``step_telemetry`` (Milestone G2, default ``None``): passed straight
    through to ``solve_arclength_mu`` (already mu-unit-converted there) --
    fires once per accepted step with residuals/Newton/GMRES/tangent detail
    that ``BranchPoint`` does not carry, for callers building a per-step
    traversal diagnostic (docs/development/fold_plan.md Milestone G2)
    without needing to correlate against ``branch.points`` by index.
    """
    points: list[BranchPoint] = []
    s_cum = 0.0

    def _on_step(
        Xc: np.ndarray, muc: float, step_size: float,
        Xdot: np.ndarray, mu_dot: float, state_scale: float,
    ) -> None:
        nonlocal s_cum
        s_cum += step_size
        points.append(BranchPoint(
            s=s_cum, mu=muc, X=np.array(Xc, copy=True),
            t_mu=mu_dot, step_size=step_size,
        ))

    X, mu, info = solver.solve_arclength_mu(
        problem, X0, mu0, i_ref=i_ref, target_mu=mu_max, ds=ds,
        max_steps=max_steps, newton_max=newton_max, max_wall_s=max_wall_s,
        on_step=_on_step, rescale_every=rescale_every,
        max_steps_after_fold=max_steps_after_fold,
        step_control=step_control,
        newton_target=newton_target,
        step_growth_exponent=step_growth_exponent,
        step_growth_clamp=step_growth_clamp,
        ds_max=ds_max,
        refine_fold=refine_fold,
        fold_t_tol=fold_t_tol,
        fold_mu_tol=fold_mu_tol,
        fold_refine_max_iter=fold_refine_max_iter,
        step_telemetry=step_telemetry,
    )
    if info.get("reached_target"):
        points.append(BranchPoint(
            s=s_cum, mu=mu, X=np.array(X, copy=True),
            t_mu=points[-1].t_mu if points else 0.0, step_size=0.0,
        ))
    return ContinuationBranch(i_ref=i_ref, points=points, info=info)


def bracket_target(
    branch: ContinuationBranch, mu_target: float,
) -> tuple[BranchPoint, BranchPoint] | None:
    """Return the two consecutive accepted points bracketing ``mu_target``.

    Scans in traversal order, so a target crossed twice (past a fold, once
    outgoing and once on the returning segment) returns the *first*
    crossing. Returns ``None`` if ``mu_target`` is not bracketed anywhere on
    the traced branch (e.g. beyond a fold with no returning branch traced).
    """
    pts = branch.points
    for a, b in zip(pts, pts[1:]):
        if (a.mu - mu_target) * (b.mu - mu_target) <= 0.0 and a.mu != b.mu:
            return a, b
    return None


def resolve_target_on_branch(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    branch: ContinuationBranch,
    mu_target: float,
) -> tuple[np.ndarray, StepReport] | None:
    """Bracket ``mu_target`` on ``branch`` and run the exact fixed-mu Newton solve.

    Plan Section 19: PALC discovers branch geometry; ordinary Newton (via the
    existing ``solve_one``, unmodified) gives the exact requested map sample
    from a linearly-interpolated warm guess. Returns ``None`` if
    ``mu_target`` is not bracketed on this branch -- callers should treat
    that as "beyond this branch's traced extent", not as a solver failure.
    """
    bracket = bracket_target(branch, mu_target)
    if bracket is None:
        return None
    a, b = bracket
    alpha = (mu_target - a.mu) / (b.mu - a.mu)
    X_guess = (1.0 - alpha) * a.X + alpha * b.X
    k = branch.i_ref / problem.pump_current_a
    return solver.solve_one(problem, X_guess, mu_target * k)


# Milestone E (fold_plan.md): production reachability semantics.
#
# ``solve_arclength``/``solve_arclength_mu`` never intentionally stop at a
# fold -- they only ever fail to reach ``target_mu`` via a genuine corrector
# failure (minimum_step), the step/deadline budget, or a singular Jacobian.
# ``info["fold_lambda"]``/``info["fold_refined"]`` record only the FIRST
# tangent-sign flip on a trace.
#
# The functions below detect and classify what these tangent-sign flips mean,
# using the multistability probe from the Milestone D.5 validation campaign
# (docs/development/fold_plan.md, memory fold-plan-ad-validation-narrow-feature
# / fold-plan-d5-e-multistability-gate): three fixed-mu Newton solves seeded
# from the segment before, inside, and after a candidate pair.
#
# Naming is deliberately epistemic, revised after D.5's own measurement on
# designs/ipm_2c_fixed at 7.9 GHz (mu~0.5253): a detected second tangent-sign
# flip is NOT by itself evidence of a genuine bifurcation, let alone a
# specifically degenerate one (corank/transversality failure -- untested
# here, Milestone F's scope). There, the "inside" and "after" segment seeds
# converged to the SAME root (1.2e-9 relative apart) despite a clean second
# sign flip, while only the "before" seed differed materially (0.44%) --
# evidence the region is numerically UNRESOLVED at the tested step sizes, not
# evidence of a specific mathematical degeneracy. The intended hierarchy
# going into Milestone F is:
#
#     tangent sign flip -> fold candidate -> validated bifurcation event
#     -> classified event (SIMPLE_FOLD / BRANCH_POINT / HIGHER_DEGENERACY / ...)
#
# not "sign flip -> fold". Every ``FoldCandidate`` below carries
# ``kind``/``validation`` fields as placeholders for that later promotion;
# nothing in Milestone E promotes them.

FOLD_TOPOLOGY_LOCAL_PAIR = "LOCAL_FOLD_PAIR"
FOLD_TOPOLOGY_UNRESOLVED = "UNRESOLVED_NEAR_FOLD"

FOLD_CANDIDATE_KIND_TANGENT_SIGN_FLIP = "TANGENT_SIGN_FLIP"
FOLD_CANDIDATE_VALIDATION_UNVALIDATED = "UNVALIDATED"


@dataclass
class FoldCandidate:
    """One tangent-sign change (``t_mu`` crossing zero) on a traced branch.

    Named "candidate", not "fold": a sign change alone does not establish a
    genuine bifurcation (Milestone D.5). ``kind``/``validation`` are
    placeholders for Milestone F's later classification hierarchy; nothing
    here promotes them.

    ``index`` is the position in ``branch.points`` of the point AFTER the
    sign change (``points[index - 1]`` is the point before).
    """

    index: int
    mu_before: float
    mu_after: float
    s_before: float
    s_after: float
    kind: str = FOLD_CANDIDATE_KIND_TANGENT_SIGN_FLIP
    validation: str = FOLD_CANDIDATE_VALIDATION_UNVALIDATED


def find_fold_candidates(branch: ContinuationBranch) -> list[FoldCandidate]:
    """Every ``t_mu`` sign change on ``branch``, in traversal order.

    Detected purely from consecutive ``BranchPoint.t_mu`` signs already
    stored on the branch -- no new continuation solves. Recovers every
    candidate on the trace, unlike ``solve_arclength``'s own
    ``info["fold_lambda"]``, which only ever records the first.
    """
    candidates: list[FoldCandidate] = []
    pts = branch.points
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if (a.t_mu >= 0) != (b.t_mu >= 0):
            candidates.append(FoldCandidate(
                index=i, mu_before=a.mu, mu_after=b.mu, s_before=a.s, s_after=b.s,
            ))
    return candidates


def classify_fold_pair(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    branch: ContinuationBranch,
    candidate_a: FoldCandidate,
    candidate_b: FoldCandidate,
    *,
    newton_max: int = 15,
    distinctness_tol: float = 1e-6,
) -> dict:
    """Multistability-gated classification of one candidate local fold pair.

    ``candidate_a``/``candidate_b`` bracket three branch segments (before/
    inside/after) -- consecutive candidates always alternate sign by
    construction, so this works for either orientation (a middle dip between
    two higher segments, or a middle bump between two lower ones). The
    before/inside/after mu-overlap is the general triple-interval
    intersection ``[max(min_i), min(max_i)]`` over the three segments, not a
    shape-specific formula.

    Picks ``mu_star`` inside that overlap and runs ordinary fixed-mu Newton
    (``solve_one``) from a seed on each segment. Returns
    ``status=FOLD_TOPOLOGY_LOCAL_PAIR`` only if all three solves converge AND
    every pairwise relative state distance exceeds ``distinctness_tol``
    (three materially different coexisting solutions);
    ``status=FOLD_TOPOLOGY_UNRESOLVED`` otherwise, with a ``reason`` -- this
    is a report of insufficient/inconsistent evidence, not a claim that the
    event is mathematically degenerate (Milestone F territory).
    """
    pts = branch.points
    before_pts = pts[:candidate_a.index]
    inside_pts = pts[candidate_a.index:candidate_b.index]
    after_pts = pts[candidate_b.index:]
    if not before_pts or not inside_pts or not after_pts:
        return {"status": FOLD_TOPOLOGY_UNRESOLVED, "reason": "insufficient points around the pair"}

    segments = (before_pts, inside_pts, after_pts)
    mu_lo = max(min(p.mu for p in seg) for seg in segments)
    mu_hi = min(max(p.mu for p in seg) for seg in segments)
    if mu_hi <= mu_lo:
        return {"status": FOLD_TOPOLOGY_UNRESOLVED, "reason": f"no mu overlap (lo={mu_lo}, hi={mu_hi})"}
    mu_star = 0.5 * (mu_lo + mu_hi)

    def nearest(points_: list[BranchPoint]) -> BranchPoint:
        return min(points_, key=lambda p: abs(p.mu - mu_star))

    seeds = {"before": nearest(before_pts), "inside": nearest(inside_pts), "after": nearest(after_pts)}
    k = branch.i_ref / problem.pump_current_a
    roots: dict[str, dict] = {}
    for label, seed in seeds.items():
        X_star, report = solver.solve_one(problem, seed.X, mu_star * k)
        roots[label] = {
            "converged": report.converged, "coeff_rel": report.coeff_rel,
            "X": X_star, "norm": float(np.linalg.norm(X_star)), "seed_mu": seed.mu,
        }

    def _public(r: dict) -> dict:
        return {k_: v for k_, v in r.items() if k_ != "X"}

    if not all(r["converged"] for r in roots.values()):
        return {
            "status": FOLD_TOPOLOGY_UNRESOLVED, "mu_star": mu_star,
            "reason": "a segment seed failed to converge",
            "roots": {k_: _public(v) for k_, v in roots.items()},
        }

    labels = list(roots)
    pairwise: dict[str, float] = {}
    distinct = True
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            d = float(np.linalg.norm(roots[li]["X"] - roots[lj]["X"])) / max(
                float(np.linalg.norm(roots[li]["X"])), 1e-300,
            )
            pairwise[f"{li}_{lj}"] = d
            if d < distinctness_tol:
                distinct = False

    status = FOLD_TOPOLOGY_LOCAL_PAIR if distinct else FOLD_TOPOLOGY_UNRESOLVED
    return {
        "status": status, "mu_star": mu_star, "mu_lo": mu_lo, "mu_hi": mu_hi,
        "pairwise": pairwise, "roots": {k_: _public(v) for k_, v in roots.items()},
    }


def classify_adjacent_fold_candidate_pairs(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    branch: ContinuationBranch,
    *,
    newton_max: int = 15,
    distinctness_tol: float = 1e-6,
) -> list[dict]:
    """Classify every ADJACENT pair of fold candidates on ``branch``.

    Tests ``(E0,E1), (E1,E2), (E2,E3), ...`` -- every consecutive
    combination, not a disjoint two-at-a-time grouping
    (``(E0,E1), (E2,E3), ...``). Disjoint pairing can silently miss the
    meaningful pair: if the real sequence is [noise, fold A, fold B],
    disjoint pairing tests (noise, foldA) and leaves foldB trailing
    unclassified, when (foldA, foldB) is the pair that actually matters.
    Adjacent pairing surfaces every consecutive combination and leaves the
    judgment of which candidates are "real" to the caller (or Milestone F).

    Each result carries ``candidate_a_id``/``candidate_b_id`` (0-indexed
    position in ``find_fold_candidates(branch)``, i.e. E0/E1/...) so a
    result can be traced back to its two candidates.
    """
    candidates = find_fold_candidates(branch)
    out: list[dict] = []
    for i in range(len(candidates) - 1):
        candidate_a, candidate_b = candidates[i], candidates[i + 1]
        result = classify_fold_pair(
            solver, problem, branch, candidate_a, candidate_b,
            newton_max=newton_max, distinctness_tol=distinctness_tol,
        )
        result["candidate_a_id"] = i
        result["candidate_b_id"] = i + 1
        result["candidate_a_mu"] = 0.5 * (candidate_a.mu_before + candidate_a.mu_after)
        result["candidate_b_mu"] = 0.5 * (candidate_b.mu_before + candidate_b.mu_after)
        out.append(result)
    return out
