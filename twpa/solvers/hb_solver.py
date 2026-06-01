"""
twpa.solvers.hb_solver
======================

Dense Newton solver and shared harmonic-balance solver reports.

This module is intentionally the first solver layer. It is not the industrial
20,000-cell solver. It is the correctness/debugging solver used for:

    - one nonlinear element,
    - one-node HB circuits,
    - small distributed ladders,
    - comparison against future Newton-Krylov and block-banded solvers.

Industrial-scale simulation will later use:

    twpa.solvers.newton_krylov
    twpa.solvers.block_banded
    twpa.solvers.preconditioners

but those must match this dense reference solver on small problems.

Design principles
-----------------
1. Residual functions may operate on structured PyTrees.
2. Newton internally solves real systems.
3. Complex unknowns are packed as [real, imag].
4. Every failure mode returns a report.
5. Backtracking damping is explicit.
6. Solver does not know circuit physics.
7. Solver can handle residual dimension equal to unknown dimension.
8. Non-square residuals are solved by least squares.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp

from twpa.core.harmonics import complex_tree_to_real_vector
from twpa.core.params import SolverBackend, SolverConfig


ArrayLike = Any
PyTree = Any
ResidualFn = Callable[[PyTree], PyTree]


# ---------------------------------------------------------------------------
# Enums / config
# ---------------------------------------------------------------------------

class SolverStatus(str, Enum):
    """Solver termination status."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    LINE_SEARCH_FAILED = "line_search_failed"
    NONFINITE_RESIDUAL = "nonfinite_residual"
    NONFINITE_STEP = "nonfinite_step"
    LINEAR_SOLVE_FAILED = "linear_solve_failed"
    BAD_INITIAL_RESIDUAL = "bad_initial_residual"
    EXCEPTION = "exception"


class LinearSolveMethod(str, Enum):
    """Dense Newton linear solve method."""

    AUTO = "auto"
    SOLVE = "solve"
    LSTSQ = "lstsq"
    PINV = "pinv"


class NormKind(str, Enum):
    """Residual/step norm kind."""

    L2 = "l2"
    INF = "inf"


@dataclass(frozen=True)
class DenseNewtonConfig:
    """
    Dense Newton configuration.

    Parameters
    ----------
    max_iter:
        Maximum Newton iterations.
    abs_tol:
        Absolute residual norm tolerance.
    rel_tol:
        Relative residual norm tolerance relative to the initial norm.
    step_tol:
        Absolute step norm tolerance.
    damping_initial:
        Initial trial damping.
    damping_min:
        Minimum allowed damping.
    damping_backtracking_factor:
        Factor in (0, 1) applied during backtracking.
    max_backtracking_steps:
        Maximum line-search backtracking attempts.
    regularization:
        Optional diagonal regularization for square Jacobian solves.
    linear_solve_method:
        Dense linear-solve method.
    norm:
        Norm used for convergence and line search.
    accept_residual_increase:
        If true, accepts full Newton step even if residual increases. This is
        useful only for debugging. Keep false for production.
    jit_residual:
        If true, jit-compile the real residual function.
    jit_jacobian:
        If true, jit-compile the Jacobian function.
    fail_on_nonconvergence:
        If true, raise RuntimeError on non-convergence after producing report.
    verbose:
        Whether scripts may print iteration logs.
    """

    max_iter: int = 50
    abs_tol: float = 1e-10
    rel_tol: float = 1e-10
    step_tol: float = 1e-12
    damping_initial: float = 1.0
    damping_min: float = 1e-6
    damping_backtracking_factor: float = 0.5
    max_backtracking_steps: int = 20
    regularization: float = 0.0
    linear_solve_method: LinearSolveMethod = LinearSolveMethod.AUTO
    norm: NormKind = NormKind.L2
    accept_residual_increase: bool = False
    jit_residual: bool = False
    jit_jacobian: bool = False
    fail_on_nonconvergence: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear_solve_method", LinearSolveMethod(self.linear_solve_method))
        object.__setattr__(self, "norm", NormKind(self.norm))

        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        object.__setattr__(self, "max_iter", int(self.max_iter))

        for name in ["abs_tol", "rel_tol", "step_tol", "damping_initial", "damping_min"]:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

        if not (0.0 < float(self.damping_backtracking_factor) < 1.0):
            raise ValueError("damping_backtracking_factor must be in (0, 1)")

        if int(self.max_backtracking_steps) < 0:
            raise ValueError("max_backtracking_steps must be non-negative")
        object.__setattr__(
            self,
            "max_backtracking_steps",
            int(self.max_backtracking_steps),
        )

        if float(self.regularization) < 0.0:
            raise ValueError("regularization must be non-negative")

    @classmethod
    def from_solver_config(cls, config: SolverConfig) -> "DenseNewtonConfig":
        """
        Build DenseNewtonConfig from the shared core SolverConfig.
        """
        return cls(
            max_iter=config.max_iter,
            abs_tol=config.abs_tol,
            rel_tol=config.rel_tol,
            step_tol=config.step_tol,
            damping_initial=config.damping_initial,
            damping_min=config.damping_min,
            damping_backtracking_factor=config.damping_backtracking_factor,
            max_backtracking_steps=config.max_backtracking_steps,
            regularization=config.regularization,
            linear_solve_method=LinearSolveMethod.AUTO,
            fail_on_nonconvergence=config.fail_on_nonconvergence,
            verbose=config.verbose,
        )

    def with_updates(self, **kwargs: Any) -> "DenseNewtonConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "step_tol": self.step_tol,
            "damping_initial": self.damping_initial,
            "damping_min": self.damping_min,
            "damping_backtracking_factor": self.damping_backtracking_factor,
            "max_backtracking_steps": self.max_backtracking_steps,
            "regularization": self.regularization,
            "linear_solve_method": self.linear_solve_method.value,
            "norm": self.norm.value,
            "accept_residual_increase": self.accept_residual_increase,
            "jit_residual": self.jit_residual,
            "jit_jacobian": self.jit_jacobian,
            "fail_on_nonconvergence": self.fail_on_nonconvergence,
            "verbose": self.verbose,
        }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewtonIterationRecord:
    """
    One Newton iteration diagnostic record.
    """

    iteration: int
    residual_norm: float
    residual_norm_relative: float
    step_norm: float | None
    damping: float | None
    accepted: bool
    backtracking_steps: int
    linear_solve_method: str | None
    linear_residual_norm: float | None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "residual_norm": self.residual_norm,
            "residual_norm_relative": self.residual_norm_relative,
            "step_norm": self.step_norm,
            "damping": self.damping,
            "accepted": self.accepted,
            "backtracking_steps": self.backtracking_steps,
            "linear_solve_method": self.linear_solve_method,
            "linear_residual_norm": self.linear_residual_norm,
            "message": self.message,
        }


@dataclass(frozen=True)
class HBSolverReport:
    """
    Solver report returned by all HB/Newton solvers.
    """

    status: SolverStatus
    converged: bool
    iterations: int
    initial_residual_norm: float
    final_residual_norm: float
    final_relative_residual_norm: float
    final_step_norm: float | None
    unknown_size: int
    residual_size: int
    config: Mapping[str, Any]
    records: tuple[NewtonIterationRecord, ...]
    metadata: Mapping[str, Any] | None = None
    exception_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": self.initial_residual_norm,
            "final_residual_norm": self.final_residual_norm,
            "final_relative_residual_norm": self.final_relative_residual_norm,
            "final_step_norm": self.final_step_norm,
            "unknown_size": self.unknown_size,
            "residual_size": self.residual_size,
            "config": dict(self.config),
            "records": [r.to_dict() for r in self.records],
            "metadata": dict(self.metadata or {}),
            "exception_message": self.exception_message,
        }

    def summary_line(self) -> str:
        return (
            f"{self.status.value}: converged={self.converged}, "
            f"iters={self.iterations}, "
            f"res0={self.initial_residual_norm:.3e}, "
            f"res={self.final_residual_norm:.3e}, "
            f"rel={self.final_relative_residual_norm:.3e}"
        )


@dataclass(frozen=True)
class HBSolverResult:
    """
    Solver result.

    Attributes
    ----------
    x:
        Solution PyTree with same structure as initial guess.
    residual:
        Residual PyTree evaluated at x.
    report:
        Solver report.
    """

    x: PyTree
    residual: PyTree
    report: HBSolverReport

    @property
    def converged(self) -> bool:
        return self.report.converged

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report.to_dict(),
        }


# ---------------------------------------------------------------------------
# Norms and packing
# ---------------------------------------------------------------------------

def real_norm(x: ArrayLike, kind: NormKind | str = NormKind.L2) -> jax.Array:
    """
    Compute vector norm.
    """
    kind = NormKind(kind)
    xx = jnp.asarray(x, dtype=jnp.float64)
    if kind == NormKind.L2:
        return jnp.linalg.norm(xx)
    if kind == NormKind.INF:
        return jnp.max(jnp.abs(xx))
    raise ValueError(f"Unsupported norm kind {kind}")


def residual_relative_norm(norm_value: ArrayLike, initial_norm: ArrayLike) -> jax.Array:
    """
    Safe residual relative norm.
    """
    return jnp.asarray(norm_value) / jnp.maximum(jnp.asarray(initial_norm), 1e-300)


def pack_unknown_tree(x: PyTree) -> tuple[jax.Array, Callable[[ArrayLike], PyTree]]:
    """
    Pack an unknown PyTree into a real vector and return an unravel function.
    """
    return complex_tree_to_real_vector(x)


def pack_residual_tree(r: PyTree) -> jax.Array:
    """
    Pack a residual PyTree into a real vector.
    """
    vec, _ = complex_tree_to_real_vector(r)
    return vec


def make_real_residual_function(
    residual_fn: ResidualFn,
    unravel_x: Callable[[ArrayLike], PyTree],
) -> Callable[[jax.Array], jax.Array]:
    """
    Convert a structured residual function into real-vector form.
    """

    def real_residual(x_vec: jax.Array) -> jax.Array:
        x_tree = unravel_x(x_vec)
        r_tree = residual_fn(x_tree)
        return pack_residual_tree(r_tree)

    return real_residual


def has_nonfinite(x: ArrayLike) -> bool:
    """
    Eager nonfinite check for a vector/array.
    """
    arr = jnp.asarray(x)
    return bool(jnp.any(jnp.isnan(arr)) or jnp.any(jnp.isinf(arr)))


# ---------------------------------------------------------------------------
# Dense linear solves
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DenseLinearSolveResult:
    """
    Result of a dense Newton linear solve.
    """

    step: jax.Array
    method: LinearSolveMethod
    linear_residual_norm: float
    success: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "linear_residual_norm": self.linear_residual_norm,
            "success": self.success,
            "message": self.message,
        }


def dense_linear_solve(
    jacobian: ArrayLike,
    rhs: ArrayLike,
    *,
    method: LinearSolveMethod | str = LinearSolveMethod.AUTO,
    regularization: float = 0.0,
) -> DenseLinearSolveResult:
    """
    Solve J step = rhs using dense methods.

    If J is square, AUTO tries solve first. If that fails or J is rectangular,
    AUTO uses least squares.
    """
    J = jnp.asarray(jacobian, dtype=jnp.float64)
    b = jnp.asarray(rhs, dtype=jnp.float64)
    method = LinearSolveMethod(method)

    if J.ndim != 2:
        raise ValueError(f"jacobian must be 2D, got {J.shape}")
    if b.ndim != 1 or b.shape[0] != J.shape[0]:
        raise ValueError(f"rhs must have shape ({J.shape[0]},), got {b.shape}")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")

    n_res, n_unknown = int(J.shape[0]), int(J.shape[1])
    square = n_res == n_unknown

    chosen = method
    if method == LinearSolveMethod.AUTO:
        chosen = LinearSolveMethod.SOLVE if square else LinearSolveMethod.LSTSQ

    try:
        if chosen == LinearSolveMethod.SOLVE:
            if not square:
                raise ValueError("SOLVE requires square Jacobian")
            J_eff = J
            if regularization > 0.0:
                J_eff = J + regularization * jnp.eye(n_unknown, dtype=J.dtype)
            step = jnp.linalg.solve(J_eff, b)

        elif chosen == LinearSolveMethod.LSTSQ:
            if regularization > 0.0:
                # Tikhonov-regularized least squares:
                # [J        ] step ≈ [b]
                # [sqrt(lam)I]        [0]
                J_aug = jnp.concatenate(
                    [
                        J,
                        jnp.sqrt(regularization) * jnp.eye(n_unknown, dtype=J.dtype),
                    ],
                    axis=0,
                )
                b_aug = jnp.concatenate([b, jnp.zeros((n_unknown,), dtype=b.dtype)])
                step = jnp.linalg.lstsq(J_aug, b_aug, rcond=None)[0]
            else:
                step = jnp.linalg.lstsq(J, b, rcond=None)[0]

        elif chosen == LinearSolveMethod.PINV:
            step = jnp.linalg.pinv(J) @ b

        else:
            raise ValueError(f"Unsupported linear solve method {chosen}")

        lin_res = J @ step - b
        lin_norm = float(jnp.linalg.norm(lin_res))
        success = not has_nonfinite(step)

        return DenseLinearSolveResult(
            step=step,
            method=chosen,
            linear_residual_norm=lin_norm,
            success=bool(success),
            message="ok" if success else "nonfinite step",
        )

    except Exception as exc:
        if method == LinearSolveMethod.AUTO and chosen == LinearSolveMethod.SOLVE:
            # Fall back to least squares.
            try:
                step = jnp.linalg.lstsq(J, b, rcond=None)[0]
                lin_res = J @ step - b
                return DenseLinearSolveResult(
                    step=step,
                    method=LinearSolveMethod.LSTSQ,
                    linear_residual_norm=float(jnp.linalg.norm(lin_res)),
                    success=not has_nonfinite(step),
                    message=f"solve failed; fell back to lstsq: {exc}",
                )
            except Exception as exc2:
                return DenseLinearSolveResult(
                    step=jnp.full((n_unknown,), jnp.nan),
                    method=LinearSolveMethod.LSTSQ,
                    linear_residual_norm=float("nan"),
                    success=False,
                    message=f"solve and lstsq failed: {exc}; {exc2}",
                )

        return DenseLinearSolveResult(
            step=jnp.full((n_unknown,), jnp.nan),
            method=chosen,
            linear_residual_norm=float("nan"),
            success=False,
            message=str(exc),
        )


# ---------------------------------------------------------------------------
# Line search
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineSearchResult:
    """
    Backtracking line-search result.
    """

    x_new: jax.Array
    r_new: jax.Array
    residual_norm_new: float
    damping: float
    accepted: bool
    backtracking_steps: int
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_norm_new": self.residual_norm_new,
            "damping": self.damping,
            "accepted": self.accepted,
            "backtracking_steps": self.backtracking_steps,
            "message": self.message,
        }


def backtracking_line_search(
    real_residual_fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    r: jax.Array,
    step: jax.Array,
    *,
    current_norm: float,
    config: DenseNewtonConfig,
) -> LineSearchResult:
    """
    Backtracking line search for Newton step.

    Tries x + damping * step.
    """
    damping = float(config.damping_initial)
    best_x = x
    best_r = r
    best_norm = float(current_norm)
    best_message = "not attempted"

    for bt in range(config.max_backtracking_steps + 1):
        x_trial = x + damping * step
        r_trial = real_residual_fn(x_trial)

        if has_nonfinite(r_trial):
            accepted = False
            trial_norm = float("inf")
            message = "nonfinite residual"
        else:
            trial_norm_j = real_norm(r_trial, config.norm)
            trial_norm = float(trial_norm_j)
            accepted = (
                trial_norm < current_norm
                or config.accept_residual_increase
                or trial_norm <= config.abs_tol
            )
            message = "accepted" if accepted else "residual did not decrease"

        if trial_norm < best_norm:
            best_x = x_trial
            best_r = r_trial
            best_norm = trial_norm
            best_message = "best decreasing trial"

        if accepted:
            return LineSearchResult(
                x_new=x_trial,
                r_new=r_trial,
                residual_norm_new=trial_norm,
                damping=damping,
                accepted=True,
                backtracking_steps=bt,
                message=message,
            )

        damping *= config.damping_backtracking_factor
        if damping < config.damping_min:
            break

    return LineSearchResult(
        x_new=best_x,
        r_new=best_r,
        residual_norm_new=best_norm,
        damping=damping,
        accepted=False,
        backtracking_steps=config.max_backtracking_steps,
        message=f"line search failed; {best_message}",
    )


# ---------------------------------------------------------------------------
# Main dense Newton solver
# ---------------------------------------------------------------------------

def solve_dense_newton(
    residual_fn: ResidualFn,
    x0: PyTree,
    *,
    config: DenseNewtonConfig | SolverConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HBSolverResult:
    """
    Solve residual_fn(x) = 0 using dense real Newton.

    Parameters
    ----------
    residual_fn:
        Function accepting a PyTree x and returning a residual PyTree.
    x0:
        Initial unknown PyTree.
    config:
        DenseNewtonConfig or shared SolverConfig.
    metadata:
        Extra report metadata.

    Returns
    -------
    HBSolverResult
    """
    if config is None:
        cfg = DenseNewtonConfig()
    elif isinstance(config, SolverConfig):
        cfg = DenseNewtonConfig.from_solver_config(config)
    else:
        cfg = config

    x_vec0, unravel_x = pack_unknown_tree(x0)
    real_residual_fn = make_real_residual_function(residual_fn, unravel_x)

    if cfg.jit_residual:
        real_residual_fn_eval = jax.jit(real_residual_fn)
    else:
        real_residual_fn_eval = real_residual_fn

    jac_fn = jax.jacfwd(real_residual_fn)
    if cfg.jit_jacobian:
        jac_fn_eval = jax.jit(jac_fn)
    else:
        jac_fn_eval = jac_fn

    records: list[NewtonIterationRecord] = []

    try:
        x = jnp.asarray(x_vec0, dtype=jnp.float64)
        r = real_residual_fn_eval(x)

        unknown_size = int(x.shape[0])
        residual_size = int(r.shape[0])

        if has_nonfinite(r):
            report = HBSolverReport(
                status=SolverStatus.BAD_INITIAL_RESIDUAL,
                converged=False,
                iterations=0,
                initial_residual_norm=float("inf"),
                final_residual_norm=float("inf"),
                final_relative_residual_norm=float("inf"),
                final_step_norm=None,
                unknown_size=unknown_size,
                residual_size=residual_size,
                config=cfg.to_dict(),
                records=tuple(records),
                metadata=dict(metadata or {}),
                exception_message=None,
            )
            return HBSolverResult(
                x=unravel_x(x),
                residual=residual_fn(unravel_x(x)),
                report=report,
            )

        r_norm = float(real_norm(r, cfg.norm))
        r0_norm = max(r_norm, 1e-300)
        rel_norm = float(residual_relative_norm(r_norm, r0_norm))

        records.append(
            NewtonIterationRecord(
                iteration=0,
                residual_norm=r_norm,
                residual_norm_relative=rel_norm,
                step_norm=None,
                damping=None,
                accepted=True,
                backtracking_steps=0,
                linear_solve_method=None,
                linear_residual_norm=None,
                message="initial residual",
            )
        )

        if r_norm <= cfg.abs_tol or rel_norm <= cfg.rel_tol:
            x_tree = unravel_x(x)
            report = HBSolverReport(
                status=SolverStatus.CONVERGED,
                converged=True,
                iterations=0,
                initial_residual_norm=r0_norm,
                final_residual_norm=r_norm,
                final_relative_residual_norm=rel_norm,
                final_step_norm=None,
                unknown_size=unknown_size,
                residual_size=residual_size,
                config=cfg.to_dict(),
                records=tuple(records),
                metadata=dict(metadata or {}),
            )
            return HBSolverResult(
                x=x_tree,
                residual=residual_fn(x_tree),
                report=report,
            )

        final_step_norm: float | None = None
        status = SolverStatus.MAX_ITERATIONS

        for it in range(1, cfg.max_iter + 1):
            J = jac_fn_eval(x)

            if has_nonfinite(J):
                status = SolverStatus.NONFINITE_RESIDUAL
                records.append(
                    NewtonIterationRecord(
                        iteration=it,
                        residual_norm=r_norm,
                        residual_norm_relative=rel_norm,
                        step_norm=None,
                        damping=None,
                        accepted=False,
                        backtracking_steps=0,
                        linear_solve_method=None,
                        linear_residual_norm=None,
                        message="nonfinite Jacobian",
                    )
                )
                break

            linear = dense_linear_solve(
                J,
                -r,
                method=cfg.linear_solve_method,
                regularization=cfg.regularization,
            )

            if not linear.success:
                status = SolverStatus.LINEAR_SOLVE_FAILED
                records.append(
                    NewtonIterationRecord(
                        iteration=it,
                        residual_norm=r_norm,
                        residual_norm_relative=rel_norm,
                        step_norm=None,
                        damping=None,
                        accepted=False,
                        backtracking_steps=0,
                        linear_solve_method=linear.method.value,
                        linear_residual_norm=linear.linear_residual_norm,
                        message=f"linear solve failed: {linear.message}",
                    )
                )
                break

            step = linear.step
            if has_nonfinite(step):
                status = SolverStatus.NONFINITE_STEP
                records.append(
                    NewtonIterationRecord(
                        iteration=it,
                        residual_norm=r_norm,
                        residual_norm_relative=rel_norm,
                        step_norm=None,
                        damping=None,
                        accepted=False,
                        backtracking_steps=0,
                        linear_solve_method=linear.method.value,
                        linear_residual_norm=linear.linear_residual_norm,
                        message="nonfinite step",
                    )
                )
                break

            step_norm = float(real_norm(step, cfg.norm))
            final_step_norm = step_norm

            if step_norm <= cfg.step_tol:
                # Try evaluating at current point. If residual is still too big,
                # this is stagnation, not convergence.
                if r_norm <= cfg.abs_tol or rel_norm <= cfg.rel_tol:
                    status = SolverStatus.CONVERGED
                    records.append(
                        NewtonIterationRecord(
                            iteration=it,
                            residual_norm=r_norm,
                            residual_norm_relative=rel_norm,
                            step_norm=step_norm,
                            damping=0.0,
                            accepted=True,
                            backtracking_steps=0,
                            linear_solve_method=linear.method.value,
                            linear_residual_norm=linear.linear_residual_norm,
                            message="step tolerance and residual tolerance reached",
                        )
                    )
                    break

            ls = backtracking_line_search(
                real_residual_fn_eval,
                x,
                r,
                step,
                current_norm=r_norm,
                config=cfg,
            )

            x = ls.x_new
            r = ls.r_new
            r_norm = float(ls.residual_norm_new)
            rel_norm = float(residual_relative_norm(r_norm, r0_norm))

            records.append(
                NewtonIterationRecord(
                    iteration=it,
                    residual_norm=r_norm,
                    residual_norm_relative=rel_norm,
                    step_norm=step_norm,
                    damping=ls.damping,
                    accepted=ls.accepted,
                    backtracking_steps=ls.backtracking_steps,
                    linear_solve_method=linear.method.value,
                    linear_residual_norm=linear.linear_residual_norm,
                    message=ls.message,
                )
            )

            if has_nonfinite(r):
                status = SolverStatus.NONFINITE_RESIDUAL
                break

            if not ls.accepted and not cfg.accept_residual_increase:
                status = SolverStatus.LINE_SEARCH_FAILED
                break

            if r_norm <= cfg.abs_tol or rel_norm <= cfg.rel_tol:
                status = SolverStatus.CONVERGED
                break

        converged = status == SolverStatus.CONVERGED
        x_tree = unravel_x(x)
        r_tree = residual_fn(x_tree)

        report = HBSolverReport(
            status=status,
            converged=converged,
            iterations=len(records) - 1,
            initial_residual_norm=r0_norm,
            final_residual_norm=r_norm,
            final_relative_residual_norm=rel_norm,
            final_step_norm=final_step_norm,
            unknown_size=unknown_size,
            residual_size=residual_size,
            config=cfg.to_dict(),
            records=tuple(records),
            metadata=dict(metadata or {}),
        )

        if cfg.fail_on_nonconvergence and not converged:
            raise RuntimeError(report.summary_line())

        return HBSolverResult(
            x=x_tree,
            residual=r_tree,
            report=report,
        )

    except Exception as exc:
        # Try to return as much state as possible.
        try:
            x_tree = unravel_x(x_vec0)
            r_tree = residual_fn(x_tree)
            r_vec = pack_residual_tree(r_tree)
            final_norm = float(real_norm(r_vec, cfg.norm)) if not has_nonfinite(r_vec) else float("inf")
            residual_size = int(r_vec.shape[0])
        except Exception:
            x_tree = x0
            r_tree = None
            final_norm = float("inf")
            residual_size = -1

        report = HBSolverReport(
            status=SolverStatus.EXCEPTION,
            converged=False,
            iterations=0,
            initial_residual_norm=final_norm,
            final_residual_norm=final_norm,
            final_relative_residual_norm=float("inf"),
            final_step_norm=None,
            unknown_size=int(x_vec0.shape[0]),
            residual_size=residual_size,
            config=cfg.to_dict(),
            records=tuple(records),
            metadata=dict(metadata or {}),
            exception_message=str(exc),
        )

        if cfg.fail_on_nonconvergence:
            raise

        return HBSolverResult(
            x=x_tree,
            residual=r_tree,
            report=report,
        )


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def solve_hb_dense(
    residual_fn: ResidualFn,
    x0: PyTree,
    *,
    config: DenseNewtonConfig | SolverConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HBSolverResult:
    """
    Alias for solve_dense_newton.
    """
    return solve_dense_newton(
        residual_fn,
        x0,
        config=config,
        metadata=metadata,
    )


def solve_hb_newton_krylov(
    residual_fn: ResidualFn,
    x0: PyTree,
    *,
    config: SolverConfig,
    preconditioner_factory: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HBSolverResult:
    """Solve a structured HB residual through matrix-free JAX JVPs and GMRES."""
    from twpa.solvers.linear_solvers import (
        IterativeLinearSolveConfig,
        LinearSolverMethod,
    )
    from twpa.solvers.newton_krylov import (
        NewtonKrylovConfig,
        NewtonKrylovStatus,
        newton_krylov_solve,
    )

    x_vec0, unravel_x = pack_unknown_tree(x0)
    real_residual_fn = make_real_residual_function(residual_fn, unravel_x)
    nk_config = NewtonKrylovConfig(
        max_iter=config.max_iter,
        abs_tol=config.abs_tol,
        rel_tol=config.rel_tol,
        step_tol=config.step_tol,
        linear_solver=IterativeLinearSolveConfig(
            method=LinearSolverMethod.GMRES,
            max_iter=500,
            atol=max(config.abs_tol, 1e-12),
            rtol=max(config.rel_tol, 1e-8),
            restart=50,
            allow_dense_fallback=False,
        ),
        damping_initial=config.damping_initial,
        damping_min=config.damping_min,
        damping_shrink=config.damping_backtracking_factor,
        max_line_search_steps=max(config.max_backtracking_steps, 1),
        use_preconditioner=preconditioner_factory is not None,
        require_linear_convergence=True,
        verbose=config.verbose,
        name="hb_newton_krylov",
    )
    nk = newton_krylov_solve(
        real_residual_fn,
        x_vec0,
        config=nk_config,
        preconditioner_factory=preconditioner_factory,
        metadata={
            "solver_backend": SolverBackend.NEWTON_KRYLOV.value,
            **dict(metadata or {}),
        },
    )

    status_map = {
        NewtonKrylovStatus.CONVERGED: SolverStatus.CONVERGED,
        NewtonKrylovStatus.MAX_ITER: SolverStatus.MAX_ITERATIONS,
        NewtonKrylovStatus.LINE_SEARCH_FAILED: SolverStatus.LINE_SEARCH_FAILED,
        NewtonKrylovStatus.LINEAR_SOLVE_FAILED: SolverStatus.LINEAR_SOLVE_FAILED,
        NewtonKrylovStatus.NONFINITE_RESIDUAL: SolverStatus.NONFINITE_RESIDUAL,
        NewtonKrylovStatus.NONFINITE_STEP: SolverStatus.NONFINITE_STEP,
        NewtonKrylovStatus.FAILED: SolverStatus.EXCEPTION,
    }
    records = tuple(
        NewtonIterationRecord(
            iteration=record.iteration,
            residual_norm=record.residual_norm,
            residual_norm_relative=record.relative_residual_norm,
            step_norm=record.step_norm,
            damping=record.damping,
            accepted=record.accepted,
            backtracking_steps=0,
            linear_solve_method=(
                None
                if record.linear_result is None
                else record.linear_result.method.value
            ),
            linear_residual_norm=(
                None
                if record.linear_result is None
                else record.linear_result.residual_norm
            ),
            message=record.message,
        )
        for record in nk.records
    )
    x_tree = unravel_x(nk.x)
    report = HBSolverReport(
        status=status_map[nk.status],
        converged=nk.converged,
        iterations=nk.iterations,
        initial_residual_norm=nk.initial_residual_norm,
        final_residual_norm=nk.final_residual_norm,
        final_relative_residual_norm=nk.final_relative_residual_norm,
        final_step_norm=records[-1].step_norm if records else None,
        unknown_size=nk.unknown_size,
        residual_size=nk.residual_size,
        config={
            "backend": SolverBackend.NEWTON_KRYLOV.value,
            "solver": nk.config.to_dict(),
        },
        records=records,
        metadata={
            "matrix_free": True,
            "newton_krylov": nk.to_dict(),
            "preconditioner_summary": {
                "used": bool(preconditioner_factory is not None),
                "rebuild_iterations": list(
                    (nk.metadata or {}).get("preconditioner_rebuild_iterations", [])
                ),
                "kind_history": list(
                    (nk.metadata or {}).get("preconditioner_kind_history", [])
                ),
                "source_history": list(
                    (nk.metadata or {}).get("preconditioner_source_history", [])
                ),
            },
            **dict(metadata or {}),
        },
        exception_message=None if nk.converged else nk.message,
    )
    if config.fail_on_nonconvergence and not nk.converged:
        raise RuntimeError(report.summary_line())
    return HBSolverResult(
        x=x_tree,
        residual=residual_fn(x_tree),
        report=report,
    )


def solve_hb(
    residual_fn: ResidualFn,
    x0: PyTree,
    *,
    config: DenseNewtonConfig | SolverConfig | None = None,
    preconditioner_factory: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HBSolverResult:
    """
    Generic HB solve dispatcher.

    Dispatch to dense Newton or matrix-free Newton-Krylov.
    """
    if isinstance(config, SolverConfig):
        if config.backend == SolverBackend.NEWTON_KRYLOV:
            return solve_hb_newton_krylov(
                residual_fn,
                x0,
                config=config,
                preconditioner_factory=preconditioner_factory,
                metadata=metadata,
            )
        if config.backend == SolverBackend.BLOCK_BANDED:
            raise NotImplementedError(
                f"Solver backend {config.backend.value!r} is not implemented in hb_solver.py. "
                "Use newton_krylov for matrix-free structured solves."
            )
    return solve_dense_newton(
        residual_fn,
        x0,
        config=config,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Diagnostics / validation utilities
# ---------------------------------------------------------------------------

def check_residual_jacobian_consistency(
    residual_fn: ResidualFn,
    x0: PyTree,
    *,
    direction: PyTree | None = None,
    eps_values: ArrayLike = jnp.asarray([1e-3, 1e-4, 1e-5, 1e-6]),
) -> dict[str, Any]:
    """
    Compare JVP/Jacobian action to finite differences.

    This is a small-problem diagnostic for residual implementations.
    """
    x_vec, unravel = pack_unknown_tree(x0)
    real_residual = make_real_residual_function(residual_fn, unravel)
    r0 = real_residual(x_vec)

    if direction is None:
        # Deterministic nonzero direction.
        d = jnp.sin(jnp.arange(x_vec.shape[0], dtype=jnp.float64) + 1.0)
        d = d / jnp.maximum(jnp.linalg.norm(d), 1e-300)
    else:
        d, _ = pack_unknown_tree(direction)
        d = d / jnp.maximum(jnp.linalg.norm(d), 1e-300)

    J = jax.jacfwd(real_residual)(x_vec)
    Jd = J @ d

    eps = jnp.asarray(eps_values, dtype=jnp.float64)
    errors = []
    for e in eps.tolist():
        fd = (real_residual(x_vec + e * d) - r0) / e
        err = jnp.linalg.norm(fd - Jd) / jnp.maximum(jnp.linalg.norm(Jd), 1e-300)
        errors.append(float(err))

    return {
        "unknown_size": int(x_vec.shape[0]),
        "residual_size": int(r0.shape[0]),
        "eps_values": [float(v) for v in eps.tolist()],
        "relative_errors": errors,
        "min_relative_error": float(jnp.min(jnp.asarray(errors))),
        "passed_loose": bool(float(jnp.min(jnp.asarray(errors))) < 1e-3),
    }


def residual_summary(residual: PyTree, *, norm: NormKind | str = NormKind.L2) -> dict[str, Any]:
    """
    Summarize a residual PyTree.
    """
    vec = pack_residual_tree(residual)
    return {
        "size": int(vec.shape[0]),
        "norm": float(real_norm(vec, norm)),
        "max_abs": float(jnp.max(jnp.abs(vec))) if vec.size else 0.0,
        "has_nonfinite": has_nonfinite(vec),
    }


def unknown_summary(x: PyTree, *, norm: NormKind | str = NormKind.L2) -> dict[str, Any]:
    """
    Summarize an unknown PyTree.
    """
    vec, _ = pack_unknown_tree(x)
    return {
        "size": int(vec.shape[0]),
        "norm": float(real_norm(vec, norm)),
        "max_abs": float(jnp.max(jnp.abs(vec))) if vec.size else 0.0,
        "has_nonfinite": has_nonfinite(vec),
    }


__all__ = [
    "ArrayLike",
    "PyTree",
    "ResidualFn",
    "SolverStatus",
    "LinearSolveMethod",
    "NormKind",
    "DenseNewtonConfig",
    "NewtonIterationRecord",
    "HBSolverReport",
    "HBSolverResult",
    "real_norm",
    "residual_relative_norm",
    "pack_unknown_tree",
    "pack_residual_tree",
    "make_real_residual_function",
    "has_nonfinite",
    "DenseLinearSolveResult",
    "dense_linear_solve",
    "LineSearchResult",
    "backtracking_line_search",
    "solve_dense_newton",
    "solve_hb_dense",
    "solve_hb_newton_krylov",
    "solve_hb",
    "check_residual_jacobian_consistency",
    "residual_summary",
    "unknown_summary",
]
