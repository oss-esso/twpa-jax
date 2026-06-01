"""
twpa.solvers.newton_krylov
==========================

Matrix-free Newton-Krylov solver for large harmonic-balance systems.

This module is the production-facing replacement path for dense Newton solves
when the unknown vector becomes too large for explicit dense Jacobians.

Core idea
---------
Given a nonlinear residual

    F(x) = 0

Newton's method requires solving

    J(x_k) dx = -F(x_k)

where J is the Jacobian of F. In a Newton-Krylov method, J is not explicitly
formed. Instead, the linear solver only needs Jacobian-vector products:

    J(x) v

These can be obtained by:

    - JAX jvp,
    - finite difference,
    - a user-provided matvec.

This module provides:
    - NewtonKrylovConfig
    - NewtonKrylovIterationRecord
    - NewtonKrylovResult
    - newton_krylov_solve

It uses twpa.solvers.linear_solvers for GMRES/CG/BiCGSTAB wrapping and
twpa.solvers.preconditioners for optional preconditioners.

Scope
-----
This is a solver infrastructure layer. It does not know anything about TWPAs,
layouts, HB tones, or circuit equations. It operates on flat vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np

import jax
import jax.numpy as jnp

from twpa.solvers.linear_solvers import (
    IterativeLinearSolveConfig,
    IterativeLinearSolveResult,
    LinearOperator,
    LinearSolverMethod,
    solve_linear_system,
)
from twpa.solvers.preconditioners import (
    Preconditioner,
    PreconditionerConfig,
    PreconditionerKind,
    build_preconditioner,
)


ArrayLike = Any
ResidualFunction = Callable[[jax.Array], jax.Array]
JacobianVectorProduct = Callable[[jax.Array, jax.Array], jax.Array]
PreconditionerFactory = Callable[[jax.Array, jax.Array], Preconditioner | None]


class JacobianVectorProductMethod(str, Enum):
    """How to compute J(x) @ v."""

    JAX_JVP = "jax_jvp"
    FINITE_DIFFERENCE = "finite_difference"
    USER_SUPPLIED = "user_supplied"


class NewtonKrylovStatus(str, Enum):
    """Newton-Krylov solve status."""

    CONVERGED = "converged"
    MAX_ITER = "max_iter"
    LINE_SEARCH_FAILED = "line_search_failed"
    LINEAR_SOLVE_FAILED = "linear_solve_failed"
    NONFINITE_RESIDUAL = "nonfinite_residual"
    NONFINITE_STEP = "nonfinite_step"
    FAILED = "failed"


@dataclass(frozen=True)
class NewtonKrylovConfig:
    """
    Matrix-free Newton-Krylov solver configuration.

    Parameters
    ----------
    max_iter:
        Maximum Newton iterations.
    abs_tol:
        Absolute nonlinear residual tolerance.
    rel_tol:
        Relative nonlinear residual tolerance with respect to the initial
        residual.
    step_tol:
        Stop if the accepted step norm falls below this threshold.
    linear_solver:
        Configuration for each Krylov linear solve.
    jvp_method:
        JVP construction method.
    finite_difference_epsilon:
        Finite-difference scale used when jvp_method=FINITE_DIFFERENCE.
    damping_initial:
        Initial line-search damping factor.
    damping_min:
        Minimum damping before declaring line-search failure.
    damping_shrink:
        Factor by which damping is reduced.
    armijo_c1:
        Armijo decrease coefficient.
    max_line_search_steps:
        Maximum damping attempts per Newton iteration.
    preconditioner_config:
        Optional preconditioner construction config.
    use_preconditioner:
        Whether to build/use preconditioners.
    rebuild_preconditioner_every:
        Rebuild cadence. A value of 1 rebuilds every Newton iteration.
    require_linear_convergence:
        Treat a failed Krylov solve as fatal.
    require_residual_decrease:
        Require line search to decrease residual norm.
    verbose:
        Print iteration diagnostics.
    name:
        Diagnostic name.
    """

    max_iter: int = 30
    abs_tol: float = 1e-10
    rel_tol: float = 1e-10
    step_tol: float = 1e-12
    linear_solver: IterativeLinearSolveConfig = IterativeLinearSolveConfig(
        method=LinearSolverMethod.GMRES,
        max_iter=500,
        atol=1e-12,
        rtol=1e-8,
        restart=50,
        allow_dense_fallback=False,
    )
    jvp_method: JacobianVectorProductMethod = JacobianVectorProductMethod.JAX_JVP
    finite_difference_epsilon: float = 1e-7
    damping_initial: float = 1.0
    damping_min: float = 1e-6
    damping_shrink: float = 0.5
    armijo_c1: float = 1e-4
    max_line_search_steps: int = 20
    preconditioner_config: PreconditionerConfig = PreconditionerConfig(
        kind=PreconditionerKind.IDENTITY
    )
    use_preconditioner: bool = False
    rebuild_preconditioner_every: int = 1
    require_linear_convergence: bool = False
    require_residual_decrease: bool = True
    verbose: bool = False
    name: str = "newton_krylov"

    def __post_init__(self) -> None:
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if self.abs_tol < 0.0:
            raise ValueError("abs_tol must be non-negative")
        if self.rel_tol < 0.0:
            raise ValueError("rel_tol must be non-negative")
        if self.step_tol < 0.0:
            raise ValueError("step_tol must be non-negative")
        if self.finite_difference_epsilon <= 0.0:
            raise ValueError("finite_difference_epsilon must be positive")
        if self.damping_initial <= 0.0:
            raise ValueError("damping_initial must be positive")
        if self.damping_min <= 0.0:
            raise ValueError("damping_min must be positive")
        if not (0.0 < self.damping_shrink < 1.0):
            raise ValueError("damping_shrink must be in (0, 1)")
        if self.armijo_c1 < 0.0:
            raise ValueError("armijo_c1 must be non-negative")
        if int(self.max_line_search_steps) <= 0:
            raise ValueError("max_line_search_steps must be positive")
        if int(self.rebuild_preconditioner_every) <= 0:
            raise ValueError("rebuild_preconditioner_every must be positive")

        object.__setattr__(self, "max_iter", int(self.max_iter))
        object.__setattr__(self, "max_line_search_steps", int(self.max_line_search_steps))
        object.__setattr__(
            self,
            "rebuild_preconditioner_every",
            int(self.rebuild_preconditioner_every),
        )
        object.__setattr__(self, "jvp_method", JacobianVectorProductMethod(self.jvp_method))

    def with_updates(self, **kwargs: Any) -> "NewtonKrylovConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "step_tol": self.step_tol,
            "linear_solver": self.linear_solver.to_dict(),
            "jvp_method": self.jvp_method.value,
            "finite_difference_epsilon": self.finite_difference_epsilon,
            "damping_initial": self.damping_initial,
            "damping_min": self.damping_min,
            "damping_shrink": self.damping_shrink,
            "armijo_c1": self.armijo_c1,
            "max_line_search_steps": self.max_line_search_steps,
            "preconditioner_config": self.preconditioner_config.to_dict(),
            "use_preconditioner": self.use_preconditioner,
            "rebuild_preconditioner_every": self.rebuild_preconditioner_every,
            "require_linear_convergence": self.require_linear_convergence,
            "require_residual_decrease": self.require_residual_decrease,
            "verbose": self.verbose,
            "name": self.name,
        }


@dataclass(frozen=True)
class NewtonKrylovIterationRecord:
    """
    One Newton-Krylov iteration record.
    """

    iteration: int
    residual_norm: float
    relative_residual_norm: float
    step_norm: float | None
    damping: float | None
    accepted: bool
    linear_result: IterativeLinearSolveResult | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "residual_norm": self.residual_norm,
            "relative_residual_norm": self.relative_residual_norm,
            "step_norm": self.step_norm,
            "damping": self.damping,
            "accepted": self.accepted,
            "linear_result": (
                None if self.linear_result is None else self.linear_result.to_dict()
            ),
            "message": self.message,
        }


@dataclass(frozen=True)
class NewtonKrylovResult:
    """
    Result of a Newton-Krylov nonlinear solve.
    """

    x: jax.Array
    residual: jax.Array
    status: NewtonKrylovStatus
    converged: bool
    records: tuple[NewtonKrylovIterationRecord, ...]
    config: NewtonKrylovConfig
    initial_residual_norm: float
    final_residual_norm: float
    final_relative_residual_norm: float
    message: str
    metadata: Mapping[str, Any] | None = None

    @property
    def iterations(self) -> int:
        return max(0, len(self.records) - 1)

    @property
    def unknown_size(self) -> int:
        return int(self.x.shape[0])

    @property
    def residual_size(self) -> int:
        return int(self.residual.shape[0])

    def summary_line(self) -> str:
        return (
            f"NewtonKrylov(status={self.status.value}, "
            f"converged={self.converged}, "
            f"iterations={self.iterations}, "
            f"final_residual={self.final_residual_norm:.6e}, "
            f"relative={self.final_relative_residual_norm:.6e})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "converged": self.converged,
            "iterations": self.iterations,
            "unknown_size": self.unknown_size,
            "residual_size": self.residual_size,
            "initial_residual_norm": self.initial_residual_norm,
            "final_residual_norm": self.final_residual_norm,
            "final_relative_residual_norm": self.final_relative_residual_norm,
            "solution_shape": tuple(int(v) for v in self.x.shape),
            "solution_dtype": str(np.asarray(self.x).dtype),
            "residual_shape": tuple(int(v) for v in self.residual.shape),
            "config": self.config.to_dict(),
            "records": [r.to_dict() for r in self.records],
            "message": self.message,
            "metadata": dict(self.metadata or {}),
        }


def _norm(x: jax.Array) -> float:
    return float(jnp.linalg.norm(x))


def _finite(x: jax.Array) -> bool:
    return bool(jnp.all(jnp.isfinite(x)))


def make_jvp_operator(
    residual_fn: ResidualFunction,
    x: jax.Array,
    *,
    method: JacobianVectorProductMethod,
    user_jvp: JacobianVectorProduct | None = None,
    finite_difference_epsilon: float = 1e-7,
    name: str = "newton_krylov_jacobian",
) -> LinearOperator:
    """
    Build a matrix-free Jacobian LinearOperator around x.
    """
    method = JacobianVectorProductMethod(method)
    x = jnp.asarray(x)
    f0 = jnp.asarray(residual_fn(x))

    if f0.ndim != 1:
        raise ValueError(f"residual_fn must return a flat vector, got shape {f0.shape}")

    if method == JacobianVectorProductMethod.USER_SUPPLIED:
        if user_jvp is None:
            raise ValueError("user_jvp is required when jvp_method=USER_SUPPLIED")

        def mv(v: jax.Array) -> jax.Array:
            return jnp.asarray(user_jvp(x, v), dtype=f0.dtype)

    elif method == JacobianVectorProductMethod.JAX_JVP:

        def mv(v: jax.Array) -> jax.Array:
            _, jv = jax.jvp(residual_fn, (x,), (v,))
            return jnp.asarray(jv, dtype=f0.dtype)

    elif method == JacobianVectorProductMethod.FINITE_DIFFERENCE:

        def mv(v: jax.Array) -> jax.Array:
            v_norm = jnp.linalg.norm(v)
            x_norm = jnp.maximum(jnp.linalg.norm(x), 1.0)
            h = finite_difference_epsilon * x_norm / jnp.maximum(v_norm, 1e-300)
            return (jnp.asarray(residual_fn(x + h * v), dtype=f0.dtype) - f0) / h

    else:
        raise ValueError(f"Unsupported JVP method {method}")

    return LinearOperator(
        shape=(int(f0.shape[0]), int(x.shape[0])),
        matvec=mv,
        rmatvec=None,
        dense_matrix=None,
        dtype=f0.dtype,
        name=name,
        metadata={
            "source": "make_jvp_operator",
            "jvp_method": method.value,
            "unknown_size": int(x.shape[0]),
            "residual_size": int(f0.shape[0]),
        },
    )


def _line_search(
    residual_fn: ResidualFunction,
    x: jax.Array,
    f: jax.Array,
    dx: jax.Array,
    *,
    current_norm: float,
    config: NewtonKrylovConfig,
) -> tuple[jax.Array, jax.Array, float, bool, str]:
    """
    Backtracking line search on residual norm.
    """
    damping = float(config.damping_initial)
    current_sq = current_norm * current_norm

    best_x = x
    best_f = f
    best_norm = current_norm
    best_damping = 0.0

    for _ in range(config.max_line_search_steps):
        candidate_x = x + damping * dx
        candidate_f = jnp.asarray(residual_fn(candidate_x))

        if not _finite(candidate_x) or not _finite(candidate_f):
            damping *= config.damping_shrink
            if damping < config.damping_min:
                break
            continue

        candidate_norm = _norm(candidate_f)

        if candidate_norm < best_norm:
            best_x = candidate_x
            best_f = candidate_f
            best_norm = candidate_norm
            best_damping = damping

        if not config.require_residual_decrease:
            return candidate_x, candidate_f, damping, True, "accepted without residual-decrease requirement"

        candidate_sq = candidate_norm * candidate_norm
        sufficient_decrease = candidate_sq <= (
            1.0 - config.armijo_c1 * damping
        ) * current_sq

        if sufficient_decrease or candidate_norm < current_norm:
            return candidate_x, candidate_f, damping, True, "accepted by line search"

        damping *= config.damping_shrink

        if damping < config.damping_min:
            break

    if best_norm < current_norm:
        return best_x, best_f, best_damping, True, "accepted best decreasing damping"

    return x, f, 0.0, False, "line search failed to decrease residual"


def _default_preconditioner_factory(
    operator: LinearOperator,
    config: NewtonKrylovConfig,
) -> Preconditioner | None:
    if not config.use_preconditioner:
        return None

    return build_preconditioner(
        operator,
        config.preconditioner_config,
        shape=operator.shape,
        dtype=operator.dtype,
    )


def newton_krylov_solve(
    residual_fn: ResidualFunction,
    x0: ArrayLike,
    *,
    config: NewtonKrylovConfig | None = None,
    user_jvp: JacobianVectorProduct | None = None,
    preconditioner_factory: PreconditionerFactory | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NewtonKrylovResult:
    """
    Solve F(x)=0 using matrix-free Newton-Krylov.

    Parameters
    ----------
    residual_fn:
        Callable mapping flat vector x to flat residual vector F(x).
    x0:
        Initial guess.
    config:
        Newton-Krylov configuration.
    user_jvp:
        Optional custom Jacobian-vector product ``user_jvp(x, v)``.
    preconditioner_factory:
        Optional callable ``factory(x, f) -> Preconditioner``. If omitted and
        config.use_preconditioner=True, a generic preconditioner is attempted
        from the LinearOperator.
    metadata:
        Optional metadata included in the result.
    """
    cfg = config or NewtonKrylovConfig()
    x = jnp.asarray(x0)

    if x.ndim != 1:
        raise ValueError(f"x0 must be a flat vector, got shape {x.shape}")

    f = jnp.asarray(residual_fn(x))
    if f.ndim != 1:
        raise ValueError(f"residual_fn must return a flat vector, got shape {f.shape}")

    preconditioner_kind_history: list[str | None] = []
    preconditioner_source_history: list[str | None] = []
    preconditioner_rebuild_iterations: list[int] = []

    def _result_metadata(records_now: list[NewtonKrylovIterationRecord]) -> dict[str, Any]:
        return {
            **dict(metadata or {}),
            "linear_iterations_history": [
                None if r.linear_result is None else r.linear_result.iterations
                for r in records_now[1:]
            ],
            "linear_status_history": [
                None if r.linear_result is None else r.linear_result.status.value
                for r in records_now[1:]
            ],
            "preconditioner_kind_history": list(preconditioner_kind_history),
            "preconditioner_source_history": list(preconditioner_source_history),
            "preconditioner_rebuild_iterations": list(preconditioner_rebuild_iterations),
            "accepted_damping_history": [r.damping for r in records_now[1:]],
            "residual_history": [r.residual_norm for r in records_now],
            "step_norm_history": [r.step_norm for r in records_now[1:]],
        }

    if not _finite(x) or not _finite(f):
        final_norm = _norm(f) if _finite(f) else float("inf")
        record = NewtonKrylovIterationRecord(
            iteration=0,
            residual_norm=final_norm,
            relative_residual_norm=float("inf"),
            step_norm=None,
            damping=None,
            accepted=False,
            linear_result=None,
            message="non-finite initial residual or initial state",
        )
        return NewtonKrylovResult(
            x=x,
            residual=f,
            status=NewtonKrylovStatus.NONFINITE_RESIDUAL,
            converged=False,
            records=(record,),
            config=cfg,
            initial_residual_norm=final_norm,
            final_residual_norm=final_norm,
            final_relative_residual_norm=float("inf"),
            message="non-finite initial residual or initial state",
            metadata=_result_metadata([record]),
        )

    initial_norm = _norm(f)
    current_norm = initial_norm
    rel_norm = current_norm / max(initial_norm, 1e-300)

    records: list[NewtonKrylovIterationRecord] = [
        NewtonKrylovIterationRecord(
            iteration=0,
            residual_norm=current_norm,
            relative_residual_norm=rel_norm,
            step_norm=None,
            damping=None,
            accepted=True,
            linear_result=None,
            message="initial",
        )
    ]

    if cfg.verbose:
        print(
            f"[newton-krylov] iter=0 residual={current_norm:.6e} "
            f"relative={rel_norm:.6e}"
        )

    if current_norm <= cfg.abs_tol or rel_norm <= cfg.rel_tol:
        return NewtonKrylovResult(
            x=x,
            residual=f,
            status=NewtonKrylovStatus.CONVERGED,
            converged=True,
            records=tuple(records),
            config=cfg,
            initial_residual_norm=initial_norm,
            final_residual_norm=current_norm,
            final_relative_residual_norm=rel_norm,
            message="initial guess satisfies tolerance",
            metadata=_result_metadata(records),
        )

    cached_preconditioner: Preconditioner | None = None

    for iteration in range(1, cfg.max_iter + 1):
        operator = make_jvp_operator(
            residual_fn,
            x,
            method=cfg.jvp_method,
            user_jvp=user_jvp,
            finite_difference_epsilon=cfg.finite_difference_epsilon,
            name=f"{cfg.name}_J_iter{iteration}",
        )

        preconditioner = None
        if cfg.use_preconditioner:
            rebuild = (
                cached_preconditioner is None
                or ((iteration - 1) % cfg.rebuild_preconditioner_every == 0)
            )

            if rebuild:
                if preconditioner_factory is not None:
                    cached_preconditioner = preconditioner_factory(x, f)
                else:
                    cached_preconditioner = _default_preconditioner_factory(operator, cfg)
                preconditioner_rebuild_iterations.append(iteration)

            preconditioner = cached_preconditioner
        preconditioner_kind_history.append(
            None if preconditioner is None else preconditioner.kind.value
        )
        preconditioner_source_history.append(
            None
            if preconditioner is None
            else str((preconditioner.metadata or {}).get("source", preconditioner.kind.value))
        )

        rhs = -f

        linear_result = solve_linear_system(
            operator,
            rhs,
            config=cfg.linear_solver,
            preconditioner=None if preconditioner is None else preconditioner.to_linear_operator(),
            x0=None,
            name=f"{cfg.name}_linear_iter{iteration}",
        )

        dx = linear_result.x
        step_norm = _norm(dx) if _finite(dx) else float("inf")

        if cfg.verbose:
            print(
                f"[newton-krylov] iter={iteration} "
                f"linear_status={linear_result.status.value} "
                f"linear_residual={linear_result.residual_norm:.6e} "
                f"step={step_norm:.6e}"
            )

        if not linear_result.converged and cfg.require_linear_convergence:
            records.append(
                NewtonKrylovIterationRecord(
                    iteration=iteration,
                    residual_norm=current_norm,
                    relative_residual_norm=rel_norm,
                    step_norm=step_norm,
                    damping=None,
                    accepted=False,
                    linear_result=linear_result,
                    message="linear solve failed and convergence is required",
                )
            )
            return NewtonKrylovResult(
                x=x,
                residual=f,
                status=NewtonKrylovStatus.LINEAR_SOLVE_FAILED,
                converged=False,
                records=tuple(records),
                config=cfg,
                initial_residual_norm=initial_norm,
                final_residual_norm=current_norm,
                final_relative_residual_norm=rel_norm,
                message="linear solve failed",
                metadata=_result_metadata(records),
            )

        if not _finite(dx):
            records.append(
                NewtonKrylovIterationRecord(
                    iteration=iteration,
                    residual_norm=current_norm,
                    relative_residual_norm=rel_norm,
                    step_norm=step_norm,
                    damping=None,
                    accepted=False,
                    linear_result=linear_result,
                    message="non-finite Newton step",
                )
            )
            return NewtonKrylovResult(
                x=x,
                residual=f,
                status=NewtonKrylovStatus.NONFINITE_STEP,
                converged=False,
                records=tuple(records),
                config=cfg,
                initial_residual_norm=initial_norm,
                final_residual_norm=current_norm,
                final_relative_residual_norm=rel_norm,
                message="non-finite Newton step",
                metadata=_result_metadata(records),
            )

        if step_norm <= cfg.step_tol:
            records.append(
                NewtonKrylovIterationRecord(
                    iteration=iteration,
                    residual_norm=current_norm,
                    relative_residual_norm=rel_norm,
                    step_norm=step_norm,
                    damping=0.0,
                    accepted=True,
                    linear_result=linear_result,
                    message="step tolerance reached",
                )
            )
            return NewtonKrylovResult(
                x=x,
                residual=f,
                status=NewtonKrylovStatus.CONVERGED if current_norm <= cfg.abs_tol or rel_norm <= cfg.rel_tol else NewtonKrylovStatus.FAILED,
                converged=current_norm <= cfg.abs_tol or rel_norm <= cfg.rel_tol,
                records=tuple(records),
                config=cfg,
                initial_residual_norm=initial_norm,
                final_residual_norm=current_norm,
                final_relative_residual_norm=rel_norm,
                message="step tolerance reached",
                metadata=_result_metadata(records),
            )

        x_new, f_new, damping, accepted, line_message = _line_search(
            residual_fn,
            x,
            f,
            dx,
            current_norm=current_norm,
            config=cfg,
        )

        if not accepted:
            records.append(
                NewtonKrylovIterationRecord(
                    iteration=iteration,
                    residual_norm=current_norm,
                    relative_residual_norm=rel_norm,
                    step_norm=step_norm,
                    damping=damping,
                    accepted=False,
                    linear_result=linear_result,
                    message=line_message,
                )
            )
            return NewtonKrylovResult(
                x=x,
                residual=f,
                status=NewtonKrylovStatus.LINE_SEARCH_FAILED,
                converged=False,
                records=tuple(records),
                config=cfg,
                initial_residual_norm=initial_norm,
                final_residual_norm=current_norm,
                final_relative_residual_norm=rel_norm,
                message=line_message,
                metadata=_result_metadata(records),
            )

        x = x_new
        f = f_new
        current_norm = _norm(f)
        rel_norm = current_norm / max(initial_norm, 1e-300)

        records.append(
            NewtonKrylovIterationRecord(
                iteration=iteration,
                residual_norm=current_norm,
                relative_residual_norm=rel_norm,
                step_norm=step_norm,
                damping=damping,
                accepted=True,
                linear_result=linear_result,
                message=line_message,
            )
        )

        if cfg.verbose:
            print(
                f"[newton-krylov] iter={iteration} accepted damping={damping:.3e} "
                f"residual={current_norm:.6e} relative={rel_norm:.6e}"
            )

        if not _finite(f):
            return NewtonKrylovResult(
                x=x,
                residual=f,
                status=NewtonKrylovStatus.NONFINITE_RESIDUAL,
                converged=False,
                records=tuple(records),
                config=cfg,
                initial_residual_norm=initial_norm,
                final_residual_norm=float("inf"),
                final_relative_residual_norm=float("inf"),
                message="non-finite residual after accepted step",
                metadata=_result_metadata(records),
            )

        if current_norm <= cfg.abs_tol or rel_norm <= cfg.rel_tol:
            return NewtonKrylovResult(
                x=x,
                residual=f,
                status=NewtonKrylovStatus.CONVERGED,
                converged=True,
                records=tuple(records),
                config=cfg,
                initial_residual_norm=initial_norm,
                final_residual_norm=current_norm,
                final_relative_residual_norm=rel_norm,
                message="Newton-Krylov converged",
                metadata=_result_metadata(records),
            )

    return NewtonKrylovResult(
        x=x,
        residual=f,
        status=NewtonKrylovStatus.MAX_ITER,
        converged=False,
        records=tuple(records),
        config=cfg,
        initial_residual_norm=initial_norm,
        final_residual_norm=current_norm,
        final_relative_residual_norm=rel_norm,
        message="maximum Newton iterations reached",
        metadata=_result_metadata(records),
    )


def finite_difference_jvp(
    residual_fn: ResidualFunction,
    x: ArrayLike,
    v: ArrayLike,
    *,
    epsilon: float = 1e-7,
) -> jax.Array:
    """
    Standalone finite-difference JVP helper.
    """
    x_arr = jnp.asarray(x)
    v_arr = jnp.asarray(v, dtype=x_arr.dtype)

    v_norm = jnp.linalg.norm(v_arr)
    x_norm = jnp.maximum(jnp.linalg.norm(x_arr), 1.0)
    h = epsilon * x_norm / jnp.maximum(v_norm, 1e-300)

    return (jnp.asarray(residual_fn(x_arr + h * v_arr)) - jnp.asarray(residual_fn(x_arr))) / h


def validate_jvp(
    residual_fn: ResidualFunction,
    x: ArrayLike,
    *,
    user_jvp: JacobianVectorProduct | None = None,
    n_random_tests: int = 3,
    finite_difference_epsilon: float = 1e-7,
    seed: int = 123,
) -> dict[str, Any]:
    """
    Compare JAX/user JVP against finite-difference JVP on random directions.
    """
    x_arr = jnp.asarray(x)
    rng = np.random.default_rng(seed)

    errors: list[float] = []
    messages: list[str] = []
    passed = True

    for k in range(n_random_tests):
        v_np = rng.standard_normal(x_arr.shape)
        if jnp.iscomplexobj(x_arr):
            v_np = v_np + 1j * rng.standard_normal(x_arr.shape)

        v = jnp.asarray(v_np, dtype=x_arr.dtype)

        try:
            if user_jvp is not None:
                jv_ref = jnp.asarray(user_jvp(x_arr, v))
            else:
                _, jv_ref = jax.jvp(residual_fn, (x_arr,), (v,))

            jv_fd = finite_difference_jvp(
                residual_fn,
                x_arr,
                v,
                epsilon=finite_difference_epsilon,
            )

            err = float(jnp.linalg.norm(jv_ref - jv_fd))
            scale = max(float(jnp.linalg.norm(jv_ref)), float(jnp.linalg.norm(jv_fd)), 1.0)
            rel = err / scale
            errors.append(rel)

            if rel > 1e-4:
                passed = False
                messages.append(f"large JVP relative error on test {k}: {rel:.3e}")

        except Exception as exc:
            passed = False
            messages.append(f"JVP validation raised on test {k}: {exc}")

    if passed:
        messages.append("PASS: JVP validation checks passed.")

    return {
        "passed": passed,
        "messages": messages,
        "n_random_tests": n_random_tests,
        "relative_error_min": float(np.min(errors)) if errors else None,
        "relative_error_max": float(np.max(errors)) if errors else None,
        "relative_error_mean": float(np.mean(errors)) if errors else None,
        "errors": errors,
    }


__all__ = [
    "ArrayLike",
    "ResidualFunction",
    "JacobianVectorProduct",
    "PreconditionerFactory",
    "JacobianVectorProductMethod",
    "NewtonKrylovStatus",
    "NewtonKrylovConfig",
    "NewtonKrylovIterationRecord",
    "NewtonKrylovResult",
    "make_jvp_operator",
    "newton_krylov_solve",
    "finite_difference_jvp",
    "validate_jvp",
]
