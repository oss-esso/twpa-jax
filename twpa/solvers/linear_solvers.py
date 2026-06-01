"""
twpa.solvers.linear_solvers
===========================

Shared linear-solver wrappers for the TWPA harmonic-balance stack.

This module provides a small abstraction layer around dense and iterative
linear solves. It is designed to be used by:

    - dense/reference HB solvers,
    - Newton-Krylov large-system solvers,
    - block-banded industrial backends,
    - preconditioner experiments.

The API is intentionally conservative:

    LinearOperator
        Matrix-free operator wrapper with optional dense representation.

    IterativeLinearSolveConfig
        Solver method, tolerances, iteration limits, and fallback policy.

    IterativeLinearSolveResult
        Solution vector plus residual/convergence diagnostics.

The implementation prefers robust behavior over cleverness. If SciPy is
available, SciPy Krylov solvers are used for non-JIT workflow solves. If SciPy
is unavailable, the module falls back to JAX dense/direct solves when a dense
matrix is available.

This file does not depend on the TWPA circuit model, so it can be used by all
higher-level modules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

import jax
import jax.numpy as jnp

try:
    import scipy.sparse.linalg as scipy_spla

    SCIPY_AVAILABLE = True
except Exception:
    scipy_spla = None
    SCIPY_AVAILABLE = False


ArrayLike = Any
Matvec = Callable[[jax.Array], jax.Array]


class LinearSolveStatus(str, Enum):
    """Linear-solve status."""

    CONVERGED = "converged"
    FAILED = "failed"
    FALLBACK_DENSE = "fallback_dense"
    MAX_ITER = "max_iter"
    SINGULAR = "singular"
    INVALID_INPUT = "invalid_input"


class LinearSolverMethod(str, Enum):
    """Supported linear-solver methods."""

    AUTO = "auto"
    DENSE = "dense"
    NORMAL_EQUATIONS = "normal_equations"
    CG = "cg"
    GMRES = "gmres"
    BICGSTAB = "bicgstab"


class PreconditionerProtocol(Protocol):
    """Callable preconditioner protocol."""

    def __call__(self, x: jax.Array) -> jax.Array:
        ...


@dataclass(frozen=True)
class LinearOperator:
    """
    Matrix-free linear operator.

    Parameters
    ----------
    shape:
        Operator shape ``(m, n)``.
    matvec:
        Function computing ``A @ x``.
    rmatvec:
        Optional function computing ``A^H @ x``.
    dense_matrix:
        Optional dense matrix. When present, direct solves and dense fallback
        are available.
    dtype:
        Preferred dtype.
    name:
        Diagnostic name.
    metadata:
        Extra metadata.

    Notes
    -----
    This object is deliberately lightweight. It is not a scipy LinearOperator,
    but it can be converted to one when SciPy is available.
    """

    shape: tuple[int, int]
    matvec: Matvec
    rmatvec: Matvec | None = None
    dense_matrix: jax.Array | None = None
    dtype: Any = jnp.complex128
    name: str = "linear_operator"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("shape must be a pair (m, n)")
        m, n = int(self.shape[0]), int(self.shape[1])
        if m <= 0 or n <= 0:
            raise ValueError("operator dimensions must be positive")
        object.__setattr__(self, "shape", (m, n))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

        if self.dense_matrix is not None:
            dense = jnp.asarray(self.dense_matrix, dtype=self.dtype)
            if dense.shape != (m, n):
                raise ValueError(f"dense_matrix shape {dense.shape} != {(m, n)}")
            object.__setattr__(self, "dense_matrix", dense)

    @classmethod
    def from_dense(
        cls,
        matrix: ArrayLike,
        *,
        name: str = "dense_operator",
        metadata: Mapping[str, Any] | None = None,
    ) -> "LinearOperator":
        """
        Build a LinearOperator from a dense matrix.
        """
        A = jnp.asarray(matrix)
        if A.ndim != 2:
            raise ValueError(f"matrix must be 2D, got shape {A.shape}")

        def mv(x: jax.Array) -> jax.Array:
            return A @ x

        def rmv(x: jax.Array) -> jax.Array:
            return jnp.conj(A.T) @ x

        return cls(
            shape=(int(A.shape[0]), int(A.shape[1])),
            matvec=mv,
            rmatvec=rmv,
            dense_matrix=A,
            dtype=A.dtype,
            name=name,
            metadata=metadata,
        )

    @classmethod
    def identity(
        cls,
        size: int,
        *,
        dtype: Any = jnp.complex128,
        name: str = "identity_operator",
    ) -> "LinearOperator":
        """
        Identity operator.
        """
        if int(size) <= 0:
            raise ValueError("size must be positive")
        n = int(size)
        eye = jnp.eye(n, dtype=dtype)
        return cls.from_dense(eye, name=name)

    def __matmul__(self, x: ArrayLike) -> jax.Array:
        return self.apply(x)

    @property
    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    @property
    def size_in(self) -> int:
        return self.shape[1]

    @property
    def size_out(self) -> int:
        return self.shape[0]

    @property
    def has_dense(self) -> bool:
        return self.dense_matrix is not None

    def apply(self, x: ArrayLike) -> jax.Array:
        """
        Compute ``A @ x``.
        """
        x_arr = jnp.asarray(x, dtype=self.dtype)
        if x_arr.shape != (self.size_in,):
            raise ValueError(f"x must have shape {(self.size_in,)}, got {x_arr.shape}")
        return jnp.asarray(self.matvec(x_arr), dtype=self.dtype)

    def apply_adjoint(self, x: ArrayLike) -> jax.Array:
        """
        Compute ``A^H @ x``.
        """
        x_arr = jnp.asarray(x, dtype=self.dtype)
        if x_arr.shape != (self.size_out,):
            raise ValueError(f"x must have shape {(self.size_out,)}, got {x_arr.shape}")

        if self.rmatvec is not None:
            return jnp.asarray(self.rmatvec(x_arr), dtype=self.dtype)

        if self.dense_matrix is not None:
            return jnp.conj(self.dense_matrix.T) @ x_arr

        raise ValueError("Adjoint action unavailable: rmatvec and dense_matrix are both None")

    def to_dense(self) -> jax.Array:
        """
        Return dense matrix representation.

        If dense_matrix is absent, the dense matrix is reconstructed by applying
        matvec to basis vectors. This can be expensive and should only be used
        for small diagnostic systems.
        """
        if self.dense_matrix is not None:
            return self.dense_matrix

        basis = jnp.eye(self.size_in, dtype=self.dtype)
        cols = [self.apply(basis[:, i]) for i in range(self.size_in)]
        return jnp.stack(cols, axis=1)

    def normal_operator(self) -> "LinearOperator":
        """
        Return the normal-equation operator ``A^H A``.
        """
        if self.rmatvec is None and self.dense_matrix is None:
            raise ValueError("Cannot form normal operator without adjoint action")

        def mv(x: jax.Array) -> jax.Array:
            return self.apply_adjoint(self.apply(x))

        dense = None
        if self.dense_matrix is not None:
            dense = jnp.conj(self.dense_matrix.T) @ self.dense_matrix

        return LinearOperator(
            shape=(self.size_in, self.size_in),
            matvec=mv,
            dense_matrix=dense,
            dtype=self.dtype,
            name=f"{self.name}_normal",
            metadata={
                "source": "normal_operator",
                "original_operator": self.name,
            },
        )

    def to_scipy(self) -> Any:
        """
        Convert to scipy.sparse.linalg.LinearOperator.
        """
        if not SCIPY_AVAILABLE or scipy_spla is None:
            raise RuntimeError("SciPy is not available")

        dtype = np.dtype(np.asarray(jnp.zeros((), dtype=self.dtype)).dtype)

        def mv_np(x: np.ndarray) -> np.ndarray:
            return np.array(self.apply(jnp.asarray(x, dtype=self.dtype)), copy=True)

        def rmv_np(x: np.ndarray) -> np.ndarray:
            return np.array(self.apply_adjoint(jnp.asarray(x, dtype=self.dtype)), copy=True)

        return scipy_spla.LinearOperator(
            shape=self.shape,
            matvec=mv_np,
            rmatvec=rmv_np if (self.rmatvec is not None or self.dense_matrix is not None) else None,
            dtype=dtype,
        )

    def to_dict(self, *, include_dense: bool = False) -> dict[str, Any]:
        out = {
            "name": self.name,
            "shape": tuple(int(v) for v in self.shape),
            "dtype": str(np.dtype(np.asarray(jnp.zeros((), dtype=self.dtype)).dtype)),
            "has_dense": self.has_dense,
            "has_rmatvec": self.rmatvec is not None,
            "metadata": dict(self.metadata or {}),
        }
        if include_dense and self.dense_matrix is not None:
            dense = np.asarray(self.dense_matrix)
            out["dense"] = {
                "shape": tuple(int(v) for v in dense.shape),
                "dtype": str(dense.dtype),
                "min_abs": float(np.nanmin(np.abs(dense))) if dense.size else None,
                "max_abs": float(np.nanmax(np.abs(dense))) if dense.size else None,
            }
        return out


@dataclass(frozen=True)
class IterativeLinearSolveConfig:
    """
    Configuration for direct/iterative linear solves.
    """

    method: LinearSolverMethod = LinearSolverMethod.AUTO
    max_iter: int = 500
    atol: float = 1e-12
    rtol: float = 1e-10
    restart: int | None = None
    regularization: float = 0.0
    allow_dense_fallback: bool = True
    require_convergence: bool = False
    use_scipy_if_available: bool = True
    name: str = "linear_solve"

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", LinearSolverMethod(self.method))
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        object.__setattr__(self, "max_iter", int(self.max_iter))
        if self.atol < 0.0:
            raise ValueError("atol must be non-negative")
        if self.rtol < 0.0:
            raise ValueError("rtol must be non-negative")
        if self.restart is not None and int(self.restart) <= 0:
            raise ValueError("restart must be positive when provided")
        if self.restart is not None:
            object.__setattr__(self, "restart", int(self.restart))
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")

    def selected_method(self, operator: LinearOperator) -> LinearSolverMethod:
        """
        Resolve AUTO into a concrete method.
        """
        if self.method != LinearSolverMethod.AUTO:
            return self.method

        if operator.has_dense:
            return LinearSolverMethod.DENSE

        if operator.is_square:
            return LinearSolverMethod.GMRES

        return LinearSolverMethod.NORMAL_EQUATIONS

    def with_updates(self, **kwargs: Any) -> "IterativeLinearSolveConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "max_iter": self.max_iter,
            "atol": self.atol,
            "rtol": self.rtol,
            "restart": self.restart,
            "regularization": self.regularization,
            "allow_dense_fallback": self.allow_dense_fallback,
            "require_convergence": self.require_convergence,
            "use_scipy_if_available": self.use_scipy_if_available,
            "scipy_available": SCIPY_AVAILABLE,
            "name": self.name,
        }


@dataclass(frozen=True)
class IterativeLinearSolveResult:
    """
    Result of a linear solve.
    """

    x: jax.Array
    status: LinearSolveStatus
    method: LinearSolverMethod
    converged: bool
    residual_norm: float
    relative_residual_norm: float
    rhs_norm: float
    iterations: int | None
    operator_shape: tuple[int, int]
    message: str = ""
    metadata: Mapping[str, Any] | None = None

    @property
    def success(self) -> bool:
        """Dense-solver compatibility alias."""
        return self.converged

    @property
    def step(self) -> jax.Array:
        """Dense-solver compatibility alias."""
        return self.x

    @property
    def linear_residual_norm(self) -> float:
        """Dense-solver compatibility alias."""
        return self.residual_norm

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "method": self.method.value,
            "converged": self.converged,
            "residual_norm": self.residual_norm,
            "relative_residual_norm": self.relative_residual_norm,
            "rhs_norm": self.rhs_norm,
            "iterations": self.iterations,
            "operator_shape": tuple(int(v) for v in self.operator_shape),
            "solution_shape": tuple(int(v) for v in self.x.shape),
            "solution_dtype": str(np.asarray(self.x).dtype),
            "message": self.message,
            "metadata": dict(self.metadata or {}),
        }


def as_linear_operator(
    operator: LinearOperator | ArrayLike | Matvec,
    *,
    shape: tuple[int, int] | None = None,
    dtype: Any = jnp.complex128,
    name: str = "linear_operator",
) -> LinearOperator:
    """
    Convert a dense matrix, callable, or LinearOperator into LinearOperator.
    """
    if isinstance(operator, LinearOperator):
        return operator

    if callable(operator):
        if shape is None:
            raise ValueError("shape is required when operator is a callable")
        return LinearOperator(
            shape=shape,
            matvec=lambda x: jnp.asarray(operator(x), dtype=dtype),
            dtype=dtype,
            name=name,
        )

    arr = jnp.asarray(operator)
    return LinearOperator.from_dense(arr, name=name)


def _regularized_dense_matrix(operator: LinearOperator, regularization: float) -> jax.Array:
    A = operator.to_dense()
    if regularization <= 0.0:
        return A

    if A.shape[0] == A.shape[1]:
        return A + regularization * jnp.eye(A.shape[0], dtype=A.dtype)

    # For rectangular matrices, regularization is handled in normal equations.
    return A


def _dense_solve(
    operator: LinearOperator,
    b: jax.Array,
    config: IterativeLinearSolveConfig,
    *,
    method_status: LinearSolveStatus = LinearSolveStatus.CONVERGED,
) -> IterativeLinearSolveResult:
    """
    Dense direct or least-squares solve.
    """
    method = LinearSolverMethod.DENSE

    try:
        A = operator.to_dense()

        if A.shape[0] == A.shape[1]:
            Areg = _regularized_dense_matrix(operator, config.regularization)
            x = jnp.linalg.solve(Areg, b)
        else:
            # Least-squares through normal equations.
            AH = jnp.conj(A.T)
            normal = AH @ A
            rhs = AH @ b
            if config.regularization > 0.0:
                normal = normal + config.regularization * jnp.eye(normal.shape[0], dtype=normal.dtype)
            x = jnp.linalg.solve(normal, rhs)
            method = LinearSolverMethod.NORMAL_EQUATIONS

        residual = operator.apply(x) - b
        rhs_norm = float(jnp.linalg.norm(b))
        residual_norm = float(jnp.linalg.norm(residual))
        rel = residual_norm / max(rhs_norm, 1e-300)

        finite = bool(jnp.all(jnp.isfinite(x)) and jnp.isfinite(residual_norm))
        converged = finite and (
            residual_norm <= config.atol
            or rel <= config.rtol
            or method_status == LinearSolveStatus.FALLBACK_DENSE
        )

        status = LinearSolveStatus.CONVERGED if converged else method_status

        return IterativeLinearSolveResult(
            x=x,
            status=status,
            method=method,
            converged=converged,
            residual_norm=residual_norm,
            relative_residual_norm=rel,
            rhs_norm=rhs_norm,
            iterations=None,
            operator_shape=operator.shape,
            message="dense solve completed",
            metadata={
                "regularization": config.regularization,
                "dense_shape": tuple(int(v) for v in A.shape),
            },
        )

    except Exception as exc:
        x = jnp.full((operator.size_in,), jnp.nan + 0j, dtype=operator.dtype)
        rhs_norm = float(jnp.linalg.norm(b))
        return IterativeLinearSolveResult(
            x=x,
            status=LinearSolveStatus.SINGULAR,
            method=method,
            converged=False,
            residual_norm=float("inf"),
            relative_residual_norm=float("inf"),
            rhs_norm=rhs_norm,
            iterations=None,
            operator_shape=operator.shape,
            message=f"dense solve failed: {exc}",
            metadata={"regularization": config.regularization},
        )


def _scipy_iterative_solve(
    operator: LinearOperator,
    b: jax.Array,
    config: IterativeLinearSolveConfig,
    method: LinearSolverMethod,
    *,
    preconditioner: PreconditionerProtocol | LinearOperator | None = None,
    x0: ArrayLike | None = None,
) -> IterativeLinearSolveResult:
    """
    SciPy Krylov solver wrapper.
    """
    if not SCIPY_AVAILABLE or scipy_spla is None:
        raise RuntimeError("SciPy is not available")

    A_sp = operator.to_scipy()
    b_np = np.array(b, copy=True)

    M_sp = None
    if preconditioner is not None:
        if isinstance(preconditioner, LinearOperator):
            M_sp = preconditioner.to_scipy()
        else:
            dtype = b_np.dtype

            def m_mv(x: np.ndarray) -> np.ndarray:
                return np.array(preconditioner(jnp.asarray(x, dtype=operator.dtype)), copy=True)

            M_sp = scipy_spla.LinearOperator(
                shape=(operator.size_in, operator.size_in),
                matvec=m_mv,
                dtype=dtype,
            )

    callback_count = {"n": 0}

    def callback(_: Any) -> None:
        callback_count["n"] += 1

    x0_np = None if x0 is None else np.array(x0, copy=True)

    kwargs_common = {
        "A": A_sp,
        "b": b_np,
        "x0": x0_np,
        "rtol": config.rtol,
        "atol": config.atol,
        "maxiter": config.max_iter,
        "M": M_sp,
    }

    if method == LinearSolverMethod.CG:
        x_np, info = scipy_spla.cg(
            **kwargs_common,
            callback=callback,
        )

    elif method == LinearSolverMethod.GMRES:
        gmres_kwargs = dict(kwargs_common)
        if config.restart is not None:
            gmres_kwargs["restart"] = config.restart
        x_np, info = scipy_spla.gmres(
            **gmres_kwargs,
            callback=callback,
            callback_type="legacy",
        )

    elif method == LinearSolverMethod.BICGSTAB:
        x_np, info = scipy_spla.bicgstab(
            **kwargs_common,
            callback=callback,
        )

    else:
        raise ValueError(f"Unsupported SciPy iterative method {method}")

    x = jnp.asarray(x_np, dtype=operator.dtype)
    residual = operator.apply(x) - b
    rhs_norm = float(jnp.linalg.norm(b))
    residual_norm = float(jnp.linalg.norm(residual))
    rel = residual_norm / max(rhs_norm, 1e-300)

    converged = int(info) == 0 and (
        residual_norm <= config.atol or rel <= config.rtol
    )

    if converged:
        status = LinearSolveStatus.CONVERGED
        message = "SciPy iterative solve converged"
    elif int(info) > 0:
        status = LinearSolveStatus.MAX_ITER
        message = f"SciPy iterative solve reached max iterations: info={info}"
    else:
        status = LinearSolveStatus.FAILED
        message = f"SciPy iterative solve failed: info={info}"

    return IterativeLinearSolveResult(
        x=x,
        status=status,
        method=method,
        converged=converged,
        residual_norm=residual_norm,
        relative_residual_norm=rel,
        rhs_norm=rhs_norm,
        iterations=callback_count["n"],
        operator_shape=operator.shape,
        message=message,
        metadata={
            "scipy_info": int(info),
            "scipy_available": SCIPY_AVAILABLE,
            "used_preconditioner": preconditioner is not None,
        },
    )


def _solve_normal_equations(
    operator: LinearOperator,
    b: jax.Array,
    config: IterativeLinearSolveConfig,
    *,
    preconditioner: PreconditionerProtocol | LinearOperator | None = None,
    x0: ArrayLike | None = None,
) -> IterativeLinearSolveResult:
    """
    Solve least-squares system through normal equations.
    """
    normal = operator.normal_operator()
    rhs = operator.apply_adjoint(b)

    if config.regularization > 0.0 and normal.dense_matrix is not None:
        A = normal.dense_matrix + config.regularization * jnp.eye(
            normal.size_in,
            dtype=normal.dense_matrix.dtype,
        )
        normal = LinearOperator.from_dense(A, name=f"{operator.name}_normal_regularized")

    normal_config = config.with_updates(method=LinearSolverMethod.AUTO)

    result = solve_linear_system(
        normal,
        rhs,
        config=normal_config,
        preconditioner=preconditioner,
        x0=x0,
    )

    residual = operator.apply(result.x) - b
    rhs_norm = float(jnp.linalg.norm(b))
    residual_norm = float(jnp.linalg.norm(residual))
    rel = residual_norm / max(rhs_norm, 1e-300)

    return replace(
        result,
        method=LinearSolverMethod.NORMAL_EQUATIONS,
        residual_norm=residual_norm,
        relative_residual_norm=rel,
        rhs_norm=rhs_norm,
        operator_shape=operator.shape,
        message=f"normal equations solve: {result.message}",
        metadata={
            **dict(result.metadata or {}),
            "outer_method": "normal_equations",
            "normal_residual_norm": result.residual_norm,
        },
    )


def solve_linear_system(
    operator: LinearOperator | ArrayLike | Matvec,
    b: ArrayLike,
    *,
    config: IterativeLinearSolveConfig | None = None,
    preconditioner: PreconditionerProtocol | LinearOperator | None = None,
    x0: ArrayLike | None = None,
    shape: tuple[int, int] | None = None,
    dtype: Any = jnp.complex128,
    name: str = "linear_system",
) -> IterativeLinearSolveResult:
    """
    Solve ``A x = b`` using dense or iterative methods.

    Parameters
    ----------
    operator:
        LinearOperator, dense matrix, or matvec callable.
    b:
        Right-hand side vector.
    config:
        Solver configuration.
    preconditioner:
        Optional preconditioner ``M`` approximating ``A^{-1}``.
    x0:
        Optional initial guess for iterative methods.
    shape:
        Required when operator is a matvec callable.
    dtype:
        Dtype used for callable operators.
    name:
        Diagnostic operator name when converting from non-LinearOperator input.
    """
    cfg = config or IterativeLinearSolveConfig()
    op = as_linear_operator(operator, shape=shape, dtype=dtype, name=name)
    rhs = jnp.asarray(b, dtype=op.dtype)

    if rhs.shape != (op.size_out,):
        raise ValueError(f"b must have shape {(op.size_out,)}, got {rhs.shape}")

    method = cfg.selected_method(op)

    if method == LinearSolverMethod.DENSE:
        result = _dense_solve(op, rhs, cfg)

    elif method == LinearSolverMethod.NORMAL_EQUATIONS:
        result = _solve_normal_equations(
            op,
            rhs,
            cfg,
            preconditioner=preconditioner,
            x0=x0,
        )

    elif method in {
        LinearSolverMethod.CG,
        LinearSolverMethod.GMRES,
        LinearSolverMethod.BICGSTAB,
    }:
        if cfg.use_scipy_if_available and SCIPY_AVAILABLE:
            try:
                result = _scipy_iterative_solve(
                    op,
                    rhs,
                    cfg,
                    method,
                    preconditioner=preconditioner,
                    x0=x0,
                )
            except Exception as exc:
                if cfg.allow_dense_fallback and op.has_dense:
                    result = _dense_solve(
                        op,
                        rhs,
                        cfg,
                        method_status=LinearSolveStatus.FALLBACK_DENSE,
                    )
                    result = replace(
                        result,
                        status=LinearSolveStatus.FALLBACK_DENSE,
                        message=f"iterative solve failed; dense fallback used: {exc}",
                    )
                else:
                    x = jnp.full((op.size_in,), jnp.nan + 0j, dtype=op.dtype)
                    rhs_norm = float(jnp.linalg.norm(rhs))
                    result = IterativeLinearSolveResult(
                        x=x,
                        status=LinearSolveStatus.FAILED,
                        method=method,
                        converged=False,
                        residual_norm=float("inf"),
                        relative_residual_norm=float("inf"),
                        rhs_norm=rhs_norm,
                        iterations=None,
                        operator_shape=op.shape,
                        message=f"iterative solve failed: {exc}",
                        metadata={"scipy_available": SCIPY_AVAILABLE},
                    )
        else:
            if cfg.allow_dense_fallback and op.has_dense:
                result = _dense_solve(
                    op,
                    rhs,
                    cfg,
                    method_status=LinearSolveStatus.FALLBACK_DENSE,
                )
                result = replace(
                    result,
                    status=LinearSolveStatus.FALLBACK_DENSE,
                    message="SciPy unavailable or disabled; dense fallback used",
                )
            else:
                x = jnp.full((op.size_in,), jnp.nan + 0j, dtype=op.dtype)
                rhs_norm = float(jnp.linalg.norm(rhs))
                result = IterativeLinearSolveResult(
                    x=x,
                    status=LinearSolveStatus.FAILED,
                    method=method,
                    converged=False,
                    residual_norm=float("inf"),
                    relative_residual_norm=float("inf"),
                    rhs_norm=rhs_norm,
                    iterations=None,
                    operator_shape=op.shape,
                    message="iterative solve unavailable without SciPy and no dense fallback",
                    metadata={"scipy_available": SCIPY_AVAILABLE},
                )

    else:
        raise ValueError(f"Unsupported linear solver method {method}")

    if cfg.require_convergence and not result.converged:
        raise RuntimeError(f"Linear solve failed: {result.message}")

    return result


def estimate_operator_norm(
    operator: LinearOperator | ArrayLike,
    *,
    n_iter: int = 20,
    seed: int = 1234,
) -> float:
    """
    Estimate spectral norm with power iteration.

    For non-square operators, estimates ``sqrt(lambda_max(A^H A))``.
    """
    op = as_linear_operator(operator)
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(op.size_in) + 1j * rng.standard_normal(op.size_in)
    x = jnp.asarray(x_np, dtype=op.dtype)
    x = x / jnp.maximum(jnp.linalg.norm(x), 1e-300)

    for _ in range(n_iter):
        y = op.apply(x)
        z = op.apply_adjoint(y)
        norm_z = jnp.linalg.norm(z)
        x = z / jnp.maximum(norm_z, 1e-300)

    y = op.apply(x)
    return float(jnp.linalg.norm(y))


def residual_diagnostics(
    operator: LinearOperator | ArrayLike,
    x: ArrayLike,
    b: ArrayLike,
) -> dict[str, Any]:
    """
    Compute residual diagnostics for ``A x ≈ b``.
    """
    op = as_linear_operator(operator)
    x_arr = jnp.asarray(x, dtype=op.dtype)
    b_arr = jnp.asarray(b, dtype=op.dtype)

    residual = op.apply(x_arr) - b_arr
    residual_norm = float(jnp.linalg.norm(residual))
    rhs_norm = float(jnp.linalg.norm(b_arr))
    x_norm = float(jnp.linalg.norm(x_arr))

    return {
        "operator_shape": tuple(int(v) for v in op.shape),
        "x_norm": x_norm,
        "rhs_norm": rhs_norm,
        "residual_norm": residual_norm,
        "relative_residual_norm": residual_norm / max(rhs_norm, 1e-300),
        "max_abs_residual": float(jnp.max(jnp.abs(residual))),
    }


def solve_multiple_rhs(
    operator: LinearOperator | ArrayLike,
    B: ArrayLike,
    *,
    config: IterativeLinearSolveConfig | None = None,
    preconditioner: PreconditionerProtocol | LinearOperator | None = None,
    shape: tuple[int, int] | None = None,
    dtype: Any = jnp.complex128,
    name: str = "multi_rhs_linear_system",
) -> tuple[jax.Array, tuple[IterativeLinearSolveResult, ...]]:
    """
    Solve ``A X = B`` column by column.

    Returns
    -------
    X:
        Matrix of solution columns.
    results:
        Per-column solve diagnostics.
    """
    op = as_linear_operator(operator, shape=shape, dtype=dtype, name=name)
    B_arr = jnp.asarray(B, dtype=op.dtype)

    if B_arr.ndim == 1:
        result = solve_linear_system(
            op,
            B_arr,
            config=config,
            preconditioner=preconditioner,
        )
        return result.x[:, None], (result,)

    if B_arr.ndim != 2:
        raise ValueError(f"B must be 1D or 2D, got shape {B_arr.shape}")
    if B_arr.shape[0] != op.size_out:
        raise ValueError(f"B first dimension must be {op.size_out}, got {B_arr.shape[0]}")

    xs = []
    results = []

    for col in range(B_arr.shape[1]):
        result = solve_linear_system(
            op,
            B_arr[:, col],
            config=config,
            preconditioner=preconditioner,
        )
        xs.append(result.x)
        results.append(result)

    return jnp.stack(xs, axis=1), tuple(results)


def validate_linear_operator(
    operator: LinearOperator,
    *,
    n_random_tests: int = 3,
    seed: int = 123,
    adjoint_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """
    Validate basic operator shape and adjoint consistency.

    For complex operators, checks:

        <A x, y> ≈ <x, A^H y>
    """
    rng = np.random.default_rng(seed)
    messages: list[str] = []
    passed = True

    for k in range(n_random_tests):
        x = jnp.asarray(
            rng.standard_normal(operator.size_in) + 1j * rng.standard_normal(operator.size_in),
            dtype=operator.dtype,
        )
        y = jnp.asarray(
            rng.standard_normal(operator.size_out) + 1j * rng.standard_normal(operator.size_out),
            dtype=operator.dtype,
        )

        try:
            Ax = operator.apply(x)
            if Ax.shape != (operator.size_out,):
                passed = False
                messages.append(f"FAIL: matvec output shape {Ax.shape} on test {k}")
        except Exception as exc:
            passed = False
            messages.append(f"FAIL: matvec raised on test {k}: {exc}")
            continue

        if operator.rmatvec is not None or operator.dense_matrix is not None:
            try:
                AHy = operator.apply_adjoint(y)
                lhs = jnp.vdot(Ax, y)
                rhs = jnp.vdot(x, AHy)
                err = float(jnp.abs(lhs - rhs))
                scale = max(float(jnp.abs(lhs)), float(jnp.abs(rhs)), 1.0)
                rel = err / scale
                if rel > adjoint_tolerance:
                    passed = False
                    messages.append(
                        f"FAIL: adjoint relative error {rel:.3e} exceeds "
                        f"{adjoint_tolerance:.3e} on test {k}"
                    )
            except Exception as exc:
                passed = False
                messages.append(f"FAIL: adjoint raised on test {k}: {exc}")

    if passed:
        messages.append("PASS: linear operator validation checks passed.")

    return {
        "passed": passed,
        "messages": messages,
        "operator": operator.to_dict(include_dense=False),
        "n_random_tests": n_random_tests,
        "adjoint_tolerance": adjoint_tolerance,
    }


__all__ = [
    "SCIPY_AVAILABLE",
    "ArrayLike",
    "Matvec",
    "LinearSolveStatus",
    "LinearSolverMethod",
    "PreconditionerProtocol",
    "LinearOperator",
    "IterativeLinearSolveConfig",
    "IterativeLinearSolveResult",
    "as_linear_operator",
    "solve_linear_system",
    "estimate_operator_norm",
    "residual_diagnostics",
    "solve_multiple_rhs",
    "validate_linear_operator",
]
