"""
twpa.solvers.preconditioners
============================

Preconditioners for structured TWPA harmonic-balance linear systems.

This module provides lightweight preconditioner objects used by Newton-Krylov
and iterative linear solvers.

Supported preconditioners
-------------------------
IDENTITY
    No-op preconditioner.

DIAGONAL
    Elementwise inverse diagonal preconditioner.

BLOCK_JACOBI
    Inverse of local diagonal blocks. This is the most important first
    preconditioner for block-banded TWPA Jacobians.

DENSE_INVERSE
    Dense inverse preconditioner for small diagnostics only.

CUSTOM
    User-supplied callable.

Design goal
-----------
The API is intentionally simple:

    P = build_preconditioner(operator_or_matrix, config)
    y = P(x)

where P approximates A^{-1} x.

For large industrial 100 mm / 20,000-cell problems, use BLOCK_JACOBI or later
structured variants. DENSE_INVERSE is only for small reduced systems and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

import numpy as np

import jax
import jax.numpy as jnp

from twpa.solvers.linear_solvers import LinearOperator, as_linear_operator
from twpa.solvers.block_banded import BlockBandedMatrix


ArrayLike = Any
PreconditionerCallable = Callable[[jax.Array], jax.Array]


class PreconditionerKind(str, Enum):
    """Supported preconditioner families."""

    IDENTITY = "identity"
    DIAGONAL = "diagonal"
    BLOCK_JACOBI = "block_jacobi"
    DENSE_INVERSE = "dense_inverse"
    CUSTOM = "custom"


class PreconditionerStatus(str, Enum):
    """Preconditioner construction status."""

    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class PreconditionerConfig:
    """
    Preconditioner configuration.

    Parameters
    ----------
    kind:
        Preconditioner family.
    regularization:
        Non-negative diagonal regularization added before inversion.
    diagonal_floor:
        Minimum absolute diagonal magnitude used by DIAGONAL to avoid division
        by zero.
    allow_dense_materialization:
        Whether matrix-free operators may be materialized as dense matrices.
        This should be false for industrial-size systems.
    require_success:
        Raise if construction falls back or fails.
    name:
        Diagnostic name.
    """

    kind: PreconditionerKind = PreconditionerKind.IDENTITY
    regularization: float = 0.0
    diagonal_floor: float = 1e-30
    allow_dense_materialization: bool = False
    require_success: bool = False
    name: str = "preconditioner"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PreconditionerKind(self.kind))
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if self.diagonal_floor <= 0.0:
            raise ValueError("diagonal_floor must be positive")

    def with_updates(self, **kwargs: Any) -> "PreconditionerConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "regularization": self.regularization,
            "diagonal_floor": self.diagonal_floor,
            "allow_dense_materialization": self.allow_dense_materialization,
            "require_success": self.require_success,
            "name": self.name,
        }


@dataclass(frozen=True)
class Preconditioner:
    """
    Callable preconditioner approximating ``A^{-1}``.

    Parameters
    ----------
    apply:
        Function returning ``M^{-1} x``.
    shape:
        Shape of the operator being preconditioned.
    kind:
        Preconditioner kind.
    status:
        Construction status.
    dtype:
        Dtype of vectors expected by the preconditioner.
    message:
        Human-readable construction note.
    metadata:
        Additional diagnostics.
    """

    apply: PreconditionerCallable
    shape: tuple[int, int]
    kind: PreconditionerKind
    status: PreconditionerStatus = PreconditionerStatus.READY
    dtype: Any = jnp.complex128
    message: str = ""
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("shape must be a pair")
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape dimensions must be positive")
        if self.shape[0] != self.shape[1]:
            raise ValueError("preconditioners currently require square shape")
        object.__setattr__(self, "shape", (int(self.shape[0]), int(self.shape[1])))
        object.__setattr__(self, "kind", PreconditionerKind(self.kind))
        object.__setattr__(self, "status", PreconditionerStatus(self.status))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def size(self) -> int:
        return self.shape[0]

    @property
    def ready(self) -> bool:
        return self.status in {PreconditionerStatus.READY, PreconditionerStatus.DEGRADED}

    def __call__(self, x: ArrayLike) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=self.dtype)
        if x_arr.shape != (self.size,):
            raise ValueError(f"x must have shape {(self.size,)}, got {x_arr.shape}")
        return jnp.asarray(self.apply(x_arr), dtype=self.dtype)

    def to_linear_operator(self) -> LinearOperator:
        """
        Convert the preconditioner into a LinearOperator.
        """
        return LinearOperator(
            shape=self.shape,
            matvec=lambda x: self(x),
            rmatvec=None,
            dense_matrix=None,
            dtype=self.dtype,
            name=f"{self.kind.value}_preconditioner",
            metadata={
                **dict(self.metadata or {}),
                "source": "Preconditioner.to_linear_operator",
                "status": self.status.value,
                "message": self.message,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "shape": tuple(int(v) for v in self.shape),
            "dtype": str(np.asarray(jnp.zeros((), dtype=self.dtype)).dtype),
            "message": self.message,
            "metadata": dict(self.metadata or {}),
        }


def identity_preconditioner(
    size: int,
    *,
    dtype: Any = jnp.complex128,
    message: str = "identity preconditioner",
) -> Preconditioner:
    """
    Build an identity/no-op preconditioner.
    """
    if int(size) <= 0:
        raise ValueError("size must be positive")
    n = int(size)

    return Preconditioner(
        apply=lambda x: x,
        shape=(n, n),
        kind=PreconditionerKind.IDENTITY,
        status=PreconditionerStatus.READY,
        dtype=dtype,
        message=message,
        metadata={"source": "identity_preconditioner"},
    )


def diagonal_preconditioner_from_diagonal(
    diagonal: ArrayLike,
    *,
    regularization: float = 0.0,
    diagonal_floor: float = 1e-30,
    name: str = "diagonal_preconditioner",
) -> Preconditioner:
    """
    Build an elementwise inverse-diagonal preconditioner.
    """
    d = jnp.asarray(diagonal)
    if d.ndim != 1:
        raise ValueError(f"diagonal must be 1D, got shape {d.shape}")
    if d.shape[0] <= 0:
        raise ValueError("diagonal may not be empty")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if diagonal_floor <= 0.0:
        raise ValueError("diagonal_floor must be positive")

    d_reg = d + regularization
    safe_abs = jnp.maximum(jnp.abs(d_reg), diagonal_floor)
    d_safe = jnp.where(jnp.abs(d_reg) >= diagonal_floor, d_reg, safe_abs + 0j)
    inv_d = 1.0 / d_safe

    def apply(x: jax.Array) -> jax.Array:
        return inv_d * x

    return Preconditioner(
        apply=apply,
        shape=(int(d.shape[0]), int(d.shape[0])),
        kind=PreconditionerKind.DIAGONAL,
        status=PreconditionerStatus.READY,
        dtype=d.dtype,
        message="diagonal inverse preconditioner constructed",
        metadata={
            "source": name,
            "regularization": regularization,
            "diagonal_floor": diagonal_floor,
            "min_abs_diagonal": float(jnp.min(jnp.abs(d))),
            "max_abs_diagonal": float(jnp.max(jnp.abs(d))),
        },
    )


def diagonal_preconditioner_from_dense(
    matrix: ArrayLike,
    *,
    regularization: float = 0.0,
    diagonal_floor: float = 1e-30,
) -> Preconditioner:
    """
    Build a diagonal preconditioner from the diagonal of a dense matrix.
    """
    A = jnp.asarray(matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    return diagonal_preconditioner_from_diagonal(
        jnp.diag(A),
        regularization=regularization,
        diagonal_floor=diagonal_floor,
        name="diagonal_preconditioner_from_dense",
    )


def dense_inverse_preconditioner(
    matrix: ArrayLike,
    *,
    regularization: float = 0.0,
    name: str = "dense_inverse_preconditioner",
) -> Preconditioner:
    """
    Build a dense inverse preconditioner.

    This is for small diagnostic systems only.
    """
    A = jnp.asarray(matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")

    if regularization > 0.0:
        A_eff = A + regularization * jnp.eye(A.shape[0], dtype=A.dtype)
    else:
        A_eff = A

    inv_A = jnp.linalg.inv(A_eff)

    def apply(x: jax.Array) -> jax.Array:
        return inv_A @ x

    return Preconditioner(
        apply=apply,
        shape=(int(A.shape[0]), int(A.shape[1])),
        kind=PreconditionerKind.DENSE_INVERSE,
        status=PreconditionerStatus.READY,
        dtype=A.dtype,
        message="dense inverse preconditioner constructed",
        metadata={
            "source": name,
            "regularization": regularization,
            "matrix_shape": tuple(int(v) for v in A.shape),
            "matrix_norm": float(jnp.linalg.norm(A)),
            "inverse_norm": float(jnp.linalg.norm(inv_A)),
        },
    )


def block_jacobi_preconditioner(
    matrix: BlockBandedMatrix,
    *,
    regularization: float = 0.0,
) -> Preconditioner:
    """
    Build a block-Jacobi preconditioner from a BlockBandedMatrix.

    The preconditioner applies inverse diagonal blocks:

        y_i = D_i^{-1} x_i
    """
    inv_blocks = matrix.diagonal_preconditioner_blocks(regularization=regularization)

    def apply(x: jax.Array) -> jax.Array:
        return matrix.apply_block_jacobi_preconditioner(
            x,
            inverse_diagonal_blocks=inv_blocks,
        )

    return Preconditioner(
        apply=apply,
        shape=matrix.shape,
        kind=PreconditionerKind.BLOCK_JACOBI,
        status=PreconditionerStatus.READY,
        dtype=matrix.dtype,
        message="block-Jacobi preconditioner constructed",
        metadata={
            "source": "block_jacobi_preconditioner",
            "regularization": regularization,
            "matrix": matrix.to_dict(include_block_stats=False),
            "inverse_block_shape": tuple(int(v) for v in inv_blocks.shape),
        },
    )


def custom_preconditioner(
    apply: PreconditionerCallable,
    *,
    shape: tuple[int, int],
    dtype: Any = jnp.complex128,
    message: str = "custom preconditioner",
    metadata: Mapping[str, Any] | None = None,
) -> Preconditioner:
    """
    Wrap a user-supplied callable as a Preconditioner.
    """
    return Preconditioner(
        apply=apply,
        shape=shape,
        kind=PreconditionerKind.CUSTOM,
        status=PreconditionerStatus.READY,
        dtype=dtype,
        message=message,
        metadata={
            "source": "custom_preconditioner",
            **dict(metadata or {}),
        },
    )


def build_preconditioner(
    operator_or_matrix: LinearOperator | BlockBandedMatrix | ArrayLike | None,
    config: PreconditionerConfig | None = None,
    *,
    custom_apply: PreconditionerCallable | None = None,
    shape: tuple[int, int] | None = None,
    dtype: Any = jnp.complex128,
) -> Preconditioner:
    """
    Build a preconditioner from an operator, matrix, or block-banded matrix.

    Parameters
    ----------
    operator_or_matrix:
        Source matrix/operator. May be None only for IDENTITY or CUSTOM when
        shape is provided.
    config:
        Preconditioner configuration.
    custom_apply:
        Callable used when kind=CUSTOM.
    shape:
        Shape used when operator_or_matrix is None.
    dtype:
        Dtype used when shape-only construction is needed.
    """
    cfg = config or PreconditionerConfig()

    try:
        if cfg.kind == PreconditionerKind.IDENTITY:
            if operator_or_matrix is None:
                if shape is None:
                    raise ValueError("shape is required for identity preconditioner without operator")
                size = int(shape[0])
                dtype_out = dtype
            elif isinstance(operator_or_matrix, BlockBandedMatrix):
                size = operator_or_matrix.size
                dtype_out = operator_or_matrix.dtype
            else:
                op = as_linear_operator(operator_or_matrix)
                size = op.size_in
                dtype_out = op.dtype

            return identity_preconditioner(
                size,
                dtype=dtype_out,
                message="identity preconditioner selected",
            )

        if cfg.kind == PreconditionerKind.CUSTOM:
            if custom_apply is None:
                raise ValueError("custom_apply is required for CUSTOM preconditioner")
            if shape is None:
                if operator_or_matrix is None:
                    raise ValueError("shape is required for custom preconditioner without operator")
                if isinstance(operator_or_matrix, BlockBandedMatrix):
                    shape = operator_or_matrix.shape
                    dtype = operator_or_matrix.dtype
                else:
                    op = as_linear_operator(operator_or_matrix)
                    shape = op.shape
                    dtype = op.dtype

            return custom_preconditioner(
                custom_apply,
                shape=shape,
                dtype=dtype,
                message="custom preconditioner selected",
                metadata={"config": cfg.to_dict()},
            )

        if operator_or_matrix is None:
            raise ValueError(f"{cfg.kind.value} preconditioner requires an operator or matrix")

        if cfg.kind == PreconditionerKind.BLOCK_JACOBI:
            if isinstance(operator_or_matrix, BlockBandedMatrix):
                return block_jacobi_preconditioner(
                    operator_or_matrix,
                    regularization=cfg.regularization,
                )

            if cfg.allow_dense_materialization:
                op = as_linear_operator(operator_or_matrix)
                dense = op.to_dense()
                # Fall back to diagonal if we only have a dense generic matrix.
                return diagonal_preconditioner_from_dense(
                    dense,
                    regularization=cfg.regularization,
                    diagonal_floor=cfg.diagonal_floor,
                )

            raise ValueError(
                "BLOCK_JACOBI requires a BlockBandedMatrix unless "
                "allow_dense_materialization=True"
            )

        if cfg.kind == PreconditionerKind.DIAGONAL:
            if isinstance(operator_or_matrix, BlockBandedMatrix):
                diag_blocks = operator_or_matrix.diagonal_blocks
                diag = jnp.reshape(
                    jnp.stack([jnp.diag(diag_blocks[i]) for i in range(operator_or_matrix.n_blocks)]),
                    (operator_or_matrix.size,),
                )
                return diagonal_preconditioner_from_diagonal(
                    diag,
                    regularization=cfg.regularization,
                    diagonal_floor=cfg.diagonal_floor,
                    name="diagonal_from_block_banded",
                )

            op = as_linear_operator(operator_or_matrix)
            if not op.has_dense and not cfg.allow_dense_materialization:
                raise ValueError(
                    "DIAGONAL preconditioner requires dense_matrix or "
                    "allow_dense_materialization=True"
                )
            dense = op.to_dense()
            return diagonal_preconditioner_from_dense(
                dense,
                regularization=cfg.regularization,
                diagonal_floor=cfg.diagonal_floor,
            )

        if cfg.kind == PreconditionerKind.DENSE_INVERSE:
            if isinstance(operator_or_matrix, BlockBandedMatrix):
                dense = operator_or_matrix.to_dense()
            else:
                op = as_linear_operator(operator_or_matrix)
                if not op.has_dense and not cfg.allow_dense_materialization:
                    raise ValueError(
                        "DENSE_INVERSE requires dense_matrix or "
                        "allow_dense_materialization=True"
                    )
                dense = op.to_dense()

            return dense_inverse_preconditioner(
                dense,
                regularization=cfg.regularization,
            )

        raise ValueError(f"Unsupported preconditioner kind {cfg.kind}")

    except Exception as exc:
        if cfg.require_success:
            raise

        if operator_or_matrix is None:
            if shape is None:
                raise
            fallback_size = int(shape[0])
            fallback_dtype = dtype
        elif isinstance(operator_or_matrix, BlockBandedMatrix):
            fallback_size = operator_or_matrix.size
            fallback_dtype = operator_or_matrix.dtype
        else:
            op = as_linear_operator(operator_or_matrix)
            fallback_size = op.size_in
            fallback_dtype = op.dtype

        return Preconditioner(
            apply=lambda x: x,
            shape=(fallback_size, fallback_size),
            kind=PreconditionerKind.IDENTITY,
            status=PreconditionerStatus.DEGRADED,
            dtype=fallback_dtype,
            message=f"preconditioner construction failed; using identity fallback: {exc}",
            metadata={
                "requested_config": cfg.to_dict(),
                "fallback": "identity",
                "error": str(exc),
            },
        )


def validate_preconditioner(
    preconditioner: Preconditioner,
    *,
    n_random_tests: int = 3,
    seed: int = 123,
) -> dict[str, Any]:
    """
    Validate preconditioner shape, finite output, and deterministic application.
    """
    rng = np.random.default_rng(seed)
    messages: list[str] = []
    passed = True

    for k in range(n_random_tests):
        x_np = rng.standard_normal(preconditioner.size) + 1j * rng.standard_normal(preconditioner.size)
        x = jnp.asarray(x_np, dtype=preconditioner.dtype)

        try:
            y1 = preconditioner(x)
            y2 = preconditioner(x)
        except Exception as exc:
            passed = False
            messages.append(f"FAIL: preconditioner raised on test {k}: {exc}")
            continue

        if y1.shape != x.shape:
            passed = False
            messages.append(f"FAIL: output shape {y1.shape} != input shape {x.shape}")

        if not bool(jnp.all(jnp.isfinite(y1))):
            passed = False
            messages.append(f"FAIL: non-finite output on test {k}")

        deterministic_error = float(jnp.linalg.norm(y1 - y2))
        if deterministic_error > 0.0:
            passed = False
            messages.append(
                f"FAIL: non-deterministic output on test {k}: error={deterministic_error:.3e}"
            )

    if passed:
        messages.append("PASS: preconditioner validation checks passed.")

    return {
        "passed": passed,
        "messages": messages,
        "preconditioner": preconditioner.to_dict(),
        "n_random_tests": n_random_tests,
    }


def preconditioned_residual_diagnostics(
    operator_or_matrix: LinearOperator | BlockBandedMatrix | ArrayLike,
    preconditioner: Preconditioner,
    *,
    n_random_tests: int = 3,
    seed: int = 123,
) -> dict[str, Any]:
    """
    Estimate how well P approximates A^{-1} using random vectors.

    Reports norms of:

        P A x - x

    This is not a proof of quality, but it is useful for catching completely
    broken preconditioners.
    """
    if isinstance(operator_or_matrix, BlockBandedMatrix):
        op = operator_or_matrix.to_linear_operator()
    else:
        op = as_linear_operator(operator_or_matrix)

    if op.shape != preconditioner.shape:
        raise ValueError(f"operator shape {op.shape} != preconditioner shape {preconditioner.shape}")

    rng = np.random.default_rng(seed)
    errors = []

    for _ in range(n_random_tests):
        x_np = rng.standard_normal(op.size_in) + 1j * rng.standard_normal(op.size_in)
        x = jnp.asarray(x_np, dtype=op.dtype)
        y = preconditioner(op.apply(x))
        err = jnp.linalg.norm(y - x) / jnp.maximum(jnp.linalg.norm(x), 1e-300)
        errors.append(float(err))

    return {
        "operator_shape": op.shape,
        "preconditioner": preconditioner.to_dict(),
        "n_random_tests": n_random_tests,
        "relative_error_min": float(np.min(errors)) if errors else None,
        "relative_error_max": float(np.max(errors)) if errors else None,
        "relative_error_mean": float(np.mean(errors)) if errors else None,
        "errors": errors,
    }


__all__ = [
    "ArrayLike",
    "PreconditionerCallable",
    "PreconditionerKind",
    "PreconditionerStatus",
    "PreconditionerConfig",
    "Preconditioner",
    "identity_preconditioner",
    "diagonal_preconditioner_from_diagonal",
    "diagonal_preconditioner_from_dense",
    "dense_inverse_preconditioner",
    "block_jacobi_preconditioner",
    "custom_preconditioner",
    "build_preconditioner",
    "validate_preconditioner",
    "preconditioned_residual_diagnostics",
]