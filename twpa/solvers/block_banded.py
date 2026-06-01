"""
twpa.solvers.block_banded
=========================

Block-banded matrix containers and operations for structured TWPA solvers.

The industrial 100 mm / 20,000-cell harmonic-balance Jacobian is not a generic
dense matrix. For nearest-neighbour ladder models, the Jacobian has a repeated
local block structure:

    [D0 U0  0  0 ...]
    [L1 D1 U1 0 ...]
    [0  L2 D2 U2 ...]
    ...

where each block corresponds to all harmonic-balance unknowns at one cell/node
or one local stencil group.

This module provides a production-facing representation for such matrices:

    BlockBandedMatrix
        General block-banded container with arbitrary lower/upper block
        bandwidth.

    BlockBandedConfig
        Construction and validation settings.

    build_block_banded_from_dense
        Diagnostic converter from dense matrices to block-banded form.

The implementation is intentionally conservative and JAX-friendly. It supports:

    - shape validation,
    - dense reconstruction for small tests,
    - block-banded matvec,
    - diagonal/block-Jacobi extraction,
    - block-tridiagonal Thomas solve,
    - conversion to a LinearOperator for Newton-Krylov code.

The general direct solver is only implemented for block-tridiagonal systems.
For wider block bandwidths, use the matrix-free matvec with iterative solvers
or convert to dense only for small diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from twpa.solvers.linear_solvers import LinearOperator


ArrayLike = Any


class BlockBandedStorage(str, Enum):
    """Storage convention for block-banded matrices."""

    DIAGONAL_MAP = "diagonal_map"


class BlockBandedSolveMethod(str, Enum):
    """Supported block-banded solve methods."""

    AUTO = "auto"
    BLOCK_TRIDIAGONAL_THOMAS = "block_tridiagonal_thomas"
    DENSE = "dense"


@dataclass(frozen=True)
class BlockBandedConfig:
    """
    Configuration for block-banded construction and validation.

    Parameters
    ----------
    block_size:
        Size of each square block.
    lower_block_bandwidth:
        Number of block subdiagonals.
    upper_block_bandwidth:
        Number of block superdiagonals.
    drop_tolerance:
        Values below this absolute magnitude may be dropped during dense
        conversion.
    validate_blocks:
        Whether constructors should run shape checks.
    name:
        Diagnostic name.
    """

    block_size: int
    lower_block_bandwidth: int = 1
    upper_block_bandwidth: int = 1
    drop_tolerance: float = 0.0
    validate_blocks: bool = True
    storage: BlockBandedStorage = BlockBandedStorage.DIAGONAL_MAP
    name: str = "block_banded"

    def __post_init__(self) -> None:
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if int(self.lower_block_bandwidth) < 0:
            raise ValueError("lower_block_bandwidth must be non-negative")
        if int(self.upper_block_bandwidth) < 0:
            raise ValueError("upper_block_bandwidth must be non-negative")
        if self.drop_tolerance < 0.0:
            raise ValueError("drop_tolerance must be non-negative")

        object.__setattr__(self, "block_size", int(self.block_size))
        object.__setattr__(self, "lower_block_bandwidth", int(self.lower_block_bandwidth))
        object.__setattr__(self, "upper_block_bandwidth", int(self.upper_block_bandwidth))
        object.__setattr__(self, "storage", BlockBandedStorage(self.storage))

    @property
    def is_block_tridiagonal(self) -> bool:
        return self.lower_block_bandwidth <= 1 and self.upper_block_bandwidth <= 1

    def with_updates(self, **kwargs: Any) -> "BlockBandedConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "lower_block_bandwidth": self.lower_block_bandwidth,
            "upper_block_bandwidth": self.upper_block_bandwidth,
            "drop_tolerance": self.drop_tolerance,
            "validate_blocks": self.validate_blocks,
            "storage": self.storage.value,
            "is_block_tridiagonal": self.is_block_tridiagonal,
            "name": self.name,
        }


@dataclass(frozen=True)
class BlockBandedMatrix:
    """
    Block-banded matrix.

    Blocks are stored by diagonal offset:

        blocks[0]    has shape (n_blocks, block_size, block_size)
        blocks[+1]   has shape (n_blocks - 1, block_size, block_size)
        blocks[-1]   has shape (n_blocks - 1, block_size, block_size)

    More generally:

        blocks[k] has shape (n_blocks - abs(k), block_size, block_size)

    For offset k:
        block index b maps row block i and column block j where j = i + k.
        Therefore:
            k > 0: upper diagonal
            k < 0: lower diagonal
    """

    blocks: Mapping[int, jax.Array]
    block_size: int
    n_blocks: int
    dtype: Any = jnp.complex128
    name: str = "block_banded_matrix"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if int(self.n_blocks) <= 0:
            raise ValueError("n_blocks must be positive")

        bsize = int(self.block_size)
        nblk = int(self.n_blocks)

        normalized: dict[int, jax.Array] = {}
        for raw_offset, raw_blocks in self.blocks.items():
            offset = int(raw_offset)
            expected_n = nblk - abs(offset)
            if expected_n <= 0:
                raise ValueError(
                    f"Offset {offset} invalid for n_blocks={nblk}; no blocks fit."
                )

            arr = jnp.asarray(raw_blocks, dtype=self.dtype)
            expected_shape = (expected_n, bsize, bsize)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"blocks[{offset}] must have shape {expected_shape}, got {arr.shape}"
                )
            normalized[offset] = arr

        if 0 not in normalized:
            raise ValueError("Main block diagonal blocks[0] is required")

        object.__setattr__(self, "blocks", normalized)
        object.__setattr__(self, "block_size", bsize)
        object.__setattr__(self, "n_blocks", nblk)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_diagonals(
        cls,
        *,
        diagonal: ArrayLike,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        name: str = "block_tridiagonal_matrix",
        metadata: Mapping[str, Any] | None = None,
    ) -> "BlockBandedMatrix":
        """
        Build a block-tridiagonal matrix from lower/diagonal/upper block arrays.

        Shapes:
            diagonal: (N, B, B)
            lower:    (N-1, B, B), optional
            upper:    (N-1, B, B), optional
        """
        diag = jnp.asarray(diagonal)
        if diag.ndim != 3 or diag.shape[1] != diag.shape[2]:
            raise ValueError("diagonal must have shape (n_blocks, block_size, block_size)")

        n_blocks = int(diag.shape[0])
        block_size = int(diag.shape[1])
        blocks: dict[int, jax.Array] = {0: diag}

        if lower is not None:
            blocks[-1] = jnp.asarray(lower, dtype=diag.dtype)

        if upper is not None:
            blocks[1] = jnp.asarray(upper, dtype=diag.dtype)

        return cls(
            blocks=blocks,
            block_size=block_size,
            n_blocks=n_blocks,
            dtype=diag.dtype,
            name=name,
            metadata=metadata,
        )

    @classmethod
    def diagonal(
        cls,
        diagonal: ArrayLike,
        *,
        name: str = "block_diagonal_matrix",
        metadata: Mapping[str, Any] | None = None,
    ) -> "BlockBandedMatrix":
        """
        Build a block-diagonal matrix.
        """
        return cls.from_diagonals(
            diagonal=diagonal,
            lower=None,
            upper=None,
            name=name,
            metadata=metadata,
        )

    @property
    def shape(self) -> tuple[int, int]:
        n = self.n_blocks * self.block_size
        return (n, n)

    @property
    def size(self) -> int:
        return self.n_blocks * self.block_size

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(sorted(self.blocks.keys()))

    @property
    def lower_block_bandwidth(self) -> int:
        return max((max(0, -k) for k in self.blocks.keys()), default=0)

    @property
    def upper_block_bandwidth(self) -> int:
        return max((max(0, k) for k in self.blocks.keys()), default=0)

    @property
    def is_block_tridiagonal(self) -> bool:
        return self.lower_block_bandwidth <= 1 and self.upper_block_bandwidth <= 1

    @property
    def diagonal_blocks(self) -> jax.Array:
        return self.blocks[0]

    @property
    def lower_blocks(self) -> jax.Array | None:
        return self.blocks.get(-1)

    @property
    def upper_blocks(self) -> jax.Array | None:
        return self.blocks.get(1)

    def with_updates(self, **kwargs: Any) -> "BlockBandedMatrix":
        return replace(self, **kwargs)

    def block(self, row_block: int, col_block: int) -> jax.Array:
        """
        Return block A[row_block, col_block], or zeros if outside the band.
        """
        i = int(row_block)
        j = int(col_block)
        if not (0 <= i < self.n_blocks and 0 <= j < self.n_blocks):
            raise IndexError("block indices out of range")

        offset = j - i
        if offset not in self.blocks:
            return jnp.zeros((self.block_size, self.block_size), dtype=self.dtype)

        local_index = i if offset >= 0 else j
        return self.blocks[offset][local_index]

    def matvec(self, x: ArrayLike) -> jax.Array:
        """
        Compute y = A @ x.
        """
        x_arr = jnp.asarray(x, dtype=self.dtype)
        if x_arr.shape != (self.size,):
            raise ValueError(f"x must have shape {(self.size,)}, got {x_arr.shape}")

        xb = jnp.reshape(x_arr, (self.n_blocks, self.block_size))
        yb = jnp.zeros_like(xb)

        for offset, block_array in self.blocks.items():
            if offset >= 0:
                n = self.n_blocks - offset
                rows = jnp.arange(n)
                cols = rows + offset
                contribution = jnp.einsum("nij,nj->ni", block_array, xb[cols])
                yb = yb.at[rows].add(contribution)
            else:
                n = self.n_blocks + offset
                cols = jnp.arange(n)
                rows = cols - offset
                contribution = jnp.einsum("nij,nj->ni", block_array, xb[cols])
                yb = yb.at[rows].add(contribution)

        return jnp.reshape(yb, (self.size,))

    def rmatvec(self, x: ArrayLike) -> jax.Array:
        """
        Compute y = A^H @ x.
        """
        x_arr = jnp.asarray(x, dtype=self.dtype)
        if x_arr.shape != (self.size,):
            raise ValueError(f"x must have shape {(self.size,)}, got {x_arr.shape}")

        xb = jnp.reshape(x_arr, (self.n_blocks, self.block_size))
        yb = jnp.zeros_like(xb)

        for offset, block_array in self.blocks.items():
            if offset >= 0:
                n = self.n_blocks - offset
                rows = jnp.arange(n)
                cols = rows + offset
                contribution = jnp.einsum(
                    "nji,ni->nj",
                    jnp.conj(block_array),
                    xb[rows],
                )
                yb = yb.at[cols].add(contribution)
            else:
                n = self.n_blocks + offset
                cols = jnp.arange(n)
                rows = cols - offset
                contribution = jnp.einsum(
                    "nji,ni->nj",
                    jnp.conj(block_array),
                    xb[rows],
                )
                yb = yb.at[cols].add(contribution)

        return jnp.reshape(yb, (self.size,))

    def to_dense(self) -> jax.Array:
        """
        Reconstruct the full dense matrix.

        Intended for tests and diagnostics only.
        """
        A = jnp.zeros((self.size, self.size), dtype=self.dtype)

        for offset, block_array in self.blocks.items():
            if offset >= 0:
                n = self.n_blocks - offset
                for i in range(n):
                    j = i + offset
                    rs = slice(i * self.block_size, (i + 1) * self.block_size)
                    cs = slice(j * self.block_size, (j + 1) * self.block_size)
                    A = A.at[rs, cs].set(block_array[i])
            else:
                n = self.n_blocks + offset
                for j in range(n):
                    i = j - offset
                    rs = slice(i * self.block_size, (i + 1) * self.block_size)
                    cs = slice(j * self.block_size, (j + 1) * self.block_size)
                    A = A.at[rs, cs].set(block_array[j])

        return A

    def to_linear_operator(self) -> LinearOperator:
        """
        Convert to a matrix-free LinearOperator.
        """
        return LinearOperator(
            shape=self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dense_matrix=None,
            dtype=self.dtype,
            name=self.name,
            metadata={
                **dict(self.metadata or {}),
                "source": "BlockBandedMatrix.to_linear_operator",
                "offsets": self.offsets,
                "block_size": self.block_size,
                "n_blocks": self.n_blocks,
            },
        )

    def diagonal_preconditioner_blocks(
        self,
        *,
        regularization: float = 0.0,
    ) -> jax.Array:
        """
        Return inverse diagonal blocks for block-Jacobi preconditioning.
        """
        D = self.diagonal_blocks
        if regularization > 0.0:
            eye = jnp.eye(self.block_size, dtype=self.dtype)
            D = D + regularization * eye[None, :, :]

        inv_blocks = []
        for i in range(self.n_blocks):
            inv_blocks.append(jnp.linalg.inv(D[i]))
        return jnp.stack(inv_blocks, axis=0)

    def apply_block_jacobi_preconditioner(
        self,
        x: ArrayLike,
        *,
        inverse_diagonal_blocks: ArrayLike | None = None,
        regularization: float = 0.0,
    ) -> jax.Array:
        """
        Apply block-Jacobi preconditioner M^{-1} x.
        """
        x_arr = jnp.asarray(x, dtype=self.dtype)
        if x_arr.shape != (self.size,):
            raise ValueError(f"x must have shape {(self.size,)}, got {x_arr.shape}")

        invD = (
            self.diagonal_preconditioner_blocks(regularization=regularization)
            if inverse_diagonal_blocks is None
            else jnp.asarray(inverse_diagonal_blocks, dtype=self.dtype)
        )

        xb = jnp.reshape(x_arr, (self.n_blocks, self.block_size))
        yb = jnp.einsum("nij,nj->ni", invD, xb)
        return jnp.reshape(yb, (self.size,))

    def transpose_conjugate(self) -> "BlockBandedMatrix":
        """
        Return A^H as a BlockBandedMatrix.
        """
        new_blocks: dict[int, jax.Array] = {}

        for offset, block_array in self.blocks.items():
            new_offset = -offset
            new_blocks[new_offset] = jnp.conj(jnp.swapaxes(block_array, -1, -2))

        return BlockBandedMatrix(
            blocks=new_blocks,
            block_size=self.block_size,
            n_blocks=self.n_blocks,
            dtype=self.dtype,
            name=f"{self.name}_H",
            metadata={
                **dict(self.metadata or {}),
                "source": "transpose_conjugate",
                "original_name": self.name,
            },
        )

    def scale(self, factor: complex | float) -> "BlockBandedMatrix":
        """
        Return factor * A.
        """
        return BlockBandedMatrix(
            blocks={k: factor * v for k, v in self.blocks.items()},
            block_size=self.block_size,
            n_blocks=self.n_blocks,
            dtype=self.dtype,
            name=f"{self.name}_scaled",
            metadata={
                **dict(self.metadata or {}),
                "scale_factor": {
                    "real": float(np.real(factor)),
                    "imag": float(np.imag(factor)),
                },
            },
        )

    def add_diagonal_regularization(self, regularization: float) -> "BlockBandedMatrix":
        """
        Return A + regularization * I.
        """
        if regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if regularization == 0.0:
            return self

        eye = jnp.eye(self.block_size, dtype=self.dtype)
        blocks = dict(self.blocks)
        blocks[0] = blocks[0] + regularization * eye[None, :, :]

        return BlockBandedMatrix(
            blocks=blocks,
            block_size=self.block_size,
            n_blocks=self.n_blocks,
            dtype=self.dtype,
            name=f"{self.name}_regularized",
            metadata={
                **dict(self.metadata or {}),
                "regularization": regularization,
            },
        )

    def to_dict(self, *, include_block_stats: bool = True) -> dict[str, Any]:
        out = {
            "name": self.name,
            "shape": self.shape,
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
            "offsets": list(self.offsets),
            "lower_block_bandwidth": self.lower_block_bandwidth,
            "upper_block_bandwidth": self.upper_block_bandwidth,
            "is_block_tridiagonal": self.is_block_tridiagonal,
            "dtype": str(np.asarray(jnp.zeros((), dtype=self.dtype)).dtype),
            "metadata": dict(self.metadata or {}),
        }

        if include_block_stats:
            out["block_stats"] = {
                str(k): {
                    "shape": tuple(int(v) for v in arr.shape),
                    "min_abs": float(jnp.min(jnp.abs(arr))) if arr.size else None,
                    "max_abs": float(jnp.max(jnp.abs(arr))) if arr.size else None,
                    "fro_norm": float(jnp.linalg.norm(arr)),
                }
                for k, arr in self.blocks.items()
            }

        return out


def build_block_banded_from_dense(
    dense: ArrayLike,
    config: BlockBandedConfig,
    *,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BlockBandedMatrix:
    """
    Convert a dense square matrix to BlockBandedMatrix by extracting block bands.
    """
    A = jnp.asarray(dense)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("dense matrix must be square")

    block_size = config.block_size
    n = int(A.shape[0])
    if n % block_size != 0:
        raise ValueError(
            f"dense size {n} is not divisible by block_size {block_size}"
        )

    n_blocks = n // block_size
    blocks: dict[int, jax.Array] = {}

    for offset in range(-config.lower_block_bandwidth, config.upper_block_bandwidth + 1):
        n_diag = n_blocks - abs(offset)
        if n_diag <= 0:
            continue

        extracted = []
        for local in range(n_diag):
            if offset >= 0:
                i = local
                j = local + offset
            else:
                j = local
                i = local - offset

            rs = slice(i * block_size, (i + 1) * block_size)
            cs = slice(j * block_size, (j + 1) * block_size)
            block = A[rs, cs]

            if config.drop_tolerance > 0.0:
                block = jnp.where(jnp.abs(block) < config.drop_tolerance, 0.0, block)

            extracted.append(block)

        blocks[offset] = jnp.stack(extracted, axis=0)

    return BlockBandedMatrix(
        blocks=blocks,
        block_size=block_size,
        n_blocks=n_blocks,
        dtype=A.dtype,
        name=name or config.name,
        metadata={
            "source": "build_block_banded_from_dense",
            "dense_shape": tuple(int(v) for v in A.shape),
            "config": config.to_dict(),
            **dict(metadata or {}),
        },
    )


def build_block_tridiagonal_from_dense(
    dense: ArrayLike,
    *,
    block_size: int,
    drop_tolerance: float = 0.0,
    name: str = "block_tridiagonal_from_dense",
) -> BlockBandedMatrix:
    """
    Convenience dense-to-block-tridiagonal converter.
    """
    return build_block_banded_from_dense(
        dense,
        BlockBandedConfig(
            block_size=block_size,
            lower_block_bandwidth=1,
            upper_block_bandwidth=1,
            drop_tolerance=drop_tolerance,
            name=name,
        ),
    )


def block_banded_residual_against_dense(
    matrix: BlockBandedMatrix,
    dense: ArrayLike,
) -> dict[str, Any]:
    """
    Compare dense reconstruction against a reference dense matrix.
    """
    A_ref = jnp.asarray(dense, dtype=matrix.dtype)
    A = matrix.to_dense()

    if A.shape != A_ref.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {A_ref.shape}")

    diff = A - A_ref
    ref_norm = float(jnp.linalg.norm(A_ref))
    diff_norm = float(jnp.linalg.norm(diff))

    return {
        "dense_shape": tuple(int(v) for v in A_ref.shape),
        "block_banded": matrix.to_dict(include_block_stats=False),
        "diff_norm": diff_norm,
        "relative_diff_norm": diff_norm / max(ref_norm, 1e-300),
        "max_abs_diff": float(jnp.max(jnp.abs(diff))),
        "reference_norm": ref_norm,
    }


def block_tridiagonal_solve(
    matrix: BlockBandedMatrix,
    rhs: ArrayLike,
    *,
    regularization: float = 0.0,
) -> jax.Array:
    """
    Solve a block-tridiagonal linear system using block Thomas elimination.

    This supports square systems with offsets subset of {-1, 0, +1}.

    Parameters
    ----------
    matrix:
        Block-tridiagonal matrix.
    rhs:
        RHS vector of shape (n_blocks * block_size,) or matrix of shape
        (n_blocks * block_size, n_rhs).
    regularization:
        Optional diagonal regularization.

    Returns
    -------
    x:
        Solution with same trailing RHS shape as input.
    """
    if not matrix.is_block_tridiagonal:
        raise ValueError("block_tridiagonal_solve requires block-tridiagonal matrix")

    b = jnp.asarray(rhs, dtype=matrix.dtype)
    rhs_was_vector = b.ndim == 1

    if b.ndim == 1:
        if b.shape != (matrix.size,):
            raise ValueError(f"rhs must have shape {(matrix.size,)}, got {b.shape}")
        B = jnp.reshape(b, (matrix.n_blocks, matrix.block_size, 1))
    elif b.ndim == 2:
        if b.shape[0] != matrix.size:
            raise ValueError(f"rhs first dimension must be {matrix.size}, got {b.shape[0]}")
        B = jnp.reshape(b, (matrix.n_blocks, matrix.block_size, b.shape[1]))
    else:
        raise ValueError("rhs must be 1D or 2D")

    D = matrix.diagonal_blocks
    if regularization > 0.0:
        D = D + regularization * jnp.eye(matrix.block_size, dtype=matrix.dtype)[None, :, :]

    L = matrix.lower_blocks
    U = matrix.upper_blocks

    if L is None:
        L = jnp.zeros((matrix.n_blocks - 1, matrix.block_size, matrix.block_size), dtype=matrix.dtype)
    if U is None:
        U = jnp.zeros((matrix.n_blocks - 1, matrix.block_size, matrix.block_size), dtype=matrix.dtype)

    # Forward elimination.
    C_prime = []
    D_prime = []
    B_prime = []

    D0_inv = jnp.linalg.inv(D[0])
    if matrix.n_blocks > 1:
        C_prime.append(D0_inv @ U[0])
    B_prime.append(D0_inv @ B[0])
    D_prime.append(D[0])

    for i in range(1, matrix.n_blocks):
        effective_D = D[i] - L[i - 1] @ C_prime[i - 1]
        effective_D_inv = jnp.linalg.inv(effective_D)

        if i < matrix.n_blocks - 1:
            C_prime.append(effective_D_inv @ U[i])

        effective_B = B[i] - L[i - 1] @ B_prime[i - 1]
        B_prime.append(effective_D_inv @ effective_B)
        D_prime.append(effective_D)

    # Back substitution.
    X = [None for _ in range(matrix.n_blocks)]
    X[-1] = B_prime[-1]

    for i in range(matrix.n_blocks - 2, -1, -1):
        X[i] = B_prime[i] - C_prime[i] @ X[i + 1]

    Xb = jnp.stack(X, axis=0)

    if rhs_was_vector:
        return jnp.reshape(Xb[:, :, 0], (matrix.size,))

    return jnp.reshape(Xb, (matrix.size, B.shape[-1]))


def solve_block_banded(
    matrix: BlockBandedMatrix,
    rhs: ArrayLike,
    *,
    method: BlockBandedSolveMethod = BlockBandedSolveMethod.AUTO,
    regularization: float = 0.0,
) -> jax.Array:
    """
    Solve a block-banded linear system.

    For block-tridiagonal matrices this uses block Thomas elimination.
    For wider matrices, dense solve is available only when requested.
    """
    method = BlockBandedSolveMethod(method)

    if method == BlockBandedSolveMethod.AUTO:
        method = (
            BlockBandedSolveMethod.BLOCK_TRIDIAGONAL_THOMAS
            if matrix.is_block_tridiagonal
            else BlockBandedSolveMethod.DENSE
        )

    if method == BlockBandedSolveMethod.BLOCK_TRIDIAGONAL_THOMAS:
        return block_tridiagonal_solve(
            matrix,
            rhs,
            regularization=regularization,
        )

    if method == BlockBandedSolveMethod.DENSE:
        A = matrix.to_dense()
        if regularization > 0.0:
            A = A + regularization * jnp.eye(matrix.size, dtype=matrix.dtype)
        b = jnp.asarray(rhs, dtype=matrix.dtype)
        return jnp.linalg.solve(A, b)

    raise ValueError(f"Unsupported block-banded solve method {method}")


def make_block_jacobi_linear_operator(
    matrix: BlockBandedMatrix,
    *,
    regularization: float = 0.0,
    name: str | None = None,
) -> LinearOperator:
    """
    Build a LinearOperator that applies the block-Jacobi inverse.
    """
    inv_blocks = matrix.diagonal_preconditioner_blocks(regularization=regularization)

    def mv(x: jax.Array) -> jax.Array:
        return matrix.apply_block_jacobi_preconditioner(
            x,
            inverse_diagonal_blocks=inv_blocks,
        )

    return LinearOperator(
        shape=matrix.shape,
        matvec=mv,
        rmatvec=mv,
        dense_matrix=None,
        dtype=matrix.dtype,
        name=name or f"{matrix.name}_block_jacobi_preconditioner",
        metadata={
            "source": "make_block_jacobi_linear_operator",
            "matrix": matrix.to_dict(include_block_stats=False),
            "regularization": regularization,
        },
    )


def validate_block_banded_matrix(
    matrix: BlockBandedMatrix,
    *,
    n_random_tests: int = 3,
    seed: int = 123,
    dense_tolerance: float = 1e-9,
    solve_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """
    Validate block-banded matvec and optional block-tridiagonal solve.
    """
    rng = np.random.default_rng(seed)
    messages: list[str] = []
    passed = True

    A_dense = matrix.to_dense()

    for k in range(n_random_tests):
        x_np = rng.standard_normal(matrix.size) + 1j * rng.standard_normal(matrix.size)
        x = jnp.asarray(x_np, dtype=matrix.dtype)

        y_banded = matrix.matvec(x)
        y_dense = A_dense @ x

        err = float(jnp.linalg.norm(y_banded - y_dense))
        scale = max(float(jnp.linalg.norm(y_dense)), 1.0)
        rel = err / scale

        if rel > dense_tolerance:
            passed = False
            messages.append(
                f"FAIL: matvec relative error {rel:.3e} exceeds "
                f"{dense_tolerance:.3e} on test {k}"
            )

    if matrix.is_block_tridiagonal:
        for k in range(n_random_tests):
            x_true_np = rng.standard_normal(matrix.size) + 1j * rng.standard_normal(matrix.size)
            x_true = jnp.asarray(x_true_np, dtype=matrix.dtype)
            b = matrix.matvec(x_true)

            try:
                x_sol = block_tridiagonal_solve(matrix, b, regularization=0.0)
                residual = matrix.matvec(x_sol) - b
                residual_norm = float(jnp.linalg.norm(residual))
                rhs_norm = float(jnp.linalg.norm(b))
                rel = residual_norm / max(rhs_norm, 1e-300)

                if rel > solve_tolerance:
                    passed = False
                    messages.append(
                        f"FAIL: block solve relative residual {rel:.3e} exceeds "
                        f"{solve_tolerance:.3e} on test {k}"
                    )
            except Exception as exc:
                passed = False
                messages.append(f"FAIL: block tridiagonal solve raised on test {k}: {exc}")

    if passed:
        messages.append("PASS: block-banded matrix validation checks passed.")

    return {
        "passed": passed,
        "messages": messages,
        "matrix": matrix.to_dict(include_block_stats=False),
        "n_random_tests": n_random_tests,
        "dense_tolerance": dense_tolerance,
        "solve_tolerance": solve_tolerance,
    }


def block_banded_memory_estimate(
    matrix: BlockBandedMatrix,
) -> dict[str, Any]:
    """
    Estimate dense versus block-banded storage.
    """
    block_entries = sum(int(np.prod(arr.shape)) for arr in matrix.blocks.values())
    dense_entries = matrix.size * matrix.size
    itemsize = np.dtype(np.asarray(jnp.zeros((), dtype=matrix.dtype)).dtype).itemsize

    return {
        "matrix_shape": matrix.shape,
        "block_size": matrix.block_size,
        "n_blocks": matrix.n_blocks,
        "offsets": list(matrix.offsets),
        "block_entries": block_entries,
        "dense_entries": dense_entries,
        "block_storage_bytes": block_entries * itemsize,
        "dense_storage_bytes": dense_entries * itemsize,
        "compression_ratio_dense_over_block": (
            dense_entries / max(block_entries, 1)
        ),
        "dtype_itemsize_bytes": itemsize,
    }


def random_block_tridiagonal_matrix(
    *,
    n_blocks: int,
    block_size: int,
    diagonal_shift: float = 5.0,
    offdiag_scale: float = 0.1,
    seed: int = 123,
    dtype: Any = jnp.complex128,
    name: str = "random_block_tridiagonal",
) -> BlockBandedMatrix:
    """
    Generate a diagonally dominant random block-tridiagonal matrix for tests.
    """
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = np.random.default_rng(seed)

    diag = []
    lower = []
    upper = []

    for _ in range(n_blocks):
        A = rng.standard_normal((block_size, block_size)) + 1j * rng.standard_normal((block_size, block_size))
        A = 0.1 * A
        A = A + diagonal_shift * np.eye(block_size)
        diag.append(A)

    for _ in range(n_blocks - 1):
        L = offdiag_scale * (
            rng.standard_normal((block_size, block_size))
            + 1j * rng.standard_normal((block_size, block_size))
        )
        U = offdiag_scale * (
            rng.standard_normal((block_size, block_size))
            + 1j * rng.standard_normal((block_size, block_size))
        )
        lower.append(L)
        upper.append(U)

    return BlockBandedMatrix.from_diagonals(
        diagonal=jnp.asarray(np.stack(diag), dtype=dtype),
        lower=None if not lower else jnp.asarray(np.stack(lower), dtype=dtype),
        upper=None if not upper else jnp.asarray(np.stack(upper), dtype=dtype),
        name=name,
        metadata={
            "source": "random_block_tridiagonal_matrix",
            "diagonal_shift": diagonal_shift,
            "offdiag_scale": offdiag_scale,
            "seed": seed,
        },
    )


__all__ = [
    "ArrayLike",
    "BlockBandedStorage",
    "BlockBandedSolveMethod",
    "BlockBandedConfig",
    "BlockBandedMatrix",
    "build_block_banded_from_dense",
    "build_block_tridiagonal_from_dense",
    "block_banded_residual_against_dense",
    "block_tridiagonal_solve",
    "solve_block_banded",
    "make_block_jacobi_linear_operator",
    "validate_block_banded_matrix",
    "block_banded_memory_estimate",
    "random_block_tridiagonal_matrix",
]