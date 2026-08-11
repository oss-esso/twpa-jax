"""Arnoldi extraction and classification of matrix-free Floquet multipliers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from twpa_solver.stability.monodromy import MonodromyOperator


@dataclass(frozen=True)
class FloquetResult:
    """Dominant multipliers returned by one matrix-free Arnoldi solve."""

    multipliers: np.ndarray
    vectors: np.ndarray
    spectral_radius: float
    runtime_s: float
    matvecs: int
    requested_eigenvalues: int
    which: str
    converged: bool
    message: str

    @property
    def magnitudes(self) -> np.ndarray:
        return np.abs(self.multipliers)

    @property
    def angles(self) -> np.ndarray:
        return np.angle(self.multipliers)


def compute_floquet_multipliers(
    operator: MonodromyOperator,
    *,
    eigenvalues: int = 12,
    which: str = "LM",
    maxiter: int | None = None,
    ncv: int | None = None,
    tol: float = 1.0e-8,
    seed: int = 0,
) -> FloquetResult:
    """Compute leading multipliers without materializing the monodromy matrix."""
    if eigenvalues < 1:
        raise ValueError("eigenvalues must be positive")
    if which not in {"LM", "SM"}:
        raise ValueError("which must be 'LM' or 'SM'")
    dimension = operator.shape[0]
    if eigenvalues >= dimension - 1:
        raise ValueError(
            "matrix-free Arnoldi requires eigenvalues < state_dimension - 1"
        )
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(dimension)
    v0 /= np.linalg.norm(v0)
    before = operator.matvec_count
    started = time.perf_counter()
    effective_ncv = ncv if ncv is not None else max(2 * eigenvalues + 2, 20)
    effective_ncv = min(effective_ncv, dimension)
    converged = True
    message = "converged"
    try:
        values, vectors = spla.eigs(
            operator.as_linear_operator(),
            k=eigenvalues,
            which=which,
            v0=v0,
            tol=tol,
            maxiter=maxiter,
            ncv=effective_ncv,
        )
    except spla.ArpackNoConvergence as exc:
        values = exc.eigenvalues
        vectors = exc.eigenvectors
        converged = False
        message = str(exc)
        if values is None or vectors is None or values.size == 0:
            values = np.empty(0, dtype=np.complex128)
            vectors = np.empty((dimension, 0), dtype=np.complex128)
    runtime_s = time.perf_counter() - started
    values = np.asarray(values, dtype=np.complex128)
    order = np.argsort(-np.abs(values))
    values = values[order]
    vectors = np.asarray(vectors[:, order], dtype=np.complex128)
    return FloquetResult(
        multipliers=values,
        vectors=vectors,
        spectral_radius=float(np.max(np.abs(values))) if values.size else float("nan"),
        runtime_s=float(runtime_s),
        matvecs=int(operator.matvec_count - before),
        requested_eigenvalues=eigenvalues,
        which=which,
        converged=converged,
        message=message,
    )


def floquet_exponents(
    multipliers: np.ndarray,
    period_seconds: float,
) -> np.ndarray:
    """Return principal-branch ``mu=log(lambda)/T`` Floquet exponents."""
    if period_seconds <= 0.0:
        raise ValueError("period_seconds must be positive")
    values = np.asarray(multipliers, dtype=np.complex128)
    return np.log(values) / period_seconds


def classify_multiplier(
    multiplier: complex,
    *,
    unit_circle_tolerance: float = 1.0e-3,
    crossing_tolerance_rad: float = 0.05,
) -> str:
    """Classify a near-unit multiplier as a bifurcation candidate only."""
    magnitude = abs(multiplier)
    if abs(magnitude - 1.0) > unit_circle_tolerance:
        return "STABLE_MODE" if magnitude < 1.0 else "UNSTABLE_MODE"
    angle = abs(float(np.angle(multiplier)))
    if angle <= crossing_tolerance_rad:
        return "+1_CROSSING_CANDIDATE"
    if abs(angle - math.pi) <= crossing_tolerance_rad:
        return "-1_CROSSING_CANDIDATE"
    return "COMPLEX_UNIT_CIRCLE_CANDIDATE"
