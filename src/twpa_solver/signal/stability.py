"""Floquet/Hill stability diagnostics on top of the existing gain-solve matrices.

The conversion matrix ``A(omega_s)`` assembled by
``twpa_solver.signal.floquet.assemble_conversion_matrix`` is exactly the
harmonic-balance Floquet/Hill determinant for small perturbations around a
converged pump state: ``A(omega) v = 0`` has a nontrivial solution iff
``omega`` is a Floquet exponent of the linear time-periodic system. Gain
solves only ever evaluate ``A`` at a real, forced probe frequency. This module
adds a cheap real-frequency stability proxy -- the smallest singular value of
``A(omega_s)`` -- without introducing any new solver math: same matrix, same
factorization cost as one gain solve, a handful of extra triangular solves.

A small ``sigma_min(A)`` means ``A`` is nearly singular at that frequency,
i.e. the system is close to (but not yet at, since this is a real-omega
sweep) a Floquet resonance. This is Tier 1 of a staged stability
investigation: a resonance found here is a candidate for Tier 2 (complex-omega
root refinement to get an actual growth/decay verdict), not a stability
verdict by itself.

Tier 2 (``refine_complex_resonance`` / ``refine_resonances``) takes a Tier 1
candidate frequency and refines it into the complex-omega plane by tracking
the eigenvalue of ``A(omega)`` nearest zero (ARPACK shift-invert at
``sigma=0``) and driving it to zero with a complex secant iteration. This
requires ``D(omega) = K - omega^2*C + i*omega*G`` to be analytic in omega --
true for ``current_complex_c`` and most loss models here (they are polynomial
in omega), false for anything using ``abs(omega)`` or a sign(omega) branch.
Under this codebase's ``e^{+i*omega*t}`` convention, a perturbation grows like
``e^{-Im(omega)*t}``, so ``growth_rate_per_s = -Im(omega) > 0`` is the
stability verdict Tier 1 alone cannot give.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.core.linear import LOSS_MODELS
from twpa_solver.signal.floquet import assemble_conversion_matrix

# These loss models are not analytic in omega, either via abs(omega) or a
# sign(omega)/sign(Re(omega)) branch. Fine for a real-omega sweep (Tier 1),
# but a complex-omega root search (Tier 2/3) must not be run against them
# without first replacing the discontinuity with an analytic continuation.
# ``complex_c_sign_omega`` compares ``omega >= 0.0``, which raises a bare
# TypeError for complex omega -- listed here so Tier 2 rejects it with a clear
# message instead of an obscure crash deep inside dynamic_block.
NON_ANALYTIC_LOSS_MODELS = (
    "conductance_abs_omega",
    "conductance_abs_omega_opposite",
    "complex_c_sign_omega",
)


@dataclass
class SigmaMinEstimate:
    sigma_min: float
    convergence_ratio: float
    matrix_size: int


def estimate_sigma_min(A: sp.spmatrix, iters: int = 8, seed: int = 0) -> SigmaMinEstimate:
    """Inverse-iteration estimate of the smallest singular value of ``A``.

    Reuses a single LU factorization of ``A`` (same cost as a normal gain
    solve) and applies ``(A^H A)^-1`` via two triangular solves per step. The
    dominant eigenvalue of ``(A^H A)^-1`` is ``1/sigma_min(A)^2``, so this
    converges to the smallest singular value, not the largest -- the
    direction a resonance shows up in.

    ``convergence_ratio`` is the ratio of the last two Rayleigh estimates; it
    should be close to 1.0 at convergence. A value far from 1.0 means
    ``iters`` was too few for this point (near-degenerate spectrum), and the
    returned ``sigma_min`` should be treated as approximate only.
    """
    lu = spla.splu(A.tocsc())
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v /= np.linalg.norm(v)

    prev_lambda: float | None = None
    ratio = float("nan")
    for _ in range(max(1, iters)):
        w = lu.solve(v, trans="H")
        x = lu.solve(w, trans="N")
        lam = float(np.linalg.norm(x))
        if not np.isfinite(lam) or lam <= 0.0:
            return SigmaMinEstimate(sigma_min=0.0, convergence_ratio=float("nan"), matrix_size=n)
        if prev_lambda is not None and prev_lambda > 0.0:
            ratio = lam / prev_lambda
        v = x / lam
        prev_lambda = lam

    sigma_min = 1.0 / math.sqrt(prev_lambda) if prev_lambda and prev_lambda > 0.0 else float("inf")
    return SigmaMinEstimate(sigma_min=sigma_min, convergence_ratio=ratio, matrix_size=n)


def sigma_min_at_signal_ghz(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    omega_p: float,
    signal_ghz: float,
    ms: list[int],
    loss_model: str = "current_complex_c",
    iters: int = 8,
    seed: int = 0,
) -> SigmaMinEstimate:
    omega_s = 2.0 * math.pi * float(signal_ghz) * 1e9
    A = assemble_conversion_matrix(
        circuit=circuit, khat=khat, omega_s=omega_s, omega_p=omega_p, ms=ms,
        loss_model=loss_model,
    )
    return estimate_sigma_min(A, iters=iters, seed=seed)


def sweep_sigma_min(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    omega_p: float,
    signal_ghz_grid: list[float],
    ms: list[int],
    loss_model: str = "current_complex_c",
    iters: int = 8,
    seed: int = 0,
) -> list[SigmaMinEstimate]:
    if loss_model not in LOSS_MODELS:
        raise ValueError(f"unknown loss_model={loss_model!r}")
    return [
        sigma_min_at_signal_ghz(
            circuit=circuit, khat=khat, omega_p=omega_p, signal_ghz=fs, ms=ms,
            loss_model=loss_model, iters=iters, seed=seed,
        )
        for fs in signal_ghz_grid
    ]


def local_minima(values: list[float], k: int = 8) -> list[int]:
    """Indices of local minima in ``values``, ranked by depth (smallest first).

    Endpoints are excluded since they are not bracketed. Returns at most ``k``
    indices.
    """
    idx = [
        i for i in range(1, len(values) - 1)
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]
    ]
    idx.sort(key=lambda i: values[i])
    return idx[:k]


@dataclass
class ComplexResonance:
    omega: complex  # rad/s
    signal_ghz: complex
    eig_min: complex
    growth_rate_per_s: float  # -Im(omega); > 0 means a growing (unstable) mode
    converged: bool
    iterations: int
    residual: float


def _nearest_zero_eigenpair(
    A: sp.spmatrix, v0: np.ndarray | None = None
) -> tuple[complex, np.ndarray]:
    """Eigenvalue (and eigenvector) of ``A`` closest to 0 via shift-invert Arnoldi.

    ``sigma=0`` shift-invert factorizes ``A`` once (ARPACK's own LU, not
    reused across calls -- refinement only needs a handful of these) and
    finds the largest-magnitude eigenvalue of ``A^-1``, i.e. the
    smallest-magnitude eigenvalue of ``A``. ``v0`` seeds Arnoldi with the
    previous iterate's eigenvector so consecutive omega steps track the same
    eigenvalue branch instead of jumping to an unrelated one.
    """
    n = A.shape[0]
    try:
        vals, vecs = spla.eigs(
            A.tocsc(), k=1, sigma=0.0, which="LM", v0=v0, maxiter=max(200, n * 50)
        )
    except spla.ArpackNoConvergence as exc:
        if exc.eigenvalues.size == 0:
            raise
        vals, vecs = exc.eigenvalues, exc.eigenvectors
    idx = int(np.argmin(np.abs(vals)))
    return complex(vals[idx]), np.asarray(vecs[:, idx])


def refine_singular_omega(
    assemble_fn: Callable[[complex], sp.spmatrix],
    omega0: complex,
    omega1: complex,
    max_iters: int = 30,
    tol: float = 1e-9,
) -> ComplexResonance:
    """Complex secant search for omega where ``assemble_fn(omega)`` is singular.

    Tracks ``f(omega) = eigenvalue of assemble_fn(omega) nearest zero`` (an
    analytic function of omega when ``assemble_fn`` is) and drives it to zero
    with the standard complex secant recurrence. Pure linear algebra -- no
    circuit/pump dependency -- so it is unit-tested directly against a known
    complex eigenvalue of a synthetic matrix pencil.
    """
    w_prev, w_curr = complex(omega0), complex(omega1)
    _, v_seed = _nearest_zero_eigenpair(assemble_fn(w_prev))
    f_prev, _ = _nearest_zero_eigenpair(assemble_fn(w_prev), v0=v_seed)
    f_curr, v_curr = _nearest_zero_eigenpair(assemble_fn(w_curr), v0=v_seed)

    converged = False
    iterations = 0
    for iterations in range(1, max_iters + 1):
        denom = f_curr - f_prev
        if abs(denom) < 1e-300:
            break
        w_next = w_curr - f_curr * (w_curr - w_prev) / denom
        f_next, v_next = _nearest_zero_eigenpair(assemble_fn(w_next), v0=v_curr)

        step_rel = abs(w_next - w_curr) / max(abs(w_next), 1e-300)
        w_prev, f_prev = w_curr, f_curr
        w_curr, f_curr, v_curr = w_next, f_next, v_next

        if step_rel < tol:
            converged = True
            break

    return ComplexResonance(
        omega=w_curr,
        signal_ghz=w_curr / (2.0 * math.pi * 1e9),
        eig_min=f_curr,
        growth_rate_per_s=float(-w_curr.imag),
        converged=converged,
        iterations=iterations,
        residual=float(abs(f_curr)),
    )


def refine_complex_resonance(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    omega_p: float,
    ms: list[int],
    signal_ghz_guess: float,
    loss_model: str = "current_complex_c",
    max_iters: int = 30,
    tol: float = 1e-9,
    perturbation: complex = 1e-4 + 1e-4j,
) -> ComplexResonance:
    """Refine a Tier 1 real-omega candidate into a complex Floquet exponent.

    ``perturbation`` (a small relative complex offset) seeds the secant
    search's second point off the real axis so it can move into the complex
    plane; the real-only Tier 1 guess is otherwise a degenerate 1-D slice of
    this 2-D root search.
    """
    if loss_model in NON_ANALYTIC_LOSS_MODELS:
        raise ValueError(
            f"loss_model={loss_model!r} is not analytic in omega (see "
            "NON_ANALYTIC_LOSS_MODELS); complex-omega root refinement (Tier 2) "
            "requires an analytic D(omega) and cannot be run against it."
        )

    def assemble(omega_complex: complex) -> sp.spmatrix:
        return assemble_conversion_matrix(
            circuit=circuit, khat=khat, omega_s=omega_complex, omega_p=omega_p,
            ms=ms, loss_model=loss_model,
        )

    omega0 = 2.0 * math.pi * complex(signal_ghz_guess) * 1e9
    omega1 = omega0 * (1.0 + perturbation)
    return refine_singular_omega(assemble, omega0, omega1, max_iters=max_iters, tol=tol)


def refine_resonances(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    omega_p: float,
    ms: list[int],
    candidates_ghz: list[float],
    loss_model: str = "current_complex_c",
    max_iters: int = 30,
    tol: float = 1e-9,
) -> list[ComplexResonance]:
    return [
        refine_complex_resonance(
            circuit=circuit, khat=khat, omega_p=omega_p, ms=ms,
            signal_ghz_guess=fs, loss_model=loss_model, max_iters=max_iters, tol=tol,
        )
        for fs in candidates_ghz
    ]
