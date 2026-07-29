"""The cached scatter map must reproduce the reference coupled Jacobian exactly.

``real_coupled_fast`` skips the reference assembly entirely: it rebuilds
``M.data`` from a precomputed map of khat entries to matrix slots. Nothing else
in the suite compares the two, so a wrong index or a dropped sign would show up
only as a silently worse preconditioner -- GMRES would still converge to the
same root, just slower, which no physics gate would catch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from twpa_solver.builders.jc_doc import build_jpa, build_jtwpa
from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import build_sideband_matched_basis
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump.backends.fast_coupled import FastCoupledPreconditioner


def _problem(sidebands: int = 2) -> FullMultiToneProblem:
    builder, _ = build_jpa()
    arrays = builder.assemble()
    circuit = CircuitMatrices(
        C=arrays["C"], G=arrays["G"], K=arrays["K"], Bphi=arrays["Bphi"],
        Ic=arrays["Ic"], port_to_index=arrays["ports"],
    )
    omega_p = 2.0 * math.pi * 4.75001e9
    delta = omega_p - 2.0 * math.pi * 4.75e9
    basis = build_sideband_matched_basis(
        [1, 3, 5], sidebands, omega_p, delta, omega_p * 22.0
    )
    drive = MultiToneDrive(
        basis.pump_tone, circuit.port_to_index[1], 1.13e-8
    ).to_coeffs(basis, circuit.node_count)
    return FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(drive)
    )


def _chain_problem() -> FullMultiToneProblem:
    """A distributed device, where the node-major band actually exists.

    The JPA is lumped -- a handful of nodes -- so its packed Jacobian is
    effectively dense and says nothing about banded storage. The banded backend
    targets 1-D chains, so it has to be exercised on one.
    """
    builder, _ = build_jtwpa()
    arrays = builder.assemble()
    circuit = CircuitMatrices(
        C=arrays["C"], G=arrays["G"], K=arrays["K"], Bphi=arrays["Bphi"],
        Ic=arrays["Ic"], port_to_index=arrays["ports"],
    )
    omega_p = 2.0 * math.pi * 7.12e9
    delta = omega_p - 2.0 * math.pi * 6.6e9
    basis = build_sideband_matched_basis([1, 3], 2, omega_p, delta, omega_p * 22.0)
    drive = MultiToneDrive(
        basis.pump_tone, circuit.port_to_index[1], 3.7e-6
    ).to_coeffs(basis, circuit.node_count)
    return FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(drive)
    )


def _excited_state(problem: FullMultiToneProblem) -> np.ndarray:
    """A state with every tone populated, so no block is accidentally zero."""
    rng = np.random.default_rng(7)
    X = problem.zeros()
    X[...] = 1e-8 * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
    X[problem.basis.index_of(problem.basis.pump_tone)] += 1e-7
    return X


def test_fast_scatter_reproduces_the_reference_coupled_matrix() -> None:
    problem = _problem()
    X = _excited_state(problem)
    tangent = problem.tangent_state(X)

    fast = FastCoupledPreconditioner(problem)
    fast.refactor(tangent)

    reference = problem.real_coupled_matrix(
        problem.spectral_tangent_state(tangent)
    ).tocsr()
    reference.sort_indices()
    difference = abs(fast.M - reference)

    assert difference.max() <= 1e-12 * abs(reference).max()


def test_fast_scatter_is_exact_for_every_quadrant_of_the_real_packing() -> None:
    """Check each quadrant separately.

    A sign error confined to one quadrant (the conjugate ``k+q`` term enters
    ``rr``/``ii`` with opposite signs) can stay small relative to the whole
    matrix norm while being completely wrong where it lives.
    """
    problem = _problem()
    X = _excited_state(problem)
    tangent = problem.tangent_state(X)

    fast = FastCoupledPreconditioner(problem)
    fast.refactor(tangent)
    reference = problem.real_coupled_matrix(
        problem.spectral_tangent_state(tangent)
    ).tocsr()

    half = reference.shape[0] // 2
    quadrants = {
        "rr": (slice(0, half), slice(0, half)),
        "ri": (slice(0, half), slice(half, None)),
        "ir": (slice(half, None), slice(0, half)),
        "ii": (slice(half, None), slice(half, None)),
    }
    for name, (rows, cols) in quadrants.items():
        expected = reference[rows, cols]
        actual = fast.M.tocsr()[rows, cols]
        scale = abs(expected).max()
        assert scale > 0.0, f"quadrant {name} is identically zero -- test is vacuous"
        assert abs(actual - expected).max() <= 1e-12 * scale, f"quadrant {name}"


@pytest.mark.parametrize("sidebands", [2, 4])
def test_fast_scatter_holds_as_the_tone_basis_grows(sidebands: int) -> None:
    """Tone count drives which (k-q, k+q) pairs exist, so vary it."""
    problem = _problem(sidebands)
    X = _excited_state(problem)
    tangent = problem.tangent_state(X)

    fast = FastCoupledPreconditioner(problem)
    fast.refactor(tangent)
    reference = problem.real_coupled_matrix(
        problem.spectral_tangent_state(tangent)
    ).tocsr()

    assert abs(fast.M - reference).max() <= 1e-12 * abs(reference).max()


def test_banded_backend_solves_the_same_system_as_the_sparse_one() -> None:
    """The banded factorization must be a factorization of the same matrix.

    It reorders to node-major and stores the factors in LAPACK band form; a
    wrong permutation or an off-by-one in the band offset still produces a
    plausible-looking solve, so compare against the sparse backend rather than
    only checking that it runs.
    """
    problem = _chain_problem()
    tangent = problem.tangent_state(_excited_state(problem))

    sparse = FastCoupledPreconditioner(problem, use_banded=False)
    banded = FastCoupledPreconditioner(problem, use_banded=True)
    sparse.refactor(tangent)
    banded.refactor(tangent)
    assert banded.last_factor_backend == "banded"

    rhs = np.random.default_rng(3).standard_normal(sparse.M.shape[0])
    expected = sparse.solve(rhs)
    actual = banded.solve(rhs)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_banded_backend_residual_is_small_against_the_assembled_matrix() -> None:
    """An independent check that does not trust the sparse backend either."""
    problem = _chain_problem()
    tangent = problem.tangent_state(_excited_state(problem))

    banded = FastCoupledPreconditioner(problem, use_banded=True)
    banded.refactor(tangent)

    rhs = np.random.default_rng(11).standard_normal(banded.M.shape[0])
    x = banded.solve(rhs)
    residual = np.linalg.norm(banded.M @ x - rhs) / np.linalg.norm(rhs)

    assert residual < 1e-8, f"banded solve residual {residual:.3e}"


def test_banded_bandwidth_is_measured_not_assumed() -> None:
    """The band must be wide enough to hold every nonzero it was built from."""
    problem = _chain_problem()
    banded = FastCoupledPreconditioner(problem, use_banded=True)
    banded.refactor(problem.tangent_state(_excited_state(problem)))

    coo = banded.M.tocoo()
    permutation = banded._band_permutation
    spread = np.abs(permutation[coo.row] - permutation[coo.col]).max()

    assert banded._band_kl >= spread
    # A band that degenerated to the full matrix would defeat the purpose.
    assert banded._band_kl < banded.M.shape[0] // 4


def test_refactor_is_idempotent_across_repeated_tangents() -> None:
    """Assembly must not accumulate into M.data across calls.

    ``M.data`` is rebuilt from ``Mconst`` every step; forgetting the copy makes
    the second factorization silently use twice the nonlinear term.
    """
    problem = _problem()
    tangent = problem.tangent_state(_excited_state(problem))

    fast = FastCoupledPreconditioner(problem)
    fast.refactor(tangent)
    first = fast.M.data.copy()
    fast.refactor(tangent)

    np.testing.assert_allclose(fast.M.data, first, rtol=0.0, atol=0.0)
