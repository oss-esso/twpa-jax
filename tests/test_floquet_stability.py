"""Tests for twpa_solver.signal.stability: the Tier-1 Floquet stability proxy.

estimate_sigma_min is pure linear algebra (no circuit/pump state), so it is
tested directly against a dense SVD ground truth on synthetic sparse
matrices -- no CircuitMatrices/pump fixtures needed.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scripts.floquet_stability_sweep as floquet_sweep

from twpa_solver.signal.stability import (
    ComplexResonance,
    classify_floquet_resonance,
    estimate_sigma_min,
    local_minima,
    require_explicit_loss_model,
    refine_complex_resonance,
    refine_singular_omega,
)
from scripts.floquet_stability_sweep import (
    enforce_scan_density_guard,
    parse_args,
    recommended_scan_points,
    track_complex_resonance_branches,
)


def _random_complex_sparse(n: int, seed: int, density: float = 0.6) -> sp.csc_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    mask = rng.random((n, n)) < density
    dense = dense * mask
    # Keep the matrix well-conditioned but non-singular: push the diagonal up.
    dense += 3.0 * np.eye(n)
    return sp.csc_matrix(dense)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_estimate_sigma_min_matches_dense_svd(seed: int) -> None:
    A = _random_complex_sparse(n=12, seed=seed)
    expected = np.linalg.svd(A.toarray(), compute_uv=False).min()

    est = estimate_sigma_min(A, iters=25, seed=seed)

    assert est.matrix_size == 12
    assert est.convergence_ratio == pytest.approx(1.0, abs=0.05)
    assert est.sigma_min == pytest.approx(expected, rel=1e-3)


def test_estimate_sigma_min_near_singular_matrix_is_small() -> None:
    n = 8
    A = sp.identity(n, format="lil", dtype=np.complex128)
    A[0, 0] = 1e-8
    A = A.tocsc()

    est = estimate_sigma_min(A, iters=20, seed=0)

    assert est.sigma_min == pytest.approx(1e-8, rel=1e-2)


def test_estimate_sigma_min_is_deterministic_for_fixed_seed() -> None:
    A = _random_complex_sparse(n=10, seed=7)
    est1 = estimate_sigma_min(A, iters=10, seed=42)
    est2 = estimate_sigma_min(A, iters=10, seed=42)
    assert est1.sigma_min == est2.sigma_min


def test_local_minima_finds_dip_and_excludes_endpoints() -> None:
    values = [5.0, 4.0, 1.0, 3.0, 0.5, 2.0, 4.0]
    idx = local_minima(values, k=8)
    assert 4 in idx  # the deepest interior dip (value 0.5)
    assert idx[0] == 4  # ranked shallowest-first by depth (smallest first)
    assert 0 not in idx and len(values) - 1 not in idx


def test_local_minima_respects_k_limit() -> None:
    values = [5.0, 1.0, 5.0, 2.0, 5.0, 0.5, 5.0]
    idx = local_minima(values, k=1)
    assert len(idx) == 1
    assert idx[0] == 5  # the global minimum among interior points


def test_refine_singular_omega_finds_known_complex_eigenvalue() -> None:
    # A(omega) = omega*I - M is singular exactly at omega = eigenvalue of M --
    # a linear-in-omega matrix pencil, same singularity structure as the
    # physical (quadratic-in-omega) conversion matrix, ground-truthable via
    # a dense eig.
    rng = np.random.default_rng(3)
    n = 10
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    eigvals = np.linalg.eigvals(M)
    target = complex(eigvals[np.argmax(np.abs(eigvals.imag))])

    def assemble(omega: complex) -> sp.csc_matrix:
        return sp.csc_matrix(omega * np.eye(n, dtype=np.complex128) - M)

    omega0 = target + (0.05 + 0.05j)
    omega1 = target + (0.1 - 0.03j)
    result = refine_singular_omega(assemble, omega0, omega1, max_iters=50, tol=1e-10)

    assert result.converged
    assert abs(result.omega - target) < 1e-6
    assert result.residual < 1e-6


def test_seeded_refinement_stays_on_target_in_a_degenerate_cluster() -> None:
    """A supplied mode vector prevents a synthetic cluster hop."""
    import twpa_solver.signal.stability as stability

    cluster_offsets = np.array([0.0, 5.0e-10, -5.0e-10], dtype=float)
    target_vector = np.array([1.0, 0.0, 0.0], dtype=np.complex128)

    def run(offset: float, seeded: bool) -> complex:
        roots = 1.0 + offset + cluster_offsets + 0.2j
        matrix = np.diag(roots)

        def assemble(omega: complex) -> sp.csc_matrix:
            return sp.csc_matrix(omega * np.eye(3) - matrix)

        original_eigs = stability.spla.eigs

        def choose_cluster_member(
            matrix_arg: sp.spmatrix, **kwargs: object
        ) -> tuple[np.ndarray, np.ndarray]:
            vector = kwargs.get("v0")
            vector_array = None if vector is None else np.asarray(vector)
            index = (
                0
                if vector_array is not None and abs(vector_array[0]) > 0.5
                else 1
            )
            diagonal = np.asarray(matrix_arg.diagonal(), dtype=np.complex128)
            eigenvalue = diagonal[index]
            eigenvector = np.zeros((3, 1), dtype=np.complex128)
            eigenvector[index, 0] = 1.0
            return np.array([eigenvalue]), eigenvector

        stability.spla.eigs = choose_cluster_member
        try:
            result = refine_singular_omega(
                assemble,
                1.0 + offset + 0.01j,
                1.0 + offset + 0.02j,
                max_iters=10,
                tol=1e-12,
                v0=target_vector if seeded else None,
            )
        finally:
            stability.spla.eigs = original_eigs
        assert result.converged
        return result.omega

    seeded_roots = [run(offset, seeded=True) for offset in (0.0, 0.03)]
    blind_roots = [run(offset, seeded=False) for offset in (0.0, 0.03)]

    np.testing.assert_allclose(
        seeded_roots, [1.0 + offset + 0.2j for offset in (0.0, 0.03)]
    )
    assert any(
        abs(blind - seeded) > 1.0e-10
        for blind, seeded in zip(blind_roots, seeded_roots)
    )


def test_refine_complex_resonance_rejects_non_analytic_loss_model() -> None:
    with pytest.raises(ValueError, match="not analytic"):
        refine_complex_resonance(
            circuit=None,
            khat=None,
            omega_p=1.0,
            ms=[0],
            signal_ghz_guess=1.0,
            loss_model="conductance_abs_omega",
        )


def test_stability_requires_explicit_loss_model() -> None:
    with pytest.raises(ValueError, match="require an explicit loss_model"):
        require_explicit_loss_model(None)
    with pytest.raises(ValueError, match="require an explicit loss_model"):
        require_explicit_loss_model("default")
    assert require_explicit_loss_model("current_complex_c") == "current_complex_c"


def test_floquet_classification_identifies_period_doubling_multiplier() -> None:
    omega_p = 2.0 * np.pi * 10.0e9
    resonance = ComplexResonance(
        omega=0.5 * omega_p - 1.0j,
        signal_ghz=5.0 - 1.0j / (2.0 * np.pi * 1.0e9),
        eig_min=0.0j,
        growth_rate_per_s=1.0,
        converged=True,
        iterations=1,
        residual=0.0,
    )

    result = classify_floquet_resonance(resonance, omega_p)

    assert result.kind == "PERIOD_DOUBLING_CANDIDATE"
    assert result.multiplier.real == pytest.approx(-1.0, abs=1.0e-6)
    assert result.magnitude == pytest.approx(1.0, rel=1.0e-7)


def test_scan_density_guard_uses_measured_comb_spacing() -> None:
    args = parse_args([
        "--circuit-dir", "c", "--pump-dir", "p", "--out", "o",
        "--loss-model", "current_complex_c",
    ])
    recommended = recommended_scan_points(7.9, 241.7)

    assert 690 <= recommended <= 710
    args.n_points = recommended // 4
    enforce_scan_density_guard(args, 7.9)
    args.n_points -= 1
    with pytest.raises(ValueError, match="under-resolves"):
        enforce_scan_density_guard(args, 7.9)


def test_floquet_sweep_tracks_refined_roots_across_powers() -> None:
    def root(value: complex) -> dict[str, object]:
        return {"floquet": {
            "multiplier_real": value.real,
            "multiplier_imag": value.imag,
        }}

    sweeps = [
        {"complex_resonances": [root(0.8 + 0.1j), root(0.2 + 0.7j)]},
        {"complex_resonances": [root(0.25 + 0.68j), root(0.82 + 0.08j)]},
    ]

    track_complex_resonance_branches(sweeps)

    assert sweeps[1]["complex_resonances"][0]["tracked_branch_index"] == 0
    assert sweeps[1]["complex_resonances"][0]["floquet"]["multiplier_real"] == pytest.approx(0.82)
    assert sweeps[1]["max_abs_lambda"] == pytest.approx(abs(0.82 + 0.08j))


def test_hill_cli_writes_json_before_summary_printing(monkeypatch, tmp_path) -> None:
    target = {
        "pump_freq_ghz": 7.9,
        "signal_ghz": [1.0, 2.0],
        "resonances": [],
        "runtime_s": 0.0,
    }
    monkeypatch.setattr(floquet_sweep, "load_circuit", lambda _: object())
    monkeypatch.setattr(floquet_sweep, "_run_sweep", lambda *args: target.copy())
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("summary failure")))
    output = tmp_path / "sweep.json"

    with pytest.raises(RuntimeError, match="summary failure"):
        floquet_sweep.main([
            "--circuit-dir", "c",
            "--pump-dir", "p",
                "--out", str(output),
                "--n-points", "200",
                "--loss-model", "current_complex_c",
            ])

    assert output.exists()
