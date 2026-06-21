"""Regression test for the fxjtwpa JC-seed node-order fix.

Root cause of the long-standing fxjtwpa non-convergence: exp10's CircuitBuilder
inserts nodes per cell in the order (node, node+3, node+2, node+1, node+4), which
is NOT sorted by node number, while JosephsonCircuits.jl orders nodes by sorted
node number. For the jtwpa chain the two orders coincide (so the identity seed
worked); for fxjtwpa they differ, and importing JC's nodeflux under the identity
assumption left a real ~45 pump residual on the SQUID nodes.

These tests pin the fix: the sorted-node-number permutation is a non-identity
bijection, and (when the seed artifacts are present) applying it drops the seed
pump residual from O(10) to the numerical floor.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = _ROOT / "experiments"
sys.path.insert(0, str(_EXPERIMENTS))

import exp08_full_ipm_pump_solve as exp08  # noqa: E402
import exp10_jc_doc_python_design_builders as builders  # noqa: E402


def _sorted_rank_perm() -> np.ndarray:
    """perm[i] = JC row (sorted node-number order) holding Python node i's flux."""
    cb, _ = builders.build_fxjtwpa()
    node_map = cb.node_map
    inv = {v: k for k, v in node_map.items()}
    labels_sorted = sorted(node_map.keys(), key=lambda s: int(s))
    rank = {lab: r for r, lab in enumerate(labels_sorted)}
    return np.array([rank[inv[i]] for i in range(len(node_map))], dtype=np.int64)


def test_fxjtwpa_node_order_is_unsorted() -> None:
    # The builder's index order must differ from sorted node number, otherwise
    # there is no bug to fix and the identity seed would already work.
    perm = _sorted_rank_perm()
    assert len(set(perm.tolist())) == perm.size, "perm must be a bijection"
    assert not np.array_equal(perm, np.arange(perm.size)), (
        "expected exp10 fxjtwpa node order to differ from sorted node number"
    )


def test_fxjtwpa_permutation_drops_seed_residual() -> None:
    design = _ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
    seed = _ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "pump" / "pump_solution.npz"
    dc = _ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "dc" / "dc_solution.npz"
    if not (design.exists() and seed.exists() and dc.exists()):
        pytest.skip("fxjtwpa seed artifacts not present")

    ipm = exp08.load_ipm(design)
    s = np.load(seed)
    X = (s["X_real"] + 1j * s["X_imag"]).astype(np.complex128)
    modes = [int(m) for m in s["pump_modes"]]
    psi_dc = np.asarray(np.load(dc)["psi_dc"], dtype=np.float64).reshape(-1)

    grid = exp08.HarmonicGrid(modes=np.asarray(modes), nt=64, omega=2 * math.pi * 20.0e9)
    branch = exp08.JosephsonBranchArray(Ic=ipm.Ic, phi0=ipm.phi0)
    prob = exp08.FullIPMPumpProblem(
        C=ipm.C, G=ipm.G, K=ipm.K, Bphi=ipm.Bphi, branch=branch, grid=grid,
        pump_node_index=int(ipm.port_to_index[3]), pump_current_a=1.195e-5,
        dc_branch_flux=psi_dc, source_mode=1,
    )

    r_identity = float(np.linalg.norm(prob.residual_coeffs(X, source_scale=2.0)))
    perm = _sorted_rank_perm()
    r_permuted = float(np.linalg.norm(prob.residual_coeffs(X[:, perm], source_scale=2.0)))

    assert r_identity > 1.0, f"expected large identity residual, got {r_identity}"
    assert r_permuted < 1e-3, f"permuted residual not at floor: {r_permuted}"
