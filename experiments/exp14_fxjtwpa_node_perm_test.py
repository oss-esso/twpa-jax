# experiments/exp14_fxjtwpa_node_perm_test.py
"""Test the node-ordering hypothesis for the fxjtwpa JC seed.

Hypothesis: JC orders circuit nodes by sorted node-number, while exp10's
CircuitBuilder inserts them per cell in the order (node, node+3, node+2, node+1,
node+4), which is NOT sorted. For the simple jtwpa chain both orders coincide,
so the identity seed worked; for fxjtwpa they differ and the identity seed has a
real ~45 residual on the SQUID nodes.

The current seed X is indexed by JC's row order (== Python index under the
identity assumption). The correct value at Python index i is the JC row whose
node-name equals Python's label(i). If JC's order is sorted node-numbers, then
that JC row = sorted-rank(label(i)). So permute and re-measure the residual.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import exp08_full_ipm_pump_solve as exp08
import exp10_jc_doc_python_design_builders as builders

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
DESIGN = ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
SEED = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "pump" / "pump_solution.npz"
DC = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "dc" / "dc_solution.npz"
PUMP_FREQ_HZ = 20.0e9
NT = 64


def residual_l2(prob: exp08.FullIPMPumpProblem, X: np.ndarray) -> tuple[float, np.ndarray]:
    R = prob.residual_coeffs(X, source_scale=2.0)
    per = np.array([float(np.linalg.norm(R[h])) for h in range(X.shape[0])])
    return float(np.linalg.norm(R)), per


def main() -> None:
    ipm = exp08.load_ipm(DESIGN)
    cb, _ = builders.build_fxjtwpa()
    node_map = cb.node_map                      # label(str) -> python index
    inv = {v: k for k, v in node_map.items()}   # python index -> label(str)
    n = len(node_map)

    # JC order hypothesis = sorted by integer node number (ground "0" excluded
    # from the matrix; exp10 also excludes ground). Build sorted rank of labels.
    labels_sorted = sorted(node_map.keys(), key=lambda s: int(s))
    rank = {lab: r for r, lab in enumerate(labels_sorted)}
    # perm[i] = JC row index that holds Python node i's physical flux
    perm = np.array([rank[inv[i]] for i in range(n)], dtype=np.int64)

    seed = np.load(SEED)
    X = (seed["X_real"] + 1j * seed["X_imag"]).astype(np.complex128)
    modes = [int(m) for m in seed["pump_modes"]]
    dc = np.load(DC)
    psi_dc = np.asarray(dc["psi_dc"], dtype=np.float64).reshape(-1)

    grid = exp08.HarmonicGrid(modes=np.asarray(modes), nt=NT, omega=2 * math.pi * PUMP_FREQ_HZ)
    branch = exp08.JosephsonBranchArray(Ic=ipm.Ic, phi0=ipm.phi0)
    pump_node = int(ipm.port_to_index[3])
    prob = exp08.FullIPMPumpProblem(
        C=ipm.C, G=ipm.G, K=ipm.K, Bphi=ipm.Bphi, branch=branch, grid=grid,
        pump_node_index=pump_node, pump_current_a=1.195e-5,
        dc_branch_flux=psi_dc, source_mode=1,
    )

    tot0, per0 = residual_l2(prob, X)
    print(f"identity seed:   total |R|={tot0:.4e}  mode1={per0[0]:.4e}")

    Xp = X[:, perm]
    totp, perp = residual_l2(prob, Xp)
    print(f"permuted seed:   total |R|={totp:.4e}  mode1={perp[0]:.4e}")

    # also try the DC branch flux permuted? psi_dc is per-branch, not per-node,
    # so node perm doesn't apply directly; report node-perm effect on AC only.
    # sanity: is perm a true permutation?
    print(f"perm is bijection: {len(set(perm.tolist())) == n}")


if __name__ == "__main__":
    main()
