# experiments/exp14_fxjtwpa_residual_decomp.py
"""Decompose the JC-seed pump residual for fxjtwpa into its three physical parts.

The seed X = JC_nodeflux * phi0 gives mode-1 residual l2 ~= 45 in Python's pump
equation. The huge mutual-inductor K entries (proven identical to JC's calcinvLn)
nearly cancel, so the O(1) residual must come from the source injection or the
Josephson tangent/DC operating point. This script splits

    R_k = D_k X_k + N_k(X) - S_k

into the three terms per mode and prints which one is unbalanced, plus the nodes
that dominate the mode-1 residual.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import exp08_full_ipm_pump_solve as exp08

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
DESIGN = ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
SEED = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "pump" / "pump_solution.npz"
DC = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "dc" / "dc_solution.npz"

PUMP_FREQ_HZ = 20.0e9
NT = 64
SOURCE_SCALE = 2.0
PUMP_CURRENT_DESIGN_A = 1.195e-5  # design Ip; * scale 2 = 2.39e-5 source current
PUMP_PORT = 3


def main() -> None:
    ipm = exp08.load_ipm(DESIGN)
    seed = np.load(SEED)
    X = (seed["X_real"] + 1j * seed["X_imag"]).astype(np.complex128)
    modes = [int(m) for m in seed["pump_modes"]]
    H = X.shape[0]
    print(f"modes={modes} X.shape={X.shape} nodes={ipm.C.shape[0]}")

    dc = np.load(DC)
    dc_branch_flux = np.asarray(dc["psi_dc"], dtype=np.float64).reshape(-1)
    print(f"dc_branch_flux max/phi0={np.abs(dc_branch_flux).max()/ipm.phi0:.4f}")

    grid = exp08.HarmonicGrid(modes=np.asarray(modes), nt=NT, omega=2 * math.pi * PUMP_FREQ_HZ)
    branch = exp08.JosephsonBranchArray(Ic=ipm.Ic, phi0=ipm.phi0)

    # pump node index from port
    pump_node_index = int(ipm.port_to_index[PUMP_PORT])
    print(f"pump_node_index={pump_node_index}")

    prob = exp08.FullIPMPumpProblem(
        C=ipm.C, G=ipm.G, K=ipm.K, Bphi=ipm.Bphi, branch=branch, grid=grid,
        pump_node_index=pump_node_index, pump_current_a=PUMP_CURRENT_DESIGN_A,
        dc_branch_flux=dc_branch_flux, source_mode=1,
    )

    Ncoeff = prob.nonlinear_current_coeffs(X)
    S = prob.source_coeffs(SOURCE_SCALE)
    lin = np.empty_like(X)
    for h in range(H):
        lin[h] = prob._linear_blocks[h] @ X[h]
    R = lin + Ncoeff - S

    def l2(a: np.ndarray) -> float:
        return float(np.linalg.norm(a))

    print("\nper-mode l2:  k   |D.X|        |N|          |S|          |R|")
    for h, k in enumerate(modes):
        print(f"  k={k:2d}  {l2(lin[h]):.4e}  {l2(Ncoeff[h]):.4e}  {l2(S[h]):.4e}  {l2(R[h]):.4e}")

    # mode-1 dominant residual nodes
    h1 = modes.index(1)
    r1 = R[h1]
    order = np.argsort(-np.abs(r1))[:12]
    print("\nmode-1 top residual nodes (idx |R| |D.X| |N| |S|):")
    for idx in order:
        print(f"  {idx:5d}  {abs(r1[idx]):.4e}  {abs(lin[h1][idx]):.4e}  "
              f"{abs(Ncoeff[h1][idx]):.4e}  {abs(S[h1][idx]):.4e}")

    # Is the imbalance linear-vs-source, or linear-vs-JJ? Check total balances.
    print(f"\nglobal: |sum lin|={l2(lin):.4e} |sum N|={l2(Ncoeff):.4e} "
          f"|sum S|={l2(S):.4e} |sum R|={l2(R):.4e}")
    # On the residual-dominant nodes, is N or S nonzero?
    print(f"S nonzero entries: {np.count_nonzero(S)}  at node {pump_node_index} mode1")


if __name__ == "__main__":
    main()
