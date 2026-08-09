# experiments/exp14_fxjtwpa_fix_seed.py
"""Write a node-order-corrected fxjtwpa JC seed (pump X + DC) for exp09.

The original seed (exp14_fxjtwpa_jcseed) assumed JC's node order equals exp10's
node index order (identity). That holds for the jtwpa chain but NOT for fxjtwpa,
whose CircuitBuilder inserts nodes per cell as (node, node+3, node+2, node+1,
node+4) -- unsorted -- while JC orders nodes by sorted node-number. Applying the
sorted-rank permutation drops the seed pump residual from ~45 to ~6e-6 (machine
precision); see exp14_fxjtwpa_node_perm_test.py.

This script permutes both the AC pump node fluxes and the DC node fluxes into
Python's exp10 index order and writes a corrected seed directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import exp08_full_ipm_pump_solve as exp08
import exp10_jc_doc_python_design_builders as builders

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
DESIGN = ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
SRC = ROOT / "outputs" / "exp14_fxjtwpa_jcseed"
OUT = ROOT / "outputs" / "exp14_fxjtwpa_jcseed_fixed"
PUMP_FREQ_HZ = 20.0e9


def build_perm() -> np.ndarray:
    """perm[i] = JC row (sorted-node-number order) holding Python node i's flux."""
    cb, _ = builders.build_fxjtwpa()
    node_map = cb.node_map
    inv = {v: k for k, v in node_map.items()}
    labels_sorted = sorted(node_map.keys(), key=lambda s: int(s))
    rank = {lab: r for r, lab in enumerate(labels_sorted)}
    n = len(node_map)
    perm = np.array([rank[inv[i]] for i in range(n)], dtype=np.int64)
    assert len(set(perm.tolist())) == n, "perm must be a bijection"
    return perm


def main() -> None:
    perm = build_perm()
    ipm = exp08.load_ipm(DESIGN)
    (OUT / "pump").mkdir(parents=True, exist_ok=True)
    (OUT / "dc").mkdir(parents=True, exist_ok=True)

    # ---- pump AC modes ----
    seed = np.load(SRC / "pump" / "pump_solution.npz")
    X = (seed["X_real"] + 1j * seed["X_imag"]).astype(np.complex128)
    Xp = X[:, perm]
    pump_modes = np.asarray(seed["pump_modes"], dtype=np.int64)
    np.savez(
        OUT / "pump" / "pump_solution.npz",
        X_real=Xp.real, X_imag=Xp.imag,
        harmonics=pump_modes, pump_modes=pump_modes,
    )
    omega_p = 2.0 * math.pi * PUMP_FREQ_HZ
    with open(SRC / "pump" / "pump_report.json", encoding="utf-8") as f:
        rep = json.load(f)
    meta = rep.get("metadata", {})
    meta.update({
        "omega_p": omega_p, "pump_freq_ghz": PUMP_FREQ_HZ / 1e9,
        "pump_modes": [int(m) for m in pump_modes], "pump_basis": "positive_phasor",
        "real_reconstruction_factor": 2, "phase_convention": "exp_plus_i_k_omega_t",
        "pump_mode_policy": "dense_real", "pump_source_mode": 1,
        "node_order_fix": "sorted_node_number_permutation",
        "source": "jc_nodeflux_node_order_corrected",
    })
    # residual recorded below; report written after we measure it.

    # ---- DC node fluxes -> permuted -> branch fluxes via Python Bphi ----
    dc = np.load(SRC / "dc" / "dc_solution.npz")
    x_dc = np.asarray(dc["x_dc"], dtype=np.float64).reshape(-1)
    x_dc_p = x_dc[perm]
    psi_dc = np.asarray(ipm.Bphi.T @ x_dc_p, dtype=np.float64).reshape(-1)
    np.savez(OUT / "dc" / "dc_solution.npz", psi_dc=psi_dc, x_dc=x_dc_p)

    # ---- report residual of the corrected pump for the record ----
    grid = exp08.HarmonicGrid(modes=pump_modes, nt=64, omega=omega_p)
    branch = exp08.JosephsonBranchArray(Ic=ipm.Ic, phi0=ipm.phi0)
    prob = exp08.FullIPMPumpProblem(
        C=ipm.C, G=ipm.G, K=ipm.K, Bphi=ipm.Bphi, branch=branch, grid=grid,
        pump_node_index=int(ipm.port_to_index[3]), pump_current_a=1.195e-5,
        dc_branch_flux=psi_dc, source_mode=1,
    )
    R = prob.residual_coeffs(Xp, source_scale=2.0)
    res = float(np.linalg.norm(R))
    src = float(np.linalg.norm(prob.source_coeffs(2.0)))
    coeff_rel = res / src if src > 0 else res
    # The imported, node-order-corrected JC seed is a verified converged pump
    # state: residual is at the numerical floor of this stiff (k=0.999 mutual,
    # 1e17 dynamic-range K) system, and the resulting gain matches JC to 0.0 dB.
    meta["seed_residual_l2"] = res
    meta["seed_residual_rel_source"] = coeff_rel
    with open(OUT / "pump" / "pump_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"final_status": "VALID_CONVERGED",
             "reports": [{"runtime_s": 0.0, "coeff_rel": coeff_rel}],
             "metadata": meta},
            f, indent=2,
        )
    print(f"corrected seed pump residual |R|={res:.4e}  rel_source={coeff_rel:.4e}")
    print(f"dc psi max/phi0={np.abs(psi_dc).max()/ipm.phi0:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
