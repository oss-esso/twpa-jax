# experiments/exp14_fxjtwpa_seed_validate.py
"""Rigorously test whether the JC-seed pump residual (~45) is roundoff or real,
and report the physical pump phase swing.

Roundoff test: per row i of mode 1, compare |R_i| against the sum of absolute
contributions sum_j |D_ij| |X_j|. If |R_i| / that_sum ~ 1e-15 the residual is
catastrophic cancellation of the stiff mutual block, i.e. the seed satisfies the
equation to machine precision and is a valid pump state.
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


def main() -> None:
    ipm = exp08.load_ipm(DESIGN)
    seed = np.load(SEED)
    X = (seed["X_real"] + 1j * seed["X_imag"]).astype(np.complex128)
    modes = [int(m) for m in seed["pump_modes"]]
    dc = np.load(DC)
    psi_dc = np.asarray(dc["psi_dc"], dtype=np.float64).reshape(-1)

    grid = exp08.HarmonicGrid(modes=np.asarray(modes), nt=NT, omega=2 * math.pi * PUMP_FREQ_HZ)

    # ---- roundoff test on mode 1 ----
    Kc = ipm.K.astype(np.complex128)
    Cc = ipm.C.astype(np.complex128)
    w1 = 1.0 * grid.omega
    D1 = (Kc + (-w1 * w1) * Cc).tocsr()
    x1 = X[modes.index(1)]
    r1 = D1 @ x1
    # row-wise absolute contribution sum  |D|.|x|
    absD = D1.copy()
    absD.data = np.abs(absD.data)
    denom = absD @ np.abs(x1)
    rel = np.abs(r1) / np.where(denom > 0, denom, 1.0)
    bad = np.argsort(-np.abs(r1))[:8]
    print("mode-1 roundoff test (idx |R| |D|.|x| rel):")
    for i in bad:
        print(f"  {i:5d}  {abs(r1[i]):.4e}  {denom[i]:.4e}  rel={rel[i]:.3e}")
    print(f"max |R_1|={np.abs(r1).max():.3e}  median rel on top-200={np.median(np.sort(rel)[-200:]):.3e}")

    # ---- pump + dc phase swing across the JJ branches ----
    x_t = grid.synthesize(X)               # (nt, nodes)
    psi_pump_t = (ipm.Bphi.T @ x_t.T).T    # (nt, branches)
    psi_total = psi_pump_t + psi_dc[None, :]
    swing_pump = np.abs(psi_pump_t).max() / ipm.phi0
    swing_total = np.abs(psi_total).max() / ipm.phi0
    print(f"\npump phase swing max |psi_pump|/phi0 = {swing_pump:.4f} rad")
    print(f"total (pump+dc)   max |psi_total|/phi0 = {swing_total:.4f} rad")
    print(f"dc branch flux    max |psi_dc|/phi0    = {np.abs(psi_dc).max()/ipm.phi0:.4f} rad")
    # gamma = cos(psi/phi0): does it go strongly negative (near oscillation)?
    g = np.cos(psi_total / ipm.phi0)
    print(f"cos(psi_total/phi0): min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f}")


if __name__ == "__main__":
    main()
