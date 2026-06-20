# experiments/exp14_build_jc_warmstart.py
"""Convert a JC nonlinear pump nodeflux dump into a Python warm-start seed.

Decoded convention (validated on the matched jtwpa case): JC `nodeflux` equals
the Python pump node flux divided by phi0_reduced, with identical mode order,
identical node order, and identical phase (exp(+i k omega t), 2 Re). Hence

    Python_X[mode k] = JC_nodeflux[mode k] * phi0_reduced

For a DC-biased design the JC mode (0,) row is the DC operating-point node flux;
it becomes the Python dc_branch_flux = Bphi^T @ (nodeflux_0 * phi0). The nonzero
modes become the AC pump seed in a dense [1..K] basis.

Writes:
    <out>/pump/pump_solution.npz   (+ pump_report.json)  -> for --promote-from-pump-dir
    <out>/dc/dc_solution.npz       (psi_dc)              -> for --dc-solution
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ipm-dir", required=True)
    p.add_argument("--nf-dir", required=True, help="dir with jc_nodeflux_{real,imag,modes}.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--pump-freq-ghz", type=float, required=True)
    args = p.parse_args()

    d = Path(args.ipm_dir)
    arr = np.load(d / "ipm_arrays.npz")
    phi0 = float(arr["phi0_reduced"][0])
    Bphi = sp.load_npz(d / "Bphi.npz").tocsr()
    n_nodes = Bphi.shape[0]

    nf_dir = Path(args.nf_dir)
    nfr = np.loadtxt(nf_dir / "jc_nodeflux_real.csv", delimiter=",")
    nfi = np.loadtxt(nf_dir / "jc_nodeflux_imag.csv", delimiter=",")
    NF = np.atleast_2d(nfr) + 1j * np.atleast_2d(nfi)  # (Nmodes, Nnodes)
    modes_raw = np.loadtxt(nf_dir / "jc_nodeflux_modes.csv", delimiter=",")
    modes = [int(m) for m in np.atleast_1d(modes_raw).reshape(-1)]  # single-tone -> ints

    if NF.shape[1] != n_nodes:
        raise ValueError(f"nodeflux has {NF.shape[1]} nodes but design has {n_nodes}")

    # Python node flux = JC nodeflux * phi0 (validated on jtwpa).
    Xnode = NF * phi0  # (Nmodes, Nnodes)

    # DC mode (0) -> dc branch flux; nonzero modes -> dense AC pump seed.
    ac_modes = [m for m in modes if m != 0]
    ac_modes_sorted = sorted(ac_modes)
    X = np.zeros((len(ac_modes_sorted), n_nodes), dtype=np.complex128)
    for j, m in enumerate(ac_modes_sorted):
        X[j] = Xnode[modes.index(m)]

    out = Path(args.out)
    (out / "pump").mkdir(parents=True, exist_ok=True)
    (out / "dc").mkdir(parents=True, exist_ok=True)

    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9
    pump_modes = np.asarray(ac_modes_sorted, dtype=np.int64)
    np.savez(
        out / "pump" / "pump_solution.npz",
        X_real=X.real, X_imag=X.imag,
        harmonics=pump_modes, pump_modes=pump_modes,
    )
    with open(out / "pump" / "pump_report.json", "w", encoding="utf-8") as f:
        json.dump({"final_status": "JC_SEED", "metadata": {
            "omega_p": omega_p, "pump_mode_policy": "dense_real",
            "pump_modes": ac_modes_sorted, "pump_source_mode": 1,
            "pump_basis": "positive_phasor", "source": "jc_nodeflux"}}, f, indent=2)

    if 0 in modes:
        dc_nodeflux = Xnode[modes.index(0)].real  # DC is real
        psi_dc = np.asarray(Bphi.T @ dc_nodeflux, dtype=np.float64).reshape(-1)
        np.savez(out / "dc" / "dc_solution.npz", psi_dc=psi_dc,
                 x_dc=dc_nodeflux.astype(np.float64))
        print(f"dc_branch_flux_max_over_phi0={np.max(np.abs(psi_dc/phi0)):.4f}")

    print(f"phi0={phi0:.6e} ac_modes={ac_modes_sorted}")
    print(f"X_seed max_abs={np.abs(X).max():.6e} (node-flux units)")
    print(f"wrote {out}/pump and {out}/dc")


if __name__ == "__main__":
    main()
