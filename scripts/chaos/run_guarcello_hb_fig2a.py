"""Run the production HB solver on a Guarcello 990-cell RF-SQUID ladder.

The generated circuit is the nodal RF-SQUID topology represented by the
Guarcello FDTD model: each cell has a Josephson branch in parallel with Lg,
the branch has Cj and Rj, and each cell has Cg to ground. The two end nodes
carry the stated source/load networks and boundary capacitors.

All generated files and plots are written below outputs/chaos/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
PHI0_REDUCED = 2.067833848e-15 / (2.0 * math.pi)


def _stamp(matrix: sp.lil_matrix, i: int, j: int, value: float) -> None:
    matrix[i, i] += value
    matrix[j, j] += value
    matrix[i, j] -= value
    matrix[j, i] -= value


def build_guarcello_circuit(outdir: Path) -> dict[str, object]:
    """Write the 990-cell Guarcello RF-SQUID ladder in solver matrix format."""
    n_cells = 990
    n_nodes = n_cells + 1
    ic_a = 2.0e-6
    cj_f = 200.0e-15
    cg_f = 24.0e-15
    lg_h = 120.0e-12
    rj_ohm = 20.0e3
    ri_ohm = 50.0
    rl_ohm = 50.0
    ci_f = 24.0e-15
    cl_f = 1.0e-9

    C = sp.lil_matrix((n_nodes, n_nodes), dtype=float)
    G = sp.lil_matrix((n_nodes, n_nodes), dtype=float)
    K = sp.lil_matrix((n_nodes, n_nodes), dtype=float)
    Bphi = sp.lil_matrix((n_nodes, n_cells), dtype=float)
    C[0, 0] += ci_f
    C[-1, -1] += cl_f
    G[0, 0] += 1.0 / ri_ohm
    G[-1, -1] += 1.0 / rl_ohm
    for branch in range(n_cells):
        left, right = branch, branch + 1
        Bphi[left, branch] = -1.0
        Bphi[right, branch] = 1.0
        _stamp(C, left, right, cj_f)
        _stamp(G, left, right, 1.0 / rj_ohm)
        _stamp(K, left, right, 1.0 / lg_h)
        C[right, right] += cg_f

    outdir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(outdir / "C.npz", C.tocsr())
    sp.save_npz(outdir / "G.npz", G.tocsr())
    sp.save_npz(outdir / "K.npz", K.tocsr())
    sp.save_npz(outdir / "Bphi.npz", Bphi.tocsr())
    np.savez(
        outdir / "ipm_arrays.npz",
        Ic=np.full(n_cells, ic_a),
        Lj=np.full(n_cells, lg_h),
        phi0_reduced=np.array([PHI0_REDUCED]),
        nodes=np.arange(n_nodes, dtype=np.int64),
        port_numbers=np.array([1, 2], dtype=np.int64),
        port_indices=np.array([0, n_cells], dtype=np.int64),
    )
    metadata = {
        "case": "guarcello_rf_squid_hb",
        "topology": "990-cell RF-SQUID ladder",
        "parameters": {
            "n_cells": n_cells, "Ic_a": ic_a, "Cj_f": cj_f,
            "Cg_f": cg_f, "Lg_h": lg_h, "Rj_ohm": rj_ohm,
            "Ri_ohm": ri_ohm, "Rl_ohm": rl_ohm,
            "Ci_f": ci_f, "Cl_f": cl_f,
        },
        "ports": {"1": 0, "2": n_cells},
        "source": "Guarcello FDTD Device defaults, translated to nodal matrices",
    }
    (outdir / "ipm_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def plot_gain(output: Path) -> Path:
    rows = list(csv.DictReader((output / "map_points.csv").open(encoding="utf-8")))
    rows = [row for row in rows if row.get("gain_vs_off_db") not in (None, "", "nan", "NaN")]
    if not rows:
        raise ValueError("map_points.csv contains no pump-off-normalized gain rows")
    rows.sort(key=lambda row: float(row["pump_power_dbm"]))
    x = np.array([float(row["pump_power_dbm"]) for row in rows])
    y = np.array([float(row["gain_vs_off_db"]) for row in rows])
    fig, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.plot(x, y, "o-", label="production HB, pump-off normalized")
    axis.axvspan(-54.0, -53.5, color="tab:red", alpha=0.12, label="FDTD transition band")
    axis.set(xlabel="Pump power (dBm)", ylabel="Gain vs pump-off (dB)",
             title="Guarcello Fig. 2(a) control: harmonic balance")
    axis.grid(alpha=0.25)
    axis.legend()
    path = output / "guarcello_hb_fig2a_gain.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chaos/guarcello_hb_fig2a")
    parser.add_argument("--n-power", type=int, default=51)
    parser.add_argument("--power-min-dbm", type=float, default=-70.0)
    parser.add_argument("--power-max-dbm", type=float, default=-45.0)
    parser.add_argument("--pump-ghz", type=float, default=7.0)
    parser.add_argument("--signal-ghz", type=float, default=6.42)
    parser.add_argument("--signal-dbm", type=float, default=-100.0)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    circuit_dir = output / "circuit"
    if not args.plot_only:
        build_guarcello_circuit(circuit_dir)
    if args.build_only:
        print(f"wrote {circuit_dir}")
        return 0
    if not args.plot_only:
        command = [
            sys.executable, str(ROOT / "scripts/run_gain_map.py"),
            "--mode", "warmstart", "--executor", "inprocess",
            "--circuit-dir", str(circuit_dir), "--outdir", str(output),
            "--n-power", str(args.n_power), "--n-frequency", "1",
            "--pump-power-min-dbm", str(args.power_min_dbm),
            "--pump-power-max-dbm", str(args.power_max_dbm),
            "--pump-freq-min-ghz", str(args.pump_ghz),
            "--pump-freq-max-ghz", str(args.pump_ghz),
            "--signal-ghz", str(args.signal_ghz),
            "--signal-attenuation-db", "0", "--attenuation-db", "0",
            "--no-signal-spectrum",
            "--power-convention", "legacy_traveling_wave",
            "--pump-port", "1", "--source-port", "1", "--out-port", "2",
            "--pump-mode-policy", "positive_odd_jc", "--pump-mode-count", "19",
            "--harmonics", "19", "--nt", "80", "--sidebands", "10",
            "--continuation-steps", "20", #"--overwrite",
        ]
        log = output / "hb_run.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"HB sweep failed with return code {completed.returncode}; see {log}")
    plot = plot_gain(output)
    print(f"wrote {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
