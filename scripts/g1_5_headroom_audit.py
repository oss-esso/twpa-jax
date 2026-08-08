"""Milestone G1.5 (fold_plan.md, 2026-08-08): junction headroom audit at
the blocking boundary.

Reuses G1's already-converged solutions (no new continuation) to answer a
sharper question than "source current / Ic": in a distributed JTWPA, a
collective fold can occur while the SOURCE current is well below Ic if a
few junctions are locally driven much harder than the line average. This
computes, from the stored pump solution at each column's highest converged
point, the actual per-junction/per-time-sample phase excursion
``phi_j(t) = psi_j(t)/phi0`` and Josephson-supercurrent utilization
``|sin(phi_j(t))|`` -- the same AFT reconstruction
(``bifurcation.py::_psi_total_time``/``_ic_phi0``) Milestone F.5 already
uses for the exact quadratic fold coefficient, just evaluated once per
point instead of differentiated.

``min(cos(phi_j(t)))`` is reported separately: the tangent Josephson
stiffness is ``gamma_j(t) = (Ic_j/phi0) cos(phi_j(t))``, which collapses
toward zero as ``|phi_j| -> pi/2`` even while the supercurrent is only
approaching (not exceeding) ``Ic`` -- a local near-quarter-cycle phase
excursion is a plausible fold mechanism the source-current ratio alone
cannot see.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from scripts.g1_column_recovery import _build_args  # noqa: E402
from twpa_solver.pump import basis as pump_basis  # noqa: E402
from twpa_solver.pump.bifurcation import _ic_phi0, _psi_total_time  # noqa: E402

OUT_DIR = ROOT / "outputs" / "fold_plan_milestone_g1_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = {
    "7.0": ("D:/tmp/g1_70/g1_column_recovery.csv", 7.0),
    "7.7": ("D:/tmp/g1_77/g1_column_recovery.csv", 7.7),
    "7.9": ("D:/tmp/g1_79/g1_column_recovery.csv", 7.9),
}


def _load_rows(csv_path: str) -> list[dict[str, Any]]:
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


def audit_point(freq_ghz: float, pump_dir: str, target_current_a: float) -> dict[str, Any]:
    """Reconstruct phi_j(t)/sin(phi_j(t))/cos(phi_j(t)) for one converged point."""
    # power_min/max_dbm are unused here (only engine._build_problem, called
    # directly below with the real target CURRENT, matters) -- placeholder
    # dBm values just to satisfy argparse/build_points' expected units.
    args = _build_args(ROOT / "designs" / "ipm_2c_fixed", Path("D:/tmp/g1_5_unused"), freq_ghz, 1, -20.0, -20.0)
    engine = run_gain_map.InProcessEngine(args)
    full_problem, _basis, _omega = engine._build_problem(freq_ghz, target_current_a)

    X_full, _basis2 = pump_basis.load_pump_basis_from_solution(pump_dir)
    psi_total_t = _psi_total_time(full_problem, X_full)  # (n_t, n_branches)
    Ic, phi0 = _ic_phi0(full_problem)
    ic_ref = float(np.median(Ic))

    phi_jt = psi_total_t / phi0
    sin_phi = np.sin(phi_jt)
    cos_phi = np.cos(phi_jt)
    abs_sin = np.abs(sin_phi)
    abs_phi = np.abs(phi_jt)

    max_sin_idx = np.unravel_index(np.argmax(abs_sin), abs_sin.shape)
    max_phi_idx = np.unravel_index(np.argmax(abs_phi), abs_phi.shape)
    min_cos_idx = np.unravel_index(np.argmin(cos_phi), cos_phi.shape)

    per_junction_max_sin = np.max(abs_sin, axis=0)  # (n_branches,)
    per_junction_max_phi = np.max(abs_phi, axis=0)

    return {
        "freq_ghz": freq_ghz,
        "i_source_a": target_current_a,
        "ic_ref_a": ic_ref,
        "i_source_over_ic": target_current_a / ic_ref,
        "max_abs_sin_phi": float(abs_sin[max_sin_idx]),
        "max_abs_sin_phi_junction": int(max_sin_idx[1]),
        "max_abs_sin_phi_time_idx": int(max_sin_idx[0]),
        "max_abs_phi_rad": float(abs_phi[max_phi_idx]),
        "max_abs_phi_junction": int(max_phi_idx[1]),
        "p99_per_junction_max_abs_sin_phi": float(np.percentile(per_junction_max_sin, 99)),
        "p99_per_junction_max_abs_phi_rad": float(np.percentile(per_junction_max_phi, 99)),
        "min_cos_phi": float(cos_phi[min_cos_idx]),
        "min_cos_phi_junction": int(min_cos_idx[1]),
        "min_cos_phi_time_idx": int(min_cos_idx[0]),
        "n_junctions": int(psi_total_t.shape[1]),
        "n_time_samples": int(psi_total_t.shape[0]),
    }


def main() -> int:
    summary_rows = []
    for label, (csv_path, freq_ghz) in COLUMNS.items():
        rows = _load_rows(csv_path)
        passed = [r for r in rows if r["status"] == "PASS"]
        top = max(passed, key=lambda r: float(r["pump_power_dbm"]))
        result = audit_point(freq_ghz, top["pump_dir"], float(top["pump_current_peak_a"]))
        result["label"] = label
        result["pump_power_dbm"] = float(top["pump_power_dbm"])
        summary_rows.append(result)
        print(f"\n{'=' * 80}")
        print(f"{label} GHz, highest converged point below wall: "
              f"P={result['pump_power_dbm']:.3f}dBm I={result['i_source_a']:.4e}A")
        print(f"  I_source/Ic_ref            = {result['i_source_over_ic']:.4f}")
        print(f"  max|sin(phi)|              = {result['max_abs_sin_phi']:.4f} "
              f"(junction {result['max_abs_sin_phi_junction']}, t-sample {result['max_abs_sin_phi_time_idx']})")
        print(f"  max|phi| (rad)             = {result['max_abs_phi_rad']:.4f} "
              f"(junction {result['max_abs_phi_junction']})")
        print(f"  p99 per-junction max|sin phi| = {result['p99_per_junction_max_abs_sin_phi']:.4f}")
        print(f"  p99 per-junction max|phi|     = {result['p99_per_junction_max_abs_phi_rad']:.4f}")
        print(f"  min cos(phi)               = {result['min_cos_phi']:.4f} "
              f"(junction {result['min_cos_phi_junction']}, t-sample {result['min_cos_phi_time_idx']})")
        print(f"  n_junctions={result['n_junctions']} n_time_samples={result['n_time_samples']}")

    # 7.9 GHz: sweep over all converged points, plot vs power.
    rows79 = _load_rows(COLUMNS["7.9"][0])
    passed79 = sorted(
        (r for r in rows79 if r["status"] == "PASS"), key=lambda r: float(r["pump_power_dbm"]),
    )
    sweep = []
    for r in passed79:
        res = audit_point(7.9, r["pump_dir"], float(r["pump_current_peak_a"]))
        res["pump_power_dbm"] = float(r["pump_power_dbm"])
        sweep.append(res)
        print(f"[79 sweep] P={res['pump_power_dbm']:+.3f}dBm "
              f"I/Ic={res['i_source_over_ic']:.4f} "
              f"max|sin phi|={res['max_abs_sin_phi']:.4f} max|phi|={res['max_abs_phi_rad']:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    powers = [s["pump_power_dbm"] for s in sweep]
    axes[0].plot(powers, [s["i_source_over_ic"] for s in sweep], "o-")
    axes[0].set_ylabel("I_source / Ic_ref")
    axes[0].set_title("7.9 GHz: junction headroom vs target power (all 13 converged points)")
    axes[1].plot(powers, [s["max_abs_sin_phi"] for s in sweep], "o-", color="tab:orange")
    axes[1].set_ylabel("max|sin(phi_j)|")
    axes[1].axhline(1.0, color="k", linewidth=0.5)
    axes[2].plot(powers, [s["max_abs_phi_rad"] for s in sweep], "o-", color="tab:red")
    axes[2].axhline(np.pi / 2, color="k", linewidth=0.5, label="pi/2")
    axes[2].set_ylabel("max|phi_j| (rad)")
    axes[2].set_xlabel("pump power (dBm)")
    axes[2].legend()
    fig.tight_layout()
    out_path = OUT_DIR / "headroom_vs_power_79ghz.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out_path}")
    print("DONE_G1_5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
