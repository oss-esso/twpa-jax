"""Zoom into where the cliff actually starts (right after P7), and check
whether any of the 5 available intra-cell continuation methods changes the
picture.

The first power_sweep_col3_p7_p8 pass showed the wall is NOT smoothly spread
across the 0.215 dB gap between P7 (-28.468354430379748 dBm, converges to
coeff_rel=1.13e-13) and P8 (-28.253164556962027 dBm, stalls at ~6e-3): the
very first sweep point above P7 (+0.045 dB, "old i=1", P=-28.423191123613066
dBm) already lands at coeff_rel=1.4e-4, an 11-order cliff. Everything past
that point only creeps up another ~2 orders across the remaining 83% of the
gap. This script:

1. Zooms into [P7, old_i1] with 10 points spaced GEOMETRICALLY in distance
   from P7 (d_i = gap * 2**(i-9), i=0..9 -- i=0 is ~0.00009 dB above P7,
   i=9 is exactly the old i=1 point at +0.045 dB) to see whether the cliff is
   itself a further step-function at some even smaller offset, or whether it
   is already present at arbitrarily small offsets above P7.
2. At P7 itself plus all 10 zoomed points, reruns the COLD solve (mode=
   "seed", X=0 each time -- per the prior finding that the warm_chain series
   past its first failure is measuring a frozen stale state, not real
   solves) under each of 5 --inproc-continuation methods: adaptive_secant
   (the production default -- "whatever we're doing now"), adaptive_tangent,
   affine, ptc, arclength. Toggled by mutating engine.args.inproc_continuation
   between calls (InProcessEngine._settings() reads it fresh every
   solve_point call, so no need to rebuild the engine per method).

Output (outputs/power_sweep_fine_continuation_methods/):
  * fine_methods_summary.csv -- one row per (method, point).
  * fine_methods_coeff_rel.png -- coeff_rel (log-log) vs distance-from-P7
    (dB), one line per method, to see whether any method pushes the cliff
    out or lowers the floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gain_map as rgm  # noqa: E402

OUT_DIR = ROOT / "outputs" / "power_sweep_fine_continuation_methods"
FREQ_GHZ = 7.32894736842
P7_DBM = -28.468354430379748
P1_OLD_DBM = -28.423191123613066  # first cliffed point from power_sweep_col3_p7_p8
GAP_DB = P1_OLD_DBM - P7_DBM
N_STEPS = 10

METHODS = ["adaptive_secant", "adaptive_tangent", "affine", "ptc", "arclength"]

ARGV = [
    "--executor", "inprocess", "--mode", "warmstart",
    "--circuit-dir", "outputs/ipm_python_design",
    "--outdir", str(OUT_DIR),
    "--n-power", "1", "--n-frequency", "1",
    "--pump-power-min-dbm", str(P7_DBM),
    "--pump-power-max-dbm", str(P7_DBM),
    "--pump-freq-min-ghz", str(FREQ_GHZ), "--pump-freq-max-ghz", str(FREQ_GHZ),
    "--inproc-pump-backend", "schur_cpu_mt",
    "--inproc-preconditioner", "real_coupled_fast",
    "--inproc-fold-predictor", "secant",
    "--fold-skip-patience", "2",
    "--column-power-substep",
    "--column-power-substep-min-db", "0.005",
    "--inproc-schur-cache-size", "2",
    "--inproc-max-newton", "16",
    "--inproc-solve-deadline-s", "14",
    "--pump-mode-count", "10", "--nt", "40",
    "--signal-detuning-mhz", "200", "--no-signal-spectrum",
    "--signal-backend", "direct", "--signal-solver", "superlu",
    "--sidebands", "10", "--signal-workers", "6",
]


def build_points(args) -> list[rgm.GridPoint]:
    """P7 itself (distance 0, baseline) plus 10 points geometrically spaced
    in distance-from-P7 up to the old i=1 point."""
    distances = [0.0] + [GAP_DB * 2.0 ** (i - (N_STEPS - 1)) for i in range(N_STEPS)]
    points = []
    for idx, d in enumerate(distances):
        p = P7_DBM + d
        current_a = rgm.dbm_to_peak_current_a(
            p, attenuation_db=rgm.attenuation_db_for(FREQ_GHZ, args), z0_ohm=args.z0_ohm,
        )
        points.append(rgm.GridPoint(
            index=300 + idx, i_power=idx, j_freq=0,
            power_dbm=float(p), pump_freq_ghz=FREQ_GHZ, current_a=current_a,
        ))
    return points


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = rgm.parse_args(ARGV)
    engine = rgm.InProcessEngine(args)
    points = build_points(args)

    rows: list[dict] = []
    for method in METHODS:
        print(f"=== method={method} ===")
        engine.args.inproc_continuation = method
        for pt in points:
            d_db = pt.power_dbm - P7_DBM
            row, _ = engine.solve_point(
                pt, OUT_DIR / method, mode="seed", warm_X=None, force_gain=True,
            )
            row["method"] = method
            row["distance_from_p7_db"] = d_db
            rows.append(row)
            print(
                f"  d={d_db:.6f}dB P={pt.power_dbm:.5f}dBm "
                f"pump_status={row['pump_status']} coeff_rel={row['pump_coeff_rel']:.3e} "
                f"newton={row['pump_newton_total']} steps={row['pump_continuation_steps']} "
                f"reason={row['pump_failure_reason']}"
            )

    write_summary(rows)
    plot_coeff_rel(rows)


def write_summary(rows: list[dict]) -> None:
    import csv

    cols = [
        "method", "distance_from_p7_db", "pump_power_dbm", "pump_status",
        "pump_coeff_rel", "pump_newton_total", "pump_gmres_total",
        "pump_continuation_steps", "pump_failure_reason", "gain_status", "gain_db",
    ]
    out = OUT_DIR / "fine_methods_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in cols})
    print(f"wrote {out} ({len(rows)} rows)")


def plot_coeff_rel(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in METHODS:
        sub = sorted(
            (r for r in rows if r["method"] == method),
            key=lambda r: r["distance_from_p7_db"],
        )
        xs = [max(r["distance_from_p7_db"], 1e-6) for r in sub]  # avoid log(0) at P7 itself
        ys = [r["pump_coeff_rel"] for r in sub]
        markers = ["o" if r["pump_status"] == "VALID_CONVERGED" else "x" for r in sub]
        ax.loglog(xs, ys, marker=".", linestyle="-", label=method, alpha=0.8)
        for x, y, m in zip(xs, ys, markers):
            ax.scatter([x], [y], marker=m, color="k", s=20, zorder=3)
    ax.set_xlabel("distance above P7 (dB, log scale; P7 itself plotted at 1e-6 dB)")
    ax.set_ylabel("coeff_rel (final)")
    ax.set_title(
        "fp=7.329 GHz: coeff_rel vs distance above P7, by continuation method\n"
        "(o=converged, x=failed; all cold/mode=seed)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "fine_methods_coeff_rel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
