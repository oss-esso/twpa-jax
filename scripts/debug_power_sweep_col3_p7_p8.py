"""Fine power-axis sweep between the last-converged cell (point 7) and the
first stalled cell (point 8) of the fp=7.329 GHz column
(outputs/measurement_match_debug_02_col3_trim_fixed).

Point 7 (P=-28.468354430379748 dBm) converges to coeff_rel=1.13e-13. Point 8
(P=-28.253164556962027 dBm), only 0.215 dB higher, cannot get below ~6e-3 no
matter the solve path -- a ~1e10 gap over a tiny power step (see
docs/convergence_investigation_log.md). This script bisects that 0.215 dB
gap into 10 points to see whether the coeff_rel floor breaks smoothly or
cliffs abruptly at some intermediate power.

Spacing: t = 1-(1-x)**2 for x linear 0..1, which is denser near x=1 (P8) --
finer resolution at the end already known to fail localizes the wall more
precisely than uniform spacing would.

Two passes, both with force_gain=True (run_gain_map.py InProcessEngine.
solve_point returns the last-iterate X even when Newton didn't converge only
in force_gain mode -- see CLAUDE.md "Forced-gain column resume" -- so
warm-chaining survives an intermediate failure instead of the chain going
None):
  * cold       -- mode="seed": each point solved independently from X=0.
  * warm_chain -- mode="warm" after the first point: each point warm-started
                  from the PREVIOUS point in this sweep (not from point 7's
                  disk state), whether or not that previous point converged --
                  mirrors how the real map warm-starts up a column.

Output (outputs/power_sweep_col3_p7_p8/):
  * power_sweep_summary.csv -- one row per (run, point).
  * power_sweep_coeff_rel.png -- coeff_rel (log-y) vs power_dbm, cold vs
    warm_chain, P7/P8 marked, converged points green / stalled points red.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gain_map as rgm  # noqa: E402

OUT_DIR = ROOT / "outputs" / "power_sweep_col3_p7_p8"
FREQ_GHZ = 7.32894736842
P7_DBM = -28.468354430379748
P8_DBM = -28.253164556962027
N_STEPS = 10

# Mirrors the ARGV in debug_cell_gmres_matrix_analysis.py (same production
# solver settings); the power-range flags are unused here since points are
# built manually below.
ARGV = [
    "--executor", "inprocess", "--mode", "warmstart",
    "--circuit-dir", "outputs/ipm_python_design",
    "--outdir", str(OUT_DIR),
    "--n-power", "1", "--n-frequency", "1",
    "--pump-power-min-dbm", str(P8_DBM),
    "--pump-power-max-dbm", str(P8_DBM),
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


def build_sweep_points(args) -> list[rgm.GridPoint]:
    x = np.linspace(0.0, 1.0, N_STEPS)
    t = 1.0 - (1.0 - x) ** 2
    powers = P7_DBM + t * (P8_DBM - P7_DBM)
    points = []
    for i, p in enumerate(powers):
        current_a = rgm.dbm_to_peak_current_a(
            float(p),
            attenuation_db=rgm.attenuation_db_for(FREQ_GHZ, args),
            z0_ohm=args.z0_ohm,
        )
        points.append(rgm.GridPoint(
            index=200 + i, i_power=i, j_freq=0,
            power_dbm=float(p), pump_freq_ghz=FREQ_GHZ, current_a=current_a,
        ))
    return points


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = rgm.parse_args(ARGV)
    engine = rgm.InProcessEngine(args)
    points = build_sweep_points(args)

    rows: list[dict] = []

    print("=== COLD sweep: each point solved independently from X=0 ===")
    for pt in points:
        row, _ = engine.solve_point(
            pt, OUT_DIR / "cold_sweep", mode="seed", warm_X=None, force_gain=True,
        )
        row["sweep_run"] = "cold"
        rows.append(row)
        _print_row(pt, row)

    print("=== WARM-CHAIN sweep: each point warm-started from the previous ===")
    X_prev = None
    for i, pt in enumerate(points):
        mode = "seed" if i == 0 else "warm"
        row, X_prev = engine.solve_point(
            pt, OUT_DIR / "warm_chain_sweep", mode=mode, warm_X=X_prev, force_gain=True,
        )
        row["sweep_run"] = "warm_chain"
        rows.append(row)
        _print_row(pt, row)

    write_summary(rows)
    plot_coeff_rel(rows)


def _print_row(pt: rgm.GridPoint, row: dict) -> None:
    print(
        f"  i={pt.i_power} P={pt.power_dbm:.4f}dBm dP_from_p8={pt.power_dbm - P8_DBM:+.4f}dB "
        f"pump_status={row['pump_status']} coeff_rel={row['pump_coeff_rel']:.3e} "
        f"newton={row['pump_newton_total']} gmres={row['pump_gmres_total']} "
        f"gain_db={row['gain_db']}"
    )


def write_summary(rows: list[dict]) -> None:
    import csv

    cols = [
        "sweep_run", "i_power", "pump_power_dbm", "pump_status", "pump_coeff_rel",
        "pump_newton_total", "pump_gmres_total", "pump_continuation_steps",
        "pump_failure_reason", "gain_status", "gain_db",
    ]
    out = OUT_DIR / "power_sweep_summary.csv"
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

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, marker in [("cold", "o"), ("warm_chain", "s")]:
        sub = [r for r in rows if r["sweep_run"] == label]
        xs = [r["pump_power_dbm"] for r in sub]
        ys = [r["pump_coeff_rel"] for r in sub]
        colors = ["tab:green" if r["pump_status"] == "VALID_CONVERGED" else "tab:red" for r in sub]
        ax.semilogy(xs, ys, linestyle="-", color="gray", alpha=0.4, zorder=1)
        ax.scatter(xs, ys, c=colors, marker=marker, s=60, zorder=3, label=label)
    ax.axvline(P7_DBM, color="k", linestyle=":", alpha=0.4)
    ax.axvline(P8_DBM, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("pump power (dBm)  [dotted=P7 converged, dashed=P8 stalled]")
    ax.set_ylabel("coeff_rel (final)")
    ax.set_title(
        "fp=7.329 GHz: coeff_rel across the P7->P8 power sweep\n"
        "(green=converged, red=stalled)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "power_sweep_coeff_rel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
