"""Profile what makes high-power pump solves slow.

Warm-starts a power sweep at a single pump frequency (the same in-process path
the map uses) and records, per power, the cost breakdown that drives runtime as
the operating point climbs toward the harmonic-balance fold:

    Newton iterations, GMRES inner iterations, GMRES per Newton, preconditioner
    factor time, pump-solve time, and gain-solve time.

The point is diagnostic: see *where* the time goes (more Newton steps? GMRES
blowing up per step? gain pipeline?) before changing any algorithm.

Example:
    python experiments/profile_exp10_highpower_solve.py \
        --pump-freq-ghz 7.0 --pump-power-min-dbm -34 --pump-power-max-dbm -22 \
        --n-power 25
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exp10_full_ipm_pump_map_warmstart as exp10

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = exp10.parse_args()
    args.executor = "inproc"
    freq = float(args.pump_freq_min_ghz)  # profile a single column; min == the chosen freq
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    powers = np.linspace(args.pump_power_min_dbm, args.pump_power_max_dbm, args.n_power)
    engine = exp10.InProcessEngine(args)
    print(f"profiling pump freq {freq:g} GHz, {len(powers)} powers "
          f"[{powers[0]:g} .. {powers[-1]:g}] dBm, precond={args.inproc_preconditioner}",
          flush=True)

    pass_dir = outdir / "profile_points"
    rows: list[dict] = []
    warm_X: np.ndarray | None = None
    for i, power_dbm in enumerate(powers):
        current = exp10.dbm_to_peak_current_a(
            float(power_dbm), attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm
        )
        point = exp10.GridPoint(i, i, 0, float(power_dbm), freq, current)
        mode = "cold" if warm_X is None else "warm"
        row, X = engine.solve_point(point, pass_dir, mode=mode, warm_X=warm_X)
        if X is not None:
            warm_X = X
        rows.append(row)
        nt = row["pump_newton_total"] or 0
        gm = row.get("pump_gmres_total") or 0
        gpn = gm / nt if nt else 0.0
        print(
            f"  {power_dbm:7.2f} dBm  {row['pump_status']:>16}  "
            f"newton={nt:3d}  gmres={gm:5d}  g/n={gpn:5.1f}  "
            f"factor={row['pump_factor_runtime_s']:6.3f}s  "
            f"pump={row['pump_runtime_s']:6.3f}s  "
            f"gain={(row['gain_total_runtime_s'] or 0.0):6.3f}s  "
            f"total={row['elapsed_s']:6.3f}s  "
            f"gain_db={('%.2f' % row['gain_db']) if row['gain_db'] is not None else '  --'}",
            flush=True,
        )

    csv_path = outdir / "profile.csv"
    fields = ["pump_power_dbm", "pump_status", "pump_newton_total", "pump_gmres_total",
              "pump_factor_runtime_s", "pump_runtime_s", "gain_total_runtime_s",
              "elapsed_s", "pump_coeff_rel", "pump_branch_current_max", "gain_db"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}", flush=True)

    p = np.array([r["pump_power_dbm"] for r in rows])
    newton = np.array([r["pump_newton_total"] or 0 for r in rows], float)
    gmres = np.array([r.get("pump_gmres_total") or 0 for r in rows], float)
    factor = np.array([r["pump_factor_runtime_s"] or 0.0 for r in rows], float)
    pump_t = np.array([r["pump_runtime_s"] or 0.0 for r in rows], float)
    gain_t = np.array([r["gain_total_runtime_s"] or 0.0 for r in rows], float)
    total = np.array([r["elapsed_s"] or 0.0 for r in rows], float)
    ok = np.array([r["pump_status"] == "VALID_CONVERGED" for r in rows])

    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax[0].plot(p, total, "o-", label="total / point", color="C3")
    ax[0].plot(p, pump_t, "s-", label="pump solve", color="C0")
    ax[0].plot(p, gain_t, "^-", label="gain solve", color="C1")
    ax[0].plot(p, factor, "d-", label="precond factor", color="C2")
    ax[0].axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax[0].set_ylabel("seconds")
    ax[0].set_title(f"per-point cost vs pump power @ {freq:g} GHz")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(p, newton, "o-", color="C0", label="Newton iters")
    ax[1].plot(p, gmres, "s-", color="C4", label="GMRES iters (total)")
    ax[1].set_ylabel("iterations")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)

    gpn = np.divide(gmres, newton, out=np.zeros_like(gmres), where=newton > 0)
    ax[2].plot(p, gpn, "o-", color="C4")
    ax[2].set_ylabel("GMRES per Newton")
    ax[2].set_xlabel("pump power (dBm)")
    ax[2].grid(True, alpha=0.3)
    for a in ax:
        for xp in p[~ok]:
            a.axvspan(xp - 0.1, xp + 0.1, color="red", alpha=0.08)

    fig.tight_layout()
    png = outdir / "profile.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
