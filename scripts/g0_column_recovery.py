"""Milestone G0 (fold_plan.md, 2026-08-08): single-column production
recovery prototype.

Pump solving only (no signal/gain), one frequency (default 7.9 GHz), the
real production power grid (default n_power=20, -26..-16 dBm, matching the
CLAUDE.md "Standard gain-map flag set"). Two passes over the column:

1. Baseline: the existing warm-start path -- ``InProcessEngine.solve_point``
   chained fail-fast (warm-start Newton from the last CONVERGED point; a
   point with no converged predecessor gets a cold solve). This is
   literally the production dispatch (``mode="warm"`` -> ``solve_direct`` ==
   ``solve_one(problem, x, 1.0)``), just without the map's skip-patience
   shortcut (disabled here so every point gets a real attempt -- coverage
   cannot be measured on points that were never tried).

2. Recovery: for every point the baseline failed, four tiers in increasing
   cost order, each seeded from the nearest preceding point this run has
   CONVERGED on (baseline or a previous recovery -- a recovered point
   becomes a real warm anchor for later targets, same as production would
   do in one online pass):

   - Tier 1 (previous-power warm-start Newton): NOT re-run as a separate
     solve. ``solve_direct`` (the baseline's own "warm" dispatch) already
     equals ``solve_one(problem, last_good_X, 1.0)`` bit-for-bit -- when a
     warm anchor exists, the baseline attempt for this same point already
     WAS this tier. Re-running it would be a deterministic, wasted repeat
     of the identical computation. Recorded as "already attempted by
     baseline" rather than paid for twice.
   - Tier 2: ``InProcessEngine.solve_power_substep`` -- adaptively
     subdivided fixed-frequency power continuation from the anchor.
   - Tier 3: ``InProcessEngine.solve_arclength_forward`` (new this
     milestone) -- local pseudo-arclength continuation starting at the
     anchor state, terminating as soon as the bordered corrector reaches
     the target (its own built-in behavior, no separate bracket step
     needed).
   - Tier 4: ``InProcessEngine.solve_frequency_substep`` (new this
     milestone) -- only attempted if a same-power solution at a nearby
     frequency can be established first (a short local search over a
     handful of small frequency offsets, warm-started from the same
     anchor); if found, a fixed-power frequency continuation walks it back
     to the target frequency.

   Each tier that produces a candidate state is verified by re-running the
   REAL ``solve_point`` (so gain is skipped exactly as production would
   skip it on a non-converged pump, files are written normally, and a
   tier's "success" always means the actual production convergence gate
   passed, not just that a continuation loop returned something).

Writes a per-point CSV with baseline/final status, recovery method,
iterations/GMRES/runtime, and a summary table.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from twpa_solver.pump import hb as exp08  # noqa: E402

TIER2_DEADLINE_S = 90.0
TIER3_DEADLINE_S = 60.0
TIER4_ANCHOR_OFFSETS_GHZ = (0.005, -0.005, 0.01, -0.01, 0.02, -0.02)
TIER4_ANCHOR_DEADLINE_S = 20.0  # per offset attempt
TIER4_SUBSTEP_DEADLINE_S = 60.0


def _build_args(
    circuit_dir: Path, outdir: Path, freq_ghz: float,
    n_power: int, power_min_dbm: float, power_max_dbm: float,
) -> argparse.Namespace:
    argv = [
        "--circuit-dir", str(circuit_dir.resolve()),
        "--outdir", str(outdir),
        "--executor", "inprocess", "--mode", "warmstart",
        "--inproc-pump-backend", "schur_cpu_mt",
        "--inproc-preconditioner", "real_coupled_fast",
        "--inproc-fold-predictor", "none",  # keep the baseline dispatch plain/reproducible
        "--inproc-fail-fast",
        "--fold-skip-patience", "0",  # every point must get a real attempt (coverage measurement)
        "--pump-current-jc-scale", "1.0",
        "--n-power", str(n_power), "--n-frequency", "1",
        "--pump-power-min-dbm", str(power_min_dbm),
        "--pump-power-max-dbm", str(power_max_dbm),
        "--pump-freq-min-ghz", str(freq_ghz),
        "--pump-freq-max-ghz", str(freq_ghz),
        "--signal-detuning-mhz", "500", "--no-signal-spectrum",
        "--log-level", "WARNING", "--overwrite",
    ]
    return run_gain_map.parse_args(argv)


def run_column(
    circuit_dir: Path, outdir: Path, freq_ghz: float,
    n_power: int, power_min_dbm: float, power_max_dbm: float,
) -> list[dict[str, Any]]:
    args = _build_args(circuit_dir, outdir, freq_ghz, n_power, power_min_dbm, power_max_dbm)
    points, _powers, _freqs = run_gain_map.build_points(args)
    points = sorted(points, key=lambda p: p.power_dbm)
    engine = run_gain_map.InProcessEngine(args)
    pass_dir = outdir / "pass"
    scale = args.pump_current_jc_scale

    last_good_X = None
    last_good_cur: float | None = None
    rows: list[dict[str, Any]] = []

    for point in points:
        target_cur = point.current_a * scale
        mode = "warm" if last_good_X is not None else "seed"
        t0 = time.perf_counter()
        row, X = engine.solve_point(point, pass_dir, mode=mode, warm_X=last_good_X)
        baseline_wall_s = time.perf_counter() - t0
        row["baseline_status"] = row["status"]
        row["baseline_wall_s"] = baseline_wall_s
        row["recovery_tier"] = 0
        row["recovery_method"] = "baseline_warm" if mode == "warm" else "baseline_cold"
        row["recovery_wall_s"] = 0.0
        row["tier1_note"] = (
            "n/a (no warm anchor)" if mode != "warm"
            else "identical to baseline warm dispatch, not re-run"
        )
        row["recovered"] = False
        used_X = X

        if row["status"] != "PASS" and last_good_X is not None:
            t_rec0 = time.perf_counter()
            recovered = False

            # Tier 2: adaptive fixed-frequency power continuation.
            t1 = time.perf_counter()
            guess2, info2 = engine.solve_power_substep(
                freq_ghz, last_good_X, last_good_cur, target_cur, deadline_s=TIER2_DEADLINE_S,
            )
            row["tier2_substeps"] = info2["substeps"]
            row["tier2_terminal_reason"] = info2["terminal_reason"]
            row["tier2_runtime_s"] = time.perf_counter() - t1
            if guess2 is not None:
                row2, X2 = engine.solve_point(point, pass_dir, mode="warm", warm_X=guess2)
                if row2["status"] == "PASS":
                    row = {**row2, **{k: v for k, v in row.items() if k not in row2}}
                    used_X = X2
                    row["recovery_method"] = "power_substep"
                    row["recovery_tier"] = 2
                    recovered = True

            # Tier 3: local pseudo-arclength continuation from the anchor.
            if not recovered:
                t2 = time.perf_counter()
                guess3, info3 = engine.solve_arclength_forward(
                    freq_ghz, last_good_X, last_good_cur, target_cur, deadline_s=TIER3_DEADLINE_S,
                )
                row["tier3_arclength_steps"] = info3.get("steps")
                row["tier3_fold_lambda"] = info3.get("fold_lambda")
                row["tier3_terminal_reason"] = info3.get("terminal_reason")
                row["tier3_runtime_s"] = time.perf_counter() - t2
                if guess3 is not None:
                    row3, X3 = engine.solve_point(point, pass_dir, mode="warm", warm_X=guess3)
                    if row3["status"] == "PASS":
                        row = {**row3, **{k: v for k, v in row.items() if k not in row3}}
                        used_X = X3
                        row["recovery_method"] = "arclength"
                        row["recovery_tier"] = 3
                        recovered = True

            # Tier 4: same-power nearby-frequency anchor, then frequency continuation.
            if not recovered:
                t3 = time.perf_counter()
                anchor = None
                solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
                for df in TIER4_ANCHOR_OFFSETS_GHZ:
                    nb_freq = freq_ghz + df
                    prob_nb, _b, _o = engine._build_problem(nb_freq, target_cur)
                    solve_nb = engine._make_solve_problem(prob_nb, nb_freq)
                    Xn, repn = solver.solve_one(solve_nb, last_good_X, 1.0)
                    if repn.converged:
                        anchor = (nb_freq, Xn)
                        break
                row["tier4_anchor_found"] = anchor is not None
                row["tier4_anchor_search_runtime_s"] = time.perf_counter() - t3
                if anchor is not None:
                    nb_freq, Xn = anchor
                    row["tier4_anchor_freq_ghz"] = nb_freq
                    guess4, info4 = engine.solve_frequency_substep(
                        nb_freq, freq_ghz, target_cur, Xn, deadline_s=TIER4_SUBSTEP_DEADLINE_S,
                    )
                    row["tier4_substeps"] = info4["substeps"]
                    row["tier4_terminal_reason"] = info4["terminal_reason"]
                    if guess4 is not None:
                        row4, X4 = engine.solve_point(point, pass_dir, mode="warm", warm_X=guess4)
                        if row4["status"] == "PASS":
                            row = {**row4, **{k: v for k, v in row.items() if k not in row4}}
                            used_X = X4
                            row["recovery_method"] = "frequency_continuation"
                            row["recovery_tier"] = 4
                            recovered = True

            row["recovered"] = recovered
            row["recovery_wall_s"] = time.perf_counter() - t_rec0

        if row["status"] == "PASS":
            last_good_X, last_good_cur = used_X, target_cur

        print(
            f"[G0] power={point.power_dbm:+.3f}dBm baseline={row['baseline_status']} "
            f"final={row['status']} method={row['recovery_method']} "
            f"recovery_wall_s={row['recovery_wall_s']:.2f}",
            flush=True,
        )
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys and k not in ("_spectrum",):
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    baseline_pass = sum(1 for r in rows if r["baseline_status"] == "PASS")
    final_pass = sum(1 for r in rows if r["status"] == "PASS")
    by_method: dict[str, int] = {}
    for r in rows:
        m = r["recovery_method"]
        by_method[m] = by_method.get(m, 0) + (1 if r["status"] == "PASS" else 0)
    total_baseline_wall = sum(r["baseline_wall_s"] for r in rows)
    total_recovery_wall = sum(r["recovery_wall_s"] for r in rows)

    print(f"\n{'=' * 80}")
    print(f"G0 column summary: {n} targets")
    print(f"baseline PASS: {baseline_pass}/{n} ({100 * baseline_pass / n:.1f}%)")
    print(f"final PASS:    {final_pass}/{n} ({100 * final_pass / n:.1f}%)")
    print(f"recovered by tier: {by_method}")
    print(f"total baseline wall time: {total_baseline_wall:.1f}s")
    print(f"total recovery wall time: {total_recovery_wall:.1f}s "
          f"({100 * total_recovery_wall / max(total_baseline_wall, 1e-9):.1f}% of baseline)")
    print("DONE_G0")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--freq-ghz", type=float, default=7.9)
    p.add_argument("--n-power", type=int, default=20)
    p.add_argument("--power-min-dbm", type=float, default=-26.0)
    p.add_argument("--power-max-dbm", type=float, default=-16.0)
    p.add_argument("--csv-out", type=Path, default=None)
    args = p.parse_args()

    rows = run_column(
        args.circuit_dir, args.outdir, args.freq_ghz,
        args.n_power, args.power_min_dbm, args.power_max_dbm,
    )
    csv_out = args.csv_out or (args.outdir / "g0_column_recovery.csv")
    write_csv(csv_out, rows)
    print(f"wrote {csv_out}")
    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
