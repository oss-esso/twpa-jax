"""Auto-tune the 2c design's Lj (and, once close, Cg) to close the pump-frequency
shift against the Themis measurement.

Loop, each iteration expensive (a full gain map -- default 80x20=1600 cells):

  1. Build a 2c IPM design at a trial Lj (twpa_solver.builders.ipm, coupler
     ALWAYS re-optimized via --coupler-mode-equivalent "optimize" -- the coupler
     geometry depends on Lj-driven impedance, so a cached/stale coupler would
     confound the fit).
  2. Run scripts/run_gain_map.py over the given (power, freq) range on that
     design, using the same production solver flags as run_campaign.ps1's
     $Common + c04_baseline_prod (column traversal, schur_cpu_mt backend,
     real_coupled_fast preconditioner, secant fold-predictor).
  3. Compare the resulting map to the Themis measurement
     (docs/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK) over the SAME
     (power, freq) window via scripts/align_map_to_measurement.py's fit
     (freq_shift_ghz, power_shift_db, rmse_db).
  4. Stage A (coarse): secant search on Lj driving freq_shift_ghz -> 0.
     Stage B (fine, only once |freq_shift_ghz| <= --freq-tol-ghz): bounded 1-D
     Cg line search (probe both directions, then expand geometrically in the
     winning one, clamped to --cg-min-ff/--cg-max-ff) driving rmse_db down
     further.

Comparison window: both the numeric fit (align_map_to_measurement.align_maps)
and its plots are restricted to the (power, freq) window actually simulated --
not the Themis measurement's full comb -- via fit_freq_range/fit_power_range
(scoring) and crop_for_plot (rendering).

Every evaluated (Lj, Cg) point is logged to tuning_log.csv as it happens (not
batched at the end), so a killed/interrupted run still has a usable trail.
Resumable: an (Lj, Cg) point whose design+map+fit already exist on disk is
reloaded, not recomputed.

Usage:
    python scripts/tune_lj_to_themis.py \\
        --n-power 80 --n-frequency 20 \\
        --pump-power-min-dbm -36 --pump-power-max-dbm -19 \\
        --pump-freq-min-ghz 7.25 --pump-freq-max-ghz 7.75
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import align_map_to_measurement as amm  # noqa: E402
from twpa_solver.builders import ipm  # noqa: E402

DEFAULT_MEASUREMENT_DIR = ROOT / "docs" / "14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"
STANDARD_LJ_H = 123.9e-12
STANDARD_CG_F = 66.0e-15


@dataclass
class Point:
    lj_h: float
    cg_f: float

    def tag(self) -> str:
        return f"lj{self.lj_h * 1e12:.4f}pH_cg{self.cg_f * 1e15:.4f}fF"


def build_design(point: Point, outdir: Path) -> dict[str, Any]:
    params = ipm.IPMParams(Lj=point.lj_h, Cg=point.cg_f)
    coupler = ipm.make_coupler_discrete(params, "optimize")
    circuit, ends = ipm.make_ipm(params, coupler)
    mats = ipm.build_matrices(circuit)
    summary = ipm.write_outputs(
        outdir=str(outdir), circuit=circuit, params=params, coupler=coupler,
        ends=ends, mats=mats,
    )
    return summary


def run_gain_map(circuit_dir: Path, map_dir: Path, args: argparse.Namespace) -> None:
    argv = [
        sys.executable, str(ROOT / "scripts" / "run_gain_map.py"),
        "--executor", "inprocess", "--mode", "warmstart",
        "--circuit-dir", str(circuit_dir),
        "--outdir", str(map_dir),
        "--n-power", str(args.n_power), "--n-frequency", str(args.n_frequency),
        "--pump-power-min-dbm", str(args.pump_power_min_dbm),
        "--pump-power-max-dbm", str(args.pump_power_max_dbm),
        "--pump-freq-min-ghz", str(args.pump_freq_min_ghz),
        "--pump-freq-max-ghz", str(args.pump_freq_max_ghz),
        "--inproc-pump-backend", "schur_cpu_mt",
        "--inproc-preconditioner", "real_coupled_fast",
        "--inproc-fold-predictor", "secant",
        "--traversal", "column",
        "--fold-skip-patience", "4",
        "--inproc-schur-cache-size", "2",
        "--inproc-max-newton", "16",
        "--inproc-solve-deadline-s", "14",
        "--pump-mode-count", "10", "--nt", "40",
        "--signal-detuning-mhz", "100",
        "--signal-backend", "direct", "--signal-solver", "superlu",
        "--sidebands", "10", "--signal-workers", str(args.signal_workers),
        "--no-signal-spectrum",
        "--signal-offset-count-per-side", "5", "--signal-offset-step-mhz", "500",
        "--frequency-chunk-size", "10",
        "--overwrite",
        "--log-level", "INFO",
    ]
    subprocess.run(argv, cwd=str(ROOT), check=True)


def align_to_measurement(map_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    meas = amm.load_measurement_map(args.measurement_dir, tuple(args.signal_band_ghz))
    sim = amm.load_sim_map(map_dir)
    fit = amm.align_maps(
        meas, sim,
        freq_shift_bounds=tuple(args.freq_shift_bounds),
        power_shift_bounds=tuple(args.power_shift_bounds),
        fit_freq_range=(args.pump_freq_min_ghz, args.pump_freq_max_ghz),
        fit_power_range=(args.pump_power_min_dbm, args.pump_power_max_dbm),
        min_overlap_frac=args.min_overlap_frac,
    )
    result = {
        "freq_shift_ghz": fit["freq_shift_ghz"],
        "power_shift_db": fit["power_shift_db"],
        "gain_offset_db": fit["gain_offset_db"],
        "rmse_db": fit["rmse_db"],
        "score": fit["score"],
        "overlap_cells": fit["overlap_cells"],
    }
    (map_dir / "align_fit.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate(point: Point, stage: str, iteration: int, out_root: Path,
             args: argparse.Namespace, log_rows: list[dict[str, Any]]) -> dict[str, Any]:
    point_dir = out_root / point.tag()
    design_dir = point_dir / "design"
    map_dir = point_dir / "map"
    fit_path = map_dir / "align_fit.json"

    if fit_path.exists() and not args.overwrite_existing:
        print(f"[{stage} iter={iteration}] {point.tag()}: reusing cached result")
        fit = json.loads(fit_path.read_text())
    else:
        print(f"[{stage} iter={iteration}] {point.tag()}: building design "
              f"(Lj={point.lj_h * 1e12:.4f} pH, Cg={point.cg_f * 1e15:.4f} fF, "
              f"coupler-mode=optimize)")
        build_design(point, design_dir)
        print(f"[{stage} iter={iteration}] {point.tag()}: running gain map "
              f"({args.n_power}x{args.n_frequency})")
        run_gain_map(design_dir, map_dir, args)
        print(f"[{stage} iter={iteration}] {point.tag()}: aligning to measurement")
        fit = align_to_measurement(map_dir, args)

    row = {
        "stage": stage, "iteration": iteration,
        "lj_pH": point.lj_h * 1e12, "cg_fF": point.cg_f * 1e15,
        "map_dir": str(map_dir), **fit,
    }
    log_rows.append(row)
    write_log(out_root, log_rows)
    print(f"[{stage} iter={iteration}] {point.tag()}: "
          f"freq_shift={fit['freq_shift_ghz']:+.4f} GHz "
          f"power_shift={fit['power_shift_db']:+.3f} dB "
          f"rmse={fit['rmse_db']:.3f} dB overlap={fit['overlap_cells']}")
    return row


def write_log(out_root: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "stage", "iteration", "lj_pH", "cg_fF", "map_dir",
        "freq_shift_ghz", "power_shift_db", "gain_offset_db", "rmse_db",
        "score", "overlap_cells",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "tuning_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in cols})


def is_better(candidate: float, current: float) -> bool:
    """True if candidate is a finite improvement over current (NaN/inf-safe --
    a dead overlap_cells=0 alignment returns rmse_db=nan, which must never be
    treated as an improvement just because nan comparisons are always False)."""
    return math.isfinite(candidate) and (not math.isfinite(current) or candidate < current)


def secant_next(x0: float, f0: float, x1: float, f1: float,
                 lo: float, hi: float, fallback_step: float) -> float:
    if f1 == f0:
        x2 = x1 + fallback_step
    else:
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
    return min(max(x2, lo), hi)


def run_lj_stage(args: argparse.Namespace, out_root: Path,
                  log_rows: list[dict[str, Any]]) -> Point:
    lj_lo = args.lj_min_ph * 1e-12
    lj_hi = args.lj_max_ph * 1e-12

    lj0 = args.lj_start_ph * 1e-12
    lj1 = min(max(lj0 * (1.0 + args.lj_step_frac), lj_lo), lj_hi)

    p0 = Point(lj0, args.cg_base_ff * 1e-15)
    r0 = evaluate(p0, "A_lj", 0, out_root, args, log_rows)
    if abs(r0["freq_shift_ghz"]) <= args.freq_tol_ghz:
        return p0

    p1 = Point(lj1, args.cg_base_ff * 1e-15)
    r1 = evaluate(p1, "A_lj", 1, out_root, args, log_rows)

    x0, f0 = lj0, r0["freq_shift_ghz"]
    x1, f1 = lj1, r1["freq_shift_ghz"]
    best = p1 if abs(f1) < abs(f0) else p0

    for it in range(2, args.max_lj_iters):
        if abs(f1) <= args.freq_tol_ghz:
            return Point(x1, args.cg_base_ff * 1e-15)
        x2 = secant_next(x0, f0, x1, f1, lj_lo, lj_hi,
                          fallback_step=args.lj_step_frac * x1)
        p2 = Point(x2, args.cg_base_ff * 1e-15)
        r2 = evaluate(p2, "A_lj", it, out_root, args, log_rows)
        f2 = r2["freq_shift_ghz"]
        if abs(f2) < abs(f1 if abs(f1) < abs(f0) else f0):
            best = p2
        x0, f0 = x1, f1
        x1, f1 = x2, f2

    print(f"stage A: max_lj_iters ({args.max_lj_iters}) reached without "
          f"closing freq_tol_ghz={args.freq_tol_ghz}; using best-seen "
          f"Lj={best.lj_h * 1e12:.4f} pH")
    return best


def run_cg_stage(lj_point: Point, args: argparse.Namespace, out_root: Path,
                  log_rows: list[dict[str, Any]]) -> Point:
    """Bounded 1-D line search on Cg minimizing rmse_db.

    A fixed +/-5% probe (the original spec) turned out to explore too narrow a
    band to matter -- the whole probed range collapsed to a few % around the
    base value. This instead: probes BOTH directions first (the old code broke
    on the first direction that improved, so it could silently never try the
    other side), then expands geometrically (doubling the step) in whichever
    direction wins, so the total excursion from base can be far larger than
    the initial --cg-step-frac step -- bounded by --cg-min-ff/--cg-max-ff so
    it can't wander into an unphysical capacitance.
    """
    cg_lo = args.cg_min_ff * 1e-15
    cg_hi = args.cg_max_ff * 1e-15
    step = args.cg_step_frac

    def clamp(cg: float) -> float:
        return min(max(cg, cg_lo), cg_hi)

    it = 0
    base = Point(lj_point.lj_h, clamp(args.cg_base_ff * 1e-15))
    best_row = evaluate(base, "B_cg", it, out_root, args, log_rows)
    it += 1
    best_point = base
    best_rmse = best_row["rmse_db"]

    if best_rmse <= args.rmse_tol_db or it >= args.max_cg_iters:
        return best_point

    probes: list[tuple[float, Point, float]] = []
    for sign in (+1.0, -1.0):
        cand_cg = clamp(base.cg_f * (1.0 + sign * step))
        if cand_cg == base.cg_f:
            continue  # already at a bound in this direction
        p = Point(lj_point.lj_h, cand_cg)
        row = evaluate(p, "B_cg", it, out_root, args, log_rows)
        it += 1
        probes.append((sign, p, row["rmse_db"]))
        if it >= args.max_cg_iters:
            break

    if not probes:
        return best_point
    finite_probes = [t for t in probes if math.isfinite(t[2])]
    if not finite_probes or not is_better(min(finite_probes, key=lambda t: t[2])[2], best_rmse):
        print("stage B: neither +Cg nor -Cg probe improved rmse_db; stopping at base Cg")
        return best_point
    sign, p, rmse = min(finite_probes, key=lambda t: t[2])
    best_rmse, best_point = rmse, p

    cur_step = step
    while it < args.max_cg_iters and best_rmse > args.rmse_tol_db:
        cur_step = min(cur_step * 2.0, 1.0)
        cand_cg = clamp(best_point.cg_f * (1.0 + sign * cur_step))
        if cand_cg == best_point.cg_f:
            print(f"stage B: hit Cg bound at {best_point.cg_f * 1e15:.4f} fF")
            break
        p = Point(lj_point.lj_h, cand_cg)
        row = evaluate(p, "B_cg", it, out_root, args, log_rows)
        it += 1
        if is_better(row["rmse_db"], best_rmse):
            best_rmse, best_point = row["rmse_db"], p
        else:
            print(f"stage B: Cg expansion stopped improving rmse_db at "
                  f"{best_point.cg_f * 1e15:.4f} fF (last try "
                  f"{p.cg_f * 1e15:.4f} fF)")
            break

    return best_point


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-power", type=int, default=80)
    p.add_argument("--n-frequency", type=int, default=20)
    p.add_argument("--pump-power-min-dbm", type=float, default=-36.0)
    p.add_argument("--pump-power-max-dbm", type=float, default=-19.0)
    p.add_argument("--pump-freq-min-ghz", type=float, default=7.25)
    p.add_argument("--pump-freq-max-ghz", type=float, default=7.75)
    p.add_argument("--measurement-dir", type=Path, default=DEFAULT_MEASUREMENT_DIR)
    p.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    p.add_argument("--freq-shift-bounds", type=float, nargs=2, default=(-1.5, 1.5))
    p.add_argument("--power-shift-bounds", type=float, nargs=2, default=(-6.0, 6.0))
    p.add_argument("--min-overlap-frac", type=float, default=0.25)

    p.add_argument("--lj-start-ph", type=float, default=80.0,
                   help=f"Starting Lj in pH; must be < standard {STANDARD_LJ_H * 1e12:.1f} pH.")
    p.add_argument("--lj-min-ph", type=float, default=60.0)
    p.add_argument("--lj-max-ph", type=float, default=150.0)
    p.add_argument("--lj-step-frac", type=float, default=0.05,
                   help="Initial relative Lj probe step for the secant search.")
    p.add_argument("--freq-tol-ghz", type=float, default=0.05,
                   help="Stage A stops once |freq_shift_ghz| <= this.")
    p.add_argument("--max-lj-iters", type=int, default=6)

    p.add_argument("--cg-base-ff", type=float, default=STANDARD_CG_F * 1e15)
    p.add_argument("--cg-step-frac", type=float, default=0.05,
                   help="Initial relative Cg probe step; doubles on each "
                        "successful expansion (see run_cg_stage), so the "
                        "final excursion from base is normally much larger "
                        "than this -- bounded by --cg-min-ff/--cg-max-ff.")
    p.add_argument("--cg-min-ff", type=float, default=20.0)
    p.add_argument("--cg-max-ff", type=float, default=150.0)
    p.add_argument("--rmse-tol-db", type=float, default=1.5,
                   help="Stage B stops once rmse_db <= this.")
    p.add_argument("--max-cg-iters", type=int, default=8,
                   help="Hard cap on the number of Cg points evaluated "
                        "(base + 2 direction probes + expansion steps).")

    p.add_argument("--signal-workers", type=int, default=6)
    p.add_argument("--out-root", type=Path,
                   default=ROOT / "outputs" / "lj_tune_themis")
    p.add_argument("--overwrite-existing", action="store_true",
                   help="Recompute (Lj, Cg) points even if a cached result exists.")

    args = p.parse_args()
    if args.lj_start_ph >= STANDARD_LJ_H * 1e12:
        p.error(f"--lj-start-ph must be below the standard {STANDARD_LJ_H * 1e12:.1f} pH")
    return args


def main() -> int:
    args = parse_args()
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, Any]] = []

    print("=== Stage A: coarse Lj secant search on freq_shift_ghz ===")
    lj_point = run_lj_stage(args, out_root, log_rows)

    print("=== Stage B: fine +/-5% Cg search on rmse_db ===")
    final_point = run_cg_stage(lj_point, args, out_root, log_rows)

    best = min(log_rows, key=lambda r: (abs(r["freq_shift_ghz"]), r["rmse_db"]))
    summary = {
        "final_lj_pH": final_point.lj_h * 1e12,
        "final_cg_fF": final_point.cg_f * 1e15,
        "best_row": best,
        "n_evaluations": len(log_rows),
    }
    (out_root / "tuning_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_root / 'tuning_summary.json'} and {out_root / 'tuning_log.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
