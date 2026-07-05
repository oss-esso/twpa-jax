"""Experiment 18: map-cell timing with one signal vs a small spectrum.

This mirrors the trusted exp10 in-process map path: solve one pump cell, then
solve linearized signal gain. The extra question is how much wall time changes
when the cell uses a small signal spectrum around the pump instead of the single
trailing point ``fs = fp - 100 MHz``.

Example:
    python experiments/exp18_signal_spectrum_cell_timing.py \
        --ipm-dir outputs/ipm_python_design \
        --pump-power-dbm -26 --pump-freq-ghz 7.5 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import exp09_full_ipm_gain_from_pump as exp09  # noqa: E402
import exp10_full_ipm_pump_map_warmstart as exp10  # noqa: E402


def parse_args() -> argparse.Namespace:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = exp10.parse_args()
    finally:
        sys.argv = old_argv

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "exp18_signal_spectrum_cell_timing")
    p.add_argument("--ipm-dir", type=Path, default=args.ipm_dir)
    p.add_argument("--pump-power-dbm", type=float, default=-26.0)
    p.add_argument("--warm-from-power-dbm", type=float, default=None,
                   help="Optional lower-power neighbor used to seed target as a warm map cell.")
    p.add_argument("--pump-freq-ghz", type=float, default=7.5)
    p.add_argument("--offset-start-mhz", type=float, default=100.0)
    p.add_argument("--offset-step-mhz", type=float, default=250.0)
    p.add_argument("--offset-count-per-side", type=int, default=5,
                   help="5 gives 10 spectrum points: +/-100, +/-350, ... MHz.")
    p.add_argument("--reps", type=int, default=1,
                   help="Repeat independent pump+gain cells; median reported.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sidebands", type=int, default=args.sidebands)
    p.add_argument("--gamma-nt", type=int, default=args.gamma_nt)
    p.add_argument("--pump-mode-count", type=int, default=args.pump_mode_count)
    p.add_argument("--nt", type=int, default=args.nt)
    p.add_argument("--attenuation-db", type=float, default=args.attenuation_db)
    p.add_argument("--z0-ohm", type=float, default=args.z0_ohm)
    p.add_argument("--signal-detuning-mhz", type=float, default=args.signal_detuning_mhz)
    p.add_argument("--pump-timeout-s", type=float, default=args.pump_timeout_s)
    local = p.parse_args()

    for key, value in vars(local).items():
        setattr(args, key, value)
    args.n_power = 1
    args.n_frequency = 1
    args.mode = "warmstart"
    args.executor = "inprocess"
    args.inproc_pump_backend = "schur_cpu_mt"
    args.inproc_preconditioner = "real_coupled_fast"
    args.inproc_fold_predictor = "secant"
    args.inproc_fail_fast = True
    args.fold_skip_patience = 0
    args.signal_ghz = None
    args.pump_mode_policy = "positive_odd_jc"
    return args
def offsets_mhz(args: argparse.Namespace) -> list[float]:
    pos = [args.offset_start_mhz + i * args.offset_step_mhz
           for i in range(args.offset_count_per_side)]
    vals = [-x for x in reversed(pos)] + pos
    return [float(x) for x in vals]


def compute_gain_context(engine: exp10.InProcessEngine, pump_dir: Path, fp_ghz: float, args: argparse.Namespace) -> dict[str, Any]:
    t_all = time.perf_counter()
    pump = exp09.load_pump(pump_dir, fallback_pump_freq_ghz=fp_ghz)
    ms = exp09.sideband_list(args.sidebands)
    max_ell = max(abs(m - q) for m in ms for q in ms)

    t0 = time.perf_counter()
    gamma_hat = exp09.compute_gamma_hat(
        ipm=engine.ipm09,
        pump=pump,
        max_ell=max_ell,
        gamma_nt=args.gamma_nt,
        dc_branch_flux=None,
    )
    gamma_runtime_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    khat = exp09.build_khat(Bphi=engine.ipm09.Bphi, gamma_hat=gamma_hat, drop_tol=0.0)
    khat_runtime_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    gamma_off = engine.ipm09.Ic / engine.ipm09.phi0
    khat_off_0 = (
        engine.ipm09.Bphi @ sp.diags(gamma_off, offsets=0, format="csr") @ engine.ipm09.Bphi.T
    ).astype(np.complex128).tocsr()
    khat_off_runtime_s = time.perf_counter() - t0

    return {
        "pump": pump,
        "khat": khat,
        "khat_off_0": khat_off_0,
        "gamma_runtime_s": gamma_runtime_s,
        "khat_build_runtime_s": khat_runtime_s,
        "khat_off_runtime_s": khat_off_runtime_s,
        "precompute_runtime_s": time.perf_counter() - t_all,
    }


def solve_signal(engine: exp10.InProcessEngine, ctx: dict[str, Any], signal_ghz: float, args: argparse.Namespace) -> exp09.GainResult:
    return exp09.solve_gain_one(
        ipm=engine.ipm09,
        khat=ctx["khat"],
        khat_off_0=ctx["khat_off_0"],
        omega_p=ctx["pump"].omega_p,
        signal_ghz=signal_ghz,
        sidebands=args.sidebands,
        signal_m=0,
        idler_m=-2,
        source_index=engine.source_idx,
        out_index=engine.out_idx,
        source_current_a=1.0,
        source_port=args.source_port,
        out_port=args.out_port,
        z0_ohm=args.z0_ohm,
        loss_model="current_complex_c",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rep", "kind", "offset_mhz", "signal_ghz", "status", "gain_db",
        "linear_rel_residual", "precompute_runtime_s", "assemble_runtime_s",
        "factor_solve_runtime_s", "baseline_off_runtime_s",
        "baseline_pumpdiag_runtime_s", "point_signal_runtime_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def median(values: list[float]) -> float:
    xs = [float(x) for x in values if np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def main() -> int:
    args = parse_args()
    args.pump_power_min_dbm = args.pump_power_dbm
    args.pump_power_max_dbm = args.pump_power_dbm
    args.pump_freq_min_ghz = args.pump_freq_ghz
    args.pump_freq_max_ghz = args.pump_freq_ghz
    args.n_power = 1
    args.n_frequency = 1
    args.executor = "inprocess"
    args.mode = "warmstart"

    if args.outdir.exists() and args.overwrite:
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    offs = offsets_mhz(args)
    current = exp10.dbm_to_peak_current_a(
        args.pump_power_dbm, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm
    )
    engine = exp10.InProcessEngine(args)

    point_rows: list[dict[str, Any]] = []
    rep_summaries: list[dict[str, Any]] = []

    for rep in range(args.reps):
        pass_dir = args.outdir / f"rep_{rep:02d}"
        warm_X = None
        if args.warm_from_power_dbm is not None:
            warm_current = exp10.dbm_to_peak_current_a(
                args.warm_from_power_dbm, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm
            )
            warm_point = exp10.GridPoint(rep * 2, 0, 0, args.warm_from_power_dbm, args.pump_freq_ghz, warm_current)
            warm_row, warm_X = engine.solve_point(warm_point, pass_dir / "warmup", mode="cold", warm_X=None)
            if warm_row["pump_status"] != "VALID_CONVERGED" or warm_X is None:
                raise RuntimeError(f"warmup pump failed in rep {rep}: {warm_row['pump_status']}")

        point = exp10.GridPoint(rep * 2 + 1, 1, 0, args.pump_power_dbm, args.pump_freq_ghz, current)
        mode = "warm" if warm_X is not None else "cold"
        row, _ = engine.solve_point(point, pass_dir / "map_style", mode=mode, warm_X=warm_X)
        if row["pump_status"] != "VALID_CONVERGED":
            raise RuntimeError(f"pump failed in rep {rep}: {row['pump_status']}")

        pump_dir = Path(row["pump_dir"])
        ctx = compute_gain_context(engine, pump_dir, args.pump_freq_ghz, args)
        single_fs = args.pump_freq_ghz - args.signal_detuning_mhz / 1000.0
        t0 = time.perf_counter()
        single = solve_signal(engine, ctx, single_fs, args)
        single_wall = time.perf_counter() - t0
        single_signal_total = ctx["precompute_runtime_s"] + single_wall

        point_rows.append({
            "rep": rep, "kind": "single_trailing", "offset_mhz": -args.signal_detuning_mhz,
            "signal_ghz": single.signal_ghz, "status": single.status,
            "gain_db": single.gain_db, "linear_rel_residual": single.linear_rel_residual,
            "precompute_runtime_s": ctx["precompute_runtime_s"],
            "assemble_runtime_s": single.assemble_runtime_s,
            "factor_solve_runtime_s": single.factor_solve_runtime_s,
            "baseline_off_runtime_s": single.baseline_off_runtime_s,
            "baseline_pumpdiag_runtime_s": single.baseline_pumpdiag_runtime_s,
            "point_signal_runtime_s": single_signal_total,
        })

        spec_ctx = compute_gain_context(engine, pump_dir, args.pump_freq_ghz, args)
        spectrum_wall_sum = 0.0
        spectrum_results: list[exp09.GainResult] = []
        for off in offs:
            fs = args.pump_freq_ghz + off / 1000.0
            t0 = time.perf_counter()
            g = solve_signal(engine, spec_ctx, fs, args)
            dt = time.perf_counter() - t0
            spectrum_wall_sum += dt
            spectrum_results.append(g)
            point_rows.append({
                "rep": rep, "kind": "spectrum", "offset_mhz": off,
                "signal_ghz": g.signal_ghz, "status": g.status,
                "gain_db": g.gain_db, "linear_rel_residual": g.linear_rel_residual,
                "precompute_runtime_s": spec_ctx["precompute_runtime_s"] if len(spectrum_results) == 1 else 0.0,
                "assemble_runtime_s": g.assemble_runtime_s,
                "factor_solve_runtime_s": g.factor_solve_runtime_s,
                "baseline_off_runtime_s": g.baseline_off_runtime_s,
                "baseline_pumpdiag_runtime_s": g.baseline_pumpdiag_runtime_s,
                "point_signal_runtime_s": dt,
            })

        spectrum_signal_total = spec_ctx["precompute_runtime_s"] + spectrum_wall_sum
        rep_summary = {
            "rep": rep,
            "pump_status": row["pump_status"],
            "pump_runtime_s": row["pump_runtime_s"],
            "pump_wall_runtime_s": row["pump_wall_runtime_s"],
            "pump_newton_total": row["pump_newton_total"],
            "pump_gmres_total": row["pump_gmres_total"],
            "single_signal_total_s": single_signal_total,
            "single_cell_total_s": row["pump_wall_runtime_s"] + single_signal_total,
            "spectrum_points": len(offs),
            "spectrum_signal_total_s": spectrum_signal_total,
            "spectrum_cell_total_s": row["pump_wall_runtime_s"] + spectrum_signal_total,
            "spectrum_over_single_signal": spectrum_signal_total / max(single_signal_total, 1e-300),
            "spectrum_over_single_cell": (row["pump_wall_runtime_s"] + spectrum_signal_total) / max(row["pump_wall_runtime_s"] + single_signal_total, 1e-300),
            "avg_extra_signal_point_s": (spectrum_signal_total - single_signal_total) / max(len(offs) - 1, 1),
            "single_gain_db": single.gain_db,
            "spectrum_gain_db_min": min(g.gain_db for g in spectrum_results),
            "spectrum_gain_db_max": max(g.gain_db for g in spectrum_results),
        }
        rep_summaries.append(rep_summary)
        print(
            f"rep {rep}: pump={rep_summary['pump_wall_runtime_s']:.3f}s "
            f"single_cell={rep_summary['single_cell_total_s']:.3f}s "
            f"spectrum_cell={rep_summary['spectrum_cell_total_s']:.3f}s "
            f"ratio={rep_summary['spectrum_over_single_cell']:.2f}x",
            flush=True,
        )

    write_csv(args.outdir / "signal_points.csv", point_rows)

    summary = {
        "pump_power_dbm": args.pump_power_dbm,
        "pump_freq_ghz": args.pump_freq_ghz,
        "current_a": current,
        "offsets_mhz": offs,
        "settings": {
            "ipm_dir": str(args.ipm_dir),
            "sidebands": args.sidebands,
            "gamma_nt": args.gamma_nt,
            "pump_backend": args.inproc_pump_backend,
            "preconditioner": args.inproc_preconditioner,
            "pump_mode_policy": args.pump_mode_policy,
            "pump_mode_count": args.pump_mode_count,
            "nt": args.nt,
        },
        "reps": rep_summaries,
        "median": {
            "pump_wall_runtime_s": median([r["pump_wall_runtime_s"] for r in rep_summaries]),
            "single_signal_total_s": median([r["single_signal_total_s"] for r in rep_summaries]),
            "single_cell_total_s": median([r["single_cell_total_s"] for r in rep_summaries]),
            "spectrum_signal_total_s": median([r["spectrum_signal_total_s"] for r in rep_summaries]),
            "spectrum_cell_total_s": median([r["spectrum_cell_total_s"] for r in rep_summaries]),
            "spectrum_over_single_signal": median([r["spectrum_over_single_signal"] for r in rep_summaries]),
            "spectrum_over_single_cell": median([r["spectrum_over_single_cell"] for r in rep_summaries]),
            "avg_extra_signal_point_s": median([r["avg_extra_signal_point_s"] for r in rep_summaries]),
        },
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    m = summary["median"]
    md = [
        "# exp18 signal-spectrum cell timing",
        "",
        f"Cell: fp={args.pump_freq_ghz:g} GHz, P={args.pump_power_dbm:g} dBm, {len(offs)} spectrum points.",
        "",
        "| metric | median s |",
        "| --- | ---: |",
        f"| pump wall | {m['pump_wall_runtime_s']:.6g} |",
        f"| single signal stage | {m['single_signal_total_s']:.6g} |",
        f"| single full cell | {m['single_cell_total_s']:.6g} |",
        f"| spectrum signal stage | {m['spectrum_signal_total_s']:.6g} |",
        f"| spectrum full cell | {m['spectrum_cell_total_s']:.6g} |",
        "",
        f"Spectrum/single signal-stage ratio: **{m['spectrum_over_single_signal']:.3g}x**.",
        f"Spectrum/single full-cell ratio: **{m['spectrum_over_single_cell']:.3g}x**.",
        f"Average marginal added signal point: **{m['avg_extra_signal_point_s']:.3g} s**.",
        "",
        f"Outputs: `{args.outdir / 'signal_points.csv'}`, `{args.outdir / 'summary.json'}`.",
    ]
    (args.outdir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.outdir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



