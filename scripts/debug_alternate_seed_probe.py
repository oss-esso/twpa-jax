"""Phase A2 (docs/multistability_and_cli_trim_plan.md): direct empirical test
for multistability at a Phase-A1-flagged discontinuity -- a cell where the
column traversal's cold reseed (warm_retry_reseed=True, run_gain_map.py:1622)
converged.

First attempt targeted outputs/campaign_continuation_methods/c04_baseline_prod
(archived campaign, fp=7.70408163265 GHz, i_power=27). That cell's recorded
convergence (PASS, gain_db=23.62) did NOT reproduce under current solver code
-- replaying the exact deterministic cold-reseed path (3x, bit-identical)
gave FAIL / gain_db=19.77 every time. The archived map is not reproducible
under the current checkout and inputs; its available provenance does not
establish whether solver code, circuit inputs, dependencies, or configuration
changed. It is not a live discontinuity to probe. Confirmed by regenerating
that exact column fresh under current code: zero reseed events (the cell just
fails now, matching the repro). A fresh full
50x50 map under current code (outputs/a2_fresh_map_50x50_full) found 3 live
reseed events elsewhere; this script targets those instead.

This solves the SAME target (P,f) three ways, all in-process (no disk
round-trip beyond loading the two NEIGHBOR solutions -- avoids the full-node
vs Schur-retained-port shape mismatch --initial-pump-dir has, see CLAUDE.md
"Pseudo-arclength" note) via InProcessEngine.solve_point (run_gain_map.py:765):

  (a) cold     -- mode="seed": X=0 Newton ladder (current production reseed
      behavior).
  (b) warm_lo  -- mode="warm", warm_X = the lower-power neighbor's real
      converged X (restricted to retained ports via
      twpa_solver.pump.backends.schur_partition.restrict), solver.solve_direct
      at lambda=1 directly (no ladder).
  (c) warm_hi  -- mode="warm", warm_X = the higher-power neighbor's real X,
      same.

If (b)/(c) converge to a materially different X / gain_db than (a), that is
two genuinely different fixed points at the same (P,f) -- confirmed
multistability, not a solver artifact. Per docs/convergence_investigation_log.md
terminology rule: report as "branch switch confirmed" only if the numbers
show it, not "fold".

Usage:
    python scripts/debug_alternate_seed_probe.py \\
        --freq-ghz 8.234693877553333 --i-power 27 \\
        --power-dbm -28.387755102040817 \\
        --neighbor-lo-pump-dir outputs/a2_fresh_map_50x50_full/.../pump \\
        --neighbor-hi-pump-dir outputs/a2_fresh_map_50x50_full/.../pump \\
        --prod-pump-dir outputs/a2_fresh_map_50x50_full/.../pump

Output (outputs/alternate_seed_probe_<slug>/):
  * probe_summary.csv -- one row per solve mode with X-norm diff vs cold.
  * printed comparison table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gain_map as rgm  # noqa: E402
import twpa_solver.pump.basis as pump_basis  # noqa: E402
from twpa_solver.pump.backends import schur_partition  # noqa: E402

# $Common + c04_baseline_prod flags from run_campaign.ps1 (the config every
# target cell in this investigation was produced under), minus grid/outdir
# (points built manually so only the target frequency needs solving).
BASE_ARGV = [
    "--executor", "inprocess", "--mode", "warmstart",
    "--n-power", "1", "--n-frequency", "1",
    "--inproc-pump-backend", "schur_cpu_mt",
    "--inproc-preconditioner", "real_coupled_fast",
    "--inproc-fold-predictor", "secant",
    "--fold-skip-patience", "4",
    "--inproc-schur-cache-size", "2",
    "--inproc-max-newton", "16",
    "--inproc-solve-deadline-s", "14",
    "--pump-mode-count", "10", "--nt", "40",
    "--signal-detuning-mhz", "100",
    "--signal-backend", "direct", "--signal-solver", "superlu",
    "--sidebands", "10", "--signal-workers", "6",
    "--no-signal-spectrum",
    "--signal-offset-count-per-side", "5", "--signal-offset-step-mhz", "500",
]


def make_point(index: int, i_power: int, power_dbm: float, freq_ghz: float, args) -> rgm.GridPoint:
    current_a = rgm.dbm_to_peak_current_a(
        power_dbm,
        attenuation_db=rgm.attenuation_db_for(freq_ghz, args),
        z0_ohm=args.z0_ohm,
    )
    return rgm.GridPoint(
        index=index, i_power=i_power, j_freq=0,
        power_dbm=power_dbm, pump_freq_ghz=freq_ghz, current_a=current_a,
    )


def load_retained_x(pump_dir: Path, solve_problem) -> np.ndarray:
    x_full, _basis = pump_basis.load_pump_basis_from_solution(pump_dir)
    if hasattr(solve_problem, "part"):
        return schur_partition.restrict(x_full, solve_problem.part)
    return x_full


def x_norm_diff(x_a: np.ndarray | None, x_b: np.ndarray | None) -> float | None:
    if x_a is None or x_b is None or x_a.shape != x_b.shape:
        return None
    denom = np.linalg.norm(x_a)
    if denom == 0:
        return None
    return float(np.linalg.norm(x_a - x_b) / denom)


def _print_row(label: str, pt: "rgm.GridPoint", row: dict) -> None:
    print(
        f"  [{label}] P={pt.power_dbm:.4f}dBm pump_status={row['pump_status']} "
        f"coeff_rel={row['pump_coeff_rel']:.3e} newton={row['pump_newton_total']} "
        f"gmres={row['pump_gmres_total']} gain_db={row['gain_db']}"
    )


def write_summary(out_dir: Path, rows: list[dict]) -> None:
    cols = [
        "run", "pump_status", "pump_coeff_rel", "pump_newton_total",
        "pump_gmres_total", "gain_status", "gain_db",
        "x_norm_diff_vs_cold", "x_norm_diff_vs_prod", "gain_db_diff_vs_cold",
    ]
    out = out_dir / "probe_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            record = {c: r["row"].get(c) for c in cols if c in r["row"]}
            record["run"] = r["run"]
            record["x_norm_diff_vs_cold"] = r["x_norm_diff_vs_cold"]
            record["x_norm_diff_vs_prod"] = r["x_norm_diff_vs_prod"]
            record["gain_db_diff_vs_cold"] = r["gain_db_diff_vs_cold"]
            writer.writerow(record)
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq-ghz", type=float, required=True)
    parser.add_argument("--power-dbm", type=float, required=True)
    parser.add_argument("--i-power", type=int, required=True)
    parser.add_argument("--neighbor-lo-pump-dir", type=Path, required=True)
    parser.add_argument("--neighbor-hi-pump-dir", type=Path, required=True)
    parser.add_argument("--prod-pump-dir", type=Path, default=None,
                         help="The target cell's own real converged solution "
                              "(for an x_norm_diff_vs_prod sanity column).")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--circuit-dir", type=str, default="outputs/ipm_python_design")
    args_cli = parser.parse_args()

    out_dir = args_cli.out_dir or (
        ROOT / "outputs" / f"alternate_seed_probe_fp{args_cli.freq_ghz:.6g}_i{args_cli.i_power}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = list(BASE_ARGV) + [
        "--outdir", str(out_dir),
        "--circuit-dir", args_cli.circuit_dir,
        "--pump-power-min-dbm", str(args_cli.power_dbm),
        "--pump-power-max-dbm", str(args_cli.power_dbm),
        "--pump-freq-min-ghz", str(args_cli.freq_ghz),
        "--pump-freq-max-ghz", str(args_cli.freq_ghz),
    ]
    args = rgm.parse_args(argv)
    engine = rgm.InProcessEngine(args)

    target = make_point(102, args_cli.i_power, args_cli.power_dbm, args_cli.freq_ghz, args)

    print("=== building target problem once, reused (prebuilt) across all 3 solves ===")
    prebuilt = engine.build_problem_for(target)
    full_problem, _basis, _omega, _injected = prebuilt
    solve_problem = engine._make_solve_problem(full_problem, args_cli.freq_ghz)

    print("=== loading neighbors' REAL converged X off disk, restricted to retained ports ===")
    x_lo = load_retained_x(args_cli.neighbor_lo_pump_dir, solve_problem)
    x_hi = load_retained_x(args_cli.neighbor_hi_pump_dir, solve_problem)
    x_prod_target = (
        load_retained_x(args_cli.prod_pump_dir, solve_problem)
        if args_cli.prod_pump_dir else None
    )
    print(f"  x_lo shape={x_lo.shape}  x_hi shape={x_hi.shape}")

    print("=== target: three seed strategies (prebuilt problem reused) ===")
    row_cold, x_cold = engine.solve_point(
        target, out_dir / "cold", mode="seed", warm_X=None, prebuilt=prebuilt, force_gain=True,
    )
    _print_row("cold (X=0 ladder)", target, row_cold)

    row_warm_lo, x_warm_lo = engine.solve_point(
        target, out_dir / "warm_from_lo", mode="warm", warm_X=x_lo, prebuilt=prebuilt, force_gain=True,
    )
    _print_row("warm_from_lo (neighbor real X, direct lambda=1)", target, row_warm_lo)

    row_warm_hi, x_warm_hi = engine.solve_point(
        target, out_dir / "warm_from_hi", mode="warm", warm_X=x_hi, prebuilt=prebuilt, force_gain=True,
    )
    _print_row("warm_from_hi (neighbor real X, direct lambda=1)", target, row_warm_hi)

    rows = [
        {"run": "cold", "row": row_cold, "x": x_cold},
        {"run": "warm_from_lo", "row": row_warm_lo, "x": x_warm_lo},
        {"run": "warm_from_hi", "row": row_warm_hi, "x": x_warm_hi},
    ]

    print("\n=== comparison vs cold, and vs the target's own real converged X (if given) ===")
    for r in rows:
        diff_vs_cold = x_norm_diff(x_cold, r["x"])
        diff_vs_prod = x_norm_diff(x_prod_target, r["x"]) if x_prod_target is not None else None
        gain_a = row_cold["gain_db"]
        gain_b = r["row"]["gain_db"]
        gain_diff = (
            (gain_b - gain_a) if (gain_a is not None and gain_b is not None) else None
        )
        r["x_norm_diff_vs_cold"] = diff_vs_cold
        r["x_norm_diff_vs_prod"] = diff_vs_prod
        r["gain_db_diff_vs_cold"] = gain_diff
        print(
            f"  {r['run']:14s} x_norm_diff_vs_cold={diff_vs_cold!s:>10} "
            f"x_norm_diff_vs_prod={diff_vs_prod!s:>10} "
            f"gain_db_diff_vs_cold={gain_diff!s:>10} gain_db={gain_b}"
        )

    write_summary(out_dir, rows)


if __name__ == "__main__":
    main()
