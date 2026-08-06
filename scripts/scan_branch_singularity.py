"""Diagnostic column scan: march pump power at one frequency, recording the
Jacobian singularity test functions (``twpa_solver.pump.singularity``)
alongside each point's convergence outcome.

Built for `docs/development/arclength_metric_fix_and_fold_test_function_plan.md`
Phase 5. The investigation doc's fp=7.9 GHz "no real fold found" and fp=7.0
GHz "genuine, tight fold" conclusions were both produced by the unscaled
``solve_arclength`` (natural-parameter continuation in disguise -- see the
docstring history in ``src/twpa_solver/pump/solver.py``), via
``--fold-follow``/``--fold-policy arclength``: an instrument that could not
tell a genuine fold apart from a numerical wall. This driver re-measures with
the fixed ``solve_arclength`` plus a direct singularity measurement
(``jacobian_min_eigenvalue``, ``jacobian_det_signature``) that does not
depend on continuation succeeding at all.

Reading the CSV: ``min_eigenvalue -> 0`` smoothly as power approaches the
wall, with ``det_sign`` flipping there, means a genuine fold (and gives the
fold power). ``min_eigenvalue`` staying at its low-power magnitude while
Newton/arclength fail to converge means the wall is numerical, not physical.
Repeated near-zero ``min_eigenvalue`` / multiple ``det_sign`` flips in a
narrow band means snaking (Phase 6 territory), not a single simple fold.

Standalone diagnostic; not wired into the production map. Takes the same
engine flags as ``run_gain_map.py`` via ``--extra-engine-args`` (must be
last). Windows note: use a short ``--outdir`` (MAX_PATH), matching the
matrix-tracing driver's own caveat.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.singularity import (  # noqa: E402
    jacobian_det_signature,
    jacobian_min_eigenvalue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--pump-freq-ghz", type=float, required=True)
    parser.add_argument("--power-min-dbm", type=float, required=True)
    parser.add_argument("--power-max-dbm", type=float, required=True)
    parser.add_argument("--n-power", type=int, default=15)
    parser.add_argument("--arclength-ds", type=float, default=0.02)
    parser.add_argument("--arclength-max-steps", type=int, default=60)
    parser.add_argument("--eig-iters", type=int, default=20)
    parser.add_argument("--extra-engine-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def _engine_args(cli: argparse.Namespace):
    map_argv = [
        "--circuit-dir", str(cli.circuit_dir.resolve()),
        "--outdir", str(cli.outdir / "_unused_map_dir"),
        "--executor", "inprocess", "--mode", "warmstart",
        "--inproc-pump-backend", "schur_cpu_mt",
        "--inproc-preconditioner", "real_coupled_fast",
        "--n-power", "1", "--n-frequency", "1",
        "--pump-power-min-dbm", str(cli.power_min_dbm),
        "--pump-power-max-dbm", str(cli.power_max_dbm),
        "--pump-freq-min-ghz", str(cli.pump_freq_ghz),
        "--pump-freq-max-ghz", str(cli.pump_freq_ghz),
        *cli.extra_engine_args,
    ]
    return run_gain_map.parse_args(map_argv)


def _make_point(index: int, power_dbm: float, freq_ghz: float, args) -> "run_gain_map.GridPoint":
    att = run_gain_map.attenuation_db_for(freq_ghz, args)
    current_a = run_gain_map.dbm_to_peak_current_a(
        power_dbm, attenuation_db=att, z0_ohm=args.z0_ohm,
        convention=args.power_convention,
    )
    return run_gain_map.GridPoint(
        index=index, i_power=index, j_freq=0, power_dbm=power_dbm,
        pump_freq_ghz=freq_ghz, current_a=current_a,
    )


def _converged(reports) -> bool:
    return bool(
        reports and reports[-1].converged
        and abs(reports[-1].source_scale - 1.0) < 1e-12
    )


def main() -> int:
    cli = parse_args()
    cli.outdir.mkdir(parents=True, exist_ok=True)
    engine_args = _engine_args(cli)
    engine = run_gain_map.InProcessEngine(engine_args)
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    use_schur = engine_args.inproc_pump_backend == "schur_cpu_mt"

    powers = np.linspace(cli.power_min_dbm, cli.power_max_dbm, cli.n_power)
    rows: list[dict[str, Any]] = []
    prev_X: np.ndarray | None = None
    prev_current_a: float | None = None

    for i, power_dbm in enumerate(powers):
        point = _make_point(i, float(power_dbm), cli.pump_freq_ghz, engine_args)
        full_problem, _basis, _omega, injected = engine.build_problem_for(point)
        solve_problem = engine._make_solve_problem(full_problem, point.pump_freq_ghz)

        t0 = time.perf_counter()
        if prev_X is None:
            X, reports, _trace = solver.solve_adaptive_continuation(
                solve_problem, solve_problem.zeros(),
                initial_step=engine_args.adaptive_initial_step,
                min_step=engine_args.adaptive_min_step, growth=1.5, shrink=0.5,
                fallback_fixed_steps=engine_args.inproc_fallback_fixed_steps,
                max_wall_s=engine_args.inproc_solve_deadline_s,
            )
        else:
            X, reports = solver.solve_direct(solve_problem, prev_X)
        converged = _converged(reports)

        arclength_used = False
        arc_info: dict[str, Any] = {}
        lam_reached = 1.0 if converged else 0.0
        if not converged and prev_X is not None:
            # Round the fold the same way production recovery does
            # (run_gain_map.py::_recover, --fold-policy arclength): warm-start
            # from the last converged neighbour toward this target, at the
            # fraction of the way there that neighbour already represents.
            lam0 = min(prev_current_a / injected, 0.98) if injected > 0.0 else 0.0
            X_arc, lam_arc, arc_info = solver.solve_arclength(
                solve_problem, prev_X, lam0,
                ds=cli.arclength_ds, max_steps=cli.arclength_max_steps,
                target_lam=1.0, max_wall_s=engine_args.inproc_solve_deadline_s,
            )
            arclength_used = True
            lam_reached = float(lam_arc)
            X = X_arc
            if arc_info.get("reached_target"):
                X, reports = solver.solve_direct(solve_problem, X_arc)
                converged = _converged(reports)
                lam_reached = 1.0 if converged else lam_reached

        solve_wall_s = time.perf_counter() - t0

        min_eig = jacobian_min_eigenvalue(solve_problem, X, lam_reached, iters=cli.eig_iters)
        det_sign, log_abs_det = jacobian_det_signature(solve_problem, X, lam_reached)

        X_full = solve_problem.reconstruct_full(X) if use_schur else X
        summary = exp08.summarize_solution(full_problem, X_full)
        peak_i = summary.get("branch_i_max_abs")
        peak_i_over_ic = (float(peak_i) / engine.ic_median) if peak_i is not None else None

        coeff_rel = float(reports[-1].coeff_rel) if reports else None
        row = {
            "index": i,
            "pump_power_dbm": float(power_dbm),
            "pump_current_peak_a": injected,
            "converged": converged,
            "lambda_reached": lam_reached,
            "coeff_rel": coeff_rel,
            "arclength_used": arclength_used,
            "arclength_reached_target": arc_info.get("reached_target"),
            "arclength_terminal_reason": arc_info.get("terminal_reason"),
            "arclength_fold_lambda": arc_info.get("fold_lambda"),
            "arclength_state_scale": arc_info.get("state_scale"),
            "min_eigenvalue": min_eig,
            "det_sign": det_sign,
            "log_abs_det": log_abs_det,
            "peak_i_over_ic": peak_i_over_ic,
            "solve_wall_s": solve_wall_s,
        }
        rows.append(row)
        print(
            f"[{i + 1}/{cli.n_power}] P={power_dbm:.4f} dBm converged={converged} "
            f"lambda={lam_reached:.6f} coeff_rel={coeff_rel} min_eig={min_eig:.6e} "
            f"det_sign={det_sign} peak_i/Ic={peak_i_over_ic} "
            f"arclength={arclength_used} wall_s={solve_wall_s:.2f}",
            flush=True,
        )

        # Only chain off a converged neighbour (matches run_warm_pass_inprocess);
        # a non-converged point still gets diagnosed above but does not become
        # the next point's warm anchor.
        if converged:
            prev_X = X
            prev_current_a = injected

    csv_path = cli.outdir / "singularity_scan.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_plot(rows, cli.outdir / "singularity_scan.png", cli.pump_freq_ghz)

    print(f"csv={csv_path}")
    return 0


def _write_plot(rows: list[dict[str, Any]], out_path: Path, freq_ghz: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    powers = [r["pump_power_dbm"] for r in rows]
    min_eigs = [r["min_eigenvalue"] for r in rows]
    det_signs = [r["det_sign"] for r in rows]
    peak_i = [r["peak_i_over_ic"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(powers, min_eigs, marker="o")
    axes[0].set_ylabel("min eigenvalue")
    axes[0].axhline(0.0, color="k", linewidth=0.5)
    axes[1].plot(powers, det_signs, marker="o")
    axes[1].set_ylabel("det sign")
    axes[1].set_yticks([-1, 0, 1])
    axes[2].plot(powers, peak_i, marker="o")
    axes[2].set_ylabel("peak I / Ic")
    axes[2].set_xlabel("pump power (dBm)")
    fig.suptitle(f"Singularity scan, fp={freq_ghz:.4f} GHz")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
