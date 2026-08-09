"""Milestone E real-solver validation report (user-requested follow-up,
2026-08-07, after the DEGENERATE_NEAR_FOLD -> UNRESOLVED_NEAR_FOLD rename
and disjoint -> adjacent pairing correction).

Runs the actual PRODUCTION functions (trace_branch, find_fold_candidates,
classify_adjacent_fold_candidate_pairs -- not a parallel diagnostic
reimplementation) against three branch settings already validated in the
A-D and D.5 campaigns: the 7.9 GHz problematic branch, the 7.0 GHz branch,
and a boring no-fold branch. No new simulation campaign -- same settings,
just calling the real solver.py entry points this time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from scripts.validate_fold_plan_ad import _build_ref_problem, _engine_args, _unit_tangent_at  # noqa: E402
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.solver import (  # noqa: E402
    _real_dot,
    classify_adjacent_fold_candidate_pairs,
    find_fold_candidates,
    trace_branch,
)


def _refine_candidate(solver, problem, branch, candidate) -> dict:
    """Best-effort secant refinement of ONE candidate, for the report table.

    Not part of production Milestone E (only the first candidate on a trace
    gets this via solve_arclength's own refine_fold=True); reuses the same
    _refine_fold primitive Milestone D validated, seeded from the candidate's
    own bracket. Uses branch.info["state_scale"] as the metric scale -- an
    approximation if a metric rescale fired mid-trace (info is not updated
    per rescale), fine for a diagnostic report, not for production numbers.
    """
    pts = branch.points
    a, b = pts[candidate.index - 1], pts[candidate.index]
    state_scale = branch.info.get("state_scale") or 1.0
    S = problem.source_coeffs(1.0)
    Xdot_a, lam_dot_a = _unit_tangent_at(solver, problem, a.X, S, state_scale)
    if lam_dot_a * a.t_mu < 0.0:
        Xdot_a, lam_dot_a = -Xdot_a, -lam_dot_a
    ds_ab = b.s - a.s

    def metric_x(u, v):
        return _real_dot(u, v) / (state_scale * state_scale)

    tol = max(solver.settings.newton_tol * 10.0, 1e-8)
    try:
        return solver._refine_fold(
            problem, a.X, a.mu, Xdot_a, lam_dot_a, b.mu, b.t_mu, ds_ab,
            metric_x, S, 15, 0.0, time.perf_counter(), tol,
            t_tol=1e-8, lam_tol=max(1e-9, abs(ds_ab) * 1e-4), max_iter=40,
        )
    except Exception as exc:  # noqa: BLE001 - report-only, never abort the report
        return {"converged": False, "error": repr(exc)}


def report_branch(
    label: str, engine, freq_ghz: float, ref_dbm: float,
    *, ds: float, mu_max: float, max_steps: int, max_steps_after_fold,
) -> None:
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    problem, _basis, ref_injected = _build_ref_problem(engine, freq_ghz, ref_dbm)

    t0 = time.perf_counter()
    branch = trace_branch(
        solver, problem, i_ref=ref_injected, mu0=0.0, mu_max=mu_max, ds=ds,
        max_steps=max_steps, max_steps_after_fold=max_steps_after_fold,
        step_control="adaptive", rescale_every=5, refine_fold=True,
    )
    runtime = time.perf_counter() - t0

    print(f"\n{'=' * 78}")
    print(f"{label}: {freq_ghz} GHz, ref={ref_dbm} dBm, ds={ds}, mu_max={mu_max}")
    print(f"{len(branch.points)} accepted points, terminal={branch.info.get('terminal_reason')}, "
          f"runtime={runtime:.1f}s")

    candidates = find_fold_candidates(branch)
    print(f"\n{len(candidates)} fold candidate(s):")
    print(f"{'id':<4}{'s_before':>10}{'s_after':>10}{'mu_before':>12}{'mu_after':>12}"
          f"{'sign':>8}{'refined_mu':>14}{'refine_conv':>13}{'refine_iters':>13}")
    for i, c in enumerate(candidates):
        sign = "+->-" if c.mu_after < c.mu_before else "-->+"
        refined = _refine_candidate(solver, problem, branch, c)
        r_mu = refined.get("lam")
        r_mu_s = f"{r_mu:.6f}" if r_mu is not None else "n/a"
        print(f"E{i:<3}{c.s_before:>10.5f}{c.s_after:>10.5f}{c.mu_before:>12.6f}{c.mu_after:>12.6f}"
              f"{sign:>8}{r_mu_s:>14}{str(refined.get('converged')):>13}{str(refined.get('iterations')):>13}")

    pairs = classify_adjacent_fold_candidate_pairs(solver, problem, branch)
    print(f"\n{len(pairs)} adjacent candidate pair(s):")
    for r in pairs:
        print(f"  E{r['candidate_a_id']}-E{r['candidate_b_id']}: "
              f"mu=[{r['candidate_a_mu']:.6f},{r['candidate_b_mu']:.6f}] status={r['status']}")
        if "mu_star" in r:
            print(f"    mu_star={r['mu_star']:.6f}  mu_lo={r.get('mu_lo')}  mu_hi={r.get('mu_hi')}")
        if "roots" in r:
            for seg_label, root in r["roots"].items():
                print(f"    root[{seg_label}]: seed_mu={root['seed_mu']:.6f} "
                      f"converged={root['converged']} coeff_rel={root['coeff_rel']:.3e} "
                      f"norm={root['norm']:.6e}")
        if "pairwise" in r:
            for pair_label, dist in r["pairwise"].items():
                print(f"    distance[{pair_label}] = {dist:.6e}")
        if "reason" in r:
            print(f"    reason: {r['reason']}")


def main() -> int:
    circuit_dir = ROOT / "designs" / "ipm_2c_fixed"
    scratch = Path("D:/tmp/e_real_solver_report")

    # 7.9 GHz problematic branch: same settings D.5 used to find the
    # mu~0.5253 candidate pair (ds=0.005, restart-safe anchor at mu=0,
    # short of the main mu~0.674 fold).
    engine79 = run_gain_map.InProcessEngine(_engine_args(circuit_dir, scratch, 7.9, -16.0))
    report_branch(
        "7.9 GHz problematic", engine79, 7.9, -16.0,
        ds=0.005, mu_max=0.60, max_steps=1200, max_steps_after_fold=300,
    )

    # 7.0 GHz branch: same settings as the A-D campaign's D7 trace (single
    # clean fold at mu~0.7332, metric-mistuning regression check).
    engine70 = run_gain_map.InProcessEngine(_engine_args(circuit_dir, scratch, 7.0, -21.0))
    report_branch(
        "7.0 GHz", engine70, 7.0, -21.0,
        ds=0.02, mu_max=1.0, max_steps=250, max_steps_after_fold=80,
    )

    # Boring no-fold branch: same settings as D8 (mu_max=0.3, well short of
    # any candidate).
    report_branch(
        "Boring no-fold", engine79, 7.9, -16.0,
        ds=0.02, mu_max=0.3, max_steps=100, max_steps_after_fold=None,
    )

    print("\nDONE_E_REPORT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
