"""Milestone G1.5 resolution experiment (fold_plan.md, 2026-08-08): does the
mu~0.674 fold at 7.9 GHz move under increased HB harmonic resolution?

Per the user's decision rule: the headroom audit found the wall occurring
at max|sin(phi_j)|~0.79, max|phi_j|~0.92 rad (pi/2=1.5708), min cos(phi_j)
~0.61 -- comfortably under-driven, not a junction pegged near its critical
current or a phase excursion near pi/2. That is the "still looks
under-driven" branch of the user's own decision tree, which calls for
exactly one resolution check: retrace the SAME local fold (same ds/
max_steps/max_steps_after_fold recipe as
``validate_fold_plan_f_report.py``'s ``main_fold`` branch) at baseline
resolution (``--pump-mode-count 10 --nt 40``, production default) and at
richer resolution (``--pump-mode-count 14 --nt 80``), and compare the
tangent-sign-flip location nearest mu~0.674 between the two runs.

Each resolution is a fully independent trace from mu=0 (no basis-promotion
needed -- a different ``--pump-mode-count`` is a genuinely different basis
dimension) -- one process per resolution, per the Milestone F OOM lesson.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from scripts.validate_fold_plan_ad import _build_ref_problem  # noqa: E402
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.solver import find_fold_candidates, trace_branch  # noqa: E402

FREQ_GHZ = 7.9
REF_DBM = -16.0
DS = 0.005
MU_MAX = 0.68
MAX_STEPS = 400
MAX_STEPS_AFTER_FOLD = 20

RESOLUTIONS = {
    "baseline": dict(pump_mode_count=10, nt=40),
    "rich": dict(pump_mode_count=14, nt=80),
}


def _engine_args(circuit_dir: Path, outdir: Path, pump_mode_count: int, nt: int) -> argparse.Namespace:
    argv = [
        "--circuit-dir", str(circuit_dir.resolve()),
        "--outdir", str(outdir / "_unused"),
        "--executor", "inprocess", "--mode", "warmstart",
        "--inproc-pump-backend", "schur_cpu_mt",
        "--inproc-preconditioner", "real_coupled_fast",
        "--pump-mode-count", str(pump_mode_count), "--nt", str(nt),
        "--n-power", "1", "--n-frequency", "1",
        "--pump-power-min-dbm", str(REF_DBM), "--pump-power-max-dbm", str(REF_DBM),
        "--pump-freq-min-ghz", str(FREQ_GHZ), "--pump-freq-max-ghz", str(FREQ_GHZ),
    ]
    return run_gain_map.parse_args(argv)


def run_one(which: str) -> None:
    cfg = RESOLUTIONS[which]
    circuit_dir = ROOT / "designs" / "ipm_2c_fixed"
    scratch = Path("D:/tmp/g1_5_resolution") / which
    args = _engine_args(circuit_dir, scratch, cfg["pump_mode_count"], cfg["nt"])
    engine = run_gain_map.InProcessEngine(args)
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    problem, basis, ref_injected = _build_ref_problem(engine, FREQ_GHZ, REF_DBM)

    print(f"{which}: pump_mode_count={cfg['pump_mode_count']} nt={cfg['nt']} "
          f"n_modes={len(basis.k)} basis_dim={problem.zeros().size}", flush=True)

    t0 = time.perf_counter()
    branch = trace_branch(
        solver, problem, i_ref=ref_injected, mu0=0.0, mu_max=MU_MAX, ds=DS,
        max_steps=MAX_STEPS, max_steps_after_fold=MAX_STEPS_AFTER_FOLD,
        step_control="adaptive", rescale_every=5, refine_fold=True,
    )
    runtime_s = time.perf_counter() - t0

    candidates = find_fold_candidates(branch)
    mus = [0.5 * (c.mu_before + c.mu_after) for c in candidates]
    main_fold_mu = max(mus) if mus else None

    print(f"{which}: {len(branch.points)} points, terminal={branch.info.get('terminal_reason')}, "
          f"runtime={runtime_s:.1f}s")
    print(f"{which}: {len(candidates)} candidate(s) at mu={mus}")
    print(f"{which}: main_fold_mu (largest-mu candidate) = {main_fold_mu}")
    print(f"RESULT {which} main_fold_mu={main_fold_mu}")
    print(f"DONE_G1_5_RESOLUTION_{which}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=sorted(RESOLUTIONS), required=True)
    args = p.parse_args()
    run_one(args.which)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
