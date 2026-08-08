"""Milestone F real-solver validation report (fold_plan.md, 2026-08-07).

Runs the actual PRODUCTION functions (trace_branch, find_fold_candidates,
classify_fold_candidates -- not a parallel diagnostic reimplementation)
against three branch settings on the real design/ipm_2c_fixed circuit:

- 7.9 GHz, mu extended to 0.75 so ONE trace covers both the ambiguous
  mu~0.5253 region (Milestone D.5/E) and the known main fold near mu~0.674
  -- the required positive control.
- 7.0 GHz, targeted near the known ~0.7332 fold (mu_max=0.78, a larger step
  budget than the E report's run, which found zero candidates that time on
  path variance).
- A boring no-fold branch (negative control: F must do zero expensive work).

Writes the required region/mu/sigma1/sigma2/gap/tau/alpha/bordered_cond/
class table to stdout, and a mu(s)/t_mu(s)/log10(sigma1_hat(s)) figure for
the mu~0.5253 region.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from scripts.validate_fold_plan_ad import _build_ref_problem, _engine_args  # noqa: E402
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.bifurcation import classify_fold_candidates  # noqa: E402
from twpa_solver.pump.solver import find_fold_candidates, trace_branch  # noqa: E402

OUT_DIR = ROOT / "outputs" / "fold_plan_milestone_f"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(x, spec: str = ".4g") -> str:
    if x is None:
        return "n/a"
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def report_branch(
    label: str, engine, freq_ghz: float, ref_dbm: float,
    *, ds: float, mu_max: float, max_steps: int, max_steps_after_fold,
) -> list[dict]:
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    problem, _basis, ref_injected = _build_ref_problem(engine, freq_ghz, ref_dbm)

    t0 = time.perf_counter()
    branch = trace_branch(
        solver, problem, i_ref=ref_injected, mu0=0.0, mu_max=mu_max, ds=ds,
        max_steps=max_steps, max_steps_after_fold=max_steps_after_fold,
        step_control="adaptive", rescale_every=5, refine_fold=True,
    )
    trace_s = time.perf_counter() - t0

    print(f"\n{'=' * 100}")
    print(f"{label}: {freq_ghz} GHz, ref={ref_dbm} dBm, ds={ds}, mu_max={mu_max}")
    print(f"{len(branch.points)} accepted points, terminal={branch.info.get('terminal_reason')}, "
          f"trace_runtime={trace_s:.1f}s")

    candidates = find_fold_candidates(branch)
    print(f"{len(candidates)} fold candidate(s) at mu = "
          f"{[round(0.5 * (c.mu_before + c.mu_after), 6) for c in candidates]}")

    t0 = time.perf_counter()
    rows = classify_fold_candidates(solver, problem, branch)
    classify_s = time.perf_counter() - t0
    print(f"classify_fold_candidates: {len(rows)} region(s), runtime={classify_s:.1f}s")

    header = (
        f"{'region':<10}{'mu':>12}{'sigma1':>14}{'sigma2':>14}{'sigma1/sigma2':>16}"
        f"{'tau':>12}{'alpha':>14}{'gamma':>14}{'bordered_cond':>16}{'class':>32}"
    )
    print(header)
    for row in rows:
        region_label = "E" + "/".join(str(i) for i in row["candidate_ids"])
        print(
            f"{region_label:<10}{_fmt(row['mu']):>12}{_fmt(row['sigma1']):>14}"
            f"{_fmt(row['sigma2']):>14}{_fmt(row['sigma_gap']):>16}"
            f"{_fmt(row['tau']):>12}{_fmt(row['alpha']):>14}{_fmt(row['gamma']):>14}"
            f"{_fmt(row['bordered_conditioning']):>16}{row['class']:>32}"
        )
        print(f"    reason: {row['reason']}")
        print(
            f"    alpha_hat={_fmt(row.get('alpha_hat'))} gamma_hat={_fmt(row.get('gamma_hat'))} "
            f"r_right={_fmt(row.get('singular_residual_right'))} "
            f"r_left={_fmt(row.get('singular_residual_left'))} "
            f"sigma1_local_contrast={_fmt(row.get('sigma1_local_contrast'))} "
            f"(healthy refs: {row.get('sigma1_healthy_reference')})"
        )

    for row in rows:
        row["_label"] = label
        row["_freq_ghz"] = freq_ghz
    return rows


def plot_singular_region(row: dict, out_name: str) -> None:
    profile = row.get("profile") or []
    if not profile:
        print(f"(no profile data to plot for {out_name})")
        return
    s = [p["s"] for p in profile]
    mu = [p["mu"] for p in profile]
    t_mu = [p["t_mu"] for p in profile]
    log_sigma1_hat = [
        (None if p["sigma1_hat"] is None or p["sigma1_hat"] <= 0 else
         __import__("math").log10(p["sigma1_hat"]))
        for p in profile
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(s, mu, "o-")
    axes[0].set_ylabel("mu(s)")
    axes[0].set_title(f"Milestone F local profile: {row['_label']} ({row['_freq_ghz']} GHz), "
                       f"region {row['candidate_ids']}, class={row['class']}")
    axes[1].plot(s, t_mu, "o-", color="tab:orange")
    axes[1].axhline(0.0, color="k", linewidth=0.5)
    axes[1].set_ylabel("t_mu(s)")
    axes[2].plot(s, log_sigma1_hat, "o-", color="tab:red")
    axes[2].set_ylabel("log10(sigma1_hat(s))")
    axes[2].set_xlabel("s (branch arclength proxy)")
    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


# Each entry is its own isolated process invocation (`--which NAME`) -- the
# combined single-process run (all three branches sequentially) was OOM-
# killed (exit 137) on this machine, most likely because the 7.9 GHz
# mu_max=0.75 trace crosses the genuine physical fold near mu~0.674 and then
# spends its whole max_steps_after_fold budget on post-fold recovery
# (fold_plan.md's own prior measurement: "even 150 further steps cannot
# cross back to target -- the budget is exhausted rounding the fold"), while
# smallest_singular_triplets simultaneously holds a second ~50k x 50k LU
# factorization (J^-T) alongside the production factor's own J^-1. Splitting
# into separate processes lets each run's peak memory be released before the
# next starts, and keeps the near-fold budgets small (rounding the fold is
# not needed -- only enough points to bracket the sign flip for the profile
# stencil).
BRANCHES = {
    "region_0525": dict(
        label="7.9 GHz (0.525 region)", freq_ghz=7.9, ref_dbm=-16.0,
        ds=0.005, mu_max=0.60, max_steps=1200, max_steps_after_fold=300,
    ),
    "main_fold": dict(
        label="7.9 GHz (main fold ~0.674 control)", freq_ghz=7.9, ref_dbm=-16.0,
        ds=0.005, mu_max=0.68, max_steps=400, max_steps_after_fold=20,
    ),
    "freq70": dict(
        label="7.0 GHz (targeted ~0.7332 control)", freq_ghz=7.0, ref_dbm=-21.0,
        ds=0.02, mu_max=0.75, max_steps=400, max_steps_after_fold=20,
    ),
    "boring": dict(
        label="Boring no-fold", freq_ghz=7.9, ref_dbm=-16.0,
        ds=0.02, mu_max=0.3, max_steps=100, max_steps_after_fold=None,
    ),
}


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=sorted(BRANCHES), required=True)
    parser.add_argument("--rows-out", type=Path, default=None)
    args = parser.parse_args()

    circuit_dir = ROOT / "designs" / "ipm_2c_fixed"
    scratch = Path("D:/tmp/f_real_solver_report") / args.which
    cfg = BRANCHES[args.which]

    engine = run_gain_map.InProcessEngine(
        _engine_args(circuit_dir, scratch, cfg["freq_ghz"], cfg["ref_dbm"]),
    )
    rows = report_branch(
        cfg["label"], engine, cfg["freq_ghz"], cfg["ref_dbm"],
        ds=cfg["ds"], mu_max=cfg["mu_max"], max_steps=cfg["max_steps"],
        max_steps_after_fold=cfg["max_steps_after_fold"],
    )

    if args.rows_out is not None:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_out.open("w") as fh:
            json.dump([{k: v for k, v in r.items() if k != "profile"} | {"profile": r.get("profile")}
                       for r in rows], fh, default=str, indent=2)
        print(f"wrote {args.rows_out}")

    if args.which == "region_0525":
        region_0525 = min(
            (r for r in rows if r["mu"] is not None), key=lambda r: abs(r["mu"] - 0.5253), default=None,
        )
        if region_0525 is not None:
            plot_singular_region(region_0525, "sigma1_profile_0p525.png")
    elif args.which == "main_fold":
        region_main = max((r for r in rows if r["mu"] is not None), key=lambda r: r["mu"], default=None)
        if region_main is not None:
            plot_singular_region(region_main, "sigma1_profile_main_fold.png")

    print("\nDONE_F_REPORT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
