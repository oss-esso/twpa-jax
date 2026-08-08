"""Milestone G2 (fold_plan.md, 2026-08-08): demonstrate pseudo-arclength
traversal of the already-validated main simple fold at 7.9 GHz,
mu_f ~= 0.6734293212 (established SIMPLE_FOLD, [[fold-plan-f-bifurcation-
classification]]; basis-converged to 8-9 sig figs,
[[fold-plan-g1-5-fold-is-basis-converged]]).

This is a traversal experiment, not a target-power recovery experiment: it
uses the production ``solve_arclength_mu``/``trace_branch`` machinery
completely unmodified (only additive, opt-in telemetry was added to
``solver.py`` this session -- ``step_telemetry`` on ``solve_arclength``/
``solve_arclength_mu``/``trace_branch``, default ``None``, zero behavior
change) and asks only: does the traced (X(s), mu(s)) pass smoothly through
the fold, with converged HB + PALC residuals and no branch jump?

No signal/gain solves, no new fold classifier, no two-parameter continuation,
no deflation, no transient solving -- pump-solve-only, reusing exactly the
already-validated ``trace_branch``/``find_fold_candidates`` API.

Two stages:
  1. warmup: cheap, uncorrected trace from mu=0 to an anchor near the fold
     (ANCHOR_MU), no telemetry -- just a converged pre-fold checkpoint.
  2. instrumented: fine-step trace from the anchor through and past the
     fold to TARGET_MU_MAX, with full per-step telemetry captured via the
     new ``step_telemetry`` hook.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.validate_fold_plan_ad import (  # noqa: E402
    _build_ref_problem,
    _current_to_dbm,
    _engine_args,
    _observable,
)
from scripts import run_gain_map  # noqa: E402
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.solver import (  # noqa: E402
    BranchPoint,
    find_fold_candidates,
    trace_branch,
)

FREQ_GHZ = 7.9
REF_DBM = -16.0
KNOWN_FOLD_MU = 0.6734293212367346  # F.5 / G1.5-baseline SIMPLE_FOLD location

ANCHOR_MU = 0.62  # "closest trustworthy converged pre-fold state" per spec
TARGET_MU_MAX = 0.75  # bounded post-fold budget, well past the known fold

TRAVERSAL_DS = 0.005
TRAVERSAL_MAX_STEPS = 120
TRAVERSAL_MAX_STEPS_AFTER_FOLD = 50
TRAVERSAL_MAX_WALL_S = 300.0
RESCALE_EVERY = 5

OUT_DIR = ROOT / "outputs" / "fold_plan_milestone_g2"
SUSTAINED_POST_FOLD_MIN_POINTS = 20


def build_problem():
    circuit_dir = ROOT / "designs" / "ipm_2c_fixed"
    args = _engine_args(circuit_dir, Path("D:/tmp/g2_fold_traversal_unused"), FREQ_GHZ, REF_DBM)
    engine = run_gain_map.InProcessEngine(args)
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    problem, basis, ref_injected = _build_ref_problem(engine, FREQ_GHZ, REF_DBM)
    return engine, solver, problem, basis, ref_injected


def run_warmup(engine, solver, problem, ref_injected: float):
    """Produce the fixed-mu anchor through the ordinary production route."""
    print(
        f"[stage 1] direct production solve to mu={ANCHOR_MU} "
        "(target-current map cell)", flush=True,
    )
    target_full, _basis, _omega = engine._build_problem(
        FREQ_GHZ, ANCHOR_MU * ref_injected,
    )
    target_problem = engine._make_solve_problem(target_full, FREQ_GHZ)
    X, reports, trace = solver.solve_adaptive_continuation(
        target_problem, target_problem.zeros(),
        initial_step=engine.args.adaptive_initial_step,
        min_step=engine.args.adaptive_min_step, growth=1.5, shrink=0.5,
        fallback_fixed_steps=engine.args.inproc_fallback_fixed_steps,
        max_wall_s=engine.args.inproc_solve_deadline_s,
    )
    if not reports or not reports[-1].converged:
        print("[stage 1] adaptive seed ladder stalled; retrying with 250 power substeps", flush=True)
        X, reports = solver.solve_continuation(
            target_problem, continuation_steps=250,
            max_wall_s=engine.args.inproc_solve_deadline_s,
        )
    if (
        not reports
        or not reports[-1].converged
        or abs(reports[-1].source_scale - 1.0) > 1e-12
    ):
        raise RuntimeError(
            f"direct production solve did not reach anchor mu={ANCHOR_MU}: "
            f"last_report={reports[-1] if reports else None!r}, "
            f"failure_reason={trace.failure_reason!r}"
        )
    anchor = BranchPoint(
        s=0.0, mu=ANCHOR_MU, X=np.array(X, copy=True), t_mu=0.0, step_size=0.0,
    )
    print(
        f"[stage 1] anchor reached: mu={anchor.mu:.6f}, "
        f"substeps={len(reports)}", flush=True,
    )
    return anchor


def run_traversal(solver, problem, ref_injected: float, anchor, *, rescale_every: int = RESCALE_EVERY):
    telemetry: list[dict[str, Any]] = []
    print(
        f"[stage 2] instrumented trace mu={anchor.mu:.4f} -> {TARGET_MU_MAX} "
        f"(ds={TRAVERSAL_DS}, refine_fold=True, max_steps_after_fold="
        f"{TRAVERSAL_MAX_STEPS_AFTER_FOLD}, rescale_every={rescale_every})", flush=True,
    )
    branch = trace_branch(
        solver, problem, i_ref=ref_injected, X0=anchor.X, mu0=anchor.mu,
        mu_max=TARGET_MU_MAX, ds=TRAVERSAL_DS, max_steps=TRAVERSAL_MAX_STEPS,
        max_wall_s=TRAVERSAL_MAX_WALL_S, step_control="adaptive",
        rescale_every=rescale_every, refine_fold=True,
        max_steps_after_fold=TRAVERSAL_MAX_STEPS_AFTER_FOLD,
        step_telemetry=telemetry.append,
    )
    print(f"[stage 2] done: {len(branch.points)} branch points, "
          f"{len(telemetry)} telemetry entries, "
          f"terminal_reason={branch.info.get('terminal_reason')!r}, "
          f"reached_target={branch.info.get('reached_target')}", flush=True)
    return branch, telemetry


def merge_rows(branch, telemetry, problem) -> list[dict[str, Any]]:
    """One row per accepted step: telemetry + branch.points[i].X-derived observable.

    ``branch.points`` and ``telemetry`` both fire once per accepted step, in
    the same loop iteration (see ``solver.py::solve_arclength``'s ``on_step``
    then ``step_telemetry`` call order) -- they are index-aligned except that
    ``branch.points`` may carry one extra trailing synthetic point (the
    linearly-interpolated ``target_mu`` crossing appended after the loop
    exits), which has no corresponding telemetry entry.
    """
    n = len(telemetry)
    points = branch.points[:n]
    assert len(points) == n, (len(points), n)

    rows: list[dict[str, Any]] = []
    prev_X = None
    prev_state_scale = None
    for i, (pt, tel) in enumerate(zip(points, telemetry)):
        obs = _observable(problem, pt.X)
        if prev_X is None:
            dX_weighted = 0.0
        else:
            raw = float(np.linalg.norm(pt.X - prev_X))
            scale = prev_state_scale if prev_state_scale else 1.0
            dX_weighted = raw / scale / max(tel["step_size"], 1e-12)
        row = {
            "index": i,
            "s": pt.s,
            "mu": tel["mu"],
            "ds": tel["step_size"],
            "t_lam_pred": tel["t_lam_pred"],
            "t_lam_own": tel["t_lam_own"],
            "used_newton": tel["used_newton"],
            "gmres_iterations": tel["gmres_iterations"],
            "hb_residual_rel": tel["hb_residual_rel"],
            "palc_residual": tel["palc_residual"],
            "theta_deg": tel["theta_deg"],
            "rejected_steps": tel["rejected_steps"],
            "state_scale": tel["state_scale"],
            "rescale_event": tel["rescale_event"],
            "rescale_norm": tel["rescale_norm"],
            "rescale_step_size_before": tel["rescale_step_size_before"],
            "rescale_step_size_after": tel["rescale_step_size_after"],
            "rescale_predictor_error": tel["rescale_predictor_error"],
            "dX_weighted": dX_weighted,
            "fundamental_l2": obs["fundamental_l2"],
            "l2_norm": obs["l2_norm"],
            "max_abs_flux": obs["max_abs_flux"],
        }
        rows.append(row)
        prev_X = pt.X
        prev_state_scale = tel["state_scale"]
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def classify_traversal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mus = np.array([r["mu"] for r in rows])
    t_own = np.array([r["t_lam_own"] for r in rows])
    dX = np.array([r["dX_weighted"] for r in rows[1:]])  # first row has no predecessor

    signs = t_own >= 0.0
    flips = [i for i in range(1, len(signs)) if signs[i] != signs[i - 1]]

    # The fold is the FIRST tangent-sign flip (the point after the first
    # +->- transition), matching solve_arclength's own fold_lambda semantics
    # -- NOT argmax(mu). A branch that later re-ascends past the first
    # fold's mu (e.g. inside a densely-oscillating post-fold region) would
    # otherwise have its "fold" silently relabeled to that later, higher
    # point, hiding exactly the topology this experiment exists to reveal.
    if not flips:
        fold_idx = int(np.argmax(mus))
        mu_max = float(mus[fold_idx])
    else:
        fold_idx = flips[0]
        # Midpoint of the bracketing pair, same convention as
        # solver.py::FoldCandidate (mu_before, mu_after).
        mu_max = 0.5 * (float(mus[fold_idx - 1]) + float(mus[fold_idx]))

    post = mus[fold_idx:]
    min_post_idx_rel = int(np.argmin(post))
    min_post_mu = float(post[min_post_idx_rel])
    delta_mu_retreat = mu_max - min_post_mu

    n_post_fold_accepted = len(rows) - fold_idx - 1
    post_fold_s_covered = rows[-1]["s"] - rows[fold_idx]["s"]

    median_dX = float(np.median(dX)) if len(dX) else 0.0
    max_dX = float(np.max(dX)) if len(dX) else 0.0
    jump_flag = max_dX > 10.0 * max(median_dX, 1e-30)

    max_hb_residual = float(np.max([r["hb_residual_rel"] for r in rows]))
    max_palc_residual = float(np.max(np.abs([r["palc_residual"] for r in rows])))

    sustained_negative = False
    if fold_idx < len(rows) - 1:
        tail = t_own[fold_idx + 1:]
        sustained_negative = bool(np.sum(tail < 0.0) >= min(
            SUSTAINED_POST_FOLD_MIN_POINTS, len(tail),
        ))

    turns_upward_again = min_post_idx_rel < len(post) - 1 and mus[-1] > min_post_mu + 1e-4

    # Immediate post-fold run: how many consecutive accepted steps right
    # after the first flip stay on the negative-t_mu side before the next
    # flip (if any). This is what actually crossed the fold, independent
    # of whether the branch later re-enters an oscillatory region.
    immediate_run = 0
    for i in range(fold_idx, len(rows)):
        if t_own[i] < 0.0:
            immediate_run += 1
        else:
            break

    residuals_converged = max_hb_residual < 1e-5 and max_palc_residual < 1e-5
    fold_crossed = (
        len(flips) >= 1 and fold_idx > 0 and immediate_run >= 1 and residuals_converged
    )
    sustained_to_end = fold_crossed and sustained_negative and len(flips) <= 1

    if not fold_crossed:
        status = "NOT_TRAVERSED"
    elif sustained_to_end:
        status = "TRAVERSED"
    else:
        status = "FIRST_FOLD_TRAVERSED_ENTERED_MULTIFOLD_REGION"

    return {
        "status": status,
        "immediate_post_fold_run": immediate_run,
        "residuals_converged": residuals_converged,
        "n_flips": len(flips),
        "flip_indices": flips,
        "fold_idx": fold_idx,
        "mu_at_fold": mu_max,
        "known_fold_mu": KNOWN_FOLD_MU,
        "fold_mu_delta_from_known": mu_max - KNOWN_FOLD_MU,
        "min_post_fold_mu": min_post_mu,
        "delta_mu_retreat": delta_mu_retreat,
        "n_post_fold_accepted": n_post_fold_accepted,
        "post_fold_s_covered": post_fold_s_covered,
        "sustained_negative_tail": sustained_negative,
        "turns_upward_again": turns_upward_again,
        "final_mu": float(mus[-1]),
        "median_dX_weighted": median_dX,
        "max_dX_weighted": max_dX,
        "jump_flag": jump_flag,
        "max_hb_residual_rel": max_hb_residual,
        "max_palc_residual": max_palc_residual,
    }


def representative_table(rows: list[dict[str, Any]], verdict: dict[str, Any], n_want: int = 16) -> list[dict[str, Any]]:
    n = len(rows)
    fold_idx = verdict["fold_idx"]
    idxs = sorted(set([
        0, max(0, fold_idx - 10), max(0, fold_idx - 5), max(0, fold_idx - 2),
        max(0, fold_idx - 1), fold_idx, min(n - 1, fold_idx + 1),
        min(n - 1, fold_idx + 2), min(n - 1, fold_idx + 5),
        min(n - 1, fold_idx + 10), min(n - 1, fold_idx + 20),
        min(n - 1, fold_idx + 40), n - 1,
    ]))
    # pad with evenly spaced indices if under n_want
    if len(idxs) < n_want:
        extra = np.linspace(0, n - 1, n_want, dtype=int).tolist()
        idxs = sorted(set(idxs) | set(extra))
    return [rows[i] for i in idxs]


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = ["index", "s", "mu", "ds", "t_lam_own", "hb_residual_rel",
            "palc_residual", "used_newton", "gmres_iterations",
            "dX_weighted", "fundamental_l2"]
    header = " | ".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:14.6g}")
            else:
                vals.append(f"{v:14}")
        print(" | ".join(vals))


def make_plots(rows: list[dict[str, Any]], verdict: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = np.array([r["s"] for r in rows])
    mu = np.array([r["mu"] for r in rows])
    t_own = np.array([r["t_lam_own"] for r in rows])
    ds = np.array([r["ds"] for r in rows])
    obs = np.array([r["fundamental_l2"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s, mu, ".-")
    ax.axhline(KNOWN_FOLD_MU, color="gray", linestyle="--", label="known fold mu")
    ax.set_xlabel("arclength s")
    ax.set_ylabel("mu")
    ax.set_title(f"mu(s) -- 7.9 GHz traversal, fold_idx={verdict['fold_idx']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mu_vs_s.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s, t_own, ".-")
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_xlabel("arclength s")
    ax.set_ylabel("t_mu (own tangent)")
    ax.set_title("t_mu(s)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "t_mu_vs_s.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s, ds, ".-")
    ax.set_xlabel("arclength s")
    ax.set_ylabel("step size ds (mu units)")
    ax.set_title("ds(s)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ds_vs_s.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s, obs, ".-")
    ax.set_xlabel("arclength s")
    ax.set_ylabel("fundamental_l2 (pump fundamental amplitude)")
    ax.set_title("physical observable vs s")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "observable_vs_s.png", dpi=130)
    plt.close(fig)

    print(f"[plots] written to {OUT_DIR}", flush=True)


def predictor_invariance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check whether periodic metric rescale preserves the physical predictor.

    ``solve_arclength``'s rescale block (Section "Milestone A" of
    ``arclength_metric_fix_and_fold_test_function_plan.md``) renormalizes
    the SAME physical tangent ``(Xdot, lam_dot)`` under a new ``state_scale``
    -- it multiplies the whole augmented tangent by one scalar
    ``1/rescale_norm``, it does not rotate it. That means, for the full
    augmented tangent, the vector difference
    ``ds_after*t_after - ds_before*t_before`` is *itself* a scalar multiple
    of ``t_before`` (both terms are parallel to ``t_before``), so its norm
    reduces to a pure scalar expression that needs only ``ds``/``t_lam``
    (mu-component) from consecutive telemetry rows -- no need to have
    stored the full state-vector tangent to check this:

        e_pred = |ds_after / (ds_before * rescale_norm) - 1|

    where ``rescale_norm = t_lam_own[i] / t_lam_pred[i+1]`` is read
    directly off the rescale event itself (``t_lam_pred`` of the row AFTER
    a rescale is exactly the pre-rescale ``t_lam_own`` divided by
    ``rescale_norm``, by construction of the rescale block).

    New telemetry rows carry the exact metric transition and its
    ``predictor_error``. Older CSVs remain supported by detecting a rescale
    from ``state_scale`` changing between consecutive rows and reconstructing
    the scalar check from the tangent components.
    """
    events = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if b.get("rescale_event", 0.0) > 0.5:
            events.append({
                "row_index": i,
                "mu": a["mu"],
                "state_scale_before": a["state_scale"],
                "state_scale_after": b["state_scale"],
                "scale_ratio": b["state_scale"] / a["state_scale"],
                "t_lam_before_rescale": a["t_lam_own"],
                "t_lam_after_rescale": b["t_lam_pred"],
                "rescale_norm": b["rescale_norm"],
                "ds_before": b["rescale_step_size_before"],
                "ds_after": b["rescale_step_size_after"],
                "e_pred": b["rescale_predictor_error"],
            })
            continue
        if a["state_scale"] == b["state_scale"]:
            continue
        t_before, t_after = a["t_lam_own"], b["t_lam_pred"]
        if t_after == 0.0:
            continue
        rescale_norm = t_before / t_after
        ds_before, ds_after = a["ds"], b["ds"]
        if ds_before == 0.0 or rescale_norm == 0.0:
            continue
        e_pred = abs(ds_after / (ds_before * rescale_norm) - 1.0)
        events.append({
            "row_index": a["index"],
            "mu": a["mu"],
            "state_scale_before": a["state_scale"],
            "state_scale_after": b["state_scale"],
            "scale_ratio": b["state_scale"] / a["state_scale"],
            "t_lam_before_rescale": t_before,
            "t_lam_after_rescale": t_after,
            "rescale_norm": rescale_norm,
            "ds_before": ds_before,
            "ds_after": ds_after,
            "e_pred": e_pred,
        })
    return events


def print_predictor_invariance(events: list[dict[str, Any]]) -> None:
    if not events:
        print("no rescale events detected in this CSV")
        return
    cols = ["row_index", "mu", "scale_ratio", "t_lam_before_rescale",
            "t_lam_after_rescale", "rescale_norm", "e_pred"]
    header = " | ".join(f"{c:>20}" for c in cols)
    print(header)
    print("-" * len(header))
    for e in events:
        print(" | ".join(f"{e[c]:20.6g}" for c in cols))
    e_preds = np.array([e["e_pred"] for e in events])
    print(f"\n{len(events)} rescale events. e_pred: "
          f"median={np.median(e_preds):.4g} min={e_preds.min():.4g} "
          f"max={e_preds.max():.4g}")
    print(f"e_pred < 0.1 (invariant roughly held): "
          f"{int(np.sum(e_preds < 0.1))}/{len(events)}")


def print_failure_diagnostics(branch, telemetry: list[dict[str, Any]]) -> None:
    """Print the compact continuation diagnostics needed for a failed run."""
    info = branch.info
    mus = [float(row["mu"]) for row in telemetry]
    dss = [float(row["step_size"]) for row in telemetry]
    events = [
        row for row in telemetry if float(row.get("rescale_event", 0.0)) > 0.5
    ]
    errors = [float(row["rescale_predictor_error"]) for row in events]
    print("\nFailure diagnostics:")
    print(f"terminal_reason={info.get('terminal_reason')!r}")
    print(f"last_mu={mus[-1] if mus else None}")
    print(f"max_mu={max(mus) if mus else None}")
    print(
        f"accepted_steps={len(telemetry)} "
        f"rejected_steps={info.get('rejected_steps')}",
    )
    print(f"final_ds={dss[-1] if dss else None} min_ds={min(dss) if dss else None}")
    print(f"rescale_events={len(events)}")
    print(f"max_predictor_error={max(errors) if errors else 0.0}")
    print(f"final_hb_residual={telemetry[-1].get('hb_residual_rel') if telemetry else None}")
    print(f"final_palc_residual={telemetry[-1].get('palc_residual') if telemetry else None}")
    print("last_telemetry_rows=")
    for row in telemetry[-10:]:
        print({
            key: row.get(key) for key in (
                "step", "mu", "step_size", "t_lam_pred", "t_lam_own",
                "state_scale", "rescale_event", "rescale_predictor_error",
                "hb_residual_rel", "palc_residual",
            )
        })


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = dict(r)
            row["index"] = int(row["index"])
            for k in row:
                if k != "index":
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def reanalyze(csv_path: Path) -> int:
    """Recompute verdict/table/plots from an already-saved steps CSV.

    No pump solves -- for re-deriving a corrected classification (e.g. the
    fold_idx fix above) without re-running the expensive traversal stages.
    """
    rows = _load_csv_rows(csv_path)
    verdict = classify_traversal(rows)
    for k, v in verdict.items():
        print(f"VERDICT {k}={v}")
    rep = representative_table(rows, verdict)
    print("\nRepresentative points:")
    print_table(rep)
    make_plots(rows, verdict)
    print(f"\nSTATUS {verdict['status']}")
    return 0 if verdict["status"] != "NOT_TRAVERSED" else 1


def check_predictor_invariance(csv_path: Path) -> int:
    rows = _load_csv_rows(csv_path)
    events = predictor_invariance(rows)
    print_predictor_invariance(events)
    return 0


def run(*, rescale_every: int, out_dir: Path, tag: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    engine, solver, problem, basis, ref_injected = build_problem()
    print(f"[{tag}] ref_injected_current_a={ref_injected:.6e}", flush=True)

    anchor = run_warmup(engine, solver, problem, ref_injected)
    branch, telemetry = run_traversal(
        solver, problem, ref_injected, anchor, rescale_every=rescale_every,
    )

    if not telemetry:
        print(f"[{tag}] STATUS NOT_TRAVERSED (no accepted steps in traversal stage)")
        return {"tag": tag, "status": "NOT_TRAVERSED", "n_rows": 0}

    if not branch.info.get("reached_target"):
        print_failure_diagnostics(branch, telemetry)

    rows = merge_rows(branch, telemetry, problem)
    csv_path = out_dir / "g2_fold_traversal_steps.csv"
    write_csv(csv_path, rows)

    verdict = classify_traversal(rows)
    for k, v in verdict.items():
        print(f"[{tag}] VERDICT {k}={v}")

    rep = representative_table(rows, verdict)
    print(f"\n[{tag}] Representative points:")
    print_table(rep)

    global OUT_DIR
    prev_out_dir = OUT_DIR
    OUT_DIR = out_dir
    make_plots(rows, verdict)
    OUT_DIR = prev_out_dir

    candidates = find_fold_candidates(branch)
    cand_mus = [0.5 * (c.mu_before + c.mu_after) for c in candidates]
    print(f"\n[{tag}] find_fold_candidates on this traversal: {cand_mus}")

    fold_dbm = _current_to_dbm(verdict["mu_at_fold"] * ref_injected, FREQ_GHZ, engine.args)
    print(f"[{tag}] fold physical power (dBm, on-chip pump convention): {fold_dbm}")

    print(f"\n[{tag}] STATUS {verdict['status']}")

    events = predictor_invariance(rows)
    print(f"\n[{tag}] Predictor invariance check ({len(events)} rescale events):")
    print_predictor_invariance(events)

    return {
        "tag": tag, "status": verdict["status"], "n_rows": len(rows),
        "verdict": verdict, "csv_path": str(csv_path),
        "n_rescale_events": len(events),
        "accepted_steps": branch.info.get("steps"),
        "rejected_steps": [r["rejected_steps"] for r in rows][-1] if rows else None,
        "terminal_reason": branch.info.get("terminal_reason"),
    }


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--reanalyze-csv":
        return reanalyze(Path(argv[1]))
    if argv and argv[0] == "--check-predictor-invariance":
        return check_predictor_invariance(Path(argv[1]))

    rescale_every = RESCALE_EVERY
    out_dir = OUT_DIR
    tag = "adaptive_metric"
    if argv and argv[0] == "--rescale-every":
        rescale_every = int(argv[1])
        out_dir = ROOT / "outputs" / (
            "fold_plan_milestone_g2_frozen_metric" if rescale_every == 0
            else f"fold_plan_milestone_g2_rescale{rescale_every}"
        )
        tag = "frozen_metric" if rescale_every == 0 else f"rescale_every_{rescale_every}"

    summary = run(rescale_every=rescale_every, out_dir=out_dir, tag=tag)
    return 0 if summary["status"] != "NOT_TRAVERSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
