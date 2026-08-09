"""Milestone D.5: resolve the branch topology of the mu~0.525 feature found
during the pre-Milestone-E A-D validation campaign (fold_plan.md; memory
fold-plan-ad-validation-narrow-feature).

Question: is the mu~0.525 event at 7.9 GHz a genuine local fold pair
(t_mu: + -> - -> +, a small S-bend the branch passes through and recovers
from), a numerical/tangent pathology, or a basis-truncation artifact -- before
Milestone E's topology-aware "blocking fold" semantics are built on top of it.

Does NOT modify production code or solver.py. Reuses the exact production
solve_arclength/trace_arclength_from_two_points/_refine_fold machinery via
the existing validate_fold_plan_ad.py helpers.

Local microscope only: restarts from an already-converged anchor near
mu=0.49 (well below the feature) and stops at mu_max=0.60 (well above it,
short of the main mu~0.674 fold) -- this is deliberately much cheaper than
re-running the full A-D campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from scripts.validate_fold_plan_ad import (  # noqa: E402
    _build_ref_problem,
    _current_to_dbm,
    _dbm_to_current,
    _engine_args,
    _unit_tangent_at,
    trace_branch_rich,
    write_csv,
)
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.singularity import (  # noqa: E402
    bordered_conditioning,
    jacobian_det_signature,
    jacobian_min_eigenvalue_with_estimator,
)
from twpa_solver.pump.solver import _real_dot  # noqa: E402


def _engine_args_resolution(
    circuit_dir: Path, outdir: Path, freq_ghz: float, ref_dbm: float,
    *, mode_count: int, nt: int,
) -> argparse.Namespace:
    argv = [
        "--circuit-dir", str(circuit_dir.resolve()),
        "--outdir", str(outdir / "_unused"),
        "--executor", "inprocess", "--mode", "warmstart",
        "--inproc-pump-backend", "schur_cpu_mt",
        "--inproc-preconditioner", "real_coupled_fast",
        "--n-power", "1", "--n-frequency", "1",
        "--pump-power-min-dbm", str(ref_dbm), "--pump-power-max-dbm", str(ref_dbm),
        "--pump-freq-min-ghz", str(freq_ghz), "--pump-freq-max-ghz", str(freq_ghz),
        "--pump-mode-count", str(mode_count), "--nt", str(nt),
    ]
    return run_gain_map.parse_args(argv)


def find_all_fold_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every t_mu sign change in an accepted-step row stream.

    solve_arclength's own ``info["fold_lambda"]`` only ever records the
    FIRST sign change (fold_plan.md Milestone D scope); this walks the full
    telemetry stream to find every one, which is exactly the question D.5
    is asking.
    """
    events = []
    for a, b in zip(rows, rows[1:]):
        if (a["t_mu"] >= 0) != (b["t_mu"] >= 0):
            events.append({
                "s_before": a["s"], "s_after": b["s"],
                "mu_before": a["mu"], "mu_after": b["mu"],
                "t_mu_before": a["t_mu"], "t_mu_after": b["t_mu"],
                "row_before": a, "row_after": b,
            })
    return events


def refine_event(
    solver, problem, S: np.ndarray, event: dict[str, Any], state_scale: float,
    *, newton_max: int = 15, t_tol: float = 1e-8, max_iter: int = 40,
) -> dict[str, Any]:
    """Tight secant refinement of one detected sign-change bracket.

    Reconstructs the anchor's own tangent the same way solve_arclength does
    (fresh linear solve at the anchor state), sign-aligned to the trace's own
    reported t_mu there, then reuses solver._refine_fold verbatim -- the same
    corrector/secant machinery Milestone D already validated, just invoked
    per-bracket instead of only on the first one solve_arclength finds.
    """
    a, b = event["row_before"], event["row_after"]
    Xdot_a, lam_dot_a = _unit_tangent_at(solver, problem, a["X"], S, state_scale)
    if lam_dot_a * a["t_mu"] < 0.0:
        Xdot_a, lam_dot_a = -Xdot_a, -lam_dot_a
    ds_ab = b["s"] - a["s"]

    def metric_x(u: np.ndarray, v: np.ndarray) -> float:
        return _real_dot(u, v) / (state_scale * state_scale)

    tol = max(solver.settings.newton_tol * 10.0, 1e-8)
    lam_tol = max(1e-9, abs(ds_ab) * 1e-4)
    return solver._refine_fold(
        problem, a["X"], a["mu"], Xdot_a, lam_dot_a, b["mu"], b["t_mu"], ds_ab,
        metric_x, S, newton_max, 0.0, time.perf_counter(), tol,
        t_tol=t_tol, lam_tol=lam_tol, max_iter=max_iter,
    )


def run_local_forward_trace(engine, freq_ghz: float, ref_dbm: float, *, ds: float, label: str) -> dict[str, Any]:
    """Microscope trace: adaptive step control, mu_max=0.60 (short of the
    main ~0.674 fold), refine_fold=True (covers the first event; every
    event is separately refined by refine_event below using the full row
    stream find_all_fold_events returns)."""
    return trace_branch_rich(
        engine, freq_ghz, ref_dbm, ds=ds, step_control="adaptive", mu_max=0.60,
        max_steps=1200, max_steps_after_fold=300, rescale_every=5, label=label,
    )


def run_backward_trace(engine, freq_ghz: float, ref_dbm: float, forward_rows: list[dict[str, Any]], *, ds: float) -> dict[str, Any]:
    """Seed trace_arclength_from_two_points with two adjacent forward-trace
    points in DECREASING-mu order so its secant tangent points backward, then
    let it walk on past them back down through the feature -- an independent
    check that doesn't reuse solve_arclength's ascending-only initial tangent.
    """
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    problem, _basis, ref_injected = _build_ref_problem(engine, freq_ghz, ref_dbm)

    # Seed pair: last two rows with mu comfortably above the feature (close
    # to mu_max=0.60), reversed so X0 has the LARGER mu.
    seed_b, seed_a = forward_rows[-1], forward_rows[-2]
    assert seed_b["mu"] > seed_a["mu"], "expected ascending forward rows near mu_max"

    points, info = solver.trace_arclength_from_two_points(
        problem, seed_b["X"], seed_b["mu"], seed_a["X"], seed_a["mu"],
        ds=ds, max_steps=600, newton_max=15,
    )
    rows = [{"index": i, "mu": float(lam), "X": X} for i, (X, lam) in enumerate(points)]
    # t_mu per interior point via the same secant-tangent sign the function
    # itself tracks internally (info["fold_lambdas"]); reconstruct a
    # comparable per-row t_mu sign purely from consecutive mu differences,
    # since points/points direction already encodes ascending vs descending.
    for i in range(1, len(rows) - 1):
        dmu_prev = rows[i]["mu"] - rows[i - 1]["mu"]
        rows[i]["local_slope_sign"] = 1 if dmu_prev >= 0 else -1
    print(f"[D5 backward] {freq_ghz:.2f} GHz ds={ds}: {len(points)} points, "
          f"fold_lambdas={info['fold_lambdas']}, terminal={info['terminal_reason']}", flush=True)
    return {"rows": rows, "info": info, "problem": problem, "solver": solver, "ref_injected": ref_injected}


def run_multistability_probe(
    engine, freq_ghz: float, ref_dbm: float, forward_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Test 6: fixed-mu Newton from seeds on each detected segment.

    If the feature is a genuine local S-bend, one horizontal mu=mu_star line
    inside the fold-pair's overlap should intersect three distinct branch
    points; Newton from a nearby guess on each segment should converge to
    three (materially) different roots, all satisfying R(X, mu_star) ~ 0.
    """
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    problem, _basis, _ref_injected = _build_ref_problem(engine, freq_ghz, ref_dbm)

    segments: dict[int, list[dict[str, Any]]] = {}
    for r in forward_rows:
        segments.setdefault(r["segment_id"], []).append(r)
    if len(segments) < 3:
        return {"skipped": True, "reason": f"only {len(segments)} segment(s) found, need >=3 for a fold pair"}

    seg_ids = sorted(segments)[:3]
    mu_lo = max(min(r["mu"] for r in segments[seg_ids[1]]), 1e-12)
    mu_hi = min(max(r["mu"] for r in segments[seg_ids[0]]), max(r["mu"] for r in segments[seg_ids[2]]))
    if mu_hi <= mu_lo:
        return {"skipped": True, "reason": f"no mu overlap found (lo={mu_lo}, hi={mu_hi})"}
    mu_star = 0.5 * (mu_lo + mu_hi)

    roots: dict[int, dict[str, Any]] = {}
    for seg_id in seg_ids:
        seg_rows = segments[seg_id]
        nearest = min(seg_rows, key=lambda r: abs(r["mu"] - mu_star))
        X_star, report = solver.solve_one(problem, nearest["X"], mu_star)
        roots[seg_id] = {
            "segment_id": seg_id, "seed_mu": nearest["mu"], "converged": report.converged,
            "coeff_rel": report.coeff_rel, "X": X_star, "norm": float(np.linalg.norm(X_star)),
        }
        print(f"[D5 multistability] mu_star={mu_star:.6f} seg={seg_id} seed_mu={nearest['mu']:.6f} "
              f"converged={report.converged} coeff_rel={report.coeff_rel} norm={roots[seg_id]['norm']:.6e}", flush=True)

    pairwise = []
    for i, si in enumerate(seg_ids):
        for sj in seg_ids[i + 1:]:
            Xi, Xj = roots[si]["X"], roots[sj]["X"]
            d = float(np.linalg.norm(Xi - Xj)) / max(float(np.linalg.norm(Xi)), 1e-300)
            pairwise.append({"seg_i": si, "seg_j": sj, "relative_distance": d})
            print(f"[D5 multistability] d({si},{sj}) = {d:.6e}", flush=True)

    return {
        "skipped": False, "mu_star": mu_star, "mu_lo": mu_lo, "mu_hi": mu_hi,
        "roots": {k: {kk: vv for kk, vv in v.items() if kk != "X"} for k, v in roots.items()},
        "pairwise": pairwise,
    }


def run_local_singularity_diagnostics(solver, problem, S: np.ndarray, X: np.ndarray, state_scale: float, label: str) -> dict[str, Any]:
    Xdot, lam_dot = _unit_tangent_at(solver, problem, X, S, state_scale)
    min_eig, estimator = jacobian_min_eigenvalue_with_estimator(problem, X, 1.0, iters=20)
    cond = bordered_conditioning(problem, X, 1.0, Xdot, lam_dot, state_scale, iters=20)
    det_sign, log_abs_det = jacobian_det_signature(problem, X, 1.0)
    print(f"[D5 singularity] {label}: min_eig={min_eig:.4e} bordered_cond={cond:.4e} det_sign={det_sign}", flush=True)
    return {
        "label": label, "min_eigenvalue": min_eig, "min_eigenvalue_estimator": estimator,
        "bordered_condition": cond, "det_sign": det_sign, "log_abs_det": log_abs_det,
    }


def run_resolution_check(
    circuit_dir: Path, outdir: Path, freq_ghz: float, ref_dbm: float,
    *, mode_count: int, nt: int, ds: float, events_ref: list[dict[str, Any]],
) -> dict[str, Any]:
    """Item 9: rerun ONLY the narrow window with a richer HB basis and compare
    fold locations against the production-basis (K=10, Nt=40) events."""
    engine = run_gain_map.InProcessEngine(
        _engine_args_resolution(circuit_dir, outdir, freq_ghz, ref_dbm, mode_count=mode_count, nt=nt),
    )
    trace = run_local_forward_trace(engine, freq_ghz, ref_dbm, ds=ds, label=f"D5-resolution K={mode_count} Nt={nt}")
    events = find_all_fold_events(trace["rows"])
    solver = trace["solver"]
    S = trace["problem"].source_coeffs(1.0)
    refined = []
    for ev in events:
        state_scale = ev["row_before"]["state_scale"]
        r = refine_event(solver, trace["problem"], S, ev, state_scale)
        refined.append({"mu": r.get("lam"), "converged": r.get("converged"), "bracket_width": r.get("bracket_width")})
    print(f"[D5 resolution] K={mode_count} Nt={nt}: {len(events)} events, refined={refined}", flush=True)
    return {"mode_count": mode_count, "nt": nt, "n_events": len(events), "refined": refined}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--skip-resolution-check", action="store_true")
    args = p.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    freq_ghz, ref_dbm = 7.9, -16.0
    engine = run_gain_map.InProcessEngine(_engine_args(args.circuit_dir, outdir, freq_ghz, ref_dbm))

    summary: dict[str, Any] = {"freq_ghz": freq_ghz, "ref_dbm": ref_dbm}

    # --- Items 1-4: local high-resolution forward traces (two ds settings) ---
    fine_traces = {}
    for ds in (0.005, 0.0025):
        trace = run_local_forward_trace(engine, freq_ghz, ref_dbm, ds=ds, label=f"D5 forward ds={ds}")
        write_csv(outdir / f"d5_forward_ds{ds}.csv", trace["rows"])
        events = find_all_fold_events(trace["rows"])
        solver = trace["solver"]
        S = trace["problem"].source_coeffs(1.0)
        refined_events = []
        for ev in events:
            state_scale = ev["row_before"]["state_scale"]
            r = refine_event(solver, trace["problem"], S, ev, state_scale)
            refined_events.append({
                "s_before": ev["s_before"], "s_after": ev["s_after"],
                "mu_before": ev["mu_before"], "mu_after": ev["mu_after"],
                "refined_mu": r.get("lam"), "refined_lam_dot": r.get("lam_dot"),
                "converged": r.get("converged"), "bracket_width": r.get("bracket_width"),
                "iterations": r.get("iterations"),
            })
            print(f"[D5 forward ds={ds}] event s=[{ev['s_before']:.4f},{ev['s_after']:.4f}] "
                  f"mu=[{ev['mu_before']:.6f},{ev['mu_after']:.6f}] -> refined mu={r.get('lam')} "
                  f"converged={r.get('converged')} width={r.get('bracket_width')}", flush=True)

        # Item 4: state continuity dX/ds between successive accepted points.
        rows = trace["rows"]
        continuity = []
        for a, b in zip(rows, rows[1:]):
            ds_ab = b["s"] - a["s"]
            if abs(ds_ab) < 1e-300:
                continue
            dX = float(np.linalg.norm(b["X"] - a["X"])) / abs(ds_ab)
            continuity.append(dX)

        fine_traces[ds] = {
            "trace": trace, "events": events, "refined_events": refined_events,
            "continuity_dX_ds": continuity,
            "continuity_max": max(continuity) if continuity else None,
            "continuity_mean": (sum(continuity) / len(continuity)) if continuity else None,
        }
        summary[f"ds{ds}_n_events"] = len(events)
        summary[f"ds{ds}_refined_events"] = [
            {k: v for k, v in e.items()} for e in refined_events
        ]
        summary[f"ds{ds}_continuity_max_dX_ds"] = fine_traces[ds]["continuity_max"]
        summary[f"ds{ds}_continuity_mean_dX_ds"] = fine_traces[ds]["continuity_mean"]
        summary[f"ds{ds}_n_segments"] = (rows[-1]["segment_id"] + 1) if rows else 0

    primary = fine_traces[0.0025]
    primary_rows = primary["trace"]["rows"]

    # Derived quantities (only meaningful if exactly a 2-event pair was found).
    if len(primary["refined_events"]) >= 2:
        ev1, ev2 = primary["refined_events"][0], primary["refined_events"][1]
        mu_f1, mu_f2 = ev1["refined_mu"], ev2["refined_mu"]
        s_f1 = 0.5 * (ev1["s_before"] + ev1["s_after"])
        s_f2 = 0.5 * (ev2["s_before"] + ev2["s_after"])
        mid_rows = [r for r in primary_rows if s_f1 <= r["s"] <= s_f2]
        mu_retreat_min = min((r["mu"] for r in mid_rows), default=None)
        idx_f1 = min(range(len(primary_rows)), key=lambda i: abs(primary_rows[i]["s"] - s_f1))
        idx_f2 = min(range(len(primary_rows)), key=lambda i: abs(primary_rows[i]["s"] - s_f2))
        dX_pair = float(np.linalg.norm(primary_rows[idx_f2]["X"] - primary_rows[idx_f1]["X"]))
        summary["delta_mu_pair"] = abs(mu_f2 - mu_f1) if (mu_f1 is not None and mu_f2 is not None) else None
        summary["delta_s_pair"] = abs(s_f2 - s_f1)
        summary["delta_mu_retreat"] = (mu_f1 - mu_retreat_min) if (mu_f1 is not None and mu_retreat_min is not None) else None
        summary["delta_X_pair"] = dX_pair
    else:
        summary["delta_mu_pair"] = None
        summary["delta_s_pair"] = None
        summary["delta_mu_retreat"] = None
        summary["delta_X_pair"] = None

    # --- Item 5: backward trace ---
    backward = run_backward_trace(engine, freq_ghz, ref_dbm, primary_rows, ds=0.0025)
    write_csv(outdir / "d5_backward.csv", [{"index": r["index"], "mu": r["mu"]} for r in backward["rows"]])
    summary["backward_fold_lambdas"] = backward["info"]["fold_lambdas"]
    summary["backward_terminal_reason"] = backward["info"]["terminal_reason"]
    summary["backward_n_points"] = len(backward["rows"])

    # --- Item 6: fixed-mu multistability probe ---
    multi = run_multistability_probe(engine, freq_ghz, ref_dbm, primary_rows)
    summary["multistability"] = multi

    # --- Item 7: singularity diagnostics at each refined local fold ---
    solver = primary["trace"]["solver"]
    problem = primary["trace"]["problem"]
    S = problem.source_coeffs(1.0)
    d5_singularity = []
    for i, ev in enumerate(primary["refined_events"]):
        X_at = None
        for a, b in zip(primary_rows, primary_rows[1:]):
            if a["s"] <= 0.5 * (ev["s_before"] + ev["s_after"]) <= b["s"]:
                X_at = a["X"]
                state_scale = a["state_scale"]
                break
        if X_at is None:
            continue
        d5_singularity.append(
            run_local_singularity_diagnostics(solver, problem, S, X_at, state_scale, label=f"local_fold_{i}"),
        )
    write_csv(outdir / "d5_singularity.csv", d5_singularity)
    summary["singularity"] = d5_singularity

    # --- Item 9: resolution check (richer HB basis, narrow window only) ---
    if not args.skip_resolution_check:
        resolution = run_resolution_check(
            args.circuit_dir, outdir, freq_ghz, ref_dbm,
            mode_count=12, nt=48, ds=0.005, events_ref=primary["refined_events"],
        )
        summary["resolution_check"] = resolution
        prod_mus = sorted(e["refined_mu"] for e in fine_traces[0.005]["refined_events"] if e["refined_mu"] is not None)
        rich_mus = sorted(m["mu"] for m in resolution["refined"] if m["mu"] is not None)
        summary["resolution_check_mu_shift"] = (
            [abs(a - b) for a, b in zip(prod_mus, rich_mus)] if len(prod_mus) == len(rich_mus) else None
        )

    with (outdir / "d5_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"wrote {outdir / 'd5_summary.json'}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
