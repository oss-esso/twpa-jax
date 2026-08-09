"""Pre-Milestone-E validation campaign for fold_plan.md Milestones A-D.

Runs the real production 2c circuit (not the toy 1-node unit-test fixture)
through the new mu-parameterized, adaptively-stepped, fold-refining
continuation and compares it against the existing production route, on the
two frequencies (7.0/7.9 GHz) that originally exposed the two failure modes
this work targets. Produces CSVs (branch trace, step telemetry, singularity
samples) and a JSON summary with the go/no-go table; a separate step turns
those into plots.

Deliberately does NOT use Milestone E's future "stop at the fold" semantics
-- every trace here continues past the first fold on purpose, per the user's
explicit validation-campaign request, so the branch geometry itself can be
inspected.

Standalone diagnostic, same engine-loading pattern as
``scan_branch_singularity.py``. Not wired into the production map.
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
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.singularity import (  # noqa: E402
    bordered_conditioning,
    jacobian_det_signature,
    jacobian_min_eigenvalue_with_estimator,
)
from twpa_solver.pump.solver import _real_dot  # noqa: E402


def _engine_args(circuit_dir: Path, outdir: Path, freq_ghz: float, ref_dbm: float) -> argparse.Namespace:
    argv = [
        "--circuit-dir", str(circuit_dir.resolve()),
        "--outdir", str(outdir / "_unused"),
        "--executor", "inprocess", "--mode", "warmstart",
        "--inproc-pump-backend", "schur_cpu_mt",
        "--inproc-preconditioner", "real_coupled_fast",
        "--n-power", "1", "--n-frequency", "1",
        "--pump-power-min-dbm", str(ref_dbm),
        "--pump-power-max-dbm", str(ref_dbm),
        "--pump-freq-min-ghz", str(freq_ghz),
        "--pump-freq-max-ghz", str(freq_ghz),
    ]
    return run_gain_map.parse_args(argv)


def _dbm_to_current(dbm: float, freq_ghz: float, args: argparse.Namespace) -> float:
    att = run_gain_map.attenuation_db_for(freq_ghz, args)
    phys = run_gain_map.dbm_to_peak_current_a(
        dbm, attenuation_db=att, z0_ohm=args.z0_ohm, convention=args.power_convention,
    )
    return phys * args.pump_current_jc_scale


def _current_to_dbm(current_a: float, freq_ghz: float, args: argparse.Namespace) -> float | None:
    if current_a is None or current_a <= 0.0:
        return None
    scaled = current_a / args.pump_current_jc_scale
    return run_gain_map.peak_current_to_power_dbm(scaled, freq_ghz, args)


def _build_ref_problem(engine, freq_ghz: float, ref_dbm: float):
    ref_injected = _dbm_to_current(ref_dbm, freq_ghz, engine.args)
    full_problem, basis, _omega = engine._build_problem(freq_ghz, ref_injected)
    problem = engine._make_solve_problem(full_problem, freq_ghz)
    return problem, basis, ref_injected


def _unit_tangent_at(solver, problem, X: np.ndarray, S: np.ndarray, state_scale: float):
    """Reconstruct a metric-normalized tangent at an arbitrary accepted state.

    Mirrors solve_arclength's own initial-tangent construction so the
    resulting (Xdot, lam_dot) is on the same footing bordered_conditioning
    expects, without needing to have been the point where the run actually
    computed a tangent (e.g. the D5 before/at/after sampling points).
    """
    Xdot = solver._solve_linear(problem, X, S)
    lam_dot = 1.0
    nrm = math.sqrt(_real_dot(Xdot, Xdot) / (state_scale ** 2) + lam_dot ** 2)
    return Xdot / nrm, lam_dot / nrm


def _observable(problem, X: np.ndarray) -> dict[str, float]:
    """Cheap physical observables from a pump state, for Test 0/A/B2."""
    return {
        "max_abs_flux": float(np.max(np.abs(X))),
        "l2_norm": float(np.linalg.norm(X)),
        "fundamental_l2": float(np.linalg.norm(X[0, :])) if X.shape[0] > 0 else 0.0,
    }


def _mode_table(basis, X: np.ndarray) -> dict[str, float]:
    return {f"h{int(k)}": float(np.linalg.norm(X[i, :])) for i, k in enumerate(basis.k)}


def run_test0_and_a(engine, freq_ghz: float, ref_dbm: float, targets_dbm: list[float]) -> list[dict[str, Any]]:
    """Test 0 (frozen old-route reference) + Test A (mu-route physics match)."""
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    ref_problem, basis, ref_injected = _build_ref_problem(engine, freq_ghz, ref_dbm)
    rows = []
    for target_dbm in targets_dbm:
        injected = _dbm_to_current(target_dbm, freq_ghz, engine.args)

        # Old/production route: build a problem whose OWN lambda=1 is this
        # target, cold-solved via solve_adaptive_continuation -- the actual
        # production cold-cell path (run_gain_map.py::InProcessEngine.solve_point,
        # default continuation branch), NOT a raw single-shot solve_one(zeros,
        # 1.0) at full drive (that is a much harder ask Newton alone is not
        # expected to survive, continuation exists precisely because of this).
        full_old, _b, _o = engine._build_problem(freq_ghz, injected)
        problem_old = engine._make_solve_problem(full_old, freq_ghz)
        X_old, reports_old, _trace_old = solver.solve_adaptive_continuation(
            problem_old, problem_old.zeros(),
            initial_step=engine.args.adaptive_initial_step,
            min_step=engine.args.adaptive_min_step, growth=1.5, shrink=0.5,
            fallback_fixed_steps=engine.args.inproc_fallback_fixed_steps,
            max_wall_s=engine.args.inproc_solve_deadline_s,
        )
        rep_old = reports_old[-1]

        # New mu route: fixed-reference problem, target_mu = injected/ref_injected.
        # PALC discovers the branch geometry; the raw arclength endpoint is
        # then polished by one exact fixed-mu Newton solve (fold_plan.md
        # Section 19 / Milestone B's resolve_target_on_branch), which is what
        # a caller is actually meant to consume as the answer -- not the loose
        # interpolated PALC endpoint on its own.
        mu_target = injected / ref_injected
        X_arc, mu_reached, info = solver.solve_arclength_mu(
            ref_problem, ref_problem.zeros(), 0.0, i_ref=ref_injected,
            target_mu=mu_target, ds=0.02, max_steps=400, newton_max=15,
        )
        X_new, rep_new = solver.solve_one(ref_problem, X_arc, mu_target)
        rep_rel = {"coeff_rel": rep_new.coeff_rel}

        # Round-trip check: I_p = mu * i_ref, converted back to dBm.
        current_roundtrip = mu_reached * ref_injected
        dbm_roundtrip = _current_to_dbm(current_roundtrip, freq_ghz, engine.args)

        e_x = float(np.linalg.norm(X_new - X_old)) / max(float(np.linalg.norm(X_old)), 1e-300)
        obs_old, obs_new = _observable(problem_old, X_old), _observable(ref_problem, X_new)

        rows.append({
            "freq_ghz": freq_ghz, "target_dbm": target_dbm,
            "physical_current_a": injected, "mu_target": mu_target,
            "old_converged": rep_old.converged, "old_coeff_rel": rep_old.coeff_rel,
            "new_palc_reached_target": info["reached_target"],
            "new_converged": rep_new.converged, "new_coeff_rel": rep_rel.get("coeff_rel"),
            "old_norm": obs_old["l2_norm"], "new_norm": obs_new["l2_norm"],
            "relative_state_error": e_x,
            "old_fundamental": obs_old["fundamental_l2"], "new_fundamental": obs_new["fundamental_l2"],
            "current_roundtrip_a": current_roundtrip, "dbm_roundtrip": dbm_roundtrip,
            "dbm_roundtrip_error": (dbm_roundtrip - target_dbm) if dbm_roundtrip is not None else None,
            **{f"old_{k}": v for k, v in _mode_table(basis, X_old).items()},
            **{f"new_{k}": v for k, v in _mode_table(basis, X_new).items()},
        })
        print(f"[test0/A] {freq_ghz:.2f} GHz {target_dbm} dBm: e_X={e_x:.3e} "
              f"old_conv={rep_old.converged} new_reached={info['reached_target']} "
              f"dbm_roundtrip_err={rows[-1]['dbm_roundtrip_error']}", flush=True)
    return rows


def trace_branch_rich(
    engine, freq_ghz: float, ref_dbm: float, *, ds: float, step_control: str,
    mu_max: float, max_steps: int, max_steps_after_fold: int | None,
    rescale_every: int = 5, label: str = "",
) -> dict[str, Any]:
    """One solve_arclength trace with combined on_step + step_telemetry rows."""
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    problem, basis, ref_injected = _build_ref_problem(engine, freq_ghz, ref_dbm)
    S = problem.source_coeffs(1.0)

    rows: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}
    s_cum = [0.0]

    def _on_step(Xc, lamc, step_size, Xdot, lam_dot, state_scale) -> None:
        pending.clear()
        pending.update({"X": np.array(Xc, copy=True), "mu": lamc, "t_mu": lam_dot,
                         "state_scale_pre": state_scale})

    def _step_telemetry(rec: dict[str, Any]) -> None:
        s_cum[0] += rec["step_size"]
        row = dict(pending)
        row.update(rec)
        row["s"] = s_cum[0]
        row["current_a"] = row["mu"] * ref_injected
        row["dbm"] = _current_to_dbm(row["current_a"], freq_ghz, engine.args)
        rows.append(row)

    t0 = time.perf_counter()
    X_final, mu_final, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=ds, max_steps=max_steps,
        target_lam=mu_max, newton_max=15, on_step=_on_step,
        rescale_every=rescale_every, max_steps_after_fold=max_steps_after_fold,
        step_control=step_control, refine_fold=True,
        step_telemetry=_step_telemetry,
    )
    runtime_s = time.perf_counter() - t0

    # Segment ids from the sign of t_mu (Test B4) -- not a stored field
    # anywhere yet (Section 20's full segment-ID storage is out of Milestone
    # B's actual scope), computed here purely for this validation's analysis.
    seg = 0
    prev_sign = None
    for row in rows:
        sign = 1 if row["t_mu"] >= 0 else -1
        if prev_sign is not None and sign != prev_sign:
            seg += 1
        row["segment_id"] = seg
        prev_sign = sign

    fold_refined = info.get("fold_refined")
    print(f"[{label}] {freq_ghz:.2f} GHz ds={ds} step_control={step_control}: "
          f"{len(rows)} accepted, rejected={info.get('rejected_steps')}, "
          f"fold_lambda={info.get('fold_lambda')}, "
          f"fold_refined={'None' if fold_refined is None else fold_refined.get('converged')}, "
          f"runtime={runtime_s:.1f}s", flush=True)

    return {
        "rows": rows, "info": info, "problem": problem, "basis": basis,
        "ref_injected": ref_injected, "runtime_s": runtime_s, "ds": ds,
        "step_control": step_control, "freq_ghz": freq_ghz, "ref_dbm": ref_dbm,
        "solver": solver,
    }


def run_test_b3(trace: dict[str, Any], targets_dbm: list[float], old_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bracket + interpolate + exact Newton solve, vs the Test 0 old-route rows."""
    rows = trace["rows"]
    problem = trace["problem"]
    solver = trace["solver"]
    out = []
    old_by_dbm = {r["target_dbm"]: r for r in old_rows if r["freq_ghz"] == trace["freq_ghz"]}
    for target_dbm in targets_dbm:
        if target_dbm not in old_by_dbm:
            continue
        mu_target = old_by_dbm[target_dbm]["mu_target"]
        bracket = None
        for a, b in zip(rows, rows[1:]):
            if (a["mu"] - mu_target) * (b["mu"] - mu_target) <= 0.0 and a["mu"] != b["mu"]:
                bracket = (a, b)
                break
        if bracket is None:
            out.append({"target_dbm": target_dbm, "bracketed": False})
            continue
        a, b = bracket
        alpha = (mu_target - a["mu"]) / (b["mu"] - a["mu"])
        X_guess = (1.0 - alpha) * a["X"] + alpha * b["X"]
        X_resolved, report = solver.solve_one(problem, X_guess, mu_target)
        old_norm = old_by_dbm[target_dbm]["old_norm"]
        e_x = float(np.linalg.norm(X_resolved))
        rel_err = abs(e_x - old_norm) / max(old_norm, 1e-300)
        out.append({
            "target_dbm": target_dbm, "bracketed": True, "converged": report.converged,
            "coeff_rel": report.coeff_rel, "resolved_norm": e_x, "old_norm": old_norm,
            "relative_norm_error": rel_err,
        })
        print(f"[test B3] {trace['freq_ghz']:.2f} GHz {target_dbm} dBm: "
              f"bracketed converged={report.converged} rel_norm_err={rel_err:.3e}", flush=True)
    return out


def run_test_d5(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Jacobian/bordered-system singularity diagnostics before/at/after the fold."""
    rows = trace["rows"]
    info = trace["info"]
    problem = trace["problem"]
    solver = trace["solver"]
    S = problem.source_coeffs(1.0)
    if not rows:
        return []
    # Called on a plain solve_arclength trace (not the _mu wrapper), so
    # fold_refined keys are the lambda-space names ("lam"), not "mu" --
    # numerically identical here since these traces are built with
    # pump_current_a == i_ref (k=1), so lambda IS mu.
    fold_refined = info.get("fold_refined")
    X_at = fold_refined.get("X") if fold_refined is not None else None
    fold_mu = fold_refined.get("lam") if (fold_refined is not None and X_at is not None) else None
    if fold_mu is None:
        fold_mu = info.get("fold_lambda")
    if fold_mu is None:
        return []
    fold_idx = min(range(len(rows)), key=lambda i: abs(rows[i]["mu"] - fold_mu))
    state_scale_at = rows[fold_idx]["state_scale"]
    if X_at is None:
        X_at = rows[fold_idx]["X"]

    before_idx = max(0, fold_idx - 10)
    after_idx = min(len(rows) - 1, fold_idx + 10)
    labels = ["before", "at_fold", "after"]
    idxs_X_scale = [
        (rows[before_idx]["X"], rows[before_idx]["state_scale"]),
        (X_at, state_scale_at),
        (rows[after_idx]["X"], rows[after_idx]["state_scale"]),
    ]
    out = []
    for label, (X, state_scale) in zip(labels, idxs_X_scale):
        Xdot, lam_dot = _unit_tangent_at(solver, problem, X, S, state_scale)
        min_eig, estimator = jacobian_min_eigenvalue_with_estimator(problem, X, 1.0, iters=20)
        cond = bordered_conditioning(problem, X, 1.0, Xdot, lam_dot, state_scale, iters=20)
        det_sign, log_abs_det = jacobian_det_signature(problem, X, 1.0)
        out.append({
            "label": label, "min_eigenvalue": min_eig, "min_eigenvalue_estimator": estimator,
            "bordered_condition": cond, "det_sign": det_sign, "log_abs_det": log_abs_det,
        })
        print(f"[test D5] {trace['freq_ghz']:.2f} GHz {label}: min_eig={min_eig:.3e} "
              f"bordered_cond={cond:.3e} det_sign={det_sign}", flush=True)
    return out


def run_test_d8(engine, freq_ghz: float, ref_dbm: float) -> dict[str, Any]:
    trace = trace_branch_rich(
        engine, freq_ghz, ref_dbm, ds=0.02, step_control="adaptive", mu_max=0.3,
        max_steps=100, max_steps_after_fold=None, label="test D8 boring",
    )
    n_fold_candidates = sum(1 for r in trace["rows"][1:] if r["segment_id"] != 0)
    return {
        "freq_ghz": freq_ghz, "mu_max": 0.3, "fold_lambda": trace["info"].get("fold_lambda"),
        "false_fold_count": n_fold_candidates, "accepted_points": len(trace["rows"]),
    }


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    """Copy of a solve_arclength ``info`` dict with the large ``X`` array
    (inside ``fold_refined``, if present) stripped for JSON output."""
    out = dict(info)
    fr = out.get("fold_refined")
    if isinstance(fr, dict):
        fr = dict(fr)
        if fr.get("X") is not None:
            fr["X"] = f"<array shape={np.shape(fr['X'])}, omitted>"
        out["fold_refined"] = fr
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    plain_keys = [k for k in keys if k != "X"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=plain_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    # --- 7.9 GHz: fold-exercising frequency ---
    freq79 = 7.9
    ref79 = -16.0
    engine79 = run_gain_map.InProcessEngine(_engine_args(args.circuit_dir, outdir, freq79, ref79))
    targets79 = [-22.0, -21.0, -20.5]

    t0_rows_79 = run_test0_and_a(engine79, freq79, ref79, targets79)
    write_csv(outdir / "test0_a_79ghz.csv", t0_rows_79)

    trace_legacy = trace_branch_rich(
        engine79, freq79, ref79, ds=0.02, step_control="legacy", mu_max=1.0,
        max_steps=250, max_steps_after_fold=100, label="C-fixed 7.9GHz",
    )
    write_csv(outdir / "branch_79ghz_legacy_ds02.csv", trace_legacy["rows"])

    trace_adaptive = trace_branch_rich(
        engine79, freq79, ref79, ds=0.02, step_control="adaptive", mu_max=1.0,
        max_steps=250, max_steps_after_fold=100, label="C-adaptive 7.9GHz ds=0.02",
    )
    write_csv(outdir / "branch_79ghz_adaptive_ds02.csv", trace_adaptive["rows"])

    trace_adaptive_ds01 = trace_branch_rich(
        engine79, freq79, ref79, ds=0.01, step_control="adaptive", mu_max=1.0,
        max_steps=400, max_steps_after_fold=150, label="D4 7.9GHz ds=0.01",
    )
    write_csv(outdir / "branch_79ghz_adaptive_ds01.csv", trace_adaptive_ds01["rows"])

    b3_rows = run_test_b3(trace_adaptive, targets79, t0_rows_79)
    write_csv(outdir / "test_b3_79ghz.csv", b3_rows)

    d5_rows = run_test_d5(trace_adaptive)
    write_csv(outdir / "test_d5_79ghz.csv", d5_rows)

    d8 = run_test_d8(engine79, freq79, ref79)

    # --- 7.0 GHz: regression frequency for the metric fix ---
    freq70 = 7.0
    ref70 = -21.0
    engine70 = run_gain_map.InProcessEngine(_engine_args(args.circuit_dir, outdir, freq70, ref70))
    targets70 = [-28.0]
    t0_rows_70 = run_test0_and_a(engine70, freq70, ref70, targets70)
    write_csv(outdir / "test0_a_70ghz.csv", t0_rows_70)

    trace70 = trace_branch_rich(
        engine70, freq70, ref70, ds=0.02, step_control="adaptive", mu_max=1.0,
        max_steps=250, max_steps_after_fold=80, rescale_every=5, label="D7 7.0GHz",
    )
    write_csv(outdir / "branch_70ghz_adaptive.csv", trace70["rows"])

    minimum_step_count = sum(
        1 for r in [trace70["info"]] if r.get("terminal_reason") == "minimum_step"
    )

    summary = {
        "freq79_ref_dbm": ref79, "freq70_ref_dbm": ref70,
        "test0_a_79ghz_max_relative_state_error": max(
            (r["relative_state_error"] for r in t0_rows_79), default=None,
        ),
        "test0_a_70ghz_max_relative_state_error": max(
            (r["relative_state_error"] for r in t0_rows_70), default=None,
        ),
        "test0_a_dbm_roundtrip_max_abs_error": max(
            (abs(r["dbm_roundtrip_error"]) for r in (t0_rows_79 + t0_rows_70)
             if r["dbm_roundtrip_error"] is not None), default=None,
        ),
        "legacy_info": _json_safe_info(trace_legacy["info"]),
        "adaptive_info": _json_safe_info(trace_adaptive["info"]),
        "adaptive_ds01_info": _json_safe_info(trace_adaptive_ds01["info"]),
        "legacy_fold_refined_mu": (trace_legacy["info"].get("fold_refined") or {}).get("lam"),
        "adaptive_fold_refined_mu": (trace_adaptive["info"].get("fold_refined") or {}).get("lam"),
        "adaptive_ds01_fold_refined_mu": (trace_adaptive_ds01["info"].get("fold_refined") or {}).get("lam"),
        "legacy_runtime_s": trace_legacy["runtime_s"],
        "adaptive_runtime_s": trace_adaptive["runtime_s"],
        "b3_rows": b3_rows,
        "d5_rows": d5_rows,
        "d8": d8,
        "trace70_info": _json_safe_info(trace70["info"]),
        "trace70_terminal_reason": trace70["info"].get("terminal_reason"),
        "trace70_fold_lambda": trace70["info"].get("fold_lambda"),
        "trace70_minimum_step_recurred": trace70["info"].get("terminal_reason") == "minimum_step",
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"wrote {outdir / 'summary.json'}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
