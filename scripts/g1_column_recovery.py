"""Milestone G1 (fold_plan.md, 2026-08-08): productionize the G0 recovery
ladder, cheapest-tier-first, with a per-point route label and a column-level
failure cache so a confirmed blocking boundary is not re-discovered at
~80 s/point on every higher-power target.

Still pump-only, no fold classification, no map wiring -- a driver script
that exercises the real ``InProcessEngine`` dispatch, same as G0.

Per point, one of six routes:

    DIRECT                       -- baseline warm/cold Newton (already free)
    POWER_SUBSTEP                -- Tier 2 recovered it
    ARCLENGTH_RECOVERY           -- Tier 3 recovered it
    FREQUENCY_RECOVERY           -- Tier 4 recovered it
    FAILED_NUMERICAL             -- full ladder exhausted, no prior boundary
                                     evidence yet at this column (this point
                                     becomes the column's first boundary
                                     candidate)
    PAST_CONNECTED_BRANCH_BOUNDARY -- failed at/above an already-established
                                     boundary; only a cheap DIRECT+Tier-2
                                     probe was run, Tiers 3/4 were skipped

G0's own result (fold_plan.md, 2026-08-08) found every one of six
recoverable gaps closed by Tier 2 alone, and every unrecoverable point
reporting Tier 2's own ``step_floor`` (a step-independent stall -- G0's
docstring already defines this as "a numerical/fold boundary", not a
transient solver miss). That is the empirical basis for running Tier 2
first and unconditionally (it is cheap, ~5-10 s, and is where essentially
all of G0's useful recovery work happened), while gating the expensive
Tiers 3/4 (each up to ~60-140 s) behind "no boundary established yet for
this column".
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from scripts.g0_column_recovery import (  # noqa: E402
    TIER2_DEADLINE_S,
    TIER3_DEADLINE_S,
    TIER4_ANCHOR_DEADLINE_S,
    TIER4_ANCHOR_OFFSETS_GHZ,
    TIER4_SUBSTEP_DEADLINE_S,
    _build_args,
)
from twpa_solver.pump import hb as exp08  # noqa: E402

ROUTE_DIRECT = "DIRECT"
ROUTE_POWER_SUBSTEP = "POWER_SUBSTEP"
ROUTE_ARCLENGTH = "ARCLENGTH_RECOVERY"
ROUTE_FREQUENCY = "FREQUENCY_RECOVERY"
ROUTE_FAILED_NUMERICAL = "FAILED_NUMERICAL"
ROUTE_PAST_BOUNDARY = "PAST_CONNECTED_BRANCH_BOUNDARY"


def _try_tier2(engine, freq_ghz, last_good_X, last_good_cur, target_cur, point, pass_dir):
    t0 = time.perf_counter()
    guess, info = engine.solve_power_substep(
        freq_ghz, last_good_X, last_good_cur, target_cur, deadline_s=TIER2_DEADLINE_S,
    )
    telemetry = {
        "tier2_substeps": info["substeps"], "tier2_terminal_reason": info["terminal_reason"],
        "tier2_runtime_s": time.perf_counter() - t0,
    }
    if guess is None:
        return None, None, telemetry
    row, X = engine.solve_point(point, pass_dir, mode="warm", warm_X=guess)
    return (row, X, telemetry) if row["status"] == "PASS" else (None, None, telemetry)


def _try_tier3(engine, freq_ghz, last_good_X, last_good_cur, target_cur, point, pass_dir):
    t0 = time.perf_counter()
    guess, info = engine.solve_arclength_forward(
        freq_ghz, last_good_X, last_good_cur, target_cur, deadline_s=TIER3_DEADLINE_S,
    )
    telemetry = {
        "tier3_arclength_steps": info.get("steps"), "tier3_fold_lambda": info.get("fold_lambda"),
        "tier3_terminal_reason": info.get("terminal_reason"), "tier3_runtime_s": time.perf_counter() - t0,
    }
    if guess is None:
        return None, None, telemetry
    row, X = engine.solve_point(point, pass_dir, mode="warm", warm_X=guess)
    return (row, X, telemetry) if row["status"] == "PASS" else (None, None, telemetry)


def _try_tier4(engine, freq_ghz, last_good_X, target_cur, point, pass_dir):
    t0 = time.perf_counter()
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    anchor = None
    for df in TIER4_ANCHOR_OFFSETS_GHZ:
        nb_freq = freq_ghz + df
        prob_nb, _b, _o = engine._build_problem(nb_freq, target_cur)
        solve_nb = engine._make_solve_problem(prob_nb, nb_freq)
        Xn, repn = solver.solve_one(solve_nb, last_good_X, 1.0)
        if repn.converged:
            anchor = (nb_freq, Xn)
            break
    telemetry: dict[str, Any] = {
        "tier4_anchor_found": anchor is not None,
        "tier4_anchor_search_runtime_s": time.perf_counter() - t0,
    }
    if anchor is None:
        return None, None, telemetry
    nb_freq, Xn = anchor
    telemetry["tier4_anchor_freq_ghz"] = nb_freq
    guess, info = engine.solve_frequency_substep(
        nb_freq, freq_ghz, target_cur, Xn, deadline_s=TIER4_SUBSTEP_DEADLINE_S,
    )
    telemetry["tier4_substeps"] = info["substeps"]
    telemetry["tier4_terminal_reason"] = info["terminal_reason"]
    if guess is None:
        return None, None, telemetry
    row, X = engine.solve_point(point, pass_dir, mode="warm", warm_X=guess)
    return (row, X, telemetry) if row["status"] == "PASS" else (None, None, telemetry)


def run_column(
    circuit_dir: Path, outdir: Path, freq_ghz: float,
    n_power: int, power_min_dbm: float, power_max_dbm: float,
    *, probe_interval: int = 1,
) -> list[dict[str, Any]]:
    args = _build_args(circuit_dir, outdir, freq_ghz, n_power, power_min_dbm, power_max_dbm)
    points, _powers, _freqs = run_gain_map.build_points(args)
    points = sorted(points, key=lambda p: p.power_dbm)
    engine = run_gain_map.InProcessEngine(args)
    pass_dir = outdir / "pass"
    scale = args.pump_current_jc_scale

    last_good_X = None
    last_good_cur: float | None = None
    candidate_boundary_dbm: float | None = None
    probe_counter = 0
    rows: list[dict[str, Any]] = []

    for point in points:
        target_cur = point.current_a * scale
        mode = "warm" if last_good_X is not None else "seed"
        t0 = time.perf_counter()
        row, X = engine.solve_point(point, pass_dir, mode=mode, warm_X=last_good_X)
        row["direct_wall_s"] = time.perf_counter() - t0
        row["recovery_wall_s"] = 0.0

        if row["status"] == "PASS":
            row["route"] = ROUTE_DIRECT
        else:
            past_boundary = (
                candidate_boundary_dbm is not None and point.power_dbm >= candidate_boundary_dbm
            )
            probe_this_point = (not past_boundary) or (probe_counter % probe_interval == 0)
            t_rec0 = time.perf_counter()
            recovered_row = None

            if probe_this_point and last_good_X is not None:
                recovered_row, X2, tel2 = _try_tier2(
                    engine, freq_ghz, last_good_X, last_good_cur, target_cur, point, pass_dir,
                )
                row.update(tel2)
                if recovered_row is not None:
                    row = {**recovered_row, **{k: v for k, v in row.items() if k not in recovered_row}}
                    row["route"] = ROUTE_POWER_SUBSTEP
                    X = X2

            if recovered_row is None and not past_boundary and last_good_X is not None:
                recovered_row, X3, tel3 = _try_tier3(
                    engine, freq_ghz, last_good_X, last_good_cur, target_cur, point, pass_dir,
                )
                row.update(tel3)
                if recovered_row is not None:
                    row = {**recovered_row, **{k: v for k, v in row.items() if k not in recovered_row}}
                    row["route"] = ROUTE_ARCLENGTH
                    X = X3

            if recovered_row is None and not past_boundary and last_good_X is not None:
                recovered_row, X4, tel4 = _try_tier4(
                    engine, freq_ghz, last_good_X, target_cur, point, pass_dir,
                )
                row.update(tel4)
                if recovered_row is not None:
                    row = {**recovered_row, **{k: v for k, v in row.items() if k not in recovered_row}}
                    row["route"] = ROUTE_FREQUENCY
                    X = X4

            row["recovery_wall_s"] = time.perf_counter() - t_rec0

            if recovered_row is None:
                if past_boundary:
                    row["route"] = ROUTE_PAST_BOUNDARY
                else:
                    row["route"] = ROUTE_FAILED_NUMERICAL
                    candidate_boundary_dbm = point.power_dbm
                if past_boundary:
                    probe_counter += 1

        if row["status"] == "PASS":
            last_good_X, last_good_cur = X, target_cur

        print(
            f"[G1] power={point.power_dbm:+.3f}dBm route={row['route']} "
            f"status={row['status']} direct_s={row['direct_wall_s']:.2f} "
            f"recovery_s={row['recovery_wall_s']:.2f} "
            f"boundary={candidate_boundary_dbm}",
            flush=True,
        )
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys and k not in ("_spectrum",):
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(freq_ghz: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    by_route: dict[str, int] = {}
    for r in rows:
        by_route[r["route"]] = by_route.get(r["route"], 0) + 1
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    total_direct = sum(r["direct_wall_s"] for r in rows)
    total_recovery = sum(r["recovery_wall_s"] for r in rows)
    pass_dbms = [r["pump_power_dbm"] for r in rows if r["status"] == "PASS"]
    fail_dbms = [r["pump_power_dbm"] for r in rows if r["status"] != "PASS"]
    wall_current_a = min(fail_dbms) if fail_dbms else None

    print(f"\n{'=' * 80}")
    print(f"G1 column summary: {freq_ghz} GHz, {n} targets")
    print(f"routes: {by_route}")
    print(f"PASS: {n_pass}/{n} ({100 * n_pass / n:.1f}%)")
    print(f"max converged power: {max(pass_dbms) if pass_dbms else None} dBm")
    print(f"first failing power (wall onset): {wall_current_a} dBm")
    print(f"total direct wall time: {total_direct:.1f}s")
    print(f"total recovery wall time: {total_recovery:.1f}s")
    print("DONE_G1")
    return {
        "freq_ghz": freq_ghz, "n": n, "n_pass": n_pass, "by_route": by_route,
        "total_direct_s": total_direct, "total_recovery_s": total_recovery,
        "wall_onset_dbm": wall_current_a,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--freq-ghz", type=float, default=7.9)
    p.add_argument("--n-power", type=int, default=20)
    p.add_argument("--power-min-dbm", type=float, default=-26.0)
    p.add_argument("--power-max-dbm", type=float, default=-16.0)
    p.add_argument("--probe-interval", type=int, default=1)
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--summary-json-out", type=Path, default=None)
    args = p.parse_args()

    rows = run_column(
        args.circuit_dir, args.outdir, args.freq_ghz,
        args.n_power, args.power_min_dbm, args.power_max_dbm,
        probe_interval=args.probe_interval,
    )
    csv_out = args.csv_out or (args.outdir / "g1_column_recovery.csv")
    write_csv(csv_out, rows)
    print(f"wrote {csv_out}")
    summary = summarize(args.freq_ghz, rows)
    if args.summary_json_out is not None:
        import json
        args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json_out.open("w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"wrote {args.summary_json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
