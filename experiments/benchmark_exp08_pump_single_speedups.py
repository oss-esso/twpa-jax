
# experiments/benchmark_exp08_pump_single_speedups.py
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"


def safe_rmtree(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"refusing to remove outside {root}: {path}")
    if path.exists():
        shutil.rmtree(path)


def run_cmd(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path, timeout_s: int) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        try:
            proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=err, timeout=timeout_s)
            return int(proc.returncode), time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            return 124, time.perf_counter() - t0


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def final_pump_report(pump_dir: Path) -> dict[str, Any]:
    rep = load_json(pump_dir / "pump_report.json")
    reports = rep.get("reports", [])
    final = reports[-1] if reports else {}
    return {
        "pump_status": rep.get("final_status", "MISSING"),
        "pump_runtime_s": float(sum(float(r.get("runtime_s", 0.0)) for r in reports)),
        "pump_total_runtime_s": float(rep.get("metadata", {}).get("total_runtime_s", sum(float(r.get("runtime_s", 0.0)) for r in reports))),
        "final_coeff_residual": final.get("coeff_rel"),
        "final_time_residual": final.get("time_rel"),
        "continuation_steps_completed": len(reports),
        "total_newton_iterations": int(sum(int(r.get("newton_iterations", 0)) for r in reports)),
        "total_gmres_iterations": int(sum(int(r.get("gmres_iterations_total", 0)) for r in reports)),
        "final_branch_current_max": rep.get("solution_summary", {}).get("branch_i_max_abs"),
        "final_branch_current_rms": rep.get("solution_summary", {}).get("branch_i_rms"),
        "final_branch_flux_max": rep.get("solution_summary", {}).get("branch_psi_max_abs"),
        "final_branch_flux_rms": rep.get("solution_summary", {}).get("branch_psi_rms"),
        "failure_reason": final.get("failure_reason", ""),
        "continuation_trace": rep.get("metadata", {}).get("continuation_trace", {}),
        "metadata": rep.get("metadata", {}),
    }


def final_gain_report(gain_dir: Path) -> dict[str, Any]:
    rep = load_json(gain_dir / "gain_report.json")
    rows = rep.get("results", [])
    row = rows[0] if rows else {}
    return {
        "gain_status": row.get("status", "MISSING"),
        "gain_db": row.get("gain_db"),
        "gain_vs_off_db": row.get("gain_vs_off_db"),
        "gain_vs_pumpdiag_db": row.get("gain_vs_pumpdiag_db"),
        "linear_rel_residual": row.get("linear_rel_residual"),
        "gain_total_runtime_s": rep.get("metadata", {}).get("total_runtime_s"),
        "gain_factor_solve_runtime_s": row.get("factor_solve_runtime_s"),
    }


def rel_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


def as_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    return float(x)


def run_variant(name: str, flags: list[str], out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    variant_dir = out_root / name
    pump_dir = variant_dir / "pump"
    gain_dir = variant_dir / "gain"
    if args.overwrite:
        safe_rmtree(variant_dir, out_root)
    variant_dir.mkdir(parents=True, exist_ok=True)

    pump_cmd = [sys.executable, str(EXP08), "--outdir", str(pump_dir), *flags]
    pump_rc, pump_wall = run_cmd(
        pump_cmd,
        ROOT,
        variant_dir / "pump_stdout.txt",
        variant_dir / "pump_stderr.txt",
        args.pump_timeout_s,
    )

    result: dict[str, Any] = {
        "variant": name,
        "pump_returncode": pump_rc,
        "pump_wall_runtime_s": pump_wall,
        "pump_dir": str(pump_dir.relative_to(ROOT)),
        "gain_dir": str(gain_dir.relative_to(ROOT)),
        "flags": " ".join(flags),
    }

    if pump_rc == 0 and (pump_dir / "pump_report.json").exists():
        result.update(final_pump_report(pump_dir))
    else:
        result.update({"pump_status": "RUN_FAILED", "failure_reason": f"pump returncode {pump_rc}"})
        return result

    gain_cmd = [
        sys.executable,
        str(EXP09),
        "--pump-dir",
        str(pump_dir),
        "--outdir",
        str(gain_dir),
        "--signal-ghz",
        str(args.signal_ghz),
    ]
    gain_rc, gain_wall = run_cmd(
        gain_cmd,
        ROOT,
        variant_dir / "gain_stdout.txt",
        variant_dir / "gain_stderr.txt",
        args.gain_timeout_s,
    )
    result["gain_returncode"] = gain_rc
    result["gain_wall_runtime_s"] = gain_wall

    if gain_rc == 0 and (gain_dir / "gain_report.json").exists():
        result.update(final_gain_report(gain_dir))
    else:
        result.update({"gain_status": "RUN_FAILED", "failure_reason": f"gain returncode {gain_rc}"})

    return result


def add_acceptance(row: dict[str, Any], baseline: dict[str, Any], tol_gain_db: float, tol_branch_rel: float) -> None:
    if row["variant"] == "baseline_cold_fixed":
        row.update({
            "gain_delta_db": 0.0,
            "gain_vs_off_delta_db": 0.0,
            "gain_vs_pumpdiag_delta_db": 0.0,
            "branch_current_max_rel_delta": 0.0,
            "pump_speedup_vs_baseline": 1.0,
            "accepted": True,
            "acceptance_reason": "baseline reference",
        })
        return

    gain_delta = None if row.get("gain_db") is None else float(row["gain_db"]) - float(baseline["gain_db"])
    off_delta = None if row.get("gain_vs_off_db") is None else float(row["gain_vs_off_db"]) - float(baseline["gain_vs_off_db"])
    diag_delta = None if row.get("gain_vs_pumpdiag_db") is None else float(row["gain_vs_pumpdiag_db"]) - float(baseline["gain_vs_pumpdiag_db"])
    branch_delta = rel_diff(as_float(row.get("final_branch_current_max")), as_float(baseline.get("final_branch_current_max")))
    speedup = None
    if row.get("pump_total_runtime_s") is not None and baseline.get("pump_total_runtime_s"):
        speedup = float(baseline["pump_total_runtime_s"]) / max(float(row["pump_total_runtime_s"]), 1e-300)

    reasons: list[str] = []
    if row.get("pump_status") != "VALID_CONVERGED":
        reasons.append("pump not converged")
    if row.get("gain_status") != "VALID_SOLVED":
        reasons.append("gain not solved")
    if row.get("final_coeff_residual") is None or float(row["final_coeff_residual"]) > float(row.get("metadata", {}).get("newton_tol", 1e-9)) * 10.0:
        reasons.append("coeff residual above tolerance margin")
    for label, delta in (("gain", gain_delta), ("gain_vs_off", off_delta), ("gain_vs_pumpdiag", diag_delta)):
        if delta is None or abs(delta) > tol_gain_db:
            reasons.append(f"{label} drift")
    if branch_delta is None or branch_delta > tol_branch_rel:
        reasons.append("branch current drift")
    if speedup is None or speedup <= 1.0:
        reasons.append("not faster")

    row.update({
        "gain_delta_db": gain_delta,
        "gain_vs_off_delta_db": off_delta,
        "gain_vs_pumpdiag_delta_db": diag_delta,
        "branch_current_max_rel_delta": branch_delta,
        "pump_speedup_vs_baseline": speedup,
        "accepted": not reasons,
        "acceptance_reason": "accepted" if not reasons else "; ".join(reasons),
    })


def recommend_variant(
    accepted: list[dict[str, Any]], *, noise_rel: float = 0.15
) -> dict[str, Any] | None:
    """Pick the variant to recommend from the accepted set.

    Speedups within ``noise_rel`` of the best are treated as a tie (single-point
    wall-clock timing is noisy at this scale). Among those near-best variants the
    strictest Newton tolerance wins, so a merely-looser tolerance never gets
    recommended just because it happened to time slightly faster. Ties on
    tolerance fall back to raw speedup.
    """
    if not accepted:
        return None

    def speedup(row: dict[str, Any]) -> float:
        return float(row.get("pump_speedup_vs_baseline") or 0.0)

    def newton_tol(row: dict[str, Any]) -> float:
        return float(row.get("metadata", {}).get("newton_tol", 1e-9))

    best_speedup = max(speedup(r) for r in accepted)
    near_best = [r for r in accepted if speedup(r) >= best_speedup * (1.0 - noise_rel)]
    return min(near_best, key=lambda r: (newton_tol(r), -speedup(r)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "variant", "accepted", "acceptance_reason", "pump_status", "gain_status",
        "pump_total_runtime_s", "pump_wall_runtime_s", "pump_speedup_vs_baseline",
        "final_coeff_residual", "final_time_residual", "continuation_steps_completed",
        "total_newton_iterations", "total_gmres_iterations", "final_branch_current_max",
        "final_branch_current_rms", "final_branch_flux_max", "final_branch_flux_rms",
        "gain_db", "gain_delta_db", "gain_vs_off_db", "gain_vs_off_delta_db",
        "gain_vs_pumpdiag_db", "gain_vs_pumpdiag_delta_db", "linear_rel_residual",
        "gain_total_runtime_s", "failure_reason", "flags", "pump_dir", "gain_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_report(path: Path, rows: list[dict[str, Any]], recommended: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pump Speedup Single-Point Benchmark", ""]
    if recommended is None:
        lines.append("No optimized variant passed the acceptance gate.")
    else:
        lines.append(
            f"Recommended opt-in mode: `{recommended['variant']}` "
            f"({recommended['pump_speedup_vs_baseline']:.2f}x pump speedup)."
        )
    lines.append("")
    lines.append("| variant | accepted | pump s | speedup | gain dB | gain delta dB | coeff rel | time rel |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r.get('accepted')} | "
            f"{float(r.get('pump_total_runtime_s') or math.nan):.6g} | "
            f"{float(r.get('pump_speedup_vs_baseline') or math.nan):.3g} | "
            f"{float(r.get('gain_db') or math.nan):.6g} | "
            f"{float(r.get('gain_delta_db') or math.nan):.3g} | "
            f"{float(r.get('final_coeff_residual') or math.nan):.3g} | "
            f"{float(r.get('final_time_residual') or math.nan):.3g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/pump_speedup_benchmark")
    ap.add_argument("--signal-ghz", type=float, default=6.0)
    ap.add_argument("--pump-timeout-s", type=int, default=240)
    ap.add_argument("--gain-timeout-s", type=int, default=120)
    ap.add_argument("--linear-seed-maxiter", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = (ROOT / args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    variants = [
        ("baseline_cold_fixed", ["--initial-guess", "zero", "--continuation-mode", "fixed", "--continuation-steps", "20"]),
        ("linear_seed_fixed", ["--initial-guess", "linear_phasor", "--linear-seed-maxiter", str(args.linear_seed_maxiter), "--continuation-mode", "fixed", "--continuation-steps", "20"]),
        ("linear_seed_adaptive", ["--initial-guess", "linear_phasor", "--linear-seed-maxiter", str(args.linear_seed_maxiter), "--continuation-mode", "adaptive"]),
        ("linear_seed_adaptive_tol1e8", ["--initial-guess", "linear_phasor", "--linear-seed-maxiter", str(args.linear_seed_maxiter), "--continuation-mode", "adaptive", "--newton-tol", "1e-8"]),
    ]

    rows = [run_variant(name, flags, out_root, args) for name, flags in variants]
    baseline = rows[0]
    for row in rows:
        add_acceptance(row, baseline, tol_gain_db=0.01, tol_branch_rel=1e-4)

    accepted = [r for r in rows[1:] if r.get("accepted")]
    recommended = recommend_variant(accepted)

    write_csv(out_root / "single_point_variants.csv", rows)
    summary = {
        "baseline": baseline,
        "variants": rows,
        "recommended_variant": None if recommended is None else recommended["variant"],
        "acceptance": {
            "gain_db_abs_tol": 0.01,
            "branch_current_max_relative_tol": 1e-4,
        },
    }
    (out_root / "single_point_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "baseline_single_point.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    write_report(ROOT / "docs" / "reports" / "pump_speedup_single_point.md", rows, recommended)

    print(f"wrote {out_root / 'single_point_variants.csv'}")
    print(f"wrote {out_root / 'single_point_summary.json'}")
    print(f"recommended_variant={None if recommended is None else recommended['variant']}")


if __name__ == "__main__":
    main()
