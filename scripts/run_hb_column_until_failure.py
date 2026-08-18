"""Run one authoritative production HB column upward and stop at first failure."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--circuit-dir", str(args.circuit_dir.resolve()), "--outdir", str(args.outdir),
        "--executor", "inprocess", "--mode", "warmstart",
        "--inproc-pump-backend", "schur_cpu_mt",
        "--inproc-preconditioner", "real_coupled_fast",
        "--inproc-fold-predictor", "none", "--inproc-fail-fast",
        "--fold-skip-patience", "0", "--pump-current-jc-scale", "1.0",
        "--n-power", str(args.n_power), "--n-frequency", "1",
        "--pump-power-min-dbm", str(args.power_min_dbm),
        "--pump-power-max-dbm", str(args.power_max_dbm),
        "--pump-freq-min-ghz", str(args.freq_ghz),
        "--pump-freq-max-ghz", str(args.freq_ghz),
        "--pump-port", str(args.pump_port),
        "--signal-detuning-mhz", "500", "--no-signal-spectrum",
        "--pump-mode-count", str(args.pump_mode_count),
        "--nt", str(max(args.nt, 4 * args.pump_mode_count + 4)),
        "--continuation-steps", str(args.continuation_steps),
        "--sidebands", str(getattr(args, "sidebands", 6)),
        "--inproc-continuation", getattr(args, "continuation_method", "adaptive_secant"),
        "--log-level", "WARNING", "--overwrite",
    ]
    # Deliberately do not pass --attenuation-db: the A10 profile is resolved
    # by run_gain_map from the installed loss model.
    return run_gain_map.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs/ipm_2c_fixed")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=1)
    parser.add_argument("--n-power", type=int, default=14)
    parser.add_argument(
        "--pump-mode-count", type=int, default=10,
        help="Number of retained positive odd pump modes [1, 3, ..., 2K-1].",
    )
    parser.add_argument(
        "--sidebands", type=int, default=6,
        help="Explicit small-signal sideband count recorded in every row.",
    )
    parser.add_argument(
        "--continuation-method",
        choices=("fixed", "adaptive_secant"),
        default="adaptive_secant",
        help="In-process pump continuation strategy used for the timing comparison.",
    )
    parser.add_argument(
        "--continuation-steps", type=int, default=4,
        help="Fixed-ladder step count; adaptive uses this only as its fallback.",
    )
    parser.add_argument(
        "--cold-each-point", action="store_true",
        help="Solve every power point from a zero seed for fair strategy timing.",
    )
    parser.add_argument("--nt", type=int, default=40, help="HB reconstruction grid size.")
    parser.add_argument("--power-min-dbm", type=float, default=-35.0)
    parser.add_argument("--power-max-dbm", type=float, default=-21.31578947368421)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    production = build_args(args)
    points, _powers, _freqs = run_gain_map.build_points(production)
    points = sorted(points, key=lambda point: point.power_dbm)
    engine = run_gain_map.InProcessEngine(production)
    pass_dir = args.outdir / "pass"
    rows = []
    started = time.perf_counter()
    last_x = None
    for point in points:
        use_warm = last_x is not None and not args.cold_each_point
        mode = "warm" if use_warm else "seed"
        row, x = engine.solve_point(
            point, pass_dir, mode=mode, warm_X=last_x if use_warm else None
        )
        row["column_mode"] = mode
        row["sidebands"] = args.sidebands
        rows.append(row)
        _write_artifacts(args, rows, started)
        print(
            f"[HB-UP] point={point.index} P={point.power_dbm:+.6f} dBm "
            f"status={row['status']} pump_status={row.get('pump_status')} "
            f"residual={row.get('linear_rel_residual')} runtime={row.get('pump_runtime_s')}",
            flush=True,
        )
        if row["status"] != "PASS":
            break
        last_x = None if args.cold_each_point else x
    _write_artifacts(args, rows, started)
    return 0 if rows and rows[-1]["status"] != "PASS" else 1


def _write_artifacts(args: argparse.Namespace, rows: list[dict], started: float) -> None:
    """Atomically publish the partial column after every accepted point."""
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and key != "_spectrum":
                keys.append(key)
    csv_tmp = args.outdir / "hb_up_to_failure.tmp.csv"
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    csv_tmp.replace(args.outdir / "hb_up_to_failure.csv")
    report = {
        "circuit_dir": str(args.circuit_dir.resolve()), "freq_ghz": args.freq_ghz,
        "power_min_dbm": args.power_min_dbm, "power_max_dbm": args.power_max_dbm,
        "n_power": args.n_power, "sidebands": args.sidebands,
        "continuation_method": args.continuation_method,
        "continuation_steps": args.continuation_steps,
        "cold_each_point": args.cold_each_point,
        "attenuation_override_db": None,
        "attenuation_policy": "default_A10_profile",
        "pump_modes": list(range(1, 2 * args.pump_mode_count, 2)),
        "rows": rows, "first_failure_index": next((r["point_index"] for r in rows if r["status"] != "PASS"), None),
        "runtime_s": time.perf_counter() - started,
    }
    json_tmp = args.outdir / "hb_up_to_failure.tmp.json"
    json_tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    json_tmp.replace(args.outdir / "hb_up_to_failure.json")


if __name__ == "__main__":
    raise SystemExit(main())
