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
        "--signal-detuning-mhz", "500", "--no-signal-spectrum",
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
    parser.add_argument("--n-power", type=int, default=14)
    parser.add_argument("--power-min-dbm", type=float, default=-35.0)
    parser.add_argument("--power-max-dbm", type=float, default=-21.31578947368421)
    args = parser.parse_args(argv)
    production = build_args(args)
    points, _powers, _freqs = run_gain_map.build_points(production)
    points = sorted(points, key=lambda point: point.power_dbm)
    engine = run_gain_map.InProcessEngine(production)
    pass_dir = args.outdir / "pass"
    rows = []
    started = time.perf_counter()
    last_x = None
    for point in points:
        mode = "warm" if last_x is not None else "seed"
        row, x = engine.solve_point(point, pass_dir, mode=mode, warm_X=last_x)
        row["column_mode"] = mode
        rows.append(row)
        print(
            f"[HB-UP] point={point.index} P={point.power_dbm:+.6f} dBm "
            f"status={row['status']} pump_status={row.get('pump_status')} "
            f"residual={row.get('linear_rel_residual')} runtime={row.get('pump_runtime_s')}",
            flush=True,
        )
        if row["status"] != "PASS":
            break
        last_x = x
    args.outdir.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and key != "_spectrum":
                keys.append(key)
    with (args.outdir / "hb_up_to_failure.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    report = {
        "circuit_dir": str(args.circuit_dir.resolve()), "freq_ghz": args.freq_ghz,
        "power_min_dbm": args.power_min_dbm, "power_max_dbm": args.power_max_dbm,
        "n_power": args.n_power, "attenuation_override_db": None,
        "attenuation_policy": "default_A10_profile", "pump_modes": [1,3,5,7,9,11,13,15,17,19],
        "rows": rows, "first_failure_index": next((r["point_index"] for r in rows if r["status"] != "PASS"), None),
        "runtime_s": time.perf_counter() - started,
    }
    (args.outdir / "hb_up_to_failure.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0 if rows and rows[-1]["status"] != "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
