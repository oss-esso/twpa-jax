"""Run the fixed-drive high-period 2c column from high to low power.

The campaign is intentionally serial and process-isolated.  Each target is
validated against its authoritative production HB checkpoint, while the TD
state is carried down one power point at a time when the preceding run leaves a
restart checkpoint.  Only compact output is requested.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_td_integrator_screen import process_rss_bytes


ROOT = Path(__file__).resolve().parents[1]


def monitor(process: subprocess.Popen[str], limit: int) -> tuple[int, int, bool]:
    peak = 0
    exceeded = False
    while process.poll() is None:
        rss = process_rss_bytes(process.pid)
        if rss is not None:
            peak = max(peak, rss)
            if rss > limit:
                exceeded = True
                process.terminate()
                break
        time.sleep(2.0)
    if exceeded:
        try:
            process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            process.kill()
    return int(process.wait()), peak, exceeded


def load_point(point_dir: Path) -> dict[str, Any]:
    report = json.loads((point_dir / "pump" / "pump_report.json").read_text())
    metadata = report["metadata"]
    return {
        "point_dir": str(point_dir),
        "index": int(point_dir.name.split("_", 2)[1]),
        "power_dbm": float(metadata["pump_power_dbm_requested"]),
        "target_current_a": float(metadata["pump_current_a"]),
        "checkpoint": str(point_dir / "pump"),
    }


def generated_point(point_dir: Path, reference: dict[str, Any]) -> dict[str, Any]:
    """Create a fixed-drive target from the documented 20-point power grid.

    This is deliberately not labelled as an HB checkpoint.  The validated
    point-12 fixture supplies only circuit/provenance metadata; the target
    current is generated from the same square-root power convention used by
    the production map.
    """
    match = re.match(r"point_(\d+)_p_m", point_dir.name)
    if match is None:
        raise ValueError(f"cannot parse point index from {point_dir}")
    index = int(match.group(1))
    power = -35.0 + index * (20.0 / 19.0)
    current = reference["target_current_a"] * 10.0 ** (
        (power - reference["power_dbm"]) / 20.0
    )
    return {
        "point_dir": str(point_dir), "index": index,
        "power_dbm": power, "target_current_a": current,
        "checkpoint": reference["checkpoint"],
        "target_provenance": "generated_from_20_point_power_grid",
    }


def run_target(args: argparse.Namespace, point: dict[str, Any], restart: Path | None) -> dict[str, Any]:
    outdir = args.outdir / f"point_{point['index']:04d}_{point['power_dbm']:+.4f}dbm"
    outdir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts" / "h1_transient_branch_transfer.py"),
        "--circuit-dir", str(args.circuit_dir),
        "--checkpoint", point["checkpoint"],
        "--outdir", str(outdir),
        "--freq-ghz", str(args.freq_ghz), "--pump-port", str(args.pump_port),
        "--target-current-a", str(point["target_current_a"]),
        "--ramp-periods", str(args.ramp_periods), "--hold-periods", str(args.hold_periods),
        "--method", "implicit_trapezoid", "--max-step", str(args.max_step),
        "--atol", str(args.atol), "--max-newton", str(args.max_newton),
        "--checkpoint-periods", str(args.checkpoint_periods),
        "--compact-output", "--compact-sample-count", str(args.compact_sample_count),
        "--compact-history-states", str(args.compact_history_states),
    ]
    if restart is not None:
        command += ["--transient-restart", str(restart)]
    stdout_path = outdir / "stdout.log"
    stderr_path = outdir / "stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
        code, peak, exceeded = monitor(process, args.memory_limit_gb * 1024**3)
    summary_path = outdir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {
        "point": point,
        "outdir": str(outdir),
        "source_restart": str(restart) if restart else None,
        "return_code": code,
        "runtime_s": time.perf_counter() - started,
        "peak_rss_bytes": peak,
        "memory_limit_exceeded": exceeded,
        "classification": summary.get("classification"),
        "final_status": summary.get("final_status"),
        "integrator": summary.get("integrator"),
        "stroboscopic": summary.get("stroboscopic"),
        "mean_phase_winding_cycles": summary.get("mean_phase_winding_cycles"),
        "summary_path": str(summary_path),
    }


def find_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = sorted(args.points_root.glob("point_*_fp_7p9ghz"))
    reference_path = args.points_root / args.reference_point
    reference = load_point(reference_path)
    points = []
    for path in paths:
        report_path = path / "pump" / "pump_report.json"
        if report_path.exists():
            points.append(load_point(path))
        elif args.allow_generated_targets:
            points.append(generated_point(path, reference))
        else:
            raise ValueError(
                f"missing authoritative HB checkpoint for descending target: {path}; "
                "pass --allow-generated-targets only for an explicitly non-HB diagnostic"
            )
    points.sort(key=lambda item: item["index"], reverse=True)
    return [item for item in points if args.min_index <= item["index"] <= args.start_index]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-root", type=Path, default=ROOT / "outputs/high_power_2c_column_settings_7p9_v1/pass/points")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs/ipm_2c_fixed")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=12)
    parser.add_argument("--min-index", type=int, default=0)
    parser.add_argument("--hold-periods", type=int, default=440)
    parser.add_argument("--ramp-periods", type=int, default=40)
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max-newton", type=int, default=12)
    parser.add_argument("--checkpoint-periods", type=int, default=10)
    parser.add_argument("--compact-sample-count", type=int, default=256)
    parser.add_argument("--compact-history-states", type=int, default=1024)
    parser.add_argument("--memory-limit-gb", type=int, default=6)
    parser.add_argument("--initial-restart", type=Path, required=True)
    parser.add_argument(
        "--reference-point", default="point_0012_p_m22p3684dbm_fp_7p9ghz",
        help="validated fixture used when lower target directories have no HB report",
    )
    parser.add_argument(
        "--allow-generated-targets", action="store_true",
        help="explicitly opt into non-HB fixed-drive targets; disabled by default",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    points = find_points(args)
    if not points:
        raise SystemExit("no authoritative 7.9 GHz point checkpoints found")
    results = []
    restart = args.initial_restart
    for point in points:
        print(f"starting point {point['index']} {point['power_dbm']:+.4f} dBm", flush=True)
        result = run_target(args, point, restart)
        results.append(result)
        print(json.dumps({k: result.get(k) for k in (
            "point", "source_restart", "return_code", "runtime_s", "peak_rss_bytes",
            "memory_limit_exceeded", "classification", "final_status")}, default=str), flush=True)
        if result["memory_limit_exceeded"]:
            break
        candidate = Path(result["outdir"]) / "restart_checkpoints" / "transient_restart.npz"
        if candidate.exists():
            restart = candidate
        else:
            restart = None
        if result["return_code"] != 0 and restart is None:
            break
    report = {
        "circuit_dir": str(args.circuit_dir), "freq_ghz": args.freq_ghz,
        "pump_port": args.pump_port, "method": "implicit_trapezoid",
        "hold_periods": args.hold_periods, "ramp_periods": args.ramp_periods,
        "memory_limit_gb": args.memory_limit_gb,
        "initial_restart": str(args.initial_restart), "results": results,
    }
    (args.outdir / "descending_campaign.json").write_text(json.dumps(report, indent=2, default=str))
    return 0 if len(results) == len(points) else 1


if __name__ == "__main__":
    raise SystemExit(main())
