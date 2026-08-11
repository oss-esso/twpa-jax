"""Run independent zero-pump -> upward-ramp TD classifications for 2c.

Each target is a separate child process.  The validated HB checkpoint is used
only as an authoritative circuit/source fixture; the transient initial state is
constructed as the zero-pump equilibrium.  No restart from another target is
ever passed to the child.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_td_integrator_screen import process_rss_bytes


ROOT = Path(__file__).resolve().parents[1]


def monitor(process: subprocess.Popen[str], limit_bytes: int) -> tuple[int, int, bool]:
    peak = 0
    exceeded = False
    while process.poll() is None:
        rss = process_rss_bytes(process.pid)
        if rss is not None:
            peak = max(peak, rss)
            if rss > limit_bytes:
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


def load_reference(checkpoint: Path) -> tuple[float, float]:
    report = json.loads((checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    if report.get("final_status") != "VALID_CONVERGED":
        raise ValueError(f"reference checkpoint is not validated: {checkpoint}")
    metadata = report["metadata"]
    return float(metadata["pump_power_dbm_requested"]), float(metadata["pump_current_a"])


def run_target(args: argparse.Namespace, power_dbm: float, reference_power: float,
               reference_current: float) -> dict:
    target_current = reference_current * 10.0 ** ((power_dbm - reference_power) / 20.0)
    label = f"p_{power_dbm:+.6f}dbm".replace("+", "p").replace("-", "m").replace(".", "p")
    outdir = args.outdir / f"point_{label}"
    outdir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-B", str(ROOT / "scripts" / "h1_transient_branch_transfer.py"),
        "--circuit-dir", str(args.circuit_dir),
        "--checkpoint", str(args.reference_checkpoint),
        "--outdir", str(outdir),
        "--freq-ghz", str(args.freq_ghz), "--pump-port", str(args.pump_port),
        "--target-current-a", repr(target_current),
        "--initialization-mode", "zero_pump_equilibrium",
        "--ramp-periods", str(args.ramp_periods),
        "--hold-periods", str(args.hold_periods),
        "--method", "implicit_trapezoid", "--max-step", str(args.max_step),
        "--atol", str(args.atol), "--max-newton", str(args.max_newton),
        "--checkpoint-periods", str(args.checkpoint_periods),
        "--compact-output", "--compact-sample-count", str(args.compact_sample_count),
        "--compact-history-states", str(args.compact_history_states),
    ]
    stdout_path = outdir / "stdout.log"
    stderr_path = outdir / "stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
        code, peak, exceeded = monitor(process, args.memory_limit_gb * 1024**3)
    summary_path = outdir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    record = {
        "target_power_dbm": power_dbm,
        "target_current_a": target_current,
        "reference_checkpoint": str(args.reference_checkpoint),
        "initialization_source": "zero_pump_equilibrium_q0_p0",
        "previous_target_restart_used": False,
        "ramp_periods": args.ramp_periods,
        "hold_periods": args.hold_periods,
        "outdir": str(outdir),
        "return_code": code,
        "runtime_s": time.perf_counter() - started,
        "peak_rss_bytes": peak,
        "memory_limit_exceeded": exceeded,
        "classification": summary.get("classification"),
        "final_status": summary.get("final_status"),
        "integrator": summary.get("integrator"),
        "stroboscopic": summary.get("stroboscopic"),
        "decay_aware": summary.get("decay_aware"),
        "mean_phase_winding_cycles": summary.get("mean_phase_winding_cycles"),
        "source_telemetry": summary.get("source_telemetry"),
        "summary_path": str(summary_path),
    }
    (outdir / "independent_target_metadata.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--hold-periods", type=int, default=440)
    parser.add_argument("--ramp-periods", type=int, default=40)
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max-newton", type=int, default=12)
    parser.add_argument("--checkpoint-periods", type=int, default=10)
    parser.add_argument("--compact-sample-count", type=int, default=256)
    parser.add_argument("--compact-history-states", type=int, default=1024)
    parser.add_argument("--memory-limit-gb", type=int, default=6)
    parser.add_argument("--powers-dbm", type=float, nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference_power, reference_current = load_reference(args.reference_checkpoint)
    results = []
    for power_dbm in args.powers_dbm:
        print(f"starting independent target {power_dbm:+.6f} dBm", flush=True)
        result = run_target(args, power_dbm, reference_power, reference_current)
        results.append(result)
        print(json.dumps({key: result.get(key) for key in (
            "target_power_dbm", "target_current_a", "return_code", "runtime_s",
            "peak_rss_bytes", "classification", "final_status",
            "mean_phase_winding_cycles", "source_telemetry",
        )}, default=str), flush=True)
        if result["memory_limit_exceeded"]:
            break
    campaign = {
        "protocol": "independent_zero_pump_equilibrium_upward_turn_on",
        "circuit_dir": str(args.circuit_dir),
        "freq_ghz": args.freq_ghz,
        "pump_port": args.pump_port,
        "reference_checkpoint": str(args.reference_checkpoint),
        "reference_power_dbm": reference_power,
        "reference_current_a": reference_current,
        "attenuation_override": None,
        "initialization_source": "zero_pump_equilibrium_q0_p0",
        "previous_target_restart_used": False,
        "ramp_periods": args.ramp_periods,
        "hold_periods": args.hold_periods,
        "method": "implicit_trapezoid",
        "results": results,
    }
    (args.outdir / "campaign_summary.json").write_text(
        json.dumps(campaign, indent=2, default=str), encoding="utf-8"
    )
    return 0 if len(results) == len(args.powers_dbm) else 1


if __name__ == "__main__":
    raise SystemExit(main())
