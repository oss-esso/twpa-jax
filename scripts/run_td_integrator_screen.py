"""Compare the available TD integrators on the first high-power 2c point.

The runner is deliberately serial and process-isolated.  Each method receives
the same validated fixed-drive restart state, and only compact diagnostics are
kept for the implicit-trapezoid path.  It is a campaign diagnostic, not a map
engine.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime monitor
    psutil = None


def process_rss_bytes(pid: int) -> int | None:
    """Return child RSS without requiring psutil on Windows."""
    if psutil is not None:
        try:
            return int(psutil.Process(pid).memory_info().rss)
        except psutil.Error:
            return None
    if sys.platform != "win32":  # pragma: no cover - campaign host is Windows
        return None
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000 | 0x0400, False, int(pid))
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else None
    finally:
        kernel32.CloseHandle(handle)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("implicit_euler", "implicit_trapezoid", "BDF", "Radau")


def monitor_process(
    process: subprocess.Popen[str],
    *,
    memory_limit_bytes: int,
) -> tuple[int, int, bool]:
    """Wait for a child and return code, peak RSS, and memory-limit status."""
    peak_rss = 0
    exceeded = False
    while process.poll() is None:
        rss = process_rss_bytes(process.pid)
        if rss is not None:
            peak_rss = max(peak_rss, rss)
            if rss > memory_limit_bytes:
                exceeded = True
                process.terminate()
                break
        time.sleep(2.0)
    if exceeded:
        try:
            process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            process.kill()
    return int(process.wait()), peak_rss, exceeded


def run_method(args: argparse.Namespace, method: str) -> dict[str, Any]:
    output_dir = args.outdir / method
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "h1_transient_branch_transfer.py"),
        "--circuit-dir", str(args.circuit_dir),
        "--checkpoint", str(args.checkpoint),
        "--transient-restart", str(args.transient_restart),
        "--outdir", str(output_dir),
        "--freq-ghz", str(args.freq_ghz),
        "--pump-port", str(args.pump_port),
        "--target-current-a", str(args.target_current_a),
        "--ramp-periods", "0",
        "--hold-periods", str(args.hold_periods),
        "--method", method,
        "--max-step", str(args.max_step),
        "--atol", str(args.atol),
        "--max-newton", str(args.max_newton),
        "--samples-per-period", str(args.samples_per_period),
    ]
    if method == "implicit_trapezoid":
        command.append("--compact-output")
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        return_code, peak_rss, memory_exceeded = monitor_process(
            process,
            memory_limit_bytes=args.memory_limit_gb * 1024**3,
        )
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "method": method,
        "return_code": return_code,
        "runtime_s": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "memory_limit_exceeded": memory_exceeded,
        "classification": summary.get("classification"),
        "final_status": summary.get("final_status"),
        "integrator": summary.get("integrator"),
        "stroboscopic": summary.get("stroboscopic"),
        "mean_phase_winding_cycles": summary.get("mean_phase_winding_cycles"),
        "output_dir": str(output_dir),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="validated production HB checkpoint used for provenance",
    )
    parser.add_argument(
        "--transient-restart", type=Path, required=True,
        help="common fixed-drive TD restart state for all methods",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--target-current-a", type=float, required=True)
    parser.add_argument("--hold-periods", type=int, default=100)
    parser.add_argument("--samples-per-period", type=int, default=8)
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max-newton", type=int, default=12)
    parser.add_argument("--memory-limit-gb", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for method in METHODS:
        print(f"starting {method}", flush=True)
        result = run_method(args, method)
        results.append(result)
        print(json.dumps(result, default=str), flush=True)
        if result["memory_limit_exceeded"]:
            print(f"stopping after memory limit for {method}", flush=True)
            break
    report = {
        "circuit_dir": str(args.circuit_dir),
        "checkpoint": str(args.checkpoint),
        "transient_restart": str(args.transient_restart),
        "freq_ghz": args.freq_ghz,
        "target_current_a": args.target_current_a,
        "hold_periods": args.hold_periods,
        "methods": list(METHODS),
        "results": results,
    }
    (args.outdir / "screen_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return 0 if all(item["return_code"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
