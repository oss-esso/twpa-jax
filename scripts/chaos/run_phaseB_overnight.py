#!/usr/bin/env python3
"""Fixed-grid overnight Phase B trace campaign.

The adaptive fine pass in ``run_phaseB_pump_only`` selects its bracket from the
classifier's evidence columns, and those columns are still under repair.  This
driver removes that dependency entirely: every power point is on a fixed grid
decided up front, nothing is bisected, and no verdict influences what gets
integrated.  The classifier still runs inside ``_run_point`` because that is
where the row is built, but its output is not consulted here and every point
persists ``trace.npz``, so the whole campaign can be re-reduced from disk once
the classifier is settled.

The run is resumable: a point whose ``trace.npz`` already exists is skipped, so
the driver can be re-launched after an interruption without losing work.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "phaseB_pump_only", ROOT / "scripts" / "chaos" / "run_phaseB_pump_only.py"
)
assert _SPEC is not None and _SPEC.loader is not None
PHASEB = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PHASEB
_SPEC.loader.exec_module(PHASEB)

# Measured per-point cost, seconds.  guarcello from its own recorded runtime_s;
# the JC devices from the post-JIT rates 11,300 and 12,700 steps/s.
COST_S = {"guarcello": 21.0, "jc_jtwpa": 324.0, "jc_fqjtwpa": 288.0}


def _grid(low: float, high: float, step: float) -> np.ndarray:
    """Inclusive grid from ``low`` to ``high`` at ``step`` dB."""
    count = int(round((high - low) / step)) + 1
    return np.linspace(low, high, count)


def build_plan() -> dict[str, np.ndarray]:
    """Fixed power grids, in dBm, one array per device.

    guarcello brackets its measured transition at -53.95 dBm; the JC devices
    bracket their harmonic-balance walls at -29.05 and -31.58 dBm.
    """
    guarcello = np.unique(np.round(np.concatenate([
        _grid(-54.60, -53.00, 0.05),   # full transition, 33 points
        _grid(-54.10, -53.70, 0.02),   # where residual_n1 leaves zero
    ]), 4))
    jtwpa = np.unique(np.round(_grid(-31.50, -25.50, 0.10), 4))
    fqjtwpa = np.unique(np.round(np.concatenate([
        np.linspace(-36.0, -28.58, 20),  # the coarse grid, only 3 points exist
        _grid(-33.50, -27.50, 0.10),
    ]), 4))
    ipm_2c_fixed = np.round(np.arange(0.300, 1.2001, 0.025), 6)
    rf_squid_2393_3wm = np.round(np.arange(0.100, 1.0001, 0.025), 6)
    return {
        "guarcello": guarcello, "jc_jtwpa": jtwpa, "jc_fqjtwpa": fqjtwpa,
        "ipm_2c_fixed": ipm_2c_fixed, "rf_squid_2393_3wm": rf_squid_2393_3wm,
    }


def _tmax_norm(device: str) -> float:
    """600 pump periods, the same budget ``run_phaseB_pump_only`` derives."""
    if device == "guarcello":
        omega_plasma = PHASEB.PAPER.Device().omega_plasma
        pump_hz = 7.0e9
    else:
        source = (
            PHASEB.phase5.phase_c_source_path(device)
            if device in {"ipm_2c_fixed", "rf_squid_2393_3wm"}
            else ROOT / "outputs" / "jc_doc_python_designs" / device
        )
        spec = PHASEB.phase5.derive_device_spec(source)
        omega_plasma = spec.omega_plasma
        pump_hz = PHASEB.phase5.resolve_pump_frequency(spec)
    return 600.0 * omega_plasma / pump_hz


def _point_dir(root: Path, device: str, power: float) -> Path:
    tag = f"{power:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return root / device / f"dense_{tag}"


def _is_done(path: Path) -> bool:
    trace = path / "trace.npz"
    return trace.exists() and trace.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "chaos" / "phaseB")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dt-norm", type=float, default=0.01)
    parser.add_argument("--devices", default="guarcello,jc_jtwpa,jc_fqjtwpa")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the cost estimate, integrate nothing")
    parser.add_argument("--skip-stride-check", action="store_true")
    args = parser.parse_args()

    plan = build_plan()
    devices = [name.strip() for name in args.devices.split(",") if name.strip()]
    root = args.output
    root.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, float, Path]] = []
    for device in devices:
        for power in plan[device]:
            path = _point_dir(root, device, float(power))
            if not _is_done(path):
                todo.append((device, float(power), path))

    known_cost = [COST_S[device] for device, _, _ in todo if device in COST_S]
    unknown_cost = sum(1 for device, _, _ in todo if device not in COST_S)
    total_s = sum(known_cost)
    print(f"devices      : {devices}")
    for device in devices:
        planned = len(plan[device])
        pending = sum(1 for name, _, _ in todo if name == device)
        print(f"  {device:<12} planned {planned:4d}   pending {pending:4d}")
    print(f"pending total: {len(todo)} points")
    print(f"serial cost  : {total_s / 3600.0:.2f} h for measured legacy-device rates")
    if unknown_cost:
        print(f"cost pending : {unknown_cost} points require Phase C measured rates")
    print(f"at {args.workers} workers : {total_s / 3600.0 / max(args.workers, 1):.2f} h plus pending points")
    if args.dry_run:
        return 0

    budgets = {device: _tmax_norm(device) for device in devices}
    for device, budget in budgets.items():
        print(f"  {device:<12} tmax_norm {budget:.1f} (600 pump periods)")

    log_path = root / "overnight_progress.csv"
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                PHASEB._run_point, device, power, path, args.dt_norm, budgets[device],
            ): (device, power, path)
            for device, power, path in todo
        }
        for future in as_completed(futures):
            device, power, path = futures[future]
            done += 1
            try:
                row = future.result()
                status = row.get("status", "OK")
            except Exception as exc:  # noqa: BLE001 - a bad point must not end the campaign
                status = f"DRIVER_ERROR {exc!r}"
            elapsed = time.perf_counter() - started
            rate = elapsed / max(done, 1)
            remaining = (len(todo) - done) * rate / 3600.0
            line = (f"[{done:4d}/{len(todo)}] {device:<12} {power:9.4f}  {status}  "
                    f"eta {remaining:5.2f} h")
            # Write the log directly.  A run piped through a console can be torn
            # down with its terminal, which is how the 2026-08-13 campaign died
            # at 60 of 190 points with no traceback.
            with (root / "overnight.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
            try:
                print(line, flush=True)
            except OSError:
                pass  # stdout is gone; the file log is authoritative
            manifest.append({
                "device": device, "pump_power_dbm": power,
                "point_dir": str(path.resolve().relative_to(ROOT.resolve())),
                "status": status,
            })
            with log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
                writer.writeheader()
                writer.writerows(manifest)

    (root / "overnight_manifest.json").write_text(
        json.dumps({"points": manifest, "wall_time_s": time.perf_counter() - started}, indent=2),
        encoding="utf-8",
    )
    print(f"integration complete in {(time.perf_counter() - started) / 3600.0:.2f} h")

    if not args.skip_stride_check:
        stride = ROOT / "scripts" / "chaos" / "run_guarcello_stride_check.py"
        if stride.exists():
            import subprocess
            print("running the record-stride alias check")
            subprocess.run([sys.executable, str(stride)], cwd=str(ROOT), check=False)
        else:
            print("stride check script not found, skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
