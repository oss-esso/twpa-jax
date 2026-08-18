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
    rf_squid_2393_3wm = np.unique(np.round(np.concatenate([
        np.geomspace(2.5e-6, 1.5e-5, 40),
        np.asarray([6.3246e-6, 7.0963e-6, 1.0024e-5]),
    ]), 12))
    return {
        "guarcello": guarcello, "jc_jtwpa": jtwpa, "jc_fqjtwpa": fqjtwpa,
        "ipm_2c_fixed": ipm_2c_fixed, "rf_squid_2393_3wm": rf_squid_2393_3wm,
    }


def _tmax_norm(device: str, periods: float = 600.0) -> float:
    """``periods`` pump periods; 600 is what ``run_phaseB_pump_only`` derives.

    600 is only valid for the Guarcello device.  Measured 2026-08-14, the
    near-lossless devices need about 1050 pump periods before the off-lattice
    energy reaches its floor, and anything shorter analyses residual ringing.
    """
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
    return float(periods) * omega_plasma / pump_hz


def _point_dir(root: Path, device: str, power: float) -> Path:
    """Directory name for one control value.

    The RF-SQUID control axis is an absolute on-chip current near 1e-5 A, and
    a fixed four-decimal tag rounds every one of its grid points to ``p0p0000``
    - one directory for the whole sweep, with ``_is_done`` then skipping 42 of
    43 points and the campaign reporting success on a single measurement.  Tag
    that device in nanoamps; the dBm devices keep their existing names so the
    traces already on disk still resume.
    """
    if device == "rf_squid_2393_3wm":
        tag = f"{power * 1e9:.3f}nA".replace(".", "p")
    else:
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
    parser.add_argument(
        "--record-stride", type=int, default=20,
        help="integrator steps between stored samples. 20 gives the JC devices "
             "149-304 samples per pump period but only 6.23 for guarcello, "
             "below the eight-sample guard for a stroboscopic section; pass 4 "
             "for guarcello.",
    )
    parser.add_argument("--devices", default="guarcello,jc_jtwpa,jc_fqjtwpa")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the cost estimate, integrate nothing")
    parser.add_argument("--skip-stride-check", action="store_true")
    parser.add_argument("--periods", type=float, default=600.0,
                        help="pump periods per point; 600 is valid only for guarcello, "
                             "the near-lossless devices need >~2100 (settle ~1050, "
                             "analysed window is the last half)")
    parser.add_argument("--control-min", type=float, default=None,
                        help="drop plan points below this control value")
    parser.add_argument("--control-max", type=float, default=None,
                        help="drop plan points above this control value")
    parser.add_argument("--n-points", type=int, default=None,
                        help="evenly subsample the surviving plan to this many "
                             "points per device; the endpoints are always kept")
    parser.add_argument("--control-linspace", type=float, nargs=3, default=None,
                        metavar=("LOW", "HIGH", "COUNT"),
                        help="replace the built-in plan with an even sweep. The "
                             "built-in grids bracket each device's known "
                             "transition and cannot be widened with "
                             "--control-min/--control-max, which only filter")
    parser.add_argument("--signal-current-a", type=float, default=0.0,
                        help="probe tone current for the solver devices; 0 keeps the "
                             "campaign pump-only. A signal makes the forcing "
                             "quasi-periodic, so the emitted bifurcation verdict is "
                             "not meaningful and the run is for gain and spectra only")
    parser.add_argument("--signal-dbm", type=float, default=None,
                        help="probe tone power for the guarcello paper device; the "
                             "solver devices use --signal-current-a instead")
    args = parser.parse_args()

    plan = build_plan()
    if args.control_linspace is not None:
        low, high, count = args.control_linspace
        sweep = np.round(np.linspace(low, high, int(count)), 6)
        plan = {device: sweep.copy() for device in plan}
    if args.control_min is not None or args.control_max is not None or args.n_points:
        low = -np.inf if args.control_min is None else args.control_min
        high = np.inf if args.control_max is None else args.control_max
        for device, powers in plan.items():
            kept = powers[(powers >= low) & (powers <= high)]
            if args.n_points and kept.size > args.n_points:
                index = np.unique(np.round(
                    np.linspace(0, kept.size - 1, args.n_points),
                ).astype(int))
                kept = kept[index]
            plan[device] = kept
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

    budgets = {device: _tmax_norm(device, args.periods) for device in devices}
    for device, budget in budgets.items():
        print(f"  {device:<12} tmax_norm {budget:.1f} ({args.periods:.0f} pump periods)")

    # The pump-off tone amplitude is the denominator of gain_vs_off_db and is
    # independent of pump power, so it is measured once per device here rather
    # than at every point.  A device whose reference cannot be measured runs
    # without gain_vs_off_db instead of failing the campaign.
    signal_on = args.signal_current_a > 0.0 or args.signal_dbm is not None
    references: dict[str, float | None] = {device: None for device in devices}
    if signal_on:
        # The reference costs one full integration per device and runs before
        # the pool, so it is pure serial time -- 39 min across three devices on
        # the 2026-08-15 run, which is most of that campaign's overrun.  It
        # depends only on the device, the probe level and the budget, never on
        # pump power, so cache it on disk and pay it once ever.
        cache_path = root / "pump_off_reference_cache.json"
        cache: dict[str, float] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cache = {}
        for device in devices:
            key = (f"{device}|dt={args.dt_norm}|tmax={budgets[device]:.6f}"
                   f"|i={args.signal_current_a}|dbm={args.signal_dbm}")
            if key in cache:
                references[device] = float(cache[key])
                print(f"  {device:<12} pump-off reference "
                      f"{references[device]:.6e} V (cached)")
                continue
            try:
                started_reference = time.perf_counter()
                references[device] = PHASEB.pump_off_reference(
                    device, args.dt_norm, budgets[device],
                    signal_current_a=args.signal_current_a,
                    signal_dbm=args.signal_dbm,
                )
                cache[key] = references[device]
                cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
                print(f"  {device:<12} pump-off reference "
                      f"{references[device]:.6e} V "
                      f"({time.perf_counter() - started_reference:.0f} s, cached)")
            except (ValueError, RuntimeError) as error:
                print(f"  {device:<12} pump-off reference unavailable: {error}")
        print("  signal installed: the forcing is no longer periodic at the pump, "
              "so treat every emitted bifurcation verdict as void")

    log_path = root / "overnight_progress.csv"
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                PHASEB._run_point, device, power, path, args.dt_norm, budgets[device],
                args.signal_current_a, args.signal_dbm, references[device],
                args.record_stride,
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
                "device": device,
                "control_value": power,
                "control_axis": (
                    "pump_current_peak_a" if device == "rf_squid_2393_3wm"
                    else "pump_power_dbm"
                ),
                "pump_power_dbm": power if device != "rf_squid_2393_3wm" else None,
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
