#!/usr/bin/env python3
"""JC re-measurement at a settled record length.

The 2026-08-13 campaign ran 600 pump periods and analysed the last half.  The
JC devices need about 1000 periods to settle, so that entire analysed window
was residual transient, roughly 600x above the true off-lattice floor
(measured: 0.003124/0.001715/0.000879/0.000665 at 600 periods against
0.9e-6/1.3e-6/0.9e-6/1.3e-6 at 2400).  Guarcello is unaffected; it settles
inside 600.

Two passes, deliberately different in what they trade away:

* ``cold``  - every point integrated from rest at ``COLD_PERIODS``.  No warm
  start, no shortcut.  These are the authoritative points and they cover the
  transition, where a warm start could carry the solution along a branch past
  where it would naturally lose stability.
* ``chain`` - a warm-started sweep across the full power span.  The first point
  pays the full settle; each subsequent point only relaxes a small power step.
  This is what makes the floor affordable, and running it in both directions
  turns warm-starting's own hazard into a measurement: if up and down disagree,
  that difference is hysteresis, which is a result rather than an error.

Analysis is NOT performed here.  Every point persists ``trace.npz`` and the
reduction happens separately, so the settling window can be chosen after the
fact rather than baked in by ``steady_state_start_index``.
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

# Import the kernel by its package name, never through spec_from_file_location
# under an ad-hoc alias.  Its integrator is @njit(cache=True), and numba pickles
# the importing module's name into the on-disk cache entry; a second process
# that loads the same kernel under a different name then fails to rebuild the
# environment with ModuleNotFoundError.  Every driver must agree on this name.
from scripts.chaos import run_guarcello_jc_phase5 as PHASE5

# Settling measured 2026-08-14 on jc_jtwpa at -31.5 dBm: off-lattice energy
# reaches its 1.1e-6 floor at about 1050 pump periods.  A cold point therefore
# needs >1050 of settle before its recorded window means anything.
COLD_PERIODS = 1500          # settle ~1100, record ~400
CHAIN_FIRST_PERIODS = 1500   # the chain's first point pays the same full settle
CHAIN_STEP_PERIODS = 700     # a small power step relaxes far faster than switch-on

DEVICES = {
    # device: (cold transition span, chain span, chain direction(s))
    "jc_jtwpa": ((-32.5, -26.5), (-36.0, -25.5), ("up", "down")),
    "jc_fqjtwpa": ((-35.5, -29.5), (-36.0, -27.5), ("up",)),
}
COLD_POINTS = 20
CHAIN_POINTS = 20


def _spec(device: str):
    return PHASE5.derive_device_spec(ROOT / "outputs" / "jc_doc_python_designs" / device)


def _current_for(device: str, power_dbm: float) -> float:
    """dBm -> on-chip current, matching the original campaign's mapping.

    Reproduced from ``run_phaseB_pump_only._pump_only_jc`` on purpose: this run
    has to be comparable to the campaign it replaces, so the axis label must be
    derived the same way.  The mapping extrapolates log-linearly above the HB
    table, which is a known weakness of that axis and is why every row also
    records the current itself.
    """
    hb_name = device.removeprefix("jc_")
    path = ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811" / hb_name / "hb_up_to_failure.csv"
    rows = PHASE5._read_hb_rows(path)
    valid = [
        row for row in rows
        if row.get("status") == "PASS"
        and row.get("pump_status") in {"VALID_CONVERGED", "VALID_SOLVED"}
    ]
    powers = np.array([float(row["pump_power_dbm"]) for row in valid])
    currents = np.array([float(row["pump_current_peak_a"]) for row in valid])
    if power_dbm <= powers[-1]:
        return float(np.interp(power_dbm, powers, currents))
    slope = np.polyfit(powers[-2:], np.log(currents[-2:]), 1)
    return float(np.exp(np.polyval(slope, power_dbm)))


def _tmax_norm(device: str, periods: float) -> float:
    spec = _spec(device)
    return periods * spec.omega_plasma / PHASE5.resolve_pump_frequency(spec)


def _done(path: Path) -> bool:
    trace = path / "trace.npz"
    return trace.exists() and trace.stat().st_size > 0


def _tag(power: float) -> str:
    return f"{power:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _persist(path: Path, device: str, power: float, periods: float,
             row: dict[str, Any], trace_t: np.ndarray, trace_v: np.ndarray,
             mode: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "trace.npz", t=trace_t, v_out=trace_v)
    record = dict(row)
    record.update({
        "device": device, "pump_power_dbm": power, "pump_periods_total": periods,
        "mode": mode, "settle_periods_measured_2026_08_14": 1050,
    })
    (path / "result.json").write_text(
        json.dumps(PHASE5._json_safe(record), indent=2), encoding="utf-8"
    )


def run_cold(device: str, power: float, root: Path) -> dict[str, Any]:
    path = root / device / f"cold_{_tag(power)}"
    if _done(path):
        return {"device": device, "mode": "cold", "pump_power_dbm": power, "status": "SKIP"}
    started = time.perf_counter()
    try:
        row, _, _, _, trace_t, trace_v = PHASE5._run_point(
            _spec(device), _current_for(device, power),
            dt_norm=0.01, tmax_norm=_tmax_norm(device, COLD_PERIODS),
            signal_current_a=0.0, pump_off_output=None,
        )
        _persist(path, device, power, COLD_PERIODS, row, trace_t, trace_v, "cold")
        status = "OK"
    except Exception as exc:  # noqa: BLE001 - one bad point must not end the campaign
        status = f"FAILED {exc!r}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "result.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    return {"device": device, "mode": "cold", "pump_power_dbm": power,
            "status": status, "wall_s": round(time.perf_counter() - started, 1)}


def run_chain(device: str, powers: np.ndarray, direction: str, root: Path) -> list[dict[str, Any]]:
    """Warm-started sweep; each point seeds the next."""
    results: list[dict[str, Any]] = []
    state: np.ndarray | None = None
    for index, power in enumerate(powers):
        path = root / device / f"chain_{direction}_{index:02d}_{_tag(float(power))}"
        periods = CHAIN_FIRST_PERIODS if index == 0 else CHAIN_STEP_PERIODS
        started = time.perf_counter()
        try:
            row, _, _, final_q, trace_t, trace_v = PHASE5._run_point(
                _spec(device), _current_for(device, float(power)),
                dt_norm=0.01, tmax_norm=_tmax_norm(device, periods),
                signal_current_a=0.0, pump_off_output=None,
                initial_state=state,
            )
            state = final_q
            _persist(path, device, float(power), periods, row, trace_t, trace_v,
                     f"chain_{direction}")
            status = "OK"
        except Exception as exc:  # noqa: BLE001
            status = f"FAILED {exc!r}"
            state = None  # a broken link must not poison the rest of the chain
            path.mkdir(parents=True, exist_ok=True)
            (path / "result.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        results.append({"device": device, "mode": f"chain_{direction}",
                        "pump_power_dbm": float(power), "status": status,
                        "wall_s": round(time.perf_counter() - started, 1)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "chaos" / "phaseB_jc_remeasure")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--devices", default="jc_jtwpa,jc_fqjtwpa")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    devices = [name.strip() for name in args.devices.split(",") if name.strip()]

    cold_jobs: list[tuple[str, float]] = []
    chain_jobs: list[tuple[str, np.ndarray, str]] = []
    for device in devices:
        (cold_lo, cold_hi), (chain_lo, chain_hi), directions = DEVICES[device]
        for power in np.linspace(cold_lo, cold_hi, COLD_POINTS):
            cold_jobs.append((device, float(power)))
        for direction in directions:
            powers = np.linspace(chain_lo, chain_hi, CHAIN_POINTS)
            chain_jobs.append((device, powers if direction == "up" else powers[::-1], direction))

    pending_cold = [job for job in cold_jobs
                    if not _done(root / job[0] / f"cold_{_tag(job[1])}")]
    unit_s = {"jc_jtwpa": 324.0, "jc_fqjtwpa": 288.0}  # measured, per 600 periods
    cold_s = sum(unit_s[d] * COLD_PERIODS / 600.0 for d, _ in pending_cold)
    chain_s = sum(
        unit_s[d] * (CHAIN_FIRST_PERIODS + (len(p) - 1) * CHAIN_STEP_PERIODS) / 600.0
        for d, p, _ in chain_jobs
    )
    print(f"cold  : {len(pending_cold)} points, {cold_s / 3600.0:.2f} h serial, "
          f"{cold_s / 3600.0 / max(args.workers, 1):.2f} h at {args.workers} workers")
    print(f"chains: {len(chain_jobs)} chains x {CHAIN_POINTS} points, "
          f"{chain_s / 3600.0:.2f} h serial, "
          f"{chain_s / 3600.0 / min(len(chain_jobs), args.workers):.2f} h "
          f"at {min(len(chain_jobs), args.workers)} parallel chains")
    print(f"total : ~{cold_s / 3600.0 / max(args.workers, 1) + chain_s / 3600.0 / min(len(chain_jobs), args.workers):.2f} h")
    if args.dry_run:
        return 0

    log = root / "remeasure.log"
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()

    def note(entry: dict[str, Any]) -> None:
        manifest.append(entry)
        line = (f"{entry['device']:<12} {entry['mode']:<12} "
                f"{entry['pump_power_dbm']:9.4f}  {entry['status']}")
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
        try:
            print(line, flush=True)
        except OSError:
            pass
        with (root / "progress.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)

    # Chains first: they are sequential and set the wall-clock floor, so start
    # them before the fully parallel cold pass rather than after it.
    with ThreadPoolExecutor(max_workers=min(len(chain_jobs), args.workers)) as pool:
        futures = {pool.submit(run_chain, d, p, direction, root): d
                   for d, p, direction in chain_jobs}
        for future in as_completed(futures):
            for entry in future.result():
                note(entry)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cold, d, p, root): (d, p) for d, p in pending_cold}
        for future in as_completed(futures):
            note(future.result())

    (root / "manifest.json").write_text(
        json.dumps({"points": manifest, "wall_time_s": time.perf_counter() - started,
                    "cold_periods": COLD_PERIODS,
                    "chain_first_periods": CHAIN_FIRST_PERIODS,
                    "chain_step_periods": CHAIN_STEP_PERIODS}, indent=2),
        encoding="utf-8",
    )
    print(f"complete in {(time.perf_counter() - started) / 3600.0:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
