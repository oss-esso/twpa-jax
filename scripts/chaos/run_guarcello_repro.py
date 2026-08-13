#!/usr/bin/env python3
"""Driver for the supplied Guarcello FDTD reproduction and convention audit."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def gain_vs_off_db(pumped_signal_v: float, unpumped_signal_v: float) -> float:
    """Return gain relative to the identical pump-off signal response."""
    if pumped_signal_v <= 0.0 or unpumped_signal_v <= 0.0:
        raise ValueError("signal amplitudes must be positive")
    return float(20.0 * np.log10(pumped_signal_v / unpumped_signal_v))


def add_gain_vs_off(rows: list[dict[str, Any]], unpumped_signal_v: float) -> list[dict[str, Any]]:
    """Add pump-off-normalized gain while preserving absolute ``gain_db``."""
    result = []
    for row in rows:
        normalized = gain_vs_off_db(float(row["signal_vout_peak_v"]), unpumped_signal_v)
        wideband = float(row.get("gain_wideband_db", float("nan")))
        narrowband = float(row.get("gain_db", float("nan")))
        result.append({
            **row,
            "gain_vs_off_db": normalized,
            "gain_wideband_vs_off_db": normalized + wideband - narrowband,
        })
    return result


def load_fdtd(path: Path):
    spec = importlib.util.spec_from_file_location("guarcello_jtwpa_fdtd", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FDTD module from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # source module is being executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_point(module: Any, *, power_convention: str, port_update: str,
              pump_dbm: float = -54.0, quick: bool = False,
              tmax_norm: float | None = None,
              dt_norm: float = 0.01, signal_ghz: float = 6.42,
              pump_ghz: float = 7.0, record_stride: int = 20) -> dict[str, Any]:
    device = module.Device()
    cfg = module.RunConfig(
        dt_norm=dt_norm,
        tmax_norm=20.0 if quick else (20_000.0 if tmax_norm is None else tmax_norm),
        pump_dbm=pump_dbm, power_convention=power_convention,
        port_update=port_update, signal_ghz=signal_ghz, pump_ghz=pump_ghz,
        record_stride=record_stride,
    )
    # The supplied solver is warmed by callers for a fair timing comparison.
    output, runtime = module.simulate(device, cfg)
    analysis = module.analyze(device, cfg, output)
    return {"power_convention": power_convention, "port_update": port_update,
            "pump_dbm": pump_dbm, "quick": quick, "runtime_s": runtime,
            "analysis": analysis}


def run_estimator_audit(module: Any, output: Path) -> dict[str, Any]:
    """Compare old and multi-tone estimates across three record lengths."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    t = np.arange(4097, dtype=float) * 2.0e-11
    signal_hz = 6.557e9
    pump_hz = 7.271e9
    expected = 1.0e-6
    voltage = (
        expected * np.sin(2.0 * np.pi * signal_hz * t + 0.37)
        + 10.0 ** (43.0 / 20.0) * expected
        * np.sin(2.0 * np.pi * pump_hz * t + 1.11)
    )
    old_amplitude = module.exact_tone_amplitude(t, voltage, signal_hz)
    new_amplitude = module.multitone_amplitude(t, voltage, signal_hz, pump_hz)
    synthetic = {
        "old_relative_error": abs(old_amplitude - expected) / expected,
        "multitone_relative_error": abs(new_amplitude - expected) / expected,
        "pump_to_signal_db": 43.0,
    }
    for tmax_norm in (10_000.0, 20_000.0, 40_000.0):
        device = module.Device()
        cfg = module.RunConfig(
            tmax_norm=tmax_norm, pump_dbm=-57.0, signal_ghz=6.760,
            power_convention="50ohm", port_update="stable",
        )
        raw, _ = module.simulate(device, cfg)
        analysis = module.analyze(device, cfg, raw)
        t, v = raw[0], raw[1]
        start = int(round(cfg.transient_fraction * len(t)))
        old_amp = module.exact_tone_amplitude(t[start:], v[start:], 6.760e9)
        vin = module.dbm_to_vpk(cfg.signal_dbm, cfg.power_convention, device.ri_ohm)
        rows.append({
            "tmax_norm": tmax_norm,
            "old_gain_db": 20.0 * np.log10(old_amp / vin),
            "multitone_gain_db": analysis["gain_db"],
        })
        output.write_text(json.dumps({"status": "IN_PROGRESS", "synthetic": synthetic, "rows": rows}, indent=2), encoding="utf-8")
    old_values = [row["old_gain_db"] for row in rows]
    new_values = [row["multitone_gain_db"] for row in rows]
    payload = {
        "status": "COMPLETE", "point": {"signal_ghz": 6.760, "pump_dbm": -57.0},
        "synthetic": synthetic,
        "rows": rows,
        "old_spread_db": max(old_values) - min(old_values),
        "multitone_spread_db": max(new_values) - min(new_values),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _nearest_crossings(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return upward derivative samples without crossing-time interpolation."""
    if len(v) < 3:
        return np.empty(0)
    dv = np.gradient(v, t)
    indices = np.flatnonzero(v[:-1] <= 0.0)
    indices = indices[v[indices + 1] > 0.0]
    return dv[indices]


def run_sampling_floor_audit(module: Any, output: Path) -> dict[str, Any]:
    """Compare recorded-trace Poincare sampling methods at two pump powers."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pump_dbm in (-70.0, -54.5):
        for record_stride, method in ((20, "stride20_nearest"),
                                      (1, "stride1_nearest"),
                                      (20, "stride20_linear")):
            device = module.Device()
            cfg = module.RunConfig(
                pump_dbm=pump_dbm, power_convention="50ohm", port_update="stable",
                record_stride=record_stride,
            )
            out, runtime = module.simulate(device, cfg)
            start = int(round(cfg.transient_fraction * len(out[0])))
            t, voltage = out[0][start:], out[1][start:]
            if method.endswith("nearest"):
                values = _nearest_crossings(t, voltage)
            else:
                values = module.poincare_crossings(t, voltage)
                values = values[values > 0.0]
            values = values / device.omega_plasma * 1e3
            row = {
                "pump_dbm": pump_dbm,
                "method": method,
                "record_stride": record_stride,
                "sigma_upward": float(np.std(values)),
                "branch_mean": float(np.mean(values)),
                "crossing_count": int(values.size),
                "runtime_s": runtime,
            }
            rows.append(row)
            output.write_text(json.dumps({"status": "IN_PROGRESS", "rows": rows}, indent=2), encoding="utf-8")
    payload = {"status": "COMPLETE", "rows": rows}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_transition_scan(module: Any, output: Path) -> dict[str, Any]:
    """Measure half-harmonic lines across the requested transition bracket."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pump_dbm in np.linspace(-54.0, -53.4, 13):
        result = run_point(
            module, power_convention="50ohm", port_update="stable",
            pump_dbm=float(pump_dbm), record_stride=20,
        )
        device = module.Device()
        cfg = module.RunConfig(pump_dbm=float(pump_dbm), power_convention="50ohm")
        raw, _ = module.simulate(device, cfg)
        start = int(round(cfg.transient_fraction * len(raw[0])))
        frequencies, spectrum = module.spectrum_dbm(
            raw[0][start:], raw[1][start:], device.rl_ohm,
        )
        def peak(center: float, width: float = 0.08) -> float:
            mask = np.abs(frequencies - center) <= width
            return float(np.max(spectrum[mask]))
        pump_level = peak(7.0)
        floor_mask = (frequencies >= 3.0) & (frequencies <= 4.0)
        upward = module.poincare_crossings(raw[0][start:], raw[1][start:])
        upward = upward[upward > 0.0] / device.omega_plasma * 1e3
        sorted_upward = np.sort(upward)
        cluster_scale = max(float(np.median(upward)), 1.0e-30) if upward.size else 1.0e-30
        clusters = 1 + int(np.count_nonzero(np.diff(sorted_upward) > 0.03 * cluster_scale)) if upward.size else 0
        row = {
            "pump_dbm": float(pump_dbm),
            "fp_over_2_minus_pump_db": peak(3.5) - pump_level,
            "3fp_over_2_minus_pump_db": peak(10.5) - pump_level,
            "floor_3_to_4ghz_minus_pump_db": float(np.median(spectrum[floor_mask]) - pump_level),
            "directional_clusters_upward": clusters,
            "sigma_upward": result["analysis"]["poincare_std_mV_per_tnorm_upward"],
            "branch_mean": result["analysis"].get("branch_mean"),
        }
        rows.append(row)
        output.write_text(json.dumps({"status": "IN_PROGRESS", "rows": rows}, indent=2), encoding="utf-8")
    payload = {"status": "COMPLETE", "rows": rows}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_pump_off_reference(module: Any, *, power_convention: str = "50ohm",
                           port_update: str = "stable", quick: bool = False) -> dict[str, Any]:
    """Measure the identical signal response with the pump amplitude set to zero."""
    return run_point(
        module, power_convention=power_convention, port_update=port_update,
        pump_dbm=-np.inf, quick=quick,
    )


def convention_audit(module: Any, *, quick: bool = True) -> dict[str, Any]:
    rows = [run_point(module, power_convention=power, port_update=update, quick=quick)
            for power, update in (("paper", "stable"), ("50ohm", "stable"),
                                  ("paper", "paper-centered"))]
    baseline = next(row["analysis"] for row in rows
                    if row["power_convention"] == "50ohm"
                    and row["port_update"] == "stable")
    for row in rows:
        row["gain_delta_db_vs_50ohm_stable"] = float(
            row["analysis"]["gain_db"] - baseline["gain_db"])
    return {"policy": "50ohm/stable is the reproduction gate; paper is retained as the overdrive control",
            "rows": rows}


def run_figure_sweep(module: Any, *, figure: str, output: Path, num: int,
                     workers: int = 1, quick: bool = False,
                     power_convention: str = "50ohm",
                     signal_sweep_pump_dbm: float = -54.5) -> dict[str, Any]:
    """Run one Figure-2 analogue through the supplied solver unchanged."""
    device = module.Device()
    base = module.RunConfig(
        tmax_norm=20.0 if quick else 20_000.0,
        power_convention=power_convention, port_update="stable",
        signal_ghz=6.42, pump_ghz=7.0, signal_dbm=-100.0, bias_ua=0.0,
    )
    if figure == "2a":
        values = np.linspace(-70.0, -45.0, num)
        configs = [module.RunConfig(**{**asdict(base), "pump_dbm": float(value)}) for value in values]
        name = "pump_dbm"
    elif figure == "2b":
        values = np.linspace(4.0, 10.0, num)
        configs = [module.RunConfig(**{**asdict(base), "signal_ghz": float(value), "pump_dbm": signal_sweep_pump_dbm}) for value in values]
        name = "signal_ghz"
    else:
        raise ValueError("figure must be 2a or 2b")
    warm = module.RunConfig(**{**asdict(base), "tmax_norm": 0.05, "record_stride": 1})
    module.simulate(device, warm)
    rows = module.run_sweep(device, configs, values, name, output, workers, False)
    return {"figure": figure, "power_convention": power_convention, "port_update": "stable",
            "points": len(rows), "output": str(output),
            "signal_sweep_pump_dbm": signal_sweep_pump_dbm}


def run_boundary_audit(module: Any, output: Path, *, quick: bool = False) -> dict[str, Any]:
    """Compare stable and literal centered port updates at three pump powers."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pump_dbm in (-60.0, -57.0, -55.0):
        measurements = {}
        for update in ("stable", "paper-centered"):
            result = run_point(
                module, power_convention="50ohm", port_update=update,
                pump_dbm=pump_dbm, quick=quick,
            )
            measurements[update] = result
        stable_gain = measurements["stable"]["analysis"]["gain_db"]
        centered_gain = measurements["paper-centered"]["analysis"]["gain_db"]
        rows.append({"pump_dbm": pump_dbm, "stable_gain_db": stable_gain,
                     "paper_centered_gain_db": centered_gain,
                     "gain_difference_db": centered_gain - stable_gain,
                     "stable": measurements["stable"],
                     "paper_centered": measurements["paper-centered"]})
        output.write_text(json.dumps({"status": "IN_PROGRESS", "rows": rows}, indent=2), encoding="utf-8")
    payload = {"status": "COMPLETE", "rows": rows,
               "interpretation": "differences are diagnostic only; centered update is not preferred without timestep convergence"}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_centered_timestep_audit(module: Any, output: Path) -> dict[str, Any]:
    """Retry literal centered updates at smaller timesteps after divergence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pump_dbm in (-60.0, -57.0, -55.0):
        for dt_norm in (0.01, 0.005, 0.0025):
            result = run_point(
                module, power_convention="50ohm", port_update="paper-centered",
                pump_dbm=pump_dbm, dt_norm=dt_norm,
            )
            gain = result["analysis"]["gain_db"]
            rows.append({"pump_dbm": pump_dbm, "dt_norm": dt_norm,
                         "gain_db": gain, "finite": bool(np.isfinite(gain)),
                         "result": result})
            output.write_text(json.dumps({"status": "IN_PROGRESS", "rows": rows}, indent=2), encoding="utf-8")
            if np.isfinite(gain):
                break
    payload = {"status": "COMPLETE", "rows": rows,
               "interpretation": "finite centered gains require timestep-convergence evidence before physical interpretation"}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdtd", type=Path, default=root / "docs" / "development" / "chaos_papers" / "guarcello_jtwpa_fdtd.py")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "chaos" / "phase2" / "convention_audit.json")
    parser.add_argument("--full", action="store_true", help="use paper-scale 2,000,000-step points")
    parser.add_argument("--sweep", choices=("2a", "2b"), default=None,
                        help="also run a Figure-2 pump or signal sweep")
    parser.add_argument("--num", type=int, default=101)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--power-convention", choices=("paper", "50ohm"), default="50ohm")
    parser.add_argument("--signal-sweep-pump-dbm", type=float, default=-54.5)
    parser.add_argument("--boundary-audit", action="store_true")
    parser.add_argument("--centered-timestep-audit", action="store_true")
    parser.add_argument("--reference", action="store_true",
                        help="run one pump-off signal reference point")
    parser.add_argument("--estimator-audit", action="store_true")
    parser.add_argument("--sampling-floor-audit", action="store_true")
    parser.add_argument("--transition-scan", action="store_true")
    args = parser.parse_args(argv)
    module = load_fdtd(args.fdtd)
    started = time.perf_counter()
    if args.transition_scan:
        result = run_transition_scan(module, args.output)
    elif args.sampling_floor_audit:
        result = run_sampling_floor_audit(module, args.output)
    elif args.estimator_audit:
        result = run_estimator_audit(module, args.output)
    elif args.reference:
        result = run_pump_off_reference(
            module, power_convention=args.power_convention, quick=not args.full,
        )
    elif args.centered_timestep_audit:
        result = run_centered_timestep_audit(module, args.output)
    elif args.boundary_audit:
        result = run_boundary_audit(module, args.output, quick=not args.full)
    elif args.sweep:
        result = run_figure_sweep(
            module, figure=args.sweep,
            output=args.output.with_suffix(""), num=args.num,
            workers=args.workers, quick=not args.full,
            power_convention=args.power_convention,
            signal_sweep_pump_dbm=args.signal_sweep_pump_dbm,
        )
    else:
        result = convention_audit(module, quick=not args.full)
    result["elapsed_s"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    point_count = int(result.get("points", len(result.get("rows", []))))
    print(json.dumps({"output": str(args.output), "points": point_count,
                      "elapsed_s": result["elapsed_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
