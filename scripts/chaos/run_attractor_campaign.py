#!/usr/bin/env python3
"""Run both-direction attractor continuation on an H1 HB checkpoint.

The campaign is intentionally resumable and evidence-oriented: every RCSJ
resistance setting gets its own circuit variant and every continuation point
stores the voltage trace, Poincare values, FT row, decay gate, and verdict.
The solver checkpoint remains authoritative; this script never creates an HB
orbit or enables the dormant period-doubled ansatz.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.chaos.attractor_classify import (
    Classification,
    H1AttractorAdapter,
    classify_trace,
    classify_sweep,
    envelope_decay,
    fourier_map,
    largest_lyapunov_map,
    make_h1_attractor_adapter,
    poincare_crossings,
    poincare_crossing_branches,
)
from twpa_solver.core import load_circuit, save_circuit, stamp_rcsj_shunt


DEFAULT_RATIOS = (math.inf, 1.0e6, 1.0e4, 1.0e2, 1.0)


def ratio_slug(ratio: float) -> str:
    return "inf" if math.isinf(ratio) else f"r{ratio:.0e}".replace("+", "")


def dbm_to_peak_current(power_dbm: float, resistance_ohm: float = 50.0) -> float:
    """Convert a 50-ohm sinusoidal power to the H1 Norton peak current."""
    if resistance_ohm <= 0.0:
        raise ValueError("resistance_ohm must be positive")
    power_w = 1.0e-3 * 10.0 ** (float(power_dbm) / 10.0)
    return float(math.sqrt(2.0 * power_w / resistance_ohm))


def parse_ratios(text: str) -> tuple[float, ...]:
    values = []
    for item in text.split(","):
        item = item.strip().lower()
        if item in {"inf", "infinity"}:
            values.append(math.inf)
        else:
            value = float(item)
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"invalid resistance ratio {item!r}")
            values.append(value)
    if not values:
        raise ValueError("at least one resistance ratio is required")
    return tuple(values)


def junction_capacitance_from_design(circuit_dir: Path) -> float:
    arrays = circuit_dir / "arrays.npz"
    if not arrays.exists() or "Cj" not in np.load(arrays, allow_pickle=True).files:
        arrays = circuit_dir / "ipm_arrays.npz"
    if not arrays.exists():
        raise FileNotFoundError(
            f"{circuit_dir} has no arrays.npz/ipm_arrays.npz; pass --junction-capacitance-f explicitly"
        )
    data = np.load(arrays, allow_pickle=True)
    if "Cj" not in data.files:
        raise ValueError(f"{arrays} does not contain Cj")
    values = np.asarray(data["Cj"], dtype=float).reshape(-1)
    if values.size == 0 or np.any(values <= 0.0):
        raise ValueError("design Cj must contain positive values")
    return float(np.median(values))


def make_variant(
    base_dir: Path, variant_root: Path, ratio: float, *, junction_capacitance_f: float,
    pump_frequency_hz: float, delta_ev: float,
) -> tuple[Path, dict[str, Any]]:
    base = load_circuit(base_dir)
    damped, params = stamp_rcsj_shunt(
        base, ratio, junction_capacitance_f=junction_capacitance_f,
        delta_ev=delta_ev, pump_frequency_hz=pump_frequency_hz,
    )
    summary = {
        "resistance_ratio": float(ratio),
        "junction_capacitance_f": float(junction_capacitance_f),
        "delta_ev": float(delta_ev),
        "resistance_ohm_median": float(np.median(params.resistance_ohm)),
        "damping_per_pump_period_median": float(np.median(params.damping_per_pump_period)),
    }
    if math.isinf(ratio):
        return base_dir, summary
    variant = variant_root / ratio_slug(ratio)
    if not (variant / "C.npz").exists():
        variant.mkdir(parents=True, exist_ok=True)
        save_circuit(damped, variant)
    return variant, summary


def _point_record(
    *, power_dbm: float, direction: str, adapter: H1AttractorAdapter,
    state: np.ndarray, theta: np.ndarray, voltage: np.ndarray, output_dir: Path,
    decay_limit: float, lyapunov_periods: int,
) -> dict[str, Any]:
    # H1 traces use pump phase theta, not seconds: one drive period is 2*pi.
    classification: Classification = classify_trace(
        theta, voltage * 1.0e3, drive_hz=1.0 / (2.0 * math.pi)
    )
    decay = envelope_decay(theta, voltage, period=2.0 * math.pi)
    ft = fourier_map(theta, voltage, fmax_hz=10.0)
    branches = poincare_crossing_branches(theta, voltage)
    poincare = branches["upward"]
    lyapunov = None
    if lyapunov_periods > 0:
        saved_state = adapter.state.copy()
        saved_current = adapter.current_a

        def one_map(value: np.ndarray) -> np.ndarray:
            endpoint, _theta, _voltage, _states = adapter.integrate(saved_current, value)
            adapter.state = saved_state.copy()
            adapter.current_a = saved_current
            return endpoint

        try:
            lyapunov = largest_lyapunov_map(
                one_map, state, steps=lyapunov_periods,
            )
        except (FloatingPointError, RuntimeError, ValueError):
            lyapunov = float("nan")
        adapter.state = saved_state
        adapter.current_a = saved_current
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{direction}_{power_dbm:+.6f}".replace("-", "m").replace("+", "p").replace(".", "p")
    np.savez_compressed(
        output_dir / f"{slug}.npz", theta=theta, output_voltage_v=voltage,
        state_norm=np.asarray([np.linalg.norm(state)]),
        fourier_frequency_hz=ft["frequency_hz"], fourier_amplitude=ft["amplitude"],
        poincare=poincare, poincare_upward=branches["upward"],
        poincare_downward=branches["downward"], poincare_both=branches["both"],
    )
    result = classification.as_dict()
    result.update({
        "power_dbm": float(power_dbm), "direction": direction,
        "target_current_a": adapter.current_for_power_dbm(power_dbm),
        "state_norm": float(np.linalg.norm(state)),
        "transient_decay": float(decay),
        "decay_gate": bool(decay <= decay_limit),
        "largest_lyapunov": lyapunov,
        "trace": str(output_dir / f"{slug}.npz"),
    })
    return result


def run_ratio_campaign(
    circuit_dir: Path, checkpoint: Path, outdir: Path, *, ratio: float,
    powers_dbm: np.ndarray, junction_capacitance_f: float,
    freq_ghz: float, pump_port: int, out_port: int, periods: int,
    decay_limit: float, delta_ev: float, lyapunov_periods: int,
    ramp_periods: int, max_step: float, reference_power_dbm: float | None,
) -> dict[str, Any]:
    variant, damping = make_variant(
        circuit_dir, outdir / "variants", ratio,
        junction_capacitance_f=junction_capacitance_f,
        pump_frequency_hz=freq_ghz * 1e9, delta_ev=delta_ev,
    )
    adapter = make_h1_attractor_adapter(
        variant, checkpoint, freq_ghz=freq_ghz, pump_port=pump_port,
        out_port=out_port, periods=periods, ramp_periods=ramp_periods,
        reference_power_dbm=reference_power_dbm,
    )
    adapter.max_step = float(max_step)
    ratio_dir = outdir / ratio_slug(ratio)
    state = adapter.state.copy()
    records: dict[str, list[dict[str, Any]]] = {}
    for direction, sequence in (("up", powers_dbm), ("down", powers_dbm[::-1])):
        rows = []
        for power in sequence:
            target = adapter.current_for_power_dbm(float(power))
            state, theta, voltage, _states = adapter.integrate(target, state)
            rows.append(_point_record(
                power_dbm=float(power), direction=direction, adapter=adapter,
                state=state, theta=theta, voltage=voltage,
                output_dir=ratio_dir / "traces", decay_limit=decay_limit,
                lyapunov_periods=lyapunov_periods,
            ))
        records[direction] = rows
        sigmas = np.asarray([row["sigma_vprime_ps"] for row in rows], dtype=float)
        controls = np.asarray([row["power_dbm"] for row in rows], dtype=float)
        stable = float(np.min(sigmas))
        verdicts = classify_sweep(controls, sigmas)
        for row, verdict in zip(rows, verdicts):
            row["sigma_deep_stable"] = stable
            row["sigma_ratio"] = float(row["sigma_vprime_ps"] / stable)
            row["ratio_threshold"] = 40.0
            if verdict == "NO_BIFURCATION_FOUND":
                row["verdict"] = verdict
                row["reason"] = "within-sweep ratio/shape guard rejected a scalar sigma rise"
    return {"resistance": damping, "variant": str(variant), "directions": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "chaos" / "phase3")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--power-start-dbm", type=float, default=-24.5)
    parser.add_argument("--power-stop-dbm", type=float, default=-22.0)
    parser.add_argument("--power-points", type=int, default=11)
    parser.add_argument("--resistance-ratios", default=", ".join("inf" if math.isinf(x) else str(x) for x in DEFAULT_RATIOS))
    parser.add_argument("--junction-capacitance-f", type=float, default=None)
    parser.add_argument("--delta-ev", type=float, default=180e-6)
    parser.add_argument("--periods", type=int, default=40)
    parser.add_argument("--ramp-periods", type=int, default=10)
    parser.add_argument("--reference-power-dbm", type=float, default=None,
                        help="reference pump power when the HB report omits it")
    parser.add_argument("--max-step", type=float, default=0.5,
                        help="maximum implicit-trapezoid phase step for timestep studies")
    parser.add_argument("--decay-limit", type=float, default=1e-5)
    parser.add_argument("--lyapunov-periods", type=int, default=4,
                        help="sampled-period map steps for the largest LE; 0 disables it")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.exists():
        payload = {"status": "BLOCKED_MISSING_INPUT", "required_checkpoint": str(args.checkpoint),
                   "message": "Phase-3 attractor continuation requires an authoritative converged HB checkpoint."}
        (args.outdir / "campaign_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2
    if args.power_points < 2:
        raise SystemExit("--power-points must be at least 2")
    cj = args.junction_capacitance_f
    if cj is None:
        cj = junction_capacitance_from_design(args.circuit_dir)
    powers = np.linspace(args.power_start_dbm, args.power_stop_dbm, args.power_points)
    results = {}
    for ratio in parse_ratios(args.resistance_ratios):
        results[ratio_slug(ratio)] = run_ratio_campaign(
            args.circuit_dir, args.checkpoint, args.outdir, ratio=ratio,
            powers_dbm=powers, junction_capacitance_f=cj,
            freq_ghz=args.freq_ghz, pump_port=args.pump_port,
            out_port=args.out_port, periods=args.periods,
            decay_limit=args.decay_limit, delta_ev=args.delta_ev,
            lyapunov_periods=args.lyapunov_periods, ramp_periods=args.ramp_periods,
            max_step=args.max_step, reference_power_dbm=args.reference_power_dbm,
        )
    payload = {"status": "COMPLETE", "controls_dbm": powers.tolist(),
               "resistance_ratios": list(results),
               "integrator": {"method": "implicit_trapezoid", "max_step": args.max_step},
               "results": results}
    (args.outdir / "campaign_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "ratios": list(results),
                      "output": str(args.outdir / "campaign_summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
