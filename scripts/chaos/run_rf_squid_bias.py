#!/usr/bin/env python3
"""Run the RF-SQUID DC-flux/bias transfer axis with carry-forward states.

The power-scan campaign starts from a validated harmonic-balance checkpoint.
For the RF-SQUID bias axis, each external-flux value is a distinct physical
problem, so this driver starts from the exact zero-pump equilibrium of the
shifted branch law and ramps the same reference pump current.  States are
then carried forward along each bias direction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.chaos.attractor_classify import (  # noqa: E402
    H1AttractorAdapter,
    classify_trace,
    envelope_decay,
    fourier_map,
    largest_lyapunov_map,
    poincare_crossings,
    poincare_crossing_branches,
)
from twpa_solver.core.constants import PHI0_REDUCED  # noqa: E402


def _load_h1_module() -> Any:
    source = ROOT / "scripts" / "h1_transient_branch_transfer.py"
    spec = importlib.util.spec_from_file_location("h1_transient_branch_transfer_bias", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load transient solver from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolved_parameters(circuit_dir: Path) -> dict[str, float]:
    path = circuit_dir / "design_resolved.json"
    if not path.exists():
        raise FileNotFoundError(f"missing compiled design metadata: {path}")
    values = json.loads(path.read_text(encoding="utf-8")).get("parameters", {})
    required = ("Ic", "Lm")
    if any(key not in values for key in required):
        raise ValueError(f"{path} must contain {required}")
    return {key: float(value) for key, value in values.items() if isinstance(value, (int, float))}


def external_phase(circuit_dir: Path, fraction: float) -> float:
    """Solve the RF-SQUID screening relation for the internal DC phase."""
    params = _resolved_parameters(circuit_dir)
    beta_l = params["Lm"] * params["Ic"] / PHI0_REDUCED
    external = 2.0 * math.pi * float(fraction)
    return float(brentq(
        lambda phase: phase - external + beta_l * math.sin(phase),
        external - beta_l - 0.5, external + beta_l + 0.5,
    ))


def screening_prediction(circuit_dir: Path, fraction: float) -> dict[str, float]:
    """Return Guarcello Eq. (4) predictions for both RF-SQUID inductance choices."""
    params = _resolved_parameters(circuit_dir)
    phase = external_phase(circuit_dir, fraction)
    beta_l_lm = params["Lm"] * params["Ic"] / PHI0_REDUCED
    beta_l_total = (params["Lm"] + params.get("Lpar", 0.0)) * params["Ic"] / PHI0_REDUCED
    return {
        "external_flux_fraction": float(fraction),
        "internal_dc_phase_rad": phase,
        "beta_l_lm": beta_l_lm,
        "beta_l_lm_plus_lpar": beta_l_total,
        "guarcello_beta_lm": 0.5 * beta_l_lm * math.sin(phase),
        "guarcello_gamma_lm": beta_l_lm * math.cos(phase) / 6.0,
        "guarcello_beta_lm_plus_lpar": 0.5 * beta_l_total * math.sin(phase),
        "guarcello_gamma_lm_plus_lpar": beta_l_total * math.cos(phase) / 6.0,
    }


def _reference_current(checkpoint: Path) -> tuple[float, float]:
    report = json.loads((checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    current = float(metadata["pump_current_a"])
    power = float(metadata.get("pump_power_dbm_requested", metadata.get("pump_power_dbm")))
    return current, power


def _adapter(
    module: Any, circuit_dir: Path, fraction: float, *, freq_ghz: float,
    pump_port: int, out_port: int, target_current: float, target_power: float,
    periods: int, ramp_periods: int, initial_state: np.ndarray | None = None,
    initial_current: float = 0.0,
) -> H1AttractorAdapter:
    dc_flux = module.dc_flux_from_external_fraction(circuit_dir, fraction, PHI0_REDUCED)
    system = module.build_system(circuit_dir, freq_ghz, pump_port, dc_flux)
    if initial_state is None:
        q0 = np.zeros(system.n, dtype=float)
        p0 = np.zeros(system.n, dtype=float)
        system.project_algebraic_state(q0, p0, system.source(0.0, 0.0, 0.0, 0.0))
        state = system.pack(q0, p0)
    else:
        state = np.asarray(initial_state, dtype=float).copy()
    return H1AttractorAdapter(
        module=module, system=system, state=state, current_a=initial_current,
        out_port=out_port, periods=periods, reference_current_a=target_current,
        reference_power_dbm=target_power, ramp_periods=ramp_periods,
    )


def _largest_lyapunov_at_target(
    adapter: H1AttractorAdapter, state: np.ndarray, target_current: float,
    periods: int,
) -> float | None:
    """Estimate the largest sampled-period exponent without advancing the branch."""
    if periods <= 0:
        return None
    saved_state = adapter.state.copy()
    saved_current = float(adapter.current_a)

    def one_map(value: np.ndarray) -> np.ndarray:
        endpoint, _theta, _voltage, _states = adapter.integrate(target_current, value)
        adapter.state = saved_state.copy()
        adapter.current_a = saved_current
        return endpoint

    try:
        return float(largest_lyapunov_map(one_map, state, steps=periods))
    except (FloatingPointError, RuntimeError, ValueError):
        return float("nan")
    finally:
        adapter.state = saved_state
        adapter.current_a = saved_current


def _run_direction(
    module: Any, circuit_dir: Path, fractions: np.ndarray, *, direction: str,
    freq_ghz: float, pump_port: int, out_port: int, target_current: float,
    target_power: float, periods: int, ramp_periods: int, decay_limit: float,
    output_dir: Path, lyapunov_periods: int,
) -> list[dict[str, Any]]:
    sequence = fractions if direction == "up" else fractions[::-1]
    rows: list[dict[str, Any]] = []
    carry_state: np.ndarray | None = None
    carry_current = 0.0
    for fraction in sequence:
        adapter = _adapter(
            module, circuit_dir, float(fraction), freq_ghz=freq_ghz,
            pump_port=pump_port, out_port=out_port, target_current=target_current,
            target_power=target_power, periods=periods, ramp_periods=ramp_periods,
            initial_state=carry_state, initial_current=carry_current,
        )
        # The converged state is carried to the next external-flux value.  At
        # the first point only, the pump is ramped from the zero-pump
        # equilibrium; subsequent points remain on the same pump branch.
        state, theta, voltage, _states = adapter.integrate(target_current, carry_state)
        carry_state = state.copy()
        carry_current = target_current
        classification = classify_trace(theta, voltage * 1e3, drive_hz=1.0 / (2.0 * math.pi))
        decay = envelope_decay(theta, voltage, period=2.0 * math.pi)
        ft = fourier_map(theta, voltage, fmax_hz=10.0)
        lyapunov = _largest_lyapunov_at_target(
            adapter, state, target_current, lyapunov_periods,
        )
        slug = f"{direction}_{float(fraction):.6f}".replace(".", "p")
        path = output_dir / "traces" / f"{slug}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, theta=theta, output_voltage_v=voltage,
            fourier_frequency_hz=ft["frequency_hz"], fourier_amplitude=ft["amplitude"],
            poincare=poincare_crossings(theta, voltage),
            poincare_upward=poincare_crossing_branches(theta, voltage)["upward"],
            poincare_downward=poincare_crossing_branches(theta, voltage)["downward"],
            poincare_both=poincare_crossing_branches(theta, voltage)["both"],
        )
        row = classification.as_dict()
        row.update({
            "external_flux_fraction": float(fraction), "direction": direction,
            "pump_power_dbm": target_power, "target_current_a": target_current,
            "state_norm": float(np.linalg.norm(state)), "transient_decay": float(decay),
            "decay_gate": bool(decay <= decay_limit), "trace": str(path),
            "largest_lyapunov": lyapunov,
            "prediction": screening_prediction(circuit_dir, float(fraction)),
        })
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "outputs" / "chaos" / "phase3" / "rf_compiled")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "chaos" / "phase3" / "rf_bias")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=1)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--bias-start", type=float, default=0.0)
    parser.add_argument("--bias-stop", type=float, default=1.0)
    parser.add_argument("--bias-points", type=int, default=9)
    parser.add_argument("--periods", type=int, default=10)
    parser.add_argument("--ramp-periods", type=int, default=5)
    parser.add_argument("--decay-limit", type=float, default=1e-5)
    parser.add_argument(
        "--lyapunov-periods", type=int, default=4,
        help="sampled-period map steps for the largest LE; 0 disables it",
    )
    args = parser.parse_args(argv)
    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if args.bias_points < 2:
        raise SystemExit("--bias-points must be at least 2")
    module = _load_h1_module()
    current, power = _reference_current(args.checkpoint)
    fractions = np.linspace(args.bias_start, args.bias_stop, args.bias_points)
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = {
        "up": _run_direction(
            module, args.circuit_dir, fractions, direction="up", freq_ghz=args.freq_ghz,
            pump_port=args.pump_port, out_port=args.out_port, target_current=current,
            target_power=power, periods=args.periods, ramp_periods=args.ramp_periods,
            decay_limit=args.decay_limit, output_dir=args.outdir,
            lyapunov_periods=args.lyapunov_periods,
        ),
        "down": _run_direction(
            module, args.circuit_dir, fractions, direction="down", freq_ghz=args.freq_ghz,
            pump_port=args.pump_port, out_port=args.out_port, target_current=current,
            target_power=power, periods=args.periods, ramp_periods=args.ramp_periods,
            decay_limit=args.decay_limit, output_dir=args.outdir,
            lyapunov_periods=args.lyapunov_periods,
        ),
    }
    payload = {
        "status": "COMPLETE", "reference_checkpoint": str(args.checkpoint),
        "reference_power_dbm": power, "reference_current_a": current,
        "controls_external_flux_fraction": fractions.tolist(), "results": results,
    }
    path = args.outdir / "bias_campaign_summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
