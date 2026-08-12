"""Run a fixed-pump TD signal-injection gain measurement.

This is an exploratory diagnostic built on the H1 DAE and bounded implicit
trapezoid integrator.  It starts from a completed pump-only restart state,
injects a small sinusoidal signal, and estimates the output voltage Fourier
amplitude over the retained late-time window.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.h1_transient_branch_transfer import (
    PHI0_REDUCED,
    build_system,
    checkpoint_dc_flux,
    implicit_trapezoid_ramp_bounded,
    load_circuit,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--circuit-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--restart", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--freq-ghz", type=float, default=7.9)
    p.add_argument("--signal-ghz", type=float, default=7.4)
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=2)
    p.add_argument("--pump-port", type=int, default=4)
    p.add_argument("--signal-current-a", type=float, default=1e-10)
    p.add_argument(
        "--initialization-mode",
        type=str,
        default="unknown",
        help="Initialization protocol used to create the supplied restart state.",
    )
    p.add_argument("--periods", type=int, default=800)
    p.add_argument("--max-step", type=float, default=0.19634954084936207)
    p.add_argument("--min-step-theta", type=float, default=0.03125)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_report = json.loads((args.checkpoint / "pump_report.json").read_text())
    circuit = load_circuit(args.circuit_dir)
    dc = checkpoint_dc_flux(checkpoint_report, circuit, np.zeros(circuit.branch_count), args.checkpoint)
    system = build_system(args.circuit_dir, args.freq_ghz, args.pump_port, dc)
    restart = np.load(args.restart)
    y0 = np.asarray(restart["y"], dtype=float)
    target_current = float(restart["target_current"])

    base_source = system.source
    source_idx = system.circuit.port_to_index[args.source_port]
    ratio = float(args.signal_ghz / args.freq_ghz)
    signal_current = float(args.signal_current_a)

    def source_with_signal(self, theta, start_current, target_current_arg, ramp_theta):
        source = np.asarray(base_source(theta, start_current, target_current_arg, ramp_theta), dtype=float)
        source[source_idx] += signal_current * math.cos(ratio * theta)
        return source

    system.source = MethodType(source_with_signal, system)
    total_theta = 2.0 * math.pi * int(args.periods)
    sample_theta, states, integrator = implicit_trapezoid_ramp_bounded(
        system, y0, target_current, target_current, total_theta, 0.0,
        args.max_step, 1e-6, 10,
        checkpoint_dir=args.outdir / "restart_checkpoints",
        checkpoint_periods=10, min_step_theta=args.min_step_theta,
        sample_count=256, history_states=1024,
    )
    history_theta = np.asarray(integrator.pop("_bounded_history_theta"), dtype=float)
    history_states = np.asarray(integrator.pop("_bounded_history_states"), dtype=float)
    vout = np.empty(history_states.shape[1], dtype=float)
    max_i_over_ic = np.empty_like(q)
    max_abs_phi = np.empty_like(q)
    for i in range(history_states.shape[1]):
        qi, pi = system.unpack(history_states[:, i])
        if system.algebraic.size and not system.full_state:
            pi[system.algebraic] = system.algebraic_velocity(
                qi, pi[system.differential], system.source(
                    history_theta[i], target_current, target_current, 0.0
                )
            )
        vout[i] = system.omega * PHI0_REDUCED * pi[system.circuit.port_to_index[args.out_port]]
        flux = PHI0_REDUCED * (system.circuit.Bphi.T @ qi)
        josephson_phase = flux / system.circuit.phi0
        branch_current = np.asarray(system.branch.current(flux[None, :]))[0]
        max_i_over_ic[i] = float(np.max(np.abs(branch_current / system.circuit.Ic)))
        max_abs_phi[i] = float(np.max(np.abs(josephson_phase)))
    phase = ratio * history_theta
    coeff = 2.0 * np.mean(vout * np.exp(-1j * phase))
    vout_peak = float(abs(coeff))
    z0 = 50.0
    # Match production port_s_from_unit_current_response(): an injected Norton
    # current corresponds to the incident wave normalization S=2V/(Z0 I).
    gain_db = float(20.0 * math.log10(max(2.0 * vout_peak, 1e-300) / (z0 * signal_current)))
    result = {
        "status": "PASS" if integrator.get("success") else "FAIL",
        "message": integrator.get("message"),
        "periods": int(args.periods),
        "pump_frequency_ghz": float(args.freq_ghz),
        "signal_frequency_ghz": float(args.signal_ghz),
        "signal_current_a_peak": signal_current,
        "initialization_mode": args.initialization_mode,
        "max_step": float(args.max_step),
        "max_step_theta": float(args.max_step),
        "min_step_theta": float(args.min_step_theta),
        "source_port": int(args.source_port),
        "output_port": int(args.out_port),
        "late_window_periods": 20,
        "late_window_samples": int(history_theta.size),
        "output_voltage_peak_v": vout_peak,
        "late_window_max_abs_i_over_ic": float(np.max(max_i_over_ic)),
        "late_window_max_abs_phi_rad": float(np.max(max_abs_phi)),
        "gain_db_50ohm": gain_db,
        "input_power_dbm_50ohm": float(10.0 * math.log10((signal_current**2 * z0 / 2.0) / 1e-3)),
        "output_power_dbm_50ohm": float(10.0 * math.log10(max((vout_peak**2 / (2.0 * z0)) / 1e-3, 1e-300))),
        "integrator": {k: v for k, v in integrator.items() if not k.startswith("_bounded")},
    }
    np.savez_compressed(
        args.outdir / "signal_late_window.npz",
        theta=history_theta,
        output_voltage_v=vout,
        max_abs_i_over_ic=max_i_over_ic,
        max_abs_phi=max_abs_phi,
    )
    (args.outdir / "signal_gain.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
