"""Enumerate Tier-1 Hill candidates at one solved pump drive.

This is intentionally a single-drive scan.  It generates seeds only; branch
identity over drive is established by the mode-vector continuation driver.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import track_critical_root as tracker  # noqa: E402
from twpa_solver.signal import (  # noqa: E402
    classify_floquet_resonance,
    local_minima,
    refine_complex_resonance,
    sideband_list,
    sweep_sigma_min,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the candidate enumeration arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", required=True)
    parser.add_argument("--pump-dir", required=True)
    parser.add_argument("--drive-dbm", type=float, required=True)
    parser.add_argument("--sidebands", type=int, required=True)
    parser.add_argument("--pump-port", type=int, required=True)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--scan-min-ghz", type=float, required=True)
    parser.add_argument("--scan-max-ghz", type=float, required=True)
    parser.add_argument("--scan-points", type=int, default=61)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--gamma-nt", type=int, default=4096)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Solve the pump once, scan Tier 1, and refine ranked minima."""
    output_json = Path(args.out_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    circuit = tracker.load_circuit(args.circuit_dir)
    pump = tracker.load_pump(Path(args.pump_dir), fallback_pump_freq_ghz=1.0)
    driver_args = tracker.parse_args(
        [
            "--circuit-dir",
            str(args.circuit_dir),
            "--pump-dir",
            str(args.pump_dir),
            "--drive-dbms",
            str(args.drive_dbm),
            "--sidebands",
            str(args.sidebands),
            "--initial-signal-ghz",
            str(args.scan_min_ghz),
            "--loss-model",
            args.loss_model,
            "--out-csv",
            str(Path(args.out_json).with_suffix(".csv")),
            "--pump-port",
            str(args.pump_port),
            "--gamma-nt",
            str(args.gamma_nt),
            "--pump-max-newton",
            "32",
            "--pump-stall-patience",
            "0",
            "--pump-stall-ratio",
            "0.95",
            "--pump-min-alpha",
            "0.0000152587890625",
        ]
    )
    continuation = tracker.PumpContinuation(driver_args, circuit, pump)
    pump_step = continuation.solve(args.drive_dbm, pump.X)
    if not pump_step.converged or pump_step.pump is None:
        raise RuntimeError("the candidate-drive pump did not converge")
    modes = sideband_list(args.sidebands)
    dc_flux = tracker._dc_flux(circuit, Path(args.pump_dir))
    khat, khat_base = tracker._build_hill_operator(
        circuit, pump_step.pump, modes, args.gamma_nt, dc_flux
    )
    grid = np.linspace(args.scan_min_ghz, args.scan_max_ghz, args.scan_points)
    estimates = sweep_sigma_min(
        circuit=circuit,
        khat=khat,
        khat_base=khat_base,
        omega_p=pump_step.pump.omega_p,
        signal_ghz_grid=[float(value) for value in grid],
        ms=modes,
        loss_model=args.loss_model,
        iters=8,
    )
    sigma = [float(item.sigma_min) for item in estimates]
    minima = local_minima(sigma, k=args.max_candidates)
    candidates: list[dict[str, object]] = []
    for index in minima:
        resonance = refine_complex_resonance(
            circuit=circuit,
            khat=khat,
            khat_base=khat_base,
            omega_p=pump_step.pump.omega_p,
            ms=modes,
            signal_ghz_guess=float(grid[index]),
            loss_model=args.loss_model,
        )
        classification = classify_floquet_resonance(
            resonance, pump_step.pump.omega_p
        )
        seed_path: Path | None = None
        if resonance.mode_vector is not None:
            seed_path = output_json.with_name(
                f"{output_json.stem}.candidate_{len(candidates):02d}.npz"
            )
            np.savez(
                seed_path,
                vector=np.asarray(resonance.mode_vector, dtype=np.complex128),
                sidebands=np.asarray(modes, dtype=np.int64),
            )
        candidates.append(
            {
                "grid_index": int(index),
                "tier1_signal_ghz": float(grid[index]),
                "sigma_min": sigma[index],
                "signal_real_ghz": float(resonance.signal_ghz.real),
                "signal_imag_ghz": float(resonance.signal_ghz.imag),
                "growth_rate_per_s": resonance.growth_rate_per_s,
                "multiplier_magnitude": classification.magnitude,
                "multiplier_phase_rad": classification.phase_rad,
                "floquet_kind": classification.kind,
                "converged": resonance.converged,
                "iterations": resonance.iterations,
                "residual": resonance.residual,
                "floquet_seed_npz": str(seed_path) if seed_path else None,
            }
        )
    result: dict[str, object] = {
        "drive_dbm": args.drive_dbm,
        "pump_frequency_ghz": pump_step.pump.pump_freq_ghz,
        "conversion_dimension": int(circuit.node_count * (2 * args.sidebands + 1)),
        "scan_points": int(args.scan_points),
        "scan_min_ghz": args.scan_min_ghz,
        "scan_max_ghz": args.scan_max_ghz,
        "candidates": candidates,
    }
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    """Run candidate enumeration."""
    print(json.dumps(run(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
