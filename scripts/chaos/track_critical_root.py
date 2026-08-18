"""Continue one complex Hill root in pump-drive amplitude.

The amplitude values are dimensionless multipliers of the converged pump state
stored in ``--pump-dir``.  This keeps the driver independent of a particular
pump solver while rebuilding the Hill operator at every accepted multiplier.
The first value must be a known-stable, below-transition operating point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.signal import (  # noqa: E402
    PumpSolution,
    assemble_khat_conversion_base,
    build_khat,
    compute_gamma_hat,
    load_pump,
    sideband_list,
)
from twpa_solver.signal.branch_tracking import (  # noqa: E402
    FloquetBranchPoint,
    track_floquet_point,
)


CSV_FIELDS = [
    "drive_amplitude",
    "omega_real_ghz",
    "omega_imag_ghz",
    "growth_rate_per_s",
    "multiplier_magnitude",
    "multiplier_phase_rad",
    "floquet_kind",
    "mode_overlap",
    "discontinuity",
    "stability_verdict",
    "converged",
    "iterations",
    "residual",
]


def parse_amplitudes(value: str) -> list[float]:
    """Parse a strictly increasing positive amplitude multiplier ladder."""
    try:
        values = [float(token.strip()) for token in value.split(",")]
    except ValueError as exc:
        raise ValueError("drive amplitudes must be comma-separated numbers") from exc
    if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError("drive amplitudes must be finite and positive")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("drive amplitudes must be strictly increasing")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", required=True)
    parser.add_argument("--pump-dir", required=True)
    parser.add_argument("--drive-amplitudes", required=True)
    parser.add_argument("--sidebands", type=int, required=True)
    parser.add_argument("--initial-signal-ghz", type=float, required=True)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--pump-port", type=int, default=None)
    parser.add_argument("--gamma-nt", type=int, default=4096)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--overlap-threshold", type=float, default=0.8)
    parser.add_argument("--discontinuity-threshold", type=float, default=0.25)
    parser.add_argument("--min-step", type=float, default=1.0e-4)
    return parser.parse_args(argv)


def _dc_flux(circuit: Any, pump_dir: Path) -> np.ndarray | None:
    report_path = pump_dir / "pump_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    value = metadata.get("dc_branch_flux", metadata.get("dc_branch_flux_wb"))
    if value is None:
        return None
    flux = np.asarray(value, dtype=float).reshape(-1)
    if flux.size == 1:
        flux = np.full(circuit.branch_count, float(flux[0]))
    if flux.size != circuit.branch_count:
        raise ValueError("pump report DC flux does not match circuit branches")
    return flux


def _build_hill_operator(
    circuit: Any,
    pump: PumpSolution,
    amplitude: float,
    ms: list[int],
    gamma_nt: int,
    dc_flux: np.ndarray | None,
) -> tuple[dict[int, Any], Any]:
    scaled_pump = replace(pump, X=np.asarray(pump.X) * amplitude)
    max_ell = max(abs(left - right) for left in ms for right in ms)
    gamma_hat = compute_gamma_hat(
        circuit=circuit,
        pump=scaled_pump,
        max_ell=max_ell,
        gamma_nt=gamma_nt,
        dc_branch_flux=dc_flux,
    )
    khat = build_khat(circuit.Bphi, gamma_hat, drop_tol=0.0)
    return khat, assemble_khat_conversion_base(circuit, khat, ms)


def _memory_guard(circuit: Any, sidebands: int) -> None:
    dimension = circuit.node_count * (2 * sidebands + 1)
    estimate = max(64.0 * dimension * (2 * sidebands + 1), 64.0e6)
    try:
        import psutil

        available = float(psutil.virtual_memory().available)
    except ImportError:
        available = float("inf")
    print(
        "memory_guard dimension=%d estimated_process_bytes=%.0f "
        "free_bytes=%.0f" % (dimension, estimate, available)
    )
    if estimate > available:
        raise MemoryError("estimated process footprint exceeds available memory")


def _write_point(
    writer: csv.DictWriter,
    point: FloquetBranchPoint,
    amplitude: float,
) -> None:
    signal = complex(point.resonance.signal_ghz)
    classification = point.classification
    writer.writerow(
        {
            "drive_amplitude": amplitude,
            "omega_real_ghz": signal.real,
            "omega_imag_ghz": signal.imag,
            "growth_rate_per_s": point.resonance.growth_rate_per_s,
            "multiplier_magnitude": classification.magnitude,
            "multiplier_phase_rad": classification.phase_rad,
            "floquet_kind": classification.kind,
            "mode_overlap": point.mode_overlap,
            "discontinuity": point.discontinuity,
            "stability_verdict": point.stability_verdict,
            "converged": point.resonance.converged,
            "iterations": point.resonance.iterations,
            "residual": point.resonance.residual,
        }
    )


def _crossing(
    points: list[tuple[float, FloquetBranchPoint]],
    pump_freq_ghz: float,
) -> dict[str, float] | None:
    for (left_amp, left), (right_amp, right) in zip(points, points[1:]):
        left_growth = left.resonance.growth_rate_per_s
        right_growth = right.resonance.growth_rate_per_s
        if left_growth == 0.0:
            fraction = 0.0
        elif left_growth * right_growth <= 0.0:
            fraction = -left_growth / (right_growth - left_growth)
        else:
            continue
        amplitude = left_amp + fraction * (right_amp - left_amp)
        signal_left = complex(left.resonance.signal_ghz)
        signal_right = complex(right.resonance.signal_ghz)
        generator_ghz = signal_left.real + fraction * (
            signal_right.real - signal_left.real
        )
        return {
            "drive_amplitude": amplitude,
            "generator_frequency_ghz": generator_ghz,
            "omega_a_over_omega_p": generator_ghz / pump_freq_ghz,
        }
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run adaptive continuation and return the crossing summary."""
    amplitudes = parse_amplitudes(args.drive_amplitudes)
    if args.sidebands < 0:
        raise ValueError("sidebands must be non-negative")
    if not 0.0 < args.overlap_threshold <= 1.0:
        raise ValueError("overlap threshold must lie in (0, 1]")
    circuit = load_circuit(args.circuit_dir)
    pump_dir = Path(args.pump_dir)
    pump = load_pump(pump_dir, fallback_pump_freq_ghz=1.0)
    ms = sideband_list(args.sidebands)
    _memory_guard(circuit, args.sidebands)
    dc_flux = _dc_flux(circuit, pump_dir)
    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points: list[tuple[float, FloquetBranchPoint]] = []
    seed = complex(args.initial_signal_ghz)
    previous_mode: np.ndarray | None = None
    previous_multiplier: complex | None = None
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()
        current = amplitudes[0]
        first_khat, first_base = _build_hill_operator(
            circuit, pump, current, ms, args.gamma_nt, dc_flux
        )
        first = track_floquet_point(
            circuit=circuit,
            khat=first_khat,
            khat_base=first_base,
            parameter=current,
            omega_p=pump.omega_p,
            ms=ms,
            seed_signal_ghz=seed,
            seed_mode_vector=None,
            previous_multiplier=None,
            loss_model=args.loss_model,
            max_iters=args.max_iters,
            tol=args.tol,
            discontinuity_threshold=args.discontinuity_threshold,
            mode_overlap_threshold=args.overlap_threshold,
        )
        if not first.resonance.converged:
            raise RuntimeError("initial stable root did not converge")
        _write_point(writer, first, current)
        handle.flush()
        points.append((current, first))
        seed = complex(first.resonance.signal_ghz)
        previous_mode = first.resonance.mode_vector
        previous_multiplier = first.classification.multiplier
        step = amplitudes[1] - current if len(amplitudes) > 1 else 0.0

        for target in amplitudes[1:]:
            while current < target - 1.0e-15:
                candidate = min(target, current + step)
                khat, khat_base = _build_hill_operator(
                    circuit, pump, candidate, ms, args.gamma_nt, dc_flux
                )
                point = track_floquet_point(
                    circuit=circuit,
                    khat=khat,
                    khat_base=khat_base,
                    parameter=candidate,
                    omega_p=pump.omega_p,
                    ms=ms,
                    seed_signal_ghz=seed,
                    seed_mode_vector=previous_mode,
                    previous_multiplier=previous_multiplier,
                    loss_model=args.loss_model,
                    max_iters=args.max_iters,
                    tol=args.tol,
                    discontinuity_threshold=args.discontinuity_threshold,
                    mode_overlap_threshold=args.overlap_threshold,
                )
                low_overlap = (
                    point.mode_overlap is not None
                    and point.mode_overlap < args.overlap_threshold
                )
                if not point.resonance.converged or low_overlap:
                    step *= 0.5
                    if step < args.min_step:
                        raise RuntimeError(
                            "adaptive continuation reached --min-step without "
                            "a converged, overlapping point"
                        )
                    continue
                _write_point(writer, point, candidate)
                handle.flush()
                points.append((candidate, point))
                current = candidate
                seed = complex(point.resonance.signal_ghz)
                previous_mode = point.resonance.mode_vector
                previous_multiplier = point.classification.multiplier
                if point.mode_overlap is None or not point.discontinuity:
                    step = max(step * 1.5, args.min_step)

    crossing = _crossing(points, pump.pump_freq_ghz)
    return {
        "pump_frequency_ghz": pump.pump_freq_ghz,
        "loss_model": args.loss_model,
        "pump_port": args.pump_port,
        "crossing": crossing,
        "accepted_steps": len(points),
        "csv": str(output_path),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
