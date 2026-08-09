"""Validate the deterministic production HB waveform projection round-trip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.h1_transient_branch_transfer import build_system  # noqa: E402
from twpa_solver.pump.hb import FullPumpProblem, HarmonicGrid  # noqa: E402
from twpa_solver.pump.validation import validate_production_hb_state  # noqa: E402


def run(args: argparse.Namespace) -> dict[str, object]:
    system = build_system(args.circuit_dir, args.freq_ghz, args.pump_port)
    report = json.loads((args.checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    data = np.load(args.checkpoint / "pump_solution.npz")
    state = np.asarray(data["X_real"], dtype=float) + 1j * np.asarray(
        data["X_imag"], dtype=float
    )
    modes = np.asarray(
        report.get("metadata", {}).get("pump_modes", data["pump_modes"]), dtype=int
    )
    current = float(report["metadata"]["pump_current_a"])
    grid = HarmonicGrid(
        modes=modes,
        nt=max(2 * int(modes.max()) + 1, 40),
        omega=2.0 * np.pi * args.freq_ghz * 1e9,
    )
    waveform = grid.synthesize(state)
    projected = grid.project_positive(waveform)
    coeff_error = float(
        np.linalg.norm(projected - state) / max(np.linalg.norm(state), 1e-300)
    )

    source_validation = validate_production_hb_state(
        system.circuit, system.branch,
        frequency_hz=args.freq_ghz * 1e9,
        pump_port=args.pump_port,
        pump_current_a=current,
        modes=modes,
        state=state,
        nt=grid.nt,
        metadata=report.get("metadata", {}),
    )
    projected_validation = validate_production_hb_state(
        system.circuit, system.branch,
        frequency_hz=args.freq_ghz * 1e9,
        pump_port=args.pump_port,
        pump_current_a=current,
        modes=modes,
        state=projected,
        nt=grid.nt,
        metadata=report.get("metadata", {}),
    )
    source_residual = source_validation.get("production_hb_residual_rel")
    projected_residual = projected_validation.get("production_hb_residual_rel")
    passed = bool(
        source_validation.get("checkpoint_validated", False)
        and projected_validation.get("checkpoint_validated", False)
        and coeff_error <= args.coeff_tolerance
    )
    result: dict[str, object] = {
        "status": "PASS" if passed else "FAIL",
        "checkpoint": str(args.checkpoint),
        "frequency_ghz": args.freq_ghz,
        "pump_current_a": current,
        "harmonic_modes": modes.tolist(),
        "nt": grid.nt,
        "source_residual_rel": source_residual,
        "projected_residual_rel": projected_residual,
        "coefficient_roundtrip_rel": coeff_error,
        "coefficient_tolerance": args.coeff_tolerance,
        "source_validation": source_validation,
        "projected_validation": projected_validation,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--circuit-dir", type=Path,
        default=ROOT / "designs" / "ipm_2c_fixed",
    )
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--coeff-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
