"""Track one harmonic-balance Floquet branch across pump settings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.floquet_stability_sweep import _load_pump_and_khat  # noqa: E402
from twpa_solver.signal.branch_tracking import (  # noqa: E402
    serialize_branch,
    track_floquet_branch,
)
from twpa_solver.signal.floquet import (  # noqa: E402
    assemble_khat_conversion_base,
)
from twpa_solver.core import load_circuit  # noqa: E402


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--pump-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--control-value", type=float, nargs="*")
    parser.add_argument("--pump-freq-ghz", type=float, default=None)
    parser.add_argument("--sidebands", type=int, default=4)
    parser.add_argument("--gamma-nt", type=int, default=4096)
    parser.add_argument("--seed-signal-ghz", type=float, required=True)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--refine-max-iters", type=int, default=30)
    parser.add_argument("--refine-tol", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    if args.control_value and len(args.control_value) != len(args.pump_dir):
        raise ValueError("--control-value must have one value per --pump-dir")
    controls = args.control_value or [
        float(index) for index in range(len(args.pump_dir))
    ]
    circuit = load_circuit(args.circuit_dir)
    khat_values: list[dict[int, Any]] = []
    khat_bases: list[Any] = []
    pump_frequency: float | None = None
    metadata: list[dict[str, Any]] = []
    ms = list(range(-args.sidebands, args.sidebands + 1))
    for directory, control in zip(args.pump_dir, controls):
        pump, _ms, khat = _load_pump_and_khat(
            circuit, directory, args.pump_freq_ghz, args.sidebands, args.gamma_nt
        )
        current_frequency = float(pump.omega_p)
        if pump_frequency is None:
            pump_frequency = current_frequency
        elif not np.isclose(current_frequency, pump_frequency, rtol=0.0, atol=1e-6):
            raise ValueError("pump frequency changes within a branch scan")
        khat_values.append(khat)
        khat_bases.append(assemble_khat_conversion_base(circuit, khat, ms))
        metadata.append({"pump_dir": str(directory), "control": float(control)})
    if pump_frequency is None:
        raise ValueError("no pump settings were supplied")

    branch = track_floquet_branch(
        circuit,
        khat_values,
        controls,
        omega_p=pump_frequency,
        ms=ms,
        seed_signal_ghz=args.seed_signal_ghz,
        loss_model=args.loss_model,
        khat_base_by_parameter=khat_bases,
        max_iters=args.refine_max_iters,
        tol=args.refine_tol,
    )
    points = serialize_branch(branch)["points"]
    for row, point in zip(metadata, points):
        row.update(point)
    crossings = [
        row for row in metadata
        if row["converged"] and row["multiplier"]["magnitude"] >= 1.0
    ]
    return {
        "metadata": {
            "circuit_dir": str(args.circuit_dir),
            "sidebands": args.sidebands,
            "loss_model": args.loss_model,
            "pump_frequency_rad_per_s": pump_frequency,
            "seed_signal_ghz": args.seed_signal_ghz,
            "branch_tracking": "previous_complex_root_and_multiplier_displacement",
        },
        "points": metadata,
        "first_unit_circle_crossing": crossings[0] if crossings else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_scan(args)
    for index, point in enumerate(result["points"]):
        _atomic_write(
            args.out.with_name(f"{args.out.stem}.setting_{index:03d}.json"),
            point,
        )
    _atomic_write(args.out, result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
