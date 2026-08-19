"""Measure Floquet-resonance sensitivity to dielectric loss.

This is the Phase 3 control campaign.  It builds the loss variants through
the existing IPM dielectric-loss path and runs the same explicit-loss Hill
scan at one validated HB checkpoint.  Each setting is written immediately so
an interrupted campaign retains completed measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import floquet_stability_sweep as sweep  # noqa: E402
from twpa_solver.builders.ipm import LossSpec, build_variant_design  # noqa: E402
from twpa_solver.core import load_circuit  # noqa: E402


TAN_DELTAS = (0.0, 1e-5, 1e-4, 1e-3)


def variant_name(tan_delta: float) -> str:
    return "2c_base" if tan_delta == 0.0 else f"2c_td{tan_delta:.0e}"


def build_variant(source: Path, outdir: Path, tan_delta: float) -> dict[str, Any]:
    summary = build_variant_design(
        source,
        outdir,
        loss=LossSpec(default=tan_delta),
        overwrite=False,
        coupler_mode="auto",
    )
    return {
        "circuit_dir": str(outdir),
        "tan_delta": tan_delta,
        "total_elements": summary.get("total_elements"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.loss_model = sweep.require_explicit_loss_model(args.loss_model)
    args.outroot.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for tan_delta in TAN_DELTAS:
        circuit_dir = args.outroot / variant_name(tan_delta)
        if not circuit_dir.exists():
            build_variant(args.source, circuit_dir, tan_delta)
        local_out = args.outroot / f"{variant_name(tan_delta)}.json"
        local_args = argparse.Namespace(**vars(args))
        local_args.circuit_dir = circuit_dir
        local_args.out = local_out
        local_args.pump_dir = [args.pump_dir]
        result = main_from_namespace(local_args)
        record = {
            "tan_delta": tan_delta,
            "circuit_dir": str(circuit_dir),
            "result": result,
        }
        records.append(record)
        (args.outroot / f"{variant_name(tan_delta)}.record.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    payload = {
        "metadata": {
            "source_circuit": str(args.source),
            "pump_dir": str(args.pump_dir),
            "loss_model": args.loss_model,
            "tan_delta_grid": list(TAN_DELTAS),
        },
        "records": records,
    }
    (args.outroot / "loss_sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "designs/ipm_2c_fixed")
    parser.add_argument("--pump-dir", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--sidebands", type=int, default=4)
    parser.add_argument("--gamma-nt", type=int, default=1024)
    parser.add_argument("--n-points", type=int, default=700)
    parser.add_argument("--mode-spacing-mhz", type=float, default=241.7)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--refine-complex", action="store_true")
    parser.add_argument("--refine-bifurcations", action="store_true")
    parser.add_argument("--refine-max-iters", type=int, default=30)
    parser.add_argument("--refine-tol", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pump-freq-ghz", type=float, default=7.9)
    parser.add_argument("--span-start-ghz", type=float, default=0.05)
    parser.add_argument("--span-end-fraction", type=float, default=0.99)
    parser.add_argument("--bifurcation-fractions", default="0.0,0.5")
    parser.set_defaults(baseline_pump_dir=None, out=None)
    return parser.parse_args(argv)


def main_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    """Run one explicit sweep and return its JSON-compatible payload."""
    return sweep.run_namespace(args)


if __name__ == "__main__":
    run(parse_args())
