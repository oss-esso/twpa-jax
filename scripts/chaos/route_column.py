"""Classify one pump-frequency column without running downstream solvers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.chaos import track_critical_root as tracker  # noqa: E402
from twpa_solver.chaos.routing import (  # noqa: E402
    Regime,
    RegimeVerdict,
    probe_multiplier,
    route,
)


CSV_FIELDS = [
    "drive_dbm",
    "regime",
    "evidence",
    "margin",
    "mode_overlap",
    "reason",
    "method",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately narrow one-column command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", required=True, type=Path)
    parser.add_argument("--pump-dir", required=True, type=Path)
    parser.add_argument("--column-freq-ghz", required=True, type=float)
    parser.add_argument("--drive-dbms", required=True)
    parser.add_argument("--sidebands", type=int, default=8)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/chaos/routing"))
    parser.add_argument("--initial-signal-ghz", type=complex, default=None)
    parser.add_argument("--gamma-nt", type=int, default=4096)
    parser.add_argument("--pump-port", type=int, default=None)
    parser.add_argument("--mode-overlap-threshold", type=float, default=0.99)
    parser.add_argument("--imaginary-seed-ghz", type=float, default=1.0e-3)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    return parser.parse_args(argv)


def _tracker_args(args: argparse.Namespace, saved_pump: Any) -> argparse.Namespace:
    """Build the existing continuation configuration without duplicating it."""
    initial = args.initial_signal_ghz or complex(args.column_freq_ghz, 0.0)
    return tracker.parse_args([
        "--circuit-dir", str(args.circuit_dir),
        "--pump-dir", str(args.pump_dir),
        "--drive-dbms", args.drive_dbms,
        "--sidebands", str(args.sidebands),
        "--initial-signal-ghz", str(initial),
        "--loss-model", args.loss_model,
        "--out-csv", str(args.output / "routing_hill.csv"),
        "--pump-port", str(args.pump_port or saved_pump.metadata.get("pump_port", 1)),
        "--gamma-nt", str(args.gamma_nt),
        "--max-iters", str(args.max_iters),
        "--tol", str(args.tol),
        "--overlap-threshold", str(args.mode_overlap_threshold),
        "--pump-stall-patience", "0",
        "--pump-stall-ratio", "0.95",
        "--pump-min-alpha", "0.0000152587890625",
    ])


def _undecided(reason: str, *, drive_dbm: float) -> dict[str, Any]:
    return {
        "drive_dbm": float(drive_dbm),
        "regime": Regime.UNDECIDED.value,
        "evidence": float("nan"),
        "margin": 0.0,
        "mode_overlap": None,
        "reason": reason,
        "method": "undecided",
    }


def _row(drive_dbm: float, verdict: RegimeVerdict) -> dict[str, Any]:
    return {
        "drive_dbm": float(drive_dbm),
        "regime": verdict.regime.value,
        "evidence": verdict.evidence,
        "margin": verdict.margin,
        "mode_overlap": verdict.mode_overlap,
        "reason": verdict.reason,
        "method": route(verdict),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the Hill probe across one frequency column."""
    if args.column_freq_ghz <= 0.0:
        raise ValueError("column frequency must be positive")
    drives = tracker.parse_float_list(args.drive_dbms, name="drive dbms")
    circuit = tracker.load_circuit(args.circuit_dir)
    saved_pump = tracker.load_pump(args.pump_dir, fallback_pump_freq_ghz=args.column_freq_ghz)
    continuation_args = _tracker_args(args, saved_pump)
    continuation = tracker.PumpContinuation(continuation_args, circuit, saved_pump)
    modes = tracker.sideband_list(args.sidebands)
    dc_flux = tracker._dc_flux(circuit, args.pump_dir)
    previous_mode: np.ndarray | None = None
    previous_multiplier: complex | None = None
    seed = args.initial_signal_ghz or complex(args.column_freq_ghz, 0.0)
    rows: list[dict[str, Any]] = []
    warm_state = saved_pump.X
    for drive_dbm in drives:
        pump_step = continuation.solve(drive_dbm, warm_state)
        if not pump_step.converged or pump_step.pump is None:
            rows.append(_undecided(
                f"pump solve failed: {pump_step.failure_reason or 'unknown failure'}",
                drive_dbm=drive_dbm,
            ))
            continue
        khat, khat_base = tracker._build_hill_operator(
            circuit, pump_step.pump, modes, args.gamma_nt, dc_flux,
        )
        verdict = probe_multiplier(
            circuit=circuit,
            khat=khat,
            khat_base=khat_base,
            omega_p=pump_step.pump.omega_p,
            ms=modes,
            seed_signal_ghz=seed,
            loss_model=args.loss_model,
            previous_mode_vector=previous_mode,
            previous_multiplier=previous_multiplier,
            imaginary_seed_ghz=args.imaginary_seed_ghz,
            mode_overlap_threshold=args.mode_overlap_threshold,
            max_iters=args.max_iters,
            tol=args.tol,
        )
        rows.append(_row(drive_dbm, verdict))
        if verdict.mode_vector is not None and verdict.regime is not Regime.UNDECIDED:
            previous_mode = verdict.mode_vector
            previous_multiplier = verdict.multiplier
            if verdict.signal_ghz is not None:
                seed = verdict.signal_ghz
            warm_state = pump_step.full_state

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "routing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    png_path = args.output / "routing.png"
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    colours = {
        Regime.PERIOD_1.value: "tab:blue",
        Regime.TORUS.value: "tab:orange",
        Regime.BROADBAND.value: "tab:red",
        Regime.UNDECIDED.value: "tab:gray",
    }
    for regime, colour in colours.items():
        selected = [row for row in rows if row["regime"] == regime]
        if selected:
            axis.scatter(
                [row["drive_dbm"] for row in selected],
                [row["evidence"] for row in selected],
                label=regime,
                color=colour,
            )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xlabel("drive (dBm)")
    axis.set_ylabel("named multiplier magnitude")
    axis.legend()
    figure.tight_layout()
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    summary = {
        "column_freq_ghz": args.column_freq_ghz,
        "rows": len(rows),
        "csv": str(csv_path),
        "png": str(png_path),
        "regimes": {
            regime: sum(row["regime"] == regime for row in rows)
            for regime in colours
        },
    }
    (args.output / "routing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the diagnostic driver."""
    print(json.dumps(run(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
