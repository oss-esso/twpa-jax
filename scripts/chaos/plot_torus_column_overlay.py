"""Overlay K=5 torus diagnostics on the Phase C comparison plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver.core import load_circuit
from twpa_solver.multitone.basis import build_autonomous_torus_basis
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.signal.io import load_pump


def _number(value: Any) -> float | None:
    """Return a finite float or ``None``."""
    if value in (None, "", "None", "nan", "NaN"):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a CSV or JSON list of rows."""
    if path.is_dir():
        return [
            json.loads(item.read_text(encoding="utf-8"))
            for item in sorted(path.glob("torus_palc.point_*.json"))
        ]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("points", payload.get("rows", []))


def _pump_current(pump: Any) -> float:
    """Read the achieved pump current from checkpoint metadata."""
    for key in ("pump_current_a", "pump_current_peak_a", "current_a"):
        value = pump.metadata.get(key)
        if value is not None:
            return float(value)
    raise KeyError("pump metadata has no achieved current")


def _torus_rj(
    circuit: Any,
    pump_dir: Path,
    state_path: Path,
    *,
    q_max: int,
    sideband_harmonics: int,
    pump_port: int,
) -> tuple[float, float]:
    """Return torus junction utilization and effective pump current."""
    pump = load_pump(pump_dir, fallback_pump_freq_ghz=7.9)
    with np.load(state_path) as data:
        state = np.asarray(data["state"], dtype=np.complex128)
        omega_a = float(data["omega_a"])
        source_tau = float(data["source_tau"])
    basis = build_autonomous_torus_basis(
        pump.omega_p,
        omega_a,
        pump.modes,
        q_max,
        sideband_harmonics=sideband_harmonics,
    )
    drive = MultiToneDrive(
        basis.pump_tone,
        circuit.port_to_index[pump_port],
        _pump_current(pump),
    ).to_coeffs(basis, circuit.node_count)
    problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.pump_turn_on(drive),
        loss_model="current_complex_c",
    )
    if state.shape != (basis.n_tones, circuit.node_count):
        raise ValueError(f"state shape does not match K=5 basis: {state.shape}")
    branch_flux = problem.branch_flux_time(state)
    total_flux = branch_flux + problem.dc_branch_flux[None, :]
    currents = np.asarray(problem.branch.current(total_flux), dtype=float)
    peak_current = np.max(np.abs(currents), axis=0)
    critical = np.asarray(problem.branch.critical_current, dtype=float).reshape(-1)
    ratio = float(np.max(peak_current / np.maximum(np.abs(critical), 1e-300)))
    return ratio, _pump_current(pump) * source_tau


def _timings(path: Path) -> dict[int, float]:
    """Return elapsed wall time between per-point timing events."""
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starts = {
        int(event["point_index"]): float(event["timestamp"])
        for event in events
        if event.get("stage") == "point" and event.get("status") == "before"
    }
    ends = {
        int(event["point_index"]): float(event["timestamp"])
        for event in events
        if event.get("stage") == "point" and event.get("status") == "after"
    }
    return {
        index: ends[index] - start
        for index, start in starts.items()
        if index in ends
    }


def _plot(
    base_rows: list[dict[str, Any]],
    torus_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    """Write the three-panel Phase C plot with torus diagnostics overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [1.0e6 * float(row["pump_current_peak_a_achieved"]) for row in base_rows]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.6), sharex=True)
    gain_ax, rj_ax, runtime_ax = axes
    for prefix, label, marker in (
        ("fdtd", "FDTD", "^-"),
        ("single_tone", "single-tone HB", "o-"),
        ("new_hb", "new HB", "s-"),
    ):
        pairs = [
            (xx, _number(row.get(f"{prefix}_gain_vs_off_db")))
            for xx, row in zip(x, base_rows)
        ]
        pairs = [(xx, yy) for xx, yy in pairs if yy is not None]
        if pairs:
            gain_ax.plot(
                [item[0] for item in pairs],
                [item[1] for item in pairs],
                marker,
                label=label,
            )
    gain_ax.set_ylabel("gain_vs_off (dB)")
    gain_ax.legend()
    gain_ax.text(
        0.99,
        0.04,
        "K=5 autonomous torus: gain_vs_off undefined",
        transform=gain_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    for prefix, label, marker in (
        ("fdtd", "FDTD signal-on", "^-"),
        ("single_tone", "single-tone HB", "o-"),
        ("new_hb", "new-HB pump diagnostic", "s-"),
    ):
        pairs = [
            (xx, _number(row.get(f"{prefix}_r_j")))
            for xx, row in zip(x, base_rows)
        ]
        pairs = [(xx, yy) for xx, yy in pairs if yy is not None]
        if pairs:
            rj_ax.plot(
                [item[0] for item in pairs],
                [item[1] for item in pairs],
                marker,
                label=label,
            )
    torus_x = [row["current_ua"] for row in torus_rows]
    rj_ax.plot(
        torus_x,
        [row["torus_r_j"] for row in torus_rows],
        "D--",
        color="tab:purple",
        label="K=5 torus HB",
    )
    rj_ax.set_ylabel("junction utilization")
    rj_ax.legend()
    for prefix, label, marker in (
        ("single_tone", "single-tone HB", "o-"),
        ("new_hb", "new HB", "s-"),
    ):
        pairs = [
            (xx, _number(row.get(f"{prefix}_runtime_s")))
            for xx, row in zip(x, base_rows)
        ]
        pairs = [(xx, yy) for xx, yy in pairs if yy is not None]
        if pairs:
            runtime_ax.plot(
                [item[0] for item in pairs],
                [item[1] for item in pairs],
                marker,
                label=label,
            )
    runtime_ax.plot(
        torus_x,
        [row["torus_runtime_s"] for row in torus_rows],
        "D--",
        color="tab:purple",
        label="K=5 torus HB",
    )
    runtime_ax.set_ylabel("wall time (s)")
    runtime_ax.set_xlabel("achieved pump current (microampere)")
    runtime_ax.legend()
    fig.suptitle("Phase C measured columns: ipm_2c_fixed, S=6 + K=5 torus")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    """Load Phase C and torus artifacts and write the overlay figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--torus-json", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--timing-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--pump-port", type=int, default=4)
    args = parser.parse_args()
    circuit = load_circuit(args.circuit_dir)
    base_rows = _read_rows(args.base_csv)
    torus_rows = _read_rows(args.torus_json)
    timings = _timings(args.timing_jsonl)
    enriched: list[dict[str, Any]] = []
    for row in torus_rows:
        index = int(row["point_index"])
        state_path = args.state_dir / f"point_{index:03d}.npz"
        if not state_path.exists() or not bool(row.get("converged")):
            continue
        rj, current = _torus_rj(
            circuit,
            Path(row["pump_dir"]),
            state_path,
            q_max=int(row["q_max"]),
            sideband_harmonics=5,
            pump_port=args.pump_port,
        )
        if index not in timings:
            continue
        enriched.append(
            {
                "point_index": index,
                "current_ua": current * 1.0e6,
                "torus_r_j": rj,
                "torus_runtime_s": timings[index],
                "omega_a_over_omega_p": row["omega_a_over_omega_p"],
                "torus_radius_squared": row["torus_radius_squared"],
            }
        )
    _plot(base_rows, enriched, args.output)
    print(json.dumps({"output": str(args.output), "points": enriched}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
