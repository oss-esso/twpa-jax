"""Plot the design-independent finite-signal validation panels.

The sweep is produced by ``run_compression.py --save-states all``.  This
script deliberately keeps the sector-power definition explicit: it is the
sum of squared positive-frequency coefficient magnitudes in each detuning
sector, which is sufficient for perturbative slope tests because all fixed
frequency factors cancel.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from twpa_solver.builders.jc_doc import build_jpa
from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath


def _load_fixture() -> CircuitMatrices:
    builder, _ = build_jpa()
    arrays = builder.assemble()
    return CircuitMatrices(
        C=arrays["C"],
        G=arrays["G"],
        K=arrays["K"],
        Bphi=arrays["Bphi"],
        Ic=arrays["Ic"],
        port_to_index=arrays["ports"],
    )


def _read_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            converted: dict[str, float | str] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
        return rows


def _basis(summary: dict[str, object]) -> MultiToneBasis:
    metadata = summary["basis"]
    assert isinstance(metadata, dict)
    tones = [ToneIndex(int(item["h"]), int(item["q"])) for item in metadata["tones"]]
    return MultiToneBasis(
        tones=tones,
        omega_p=float(metadata["omega_p"]),
        delta=float(metadata["delta"]),
        n_p=int(metadata["n_p"]),
        n_delta=int(metadata["n_delta"]),
    )


def _state(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["X_real"] + 1j * data["X_imag"], dtype=np.complex128)


def _sector_powers(rows: list[dict[str, float | str]], states: list[np.ndarray], basis: MultiToneBasis) -> dict[int, np.ndarray]:
    values: dict[int, list[float]] = {}
    for state in states:
        for q in sorted({tone.q for tone in basis.tones}):
            indices = [index for index, tone in enumerate(basis.tones) if tone.q == q]
            values.setdefault(q, []).append(float(np.sum(np.abs(state[indices]) ** 2)))
    return {q: np.asarray(power) for q, power in values.items()}


def _jvp_curve(
    rows: list[dict[str, float | str]], states: list[np.ndarray], basis: MultiToneBasis,
    circuit: CircuitMatrices,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    row = rows[0]
    source = np.zeros_like(states[0])
    source[basis.index_of(basis.pump_tone), circuit.port_to_index[1]] = 0.5 * 1.13e-8
    source[basis.index_of(basis.signal_tone), circuit.port_to_index[1]] = 0.5 * row["signal_current_a"]
    problem = FullMultiToneProblem(circuit, basis, AffineSourcePath.pump_turn_on(source))
    x = states[0]
    rng = np.random.default_rng(20260729)
    direction = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    direction *= np.linalg.norm(x) / np.linalg.norm(direction)
    analytic = problem.jvp_coeffs(x, direction)
    etas = np.logspace(-2, -12, 31)
    errors = np.asarray([
        np.linalg.norm(analytic - (problem.residual_coeffs(x + eta * direction, 1.0)
                                   - problem.residual_coeffs(x - eta * direction, 1.0)) / (2.0 * eta))
        / np.linalg.norm(analytic)
        for eta in etas
    ])
    descending = slice(0, int(np.argmin(errors)) + 1)
    slope = float(np.polyfit(np.log10(etas[descending]), np.log10(errors[descending]), 1)[0])
    return etas, errors, slope, float(np.min(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--floquet-db", type=float, default=None)
    args = parser.parse_args()
    summary = json.loads((args.input_dir / "compression_summary.json").read_text())
    rows = _read_rows(args.input_dir / "compression_points.csv")
    basis = _basis(summary)
    states = [_state(args.input_dir / f"multitone_solution_signal_{i:04d}.npz") for i in range(len(rows))]
    sectors = _sector_powers(rows, states, basis)
    power = np.asarray([row["signal_power_dbm"] for row in rows])
    gain = np.asarray([row["gain_vs_off_db"] for row in rows])
    compression = np.asarray([row["compression_db"] for row in rows])
    pump_depletion = np.asarray([row["pump_depletion_db"] for row in rows])
    residual = np.asarray([row["hb_residual_rel"] for row in rows])
    balance = np.asarray([row["power_balance_rel_err"] for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(power, gain, "o-", label="multitone")
    if args.floquet_db is not None:
        axes[0, 0].axhline(args.floquet_db, color="k", linestyle="--", label="Floquet")
    axes[0, 0].set_ylabel("Gain vs off (dB)")
    axes[0, 0].set_xlabel("Signal power (dBm)")
    for q, values in sorted(sectors.items(), key=lambda item: abs(item[0])):
        if q != 0:
            axes[0, 1].loglog(10.0 ** (power / 10.0), values, "o-", label=f"q={q}")
    axes[0, 1].set_xlabel("Relative signal power")
    axes[0, 1].set_ylabel("Sector coefficient power")
    axes[0, 1].legend()
    axes[1, 0].semilogx(10.0 ** (power / 10.0), compression, "o-", label="HB compression")
    axes[1, 0].semilogx(10.0 ** (power / 10.0), pump_depletion, "--", label="Pump depletion")
    axes[1, 0].set_xlabel("Relative signal power")
    axes[1, 0].set_ylabel("dB")
    axes[1, 0].legend()
    axes[1, 1].loglog(10.0 ** (power / 10.0), residual, "o-", label="HB residual")
    axes[1, 1].loglog(10.0 ** (power / 10.0), np.maximum(balance, 1e-30), "o-", label="power balance")
    axes[1, 1].set_xlabel("Relative signal power")
    axes[1, 1].set_ylabel("relative error")
    axes[1, 1].legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    etas, errors, slope, minimum = _jvp_curve(rows, states, basis, _load_fixture())
    jvp_figure, axis = plt.subplots(figsize=(6, 4))
    axis.loglog(etas, errors, "o-")
    axis.set_title(f"JVP convergence: slope={slope:.3g}, min={minimum:.3e}")
    axis.set_xlabel("eta")
    axis.set_ylabel("relative error")
    jvp_figure.tight_layout()
    jvp_figure.savefig(args.output.with_name(args.output.stem + "_jvp.png"), dpi=180)
    plt.close(jvp_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
