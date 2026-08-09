"""Map the Themis Jan28 compression cube and infer the on-chip pump power.

The measurement sweeps signal power against signal frequency at one pump
setting, so every frequency column is an independent compression curve.  Fitting
`G0` and `P1dB` per column and inverting the pump-depletion relation

    P1dB_in = Pp - G0 - 8.8786 dB          (equivalently P1dB_out = Pp - 9.8786)

gives one inferred pump power per frequency.  If saturation were depletion
limited and the line calibration were correct, that inference would be flat and
equal to the independently estimated -66.7 dBm at the device input.  Departures
measure either the calibration or the failure of the depletion picture, and the
two are separable only by their frequency dependence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

CUBE = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
SIGNAL_LINE_LOSS_DB = 72.5
PUMP_FREQ_GHZ = 7.256
PUMP_INPUT_DBM = -66.7
PUMP_EXCLUSION_GHZ = 0.15
DEPLETION_CONSTANT_DB = 10.0 * math.log10((10.0**0.1 - 1.0) / 2.0)


@dataclass(frozen=True)
class ColumnResult:
    """One frequency column's compression fit."""

    frequency_ghz: float
    g0_db: float
    p1db_input_dbm: float | None
    p1db_output_dbm: float | None
    inferred_pump_dbm: float | None
    n_crossings: int


def load_cube(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frequency GHz, device signal power dBm, gain dB)."""
    data = np.load(path, allow_pickle=True).item()
    frequency_ghz = np.asarray(data["Frequency"], dtype=float) / 1e9
    signal_dbm = np.asarray(data["SignalPower"], dtype=float) - SIGNAL_LINE_LOSS_DB
    gain_db = np.asarray(data["Response"], dtype=float)
    return frequency_ghz, signal_dbm, gain_db


def fit_column(
    signal_dbm: np.ndarray,
    gain_db: np.ndarray,
    frequency_ghz: float,
    *,
    plateau_points: int,
    savgol_window: int,
) -> ColumnResult:
    """Fit G0 and the 1 dB crossing for one frequency column."""
    g0 = float(np.median(gain_db[:plateau_points]))
    smoothed = savgol_filter(gain_db, savgol_window, 2)
    target = g0 - 1.0
    above = smoothed >= target
    # Upward crossings of the target, walking from high power back down; the
    # last one is the edge of the uncompressed plateau.
    crossings = [
        i
        for i in range(len(smoothed) - 1)
        if above[i] and not above[i + 1]
    ]
    if not crossings:
        return ColumnResult(frequency_ghz, g0, None, None, None, 0)
    index = crossings[-1]
    lo, hi = smoothed[index], smoothed[index + 1]
    span = lo - hi
    frac = (lo - target) / span if span != 0.0 else 0.0
    p1db_in = float(signal_dbm[index] + frac * (signal_dbm[index + 1] - signal_dbm[index]))
    p1db_out = p1db_in + g0 - 1.0
    inferred = p1db_in + g0 - DEPLETION_CONSTANT_DB
    return ColumnResult(frequency_ghz, g0, p1db_in, p1db_out, inferred, len(crossings))


def analyse(
    frequency_ghz: np.ndarray,
    signal_dbm: np.ndarray,
    gain_db: np.ndarray,
    *,
    min_gain_db: float,
    plateau_points: int,
    savgol_window: int,
) -> list[ColumnResult]:
    """Fit every usable frequency column."""
    results: list[ColumnResult] = []
    for j, freq in enumerate(frequency_ghz):
        if abs(freq - PUMP_FREQ_GHZ) < PUMP_EXCLUSION_GHZ:
            continue
        column = gain_db[:, j]
        if not np.all(np.isfinite(column)):
            continue
        result = fit_column(
            signal_dbm,
            column,
            float(freq),
            plateau_points=plateau_points,
            savgol_window=savgol_window,
        )
        if result.g0_db < min_gain_db:
            continue
        results.append(result)
    return results


def write_table(results: list[ColumnResult], path: Path) -> None:
    """Write the per-column fit as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_ghz",
                "g0_db",
                "p1db_input_dbm",
                "p1db_output_dbm",
                "inferred_pump_dbm",
                "n_crossings",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    f"{r.frequency_ghz:.6f}",
                    f"{r.g0_db:.6f}",
                    "" if r.p1db_input_dbm is None else f"{r.p1db_input_dbm:.6f}",
                    "" if r.p1db_output_dbm is None else f"{r.p1db_output_dbm:.6f}",
                    "" if r.inferred_pump_dbm is None else f"{r.inferred_pump_dbm:.6f}",
                    r.n_crossings,
                ]
            )


def plot(
    frequency_ghz: np.ndarray,
    signal_dbm: np.ndarray,
    gain_db: np.ndarray,
    results: list[ColumnResult],
    path: Path,
) -> None:
    """Write the four-panel map and inference figure."""
    solved = [r for r in results if r.inferred_pump_dbm is not None]
    freqs = np.array([r.frequency_ghz for r in solved])
    g0 = np.array([r.g0_db for r in solved])
    p1db_in = np.array([r.p1db_input_dbm for r in solved])
    pump = np.array([r.inferred_pump_dbm for r in solved])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    band = (frequency_ghz >= 4.5) & (frequency_ghz <= 10.0)
    mesh = axes[0, 0].pcolormesh(
        frequency_ghz[band],
        signal_dbm,
        np.clip(gain_db[:, band], -40.0, 20.0),
        cmap="magma",
        shading="auto",
    )
    axes[0, 0].axvline(PUMP_FREQ_GHZ, color="cyan", ls="--", lw=1.0)
    axes[0, 0].set_xlabel("signal frequency (GHz)")
    axes[0, 0].set_ylabel("signal power at device (dBm)")
    axes[0, 0].set_title("Themis 105C5 compression map, gain (dB)")
    fig.colorbar(mesh, ax=axes[0, 0], label="gain (dB)")

    axes[0, 1].plot(freqs, g0, lw=0.8, color="tab:blue")
    axes[0, 1].axvline(PUMP_FREQ_GHZ, color="k", ls="--", lw=0.8)
    axes[0, 1].set_xlabel("signal frequency (GHz)")
    axes[0, 1].set_ylabel("$G_0$ (dB)")
    axes[0, 1].set_title("small-signal gain")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(freqs, p1db_in, lw=0.8, color="tab:orange")
    axes[1, 0].axvline(PUMP_FREQ_GHZ, color="k", ls="--", lw=0.8)
    axes[1, 0].set_xlabel("signal frequency (GHz)")
    axes[1, 0].set_ylabel("input $P_{1dB}$ (dBm)")
    axes[1, 0].set_title("input-referred 1 dB compression point")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(freqs, pump, lw=0.8, color="tab:green", label="inferred")
    axes[1, 1].axhline(
        PUMP_INPUT_DBM,
        color="crimson",
        ls="--",
        lw=1.2,
        label=f"line estimate {PUMP_INPUT_DBM:.1f} dBm",
    )
    axes[1, 1].axvline(PUMP_FREQ_GHZ, color="k", ls="--", lw=0.8)
    axes[1, 1].set_xlabel("signal frequency (GHz)")
    axes[1, 1].set_ylabel("inferred on-chip pump (dBm)")
    axes[1, 1].set_title(r"$P_p = P_{1dB,in} + G_0 + 8.879$ dB")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=CUBE)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp30_themis_pump_inference")
    )
    parser.add_argument("--min-gain-db", type=float, default=4.0)
    parser.add_argument("--plateau-points", type=int, default=10)
    parser.add_argument("--savgol-window", type=int, default=11)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frequency_ghz, signal_dbm, gain_db = load_cube(args.cube)
    results = analyse(
        frequency_ghz,
        signal_dbm,
        gain_db,
        min_gain_db=args.min_gain_db,
        plateau_points=args.plateau_points,
        savgol_window=args.savgol_window,
    )
    write_table(results, args.output_dir / "themis_column_fits.csv")
    plot(
        frequency_ghz,
        signal_dbm,
        gain_db,
        results,
        args.output_dir / "themis_map_and_pump_inference.png",
    )

    solved = [r for r in results if r.inferred_pump_dbm is not None]
    pump = np.array([r.inferred_pump_dbm for r in solved])
    g0 = np.array([r.g0_db for r in solved])
    summary = {
        "n_columns_fit": len(results),
        "n_columns_with_p1db": len(solved),
        "signal_line_loss_db": SIGNAL_LINE_LOSS_DB,
        "depletion_constant_db": DEPLETION_CONSTANT_DB,
        "pump_line_estimate_dbm": PUMP_INPUT_DBM,
        "inferred_pump_dbm_median": float(np.median(pump)) if len(pump) else None,
        "inferred_pump_dbm_mean": float(np.mean(pump)) if len(pump) else None,
        "inferred_pump_dbm_std": float(np.std(pump)) if len(pump) else None,
        "inferred_pump_dbm_p10_p90": (
            [float(np.percentile(pump, 10)), float(np.percentile(pump, 90))]
            if len(pump)
            else None
        ),
        "gain_span_db": [float(g0.min()), float(g0.max())] if len(g0) else None,
        "offset_from_line_estimate_db": (
            float(np.median(pump) - PUMP_INPUT_DBM) if len(pump) else None
        ),
    }
    (args.output_dir / "pump_inference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for key, value in summary.items():
        print(f"{key:34s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
