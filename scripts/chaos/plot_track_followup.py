"""Plot the recorded pump-continuation and Hill-branch diagnostics."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "track_followup_plots"


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a recorded continuation CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], name: str) -> float | None:
    """Return a finite CSV float, or ``None`` for an empty/non-finite value."""
    value = row.get(name, "")
    if not value:
        return None
    result = float(value)
    return result if result == result else None


def _save(fig: plt.Figure, name: str) -> None:
    """Save one figure in both PNG and SVG formats."""
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=180)
    fig.savefig(OUTPUT / f"{name}.svg")
    plt.close(fig)


def _plot_pump_fold() -> None:
    """Plot the 2c source continuation and its first failed pump solve."""
    rows = _read_csv(ROOT / "track_followup_p1_2c_01db.csv")
    drive = [float(row["requested_drive_dbm"]) for row in rows]
    current = [float(row["requested_source_current_a"]) * 1e6 for row in rows]
    residual = [
        max(float(row["pump_coeff_rel"]), 1e-16)
        for row in rows
    ]
    converged = [row["pump_converged"] == "True" for row in rows]
    bound_ua = 1.1628e-5 * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(drive, current, "o-", ms=3.5, label="pump solve")
    axes[0].axhline(bound_ua, color="tab:red", ls="--", label="I_bound")
    fail = next(index for index, value in enumerate(converged) if not value)
    axes[0].plot(drive[fail], current[fail], "x", ms=9, mew=2, color="tab:red")
    axes[0].set_xlabel("Requested pump drive (dBm)")
    axes[0].set_ylabel("Requested/on-chip current (µA)")
    axes[0].set_title("ipm_2c_fixed pump continuation")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].semilogy(drive, residual, "o-", ms=3.5)
    axes[1].axvline(drive[fail], color="tab:red", ls="--")
    axes[1].set_xlabel("Requested pump drive (dBm)")
    axes[1].set_ylabel("Pump coefficient residual")
    axes[1].set_title("Residual at each attempted step")
    axes[1].grid(alpha=0.25, which="both")
    _save(fig, "ipm_2c_pump_fold")


def _candidate_labels() -> dict[str, str]:
    """Return candidate labels from the single-drive enumeration artifact."""
    path = ROOT / "track_followup_jtwpa_candidates.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    return {
        f"{index + 1:02d}": f"B{index + 1}: {item['signal_real_ghz']:.3f} GHz"
        for index, item in enumerate(candidates)
    }


def _plot_candidate_growth() -> None:
    """Plot all independently tracked JTWPA candidate growth trajectories."""
    labels = _candidate_labels()
    fig, axis = plt.subplots(figsize=(10, 5.2))
    for index in range(1, 7):
        branch = f"{index:02d}"
        rows = _read_csv(ROOT / f"track_followup_jtwpa_branch_{branch}.csv")
        drive = [float(row["requested_drive_dbm"]) for row in rows]
        growth = [float(row["growth_rate_per_s"]) / 1e8 for row in rows]
        axis.plot(drive, growth, "o-", ms=3.2, label=labels[branch])
    axis.axhline(0.0, color="black", lw=0.9)
    axis.axvspan(-29.3, -28.2, color="tab:orange", alpha=0.15)
    axis.set_xlabel("Requested pump drive (dBm)")
    axis.set_ylabel("Growth rate (10⁸ s⁻¹)")
    axis.set_title("jc_jtwpa candidate Hill-root trajectories")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.25)
    _save(fig, "jc_jtwpa_candidate_growth")


def _plot_jtwpa_tracking_diagnostics() -> None:
    """Plot growth, overlap, and multiplier magnitude for the primary branch."""
    rows = _read_csv(ROOT / "track_followup_p2_jtwpa_adaptive.csv")
    drive = [float(row["requested_drive_dbm"]) for row in rows]
    growth = [float(row["growth_rate_per_s"]) / 1e8 for row in rows]
    overlap = [_float(row, "mode_overlap") for row in rows]
    magnitude = [float(row["multiplier_magnitude"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(drive, growth, "o-", ms=3.5)
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set_ylabel("Growth (10⁸ s⁻¹)")
    axes[0].set_title("jc_jtwpa corrected continuation diagnostics")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        drive,
        [value if value is not None else float("nan") for value in overlap],
        "o-",
        ms=3.5,
    )
    axes[1].axhline(0.8, color="tab:red", ls="--", label="retry threshold")
    axes[1].set_ylabel("Consecutive mode overlap")
    axes[1].set_ylim(0.75, 1.01)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    axes[2].plot(drive, magnitude, "o-", ms=3.5)
    axes[2].axhline(1.0, color="black", lw=0.9)
    axes[2].set_xlabel("Requested pump drive (dBm)")
    axes[2].set_ylabel("Multiplier magnitude")
    axes[2].grid(alpha=0.25)
    _save(fig, "jc_jtwpa_tracking_diagnostics")


def _plot_reference_ratios() -> None:
    """Plot corrected low-drive ratios against the independent references."""
    values = {
        "jc_jtwpa": 0.817806653925515 / 7.12,
        "ipm_2c_fixed": 0.6844490915635878 / 7.9,
    }
    reference = {"jc_jtwpa": 0.1217, "ipm_2c_fixed": 0.0917}
    labels = list(values)
    positions = list(range(len(labels)))
    width = 0.35
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.bar(
        [position - width / 2 for position in positions],
        [values[label] for label in labels],
        width,
        label="Corrected pump solve",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [reference[label] for label in labels],
        width,
        label="FDTD reference",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("f_a / f_p")
    axis.set_title("Low-drive reference-ratio recheck")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, "low_drive_reference_ratios")


def main(argv: list[str] | None = None) -> None:
    """Generate all continuation plots."""
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    OUTPUT = args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _plot_pump_fold()
    _plot_candidate_growth()
    _plot_jtwpa_tracking_diagnostics()
    _plot_reference_ratios()
    print(f"wrote plots to {OUTPUT}")


if __name__ == "__main__":
    main()
