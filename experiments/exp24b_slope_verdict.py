"""Fit q-axis P1dB slopes and generate the exp24b comparison plot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_SLOPES = {
    "production basis": -0.400,
    "pump-depletion bound": -1.000,
    "measured hardware": -2.231,
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read the q-axis summary rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fit_level(rows: list[dict[str, Any]], sidebands: int) -> dict[str, Any]:
    """Fit P1dB against gain for one q level."""
    usable = [
        row for row in rows
        if int(row["q"]) == sidebands
        and row.get("p1db_input_dbm") not in (None, "")
        and row.get("number_of_crossings") == "1"
        and float(row.get("max_power_balance_rel_err", "inf")) < 1e-6
    ]
    x = np.asarray([float(row["small_signal_gain_db"]) for row in usable])
    y = np.asarray([float(row["p1db_input_dbm"]) for row in usable])
    if x.size < 3:
        return {
            "q": sidebands,
            "n": int(x.size),
            "frequencies_ghz": [float(row["signal_ghz"]) for row in usable],
            "slope": None,
            "slope_standard_error": None,
            "intercept": None,
            "fit_rms_db": None,
        }
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    residual_variance = float(np.sum(residual**2) / (x.size - 2))
    standard_error = math.sqrt(residual_variance / sxx)
    return {
        "q": sidebands,
        "n": int(x.size),
        "frequencies_ghz": [float(row["signal_ghz"]) for row in usable],
        "slope": float(slope),
        "slope_standard_error": float(standard_error),
        "intercept": float(intercept),
        "fit_rms_db": float(np.sqrt(np.mean(residual**2))),
    }


def plot_fits(
    rows: list[dict[str, Any]],
    fits: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot both q families and slope-only reference lines."""
    usable = [
        row for row in rows
        if row.get("p1db_input_dbm") not in (None, "")
        and row.get("number_of_crossings") == "1"
        and float(row.get("max_power_balance_rel_err", "inf")) < 1e-6
    ]
    x = np.asarray([float(row["small_signal_gain_db"]) for row in usable])
    y = np.asarray([float(row["p1db_input_dbm"]) for row in usable])
    x_grid = np.linspace(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5, 200)
    x_center = float(np.mean(x))
    y_center = float(np.mean(y))
    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    colors = {1: "tab:blue", 2: "tab:orange"}
    for fit in fits:
        level = int(fit["q"])
        selected = [row for row in usable if int(row["q"]) == level]
        axis.scatter(
            [float(row["small_signal_gain_db"]) for row in selected],
            [float(row["p1db_input_dbm"]) for row in selected],
            color=colors[level],
            label=f"q<={level} data (n={fit['n']})",
        )
        if fit["slope"] is not None:
            axis.plot(
                x_grid,
                float(fit["slope"]) * x_grid + float(fit["intercept"]),
                color=colors[level],
                linewidth=2.0,
                label=(
                    f"q<={level} fit: {fit['slope']:+.3f} "
                    f"+/- {fit['slope_standard_error']:.3f}"
                ),
            )
    for name, slope in REFERENCE_SLOPES.items():
        axis.plot(
            x_grid,
            y_center + slope * (x_grid - x_center),
            linestyle="--",
            linewidth=1.3,
            label=f"{name}: {slope:+.3f} dB/dB",
        )
    axis.set_xlabel("small-signal gain G (dB)")
    axis.set_ylabel("input P1dB (dBm)")
    axis.set_title("exp24b q-axis slope measurement")
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/exp24b_q_axis_slope/q_axis_summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp24b_q_axis_slope")
    )
    args = parser.parse_args()
    rows = read_rows(args.summary)
    common = {
        float(row["signal_ghz"])
        for row in rows
        if row.get("p1db_input_dbm") not in (None, "")
        and row.get("number_of_crossings") == "1"
        and float(row.get("max_power_balance_rel_err", "inf")) < 1e-6
    }
    common = {
        frequency for frequency in common
        if all(
            any(
                int(row["q"]) == sidebands
                and float(row["signal_ghz"]) == frequency
                and row.get("p1db_input_dbm") not in (None, "")
                and row.get("number_of_crossings") == "1"
                and float(row.get("max_power_balance_rel_err", "inf")) < 1e-6
                for row in rows
            )
            for sidebands in (1, 2)
        )
    }
    fit_rows = [row for row in rows if float(row["signal_ghz"]) in common]
    fits = [fit_level(fit_rows, sidebands) for sidebands in (1, 2)]
    slope_difference = None
    combined_error = None
    sigma = None
    if all(fit["slope"] is not None for fit in fits):
        slope_difference = float(fits[1]["slope"] - fits[0]["slope"])
        combined_error = math.sqrt(
            fits[0]["slope_standard_error"] ** 2
            + fits[1]["slope_standard_error"] ** 2
        )
        sigma = abs(slope_difference) / combined_error
    if sigma is None or sigma < 2.0:
        verdict = "NOT_RESOLVED"
        recommendation = "Report the slope change and collect more frequencies."
    elif fits[1]["slope"] < -1.0:
        verdict = "Q_DOMINANT_CAUSE"
        recommendation = "Recommend redesigning build_sideband_matched_basis."
    elif fits[1]["slope"] < fits[0]["slope"]:
        verdict = "Q_CONTRIBUTES_SECOND_MECHANISM"
        recommendation = "Recommend the basis fix and continued investigation."
    else:
        verdict = "INTERCEPT_ONLY"
        recommendation = "Recommend against redesigning the matched basis."
    spot_path = args.output_dir / "q3_spot_check.json"
    spot_checks = (
        json.loads(spot_path.read_text(encoding="utf-8"))
        if spot_path.exists()
        else []
    )
    q3_convergence_ok = bool(spot_checks) and all(
        bool(check["gate_ok"]) for check in spot_checks
    )
    if not q3_convergence_ok:
        recommendation += " The slope verdict is provisional because q<=3 is not converged at every spot check."
    report = {
        "common_frequencies_ghz": sorted(common),
        "dropped_frequencies_ghz": sorted(
            {float(row["signal_ghz"]) for row in rows} - common
        ),
        "fits": fits,
        "slope_difference_q2_minus_q1": slope_difference,
        "combined_slope_standard_error": combined_error,
        "difference_sigma": sigma,
        "verdict": verdict,
        "recommendation": recommendation,
        "q3_spot_checks": spot_checks,
        "q3_convergence_ok": q3_convergence_ok,
        "references": REFERENCE_SLOPES,
        "absolute_p1db_caveat": (
            "Dense h=[1..6] changes absolute gain relative to the odd-only "
            "production basis; only within-q slopes are interpreted."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "slope_verdict.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    plot_fits(rows, fits, args.output_dir / "p1db_vs_gain_q_axis.png")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
