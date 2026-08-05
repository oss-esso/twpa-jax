"""Plot the Phase-6 model compression curves paired with measurement, two ways.

Same four-panel layout as ``exp45_curves_two_way.py`` (same-frequency pairing,
gain-matched pairing, model-alone, measured-alone), but sourced from the
post-Phase-1-3 pipeline instead of exp45's pre-fix constants: measured
on-chip power via the frequency-dependent ``loss_B1`` model
(``MeasurementCube.on_chip_power_dbm``), not exp45's fabricated flat 72.5 dB;
model power computed directly on-chip from ``signal_current_a`` (Norton,
zero attenuation), NOT from the CSV's own ``signal_power_dbm`` column.

That column is on-chip power PLUS ``run_compression.py``'s
``_resolve_attenuation()``, which for a ``--circuit-dir`` run with no
explicit ``--attenuation-db`` evaluates ``default_loss_model()`` (``loss_A10``,
the PUMP line) at the PUMP frequency -- wrong model and wrong frequency for a
SIGNAL-power label (should be ``signal_line_loss_model()``/``loss_B1`` at the
signal frequency; measured ~25.1-25.3 dB too low across 6.55-8.15 GHz). Even
with that fixed, comparing it against ``cube.on_chip_power_dbm`` (instrument
power MINUS attenuation) would still be backwards in direction. On-chip vs
on-chip sidesteps both: the model's current already IS the on-chip current,
no cable exists in the simulation, so no loss model is needed on that side at
all. This is the current version. See
``docs/development/psat_comparison_fix_plan.md`` Phase 2, which fixed the
measured side's calibration but never touched
``run_compression.py``'s own signal-power labeling -- that is still a live
defect in ``signal_power_dbm``/``p1db_input_dbm`` whenever those fields are
read as instrument-referred numbers, just not one this plot needs to care
about.

Default ``--run-dir`` is the merged sweep (``D:/tmp/phase6_model_sweep_merged``):
a low-current run (1e-11..3e-7 A, resolves the small-signal G0 anchor) and a
high-current run (5e-7..6e-6 A, reaches real compression) concatenated per
frequency at the CSV level and sorted by ``signal_current_a`` -- no re-solve,
see ``docs/development/psat_comparison_fix_plan.md`` Phase 5/7. 7 of 8
frequencies cross 1 dB compression in the merged data (fp=7.725 GHz,
I=1.077e-5 A pump, provisional/unrefined operating point -- see
``outputs/fit_operating_point_phase4/fit_operating_point.json``); the
low-current-only sweep this module originally shipped against crossed only 1
of 8, since it never reached compression current. The 8th frequency
(8.15 GHz) has no high-current points (that run was killed before reaching
it) and still reads as a still-rising small-signal trace, not a knee.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.measured_psat_pipeline import (
    DEFAULT_CUBE_PATH, MEAS_PUMP_GHZ, PUMP_EXCLUSION_GHZ, load_cube,
)
from twpa_solver.ports import port_available_power_w

SAVGOL_WINDOW = 21
SAVGOL_ORDER = 3
Z0_OHM = 50.0


def model_curves(run_dir: Path) -> list[dict[str, object]]:
    """Solved (frequency, power, gain, G0) per model frequency.

    ``power_dbm`` is computed directly from ``signal_current_a`` via the
    Norton on-chip relation (``port_available_power_w``, zero attenuation)
    -- NOT from the CSV's own ``signal_power_dbm``, which is on-chip power
    PLUS ``run_compression.py``'s ``attenuation_db`` (an external/instrument-
    referred label, and one that itself uses the wrong loss model for a
    signal-power quantity -- see this module's docstring). The measured
    side's ``cube.on_chip_power_dbm`` is instrument power MINUS attenuation,
    i.e. already on-chip -- so on-chip-to-on-chip is the only combination
    that is not mismatched in convention, and it needs no loss model on the
    model side at all: the model's current already IS the on-chip current,
    no cable exists in the simulation.
    """
    curves: list[dict[str, object]] = []
    for directory in sorted(run_dir.glob("frequency_*")):
        points = directory / "compression_points.csv"
        if not points.exists():
            continue
        match = re.search(r"_(\d+\.\d+)ghz", directory.name)
        if match is None:
            continue
        with points.open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(stream) if r["status"] == "VALID_SOLVED"]
        if not rows:
            continue
        rows.sort(key=lambda r: float(r["signal_current_a"]))
        current = np.array([float(r["signal_current_a"]) for r in rows])
        power = np.array([
            10.0 * np.log10(port_available_power_w(i, Z0_OHM, convention="norton") / 1.0e-3)
            for i in current
        ])
        gain = np.array([float(r["gain_vs_off_db"]) for r in rows])
        curves.append({
            "signal_ghz": float(match.group(1)),
            "power_dbm": power,
            "gain_db": gain,
            "g0_db": float(gain[0]),
        })
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("D:/tmp/phase6_model_sweep_merged"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/presentation/phase6_2c_curves_two_way.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cube = load_cube(DEFAULT_CUBE_PATH)
    freq = cube.frequency_ghz
    smooth = savgol_filter(cube.response_db, SAVGOL_WINDOW, SAVGOL_ORDER, axis=0)
    g0 = np.median(cube.response_db[:10, :], axis=0)
    usable = np.abs(freq - MEAS_PUMP_GHZ) > PUMP_EXCLUSION_GHZ

    curves = model_curves(args.run_dir)
    if not curves:
        raise SystemExit(f"no model curves under {args.run_dir}")

    ghz = np.array([c["signal_ghz"] for c in curves])
    norm = Normalize(vmin=ghz.min(), vmax=ghz.max())
    colormap = plt.get_cmap("viridis")

    fig, grid = plt.subplots(2, 2, figsize=(16.0, 12.0), sharex=True, sharey=True)
    axes = grid.ravel()
    pairing: list[dict[str, float]] = []
    # xlim must cover BOTH sides, not just the model curves -- otherwise a
    # narrower model current range silently crops the measured knee out of
    # the plot, making it look like "measurement isn't in the picture" when
    # it's actually just off-frame (caught 2026-08-05: model's tested range
    # topped out well short of where Themis compresses, and an xlim sized
    # only to the model curves cut the real measured knee off entirely).
    # Only the specific measured columns actually plotted (same-frequency +
    # gain-matched) count, not the whole cube -- unrelated columns elsewhere
    # in the band would otherwise pull the axis out further than needed.
    all_power = [c["power_dbm"] for c in curves]
    for curve in curves:
        same_index = int(np.argmin(np.abs(freq - curve["signal_ghz"])))
        all_power.append(cube.on_chip_power_dbm(same_index))
        candidates = np.where(usable, np.abs(g0 - curve["g0_db"]), np.inf)
        gain_index = int(np.argmin(candidates))
        all_power.append(cube.on_chip_power_dbm(gain_index))
    power_lo = min(float(p.min()) for p in all_power) - 5.0
    power_hi = max(float(p.max()) for p in all_power) + 5.0
    for curve in curves:
        color = colormap(norm(curve["signal_ghz"]))
        same_index = int(np.argmin(np.abs(freq - curve["signal_ghz"])))
        power_at_same = cube.on_chip_power_dbm(same_index)
        candidates = np.where(usable, np.abs(g0 - curve["g0_db"]), np.inf)
        gain_index = int(np.argmin(candidates))
        power_at_gain = cube.on_chip_power_dbm(gain_index)

        for ax, index, power_axis in (
            (axes[0], same_index, power_at_same), (axes[1], gain_index, power_at_gain),
        ):
            ax.plot(power_axis, smooth[:, index], lw=1.6, color=color, alpha=0.85)
            ax.plot(curve["power_dbm"], curve["gain_db"], lw=1.4, ls="--",
                    color=color, marker="o", ms=3.5, mfc="white", mew=1.0)
        axes[2].plot(curve["power_dbm"], curve["gain_db"], lw=1.6, ls="--",
                     color=color, marker="o", ms=4.0, mfc="white", mew=1.1)
        axes[3].plot(power_at_same, smooth[:, same_index], lw=1.8, color=color)

        pairing.append({
            "model_ghz": float(curve["signal_ghz"]),
            "model_g0_db": float(curve["g0_db"]),
            "same_frequency_meas_g0_db": float(g0[same_index]),
            "gain_matched_meas_ghz": float(freq[gain_index]),
            "gain_matched_meas_g0_db": float(g0[gain_index]),
            "gain_match_residual_db": float(abs(g0[gain_index] - curve["g0_db"])),
        })

    mean_same = float(np.mean([
        p["same_frequency_meas_g0_db"] - p["model_g0_db"] for p in pairing
    ]))
    mean_matched = float(np.mean([p["gain_match_residual_db"] for p in pairing]))
    axes[0].set_title(
        "both -- paired at the SAME FREQUENCY\n"
        f"measured $G_0$ exceeds model by {mean_same:.2f} dB on average, "
        "so the curves start apart"
    )
    axes[1].set_title(
        "both -- paired at MATCHED $G_0$\n"
        f"gain matched to {mean_matched:.3f} dB; "
        "this is the comparison that isolates saturation"
    )
    axes[2].set_title(f"model alone -- {len(curves)} solved frequencies")
    axes[3].set_title(
        "Themis alone -- the same frequencies\n"
        f"Savitzky-Golay, {SAVGOL_WINDOW}-point window, order {SAVGOL_ORDER}"
    )
    for ax in axes:
        ax.set_xlim(power_lo, power_hi)
        ax.grid(alpha=0.3)
    for ax in axes[2:]:
        ax.set_xlabel("signal power at device (dBm)")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("gain (dB)")

    solid = plt.Line2D([], [], color="0.3", lw=1.6, label="Themis measurement")
    dashed = plt.Line2D([], [], color="0.3", lw=1.4, ls="--", marker="o",
                        ms=3.5, mfc="white", label="multitone HB model")
    axes[0].legend(handles=[solid, dashed], fontsize=9, loc="lower left")

    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=colormap), ax=axes, fraction=0.025, pad=0.015
    )
    bar.set_label("model signal frequency (GHz)")
    n_crossed = sum(
        1 for c in curves if float(c["gain_db"].min()) <= float(c["g0_db"]) - 1.0
    )
    fig.suptitle(
        "2c compression curves (Phase 6, corrected loss_B1/Norton pipeline) -- "
        f"model (pump 7.725 GHz, provisional I=1.077e-5 A) vs Themis 105C5 "
        f"(pump {MEAS_PUMP_GHZ} GHz), {len(curves)} frequencies, "
        f"{n_crossed} of {len(curves)} crossed 1 dB compression",
        fontsize=12,
    )
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.output}")
    print(f'{"model GHz":>10} {"model G0":>9} {"same-f G0":>10} '
          f'{"matched GHz":>12} {"matched G0":>11} {"resid":>7}')
    for row in pairing:
        print(
            f'{row["model_ghz"]:10.2f} {row["model_g0_db"]:9.2f} '
            f'{row["same_frequency_meas_g0_db"]:10.2f} '
            f'{row["gain_matched_meas_ghz"]:12.3f} '
            f'{row["gain_matched_meas_g0_db"]:11.2f} '
            f'{row["gain_match_residual_db"]:7.3f}'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
