"""Plot the signal-installed campaign in the Guarcello figure 2(a) layout.

One figure per device, three stacked panels sharing the control axis, matching
``plot_guarcello_mtls.py::plot_fig2a`` so the two can be read side by side:

  * narrowband and wideband gain against the control parameter;
  * the Poincare section, one column of ``V'_PS`` points per control value;
  * a frequency-versus-control spectrogram, with ``f_s``, ``f_p/2``, ``f_p``,
    ``3f_p/2`` and ``2f_p`` marked, since a period doubling shows at the
    half-integer lines and a transition to broadband shows as the whole map
    filling in.

The classifier verdict is deliberately NOT drawn.  With a probe tone installed
the forcing is quasi-periodic, the strobe map no longer has a fixed point, and
every integer-period test in that classifier reports chaos regardless of the
physics; the emitted verdicts are void by construction and printing them beside
these panels would only invite reading them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]

# White where there is no spectral content, through blue for the ordinary comb,
# to red where a line is strong.  A single-hue ramp made a strong line and a
# merely present one hard to separate.
SPECTRUM_CMAP = LinearSegmentedColormap.from_list(
    "spectrum_white_blue_red",
    ["#ffffff", "#cfe0f3", "#5b8fc9", "#1f4e8c", "#7b2d8e", "#c1272d", "#f0a202"],
)
# A third colour, used only for the marked frequencies so they never read as
# data.  Green sits outside the white-blue-red ramp at every level.
MARK_COLOR = "#00a651"


def _signal_hz_fallback(device: str) -> float:
    """Signal frequency for campaigns predating the recorded ``signal_hz``.

    Imported lazily: the kernel module pulls in numba, which costs seconds and
    is pure waste for a plot whose data already carries the field.
    """
    try:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.chaos import run_guarcello_jc_phase5 as phase5
        source = (
            phase5.phase_c_source_path(device)
            if device in {"ipm_2c_fixed", "rf_squid_2393_3wm"}
            else ROOT / "outputs" / "jc_doc_python_designs" / device
        )
        return float(phase5.derive_device_spec(source).signal_ghz) * 1e9
    except Exception:                                            # noqa: BLE001
        return 0.0


def _mark_frequencies(
    pump_hz: float, signal_hz: float, top_ghz: float,
) -> list[tuple[float, str]]:
    """Pump harmonics, pump subharmonics, signal and idler, within the axis."""
    marks: list[tuple[float, str]] = []
    if pump_hz > 0.0:
        for order in range(1, 9):
            marks.append((order * pump_hz / 1e9,
                          "$f_p$" if order == 1 else f"${order}f_p$"))
        # Subharmonics below f_p/3 crowd into an unreadable stack at the axis
        # floor and none of them is a period-doubling indicator anyway.
        for order in (2, 3):
            marks.append((pump_hz / order / 1e9, f"$f_p/{order}$"))
    if signal_hz > 0.0:
        marks.append((signal_hz / 1e9, "$f_s$"))
        if pump_hz > 0.0:
            idler = abs(pump_hz - signal_hz)
            if idler > 0.0:
                marks.append((idler / 1e9, "$f_i$"))
    return [entry for entry in marks if 0.0 < entry[0] <= top_ghz]


def load_points(device_dir: Path) -> list[dict[str, Any]]:
    points = []
    for path in sorted(device_dir.glob("dense_*/result.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("control_value") is None:
            continue
        record["_dir"] = path.parent
        points.append(record)
    points.sort(key=lambda r: float(r["control_value"]))
    return points


def _control_label(points: list[dict[str, Any]]) -> str:
    axis = str(points[0].get("control_axis", "control"))
    return {
        "I_over_I_bound": "pump current / fold boundary current",
        "pump_power_dbm": "pump power [dBm]",
        "pump_current_peak_a": "pump peak current [A]",
    }.get(axis, axis)


# Harmonic-balance reference columns.  Each entry gives the summary CSV and how
# to map its pump_current_peak_a / pump_power_dbm onto that device's control
# axis, since the FDTD campaigns do not all sweep the same quantity.
HB_REFERENCES: dict[str, dict[str, Any]] = {
    "jc_jtwpa": {
        "path": ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811"
                / "jtwpa" / "hb_up_to_failure.csv",
        "control": "pump_power_dbm",
    },
    "jc_fqjtwpa": {
        "path": ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811"
                / "fqjtwpa" / "hb_up_to_failure.csv",
        "control": "pump_power_dbm",
    },
    "ipm_2c_fixed": {
        "path": ROOT / ".hybrid_outputs" / "hb_up_7p9_m35_to_m21"
                / "hb_up_to_failure.csv",
        # The 2c campaign sweeps current as a fraction of the fold boundary.
        "control": "pump_current_peak_a",
        "scale": 1.0 / 1.1628e-05,
    },
}


def load_hb(device: str) -> tuple[np.ndarray, np.ndarray, float | None] | None:
    """Return (control, gain_vs_off_db, signal_ghz) for the converged HB rows.

    Harmonic balance stops where its Newton solve stops converging, so the
    curve simply ends; that endpoint is itself the comparison of interest and
    is not padded or extrapolated.
    """
    entry = HB_REFERENCES.get(device)
    if entry is None or not entry["path"].exists():
        return None
    control, gain, signal_ghz = [], [], None
    with entry["path"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("pump_status") not in {"VALID_CONVERGED", "VALID_SOLVED"}:
                continue
            value = row.get("gain_vs_off_db", "")
            if not value:
                continue
            try:
                control.append(float(row[entry["control"]]) * entry.get("scale", 1.0))
                gain.append(float(value))
            except (KeyError, ValueError):
                continue
            if signal_ghz is None and row.get("signal_ghz"):
                signal_ghz = float(row["signal_ghz"])
    if not control:
        return None
    order = np.argsort(control)
    return np.array(control)[order], np.array(gain)[order], signal_ghz


def _spectrogram(points: list[dict[str, Any]]) -> tuple[np.ndarray, ...] | None:
    """Stack the per-point spectra onto one frequency grid.

    Every point of one device runs the same step count and record stride, so the
    grids agree; a point whose grid differs is dropped rather than resampled,
    because silently interpolating one onto another would blur exactly the
    sharp lines this panel exists to show.
    """
    usable = []
    for point in points:
        path = point["_dir"] / "spectrum.npz"
        if not path.exists():
            continue
        data = np.load(path)
        frequency, spectrum = data["frequency_hz"], data["spectrum_db_relative_pump"]
        if frequency.size == 0:
            continue
        usable.append((float(point["control_value"]), frequency, spectrum))
    if not usable:
        return None
    reference = usable[0][1]
    kept = [entry for entry in usable if entry[1].shape == reference.shape]
    dropped = len(usable) - len(kept)
    control = np.array([entry[0] for entry in kept])
    matrix = np.vstack([entry[2] for entry in kept])
    order = np.argsort(control)
    return control[order], reference / 1e9, matrix[order], dropped


def plot_device(device: str, device_dir: Path, output: Path) -> None:
    points = load_points(device_dir)
    if not points:
        print(f"  {device}: no usable points")
        return
    control = np.array([float(p["control_value"]) for p in points])
    limits = (float(control.min()), float(control.max()))
    pump_hz = float(points[0].get("pump_hz", 0.0) or 0.0)
    signal_hz = float(points[0].get("signal_hz", 0.0) or 0.0)
    if signal_hz <= 0.0 and device != "guarcello":
        signal_hz = _signal_hz_fallback(device)

    figure, axes = plt.subplots(3, 1, figsize=(9, 11))

    def column(key: str) -> np.ndarray:
        return np.array([
            np.nan if p.get(key) is None else float(p[key]) for p in points
        ])

    # Panel 1: FDTD against harmonic balance.  Both are pump-on/pump-off
    # ratios, so the comparison carries no power-convention or line-loss
    # assumption.  The HB curve ends where its Newton solve stops converging.
    narrow = column("gain_vs_off_db")
    axes[0].plot(control, narrow, "o-", color="tab:blue", label="FDTD")
    reference = load_hb(device)
    if reference is not None:
        hb_control, hb_gain, hb_signal_ghz = reference
        inside = (hb_control >= limits[0] - 1e-9) & (hb_control <= limits[1] + 1e-9)
        label = "harmonic balance"
        if hb_signal_ghz is not None and signal_hz > 0.0:
            label += f" ($f_s$={hb_signal_ghz:.2f} GHz)"
        axes[0].plot(hb_control[inside], hb_gain[inside], "s--", color="tab:green",
                     markersize=5, label=label)
        if np.any(inside):
            axes[0].plot(hb_control[inside][-1], hb_gain[inside][-1], "x",
                         color="tab:green", markersize=11, markeredgewidth=2,
                         label="HB last converged")
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].set_ylabel("Gain (dB)")
    axes[0].set_xlim(*limits)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    utilization = column("r_j")
    if np.any(np.isfinite(utilization)):
        twin = axes[0].twinx()
        twin.plot(control, utilization, "s--", color="tab:red", markersize=4,
                  linewidth=1.0, alpha=0.7)
        twin.axhline(1.0, color="tab:red", linestyle=":", linewidth=0.8)
        twin.set_ylabel(r"max $|\sin\phi|$", color="tab:red")
        twin.tick_params(axis="y", labelcolor="tab:red")
        twin.set_ylim(0.0, 1.05)

    # Panel 2: Poincare section.  The stored branches are dv/dt at the v = 0
    # crossings, which is the V'_PS of the paper figure, not a voltage.
    drawn = 0
    for point in points:
        path = point["_dir"] / "poincare_branches.npz"
        if not path.exists():
            continue
        branches = np.load(path)["upward"]
        branches = branches[np.isfinite(branches)]
        if branches.size == 0:
            continue
        axes[1].scatter(np.full(branches.size, float(point["control_value"])),
                        branches, s=2, alpha=0.25, color="tab:blue")
        drawn += 1
    axes[1].set_ylabel(r"$V'_{PS}$ (V/s)")
    axes[1].set_xlim(*limits)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: spectrogram.
    stacked = _spectrogram(points)
    if stacked is not None:
        spectra_control, frequency_ghz, matrix, dropped = stacked
        # The weakest drive sits near the denormal floor and the strongest near
        # 0 dB, so a percentile floor spans ~250 dB and the map shows only the
        # overall level shift.  Clip to a fixed dynamic range below the peak,
        # which is what makes the spectral lines visible.
        finite = matrix[np.isfinite(matrix)]
        vmax = float(np.percentile(finite, 99.9)) if finite.size else -70.0
        vmin = vmax - 120.0
        mesh = axes[2].pcolormesh(
            spectra_control, frequency_ghz, matrix.T, shading="auto",
            cmap=SPECTRUM_CMAP, vmin=vmin, vmax=vmax,
        )
        colorbar = figure.colorbar(mesh, ax=axes[2], pad=0.01)
        colorbar.set_label("Fourier amplitude (dB rel. pump)")
        # Leave headroom above 3 f_p so its label is not clipped by the frame.
        top = 3.3 * pump_hz / 1e9 if pump_hz > 0.0 else float(frequency_ghz.max())
        span = limits[1] - limits[0]
        for index, (frequency, label) in enumerate(
            _mark_frequencies(pump_hz, signal_hz, top)
        ):
            axes[2].axhline(frequency, color=MARK_COLOR, linestyle=":",
                            linewidth=1.0, alpha=0.85)
            # Stagger horizontally: f_s and f_i sit within a few hundred MHz of
            # f_p on these devices and their labels would otherwise overprint.
            axes[2].text(limits[0] + 0.012 * span * (index % 3), frequency, label,
                         va="bottom", fontsize=7, color=MARK_COLOR,
                         fontweight="bold")
        axes[2].set_ylim(0.0, top)
        if dropped:
            axes[2].set_title(f"{dropped} point(s) dropped: frequency grid differs",
                              fontsize=8)
    axes[2].set_xlim(*limits)
    axes[2].set_ylabel("Frequency (GHz)")
    axes[2].set_xlabel(_control_label(points))
    axes[2].grid(True, alpha=0.3)

    missing = int(np.sum(~np.isfinite(narrow)))
    figure.suptitle(
        f"{device}: probe-tone gain, bifurcation points, spectrum"
        + (f"  ({missing} point(s) without gain)" if missing else "")
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    runtimes = [float(p["runtime_s"]) for p in points if p.get("runtime_s")]
    if runtimes:
        figure.text(0.5, 0.008,
                    f"Average runtime per point: {np.mean(runtimes):.1f} s",
                    ha="center", va="bottom", fontsize=10)
    for path in (output / f"{device}_signal.png", output / f"{device}_signal.svg"):
        figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"  {device}: {len(points)} points, {drawn} sections "
          f"-> {output / (device + '_signal.png')}")



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "outputs" / "chaos" / "phaseB_signal")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--devices", default=None,
                        help="comma-separated subset; default is every device present")
    args = parser.parse_args()
    output = args.output or (args.input / "figures")
    output.mkdir(parents=True, exist_ok=True)
    devices = (
        [name.strip() for name in args.devices.split(",") if name.strip()]
        if args.devices else
        sorted(path.name for path in args.input.iterdir() if path.is_dir()
               and path.name != "figures")
    )
    print(f"plotting from {args.input}")
    for device in devices:
        plot_device(device, args.input / device, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
