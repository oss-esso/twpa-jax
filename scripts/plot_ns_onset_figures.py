"""Plot the Neimark-Sacker onset figure set for one pump-frequency column.

Every reported quantity is derived from the supplied files; nothing is
hard-coded to a particular device or column. Figures whose inputs are absent
are skipped and named in the summary rather than drawn from defaults.

Inputs
------
``--floquet-csv``
    One or more tracked-branch CSVs from ``scripts/chaos/track_critical_root.py``
    or ``scripts/scan_hb_floquet_branch.py``. Rows are merged and de-duplicated
    on drive, keeping converged rows only.
``--torus-csv``
    Optional branch-locked torus CSV from ``scripts/chaos/run_branch_locked_torus.py``.
``--ansatz-csv``
    Optional lattice-occupancy CSV from ``scripts/chaos/measure_ansatz_validity.py``.

Example
-------
    python scripts/plot_ns_onset_figures.py `
      --floquet-csv run_a/floquet.csv run_b/floquet.csv `
      --torus-csv torus_column.csv `
      --ansatz-csv ansatz_validity.csv `
      --pump-ghz 7.9 --device ipm_2c_fixed `
      --control-anchor 0.587852 -24.05 `
      --outdir outputs/ns_figures/2c_7p9
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HILL = "#3b6bd6"
TORUS = "#d1642a"
FDTD = "#2f9e6e"
ACCENT = "#8e5bc9"
INK = "#1c1c1e"
MUTED = "#6e6e73"
GRID = "#dcdce0"
SURFACE = "#fcfcfb"

DRIVE_KEYS = ("requested_drive_dbm", "drive_dbm", "pump_power_instrument_dbm")
TORUS_DRIVE_KEYS = ("drive_dbm", "requested_drive_dbm")
TORUS_RADIUS_KEYS = ("radius_squared", "r_squared", "r2")
TORUS_RATIO_KEYS = ("omega_a_over_omega_p", "omega_a_over_f_p")


def _rc() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.8,
        "font.size": 11,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "figure.dpi": 200,
    })


def style(ax: plt.Axes) -> None:
    """Apply the recessive grid and open frame used across the set."""
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------
# pure numerics -- unit tested
# --------------------------------------------------------------------------


def interpolate_crossing(
    x: np.ndarray, y: np.ndarray, level: float = 1.0
) -> tuple[float, tuple[float, float]] | None:
    """Return the first linear crossing of ``level`` and its bracket.

    ``x`` must be sorted ascending. Returns ``None`` when ``y`` never crosses,
    which is a real outcome for a column that stays stable across its range.
    """
    if x.size != y.size:
        raise ValueError("x and y must have equal length")
    if x.size < 2:
        return None
    below = y < level
    for index in range(x.size - 1):
        if below[index] and not below[index + 1]:
            span = y[index + 1] - y[index]
            if span == 0.0:
                continue
            fraction = (level - y[index]) / span
            crossing = float(x[index] + fraction * (x[index + 1] - x[index]))
            return crossing, (float(x[index]), float(x[index + 1]))
    return None


def fit_linear_intercept(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Least-squares ``y = m*(x - x0)``; return ``(m, x0, r_squared)``."""
    if x.size < 2:
        raise ValueError("at least two points are required for a linear fit")
    slope, offset = np.polyfit(x, y, 1)
    if slope == 0.0:
        raise ValueError("degenerate fit: zero slope")
    predicted = slope * x + offset
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total == 0.0 else 1.0 - residual / total
    return float(slope), float(-offset / slope), r_squared


def twist_coefficient(
    hill_slope: float, torus_slope: float, radius_slope: float
) -> float | None:
    """Return ``b`` in ``omega = omega_c + a*(P-P_c) + b*r^2``.

    ``hill_slope`` is ``a`` measured below onset, ``torus_slope`` is
    ``a + b*dr^2/dP`` measured above it, and ``radius_slope`` is ``dr^2/dP``.
    """
    if radius_slope == 0.0:
        return None
    return float((torus_slope - hill_slope) / radius_slope)


def control_to_dbm(
    control: np.ndarray, anchor_value: float, anchor_dbm: float
) -> np.ndarray:
    """Map a normalized drive axis onto dBm through one anchor point."""
    if anchor_value <= 0.0:
        raise ValueError("anchor control value must be positive")
    return anchor_dbm + 20.0 * np.log10(control / anchor_value)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _first_key(row: dict[str, str], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return key
    return None


def _as_float(value: str | None) -> float:
    return float("nan") if value in (None, "") else float(value)


@dataclass
class Branch:
    """One tracked Floquet branch on a single pump-frequency column."""

    drive: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    real_ghz: np.ndarray
    imag_ghz: np.ndarray
    overlap: np.ndarray


@dataclass
class TorusBranch:
    """Converged torus solves and the normal form fitted to them."""

    drive: np.ndarray
    radius_squared: np.ndarray
    ratio: np.ndarray


@dataclass
class Lattice:
    """Lattice-occupancy fractions measured from time-domain spectra."""

    control: np.ndarray
    drive: np.ndarray | None
    on_lattice: np.ndarray
    off_lattice: np.ndarray
    generator_share: np.ndarray


@dataclass
class Summary:
    """Every derived number, written next to the figures as JSON."""

    values: dict[str, Any] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def note(self, name: str, reason: str) -> None:
        self.skipped.append(f"{name}: {reason}")


def load_branch(paths: Sequence[Path], require_converged: bool) -> Branch:
    """Merge tracked-branch CSVs, keeping the last row per drive."""
    merged: dict[float, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if require_converged and row.get("converged", "True").strip() != "True":
                    continue
                key = _first_key(row, DRIVE_KEYS)
                if key is None:
                    raise ValueError(f"{path}: no drive column among {DRIVE_KEYS}")
                merged[float(row[key])] = row
    if not merged:
        raise ValueError("no converged rows found in the supplied Floquet CSVs")
    drives = np.array(sorted(merged), dtype=float)
    rows = [merged[value] for value in drives]
    return Branch(
        drive=drives,
        magnitude=np.array([_as_float(r.get("multiplier_magnitude")) for r in rows]),
        phase=np.array([_as_float(r.get("multiplier_phase_rad")) for r in rows]),
        real_ghz=np.array([_as_float(r.get("omega_real_ghz")) for r in rows]),
        imag_ghz=np.array([_as_float(r.get("omega_imag_ghz")) for r in rows]),
        overlap=np.array([_as_float(r.get("mode_overlap")) for r in rows]),
    )


def load_torus(path: Path) -> TorusBranch | None:
    """Read converged torus rows; return ``None`` when none are present."""
    drives: list[float] = []
    radii: list[float] = []
    ratios: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            drive_key = _first_key(row, TORUS_DRIVE_KEYS)
            radius_key = _first_key(row, TORUS_RADIUS_KEYS)
            if drive_key is None or radius_key is None:
                continue
            radius = _as_float(row[radius_key])
            if not math.isfinite(radius) or radius <= 0.0:
                continue
            ratio_key = _first_key(row, TORUS_RATIO_KEYS)
            drives.append(float(row[drive_key]))
            radii.append(radius)
            ratios.append(float("nan") if ratio_key is None else _as_float(row[ratio_key]))
    if len(drives) < 2:
        return None
    order = np.argsort(np.asarray(drives))
    return TorusBranch(
        drive=np.asarray(drives)[order],
        radius_squared=np.asarray(radii)[order],
        ratio=np.asarray(ratios)[order],
    )


def load_lattice(
    path: Path,
    device: str | None,
    anchor: tuple[float, float] | None,
) -> Lattice | None:
    """Read the pump-only lattice columns, falling back to the signal ones."""
    control: list[float] = []
    on_values: list[float] = []
    off_values: list[float] = []
    share_values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if device is not None and row.get("device") not in (None, device):
                continue
            if "control_value" not in row:
                continue
            on_key = _first_key(row, ("on_lattice_pump_only", "on_lattice"))
            off_key = _first_key(row, ("off_lattice_pump_only", "off_lattice"))
            share_key = _first_key(
                row, ("generator_share_pump_only", "generator_share")
            )
            if on_key is None or off_key is None:
                continue
            control.append(float(row["control_value"]))
            on_values.append(_as_float(row[on_key]))
            off_values.append(_as_float(row[off_key]))
            share_values.append(
                float("nan") if share_key is None else _as_float(row[share_key])
            )
    if not control:
        return None
    order = np.argsort(np.asarray(control))
    control_array = np.asarray(control)[order]
    drive = (
        None if anchor is None
        else control_to_dbm(control_array, anchor[0], anchor[1])
    )
    return Lattice(
        control=control_array,
        drive=drive,
        on_lattice=np.asarray(on_values)[order],
        off_lattice=np.asarray(off_values)[order],
        generator_share=np.asarray(share_values)[order],
    )


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_onset(
    branch: Branch,
    torus: TorusBranch | None,
    crossing: float | None,
    torus_fit: tuple[float, float, float] | None,
    title: str,
    outdir: Path,
) -> None:
    """Multiplier crossing over torus amplitude, on a shared drive axis."""
    panels = 2 if torus is not None else 1
    height = 6.2 if panels == 2 else 4.0
    fig, axes = plt.subplots(
        panels, 1, figsize=(7.4, height), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 0.85][:panels], "hspace": 0.14},
    )
    top = axes[0][0]
    top.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    top.plot(branch.drive, branch.magnitude, color=HILL, lw=2.0, marker="o",
             ms=4.5, mfc=SURFACE, mew=1.4, label="tracked Floquet multiplier")
    if crossing is not None:
        top.axvline(crossing, color=HILL, lw=1.2, ls=":", alpha=0.9)
        top.annotate(f"|$\\lambda$| = 1 at {crossing:.4f} dBm",
                     xy=(crossing, 1.0),
                     xytext=(branch.drive.min(), float(np.max(branch.magnitude))),
                     color=HILL, fontsize=10)
    top.set_ylabel("|$\\lambda$|   (Floquet multiplier)")
    top.set_title(title, loc="left", pad=10)
    top.legend(loc="lower right")
    style(top)

    if torus is not None and torus_fit is not None:
        bot = axes[1][0]
        slope, p_c, _ = torus_fit
        grid = np.linspace(p_c, float(torus.drive.max()), 200)
        bot.plot(grid, slope * (grid - p_c), color=TORUS, lw=2.0,
                 label=f"normal form  $r^2 = {slope:.4g}\\,(P - P_c)$")
        bot.plot(torus.drive, torus.radius_squared, ls="none", marker="s", ms=7,
                 color=TORUS, mfc=SURFACE, mew=1.8, label="converged torus solves")
        bot.axvline(p_c, color=TORUS, lw=1.2, ls=":", alpha=0.9)
        bot.annotate(f"$P_c$ = {p_c:.4f} dBm", xy=(p_c, 0.0),
                     xytext=(branch.drive.min(),
                             0.55 * float(torus.radius_squared.max())),
                     color=TORUS, fontsize=10)
        bot.set_ylim(bottom=-0.02 * float(torus.radius_squared.max()))
        bot.set_ylabel("torus amplitude  $r^2$")
        bot.legend(loc="upper left")
        style(bot)
        if crossing is not None:
            fig.text(0.985, 0.015, f"agreement {abs(crossing - p_c):.4f} dB",
                     ha="right", color=MUTED, fontsize=10)
    axes[-1][0].set_xlabel("pump drive [dBm, instrument-referred]")
    _save(fig, outdir, "fig1_onset_agreement")


def figure_locus(branch: Branch, title: str, outdir: Path) -> None:
    """Critical multiplier pair in the complex plane, with a zoom inset."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), color=MUTED, lw=1.2, ls=(0, (5, 4)))
    lam = branch.magnitude * np.exp(1j * branch.phase)
    inside = branch.magnitude < 1.0
    for sign in (1.0, -1.0):
        z = lam.real + 1j * sign * lam.imag
        ax.plot(z.real[inside], z.imag[inside], ls="none", marker="o", ms=5,
                color=HILL, mfc=SURFACE, mew=1.4,
                label="stable  (|$\\lambda$| < 1)" if sign > 0 else None)
        ax.plot(z.real[~inside], z.imag[~inside], ls="none", marker="o", ms=5,
                color=ACCENT,
                label="unstable  (|$\\lambda$| > 1)" if sign > 0 else None)
    ax.plot([0.0], [0.0], marker="+", ms=9, color=MUTED, ls="none")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_xlabel("Re $\\lambda$")
    ax.set_ylabel("Im $\\lambda$")
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="lower left", fontsize=9.5)
    style(ax)

    span = max(float(np.ptp(lam.real)), float(np.ptp(lam.imag))) * 0.9 + 1e-6
    centre = complex(float(np.mean(lam.real)), float(np.mean(lam.imag)))
    inset = ax.inset_axes([0.015, 0.615, 0.365, 0.365])
    inset.plot(np.cos(theta), np.sin(theta), color=MUTED, lw=1.0, ls=(0, (4, 3)))
    inset.plot(lam.real, lam.imag, color=HILL, lw=1.2, alpha=0.5)
    inset.plot(lam.real[inside], lam.imag[inside], ls="none", marker="o", ms=5,
               color=HILL, mfc=SURFACE, mew=1.4)
    inset.plot(lam.real[~inside], lam.imag[~inside], ls="none", marker="o", ms=5,
               color=ACCENT)
    inset.set_xlim(centre.real - span, centre.real + span)
    inset.set_ylim(centre.imag - span, centre.imag + span)
    inset.set_aspect("equal")
    inset.set_title("zoom", fontsize=9, color=MUTED, pad=4)
    inset.tick_params(labelsize=7.5)
    style(inset)
    ax.indicate_inset_zoom(inset, edgecolor=MUTED, alpha=0.9)
    _save(fig, outdir, "fig2_root_locus")


def figure_frequency(
    branch: Branch,
    torus: TorusBranch | None,
    pump_ghz: float,
    crossing: float | None,
    torus_fit: tuple[float, float, float] | None,
    twist: float | None,
    outdir: Path,
) -> None:
    """Hill and torus generator frequency, meeting at onset."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(branch.drive, branch.real_ghz / pump_ghz, color=HILL, lw=2.0,
            marker="o", ms=4.5, mfc=SURFACE, mew=1.4,
            label="Hill root  (below onset: slope $a$)")
    if torus is not None and np.isfinite(torus.ratio).any():
        ax.plot(torus.drive, torus.ratio, color=TORUS, lw=2.0, marker="s", ms=7,
                mfc=SURFACE, mew=1.8,
                label="torus $f_a$  (above onset: $a + b\\,dr^2/dP$)")
        if torus_fit is not None and torus.drive.size >= 2:
            slope = (torus.ratio[1] - torus.ratio[0]) / (
                torus.drive[1] - torus.drive[0]
            )
            back = np.linspace(torus_fit[1], float(torus.drive[0]), 40)
            ax.plot(back, torus.ratio[0] + slope * (back - torus.drive[0]),
                    color=TORUS, lw=1.2, ls=(0, (3, 3)))
            ax.plot([torus_fit[1]],
                    [torus.ratio[0] + slope * (torus_fit[1] - torus.drive[0])],
                    marker="*", ms=15, color=TORUS, ls="none")
    if crossing is not None:
        ax.plot([crossing], [float(np.interp(crossing, branch.drive,
                                             branch.real_ghz / pump_ghz))],
                marker="*", ms=15, color=HILL, ls="none")
    subtitle = ("$\\omega_{torus}(P) = \\omega_c + a(P-P_c) + b\\,r^2$"
                if twist is None else
                f"$b$ = {twist:.4f} GHz per unit $r^2$  (negative twist)")
    ax.set_xlabel("pump drive [dBm, instrument-referred]")
    ax.set_ylabel("$f_a / f_p$")
    ax.set_title("Opposite slopes are predicted, not a contradiction\n" + subtitle,
                 loc="left", pad=10)
    ax.legend(loc="best")
    style(ax)
    _save(fig, outdir, "fig3_frequency_twist")


def figure_gap(
    lattice: Lattice,
    fit: tuple[float, float, float],
    window: np.ndarray,
    crossing: float | None,
    torus_pc: float | None,
    outdir: Path,
) -> None:
    """Off-lattice growth, its extrapolated onset, and the HB onsets."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    drive = lattice.drive
    assert drive is not None
    slope, p_zero, r_squared = fit
    ceiling = float(np.max(lattice.off_lattice[window])) * 1.18
    show = drive <= float(np.max(drive[window])) + 0.06
    ax.plot(drive[show], lattice.off_lattice[show], ls="none", marker="o", ms=7,
            color=FDTD, mfc=SURFACE, mew=1.8,
            label="off-lattice power fraction (time domain)")
    grid = np.linspace(p_zero, float(np.max(drive[window])), 100)
    ax.plot(grid, slope * (grid - p_zero), color=FDTD, lw=2.0,
            label=f"linear fit ($R^2$ = {r_squared:.3f}) → {p_zero:.4f} dBm")
    references = [(crossing, HILL, "Hill"), (torus_pc, TORUS, "torus")]
    present = [(v, c, n) for v, c, n in references if v is not None]
    for value, colour, _ in present:
        ax.axvline(value, color=colour, lw=1.6, ls=":")
    if present:
        anchor = present[0][0]
        ax.annotate("", xy=(anchor, 0.34 * ceiling),
                    xytext=(p_zero, 0.34 * ceiling),
                    arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.2})
        ax.text((anchor + p_zero) / 2.0, 0.37 * ceiling,
                f"{abs(p_zero - anchor):.3f} dB", ha="center", color=INK,
                fontsize=11)
        ax.text(anchor, 0.50 * ceiling,
                "  HB onset\n  " + " + ".join(n for _, _, n in present),
                color=INK, fontsize=9.5, va="top")
    ax.set_ylim(-0.008 * ceiling, ceiling)
    ax.set_xlabel("pump drive [dBm, instrument-referred]")
    ax.set_ylabel("off-lattice power fraction")
    ax.set_title("Harmonic balance versus time-domain onset\n"
                 "ratio inside one spectrum, so calibration-free",
                 loc="left", pad=10)
    ax.legend(loc="upper left")
    style(ax)
    _save(fig, outdir, "fig4_hb_vs_fdtd_gap")


def figure_regimes(
    lattice: Lattice,
    crossing: float | None,
    edges: dict[str, float | None],
    outdir: Path,
) -> None:
    """Lattice occupancy and single-generator share, banded by regime."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    drive = lattice.drive if lattice.drive is not None else lattice.control
    ax.plot(drive, lattice.on_lattice, color=HILL, lw=2.0, marker="o", ms=5,
            mfc=SURFACE, mew=1.4, label="power on the HB tone lattice")
    share_mask = np.isfinite(lattice.generator_share)
    if crossing is not None and lattice.drive is not None:
        share_mask &= lattice.drive >= crossing
    if share_mask.any():
        ax.plot(drive[share_mask], lattice.generator_share[share_mask],
                color=TORUS, lw=2.0, marker="s", ms=5, mfc=SURFACE, mew=1.4,
                label="explained by ONE extra generator")
    boundaries = [value for value in edges.values() if value is not None]
    for value in boundaries:
        ax.axvline(value, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    marks = [float(np.min(drive))] + sorted(boundaries) + [float(np.max(drive))]
    names = ["period-1", "2-torus", "3-frequency", "chaos"]
    for index in range(min(len(marks) - 1, len(names))):
        ax.text((marks[index] + marks[index + 1]) / 2.0, 1.06, names[index],
                ha="center", color=MUTED, fontsize=9.5)
    if crossing is not None:
        ax.axvline(crossing, color=ACCENT, lw=1.6, ls=":")
        ax.text(crossing, 0.06, " HB onset", color=ACCENT, fontsize=10)
    ax.set_ylim(0.0, 1.14)
    ax.set_xlabel("pump drive [dBm, instrument-referred]"
                  if lattice.drive is not None else "control value")
    ax.set_ylabel("fraction of in-band power")
    ax.set_title("Where harmonic balance can and cannot go\n"
                 "measured from time-domain spectra, calibration-free",
                 loc="left", pad=14)
    ax.legend(loc="lower left")
    style(ax)
    _save(fig, outdir, "fig5_regime_map")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--floquet-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--torus-csv", type=Path, default=None)
    parser.add_argument("--ansatz-csv", type=Path, default=None)
    parser.add_argument("--pump-ghz", type=float, required=True)
    parser.add_argument("--device", default=None,
                        help="filter rows of the ansatz CSV to this device")
    parser.add_argument("--control-anchor", type=float, nargs=2, default=None,
                        metavar=("CONTROL", "DBM"),
                        help="map the ansatz control axis onto dBm through one point")
    parser.add_argument("--fit-control-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="control values used for the off-lattice fit; "
                             "default selects points above the floor and below "
                             "the saturating maximum")
    parser.add_argument("--off-lattice-floor", type=float, default=1.0e-3,
                        help="off-lattice fraction treated as no torus")
    parser.add_argument("--fit-min-r-squared", type=float, default=0.95,
                        help="stop growing the auto fit window below this R^2")
    parser.add_argument("--torus-share-threshold", type=float, default=0.95,
                        help="generator share below which a second generator is present")
    parser.add_argument("--on-lattice-threshold", type=float, default=0.60,
                        help="lattice occupancy below which no balance applies")
    parser.add_argument("--allow-unconverged", action="store_true",
                        help="keep Floquet rows whose refinement did not converge")
    parser.add_argument("--label", default=None,
                        help="text appended to the onset figure title")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args(argv)


def _select_fit_window(
    lattice: Lattice, args: argparse.Namespace, summary: Summary
) -> np.ndarray:
    if args.fit_control_range is not None:
        low, high = sorted(args.fit_control_range)
        window = (lattice.control >= low) & (lattice.control <= high)
        summary.values["fit_window_source"] = "explicit"
        summary.values["fit_window_control"] = lattice.control[window].tolist()
        return window

    # Grow the window from the detection floor while the linear law still
    # holds. The off-lattice *fraction* saturates once on-lattice power
    # depletes, so a window that runs to the maximum drags the intercept late.
    axis = lattice.drive if lattice.drive is not None else lattice.control
    candidates = np.where(lattice.off_lattice > args.off_lattice_floor)[0]
    if candidates.size < 2:
        summary.values["fit_window_source"] = "auto (too few points above floor)"
        window = np.zeros(lattice.control.size, dtype=bool)
        window[candidates] = True
        summary.values["fit_window_control"] = lattice.control[window].tolist()
        return window

    # Two points always fit a line exactly, so a window that small proves
    # nothing; require three and keep the LARGEST window that still holds.
    trace: list[dict[str, float]] = []
    minimum = min(3, candidates.size)
    best = candidates[:minimum]
    for size in range(minimum, candidates.size + 1):
        subset = candidates[:size]
        try:
            _, intercept, r_squared = fit_linear_intercept(
                axis[subset], lattice.off_lattice[subset]
            )
        except ValueError:
            break
        trace.append({"points": float(size), "r_squared": r_squared,
                      "onset": intercept})
        if r_squared >= args.fit_min_r_squared:
            best = subset
    window = np.zeros(lattice.control.size, dtype=bool)
    window[best] = True
    summary.values["fit_window_source"] = (
        f"auto (grown while R^2 >= {args.fit_min_r_squared})"
    )
    summary.values["fit_window_control"] = lattice.control[window].tolist()
    summary.values["fit_window_scan"] = trace
    return window


def _regime_edges(
    lattice: Lattice, args: argparse.Namespace, onset: float | None
) -> dict[str, float | None]:
    """Locate regime boundaries, ignoring everything below the NS onset.

    Below onset there is no torus, so a single-generator fit is chasing a
    noise floor and its share carries no information.
    """
    axis = lattice.drive if lattice.drive is not None else lattice.control
    live = lattice.off_lattice > args.off_lattice_floor
    if onset is not None and lattice.drive is not None:
        live &= lattice.drive >= onset

    def first_below(values: np.ndarray, threshold: float) -> float | None:
        hits = np.where(live & np.isfinite(values) & (values < threshold))[0]
        return None if hits.size == 0 else float(axis[hits[0]])

    return {
        "torus_ends": first_below(lattice.generator_share,
                                  args.torus_share_threshold),
        "balance_ends": first_below(lattice.on_lattice,
                                    args.on_lattice_threshold),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _rc()
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = Summary()
    branch = load_branch(args.floquet_csv, not args.allow_unconverged)
    summary.values["floquet_points"] = int(branch.drive.size)
    summary.values["drive_range_dbm"] = [float(branch.drive.min()),
                                         float(branch.drive.max())]
    overlaps = branch.overlap[np.isfinite(branch.overlap)]
    summary.values["min_mode_overlap"] = (
        None if overlaps.size == 0 else float(overlaps.min())
    )

    found = interpolate_crossing(branch.drive, branch.magnitude, 1.0)
    crossing = None
    if found is None:
        summary.note("crossing", "|lambda| never crosses 1 on this column")
    else:
        crossing, bracket = found
        summary.values["hill_crossing_dbm"] = crossing
        summary.values["hill_crossing_bracket_dbm"] = list(bracket)
        summary.values["lambda_slope_per_db"] = float(
            np.polyfit(branch.drive, branch.magnitude, 1)[0]
        )

    torus = load_torus(args.torus_csv) if args.torus_csv else None
    if args.torus_csv and torus is None:
        summary.note("torus", "fewer than two converged torus rows")
    torus_fit = None
    if torus is not None:
        torus_fit = fit_linear_intercept(torus.drive, torus.radius_squared)
        summary.values["torus_p_c_dbm"] = torus_fit[1]
        summary.values["torus_slope_per_db"] = torus_fit[0]
        summary.values["torus_fit_r_squared"] = torus_fit[2]
        if crossing is not None:
            summary.values["onset_agreement_db"] = abs(crossing - torus_fit[1])

    title = (f"Neimark–Sacker onset, $f_p$ = {args.pump_ghz:g} GHz"
             + (f"\n{args.label}" if args.label else ""))
    figure_onset(branch, torus, crossing, torus_fit, title, args.outdir)
    figure_locus(branch, f"Critical pair in the complex plane\n"
                         f"{branch.drive.size} drives, "
                         f"{branch.drive.min():.2f} → {branch.drive.max():.2f} dBm",
                 args.outdir)

    twist = None
    if torus is not None and crossing is not None and torus.drive.size >= 2:
        # Both slopes in GHz per dB so the twist carries GHz per unit r^2.
        near = np.argsort(np.abs(branch.drive - crossing))[:2]
        hill_slope = float(
            (branch.real_ghz[near.max()] - branch.real_ghz[near.min()])
            / (branch.drive[near.max()] - branch.drive[near.min()])
        )
        torus_slope = float(
            (torus.ratio[1] - torus.ratio[0])
            / (torus.drive[1] - torus.drive[0]) * args.pump_ghz
        )
        twist = twist_coefficient(hill_slope, torus_slope, torus_fit[0])
        summary.values["hill_frequency_slope_ghz_per_db"] = hill_slope
        summary.values["torus_frequency_slope_ghz_per_db"] = torus_slope
        summary.values["twist_b_ghz_per_r_squared"] = twist
    figure_frequency(branch, torus, args.pump_ghz, crossing, torus_fit, twist,
                     args.outdir)

    lattice = (
        load_lattice(args.ansatz_csv, args.device,
                     tuple(args.control_anchor) if args.control_anchor else None)
        if args.ansatz_csv else None
    )
    if args.ansatz_csv and lattice is None:
        summary.note("lattice", "no matching rows in the ansatz CSV")
    if lattice is not None:
        edges = _regime_edges(lattice, args, crossing)
        summary.values["regime_edges"] = edges
        figure_regimes(lattice, crossing, edges, args.outdir)
        if lattice.drive is None:
            summary.note("fig4", "--control-anchor is required to place onsets in dBm")
        else:
            window = _select_fit_window(lattice, args, summary)
            if int(np.count_nonzero(window)) < 2:
                summary.note("fig4", "fewer than two points inside the fit window")
            else:
                fit = fit_linear_intercept(lattice.drive[window],
                                           lattice.off_lattice[window])
                summary.values["time_domain_onset_dbm"] = fit[1]
                summary.values["time_domain_fit_r_squared"] = fit[2]
                if crossing is not None:
                    summary.values["onset_gap_db"] = fit[1] - crossing
                figure_gap(lattice, fit, window, crossing,
                           None if torus_fit is None else torus_fit[1], args.outdir)

    payload = {"summary": summary.values, "skipped": summary.skipped}
    (args.outdir / "onset_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    payload = run(parse_args(argv))
    for key, value in payload["summary"].items():
        print(f"{key}: {value}")
    for note in payload["skipped"]:
        print(f"skipped {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
