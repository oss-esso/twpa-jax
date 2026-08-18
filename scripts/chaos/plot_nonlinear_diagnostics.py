"""Figures for the nonlinear-dynamics reduction of the FDTD chaos campaigns.

Every measured panel is paired with a synthetic reference generated from a
textbook system, so the reader can see what the instrument reports on a case
with a known answer before reading the device panels.

Reference systems used:

* periodic          fixed point of a stable limit cycle, strobed once per period
* quasi-periodic    rigid rotation by the golden mean, i.e. an invariant circle
* chaotic           Henon map, ``a = 1.4``, ``b = 0.3``

Representative points are selected by the rules in ``DeviceSet.representatives``
rather than named per device, so adding a device changes no thresholds.

Usage::

    python scripts/chaos/plot_nonlinear_diagnostics.py \
        --device jc_jtwpa outputs/chaos/nonlinear_jtwpa_torus/jc_jtwpa.json \
        --device ipm_2c_fixed outputs/chaos/nonlinear_2c_gap/ipm_2c_fixed.json \
        --device guarcello outputs/chaos/nonlinear_guarcello/guarcello.json \
        --output outputs/chaos/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "chaos"))

import nonlinear_diagnostics as nd  # noqa: E402

# Thresholds are fixed here, before any figure is drawn, so that regime shading
# and point selection are stated rules rather than a curve-fit to the picture.
FLOOR_SIGMA = 1.0e-07
REGULAR_K = 0.30
CHAOTIC_K = 0.90

COLOUR = {
    "floor": "#bdbdbd",
    "regular": "#1a9850",
    "transitional": "#fdb863",
    "chaotic": "#d7301f",
}
LABEL = {
    "floor": "numerical floor (not classifiable)",
    "regular": "regular",
    "transitional": "transitional",
    "chaotic": "chaotic",
}
# Control axis label per device; anything unlisted falls back to dBm.
XLABEL = {
    "ipm_2c_fixed": r"control $I/I_{\rm bound}$",
}
DEFAULT_XLABEL = "pump power [dBm]"


@dataclass(frozen=True)
class Point:
    """One reduced campaign point."""

    control: float
    sigma: float
    k: float | None
    d2_curve: dict[int, float]
    d2_plateau: float | None
    point_dir: Path
    section_kind: str  # "strobe" or "poincare_upward"

    @property
    def regime(self) -> str:
        # The floor rule is calibrated on the stroboscopic voltage spread. It
        # must not be applied to the Poincare fallback, whose units and sample
        # count are different, so those points are classified on K alone.
        if self.section_kind == "strobe" and self.sigma < FLOOR_SIGMA:
            return "floor"
        if self.k is None:
            return "transitional"
        if abs(self.k) < REGULAR_K:
            return "regular"
        if self.k > CHAOTIC_K:
            return "chaotic"
        return "transitional"


@dataclass(frozen=True)
class DeviceSet:
    """All reduced points for one device."""

    name: str
    points: list[Point]

    @property
    def xlabel(self) -> str:
        return XLABEL.get(self.name, DEFAULT_XLABEL)

    @property
    def section_kind(self) -> str:
        if any(p.section_kind == "poincare_upward" for p in self.points):
            return "poincare_upward"
        return "strobe"

    @property
    def sigma_label(self) -> str:
        if self.section_kind == "poincare_upward":
            return r"$\sigma$ of $\dot v$ at upward crossings  [device units]"
        return r"$\sigma$ of the stroboscopic section  [V]"

    @property
    def section_note(self) -> str:
        if self.section_kind == "poincare_upward":
            return (
                "NO stroboscopic section: the trace stores < 8 samples per pump "
                "period, so this device falls back to the stored Poincare upward "
                "crossings. Different observable, different units, 300 samples "
                "instead of 1049 — the floor threshold does not apply and sigma "
                "is NOT comparable to the other devices."
            )
        return ""

    def representatives(self) -> list[tuple[Point, str]]:
        """Pick one point per regime by rule, never by name.

        floor           the best-resolved floor point, i.e. the largest sigma
                        still below the floor threshold
        approach        above the floor, the point whose D2 plateau is closest
                        to 1 -- an invariant circle if one exists at all
        regular window  above the floor, the smallest |K|
        chaotic         the largest K
        """
        if self.section_kind == "strobe":
            above = [p for p in self.points if p.sigma >= FLOOR_SIGMA]
            below = [p for p in self.points if p.sigma < FLOOR_SIGMA]
        else:
            above, below = list(self.points), []
        chosen: list[tuple[Point, str]] = []

        if below:
            chosen.append((max(below, key=lambda p: p.sigma), "floor"))
        elif self.points:
            chosen.append((min(self.points, key=lambda p: p.sigma), "lowest sigma"))

        plateaued = [p for p in above if p.d2_plateau is not None]
        if plateaued:
            chosen.append(
                (min(plateaued, key=lambda p: abs(p.d2_plateau - 1.0)), "approach")
            )
        elif above:
            chosen.append((min(above, key=lambda p: p.sigma), "approach"))

        keyed = [p for p in above if p.k is not None]
        if keyed:
            chosen.append((min(keyed, key=lambda p: abs(p.k)), "smallest |K|"))
            chosen.append((max(keyed, key=lambda p: p.k), "largest K"))
        return chosen


def load_device(name: str, path: Path) -> DeviceSet:
    rows = json.loads(path.read_text(encoding="utf-8"))
    points: list[Point] = []
    for row in rows:
        if row.get("status") != "OK" or not row.get("strobe_std"):
            continue
        zero_one = row.get("zero_one_test") or {}
        dimension = row.get("correlation_dimension") or {}
        curve = {int(m): float(d) for m, d in (dimension.get("d2_curve") or {}).items()}
        plateau = dimension.get("d2_plateau")
        points.append(
            Point(
                control=float(row["control_value"]),
                sigma=float(row["strobe_std"]),
                k=(None if zero_one.get("k_median") is None
                   else float(zero_one["k_median"])),
                d2_curve=curve,
                d2_plateau=None if plateau is None else float(plateau),
                point_dir=Path(row["point_dir"]),
                section_kind=(
                    "strobe" if int(row.get("strobe_points") or 0) > 0
                    else "poincare_upward"
                ),
            )
        )
    points.sort(key=lambda p: p.control)
    return DeviceSet(name=name, points=points)


# --------------------------------------------------------------------------
# Reference systems with known answers
# --------------------------------------------------------------------------


def reference_periodic(n: int = 1049, noise: float = 1e-12) -> np.ndarray:
    """A stable limit cycle strobed on its own period: a single point."""
    rng = np.random.default_rng(0)
    return 1.0 + noise * rng.standard_normal(n)


def reference_quasiperiodic(n: int = 1049) -> np.ndarray:
    """Rigid rotation by the golden mean: densely fills an invariant circle.

    A pure cosine is used so the return map is an exact ellipse, which is the
    shape a 2-torus section must produce.
    """
    gamma = 0.5 * (np.sqrt(5.0) - 1.0)
    theta = 2.0 * np.pi * gamma * np.arange(n)
    return np.cos(theta)


def reference_chaotic(n: int = 1049, burn: int = 500) -> np.ndarray:
    """Henon map at the classical parameters."""
    x, y = 0.1, 0.1
    out = np.empty(n)
    for i in range(n + burn):
        x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
        if i >= burn:
            out[i - burn] = x
    return out


def zero_one_translation(series: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """The (p, q) translation variables of the Gottwald-Melbourne 0-1 test.

    Bounded for a regular signal, an unbounded random walk for a chaotic one.
    This is the test's own internal state, plotted directly rather than
    summarised into ``K``.
    """
    phi = np.asarray(series, dtype=np.float64)
    phi = phi - phi.mean()
    n = np.arange(phi.size)
    return np.cumsum(phi * np.cos(c * n)), np.cumsum(phi * np.sin(c * n))


def measured_strobe(point: Point, device: str) -> np.ndarray:
    """Recompute one section from the stored trace.

    Mirrors ``analyse_point``: a stroboscopic section where the record is fine
    enough, otherwise the stored Poincare upward crossings.
    """
    if point.section_kind == "poincare_upward":
        branches = point.point_dir / "poincare_branches.npz"
        if not branches.exists():
            return np.empty(0)
        with np.load(branches) as data:
            return np.asarray(data["upward"], dtype=np.float64)
    result = json.loads((point.point_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(point.point_dir / "trace.npz") as data:
        t = np.asarray(data["t"], dtype=np.float64)
        v = np.asarray(data["v_out"], dtype=np.float64)
    start = int(result.get("steady_state_start_index", 0))
    start = min(start, max(t.size - 16, 0))
    return nd.stroboscopic_section(
        v[start:],
        nd.PUMP_HZ[device],
        float(result["dt_s"]),
        int(result.get("record_stride", 1)),
    )


# --------------------------------------------------------------------------
# Figure 1 -- regime map
# --------------------------------------------------------------------------


def figure_regime_map(devices: list[DeviceSet], out_path: Path) -> None:
    fig, axes = plt.subplots(
        len(devices), 1, figsize=(11, 4.5 * len(devices)), constrained_layout=True,
        squeeze=False,
    )
    for ax, device in zip(axes[:, 0], devices):
        points = device.points
        control = np.array([p.control for p in points])
        sigma = np.array([p.sigma for p in points])
        k = np.array([np.nan if p.k is None else p.k for p in points])

        edges = 0.5 * (control[1:] + control[:-1])
        lo = np.concatenate([[control[0] - (edges[0] - control[0])], edges])
        hi = np.concatenate([edges, [control[-1] + (control[-1] - edges[-1])]])
        seen: set[str] = set()
        for point, left, right in zip(points, lo, hi):
            ax.axvspan(
                left, right, color=COLOUR[point.regime], alpha=0.25, linewidth=0,
                label=LABEL[point.regime] if point.regime not in seen else None,
            )
            seen.add(point.regime)

        ax.semilogy(control, sigma, "o-", color="#252525", markersize=4,
                    label=r"$\sigma$ (section spread)")
        if device.section_kind == "strobe":
            ax.axhline(FLOOR_SIGMA, color="#252525", linestyle=":", linewidth=1.2)
            ax.text(control[0], FLOOR_SIGMA * 1.3,
                    f"floor threshold {FLOOR_SIGMA:.0e}", fontsize=8, color="#252525")
        else:
            ax.text(
                0.015, 0.03, device.section_note, fontsize=7.5, style="italic",
                color="#7a2020", transform=ax.transAxes, wrap=True,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#7a2020",
                      "linewidth": 0.6},
            )
        ax.set_ylabel(device.sigma_label, fontsize=9)
        ax.set_xlabel(device.xlabel)
        section = ("stroboscopic" if device.section_kind == "strobe"
                   else "Poincare upward (fallback)")
        ax.set_title(
            f"{device.name} — {section} section spread and 0-1 test, pump-only "
            f"({len(points)} points)"
        )

        twin = ax.twinx()
        twin.plot(control, k, "s--", color="#2166ac", markersize=3,
                  label=r"$K$ (0-1 test)")
        twin.axhline(0.0, color="#2166ac", linewidth=0.6, alpha=0.5)
        twin.axhline(1.0, color="#2166ac", linewidth=0.6, alpha=0.5)
        twin.set_ylim(-0.6, 1.15)
        twin.set_ylabel(r"$K$    (0 regular, 1 chaotic)", color="#2166ac")
        twin.tick_params(axis="y", labelcolor="#2166ac")

        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, loc="upper left", fontsize=8,
                  framealpha=0.9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 -- return maps against theory
# --------------------------------------------------------------------------


def _return_map(ax, series: np.ndarray, title: str, colour: str) -> None:
    series = np.asarray(series, dtype=np.float64)
    if series.size < 4:
        ax.text(0.5, 0.5, "no section", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    spread = float(np.std(series))
    centred = (series - series.mean()) / spread if spread > 0 else series - series.mean()
    ax.plot(centred[:-1], centred[1:], ".", markersize=1.6, color=colour, alpha=0.7)
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def figure_return_maps(devices: list[DeviceSet], out_path: Path) -> None:
    rows = 1 + len(devices)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3.1 * rows),
                             constrained_layout=True, squeeze=False)

    _return_map(axes[0, 0], reference_periodic(),
                "theory: periodic + roundoff\n(a point, magnified $10^{12}$x)",
                "#252525")
    _return_map(axes[0, 1], reference_quasiperiodic(),
                "theory: quasi-periodic\n(invariant circle)", "#1a9850")
    _return_map(axes[0, 2], reference_chaotic(),
                "theory: chaotic\n(Henon map)", "#d7301f")
    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.0, 0.5,
        "Return map of the stroboscopic\nsection, $x_{n+1}$ vs $x_n$.\n\n"
        "EVERY panel is normalised to\nunit variance, so shape is\n"
        "readable but scale is not --\nread $\\sigma$ in each title.\n\n"
        "A period-1 orbit collapses to a\npoint, a 2-torus to a closed\n"
        "curve, chaos to a fractal set.\n\n"
        "The leftmost column is pure\nroundoff blown up to fill the\n"
        "axes: structureless by\nconstruction, not measured\ndynamics.",
        fontsize=8.5, va="center", transform=axes[0, 3].transAxes,
    )

    for row, device in zip(axes[1:], devices):
        chosen = device.representatives()
        for ax in row:
            ax.axis("off")
        for ax, (point, tag) in zip(row, chosen):
            ax.axis("on")
            strobe = measured_strobe(point, device.name)
            k_text = "n/a" if point.k is None else f"{point.k:+.4f}"
            _return_map(
                ax, strobe,
                f"{device.name} {point.control:.4f} — {tag}\n"
                f"K={k_text}   $\\sigma$={point.sigma:.2e}",
                COLOUR[point.regime],
            )

    fig.suptitle(
        "Stroboscopic return maps: reference systems (top) against measured devices",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 -- the 0-1 test's own translation variables
# --------------------------------------------------------------------------


def figure_zero_one_translation(devices: list[DeviceSet], out_path: Path) -> None:
    columns = 2 + 2 * len(devices)
    fig, axes = plt.subplots(1, columns, figsize=(3.3 * columns, 3.7),
                             constrained_layout=True, squeeze=False)
    flat = axes[0]
    c = 1.1  # one fixed frequency; the test itself medians over many

    for ax, (series, title, colour) in zip(
        flat,
        [
            (reference_quasiperiodic(), "theory: quasi-periodic\nbounded", "#1a9850"),
            (reference_chaotic(), "theory: chaotic\nunbounded random walk", "#d7301f"),
        ],
    ):
        p, q = zero_one_translation(series, c)
        ax.plot(p, q, "-", linewidth=0.7, color=colour)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("p")
        ax.set_ylabel("q")
        ax.set_aspect("equal", adjustable="datalim")

    index = 2
    for device in devices:
        chosen = {tag: point for point, tag in device.representatives()}
        for tag in ("smallest |K|", "largest K"):
            ax = flat[index]
            index += 1
            point = chosen.get(tag)
            if point is None:
                ax.axis("off")
                continue
            strobe = measured_strobe(point, device.name)
            if strobe.size < 4:
                ax.axis("off")
                continue
            p, q = zero_one_translation(strobe, c)
            ax.plot(p, q, "-", linewidth=0.7, color=COLOUR[point.regime])
            k_text = "n/a" if point.k is None else f"{point.k:+.4f}"
            ax.set_title(
                f"{device.name} {point.control:.4f}\nK={k_text}", fontsize=9
            )
            ax.set_xlabel("p")
            ax.set_ylabel("q")
            ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "0-1 test translation variables: the quantity $K$ summarises these",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4 -- correlation dimension and the normal-form law
# --------------------------------------------------------------------------


def figure_d2_and_scaling(devices: list[DeviceSet], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    markers = ["o", "s", "^", "v"]
    for device, marker in zip(devices, markers):
        for point in device.points:
            if not point.d2_curve:
                continue
            dims = sorted(point.d2_curve)
            ax.plot(
                dims, [point.d2_curve[m] for m in dims], marker + "-",
                markersize=3, linewidth=0.8, alpha=0.5, color=COLOUR[point.regime],
            )
    ax.plot([2, 3, 4, 5], [2, 3, 4, 5], "k--", linewidth=1.0,
            label=r"$D_2 = m$  (no attractor: noise fills the embedding)")
    ax.axhline(1.0, color="#1a9850", linewidth=1.4,
               label=r"$D_2 = 1$: invariant circle, i.e. a 2-torus section")
    ax.axhline(2.05, color="#2166ac", linestyle=":", linewidth=1.2,
               label=r"Lorenz $D_2 = 2.05$ (literature)")
    ax.set_xlabel("embedding dimension $m$")
    ax.set_ylabel("$D_2$")
    ax.set_title(
        "Correlation dimension: saturates at $D_2\\simeq1$ on the torus branch,\n"
        "grows with $m$ once chaotic  (marker = device, colour = regime)"
    )
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    # Supercritical Neimark-Sacker: the invariant circle grows as
    # (mu - mu_c)**0.5.  The reference curve is anchored on each device's first
    # above-floor point rather than fitted, so the comparison stays visual.
    styles = ["-", "--", "-.", ":"]
    for device, style in zip(devices, styles):
        above = [p for p in device.points if p.sigma >= FLOOR_SIGMA]
        if len(above) < 2:
            continue
        control = np.array([p.control for p in above])
        sigma = np.array([p.sigma for p in above])
        span = control.max() - control.min()
        # Place the reference onset just below the first above-floor point.
        mu_c = control[0] - 0.02 * span
        line, = ax.semilogy(control - control[0], sigma, "o" + style, markersize=3,
                            label=f"{device.name} measured")
        grid = np.linspace(mu_c + 1e-6, control.max(), 300)
        anchor = sigma[0] / np.sqrt(max(control[0] - mu_c, 1e-9))
        ax.semilogy(grid - control[0], anchor * np.sqrt(grid - mu_c), ":",
                    color=line.get_color(), alpha=0.8,
                    label=f"{device.name}  NS $(\\mu-\\mu_c)^{{1/2}}$")
    ax.set_xlabel("control offset from the first above-floor point")
    ax.set_ylabel(r"$\sigma$ [V]")
    ax.set_title("Growth on approach, against the supercritical\n"
                 "Neimark-Sacker square-root prediction")
    ax.legend(fontsize=7.5)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", nargs=2, action="append", metavar=("NAME", "JSON"),
        help="device name and its reduced-diagnostics JSON; repeatable",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/chaos/figures"))
    args = parser.parse_args()

    specs = args.device or [
        ["jc_jtwpa", "outputs/chaos/nonlinear_jtwpa_torus/jc_jtwpa.json"],
        ["ipm_2c_fixed", "outputs/chaos/nonlinear_2c_gap/ipm_2c_fixed.json"],
        ["guarcello", "outputs/chaos/nonlinear_guarcello/guarcello.json"],
    ]

    devices: list[DeviceSet] = []
    for name, path in specs:
        json_path = Path(path)
        if not json_path.exists():
            print(f"{name}: {json_path} missing, skipped")
            continue
        device = load_device(name, json_path)
        if not device.points:
            print(f"{name}: no usable points, skipped")
            continue
        devices.append(device)
        print(f"{name:<14} {len(device.points):>3} points  "
              + "  ".join(f"{tag}={p.control:.4f}"
                          for p, tag in device.representatives()))
    if not devices:
        print("no devices to plot")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("regime_map.png", figure_regime_map),
        ("return_maps.png", figure_return_maps),
        ("zero_one_translation.png", figure_zero_one_translation),
        ("d2_and_scaling.png", figure_d2_and_scaling),
    ]
    for name, build in jobs:
        path = args.output / name
        build(devices, path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
