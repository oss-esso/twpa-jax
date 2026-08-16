"""Figures for the nonlinear-dynamics reduction of the FDTD chaos campaigns.

Every measured panel is paired with a synthetic reference generated from a
textbook system, so the reader can see what the instrument reports on a case
with a known answer before reading the device panels.

Reference systems used:

* periodic          fixed point of a stable limit cycle, strobed once per period
* quasi-periodic    rigid rotation by the golden mean, i.e. an invariant circle
* chaotic           Henon map, ``a = 1.4``, ``b = 0.3``

Usage::

    python scripts/chaos/plot_nonlinear_diagnostics.py \
        --jtwpa outputs/chaos/nonlinear_jtwpa_torus/jc_jtwpa.json \
        --twoc  outputs/chaos/nonlinear_2c_gap/ipm_2c_fixed.json \
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
# is a stated rule rather than a curve-fit to the picture.
FLOOR_SIGMA = 1.0e-07
REGULAR_K = 0.30
CHAOTIC_K = 0.90

COLOUR = {
    "floor": "#d9d9d9",
    "regular": "#7fbf7b",
    "transitional": "#fdb863",
    "chaotic": "#d7301f",
}
LABEL = {
    "floor": "numerical floor (not classifiable)",
    "regular": "regular",
    "transitional": "transitional",
    "chaotic": "chaotic",
}


@dataclass(frozen=True)
class Point:
    """One reduced campaign point."""

    control: float
    sigma: float
    k: float | None
    d2_curve: dict[int, float]
    point_dir: Path

    @property
    def regime(self) -> str:
        if self.sigma < FLOOR_SIGMA:
            return "floor"
        if self.k is None:
            return "transitional"
        if abs(self.k) < REGULAR_K:
            return "regular"
        if self.k > CHAOTIC_K:
            return "chaotic"
        return "transitional"


def load_points(path: Path) -> list[Point]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    points: list[Point] = []
    for row in rows:
        if row.get("status") != "OK" or not row.get("strobe_std"):
            continue
        zero_one = row.get("zero_one_test") or {}
        dimension = row.get("correlation_dimension") or {}
        curve = {int(m): float(d) for m, d in (dimension.get("d2_curve") or {}).items()}
        points.append(
            Point(
                control=float(row["control_value"]),
                sigma=float(row["strobe_std"]),
                k=(None if zero_one.get("k_median") is None
                   else float(zero_one["k_median"])),
                d2_curve=curve,
                point_dir=Path(row["point_dir"]),
            )
        )
    points.sort(key=lambda p: p.control)
    return points


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
    """Recompute one stroboscopic section from the stored trace."""
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


def nearest(points: list[Point], control: float) -> Point:
    return min(points, key=lambda p: abs(p.control - control))


# --------------------------------------------------------------------------
# Figure 1 -- regime map
# --------------------------------------------------------------------------


def figure_regime_map(
    jtwpa: list[Point], twoc: list[Point], out_path: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)

    panels = [
        (axes[0], jtwpa, "jc_jtwpa", "pump power [dBm]"),
        (axes[1], twoc, "ipm_2c_fixed", r"control $I/I_{\rm bound}$"),
    ]
    for ax, points, device, xlabel in panels:
        control = np.array([p.control for p in points])
        sigma = np.array([p.sigma for p in points])
        k = np.array([np.nan if p.k is None else p.k for p in points])

        # Regime shading, drawn from the thresholds declared at module scope.
        edges = 0.5 * (control[1:] + control[:-1])
        lo = np.concatenate([[control[0] - (edges[0] - control[0])], edges])
        hi = np.concatenate([edges, [control[-1] + (control[-1] - edges[-1])]])
        seen: set[str] = set()
        for point, left, right in zip(points, lo, hi):
            ax.axvspan(
                left, right, color=COLOUR[point.regime], alpha=0.30, linewidth=0,
                label=LABEL[point.regime] if point.regime not in seen else None,
            )
            seen.add(point.regime)

        ax.semilogy(control, sigma, "o-", color="#252525", markersize=5,
                    label=r"$\sigma$ (stroboscopic spread)")
        ax.axhline(FLOOR_SIGMA, color="#252525", linestyle=":", linewidth=1.2)
        ax.text(control[0], FLOOR_SIGMA * 1.25,
                f"floor threshold {FLOOR_SIGMA:.0e}", fontsize=8, color="#252525")
        ax.set_ylabel(r"$\sigma$  [V]")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{device} — stroboscopic spread and 0-1 test, pump-only")

        twin = ax.twinx()
        twin.plot(control, k, "s--", color="#2166ac", markersize=4, label=r"$K$ (0-1 test)")
        twin.axhline(0.0, color="#2166ac", linewidth=0.6, alpha=0.5)
        twin.axhline(1.0, color="#2166ac", linewidth=0.6, alpha=0.5)
        twin.set_ylim(-0.6, 1.15)
        twin.set_ylabel(r"$K$    (0 regular, 1 chaotic)", color="#2166ac")
        twin.tick_params(axis="y", labelcolor="#2166ac")

        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, loc="upper left", fontsize=8, framealpha=0.9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 -- return maps against theory
# --------------------------------------------------------------------------


def _return_map(ax, series: np.ndarray, title: str, colour: str) -> None:
    series = np.asarray(series, dtype=np.float64)
    if series.size < 4:
        ax.text(0.5, 0.5, "no section", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return
    spread = float(np.std(series))
    if spread > 0:
        centred = (series - series.mean()) / spread
    else:
        centred = series - series.mean()
    ax.plot(centred[:-1], centred[1:], ".", markersize=1.6, color=colour, alpha=0.7)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def figure_return_maps(
    jtwpa: list[Point], twoc: list[Point], out_path: Path
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5), constrained_layout=True)

    # Row 1: reference systems, normalised the same way as the measured rows.
    _return_map(axes[0, 0], reference_periodic(),
                "theory: periodic + roundoff\n(a point, magnified $10^{12}$x)", "#252525")
    _return_map(axes[0, 1], reference_quasiperiodic(),
                "theory: quasi-periodic\n(invariant circle)", "#1a9850")
    _return_map(axes[0, 2], reference_chaotic(), "theory: chaotic\n(Henon map)", "#d7301f")
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

    # Row 2: jc_jtwpa across its regimes.
    jt_targets = [
        (-30.5000, "floor"),
        (-29.4444, "approach"),
        (-29.1629, "regular window"),
        (-28.0545, "chaotic"),
    ]
    for ax, (control, tag) in zip(axes[1], jt_targets):
        point = nearest(jtwpa, control)
        strobe = measured_strobe(point, "jc_jtwpa")
        _return_map(
            ax, strobe,
            f"jc_jtwpa {point.control:.4f} dBm — {tag}\n"
            f"K={point.k:+.4f}   $\\sigma$={point.sigma:.2e}",
            COLOUR[point.regime],
        )

    # Row 3: ipm_2c_fixed across its regimes.
    tc_targets = [
        (0.5750, "floor"),
        (0.5900, "approach"),
        (0.6000, "regular window"),
        (0.6200, "chaotic"),
    ]
    for ax, (control, tag) in zip(axes[2], tc_targets):
        point = nearest(twoc, control)
        strobe = measured_strobe(point, "ipm_2c_fixed")
        _return_map(
            ax, strobe,
            f"ipm_2c_fixed {point.control:.4f} — {tag}\n"
            f"K={point.k:+.4f}   $\\sigma$={point.sigma:.2e}",
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


def figure_zero_one_translation(
    jtwpa: list[Point], out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    c = 1.1  # one fixed frequency; the test itself medians over many

    for ax, (series, title, colour) in zip(
        axes,
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

    for ax, control in zip(axes[2:], [-29.1629, -28.0545]):
        point = nearest(jtwpa, control)
        strobe = measured_strobe(point, "jc_jtwpa")
        p, q = zero_one_translation(strobe, c)
        ax.plot(p, q, "-", linewidth=0.7, color=COLOUR[point.regime])
        ax.set_title(
            f"jc_jtwpa {point.control:.4f} dBm\nK={point.k:+.4f}", fontsize=9
        )
        ax.set_xlabel("p")
        ax.set_ylabel("q")
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "0-1 test translation variables: the quantity $K$ summarises", fontsize=11
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4 -- correlation dimension and the normal-form law
# --------------------------------------------------------------------------


def figure_d2_and_scaling(
    jtwpa: list[Point], twoc: list[Point], out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    for points, device, marker in [(jtwpa, "jc_jtwpa", "o"), (twoc, "ipm_2c_fixed", "s")]:
        for point in points:
            if not point.d2_curve:
                continue
            dims = sorted(point.d2_curve)
            ax.plot(
                dims, [point.d2_curve[m] for m in dims], marker + "-",
                markersize=3, linewidth=0.8, alpha=0.55,
                color=COLOUR[point.regime],
            )
    ax.plot([2, 3, 4, 5], [2, 3, 4, 5], "k--", linewidth=1.0,
            label=r"$D_2 = m$  (no attractor: noise fills the embedding)")
    ax.axhline(1.0, color="#1a9850", linestyle="-", linewidth=1.4,
               label=r"$D_2 = 1$: invariant circle, i.e. a 2-torus section")
    ax.axhline(2.05, color="#2166ac", linestyle=":", linewidth=1.2,
               label=r"Lorenz $D_2 = 2.05$ (literature)")
    ax.set_xlabel("embedding dimension $m$")
    ax.set_ylabel("$D_2$")
    ax.set_title(
        "Correlation dimension: saturates at $D_2\\simeq1$ on the torus branch,\n"
        "grows with $m$ once chaotic"
    )
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    # Supercritical Neimark-Sacker: the invariant circle grows as
    # (mu - mu_c)**0.5.  Anchor the reference curve on the measured approach
    # rather than fitting it, so the comparison stays visual.
    jt = [p for p in jtwpa if p.sigma >= FLOOR_SIGMA]
    control = np.array([p.control for p in jt])
    sigma = np.array([p.sigma for p in jt])
    ax.semilogy(control, sigma, "o-", color="#252525", markersize=4,
                label="jc_jtwpa measured")
    mu_c = -29.50
    grid = np.linspace(mu_c + 1e-3, control.max(), 300)
    anchor = sigma[0] / np.sqrt(max(control[0] - mu_c, 1e-9))
    ax.semilogy(grid, anchor * np.sqrt(grid - mu_c), "--", color="#1a9850",
                label=r"supercritical NS: $\sigma \propto (\mu-\mu_c)^{1/2}$")
    ax.set_xlabel("pump power [dBm]")
    ax.set_ylabel(r"$\sigma$ [V]")
    ax.set_title("Growth law on approach, against the\nsupercritical Neimark-Sacker prediction")
    ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jtwpa", type=Path,
                        default=Path("outputs/chaos/nonlinear_jtwpa_torus/jc_jtwpa.json"))
    parser.add_argument("--twoc", type=Path,
                        default=Path("outputs/chaos/nonlinear_2c_gap/ipm_2c_fixed.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/chaos/figures"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    jtwpa = load_points(args.jtwpa)
    twoc = load_points(args.twoc)
    print(f"jc_jtwpa {len(jtwpa)} points, ipm_2c_fixed {len(twoc)} points")

    jobs = [
        ("regime_map.png", lambda p: figure_regime_map(jtwpa, twoc, p)),
        ("return_maps.png", lambda p: figure_return_maps(jtwpa, twoc, p)),
        ("zero_one_translation.png", lambda p: figure_zero_one_translation(jtwpa, p)),
        ("d2_and_scaling.png", lambda p: figure_d2_and_scaling(jtwpa, twoc, p)),
    ]
    for name, build in jobs:
        path = args.output / name
        build(path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
