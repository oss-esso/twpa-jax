"""Plot swept return-map diagnostics from :mod:`return_map`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from return_map import CAMPAIGNS, analyse_device, section_coordinates  # noqa: E402


FIGURE_ROOT = ROOT / "outputs/chaos/figures_20260817"


def _atomic_savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fig.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
    os.replace(temporary, path)


def _point_arrays(
    result: dict[str, Any],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    control = np.array([float(p["control_value"]) for p in result["points"]])
    strobe = [np.asarray(p["strobe_values"], dtype=float) for p in result["points"]]
    section = [np.asarray(p["section_z1_z2"], dtype=float) for p in result["points"]]
    return control, strobe, section


def plot_successive_maxima(
    result: dict[str, Any], output_dir: Path
) -> Path | None:
    """Lorenz-style ``a_{n+1}`` against ``a_n`` for one point per regime.

    Independent of the strobe timebase, so it cross-checks the stroboscopic
    section rather than repeating it.
    """
    device = str(result["device"])
    points = result["points"]
    classes = [str(p["descriptors"].get("classification")) for p in points]
    chosen: list[tuple[int, str]] = []
    for wanted in ("period-1", "period-q", "torus", "chaos"):
        indices = [
            i
            for i, value in enumerate(classes)
            if value == wanted
            and (points[i].get("successive_maxima") or {}).get("status") == "OK"
        ]
        if indices:
            chosen.append((indices[len(indices) // 2], wanted))
    if not chosen:
        return None

    fig, axes = plt.subplots(
        1, len(chosen), figsize=(4 * len(chosen), 4.2),
        squeeze=False, constrained_layout=True,
    )
    for ax, (index, label) in zip(axes[0], chosen):
        maxima = points[index]["successive_maxima"]
        a = np.asarray(maxima["amplitudes"], dtype=float)
        ax.scatter(a[:-1], a[1:], s=2, alpha=0.35)
        ax.set_title(
            f"{label}\nmu={float(points[index]['control_value']):g}\n"
            f"n={maxima['n_maxima']}, rel. spread={maxima['relative_spread']:.2e}"
        )
        ax.set_xlabel("$a_n$")
        ax.set_ylabel("$a_{n+1}$")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"{device}: successive-maxima return map")
    path = output_dir / f"{device}_successive_maxima.png"
    _atomic_savefig(fig, path)
    plt.close(fig)
    return path


def animate_sections(result: dict[str, Any], output_dir: Path) -> Path | None:
    """Animate the section across the control axis in FIXED coordinates.

    The axes are computed once over every point and then held, so motion in the
    frame is the attractor changing and never the view rescaling.  The 3-D
    parameterized figure carries the same information in one static image; this
    exists for presentation.
    """
    device = str(result["device"])
    control, _, sections = _point_arrays(result)
    if len(sections) < 2:
        return None
    classes = [str(p["descriptors"].get("classification")) for p in result["points"]]

    stacked = np.vstack([s for s in sections if s.size])
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    pad = 0.05 * np.maximum(hi - lo, np.finfo(float).tiny)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter = ax.scatter([], [], s=3, alpha=0.45)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel("z1 = v(t_n)")
    ax.set_ylabel("z2 = v(t_n + Delta)")
    title = ax.set_title("")

    def update(frame: int):
        scatter.set_offsets(sections[frame])
        title.set_text(
            f"{device}   control={control[frame]:g}   {classes[frame]}"
        )
        return scatter, title

    animation = FuncAnimation(
        fig, update, frames=len(sections), interval=250, blit=False
    )
    path = output_dir / f"{device}_section_movie.gif"
    try:
        animation.save(path, writer=PillowWriter(fps=4))
    except (OSError, ValueError) as error:
        plt.close(fig)
        print(f"{device}: animation not written ({error})")
        return None
    plt.close(fig)
    return path


def plot_sweep(result: dict[str, Any], output_dir: Path) -> list[str]:
    device = str(result["device"])
    control, strobes, sections = _point_arrays(result)
    descriptors = [p["descriptors"] for p in result["points"]]
    d1 = np.array(
        [max(float(d["d_1"]), np.finfo(float).tiny) for d in descriptors]
    )
    radius = np.array(
        [max(float(d["r_RMS"]), np.finfo(float).tiny) for d in descriptors]
    )
    rho = np.array(
        [np.nan if d.get("rho") is None else d["rho"] for d in descriptors]
    )
    on_comb = np.array([
        np.nan if p.get("on_comb") is None else p["on_comb"]
        for p in result["points"]
    ])
    # The spectral records are joined by point basename.
    spectral = {
        Path(str(r.get("point_dir", ""))).name: r
        for r in result["spectral_cross_check"]
    }
    for i, point in enumerate(result["points"]):
        match = spectral.get(Path(str(point["point_dir"])).name)
        if match is not None:
            on_comb[i] = np.nan
    lyap_path = ROOT / "outputs/chaos/lyapunov_kantz" / f"{device}.json"
    if lyap_path.exists():
        lyap_rows = json.loads(lyap_path.read_text(encoding="utf-8"))
        by_name = {Path(str(r.get("point_dir", ""))).name: r for r in lyap_rows}
        on_comb = np.array([
            float(
                by_name.get(Path(str(p["point_dir"])).name, {})
                .get("second_generator", {})
                .get("on_comb", np.nan)
            )
            for p in result["points"]
        ])

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 13), sharex=True, constrained_layout=True
    )
    ax = axes[0]
    for mu, values in zip(control, strobes):
        ax.scatter(
            np.full(values.size, mu),
            values,
            s=1.0,
            alpha=0.30,
            color="tab:blue",
            rasterized=True,
        )
    ax.set_ylabel("settled strobe $y_n$")
    ax.set_title(f"{device}: swept stroboscopic bifurcation diagram")
    axes[1].plot(control, radius, "o-", ms=3, label="$r_{RMS}$")
    axes[1].plot(control, d1, "s-", ms=3, label="$d_1$")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("radius / recurrence")
    axes[1].legend()
    axes[2].plot(control, rho, "o-", ms=3, color="tab:purple")
    locked = np.array([d.get("locking_verdict") == "LOCKED" for d in descriptors])
    if locked.any():
        axes[2].scatter(
            control[locked],
            rho[locked],
            facecolors="none",
            edgecolors="red",
            s=55,
            label="locked",
        )
        axes[2].legend()
    axes[2].set_ylabel("$\\rho$")
    axes[3].plot(control, on_comb, "o-", ms=3, color="tab:green")
    axes[3].set_ylabel("on_comb")
    axes[3].set_xlabel(str(result["points"][0].get("control_axis", "control")))
    sweep_path = output_dir / f"{device}_sweep.png"
    _atomic_savefig(fig, sweep_path)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    for mu, section in zip(control, sections):
        ax3d.scatter(
            np.full(section.shape[0], mu),
            section[:, 0],
            section[:, 1],
            s=1.0,
            alpha=0.25,
        )
    ax3d.set_xlabel("control")
    ax3d.set_ylabel("z1 = v(t_n)")
    ax3d.set_zlabel("z2 = v(t_n + Delta)")
    ax3d.set_title(f"{device}: parameterized Poincare set")
    poincare_path = output_dir / f"{device}_poincare3d.png"
    _atomic_savefig(fig, poincare_path)
    plt.close(fig)

    classes = [str(d.get("classification")) for d in descriptors]
    representatives: list[int] = []
    for wanted in ("period-1", "period-q", "torus", "chaos"):
        indices = [i for i, value in enumerate(classes) if value == wanted]
        if indices:
            representatives.append(indices[len(indices) // 2])
    n_representatives = max(1, len(representatives))
    fig, axes = plt.subplots(
        1,
        n_representatives,
        figsize=(4 * n_representatives, 4),
        squeeze=False,
        constrained_layout=True,
    )
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, index in zip(axes[0], representatives):
        section = sections[index]
        ax.scatter(section[:, 0], section[:, 1], s=2, alpha=0.35)
        ax.set_title(f"{classes[index]}\nmu={control[index]:g}")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
    return_map_path = output_dir / f"{device}_return_maps.png"
    _atomic_savefig(fig, return_map_path)
    plt.close(fig)

    # Each panel above autoscales independently, which hides the fact that the
    # three regimes differ in radius by three to four orders of magnitude -- the
    # period-1 "point" and the chaotic cloud are drawn the same size.  Overlay
    # them on one pair of axes, linear and log, so the scale is visible.
    overlay_path = output_dir / f"{device}_return_maps_common_scale.png"
    fig, (ax_linear, ax_log) = plt.subplots(
        1, 2, figsize=(13, 6), constrained_layout=True
    )
    for order, index in enumerate(representatives):
        section = sections[index]
        centre = section.mean(axis=0)
        label = f"{classes[index]}  mu={control[index]:g}"
        ax_linear.scatter(
            section[:, 0],
            section[:, 1],
            s=3,
            alpha=0.4,
            color=colours[order % len(colours)],
            label=label,
        )
        # Radius about each set's own centroid, on a log axis: this is the only
        # view in which a floor-level point and a chaotic cloud are both legible.
        radius = np.linalg.norm(section - centre, axis=1)
        radius = np.maximum(radius, np.finfo(float).tiny)
        ax_log.plot(
            np.arange(radius.size),
            radius,
            lw=0.6,
            alpha=0.8,
            color=colours[order % len(colours)],
            label=label,
        )
    ax_linear.set_xlabel("z1 = v(t_n)")
    ax_linear.set_ylabel("z2 = v(t_n + Delta)")
    ax_linear.set_title(f"{device}: all regimes, common linear axes")
    ax_linear.set_aspect("equal", adjustable="datalim")
    ax_linear.legend(fontsize=8, loc="best")
    ax_log.set_yscale("log")
    ax_log.set_xlabel("strobe index n")
    ax_log.set_ylabel("radius about each set's centroid")
    ax_log.set_title("same three sets, log radius")
    ax_log.legend(fontsize=8, loc="best")
    _atomic_savefig(fig, overlay_path)
    plt.close(fig)

    paths = [
        str(sweep_path),
        str(poincare_path),
        str(return_map_path),
        str(overlay_path),
    ]
    maxima_path = plot_successive_maxima(result, output_dir)
    if maxima_path is not None:
        paths.append(str(maxima_path))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=sorted(CAMPAIGNS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument(
        "--animate",
        action="store_true",
        help="also write the section movie (slow; the 3-D figure is static)",
    )
    args = parser.parse_args(argv)
    devices = [args.device] if args.device else list(CAMPAIGNS)
    for device in devices:
        result = analyse_device(device, force=args.force)
        for path in plot_sweep(result, args.output_dir):
            print(path)
        if args.animate:
            movie = animate_sections(result, args.output_dir)
            if movie is not None:
                print(movie)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
