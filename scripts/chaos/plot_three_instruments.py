"""The three-instrument figure: K, return-map shape, and spectral structure.

One column per device, one row per instrument, so that the claim
"period-1 -> 2-torus -> chaos" can be read off three measurements that could
each have disagreed:

    instrument          below onset          window                chaos
    K (0-1 test)        ~0 on noise          |K| small             ~1
    return-map shape    point                closed curve          cloud
    spectrum            all power on n*f_p   one extra generator   broadband

The spectral row is the strongest of the three because it is a direct
structural measurement rather than a statistic: it asks whether the measured
power sits on a two-frequency lattice, which is exactly the question a
quasi-periodic harmonic-balance ansatz has to answer.

Usage::

    python scripts/chaos/plot_three_instruments.py --output outputs/chaos/figures_20260817
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "chaos"))

import nonlinear_diagnostics as nd  # noqa: E402
import lyapunov_kantz as lk  # noqa: E402

DEVICES = [
    ("jc_jtwpa", "outputs/chaos/nonlinear_jtwpa_torus/jc_jtwpa.json",
     7.12e9, "pump power [dBm]"),
    ("ipm_2c_fixed", "outputs/chaos/nonlinear_2c_gap/ipm_2c_fixed.json",
     7.9e9, r"control $I/I_{\rm bound}$"),
    ("guarcello", "outputs/chaos/nonlinear_guarcello_s4/guarcello.json",
     7.0e9, "pump power [dBm]"),
]
COLOUR = {"regular": "#1a9850", "chaotic": "#d7301f", "floor": "#bdbdbd",
          "transitional": "#fdb863"}
REGULAR_K = 0.30
CHAOTIC_K = 0.90


def regime(k: float | None, on_comb: float | None) -> str:
    if k is None:
        return "transitional"
    if k > CHAOTIC_K:
        return "chaotic"
    if abs(k) < REGULAR_K:
        # Separate the period-1 floor from the torus window on the spectrum,
        # which is what distinguishes them; K cannot.
        if on_comb is not None and on_comb > 0.999:
            return "floor"
        return "regular"
    return "transitional"


def spectral_metrics(point_dir: Path, pump_hz: float, *, n_trials: int = 240,
                     pump_order: int = 12) -> dict[str, float | None]:
    result = json.loads((point_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(point_dir / "trace.npz") as data:
        v = np.asarray(data["v_out"], dtype=np.float64)
    v = v[int(result.get("steady_state_start_index", 0)):]
    dt = float(result["dt_s"]) * int(result.get("record_stride", 1))
    t = dt * np.arange(v.size)
    out = lk.second_generator_share(
        t, v, pump_hz, pump_order=pump_order, generator_order=3,
        n_trials=n_trials,
    )
    if out.get("status") != "OK":
        return {"on_comb": None, "generator_share": None}
    return {
        "on_comb": float(out["on_comb"]),
        "generator_share": float(out["generator_share_of_off_comb"]),
    }


def load(device: str, path: Path, pump_hz: float, cache: Path) -> list[dict]:
    rows = [r for r in json.loads(path.read_text(encoding="utf-8"))
            if r.get("status") == "OK" and r.get("strobe_std")]
    rows.sort(key=lambda r: r["control_value"])
    stored: dict[str, dict] = {}
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
    changed = False
    for row in rows:
        key = row["point_dir"]
        if key not in stored:
            stored[key] = spectral_metrics(Path(row["point_dir"]), pump_hz)
            changed = True
            print(f"  {device} {row['control_value']:>10.4f} "
                  f"on_comb={stored[key]['on_comb']}", flush=True)
        row.update(stored[key])
        row["k"] = (row.get("zero_one_test") or {}).get("k_median")
    if changed:
        cache.write_text(json.dumps(stored, indent=1), encoding="utf-8")
    return rows


def return_map(ax, series: np.ndarray, title: str, colour: str) -> None:
    series = np.asarray(series, dtype=np.float64)
    if series.size < 4:
        ax.text(0.5, 0.5, "no section", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        return
    spread = float(np.std(series))
    centred = (series - series.mean()) / spread if spread > 0 else series
    ax.plot(centred[:-1], centred[1:], ".", markersize=1.4, color=colour, alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=8)


def strobe_of(row: dict, device: str) -> np.ndarray:
    point = Path(row["point_dir"])
    result = json.loads((point / "result.json").read_text(encoding="utf-8"))
    with np.load(point / "trace.npz") as data:
        v = np.asarray(data["v_out"], dtype=np.float64)
    v = v[int(result.get("steady_state_start_index", 0)):]
    return nd.stroboscopic_section(
        v, nd.PUMP_HZ[device], float(result["dt_s"]),
        int(result.get("record_stride", 1)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/chaos/figures_20260817"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "_spectral_cache"
    cache_dir.mkdir(exist_ok=True)

    data = {}
    for device, path, pump_hz, xlabel in DEVICES:
        p = Path(path)
        if not p.exists():
            print(f"{device}: {p} missing, skipped")
            continue
        data[device] = (load(device, p, pump_hz, cache_dir / f"{device}.json"),
                        xlabel, pump_hz)

    ncols = len(data)
    fig, axes = plt.subplots(4, ncols, figsize=(5.2 * ncols, 13.5),
                             constrained_layout=True, squeeze=False)

    for col, (device, (rows, xlabel, pump_hz)) in enumerate(data.items()):
        control = np.array([r["control_value"] for r in rows])
        k = np.array([np.nan if r["k"] is None else r["k"] for r in rows])
        on_comb = np.array([np.nan if r["on_comb"] is None else r["on_comb"]
                            for r in rows])
        # The generator share is a fraction OF the off-comb power. Where there
        # is essentially no off-comb power (a clean period-1 orbit) it divides
        # a floor by a floor and means nothing, so mask it rather than plot a
        # flat artefact -- guarcello otherwise shows a spurious constant 0.86
        # across 15 dB of period-1 operation.
        share = np.array([
            np.nan if (r["generator_share"] is None or r["on_comb"] is None
                       or (1.0 - r["on_comb"]) < 1.0e-4)
            else r["generator_share"] for r in rows
        ])
        sigma = np.array([r["strobe_std"] for r in rows])
        regimes = [regime(r["k"], r["on_comb"]) for r in rows]

        # Row 0 -- instrument 1: the 0-1 test.
        ax = axes[0, col]
        ax.plot(control, k, "s-", color="#2166ac", markersize=3, linewidth=1.0)
        ax.axhline(0.0, color="#666", linewidth=0.6)
        ax.axhline(1.0, color="#666", linewidth=0.6)
        ax.axhspan(-REGULAR_K, REGULAR_K, color=COLOUR["regular"], alpha=0.15)
        ax.axhspan(CHAOTIC_K, 1.15, color=COLOUR["chaotic"], alpha=0.15)
        ax.set_ylim(-0.85, 1.15)
        ax.set_ylabel(r"$K$  (0-1 test)")
        ax.set_title(f"{device}\ninstrument 1: $K$", fontsize=10)
        ax.set_xlabel(xlabel)

        # Row 1 -- instrument 3: spectral structure.
        ax = axes[1, col]
        ax.plot(control, on_comb, "o-", color="#252525", markersize=3,
                linewidth=1.0, label=r"power on the pump comb $n f_p$")
        ax.plot(control, share, "^--", color="#1a9850", markersize=3,
                linewidth=1.0,
                label="off-comb power one extra\ngenerator explains")
        ax.axhline(1.0, color="#666", linewidth=0.6)
        ax.set_ylim(-0.05, 1.10)
        ax.set_ylabel("fraction")
        ax.set_title("instrument 3: spectral structure", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=7, loc="lower left")

        # Row 2 -- sigma, for orientation between the two.
        ax = axes[2, col]
        for point_regime, x, y in zip(regimes, control, sigma):
            ax.semilogy([x], [y], "o", color=COLOUR[point_regime], markersize=4)
        ax.semilogy(control, sigma, "-", color="#999", linewidth=0.7, zorder=0)
        ax.set_ylabel(r"$\sigma$ of the stroboscopic section")
        ax.set_title("section spread, coloured by regime", fontsize=10)
        ax.set_xlabel(xlabel)

        # Row 3 -- instrument 2: return-map shape, three representative points.
        inner = axes[3, col]
        inner.set_xticks([]); inner.set_yticks([])
        for spine in inner.spines.values():
            spine.set_visible(False)
        picks = []
        for wanted in ("floor", "regular", "chaotic"):
            candidates = [r for r, g in zip(rows, regimes) if g == wanted]
            if candidates:
                picks.append((candidates[len(candidates) // 2], wanted))
        width = 1.0 / max(len(picks), 1)
        for index, (row, tag) in enumerate(picks):
            # inset_axes keeps the panel inside its parent's box; fig.add_axes
            # with a manually computed position collided with the row above.
            sub = inner.inset_axes([index * width + 0.02 * width, 0.0,
                                    width * 0.94, 0.84])
            return_map(sub, strobe_of(row, device),
                       f"{tag}\n{row['control_value']:.4f}", COLOUR[tag])
        inner.set_title("instrument 2: return-map shape", fontsize=10)

    path = args.output / "three_instruments.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
