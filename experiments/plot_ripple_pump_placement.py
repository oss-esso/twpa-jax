# experiments/plot_ripple_pump_placement.py
"""Plot the S42-ripple pump-placement workflow (twpa_jax port).

Reads a ``manifest.json`` written by :mod:`exp17_ripple_pump_placement` or
:mod:`exp17_ripple_map_crosscheck` and emits one PNG per operating point:

  1. Pump OFF - passive ``|S42|`` coupler ripple + ``|S21|`` through path, with
     the reference S42 peak, the placed pump ``fp``, and the measured degrees
     offset (+120 target). Map cross-check points also show the raw map ``fp``.
  2. Pump ON  - ``S21`` gain, with the ``ws = fp - detuning`` marker.
  3. Pump ON  - ``S12`` magnitude   (only if ``--extra-sparams`` was swept).
  4. Pump ON  - ``S24`` magnitude    (only if ``--extra-sparams`` was swept).

Panels 3-4 are skipped when their sweep CSVs are absent, so an S21-only run
produces a compact two-panel figure.

Usage:
    python experiments/plot_ripple_pump_placement.py \
        --rundir outputs/ripple_pump_placement_2c
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read_sweep(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    fx: list[float] = []
    gy: list[float] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    fx.append(float(row["signal_ghz"]))
                    gy.append(float(row["gain_db"]))
                except (KeyError, ValueError):
                    continue
    order = np.argsort(fx)
    return np.asarray(fx)[order], np.asarray(gy)[order]


def plot_point(
    pt: dict[str, Any],
    design: str,
    rf: np.ndarray,
    r42: np.ndarray,
    r21: np.ndarray,
    outdir: Path,
) -> Path:
    fp = pt["fp_ghz"]
    ref = pt["ref_peak_ghz"]
    ws = pt["ws_marker_ghz"]
    map_fp = pt.get("map_fp_ghz")

    sweep = pt.get("sweep", {})
    fx21, s21 = read_sweep(Path(sweep["s21"])) if "s21" in sweep else (np.array([]), np.array([]))
    have_s12 = "s12" in sweep and Path(sweep["s12"]).exists()
    have_s24 = "s24" in sweep and Path(sweep["s24"]).exists()

    peak_db = pt.get("peak_s21_db", float("nan"))
    peak_at = pt.get("peak_s21_at_ghz", float("nan"))

    if "map_gain_db" in pt:
        state = (f"map {pt['map_gain_db']:.1f} dB  |  "
                 f"snap {map_fp:.3f}->{fp:.3f} "
                 f"({pt['map_offset_deg']:+.0f}->+120 deg)")
    else:
        state = "CONVERGED"

    n_panels = 2 + int(have_s12) + int(have_s24)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels))
    ax = np.atleast_1d(axes)

    fig.suptitle(
        f"{design} IPM JTWPA - S42-ripple pump placement (point {pt['point']})   "
        f"fp={fp:.3f} GHz, Ip={pt['ic_ratio']:.2f} Ic "
        f"({pt['ic_current_a']*1e6:.3f} uA)  {state}\n"
        f"fp = S42 peak {ref:.3f} GHz {pt['offset_mhz']:+.0f} MHz "
        f"({pt['offset_deg']:+.0f} deg, target +120)   |   "
        f"peak S21 {peak_db:.1f} dB @ {peak_at:.2f} GHz",
        fontsize=9,
    )

    a = ax[0]
    a.plot(rf, r42, lw=1.3, color="0.35", label="|S42| (coupled)")
    a.plot(rf, r21, lw=1.2, color="C7", alpha=0.9, label="|S21| (through)")
    a.axvline(ref, color="C3", ls="-", lw=1.4, label=f"reference peak {ref:.3f} GHz")
    a.axvline(fp, color="C0", ls="--", lw=1.6, label=f"pump fp {fp:.3f} GHz (+120 deg)")
    if map_fp is not None:
        a.axvline(map_fp, color="0.5", ls=":", lw=1.4, label=f"map fp {map_fp:.3f} GHz")
    a.annotate(f"{pt['offset_mhz']:+.0f} MHz\n{pt['offset_deg']:+.0f} deg",
               xy=((ref + fp) / 2, r42.max() - 6), ha="center", fontsize=8, color="C0")
    xs = [ref, fp] + ([map_fp] if map_fp is not None else [])
    a.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    a.set_ylabel("PUMP OFF\n|S| (dB)")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=7, loc="lower right", ncol=2)

    a = ax[1]
    a.plot(fx21, s21, lw=1.6, color="C0")
    a.axvline(fp, color="0.5", ls=":", lw=1.0)
    if fx21.size:
        a.axvline(ws, color="C3", ls="--", lw=1.2)
        a.plot(ws, np.interp(ws, fx21, s21), "o", color="C3", ms=6)
        a.annotate(f"ws = fp-100 MHz\n= {ws:.3f} GHz",
                   xy=(ws, np.interp(ws, fx21, s21)),
                   xytext=(-95, -6), textcoords="offset points", fontsize=8, color="C3")
    a.set_ylabel("PUMP ON\nS21 gain (dB)")
    a.grid(True, alpha=0.3)

    idx = 2
    if have_s12:
        fx12, s12 = read_sweep(Path(sweep["s12"]))
        a = ax[idx]
        a.plot(fx12, s12, lw=1.6, color="C1")
        a.axvline(fp, color="0.5", ls=":", lw=1.0)
        a.set_ylabel("PUMP ON\nS12 mag (dB)")
        a.grid(True, alpha=0.3)
        idx += 1
    if have_s24:
        fx24, s24 = read_sweep(Path(sweep["s24"]))
        a = ax[idx]
        a.plot(fx24, s24, lw=1.6, color="C2")
        a.axvline(fp, color="0.5", ls=":", lw=1.0)
        a.set_ylabel("PUMP ON\nS24 mag (dB)")
        a.grid(True, alpha=0.3)
        idx += 1

    ax[-1].set_xlabel("Signal frequency (GHz)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"ripple_pump_placement_{design}_point{pt['point']}_fp{round(fp*1000)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


_VARIANTS = (
    ("cold_snap", "C0", "cold @ +120 snapped fp"),
    ("cold_orig", "C1", "cold @ original map fp"),
    ("map_warm", "C2", "map warm solution (as-run)"),
)


def plot_compare_point(
    pt: dict[str, Any],
    design: str,
    rf: np.ndarray,
    r42: np.ndarray,
    r21: np.ndarray,
    r24: np.ndarray,
    outdir: Path,
) -> Path:
    """3-panel compare figure: pump-off ripple, S21 pump-on, S24 pump-on.

    Panels 2-3 overlay the three pump variants (cold @ snapped fp, cold @
    original fp, map's own warm solution) so the snap-into-a-dip vs warm-branch
    cause of a low +120-degree gain is visible directly.
    """
    snap_fp = pt["snap_fp_ghz"]
    orig_fp = pt["map_fp_ghz"]
    ref = pt["ref_peak_ghz"]
    variants = pt["variants"]

    fig, ax = plt.subplots(3, 1, figsize=(9, 10))
    fig.suptitle(
        f"{design} IPM JTWPA - map-cell compare (point {pt['point']})   "
        f"map {pt['map_gain_db']:.1f} dB @ {orig_fp:.3f} GHz "
        f"({pt['map_power_dbm']:.1f} dBm, {pt['ic_ratio']:.2f} Ic)   "
        f"snap {orig_fp:.3f}->{snap_fp:.3f} GHz "
        f"({pt['map_offset_deg']:+.0f}->+120 deg)",
        fontsize=9,
    )

    # Panel 1: pump OFF passive ripple + placement.
    a = ax[0]
    a.plot(rf, r42, lw=1.3, color="0.35", label="|S42| coupled")
    a.plot(rf, r21, lw=1.1, color="C7", alpha=0.9, label="|S21| through")
    if r24 is not None:
        a.plot(rf, r24, lw=1.1, color="C9", alpha=0.7, label="|S24|")
    a.axvline(ref, color="C3", ls="-", lw=1.3, label=f"S42 peak {ref:.3f}")
    a.axvline(snap_fp, color="C0", ls="--", lw=1.6, label=f"snapped fp {snap_fp:.3f} (+120)")
    a.axvline(orig_fp, color="C1", ls=":", lw=1.6, label=f"original map fp {orig_fp:.3f}")
    xs = [ref, snap_fp, orig_fp]
    a.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    a.set_ylabel("PUMP OFF\n|S| (dB)")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=7, loc="lower right", ncol=2)

    # Panels 2-3: S21 / S24 pump ON, three variants overlaid.
    for panel, key, ylabel in ((1, "s21", "S21 gain"), (2, "s24", "S24 mag")):
        a = ax[panel]
        for tag, color, lbl in _VARIANTS:
            v = variants.get(tag, {})
            path = v.get(key)
            if not path:
                continue
            fx, gy = read_sweep(Path(path))
            if not fx.size:
                continue
            peak = v.get("peak_s21_db", float("nan")) if key == "s21" else float("nan")
            suffix = f"  (peak {peak:.1f} dB)" if key == "s21" and np.isfinite(peak) else ""
            a.plot(fx, gy, lw=1.6, color=color, label=f"{lbl}{suffix}")
        a.axvline(pt["ws_snap_ghz"], color="C0", ls="--", lw=1.0, alpha=0.7)
        a.axvline(pt["ws_orig_ghz"], color="C1", ls=":", lw=1.0, alpha=0.7)
        a.set_ylabel(f"PUMP ON\n{ylabel} (dB)")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7, loc="upper right")

    ax[-1].set_xlabel("Signal frequency (GHz)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"ripple_map_compare_{design}_point{pt['point']}_fp{round(orig_fp*1000)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rundir", type=Path, required=True,
                    help="Run directory containing manifest.json.")
    ap.add_argument("--label", default=None,
                    help="Label used in titles/filenames (default: design tag).")
    args = ap.parse_args()

    rundir = Path(args.rundir).resolve()
    manifest = json.loads((rundir / "manifest.json").read_text(encoding="utf-8"))
    label = args.label or manifest.get("design", "ipm")

    passive = np.load(manifest["passive_ripple_npz"])
    rf = passive["freq_ghz"]
    r42 = passive["s42_db"]
    r21 = passive["s21_db"]
    r24 = passive["s24_db"] if "s24_db" in passive.files else None

    if manifest.get("compare"):
        for pt in manifest["points"]:
            out = plot_compare_point(pt, label, rf, r42, r21, r24, rundir / "plots")
            w = pt["variants"].get("map_warm", {}).get("peak_s21_db", float("nan"))
            cs = pt["variants"].get("cold_snap", {}).get("peak_s21_db", float("nan"))
            co = pt["variants"].get("cold_orig", {}).get("peak_s21_db", float("nan"))
            print(f"[{label}] point {pt['point']} map {pt['map_gain_db']:.1f} dB: "
                  f"cold_snap {cs:.1f} | cold_orig {co:.1f} | map_warm {w:.1f} "
                  f"-> {out.name}")
        return

    for pt in manifest["points"]:
        out = plot_point(pt, label, rf, r42, r21, rundir / "plots")
        extra = ""
        if "map_gain_db" in pt:
            extra = (f", map {pt['map_gain_db']:.1f} dB, "
                     f"coherent={pt.get('coherent')}")
        print(f"[{label}] point {pt['point']} fp {pt['fp_ghz']:.3f} GHz "
              f"({pt['offset_deg']:+.0f} deg): peak S21 {pt['peak_s21_db']:.1f} dB"
              f"{extra} -> {out.name}")


if __name__ == "__main__":
    main()
