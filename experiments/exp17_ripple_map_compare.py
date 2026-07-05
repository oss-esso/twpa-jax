# experiments/exp17_ripple_map_compare.py
"""Experiment 17c: compare a map cell's operating point three ways.

For the ``top_n`` highest-gain cells of a pump/gain map, this re-runs the signal
gain (S21 and S24 vs signal frequency) under three pump conditions, to explain
*why* a headline map cell does or does not reproduce:

  cold_snap : pump solved **cold** at the +120-degree snapped ``fp`` (the coupler
              ripple design point). If this ``fp`` lands in an S42 ripple dip the
              gain collapses -- the snap detuned the pump off the ripple peak.
  cold_orig : pump solved **cold** at the cell's **original** map ``fp``. Isolates
              the snap detuning (cold_orig high, cold_snap low => the +120 point
              is in a dip) from the warm-branch effect (both cold low, warm high).
  map_warm  : the cell's **own stored map pump solution** reused directly (no
              re-solve) -- the warm-start branch the map actually reported.

Each variant is swept for S21 (1->2) and S24 (4->2). The passive (pump-off) S42 /
S21 / S24 ripple is saved for the plotter's context panel.

Usage:
    python experiments/exp17_ripple_map_compare.py \
        --design 2c --ipm-dir outputs/ipm_python_design \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20 --top-n 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ripple_common as rc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", default="2c", choices=sorted(rc.DESIGNS))
    p.add_argument("--map-dir", type=Path, required=True)
    p.add_argument("--ipm-dir", type=Path, default=None,
                   help="Reuse an existing design dir instead of rebuilding.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Default: outputs/ripple_map_compare_<design>.")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument("--ripple-band-ghz", type=float, nargs=2, default=(5.5, 10.0),
                   metavar=("LO", "HI"))
    p.add_argument("--ripple-grid-ghz", type=float, nargs=3,
                   default=(4.0, 11.0, 1401), metavar=("START", "STOP", "N"))
    p.add_argument("--signal-grid-ghz", type=float, nargs=3,
                   default=(4.0, 11.0, 121), metavar=("START", "STOP", "N"))
    p.add_argument("--sidebands", type=int, default=10)
    p.add_argument("--gamma-nt", type=int, default=96)
    p.add_argument("--pump-mode-count", type=int, default=10)
    p.add_argument("--nt", type=int, default=40)
    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--signal-detuning-mhz", type=float, default=100.0)
    p.add_argument("--pump-timeout-s", type=float, default=90.0)
    p.add_argument("--gain-timeout-s", type=float, default=300.0)
    return p.parse_args()


def load_top_cells(map_dir: Path, top_n: int) -> list[dict[str, Any]]:
    """Top-``top_n`` finite-gain cells with their stored map pump dir."""
    csv_path = next((map_dir / n for n in ("map_points.csv", "gain_map_points.csv")
                     if (map_dir / n).exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"no map_points.csv in {map_dir}")
    cells: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                gain = float(row["gain_db"])
                fp = float(row["pump_freq_ghz"])
                current = float(row["pump_current_peak_a"])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(gain):
                continue
            cells.append({
                "gain_db": gain, "fp_ghz": fp, "current_a": current,
                "power_dbm": float(row.get("pump_power_dbm", "nan") or "nan"),
                "pump_dir": (row.get("pump_dir") or "").strip(),
            })
    cells.sort(key=lambda c: c["gain_db"], reverse=True)
    return cells[:top_n]


def _sweeps_from_pump(
    ipm_dir: Path, pump_dir: Path, base: Path, fp: float, args: argparse.Namespace,
) -> dict[str, Any]:
    """Run S21 (1->2) and S24 (4->2) signal sweeps from an existing pump dir."""
    start, stop, n = args.signal_grid_ghz
    out: dict[str, Any] = {"fp_ghz": fp}
    for name, (src, o) in {"s21": (1, 2), "s24": (4, 2)}.items():
        csv_path = rc.gain_sweep(
            ipm_dir, pump_dir, base / f"gain_{name}", fp_ghz=fp,
            source_port=src, out_port=o, signal_start_ghz=start,
            signal_stop_ghz=stop, points=int(n), sidebands=args.sidebands,
            gamma_nt=args.gamma_nt, timeout_s=args.gain_timeout_s)
        out[name] = str(csv_path)
        if name == "s21":
            fx, gy = rc.read_gain_sweep(csv_path)
            out["peak_s21_db"] = float(np.nanmax(gy)) if gy.size else float("nan")
            out["peak_s21_at_ghz"] = (float(fx[int(np.nanargmax(gy))])
                                      if gy.size else float("nan"))
    return out


def cold_variant(
    ipm_dir: Path, base: Path, fp: float, ratio: float, ic_a: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    outcome = rc.solve_pump(
        ipm_dir, base / "pump", fp_ghz=fp, ratio_ic=ratio, ic_a=ic_a,
        pump_mode_count=args.pump_mode_count, nt=args.nt,
        continuation_steps=args.continuation_steps, timeout_s=args.pump_timeout_s)
    rec: dict[str, Any] = {
        "fp_ghz": fp, "accepted": bool(outcome.accepted),
        "final_status": outcome.final_status,
        "flux_over_phi0": outcome.flux_over_phi0,
        "reached_full_scale": bool(outcome.reached_full_scale),
        "peak_s21_db": float("nan"), "peak_s21_at_ghz": float("nan"),
    }
    if outcome.accepted:
        rec.update(_sweeps_from_pump(ipm_dir, outcome.pump_dir, base, fp, args))
    return rec


def map_warm_variant(
    ipm_dir: Path, base: Path, fp: float, cell: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pump_dir = Path(cell["pump_dir"]) if cell["pump_dir"] else None
    rec: dict[str, Any] = {
        "fp_ghz": fp, "pump_dir": cell["pump_dir"],
        "peak_s21_db": float("nan"), "peak_s21_at_ghz": float("nan"),
    }
    if pump_dir is not None and (pump_dir / "pump_solution.npz").exists():
        rec["available"] = True
        rec.update(_sweeps_from_pump(ipm_dir, pump_dir, base, fp, args))
    else:
        rec["available"] = False
    return rec


def main() -> None:
    args = parse_args()
    outdir = args.outdir or (ROOT / "outputs" / f"ripple_map_compare_{args.design}")
    outdir.mkdir(parents=True, exist_ok=True)

    ipm_dir = Path(args.ipm_dir) if args.ipm_dir else outdir / "ipm_design"
    if args.ipm_dir is None:
        rc.build_design(args.design, ipm_dir)
    ic_a = rc.ic_reference_a(ipm_dir)

    # Passive ripple (pump off): S42 (coupled), S21 (through), S24.
    start, stop, n = args.ripple_grid_ghz
    freqs = np.linspace(start * 1e9, stop * 1e9, int(n))
    t0 = time.perf_counter()
    S = rc.passive_s_matrix(ipm_dir, freqs, ports=(1, 2, 3, 4), z0_ohm=args.z0_ohm)
    fg = freqs / 1e9
    s42_db, s21_db, s24_db = rc.db20(S[:, 3, 1]), rc.db20(S[:, 1, 0]), rc.db20(S[:, 1, 3])
    passive_path = outdir / "passive_ripple.npz"
    np.savez(passive_path, freq_ghz=fg, s21_db=s21_db, s42_db=s42_db, s24_db=s24_db)
    print(f"passive S-matrix in {time.perf_counter()-t0:.1f}s -> {passive_path}",
          flush=True)
    peaks = rc.find_s42_peaks(fg, s42_db, tuple(args.ripple_band_ghz))

    cells = load_top_cells(args.map_dir, args.top_n)
    print(f"design={args.design}  map={args.map_dir.name}  top {len(cells)} cells",
          flush=True)

    points: list[dict[str, Any]] = []
    for i, cell in enumerate(cells, 1):
        placement, map_offset_deg = rc.snap_to_120(cell["fp_ghz"], peaks)
        snap_fp, orig_fp = placement.fp_ghz, cell["fp_ghz"]
        base = outdir / f"point{i:02d}_fp{round(orig_fp*1000)}"
        det = args.signal_detuning_mhz / 1e3
        print(f"[{i}] map {cell['gain_db']:.2f} dB @ {orig_fp:.3f} GHz "
              f"({cell['current_a']/ic_a:.2f} Ic) -> snap {snap_fp:.3f} GHz "
              f"({map_offset_deg:+.0f}->+120 deg)", flush=True)

        ratio = cell["current_a"] / ic_a
        variants = {
            "cold_snap": cold_variant(ipm_dir, base / "cold_snap", snap_fp, ratio, ic_a, args),
            "cold_orig": cold_variant(ipm_dir, base / "cold_orig", orig_fp, ratio, ic_a, args),
            "map_warm": map_warm_variant(ipm_dir, base / "map_warm", orig_fp, cell, args),
        }
        for tag, v in variants.items():
            print(f"    {tag:9s} fp={v['fp_ghz']:.3f}: peak S21 "
                  f"{v.get('peak_s21_db', float('nan')):.2f} dB", flush=True)

        points.append({
            "point": i,
            "map_gain_db": cell["gain_db"], "map_fp_ghz": orig_fp,
            "map_power_dbm": cell["power_dbm"], "map_offset_deg": map_offset_deg,
            "snap_fp_ghz": snap_fp, "ref_peak_ghz": placement.ref_peak_ghz,
            "period_ghz": placement.period_ghz, "offset_mhz": placement.offset_mhz,
            "offset_deg": placement.offset_deg,
            "ic_ratio": ratio, "ic_current_a": cell["current_a"],
            "ws_snap_ghz": snap_fp - det, "ws_orig_ghz": orig_fp - det,
            "variants": variants,
        })

    manifest = {
        "design": args.design, "ipm_dir": str(ipm_dir),
        "map_dir": str(args.map_dir), "ic_a": ic_a,
        "signal_detuning_mhz": args.signal_detuning_mhz,
        "passive_ripple_npz": str(passive_path),
        "compare": True, "points": points,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
    print(f"wrote {outdir/'manifest.json'} ({len(points)} points)", flush=True)


if __name__ == "__main__":
    main()
