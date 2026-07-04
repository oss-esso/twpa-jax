# experiments/exp17_ripple_map_crosscheck.py
"""Experiment 17b: cross-check an IPM gain map against the S42 ripple.

Port of Harmonia.jl ``ripple_map_crosscheck`` onto the twpa_jax stack. It closes
the loop with the pump maps produced by :mod:`exp16_ipm_gain_map`: the map's
headline cells are frequently past-fold **artifacts** (the pump HB has passed the
fold, so the reported gain is not a converged solution), so the map cannot be
read by raw gain. This script instead:

1. nominates a **pool** of the ``POOL_N`` highest-gain map cells;
2. **snaps** each cell's pump ``fp`` onto the nearest passive ``S42`` ripple
   ``peak + period/3`` target (the +120-degrees design point), recording the
   original map offset and the shift;
3. **re-solves** the pump + gain at that snapped ``fp`` (the cell's own pump
   current) honestly (exp08 continuation), and keeps the cell only if it is
   **physical** -- ``final_status == VALID_CONVERGED``, bounded node flux, and a
   finite gain in ``[-40, 60]`` dB.

It stops once ``top_n`` cells verify. The output uses the same ``manifest.json``
layout as :mod:`exp17_ripple_pump_placement`, so the same plotter renders it;
each record additionally carries the source map gain and the snap bookkeeping.

Usage:
    python experiments/exp17_ripple_map_crosscheck.py \
        --design 2c --map-dir outputs/ipm_gain_map_35x35_power_freq --top-n 4
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

POOL_N = 12
FLUX_MAX = 1.0e3
GAIN_WINDOW_DB = (-40.0, 60.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", default="2c", choices=sorted(rc.DESIGNS))
    p.add_argument("--map-dir", type=Path, required=True,
                   help="Map directory holding gain_map_points.csv.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Default: outputs/ripple_map_crosscheck_<design>.")
    p.add_argument("--ipm-dir", type=Path, default=None,
                   help="Reuse an existing design dir instead of rebuilding.")
    p.add_argument("--passive-npz", type=Path, default=None,
                   help="Reuse a passive_ripple.npz (else recompute).")
    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--pool-n", type=int, default=POOL_N)
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
    p.add_argument("--extra-sparams", action="store_true")
    p.add_argument("--pump-timeout-s", type=float, default=300.0)
    p.add_argument("--gain-timeout-s", type=float, default=900.0)
    return p.parse_args()


def load_map_pool(map_dir: Path, pool_n: int) -> list[dict[str, float]]:
    """Read the ``pool_n`` highest-gain cells from a gain map's points CSV.

    Accepts either the :mod:`exp16_ipm_gain_map` layout
    (``gain_map_points.csv``) or the warm-start map layout (``map_points.csv``);
    both share the ``pump_freq_ghz`` / ``pump_current_peak_a`` / ``gain_db``
    columns.
    """
    csv_path = next(
        (map_dir / name for name in ("gain_map_points.csv", "map_points.csv")
         if (map_dir / name).exists()),
        None,
    )
    if csv_path is None:
        raise FileNotFoundError(
            f"no gain_map_points.csv or map_points.csv in {map_dir}"
        )

    cells: list[dict[str, float]] = []
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
                "gain_db": gain,
                "fp_ghz": fp,
                "current_a": current,
                "power_dbm": float(row.get("pump_power_dbm", "nan") or "nan"),
            })
    cells.sort(key=lambda c: c["gain_db"], reverse=True)
    return cells[:pool_n]


def passive_peaks(
    ipm_dir: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, Path]:
    """Return S42 peak frequencies (GHz) and the passive npz path."""
    if args.passive_npz is not None and Path(args.passive_npz).exists():
        data = np.load(args.passive_npz)
        fg = data["freq_ghz"]
        s42_db = data["s42_db"]
        path = Path(args.passive_npz)
    else:
        start, stop, n = args.ripple_grid_ghz
        freqs = np.linspace(start * 1e9, stop * 1e9, int(n))
        t0 = time.perf_counter()
        S = rc.passive_s_matrix(ipm_dir, freqs, ports=(1, 2, 3, 4),
                                z0_ohm=args.z0_ohm)
        fg = freqs / 1e9
        s42_db = rc.db20(S[:, 3, 1])
        path = outdir / "passive_ripple.npz"
        np.savez(path, freq_ghz=fg, s21_db=rc.db20(S[:, 1, 0]), s42_db=s42_db,
                 s_real=S.real, s_imag=S.imag,
                 ports=np.array([1, 2, 3, 4], dtype=np.int64))
        print(f"passive S-matrix in {time.perf_counter()-t0:.1f}s -> {path}",
              flush=True)

    peaks = rc.find_s42_peaks(fg, s42_db, tuple(args.ripple_band_ghz))
    return peaks, path


def verify_cell(
    ipm_dir: Path,
    point_dir: Path,
    cell: dict[str, float],
    placement: rc.Placement,
    map_offset_deg: float,
    ic_a: float,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Re-solve one snapped cell; return a manifest record if it verifies."""
    fp = placement.fp_ghz
    ratio = cell["current_a"] / ic_a
    outcome = rc.solve_pump(
        ipm_dir,
        point_dir / "pump",
        fp_ghz=fp,
        ratio_ic=ratio,
        ic_a=ic_a,
        pump_mode_count=args.pump_mode_count,
        nt=args.nt,
        continuation_steps=args.continuation_steps,
        timeout_s=args.pump_timeout_s,
    )

    peak_db = float("nan")
    peak_at = float("nan")
    sweep_paths: dict[str, str] = {}
    if outcome.accepted:
        start, stop, n = args.signal_grid_ghz
        pairs = {"s21": (1, 2)}
        if args.extra_sparams:
            pairs.update({"s12": (2, 1), "s24": (4, 2)})
        for name, (src, out) in pairs.items():
            csv_path = rc.gain_sweep(
                ipm_dir, outcome.pump_dir, point_dir / f"gain_{name}",
                fp_ghz=fp, source_port=src, out_port=out,
                signal_start_ghz=start, signal_stop_ghz=stop, points=int(n),
                sidebands=args.sidebands, gamma_nt=args.gamma_nt,
                timeout_s=args.gain_timeout_s,
            )
            sweep_paths[name] = str(csv_path)
        fx, gy = rc.read_gain_sweep(Path(sweep_paths["s21"]))
        if gy.size:
            peak_db = float(np.nanmax(gy))
            peak_at = float(fx[int(np.nanargmax(gy))])

    lo, hi = GAIN_WINDOW_DB
    verified = (
        outcome.accepted
        and np.isfinite(peak_db)
        and lo < peak_db < hi
        and np.isfinite(outcome.flux_over_phi0)
        and outcome.flux_over_phi0 < FLUX_MAX
    )
    shift_mhz = (fp - cell["fp_ghz"]) * 1e3
    print(f"    map {cell['gain_db']:.2f} dB @ {cell['fp_ghz']:.3f} GHz "
          f"({map_offset_deg:+.0f} deg, {ratio:.2f} Ic) -> snap {fp:.3f} GHz "
          f"(shift {shift_mhz:+.1f} MHz): re-solve {peak_db:.2f} dB "
          f"flux={outcome.flux_over_phi0:.3g} status={outcome.final_status} "
          f"VERIFIED={verified}", flush=True)

    if not verified:
        return None

    return {
        "fp_ghz": fp,
        "ref_peak_ghz": placement.ref_peak_ghz,
        "period_ghz": placement.period_ghz,
        "offset_mhz": placement.offset_mhz,
        "offset_deg": placement.offset_deg,
        "ws_marker_ghz": fp - args.signal_detuning_mhz / 1e3,
        "ic_ratio": ratio,
        "ic_current_a": cell["current_a"],
        "flux_over_phi0": outcome.flux_over_phi0,
        "coeff_rel": outcome.coeff_rel,
        "converged": True,
        "strictly_converged": outcome.converged,
        "reached_full_scale": outcome.reached_full_scale,
        "coherent": True,
        "peak_s21_db": peak_db,
        "peak_s21_at_ghz": peak_at,
        "map_gain_db": cell["gain_db"],
        "map_fp_ghz": cell["fp_ghz"],
        "map_offset_deg": map_offset_deg,
        "map_power_dbm": cell["power_dbm"],
        "shift_mhz": shift_mhz,
        "sweep": sweep_paths,
        "pump_dir": str(outcome.pump_dir),
    }


def main() -> None:
    args = parse_args()
    outdir = args.outdir or (ROOT / "outputs" / f"ripple_map_crosscheck_{args.design}")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.ipm_dir is not None:
        ipm_dir = Path(args.ipm_dir)
    else:
        ipm_dir = outdir / "ipm_design"
        rc.build_design(args.design, ipm_dir)
    ic_a = rc.ic_reference_a(ipm_dir)

    peaks, passive_path = passive_peaks(ipm_dir, outdir, args)
    pool = load_map_pool(args.map_dir, args.pool_n)
    print(f"design={args.design}  map={args.map_dir.name}  "
          f"pool={len(pool)} cells  S42 peaks={peaks.size}  "
          f"keeping top {args.top_n} that verify", flush=True)

    points: list[dict[str, Any]] = []
    for i, cell in enumerate(pool, 1):
        placement, map_offset_deg = rc.snap_to_120(cell["fp_ghz"], peaks)
        point_dir = outdir / f"pool_{i:02d}_fp{round(placement.fp_ghz*1000)}"
        print(f"[pool {i}] map gain {cell['gain_db']:.2f} dB", flush=True)
        rec = verify_cell(ipm_dir, point_dir, cell, placement, map_offset_deg,
                          ic_a, args)
        if rec is not None:
            rec["point"] = len(points) + 1
            points.append(rec)
        if len(points) >= args.top_n:
            break

    manifest = {
        "design": args.design,
        "ipm_dir": str(ipm_dir),
        "map_dir": str(args.map_dir),
        "z0_ohm": args.z0_ohm,
        "ic_a": ic_a,
        "ripple_band_ghz": list(args.ripple_band_ghz),
        "signal_detuning_mhz": args.signal_detuning_mhz,
        "passive_ripple_npz": str(passive_path),
        "pool_n": args.pool_n,
        "flux_max": FLUX_MAX,
        "points": points,
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path} ({len(points)} verified candidates, "
          f"all snapped to +120 deg)", flush=True)


if __name__ == "__main__":
    main()
