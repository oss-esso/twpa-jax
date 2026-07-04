# experiments/exp17_ripple_pump_placement.py
"""Experiment 17: S42-ripple pump placement for the IPM JTWPA (2c/3c).

Port of the Harmonia.jl ``ripple_pump_placement`` workflow onto the faster
twpa_jax pump/gain stack (exp07/exp08/exp09). It chooses the pump frequency from
the device's *passive* coupler ripple -- cheap, deterministic, and needing no
expensive pump-power/frequency map -- then finds the strongest pump current that
genuinely converges, so every reported gain curve is a physical solution.

Steps (per design):

1. Build the design (:func:`ripple_common.build_design`).
2. Solve the passive (pump-off) 4-port S-matrix on a fine grid; the pump-port
   transmission ``|S42|`` shows the periodic coupler ripple.
3. Place each pump ``fp`` one third of a local ripple period (**+120 degrees**)
   above a strong ``S42`` peak, auto-selected so ``fp`` lands in the map band.
4. Ladder the pump current (x Ic) at each ``fp`` and keep the strongest solve
   that converges (exp08 ``final_status == VALID_CONVERGED``).
5. Sweep the signal frequency there (exp09) to get ``S21`` (and optionally the
   ``S12`` / ``S24`` diagnostics) at the placed operating point.

Outputs (``outputs/ripple_pump_placement_<design>/``):
    passive_ripple.npz      freq_ghz, s21_db, s42_db, full S, port order
    manifest.json           one record per operating point + sweep CSV paths
    point_<k>_fp<MHz>/...    per-point pump + gain artifacts

Usage:
    python experiments/exp17_ripple_pump_placement.py --design 2c
    python experiments/exp17_ripple_pump_placement.py --design 3c --extra-sparams
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--outdir", type=Path, default=None,
                   help="Default: outputs/ripple_pump_placement_<design>.")
    p.add_argument("--ipm-dir", type=Path, default=None,
                   help="Reuse an existing design dir instead of rebuilding.")
    p.add_argument("--z0-ohm", type=float, default=50.0)

    p.add_argument("--ripple-band-ghz", type=float, nargs=2, default=(5.5, 10.0),
                   metavar=("LO", "HI"), help="Band searched for S42 peaks.")
    p.add_argument("--ripple-grid-ghz", type=float, nargs=3,
                   default=(4.0, 11.0, 1401), metavar=("START", "STOP", "N"),
                   help="Fine passive sweep grid.")
    p.add_argument("--map-pump-band-ghz", type=float, nargs=2, default=(6.0, 8.5),
                   metavar=("LO", "HI"), help="Placed fp must land in this band.")
    p.add_argument("--n-points", type=int, default=4)

    p.add_argument("--ic-ladder", type=str, default="2.0,2.5,3.0,3.5",
                   help="Comma-separated pump-current ratios (x Ic), ascending.")
    p.add_argument("--signal-grid-ghz", type=float, nargs=3,
                   default=(4.0, 11.0, 121), metavar=("START", "STOP", "N"),
                   help="Pump-on signal sweep grid.")
    p.add_argument("--sidebands", type=int, default=10)
    p.add_argument("--gamma-nt", type=int, default=96)
    p.add_argument("--pump-mode-count", type=int, default=10)
    p.add_argument("--nt", type=int, default=40)
    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--signal-detuning-mhz", type=float, default=100.0,
                   help="ws marker: ws = fp - detuning.")
    p.add_argument("--extra-sparams", action="store_true",
                   help="Also sweep S12 (2->1) and S24 (4->2) at each point.")
    p.add_argument("--pump-timeout-s", type=float, default=300.0)
    p.add_argument("--gain-timeout-s", type=float, default=900.0)
    return p.parse_args()


def compute_passive(
    ipm_dir: Path,
    outdir: Path,
    grid: tuple[float, float, float],
    z0_ohm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Solve and cache the passive S-matrix; return grid, s42_db, s21_db, path."""
    start, stop, n = grid
    freqs = np.linspace(start * 1e9, stop * 1e9, int(n))
    t0 = time.perf_counter()
    S = rc.passive_s_matrix(ipm_dir, freqs, ports=(1, 2, 3, 4), z0_ohm=z0_ohm)
    fg = freqs / 1e9
    s21_db = rc.db20(S[:, 1, 0])
    s42_db = rc.db20(S[:, 3, 1])
    path = outdir / "passive_ripple.npz"
    np.savez(
        path,
        freq_ghz=fg,
        s21_db=s21_db,
        s42_db=s42_db,
        s_real=S.real,
        s_imag=S.imag,
        ports=np.array([1, 2, 3, 4], dtype=np.int64),
    )
    print(f"passive S-matrix: {S.shape} in {time.perf_counter()-t0:.1f}s -> {path}",
          flush=True)
    return fg, s42_db, s21_db, path


def characterize_point(
    ipm_dir: Path,
    point_dir: Path,
    placement: rc.Placement,
    ic_a: float,
    ic_ladder: list[float],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Ladder the pump then sweep the gain at one +120-degrees placement."""
    fp = placement.fp_ghz
    best = rc.ladder_pump(
        ipm_dir,
        point_dir,
        fp_ghz=fp,
        ic_a=ic_a,
        ic_ladder=ic_ladder,
        pump_mode_count=args.pump_mode_count,
        nt=args.nt,
        continuation_steps=args.continuation_steps,
        timeout_s=args.pump_timeout_s,
    )
    if best is None:
        print(f"  fp={fp:.3f} GHz: no pump ratio accepted (past fold at all rungs)",
              flush=True)
        return None

    start, stop, n = args.signal_grid_ghz
    pairs = {"s21": (1, 2)}
    if args.extra_sparams:
        pairs.update({"s12": (2, 1), "s24": (4, 2)})

    sweep_paths: dict[str, str] = {}
    for name, (src, out) in pairs.items():
        csv_path = rc.gain_sweep(
            ipm_dir,
            best.pump_dir,
            point_dir / f"gain_{name}",
            fp_ghz=fp,
            source_port=src,
            out_port=out,
            signal_start_ghz=start,
            signal_stop_ghz=stop,
            points=int(n),
            sidebands=args.sidebands,
            gamma_nt=args.gamma_nt,
            timeout_s=args.gain_timeout_s,
        )
        sweep_paths[name] = str(csv_path)

    fx, gy = rc.read_gain_sweep(Path(sweep_paths["s21"]))
    peak_db = float(np.nanmax(gy)) if gy.size else float("nan")
    peak_at = float(fx[int(np.nanargmax(gy))]) if gy.size else float("nan")
    flag = "strict" if best.converged else "fold-edge"
    print(f"  fp={fp:.3f} GHz: accepted {best.ratio_ic:g} Ic "
          f"({best.current_a*1e6:.3f} uA, {flag}, flux={best.flux_over_phi0:.2g}), "
          f"peak S21 {peak_db:.2f} dB @ {peak_at:.3f} GHz", flush=True)

    return {
        "fp_ghz": fp,
        "ref_peak_ghz": placement.ref_peak_ghz,
        "period_ghz": placement.period_ghz,
        "offset_mhz": placement.offset_mhz,
        "offset_deg": placement.offset_deg,
        "ws_marker_ghz": fp - args.signal_detuning_mhz / 1e3,
        "ic_ratio": best.ratio_ic,
        "ic_current_a": best.current_a,
        "flux_over_phi0": best.flux_over_phi0,
        "coeff_rel": best.coeff_rel,
        "converged": best.converged,
        "strictly_converged": best.converged,
        "reached_full_scale": best.reached_full_scale,
        "peak_s21_db": peak_db,
        "peak_s21_at_ghz": peak_at,
        "sweep": sweep_paths,
        "pump_dir": str(best.pump_dir),
    }


def main() -> None:
    args = parse_args()
    outdir = args.outdir or (ROOT / "outputs" / f"ripple_pump_placement_{args.design}")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.ipm_dir is not None:
        ipm_dir = Path(args.ipm_dir)
    else:
        ipm_dir = outdir / "ipm_design"
        rc.build_design(args.design, ipm_dir)
    ic_a = rc.ic_reference_a(ipm_dir)
    print(f"design={args.design}  ipm_dir={ipm_dir}  Ic_median={ic_a*1e6:.4f} uA",
          flush=True)

    fg, s42_db, s21_db, passive_path = compute_passive(
        ipm_dir, outdir, tuple(args.ripple_grid_ghz), args.z0_ohm
    )

    placements = rc.auto_targets(
        fg, s42_db, tuple(args.ripple_band_ghz),
        tuple(args.map_pump_band_ghz), args.n_points,
    )
    print(f"selected {len(placements)} +120-degrees placements:", flush=True)
    for pl in placements:
        print(f"  fp={pl.fp_ghz:.3f} GHz (peak {pl.ref_peak_ghz:.3f}, "
              f"period {pl.period_ghz*1e3:.0f} MHz, {pl.offset_deg:+.0f} deg)",
              flush=True)

    ic_ladder = [float(x) for x in args.ic_ladder.split(",")]
    points: list[dict[str, Any]] = []
    for k, pl in enumerate(placements, 1):
        point_dir = outdir / f"point_{k}_fp{round(pl.fp_ghz*1000)}"
        print(f"[point {k}] placing pump at {pl.fp_ghz:.3f} GHz", flush=True)
        rec = characterize_point(ipm_dir, point_dir, pl, ic_a, ic_ladder, args)
        if rec is not None:
            rec["point"] = k
            points.append(rec)

    manifest = {
        "design": args.design,
        "ipm_dir": str(ipm_dir),
        "z0_ohm": args.z0_ohm,
        "ic_a": ic_a,
        "ripple_band_ghz": list(args.ripple_band_ghz),
        "map_pump_band_ghz": list(args.map_pump_band_ghz),
        "signal_detuning_mhz": args.signal_detuning_mhz,
        "passive_ripple_npz": str(passive_path),
        "points": points,
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path} ({len(points)} converged points)", flush=True)


if __name__ == "__main__":
    main()
