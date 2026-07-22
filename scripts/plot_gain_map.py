"""Generate standard plots and tables from saved gain-map outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from twpa_solver.plotting.candidates import (
    compute_all_fit_metrics,
    gain_ranked_candidates,
    select_candidates,
    write_candidate_tables,
)
from twpa_solver.plotting.data import MapData, load_map_data, spectrum_for_point
from twpa_solver.plotting.maps import (
    plot_gbp_fit_map,
    plot_peak_gain_fit_map,
    plot_ripple_fit_map,
    plot_runtime_map,
    plot_selected_candidate_map,
    plot_simple_gain_map,
    plot_smoothness_fit_map,
    plot_status_map,
)
from twpa_solver.plotting.metrics import compute_fit_metrics, minus3db_band
from twpa_solver.plotting.spectrum import (
    plot_candidate_s21_bandwidth,
    plot_candidate_spectrum,
)
from twpa_solver.plotting.style import apply_thesis_style

ROOT = Path(__file__).resolve().parents[1]
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"


@dataclass
class PlotConfig:
    operation_drop_db: float = 3.0
    window_frac: float = 0.05
    polyorder: int = 3
    n_dense: int = 2000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--min-gain-db", type=float, default=10.0)
    parser.add_argument("--operation-drop-db", type=float, default=3.0)
    parser.add_argument("--window-frac", type=float, default=0.05)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--n-dense", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--save-svg", action="store_true")
    # No-spectrum maps only: auto-pick the top-gain cells and run a real S21
    # sweep for each so they can be fitted. Needs the circuit dir; if --ipm-dir
    # is omitted it is inferred from map_summary.json or the run-dir name.
    parser.add_argument("--ipm-dir", type=Path, default=None,
                        help="Circuit dir; enables candidate S21 sweeps for a "
                             "--no-signal-spectrum map. Inferred if omitted.")
    # Sweep is centered on the pump frequency fp with a fixed +-half-span so every
    # candidate shares the same window width regardless of fp (default fp +-3 GHz).
    parser.add_argument("--sweep-half-span-ghz", type=float, default=3.0,
                        help="Half-width of the S21 sweep around fp (GHz); "
                             "ignored if --sweep-start-ghz/--sweep-stop-ghz are set.")
    parser.add_argument("--sweep-start-ghz", type=float, default=None,
                        help="Absolute S21 sweep start (GHz); overrides the "
                             "fp-centred window when given with --sweep-stop-ghz.")
    parser.add_argument("--sweep-stop-ghz", type=float, default=None,
                        help="Absolute S21 sweep stop (GHz); overrides the "
                             "fp-centred window when given with --sweep-start-ghz.")
    parser.add_argument("--sweep-points", type=int, default=501,
                        help="Sweep points (default 1251 over +-3 GHz ~ 5 MHz).")
    parser.add_argument("--sweep-smooth-frac", type=float, default=0.35,
                        help="Savitzky-Golay window as a fraction of the sweep "
                             "length for the -3 dB band/GBP fit; large by default "
                             "so the band tracks the broadband envelope (0 = raw).")
    parser.add_argument("--sweep-bridge-ghz", type=float, default=0.3,
                        help="Bridge a sub-threshold gap up to this width (GHz) -- "
                             "the degenerate notch at fs~fp -- to also report a "
                             "band/GBP spanning both gain lobes (0 = off).")
    parser.add_argument("--source-port", type=int, default=1)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--sidebands", type=int, default=6)
    parser.add_argument("--gamma-nt", type=int, default=96)
    parser.add_argument("--overwrite-sweeps", action="store_true",
                        help="Re-run candidate sweeps even if gain_sweep.csv exists.")
    return parser


def infer_ipm_dir(run_dir: Path) -> Path | None:
    """Find the circuit dir for a no-spectrum run's candidate sweeps.

    Prefers a path recorded in map_summary.json; otherwise guesses from the
    run-dir name ('7c' or '3c' -> the matching design, else the default 2c
    design).
    """
    summary = run_dir / "map_summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        for key in ("circuit_dir", "ipm_dir"):
            value = data.get(key)
            if value and Path(value).exists():
                return Path(value)
    name = run_dir.name.lower()
    if "7c" in name:
        design_dir = "ipm_python_design_7c"
    elif "3c" in name:
        design_dir = "ipm_python_design_3c"
    else:
        design_dir = "ipm_python_design"
    guess = ROOT / "outputs" / design_dir
    return guess if guess.exists() else None


def _candidate_filename(label: str) -> str:
    if label.startswith("rank_"):
        return f"candidate_{label}_spectrum.png"
    return f"candidate_{label}_spectrum.png"


def _gain_candidates(points: pd.DataFrame, *, top_k: int, min_gain_db: float) -> pd.DataFrame:
    """Top-K PASS cells by trailing gain (no spectrum fit needed).

    Thin wrapper over the shared ``gain_ranked_candidates`` so the S21 re-sweep
    and ``scripts/prune_map_solutions.py`` keep exactly the same cells.
    """
    return gain_ranked_candidates(points, top_k=top_k, min_gain_db=min_gain_db)


def _run_candidate_sweep(
    row: pd.Series, args: argparse.Namespace, sweep_dir: Path, *, center_ghz: float,
) -> Path | None:
    """Run an S21 sweep centered on ``center_ghz`` (fp) via exp09; return its CSV."""
    fp = float(row["pump_freq_ghz"])
    pump_dir = Path(str(row["pump_dir"]))
    if not (pump_dir / "pump_solution.npz").exists():
        print(f"  skip point {int(row['point_index'])}: missing pump solution {pump_dir}")
        return None
    csv_out = sweep_dir / "gain_sweep.csv"
    if csv_out.exists() and not args.overwrite_sweeps:
        return csv_out
    sweep_dir.mkdir(parents=True, exist_ok=True)
    if args.sweep_start_ghz is not None and args.sweep_stop_ghz is not None:
        start_ghz, stop_ghz = float(args.sweep_start_ghz), float(args.sweep_stop_ghz)
    else:
        half = float(args.sweep_half_span_ghz)
        start_ghz, stop_ghz = center_ghz - half, center_ghz + half
    cmd = [
        sys.executable, str(EXP09),
        "--ipm-dir", str(args.ipm_dir),
        "--pump-dir", str(pump_dir),
        "--fallback-pump-freq-ghz", f"{fp:.12g}",
        "--sweep",
        "--signal-start-ghz", f"{start_ghz:.12g}",
        "--signal-stop-ghz", f"{stop_ghz:.12g}",
        "--points", str(args.sweep_points),
        "--sidebands", str(args.sidebands),
        "--gamma-nt", str(args.gamma_nt),
        "--source-port", str(args.source_port),
        "--out-port", str(args.out_port),
        "--outdir", str(sweep_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0 or not csv_out.exists():
        print(f"  sweep failed for point {int(row['point_index'])} (rc={proc.returncode})")
        return None
    return csv_out


def _load_sweep(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freqs: list[float] = []
    gains: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("status") == "VALID_SOLVED":
                freqs.append(float(record["signal_ghz"]))
                gains.append(float(record["gain_db"]))
    return np.asarray(freqs, dtype=float), np.asarray(gains, dtype=float)


def fit_gain_candidates(
    data: MapData,
    args: argparse.Namespace,
    config: PlotConfig,
    outdir: Path,
    candidates_dir: Path,
) -> None:
    """No-spectrum path: auto-pick top-gain cells, sweep S21, plot -3 dB band.

    For each candidate it sweeps S21 over fp +- half-span (same window for every
    candidate), measures the -3 dB bandwidth around the peak of the heavily
    smoothed curve, plots the annotated S21, prints the pump condition + GBP, and
    writes a summary table.
    """
    top = _gain_candidates(data.points, top_k=args.top_k, min_gain_db=args.min_gain_db)
    if top.empty:
        print("no PASS cells with finite gain -> no candidates to sweep")
        return
    if args.sweep_start_ghz is not None and args.sweep_stop_ghz is not None:
        window = f"{args.sweep_start_ghz:g}-{args.sweep_stop_ghz:g} GHz"
    else:
        window = f"fp +-{args.sweep_half_span_ghz:g} GHz"
    print(f"auto-picked {len(top)} candidate(s); S21 sweep {args.sweep_points} pts "
          f"over {window} via exp09")
    save_kwargs = {"save_pdf": args.save_pdf, "save_svg": args.save_svg}
    sweeps_root = outdir / "candidate_sweeps"
    rows_out: list[dict] = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        label = f"rank_{rank:03d}"
        point_index = int(row["point_index"])
        fp = float(row["pump_freq_ghz"])
        pp = float(row["pump_power_dbm"])
        map_gain = float(row["gain_db_num"])
        sweep_dir = sweeps_root / f"{label}_point_{point_index:04d}"
        csv_out = _run_candidate_sweep(row, args, sweep_dir, center_ghz=fp)
        if csv_out is None:
            continue
        freq, gain = _load_sweep(csv_out)
        if freq.size < 2:
            print(f"  skip {label}: <2 solved sweep points")
            continue
        band = minus3db_band(
            freq, gain, drop_db=config.operation_drop_db,
            smooth_window_frac=(args.sweep_smooth_frac or None),
            polyorder=config.polyorder, bridge_ghz=args.sweep_bridge_ghz,
        )
        meta = {
            "point_index": point_index, "pump_power_dbm": pp,
            "pump_freq_ghz": fp, "map_gain_db": map_gain,
        }
        plot_candidate_s21_bandwidth(
            freq, gain, meta, band,
            candidates_dir / f"candidate_{label}_s21.png",
            title=f"Signal Gain S21",
            drop_db=config.operation_drop_db, **save_kwargs,
        )
        if band is None:
            print(f"  {label}: Pp={pp:.3f} dBm fp={fp:.4f} GHz | band undefined")
            continue
        clip = " [clipped-widen sweep]" if band.get("band_clipped") else ""
        needle = ""
        if band.get("window_max_db", -np.inf) > band["peak_gain_db"] + config.operation_drop_db:
            needle = (f"  (window peak {band['window_max_db']:.1f} dB @ "
                      f"{band['window_max_freq_ghz']:.4f} GHz -- near-fold needle, off-band)")
        bridged = ""
        if band.get("bridged"):
            bclip = " [clipped]" if band.get("bridged_band_clipped") else ""
            bridged = (f"  || bridged(notch): BW={band['bridged_bandwidth_ghz'] * 1e3:.1f} MHz "
                       f"GBP={band['bridged_gbp_ghz']:.3g} GHz{bclip}")
        print(f"  {label}: Pp={pp:.3f} dBm  fp={fp:.4f} GHz "
              f"| Gpk={band['peak_gain_db']:.2f} dB @ {band['peak_freq_ghz']:.4f} GHz "
              f"| BW(-{config.operation_drop_db:g}dB)={band['bandwidth_ghz'] * 1e3:.1f} MHz "
              f"| GBP={band['gbp_ghz']:.3g} GHz{clip}{needle}{bridged}")
        band_scalar = {k: v for k, v in band.items() if not k.startswith("_")}
        rows_out.append({"candidate": label, "point_index": point_index,
                         "pump_power_dbm": pp, "pump_freq_ghz": fp,
                         "map_gain_db": map_gain, **band_scalar})
    if rows_out:
        tables_dir = outdir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        out_csv = tables_dir / "candidate_s21_bandwidth.csv"
        pd.DataFrame(rows_out).to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_thesis_style()
    outdir = args.outdir or (args.run_dir / "plots")
    maps_dir = outdir / "maps"
    candidates_dir = outdir / "candidates"
    maps_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    config = PlotConfig(
        operation_drop_db=args.operation_drop_db,
        window_frac=args.window_frac,
        polyorder=args.polyorder,
        n_dense=args.n_dense,
    )
    data = load_map_data(args.run_dir)
    metrics_df = compute_all_fit_metrics(data, config)
    save_kwargs = {"save_pdf": args.save_pdf, "save_svg": args.save_svg}

    # A map run with --no-signal-spectrum has no per-cell signal curve, so none
    # of the fit-based metrics (peak/GBP/ripple/smoothness) exist -- every row
    # comes back valid_fit=False. Fall back to the grids that only need the
    # trailing gain + point table, straight from data.points.
    has_spectrum_fit = (
        not metrics_df.empty
        and "valid_fit" in metrics_df.columns
        and bool(metrics_df["valid_fit"].any())
    )
    if not has_spectrum_fit:
        print("no spectrum data -> plotting simple gain, status and runtime maps")
        plot_simple_gain_map(data.points, maps_dir / "gain_map.png", **save_kwargs)
        plot_status_map(data.points, maps_dir / "status_map.png", **save_kwargs)
        plot_runtime_map(data.points, maps_dir / "runtime_map.png", **save_kwargs)
        if args.ipm_dir is None:
            args.ipm_dir = infer_ipm_dir(args.run_dir)
        if args.ipm_dir is not None and Path(args.ipm_dir).exists():
            print(f"circuit dir: {args.ipm_dir}")
            fit_gain_candidates(data, args, config, outdir, candidates_dir)
        else:
            print("no --ipm-dir given and none could be inferred; "
                  "skipping candidate S21 sweeps")
        print(f"Wrote plots to {outdir}")
        return 0

    candidates = select_candidates(
        metrics_df,
        min_gain_db=args.min_gain_db,
        top_k=args.top_k,
    )
    write_candidate_tables(candidates, metrics_df, outdir)

    plot_peak_gain_fit_map(
        metrics_df,
        maps_dir / "peak_gain_fit_map.png",
        candidates=candidates,
        **save_kwargs,
    )
    plot_gbp_fit_map(
        metrics_df,
        maps_dir / "gbp_fit_map.png",
        candidates=candidates,
        **save_kwargs,
    )
    plot_ripple_fit_map(
        metrics_df,
        maps_dir / "ripple_fit_map.png",
        candidates=candidates,
        **save_kwargs,
    )
    plot_smoothness_fit_map(
        metrics_df,
        maps_dir / "smoothness_fit_map.png",
        candidates=candidates,
        **save_kwargs,
    )
    plot_status_map(metrics_df, maps_dir / "status_map.png", **save_kwargs)
    plot_runtime_map(data.points, maps_dir / "runtime_map.png", **save_kwargs)
    plot_selected_candidate_map(
        metrics_df,
        candidates,
        maps_dir / "selected_candidate_map.png",
        **save_kwargs,
    )

    for label, row in candidates.items():
        freq, gain = spectrum_for_point(data, int(row["point_index"]))
        fit = compute_fit_metrics(
            freq,
            gain,
            row.to_dict(),
            drop_db=args.operation_drop_db,
            n_dense=args.n_dense,
            window_frac=args.window_frac,
            polyorder=args.polyorder,
        )
        plot_candidate_spectrum(
            fit,
            candidates_dir / _candidate_filename(label),
            title=label,
            **save_kwargs,
        )

    # Also emit the same fp+-3 GHz candidate S21 plots as the no-spectrum path,
    # so a spectrum map and a no-spectrum map yield identical candidate figures.
    # The stored per-cell spectrum is coarse, so re-sweep S21 at full resolution.
    if args.ipm_dir is None:
        args.ipm_dir = infer_ipm_dir(args.run_dir)
    if args.ipm_dir is not None and Path(args.ipm_dir).exists():
        print(f"circuit dir: {args.ipm_dir}")
        fit_gain_candidates(data, args, config, outdir, candidates_dir)
    else:
        print("no --ipm-dir given and none could be inferred; "
              "skipping candidate S21 sweeps")
    print(f"Wrote plots to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
