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
    # Sweep is centered on the cell's own signal ws = fp - detuning (where the
    # map's gain lives) and defaults to a fine grid so near-fold needle peaks
    # are resolved for the -3 dB bandwidth measurement.
    parser.add_argument("--sweep-half-span-ghz", type=float, default=0.5,
                        help="Half-width of the S21 sweep around ws (GHz).")
    parser.add_argument("--sweep-points", type=int, default=501,
                        help="Sweep points (default 501 over +-0.5 GHz = 2 MHz).")
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
    run-dir name ('3c' -> the 3c design, else the default 2c design).
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
    guess = ROOT / "outputs" / ("ipm_python_design_3c" if "3c" in name else "ipm_python_design")
    return guess if guess.exists() else None


def _candidate_filename(label: str) -> str:
    if label.startswith("rank_"):
        return f"candidate_{label}_spectrum.png"
    return f"candidate_{label}_spectrum.png"


def _gain_candidates(points: pd.DataFrame, *, top_k: int, min_gain_db: float) -> pd.DataFrame:
    """Top-K PASS cells by trailing gain (no spectrum fit needed)."""
    df = points.copy()
    df["gain_db_num"] = pd.to_numeric(df.get("gain_db"), errors="coerce")
    passed = df["status"].map(lambda s: str(s).upper().startswith("PASS"))
    ranked = df[passed & np.isfinite(df["gain_db_num"])].sort_values(
        "gain_db_num", ascending=False
    )
    strong = ranked[ranked["gain_db_num"] >= float(min_gain_db)]
    chosen = strong if not strong.empty else ranked
    return chosen.head(int(top_k))


def _signal_center_ghz(row: pd.Series) -> float:
    """Sweep center = the cell's own trailing signal ws (where the gain lives)."""
    s = pd.to_numeric(pd.Series([row.get("signal_ghz")]), errors="coerce").iloc[0]
    if np.isfinite(s):
        return float(s)
    return float(row["pump_freq_ghz"]) - 0.1


def _run_candidate_sweep(
    row: pd.Series, args: argparse.Namespace, sweep_dir: Path, *, center_ghz: float,
) -> Path | None:
    """Run an S21 sweep centered on ws for one cell via exp09; return its CSV."""
    fp = float(row["pump_freq_ghz"])
    pump_dir = Path(str(row["pump_dir"]))
    if not (pump_dir / "pump_solution.npz").exists():
        print(f"  skip point {int(row['point_index'])}: missing pump solution {pump_dir}")
        return None
    csv_out = sweep_dir / "gain_sweep.csv"
    if csv_out.exists() and not args.overwrite_sweeps:
        return csv_out
    sweep_dir.mkdir(parents=True, exist_ok=True)
    half = float(args.sweep_half_span_ghz)
    cmd = [
        sys.executable, str(EXP09),
        "--ipm-dir", str(args.ipm_dir),
        "--pump-dir", str(pump_dir),
        "--fallback-pump-freq-ghz", f"{fp:.12g}",
        "--sweep",
        "--signal-start-ghz", f"{center_ghz - half:.12g}",
        "--signal-stop-ghz", f"{center_ghz + half:.12g}",
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

    For each candidate it centers a fine S21 sweep on the cell's own signal ws,
    measures the -3 dB bandwidth straight from the raw sweep (so near-fold needle
    peaks survive), plots the annotated S21, prints the pump condition + GBP, and
    writes a summary table.
    """
    top = _gain_candidates(data.points, top_k=args.top_k, min_gain_db=args.min_gain_db)
    if top.empty:
        print("no PASS cells with finite gain -> no candidates to sweep")
        return
    print(f"auto-picked {len(top)} candidate(s); S21 sweep {args.sweep_points} pts "
          f"over ws +-{args.sweep_half_span_ghz} GHz via exp09")
    save_kwargs = {"save_pdf": args.save_pdf, "save_svg": args.save_svg}
    sweeps_root = outdir / "candidate_sweeps"
    rows_out: list[dict] = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        label = f"rank_{rank:03d}"
        point_index = int(row["point_index"])
        fp = float(row["pump_freq_ghz"])
        pp = float(row["pump_power_dbm"])
        map_gain = float(row["gain_db_num"])
        ws = _signal_center_ghz(row)
        sweep_dir = sweeps_root / f"{label}_point_{point_index:04d}"
        csv_out = _run_candidate_sweep(row, args, sweep_dir, center_ghz=ws)
        if csv_out is None:
            continue
        freq, gain = _load_sweep(csv_out)
        if freq.size < 2:
            print(f"  skip {label}: <2 solved sweep points")
            continue
        band = minus3db_band(freq, gain, drop_db=config.operation_drop_db)
        meta = {
            "point_index": point_index, "pump_power_dbm": pp,
            "pump_freq_ghz": fp, "map_gain_db": map_gain,
        }
        plot_candidate_s21_bandwidth(
            freq, gain, meta, band,
            candidates_dir / f"candidate_{label}_s21.png",
            title=f"{label} (point {point_index})",
            drop_db=config.operation_drop_db, **save_kwargs,
        )
        if band is None:
            print(f"  {label}: Pp={pp:.3f} dBm fp={fp:.4f} GHz | band undefined")
            continue
        clip = " [clipped-widen sweep]" if band.get("band_clipped") else ""
        print(f"  {label}: Pp={pp:.3f} dBm  fp={fp:.4f} GHz  map_gain@ws={map_gain:.2f} dB "
              f"| Gmax={band['peak_gain_db']:.2f} dB @ {band['peak_freq_ghz']:.4f} GHz "
              f"| BW(-{config.operation_drop_db:g}dB)={band['bandwidth_ghz'] * 1e3:.1f} MHz "
              f"| GBP={band['gbp_ghz']:.3g} GHz{clip}")
        rows_out.append({"candidate": label, "point_index": point_index,
                         "pump_power_dbm": pp, "pump_freq_ghz": fp,
                         "map_gain_db": map_gain, **band})
    if rows_out:
        tables_dir = outdir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        out_csv = tables_dir / "candidate_s21_bandwidth.csv"
        pd.DataFrame(rows_out).to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")


def main() -> int:
    args = build_parser().parse_args()
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
    print(f"Wrote plots to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
