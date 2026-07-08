"""Generate standard plots and tables from saved gain-map outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from twpa_solver.plotting.candidates import (
    compute_all_fit_metrics,
    select_candidates,
    write_candidate_tables,
)
from twpa_solver.plotting.data import load_map_data, spectrum_for_point
from twpa_solver.plotting.maps import (
    plot_gbp_fit_map,
    plot_peak_gain_fit_map,
    plot_ripple_fit_map,
    plot_selected_candidate_map,
    plot_smoothness_fit_map,
    plot_status_map,
)
from twpa_solver.plotting.metrics import compute_fit_metrics
from twpa_solver.plotting.spectrum import plot_candidate_spectrum


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
    return parser


def _candidate_filename(label: str) -> str:
    if label.startswith("rank_"):
        return f"candidate_{label}_spectrum.png"
    return f"candidate_{label}_spectrum.png"


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
    candidates = select_candidates(
        metrics_df,
        min_gain_db=args.min_gain_db,
        top_k=args.top_k,
    )
    write_candidate_tables(candidates, metrics_df, outdir)

    save_kwargs = {"save_pdf": args.save_pdf, "save_svg": args.save_svg}
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
