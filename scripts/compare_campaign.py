"""Compare continuation-method campaign runs on coverage, cost, and gain.

Reads each `<id>/map_summary.json` + `<id>/map_arrays.npz` under a campaign
directory, snapshots them (so a later `run_gain_map --overwrite` cannot destroy
the analysis inputs), prints a comparison table vs the baseline config, and
renders a shared-scale gain-heatmap grid plus a delta-vs-baseline grid.

Metrics mirror docs/reports/campaign_runs.md "How to read the results":
fold-skip coverage, convergence cost, single-point gain in the solved region.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class RunResult:
    """One campaign config's coverage, cost, and gain summary."""

    run_id: str
    n_cells: int
    n_pass: int
    n_error: int
    n_skip_fold: int
    pump_runtime_s: float
    gain_runtime_s: float
    elapsed_s: float
    gain_db: np.ndarray  # (n_power, n_frequency), NaN where unsolved
    power_dbm: np.ndarray
    freq_ghz: np.ndarray

    @property
    def coverage(self) -> float:
        return self.n_pass / self.n_cells if self.n_cells else 0.0

    @property
    def solved_gain(self) -> np.ndarray:
        return self.gain_db[np.isfinite(self.gain_db)]


def _status_counts(summary: dict) -> dict[str, int]:
    counts = summary.get("warm_status_counts") or {}
    if not counts:
        counts = summary.get("cold_status_counts") or {}
    return {str(k): int(v) for k, v in counts.items()}


def load_run(run_dir: Path) -> RunResult | None:
    """Load one config; return None if it has no assembled map."""
    summary_path = run_dir / "map_summary.json"
    npz_path = run_dir / "map_arrays.npz"
    if not summary_path.is_file() or not npz_path.is_file():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = _status_counts(summary)

    with np.load(npz_path) as data:
        gain_key = "gain_db_warm" if "gain_db_warm" in data.files else "gain_db_cold"
        gain = np.array(data[gain_key], dtype=float)
        power = np.array(data["pump_power_dbm"], dtype=float)
        freq = np.array(data["pump_frequency_ghz"], dtype=float)

    timing = summary.get("warm_timing_totals") or {}
    return RunResult(
        run_id=run_dir.name,
        n_cells=int(gain.size),
        n_pass=counts.get("PASS", 0),
        n_error=counts.get("ERROR", 0),
        n_skip_fold=counts.get("SKIP_PAST_FOLD", 0),
        pump_runtime_s=float(summary.get("warm_pump_runtime_s") or 0.0),
        gain_runtime_s=float(summary.get("warm_gain_runtime_s") or 0.0),
        elapsed_s=float(timing.get("elapsed_s") or summary.get("elapsed_s") or 0.0),
        gain_db=gain,
        power_dbm=power,
        freq_ghz=freq,
    )


def snapshot(run_dir: Path, dest_root: Path) -> None:
    """Copy the analysis inputs so a later --overwrite cannot destroy them."""
    dest = dest_root / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("map_summary.json", "map_arrays.npz", "map_points.csv"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)


def print_table(runs: list[RunResult], baseline: str) -> str:
    header = (
        f"{'id':<24} {'PASS':>5} {'ERR':>5} {'FOLD':>5} {'cov%':>6} "
        f"{'gmax':>6} {'gp50':>6} {'gp90':>6} {'dcov%':>6} "
        f"{'pump_s':>8} {'total_s':>8}"
    )
    base = next((r for r in runs if r.run_id == baseline), None)
    base_cov = base.coverage if base else 0.0

    lines = [header, "-" * len(header)]
    for r in sorted(runs, key=lambda x: x.coverage, reverse=True):
        g = r.solved_gain
        gmax = float(np.max(g)) if g.size else float("nan")
        gp50 = float(np.percentile(g, 50)) if g.size else float("nan")
        gp90 = float(np.percentile(g, 90)) if g.size else float("nan")
        dcov = (r.coverage - base_cov) * 100.0
        mark = "*" if r.run_id == baseline else " "
        lines.append(
            f"{mark}{r.run_id:<23} {r.n_pass:>5} {r.n_error:>5} {r.n_skip_fold:>5} "
            f"{r.coverage * 100:>6.1f} {gmax:>6.2f} {gp50:>6.2f} {gp90:>6.2f} "
            f"{dcov:>+6.1f} {r.pump_runtime_s:>8.0f} {r.elapsed_s:>8.0f}"
        )
    return "\n".join(lines)


def _grid_shape(n: int) -> tuple[int, int]:
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def plot_gain_grid(runs: list[RunResult], out_path: Path) -> None:
    vals = np.concatenate([r.solved_gain for r in runs if r.solved_gain.size])
    vmin, vmax = np.percentile(vals, [2, 98])
    nrows, ncols = _grid_shape(len(runs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    im = None
    for ax, r in zip(axes.flat, runs):
        extent = [r.freq_ghz[0], r.freq_ghz[-1], r.power_dbm[0], r.power_dbm[-1]]
        im = ax.imshow(r.gain_db, origin="lower", aspect="auto", extent=extent,
                       vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"{r.run_id}\ncov {r.coverage * 100:.0f}%", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    fig.supxlabel("pump frequency (GHz)")
    fig.supylabel("pump power (dBm)")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="gain (dB)", shrink=0.8)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_delta_grid(runs: list[RunResult], baseline: str, out_path: Path) -> None:
    base = next((r for r in runs if r.run_id == baseline), None)
    if base is None:
        return
    others = [r for r in runs if r.run_id != baseline
              and r.gain_db.shape == base.gain_db.shape]
    if not others:
        return
    nrows, ncols = _grid_shape(len(others))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    im = None
    for ax, r in zip(axes.flat, others):
        # Extra coverage vs baseline: cells this config solved that baseline did not.
        delta = r.gain_db - base.gain_db
        extent = [r.freq_ghz[0], r.freq_ghz[-1], r.power_dbm[0], r.power_dbm[-1]]
        im = ax.imshow(delta, origin="lower", aspect="auto", extent=extent,
                       vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        gained = int(np.sum(np.isfinite(r.gain_db) & ~np.isfinite(base.gain_db)))
        ax.set_title(f"{r.run_id}\n+{gained} solved cells", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(others):]:
        ax.axis("off")
    fig.supxlabel("pump frequency (GHz)")
    fig.supylabel("pump power (dBm)")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(),
                     label=f"gain − {baseline} (dB)", shrink=0.8)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path,
                        default=Path("outputs/campaign_continuation_methods"))
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Default: <campaign-dir>/comparison")
    parser.add_argument("--baseline", default="c04_baseline_prod")
    parser.add_argument("--only", default=None,
                        help="Comma-separated ids to include")
    args = parser.parse_args()

    campaign = args.campaign_dir
    outdir = args.outdir or campaign / "comparison"
    snap_root = outdir / "snapshots"
    outdir.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    runs: list[RunResult] = []
    for run_dir in sorted(p for p in campaign.iterdir() if p.is_dir()):
        if run_dir.name in {"comparison"}:
            continue
        if only is not None and run_dir.name not in only:
            continue
        result = load_run(run_dir)
        if result is None:
            continue
        snapshot(run_dir, snap_root)
        runs.append(result)

    if not runs:
        print("No assembled maps found.")
        return

    table = print_table(runs, args.baseline)
    print(table)
    (outdir / "comparison_table.txt").write_text(table + "\n", encoding="utf-8")

    plot_gain_grid(runs, outdir / "gain_grid.png")
    plot_delta_grid(runs, args.baseline, outdir / "delta_grid.png")
    print(f"\nWrote {outdir / 'gain_grid.png'}, {outdir / 'delta_grid.png'}, "
          f"snapshots -> {snap_root}")


if __name__ == "__main__":
    main()
