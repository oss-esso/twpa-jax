"""Resume a gain-map frequency column past its convergence wall, forcing gain.

Motivation
----------
On the high-power side of a column the pump harmonic-balance Newton solve stops
converging (above threshold / near the device fold). The normal map marks those
cells ERROR/SKIP and never computes a gain. But the last Newton *iterate* is
still a pump waveform, and it is an open question whether the wall is the real
device fold or a numerical boundary. This tool marches one column (or every
column) up in power, warm-starting each cell from the previous one, and runs the
gain solve on the last-iterate pump waveform **regardless of convergence**
(``InProcessEngine.solve_point(force_gain=True)``). It records, per cell, whether
the pump converged and what gain the (possibly non-converged) pump produced, so
the converged branch and the forced continuation past the wall can be compared.

This does NOT re-run a whole map from scratch: it rebuilds one column's warm
chain in-process (the easy low-power cells solve in ~seconds), which is what
makes the last-iterate above the wall physically meaningful (it is warm-started
from a genuine converged neighbour, not a cold guess).

Usage
-----
Pass the SAME engine/grid flags you would give ``run_gain_map.py`` (circuit,
power/freq bounds, grid size, backends, sidebands, signal params). Extra flags:

  --column-freq-ghz F   Only the grid frequency column nearest F. Omit to do all.
  --force-out PATH       Output directory (per-cell dirs + column CSV + PNG).
  --force-max-nonfinite N  Stop a column after N consecutive non-finite pump
                           states (warm chain has diverged; default 3).

Example (one column of the standard 2c band):

  python scripts/resume_column_force_gain.py \
    --circuit-dir outputs/ipm_python_design \
    --n-power 40 --n-frequency 51 \
    --pump-power-min-dbm -32 --pump-power-max-dbm -19 \
    --pump-freq-min-ghz 8.033 --pump-freq-max-ghz 8.363 \
    --inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast \
    --pump-mode-count 10 --nt 40 --sidebands 10 \
    --signal-detuning-mhz 100 --no-signal-spectrum \
    --signal-backend direct --signal-solver superlu \
    --column-freq-ghz 8.099 --force-out outputs/force_gain_col_8p099
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gain_map as rgm  # noqa: E402


def _finite_X(X: np.ndarray | None) -> bool:
    return X is not None and bool(np.all(np.isfinite(X)))


def march_column(
    engine: rgm.InProcessEngine,
    column: list[rgm.GridPoint],
    pass_dir: Path,
    *,
    max_nonfinite: int,
) -> list[dict[str, Any]]:
    """Warm-march one column ascending in power, forcing gain at every cell.

    Never skips and never gives up on a failed pump: the last-iterate X keeps
    seeding the next (higher-power) cell as long as it stays finite. ``force_gain``
    makes ``solve_point`` compute gain even when the pump did not converge.
    """
    column = sorted(column, key=lambda p: p.power_dbm)
    scale = engine.args.pump_current_jc_scale
    prev_X: np.ndarray | None = None
    nonfinite_run = 0
    rows: list[dict[str, Any]] = []
    fp = column[0].pump_freq_ghz
    for point in column:
        mode = "warm" if prev_X is not None else "seed"
        try:
            row, X = engine.solve_point(
                point, pass_dir, mode=mode, warm_X=prev_X, force_gain=True
            )
        except (ValueError, FloatingPointError, RuntimeError, MemoryError) as exc:
            row = {
                "point_index": point.index, "i_power": point.i_power,
                "j_freq": point.j_freq, "pump_power_dbm": point.power_dbm,
                "pump_freq_ghz": point.pump_freq_ghz, "pump_status": "EXC",
                "gain_status": "EXC", "status": "EXC", "gain_db": None,
                "pump_coeff_rel": None, "pump_failure_reason": repr(exc),
            }
            X = None
        pump_ok = row.get("pump_status") == "VALID_CONVERGED"
        row["pump_converged"] = pump_ok
        # A gain read off a non-converged pump waveform: the diagnostic signal.
        row["forced_gain"] = bool(
            not pump_ok and row.get("gain_status") == "VALID_SOLVED"
        )
        rows.append(row)
        print(
            f"[force {point.i_power + 1}/{len(column)}] fp={fp:.4g} GHz "
            f"P={point.power_dbm:.4g} dBm pump={row.get('pump_status')} "
            f"coeff_rel={row.get('pump_coeff_rel')} gain_status={row.get('gain_status')} "
            f"gain_db={row.get('gain_db')} forced={row['forced_gain']}",
            flush=True,
        )
        if _finite_X(X):
            prev_X = X
            nonfinite_run = 0
        else:
            nonfinite_run += 1
            if nonfinite_run >= max_nonfinite:
                print(
                    f"[force] fp={fp:.4g} GHz stopping column: "
                    f"{nonfinite_run} consecutive non-finite pump states "
                    f"(warm chain diverged) at P={point.power_dbm:.4g} dBm",
                    flush=True,
                )
                break
    return rows


_CSV_FIELDS = [
    "point_index", "i_power", "j_freq", "pump_power_dbm", "pump_freq_ghz",
    "pump_status", "pump_converged", "pump_coeff_rel", "pump_newton_total",
    "pump_branch_current_max", "pump_failure_reason", "gain_status",
    "gain_db", "gain_vs_off_db", "signal_ghz", "forced_gain", "status",
]


def write_column_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_column(rows: list[dict[str, Any]], out_png: Path, fp_ghz: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipped plot", flush=True)
        return
    P = np.array([r["pump_power_dbm"] for r in rows], dtype=float)
    G = np.array(
        [r.get("gain_db") if r.get("gain_db") is not None else np.nan for r in rows],
        dtype=float,
    )
    conv = np.array([bool(r.get("pump_converged")) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(P, G, "-", color="0.6", lw=1, zorder=1)
    ax.scatter(P[conv], G[conv], s=28, c="tab:blue", label="pump converged", zorder=2)
    ax.scatter(P[~conv], G[~conv], s=28, c="tab:red", marker="x",
               label="forced (pump not converged)", zorder=3)
    ax.set_xlabel("pump power (dBm)")
    ax.set_ylabel("gain (dB)")
    ax.set_title(f"forced-gain column resume  fp={fp_ghz:.4g} GHz")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--column-freq-ghz", type=float, default=None,
                     help="Only the grid column nearest this frequency; omit for all.")
    pre.add_argument("--force-out", type=Path, required=True,
                     help="Output directory for per-cell dirs, CSV, and PNG.")
    pre.add_argument("--force-max-nonfinite", type=int, default=3,
                     help="Stop a column after N consecutive non-finite pump states.")
    my_args, rest = pre.parse_known_args(argv)

    args = rgm.parse_args(rest)
    engine = rgm.InProcessEngine(args)
    points, powers, freqs = rgm.build_points(args)

    by_col: dict[int, list[rgm.GridPoint]] = {}
    for p in points:
        by_col.setdefault(p.j_freq, []).append(p)

    if my_args.column_freq_ghz is not None:
        j = int(np.argmin(np.abs(freqs - my_args.column_freq_ghz)))
        chosen = [j]
        print(f"[force] selected column j={j} fp={freqs[j]:.6g} GHz "
              f"(nearest to {my_args.column_freq_ghz} GHz)", flush=True)
    else:
        chosen = sorted(by_col)
        print(f"[force] running all {len(chosen)} columns", flush=True)

    pass_dir = my_args.force_out / "cells"
    all_rows: list[dict[str, Any]] = []
    for j in chosen:
        rows = march_column(
            engine, by_col[j], pass_dir, max_nonfinite=my_args.force_max_nonfinite
        )
        all_rows.extend(rows)
        fp = float(freqs[j])
        tag = f"col_j{j:03d}_fp_{fp:.4g}".replace(".", "p")
        write_column_csv(rows, my_args.force_out / f"{tag}.csv")
        plot_column(rows, my_args.force_out / f"{tag}.png", fp)

    write_column_csv(all_rows, my_args.force_out / "force_gain_all.csv")
    n_forced = sum(1 for r in all_rows if r.get("forced_gain"))
    n_conv = sum(1 for r in all_rows if r.get("pump_converged"))
    print(f"[force] done: {len(all_rows)} cells, {n_conv} pump-converged, "
          f"{n_forced} forced-gain (past-wall). wrote {my_args.force_out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
