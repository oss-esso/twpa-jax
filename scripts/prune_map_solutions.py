"""Prune a finished gain-map's per-point ``pump_solution.npz`` down to the cells
the plotter would re-sweep.

A 100x100 map stores ~10k pump solutions (~0.7 GB after the float32 change, more
if legacy float64). Only the top-K PASS cells by trailing gain are ever re-swept
for candidate S21 plots (``plot_gain_map.fit_gain_candidates`` ->
``gain_ranked_candidates``); the gain grid and spectrum cube live in
``map_arrays.npz`` / ``map_spectrum.npz`` and never touch the per-point files.

This keeps the pump solutions for the top-K cells (same selection the plotter
uses) and deletes the rest. Run it once you are done plotting a map.

Because candidates are re-picked at plot time, keep a generous margin above the
plotter's ``--top-k`` (default here is 50) so a later re-plot with a higher
``--top-k`` or lower ``--min-gain-db`` still finds its solutions.

Dry-run by default; pass ``--apply`` to delete. Refuses to delete everything
when the keep-set is empty unless ``--allow-empty-keep``.

    python scripts/prune_map_solutions.py outputs/<map>              # preview
    python scripts/prune_map_solutions.py outputs/<map> --apply      # delete
    python scripts/prune_map_solutions.py outputs/<map> --purge-point-dirs --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from twpa_solver.plotting.candidates import gain_ranked_candidates
from twpa_solver.plotting.data import load_map_data


def _subpath_under_map(stored_pump_dir: str, map_name: str) -> list[str] | None:
    """Path components below the map folder, from a recorded ``pump_dir``.

    ``pump_dir`` is stored with the full path (repo-relative or absolute, either
    slash style), e.g. ``outputs\\<map>\\chunks\\...\\point_0005_...\\pump``.
    Return the parts after the last occurrence of ``<map>`` -- these locate the
    solution under any copy of the map folder, independent of CWD or where the
    folder was moved. Matching on this (not a parsed point index) is the only
    correct key for chunked maps, where map_points.csv globalizes point_index
    but the on-disk point dirs use per-chunk local indices.
    """
    parts = [p for p in re.split(r"[\\/]", str(stored_pump_dir)) if p]
    if map_name not in parts:
        return None
    last = len(parts) - 1 - parts[::-1].index(map_name)
    return parts[last + 1:]


def keep_solution_paths(map_dir: Path, *, top_k: int, min_gain_db: float) -> set[Path]:
    """Resolved ``pump_solution.npz`` paths for the plotter's top-K candidates.

    Keyed off each candidate row's recorded ``pump_dir`` -- the same path the
    plotter's S21 re-sweep loads -- so the keep-set is exactly what a re-plot
    needs.
    """
    data = load_map_data(map_dir)
    chosen = gain_ranked_candidates(data.points, top_k=top_k, min_gain_db=min_gain_db)
    map_name = map_dir.resolve().name
    keep: set[Path] = set()
    for stored in chosen.get("pump_dir", []):
        sub = _subpath_under_map(str(stored), map_name)
        if sub is None:
            continue
        keep.add((map_dir.joinpath(*sub, "pump_solution.npz")).resolve())
    return keep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("map_dir", type=Path, help="finished map output folder (has map_points.csv)")
    p.add_argument("--top-k", type=int, default=50,
                   help="keep this many top-gain cells (default 50; margin over the "
                        "plotter's top-k so re-plots still have their solutions)")
    p.add_argument("--min-gain-db", type=float, default=10.0,
                   help="prefer cells at/above this gain, matching the plotter")
    p.add_argument("--purge-point-dirs", action="store_true",
                   help="delete the whole point dir (reports, gain, logs) for non-kept "
                        "cells, not just pump_solution.npz")
    p.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    p.add_argument("--allow-empty-keep", action="store_true",
                   help="proceed even if no cells qualify to keep (would prune all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    map_dir = args.map_dir
    if not (map_dir / "map_points.csv").exists():
        raise SystemExit(f"not a map folder (no map_points.csv): {map_dir}")

    keep = keep_solution_paths(map_dir, top_k=args.top_k, min_gain_db=args.min_gain_db)
    if not keep and not args.allow_empty_keep:
        raise SystemExit(
            "keep-set is empty (no PASS cells with finite gain); refusing to prune "
            "everything. Pass --allow-empty-keep to override."
        )

    solutions = sorted(p.resolve() for p in map_dir.rglob("pump_solution.npz"))
    on_disk = set(solutions)
    # Safety guard: if the keep-set resolves to nothing on disk while candidates
    # exist, our path matching is wrong -- pruning would delete everything.
    # (This is exactly the chunked global-vs-local index trap.) Abort.
    matched_on_disk = keep & on_disk
    if keep and not matched_on_disk and not args.allow_empty_keep:
        raise SystemExit(
            f"{len(keep)} keep-path(s) but NONE match an on-disk solution under "
            f"{map_dir}; refusing to prune (matching is likely wrong). "
            "Pass --allow-empty-keep only if you are certain."
        )
    print(f"keeping {len(matched_on_disk)}/{len(keep)} candidate solution(s) on disk")

    kept_bytes = 0
    freed_bytes = 0
    freed_count = 0
    for sol in solutions:
        if sol in keep:
            kept_bytes += sol.stat().st_size
            continue
        # Non-kept cell: measure then (optionally) delete.
        if args.purge_point_dirs:
            target = sol.parent.parent  # .../point_NNNN_...
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        else:
            target = sol
            size = sol.stat().st_size
        freed_bytes += size
        freed_count += 1
        if args.apply:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

    verb = "freed" if args.apply else "would free"
    print(f"\n{len(solutions)} pump solutions found")
    print(f"kept:  {len(solutions) - freed_count} cell(s), {kept_bytes / 1e9:.2f} GB")
    print(f"{verb}: {freed_count} cell(s), {freed_bytes / 1e9:.2f} GB")
    if not args.apply and freed_count:
        print("\ndry-run only. re-run with --apply to delete.")


if __name__ == "__main__":
    main()
