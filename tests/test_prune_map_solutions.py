from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twpa_solver.plotting.candidates import gain_ranked_candidates

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prune_map_solutions.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prune_map_solutions", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prune_map_solutions"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_chunked_map(root: Path) -> None:
    """Two chunks whose LOCAL point indices collide (both have point_0000..0002),
    while map_points.csv carries GLOBAL indices 0..5 -- the real layout that
    broke index-based matching. pump_dir is the reliable key.
    """
    # global index -> (chunk, local index, gain)
    layout = {
        0: ("chunk_000_cols_000_000", 0, 5.0),
        1: ("chunk_000_cols_000_000", 1, 30.0),   # top
        2: ("chunk_000_cols_000_000", 2, 8.0),
        3: ("chunk_001_cols_001_001", 0, 12.0),
        4: ("chunk_001_cols_001_001", 1, 40.0),   # top
        5: ("chunk_001_cols_001_001", 2, 9.0),
    }
    rows = []
    for gidx, (chunk, local, gain) in layout.items():
        name = f"point_{local:04d}_p_m30dbm_fp_7p5ghz"
        pump = root / "chunks" / chunk / "warm" / "points" / name / "pump"
        pump.mkdir(parents=True)
        np.savez(pump / "pump_solution.npz", X_real=np.zeros((2, 4)), X_imag=np.zeros((2, 4)))
        # pump_dir recorded repo-relative with backslashes, as run_gain_map writes it.
        stored = f"{root.name}\\chunks\\{chunk}\\warm\\points\\{name}\\pump"
        rows.append({
            "point_index": gidx, "status": "PASS", "gain_db": gain,
            "pump_freq_ghz": 7.5, "pump_power_dbm": -30, "pump_dir": stored,
        })
    with open(root / "map_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(root / "map_arrays.npz", pump_power_dbm=np.array([-30.0]),
             pump_frequency_ghz=np.array([7.5]), gain_db_warm=np.zeros((1, 1)))


def _survivor_keys(root: Path) -> set[tuple[str, str]]:
    """(chunk, point-dir name) for each surviving solution -- unique across chunks."""
    out = set()
    for s in root.rglob("pump_solution.npz"):
        parts = s.parts
        chunk = parts[parts.index("chunks") + 1]
        point = s.parent.parent.name
        out.add((chunk, point))
    return out


def test_gain_ranked_prefers_strong_then_falls_back() -> None:
    points = pd.DataFrame({
        "point_index": [0, 1, 2, 3],
        "status": ["PASS", "PASS", "ERROR", "PASS"],
        "gain_db": [40.0, 5.0, 99.0, 12.0],
    })
    chosen = gain_ranked_candidates(points, top_k=2, min_gain_db=10.0)
    assert chosen["point_index"].tolist() == [0, 3]
    fallback = gain_ranked_candidates(points, top_k=1, min_gain_db=100.0)
    assert fallback["point_index"].tolist() == [0]


def test_subpath_under_map_slices_after_map_name() -> None:
    mod = _load_script()
    stored = r"outputs\mymap\chunks\chunk_000\warm\points\point_0001_x\pump"
    assert mod._subpath_under_map(stored, "mymap") == [
        "chunks", "chunk_000", "warm", "points", "point_0001_x", "pump"
    ]
    assert mod._subpath_under_map(stored, "not_here") is None


def test_keep_paths_disambiguate_colliding_local_indices(tmp_path: Path) -> None:
    mod = _load_script()
    _build_chunked_map(tmp_path)
    keep = mod.keep_solution_paths(tmp_path, top_k=2, min_gain_db=10.0)
    # top-2 by gain = global 4 (chunk_001/point_0001) and 1 (chunk_000/point_0001).
    # Both dirs are named point_0001 -> index matching would be ambiguous; path
    # matching must pick exactly one per chunk.
    keys = {(p.parts[p.parts.index("chunks") + 1], p.parent.parent.name) for p in keep}
    assert keys == {
        ("chunk_000_cols_000_000", "point_0001_p_m30dbm_fp_7p5ghz"),
        ("chunk_001_cols_001_001", "point_0001_p_m30dbm_fp_7p5ghz"),
    }
    # every keep path actually exists on disk (the guard depends on this)
    assert all(p.exists() for p in keep)


def test_apply_deletes_only_non_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_script()
    _build_chunked_map(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["prune", str(tmp_path), "--top-k", "2", "--min-gain-db", "10", "--apply"])
    mod.main()
    assert _survivor_keys(tmp_path) == {
        ("chunk_000_cols_000_000", "point_0001_p_m30dbm_fp_7p5ghz"),
        ("chunk_001_cols_001_001", "point_0001_p_m30dbm_fp_7p5ghz"),
    }


def test_guard_aborts_when_keep_matches_nothing_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_script()
    _build_chunked_map(tmp_path)
    # Corrupt every pump_dir so it points nowhere real -> keep-set resolves to
    # paths absent on disk. Pruning must abort, not delete everything.
    df = pd.read_csv(tmp_path / "map_points.csv")
    df["pump_dir"] = df["pump_dir"].str.replace("chunks", "GHOST", regex=False)
    df.to_csv(tmp_path / "map_points.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["prune", str(tmp_path), "--top-k", "2", "--apply"])
    with pytest.raises(SystemExit):
        mod.main()
    # nothing deleted
    assert len(list(tmp_path.rglob("pump_solution.npz"))) == 6


def test_empty_keep_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_script()
    _build_chunked_map(tmp_path)
    df = pd.read_csv(tmp_path / "map_points.csv")
    df["status"] = "ERROR"  # no PASS cells -> empty keep-set
    df.to_csv(tmp_path / "map_points.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["prune", str(tmp_path), "--top-k", "2", "--apply"])
    with pytest.raises(SystemExit):
        mod.main()
    assert len(list(tmp_path.rglob("pump_solution.npz"))) == 6
