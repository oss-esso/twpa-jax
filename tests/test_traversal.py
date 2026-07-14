"""Tests for the map traversal ordering and neighbour selection (pure logic).

These exercise ``_traversal_order`` / ``_grid_dims`` / ``_nearest_solved`` and
the predictor candidate wiring in ``scripts/run_gain_map.py`` without running any
solver, so they are fast and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

from scripts.run_gain_map import (  # noqa: E402
    GridPoint,
    _grid_dims,
    _nearest_solved,
    _recover,
    _traversal_order,
)


def _grid(n_power: int, n_freq: int) -> list[GridPoint]:
    pts: list[GridPoint] = []
    idx = 0
    for i in range(n_power):
        for j in range(n_freq):
            pts.append(GridPoint(idx, i, j, -35.0 + i, 7.5 + 0.1 * j, 1e-6))
            idx += 1
    return pts


def test_grid_dims() -> None:
    assert _grid_dims(_grid(4, 3)) == (4, 3)


def test_column_order_is_column_major_ascending_power() -> None:
    pts = _grid(3, 2)
    order = _traversal_order(pts, "column", "ltr")
    keys = [(p.j_freq, p.i_power) for p in order]
    assert keys == sorted(keys)  # grouped by frequency, ascending power within


def test_all_strategies_visit_every_cell_once() -> None:
    pts = _grid(4, 5)
    for strat in ("column", "backbone", "nearest", "serpentine", "floodfill"):
        order = _traversal_order(pts, strat, "center_out")
        seen = {(p.i_power, p.j_freq) for p in order}
        assert len(order) == len(pts), strat
        assert seen == {(p.i_power, p.j_freq) for p in pts}, strat


def test_backbone_solves_lowest_power_row_first() -> None:
    pts = _grid(4, 5)
    order = _traversal_order(pts, "backbone", "ltr")
    # The first n_freq cells must all be the lowest power row (i == 0).
    n_freq = 5
    assert all(p.i_power == 0 for p in order[:n_freq])
    # ltr backbone row visits frequencies in ascending column order.
    assert [p.j_freq for p in order[:n_freq]] == list(range(n_freq))


def test_backbone_center_out_starts_at_middle_frequency() -> None:
    pts = _grid(3, 5)
    order = _traversal_order(pts, "backbone", "center_out")
    assert order[0].i_power == 0
    assert order[0].j_freq == 2  # middle of 5 columns


def test_serpentine_alternates_power_direction() -> None:
    pts = _grid(3, 2)
    order = _traversal_order(pts, "serpentine", "ltr")
    col0 = [p.i_power for p in order if p.j_freq == 0]
    col1 = [p.i_power for p in order if p.j_freq == 1]
    # order preserves emission sequence; even column ascending, odd descending.
    first_col0 = [p.i_power for p in order[:3]]
    next_col1 = [p.i_power for p in order[3:6]]
    assert first_col0 == [0, 1, 2]
    assert next_col1 == [2, 1, 0]


def test_floodfill_starts_central_low_power() -> None:
    pts = _grid(3, 5)
    order = _traversal_order(pts, "floodfill", "center_out")
    assert (order[0].i_power, order[0].j_freq) == (0, 2)


def test_nearest_solved_picks_closest() -> None:
    solved = {(0, 0): {}, (2, 2): {}}
    assert _nearest_solved(0, 1, solved, 4, 4) == (0, 0)
    assert _nearest_solved(3, 3, solved, 4, 4) == (2, 2)


def test_generic_fail_fast_does_not_force_final_reseed(tmp_path) -> None:
    class Engine:
        def solve_point(self, *args, **kwargs):
            raise AssertionError("fail-fast recovery=none must not solve again")

    args = type(
        "Args",
        (),
        {
            "recovery": "none",
            "fold_policy": "patience",
            "inproc_fail_fast": True,
        },
    )()
    failed = {"status": "ERROR", "pump_failure_reason": "stalled"}

    row, state, ok, tag = _recover(
        Engine(),
        _grid(1, 1)[0],
        tmp_path,
        None,
        None,
        1.0,
        {},
        1,
        1,
        [],
        args,
        failed,
        None,
    )

    assert row is failed
    assert state is None
    assert ok is False
    assert tag == "fail_fast"
