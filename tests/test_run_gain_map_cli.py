"""CLI defaults for the gain-map runner."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

import scripts.run_gain_map as run_gain_map


def test_inproc_fail_fast_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py"])

    args = run_gain_map.parse_args()

    assert args.inproc_fail_fast is False


def test_inproc_fail_fast_flag_enables_fast_failure(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py", "--inproc-fail-fast"])

    args = run_gain_map.parse_args()

    assert args.inproc_fail_fast is True


def test_frequency_chunk_size_defaults_to_ten_columns(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py"])

    args = run_gain_map.parse_args()

    assert args.frequency_chunk_size == 10
    assert args.resume_chunks is True
    assert args.signal_spectrum is True


def test_signal_spectrum_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py", "--no-signal-spectrum"])

    args = run_gain_map.parse_args()

    assert args.signal_spectrum is False


def test_frequency_chunk_ranges_are_half_open() -> None:
    assert run_gain_map.frequency_chunk_ranges(25, 10) == [(0, 10), (10, 20), (20, 25)]
    assert run_gain_map.frequency_chunk_ranges(8, 10) == [(0, 8)]
    assert run_gain_map.frequency_chunk_ranges(8, 0) == [(0, 8)]


def test_chunk_worker_command_strips_parent_routing_options() -> None:
    cmd = run_gain_map.chunk_worker_command(
        [
            "--outdir",
            "outputs/full",
            "--frequency-chunk-size",
            "10",
            "--gate-spotcheck=5",
            "--overwrite",
            "--n-frequency",
            "50",
            "--pump-freq-min-ghz",
            "7.5",
            "--pump-freq-max-ghz",
            "8.5",
        ],
        outdir=Path("outputs/full/chunks/chunk_000"),
        n_frequency=10,
        pump_freq_min_ghz=7.5,
        pump_freq_max_ghz=7.683673469387755,
    )

    assert "--chunk-worker" in cmd
    assert "--n-frequency" in cmd
    assert "10" in cmd
    assert "--pump-freq-min-ghz" in cmd
    assert "--pump-freq-max-ghz" in cmd
    assert "7.5" in cmd
    assert "7.68367346939" in cmd
    assert "50" not in cmd
    assert "8.5" not in cmd
    assert "--frequency-chunk-size" not in cmd
    assert "--gate-spotcheck=5" not in cmd
    assert cmd.count("--outdir") == 1


def test_read_chunk_rows_globalizes_frequency_and_point_indices(tmp_path) -> None:
    chunk_dir = tmp_path / "chunk"
    chunk_dir.mkdir()
    (chunk_dir / "map_points.csv").write_text(
        "pass,point_index,i_power,j_freq,pump_power_dbm,pump_freq_ghz,status\n"
        "warm,0,0,0,-30,7.5,PASS\n"
        "warm,3,1,1,-29,7.6,PASS\n",
        encoding="utf-8",
    )

    _cold, warm = run_gain_map.read_chunk_rows(
        [(chunk_dir, 10, 12)],
        global_n_frequency=50,
    )

    assert warm[0]["j_freq"] == 10
    assert warm[0]["point_index"] == 10
    assert warm[1]["j_freq"] == 11
    assert warm[1]["point_index"] == 61


def test_fail_fast_does_not_retry_secant_fallback(tmp_path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                inproc_fold_predictor="secant",
                pump_current_jc_scale=1.0,
                fold_skip_patience=0,
            )
            self.calls = []

        def solve_point(self, point, pass_dir, *, mode, warm_X):
            self.calls.append((point.index, mode, None if warm_X is None else warm_X.copy()))
            status = "ERROR" if point.index == 2 else "PASS"
            row = {
                "point_index": point.index,
                "status": status,
                "gain_db": None,
                "pump_newton_total": 1,
                "pump_runtime_s": 0.1,
            }
            return row, np.array([float(point.index + 1)])

    points = [
        run_gain_map.GridPoint(0, 0, 0, -35.0, 7.5, 1.0),
        run_gain_map.GridPoint(1, 1, 0, -34.0, 7.5, 2.0),
        run_gain_map.GridPoint(2, 2, 0, -33.0, 7.5, 3.0),
    ]
    engine = FakeEngine()

    rows = run_gain_map.run_warm_pass_inprocess(
        points,
        tmp_path,
        engine,
        fail_fast=True,
    )

    assert [call[0] for call in engine.calls] == [0, 1, 2]
    assert rows[-1]["status"] == "ERROR"
    assert rows[-1]["pump_predictor"] == "secant"
