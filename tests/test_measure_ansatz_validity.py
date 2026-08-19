"""Tests for pump-plus-signal and pump-only spectral reductions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.chaos.measure_ansatz_validity import analyse_point


def test_pump_only_reduction_is_additive_to_signal_lattice(tmp_path: Path) -> None:
    """The pump-only mode excludes a signal-frequency line from the lattice."""
    point = tmp_path / "ipm_2c_fixed" / "dense_p0p5950"
    point.mkdir(parents=True)
    (point / "result.json").write_text(
        json.dumps({
            "device": "ipm_2c_fixed",
            "control_axis": "I_over_I_bound",
            "control_value": 0.595,
            "pump_hz": 7.9e9,
            "signal_hz": 7.4e9,
        }),
        encoding="utf-8",
    )
    frequencies = np.concatenate([
        np.asarray([7.9e9, 15.8e9, 7.4e9, 14.8e9]),
        np.linspace(0.2e9, 40.0e9, 100),
    ])
    np.savez(
        point / "spectrum.npz",
        frequency_hz=frequencies,
        spectrum_db_relative_pump=np.asarray(
            [0.0, -20.0, -3.0, -30.0] + [-60.0] * 100,
        ),
    )

    row = analyse_point(point, n_generator_trials=8, include_pump_only=True)

    assert row is not None
    assert row["on_lattice"] > 0.99
    assert row["on_lattice_pump_only"] < row["on_lattice"]
    assert row["off_lattice_pump_only"] > row["off_lattice"]


def test_default_reduction_preserves_signal_lattice_columns(tmp_path: Path) -> None:
    """The default reducer does not change its historical output contract."""
    point = tmp_path / "device" / "point"
    point.mkdir(parents=True)
    (point / "result.json").write_text(
        json.dumps({"device": "device", "pump_hz": 8.0e9, "signal_hz": 7.0e9}),
        encoding="utf-8",
    )
    frequencies = np.concatenate([
        np.asarray([8.0e9, 16.0e9, 7.0e9, 14.0e9]),
        np.linspace(0.2e9, 40.0e9, 100),
    ])
    np.savez(
        point / "spectrum.npz",
        frequency_hz=frequencies,
        spectrum_db_relative_pump=np.asarray(
            [0.0, -20.0, -3.0, -30.0] + [-60.0] * 100,
        ),
    )

    row = analyse_point(point, n_generator_trials=8)

    assert row is not None
    assert "on_lattice" in row
    assert "on_lattice_pump_only" not in row
