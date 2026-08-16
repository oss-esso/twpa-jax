from __future__ import annotations

import numpy as np
import pytest

from scripts.chaos.lyapunov import benettin_flow, benettin_map
from scripts.chaos.symmetry import measure_solution
from scripts.chaos.tongues import _event_scan, nearest_rational


def test_refined_event_scan_detects_a_single_generator() -> None:
    pump = 10.0
    rho_true = 0.137
    frequency = np.linspace(0.02, 60.0, 3000)
    target = np.abs(2.0 * pump + rho_true * pump)
    power = np.exp(-((frequency - target) / 0.01) ** 2) + 1.0e-6
    rho, shares = _event_scan(frequency, power, pump, 6.0, 0.02, 20_001)
    measured = float(rho[np.argmax(shares)])
    assert min(abs(measured - rho_true), abs(2.0 * measured - rho_true)) < 2.0e-3


def test_nearest_rational_returns_reduced_fraction() -> None:
    assert nearest_rational(1.0 / 3.0, 13)[:2] == (1, 3)


def test_benettin_flow_matches_negative_linear_exponent() -> None:
    matrix = np.diag((-0.3, -1.1))
    result = benettin_flow(
        lambda _t, y: matrix @ y,
        lambda _t, _y: matrix,
        np.ones(2),
        h=0.01,
        steps=20_000,
        renormalize_every=10,
    )
    assert abs(float(result["lambda_1"]) + 0.3) < 0.01


def test_benettin_map_matches_known_contraction() -> None:
    result = benettin_map(
        lambda y: np.array((0.5 * y[0], 0.25 * y[1])),
        lambda _y: np.diag((0.5, 0.25)),
        np.ones(2),
        steps=2_000,
    )
    assert abs(float(result["lambda_1"]) - np.log(0.5)) < 0.01


def test_symmetry_measurement_rejects_odd_only_solution(tmp_path) -> None:
    path = tmp_path / "pump_solution.npz"
    np.savez(path, X_real=np.ones((2, 3)), X_imag=np.zeros((2, 3)), pump_modes=np.array([1, 3]))
    with pytest.raises(ValueError, match="odd-only"):
        measure_solution(path, "synthetic")


def test_symmetry_measurement_detects_even_component(tmp_path) -> None:
    path = tmp_path / "pump_solution.npz"
    np.savez(
        path,
        X_real=np.array([[1.0, 0.0], [0.2, 0.0]]),
        X_imag=np.zeros((2, 2)),
        pump_modes=np.array([1, 2]),
    )
    row = measure_solution(path, "synthetic")
    assert row["even_to_fundamental_ratio"] == pytest.approx(0.2)
    assert row["half_period_residual"] > 0.0
