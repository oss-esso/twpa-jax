"""Tests for the exp10 warm-start map gate.

The gate compares the warm-start pass against a cold reference. Two failure
modes that a naive implementation gets wrong are covered here: a sparse cold
spot-check (far fewer points than the warm pass) must not make the speedup look
like a slowdown, and a handful of stiff non-converged points must not invalidate
an otherwise-good large map.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXPERIMENTS))

import exp10_full_ipm_pump_map_warmstart as exp10  # noqa: E402


def _row(i: int, j: int, *, status: str, gain: float | None, pump_s: float) -> dict:
    return {
        "i_power": i,
        "j_freq": j,
        "status": status,
        "gain_db": gain,
        "pump_runtime_s": pump_s,
    }


def test_speedup_is_per_point_with_sparse_cold_spotcheck() -> None:
    # 1 cold spot-check point (slow) vs a full warm pass (many fast points).
    # Total cold seconds << total warm seconds, but per point warm is far faster:
    # a total-vs-total ratio would wrongly read as a slowdown.
    cold = [_row(0, 0, status="PASS", gain=5.0, pump_s=12.0)]
    warm = [_row(0, j, status="PASS", gain=5.0, pump_s=0.7) for j in range(20)]

    gate = exp10.evaluate_gate(cold, warm, gate_gain_db=0.01, min_converged_frac=0.98)

    assert gate.passed
    assert gate.pump_speedup == pytest_approx(12.0 / 0.7)
    # Totals favour cold (20*0.7 = 14 > 12), yet per-point speed is what counts.
    assert gate.warm_pump_runtime_s > gate.cold_pump_runtime_s
    assert gate.pump_speedup > 1.0


def test_sparse_nonconvergence_within_threshold_passes() -> None:
    cold = [_row(0, 0, status="PASS", gain=5.0, pump_s=14.0)]
    warm = [_row(0, j, status="PASS", gain=5.0, pump_s=0.7) for j in range(99)]
    warm.append(_row(0, 99, status="ERROR", gain=None, pump_s=14.0))  # 1/100 failed

    gate = exp10.evaluate_gate(cold, warm, gate_gain_db=0.01, min_converged_frac=0.98)

    assert gate.passed
    assert gate.n_warm_failed == 1
    assert gate.warm_converged_frac == pytest_approx(0.99)


def test_too_many_nonconverged_fails() -> None:
    cold = [_row(0, 0, status="PASS", gain=5.0, pump_s=14.0)]
    warm = [_row(0, j, status="PASS", gain=5.0, pump_s=0.7) for j in range(90)]
    warm += [_row(0, j, status="ERROR", gain=None, pump_s=14.0) for j in range(90, 100)]

    gate = exp10.evaluate_gate(cold, warm, gate_gain_db=0.01, min_converged_frac=0.98)

    assert not gate.passed
    assert any("convergence" in r for r in gate.reasons)


def test_gain_drift_over_gate_fails() -> None:
    cold = [_row(0, 0, status="PASS", gain=5.0, pump_s=14.0)]
    warm = [_row(0, 0, status="PASS", gain=5.5, pump_s=0.7)]  # 0.5 dB drift

    gate = exp10.evaluate_gate(cold, warm, gate_gain_db=0.01, min_converged_frac=0.98)

    assert not gate.passed
    assert any("drift" in r for r in gate.reasons)
    assert gate.max_gain_drift_db == pytest_approx(0.5)


def test_secant_guess_uniform_step_doubles_the_delta() -> None:
    import numpy as np

    # Uniform current spacing -> beta = 1 -> X_guess = 2*prev - prevprev.
    x_pp = np.array([1.0 + 0j, 2.0 + 0j])
    x_p = np.array([2.0 + 0j, 3.0 + 0j])
    g = exp10.secant_guess(x_pp, x_p, 1.0, 2.0, 3.0)
    assert np.allclose(g, np.array([3.0, 4.0]))


def test_secant_guess_is_exact_on_affine_state() -> None:
    import numpy as np

    # A state that is exactly affine in the current is predicted with zero error,
    # including a non-uniform target step.
    def state(cur: float) -> np.ndarray:
        return np.array([0.5 + 1.0 * cur, -2.0 + 0.25 * cur], dtype=np.complex128)

    g = exp10.secant_guess(state(1.0), state(2.0), 1.0, 2.0, 2.5)
    assert np.allclose(g, state(2.5))


def test_secant_guess_degenerate_denominator_returns_prev() -> None:
    import numpy as np

    x_pp = np.array([1.0 + 0j, 2.0 + 0j])
    x_p = np.array([2.0 + 0j, 3.0 + 0j])
    # Equal currents -> no secant available -> fall back to the last solution.
    g = exp10.secant_guess(x_pp, x_p, 2.0, 2.0, 3.0)
    assert np.allclose(g, x_p)


def pytest_approx(value: float, rel: float = 1e-9):
    import pytest

    return pytest.approx(value, rel=rel)
