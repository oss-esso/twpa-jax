from __future__ import annotations

import numpy as np

from scripts.chaos.run_rf_squid_bias import _largest_lyapunov_at_target


class _DoublingAdapter:
    def __init__(self) -> None:
        self.state = np.array([7.0])
        self.current_a = 3.0

    def integrate(self, _target: float, state: np.ndarray):
        value = 2.0 * np.asarray(state, dtype=float)
        self.state = value.copy()
        self.current_a = 99.0
        return value, np.array([0.0]), np.array([0.0]), np.zeros((1, 1))


def test_rf_bias_lyapunov_map_restores_branch_state() -> None:
    adapter = _DoublingAdapter()
    state = np.array([1.0])

    value = _largest_lyapunov_at_target(adapter, state, 5.0, periods=12)

    assert np.isclose(value, np.log(2.0))
    assert np.array_equal(adapter.state, np.array([7.0]))
    assert adapter.current_a == 3.0


def test_rf_bias_lyapunov_can_be_disabled() -> None:
    adapter = _DoublingAdapter()
    assert _largest_lyapunov_at_target(adapter, np.array([1.0]), 5.0, 0) is None
