"""Bounded CPU smoke gates for the GPU-session JAX path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.chaos.run_guarcello_jc_phase5 import (
    PHI0_REDUCED,
    integrate_jc_banded_batch,
    load_jc_device,
)


ROOT = Path(__file__).resolve().parents[1]


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(first) - np.asarray(second)))
    denominator = float(np.linalg.norm(np.asarray(second)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _settings() -> dict[str, object]:
    return {
        "pump_currents_a": np.asarray([0.0, 0.1e-6]),
        "pump_hz": 7.9e9,
        "signal_current_a": 0.0,
        "signal_hz": 7.4e9,
        "dt_s": 0.01 / (3.4e-6 / (PHI0_REDUCED * 90.0e-15)) ** 0.5,
        "n_steps": 600,
        "record_stride": 20,
        "backend": "jax",
        "jax_device": "cpu",
    }


@pytest.mark.parametrize("solve_kind", ("sequential", "scan"))
def test_chunked_recording_matches_materialized_oracle(solve_kind: str) -> None:
    device = load_jc_device(ROOT / "designs/ipm_2c_fixed")
    settings = _settings()
    materialized = integrate_jc_banded_batch(
        device, solve_kind=solve_kind, chunked_scan=False, **settings,
    )
    chunked = integrate_jc_banded_batch(
        device, solve_kind=solve_kind, chunked_scan=True, **settings,
    )
    for index in (0, 1, 2, 4):
        assert chunked[index].shape == materialized[index].shape
        assert np.all(np.isfinite(chunked[index]))
        assert _relative(chunked[index], materialized[index]) <= 1.0e-12


@pytest.mark.parametrize("solve_kind", ("sequential", "scan"))
def test_jax_float32_smoke_is_finite_and_close_to_float64(
    solve_kind: str,
) -> None:
    device = load_jc_device(ROOT / "designs/ipm_2c_fixed")
    settings = _settings()
    float64 = integrate_jc_banded_batch(
        device, solve_kind=solve_kind, dtype="float64", **settings,
    )
    float32 = integrate_jc_banded_batch(
        device, solve_kind=solve_kind, dtype="float32", **settings,
    )
    for result in (float64, float32):
        assert result[0].shape == (2, 31)
        assert result[1].shape == (2, 31)
        assert result[2].shape == (2, 31)
        assert result[4].shape == (2, device.n_nodes)
        assert all(np.all(np.isfinite(value)) for value in result[:3])
        assert np.all(np.isfinite(result[4]))
    # This is only a wiring smoke gate; the observable precision decision is
    # made by measure_kernel_precision.py on the GPU box.
    assert _relative(float32[4], float64[4]) <= 3.0e-3
