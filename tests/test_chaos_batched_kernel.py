from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.chaos.run_guarcello_jc_phase5 import (
    _integrate_jc_compiled,
    integrate_jc_banded_batch,
    load_jc_device,
)


ROOT = Path(__file__).resolve().parents[1]
DEVICE_DIR = ROOT / "outputs/jc_doc_python_designs/jc_jtwpa"


def _settings(device):
    return {
        "pump_hz": 7.12e9,
        "signal_current_a": 1.0e-9,
        "signal_hz": 6.62e9,
        "dt_s": 0.01 / (3.4e-6 / (2.067833848e-15 * 90.0e-15)) ** 0.5,
        "n_steps": 80,
        "record_stride": 20,
    }


def test_numba_batch_is_bit_identical_to_sequential_compiled_runs() -> None:
    device = load_jc_device(DEVICE_DIR)
    settings = _settings(device)
    drives = np.asarray([0.8e-6, 1.0e-6, 1.2e-6])

    sequential = [
        _integrate_jc_compiled(
            device, pump_current_a=float(drive), initial_q=None, **settings,
        )
        for drive in drives
    ]
    batched = integrate_jc_banded_batch(
        device, pump_currents_a=drives, backend="numba", **settings,
    )

    for lane, result in enumerate(sequential):
        assert np.array_equal(batched[0][lane], result[0])
        assert np.array_equal(batched[1][lane], result[1])
        assert np.array_equal(batched[2][lane], result[2])
        assert np.array_equal(batched[4][lane], result[4])


def test_batch_one_matches_compiled_scalar_with_a_custom_initial_state() -> None:
    device = load_jc_device(DEVICE_DIR)
    settings = _settings(device)
    initial = np.linspace(-1.0e-18, 1.0e-18, device.n_nodes)
    previous = initial * 0.5
    scalar = _integrate_jc_compiled(
        device,
        pump_current_a=1.0e-6,
        initial_q=initial,
        initial_q_previous=previous,
        **settings,
    )
    batch = integrate_jc_banded_batch(
        device,
        pump_currents_a=np.asarray([1.0e-6]),
        initial_q=initial[None, :],
        initial_q_previous=previous[None, :],
        backend="numba",
        **settings,
    )

    assert np.array_equal(batch[0][0], scalar[0])
    assert np.array_equal(batch[1][0], scalar[1])
    assert np.array_equal(batch[2][0], scalar[2])
    assert np.array_equal(batch[4][0], scalar[4])


def test_initial_state_perturbation_stays_in_one_batch_lane() -> None:
    device = load_jc_device(DEVICE_DIR)
    settings = _settings(device)
    initial = np.zeros((3, device.n_nodes), dtype=float)
    initial[1, :] = 1.0e-18
    baseline = integrate_jc_banded_batch(
        device, pump_currents_a=np.asarray([0.8e-6, 1.0e-6, 1.2e-6]),
        initial_q=initial, backend="numba", **settings,
    )
    changed_initial = initial.copy()
    changed_initial[1, 0] = 3.0e-16
    changed = integrate_jc_banded_batch(
        device, pump_currents_a=np.asarray([0.8e-6, 1.0e-6, 1.2e-6]),
        initial_q=changed_initial, backend="numba", **settings,
    )

    assert np.array_equal(baseline[1][0], changed[1][0])
    assert np.array_equal(baseline[1][2], changed[1][2])
    assert np.array_equal(baseline[4][0], changed[4][0])
    assert np.array_equal(baseline[4][2], changed[4][2])
    assert not np.array_equal(baseline[4][1], changed[4][1])
