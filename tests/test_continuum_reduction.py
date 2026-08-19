from __future__ import annotations

import numpy as np

from scripts.chaos.measure_continuum_closure import (
    residue_metrics,
    split_coherent_residue,
)


def test_coherent_and_residue_power_reconstruct_total() -> None:
    frequency = np.linspace(0.5, 32.0, 2001) * 1.0e9
    power = np.full(frequency.size, 1.0e-4)
    coherent_indices = [int(np.argmin(abs(frequency - target * 1.0e9))) for target in (7.9, 15.8)]
    power[coherent_indices] = [1.0, 0.5]

    coherent, residue, mask = split_coherent_residue(
        frequency,
        power,
        pump_hz=7.9e9,
        signal_hz=7.4e9,
        window_hz=3.0 * (frequency[1] - frequency[0]),
    )
    metrics = residue_metrics(
        frequency,
        coherent,
        residue,
        pump_hz=7.9e9,
        window_hz=3.0 * (frequency[1] - frequency[0]),
        coherent_mask=mask,
    )

    assert np.isclose(
        metrics.coherent_power + metrics.residue_power,
        metrics.total_power,
        rtol=1.0e-12,
    )
    assert metrics.coherent_bins > 0
    assert metrics.residue_bins > 0


def test_split_is_stable_when_fft_grid_is_shifted_within_the_window() -> None:
    base = np.linspace(0.5, 32.0, 2001) * 1.0e9
    shifted = base + 0.2 * (base[1] - base[0])
    power_base = np.exp(-((base - 12.0e9) / 2.0e9) ** 2)
    power_shifted = np.exp(-((shifted - 12.0e9) / 2.0e9) ** 2)
    window = 3.0 * (base[1] - base[0])

    first = split_coherent_residue(
        base, power_base, pump_hz=7.9e9, signal_hz=7.4e9, window_hz=window,
    )
    second = split_coherent_residue(
        shifted, power_shifted, pump_hz=7.9e9, signal_hz=7.4e9, window_hz=window,
    )
    first_fraction = float(first[1].sum() / power_base.sum())
    second_fraction = float(second[1].sum() / power_shifted.sum())

    assert abs(first_fraction - second_fraction) < 0.01
