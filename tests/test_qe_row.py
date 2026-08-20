"""Quantum-efficiency reduction of a signal-output scattering row."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from twpa_solver.signal.qe_row import reduce_signal_row

MS = [-2, 0, 2]


def _unitary_row(gain: float, *, leak: float = 0.0) -> np.ndarray:
    """Signal row whose idler carries exactly the strength unitarity requires."""
    return np.array([np.sqrt(gain - 1.0), np.sqrt(gain), leak], dtype=np.complex128)


def test_unitary_row_sits_at_its_own_quantum_limit() -> None:
    reduced = reduce_signal_row(MS, _unitary_row(4.0), signal_m=0, idler_m=-2)

    assert reduced.qe_ratio == pytest.approx(1.0, abs=1e-12)
    assert reduced.unitarity_residual == pytest.approx(1.0, abs=1e-12)


def test_power_leaking_to_other_sidebands_lowers_the_ratio() -> None:
    clean = reduce_signal_row(MS, _unitary_row(4.0), signal_m=0, idler_m=-2)
    leaky = reduce_signal_row(
        MS, _unitary_row(4.0, leak=0.5), signal_m=0, idler_m=-2
    )

    assert leaky.qe_ratio < clean.qe_ratio
    assert leaky.qe_ratio < 1.0
    # Leakage does not touch the signal/idler pair, so the residual is unmoved.
    assert leaky.unitarity_residual == pytest.approx(1.0, abs=1e-12)


def test_idler_deficit_pushes_the_ratio_above_one() -> None:
    """A row with less idler than unitarity demands exceeds its own bound.

    This is the diagnostic the ratio carries: above 1 means the solved row is
    not unitary, not that the amplifier beat the quantum limit.
    """
    short_idler = np.array([np.sqrt(2.0), np.sqrt(4.0), 0.0], dtype=np.complex128)

    reduced = reduce_signal_row(MS, short_idler, signal_m=0, idler_m=-2)

    assert reduced.unitarity_residual == pytest.approx(2.0)
    assert reduced.qe_ratio > 1.0


def test_reduction_reports_the_row_width_and_signal_amplitude() -> None:
    reduced = reduce_signal_row(MS, _unitary_row(9.0), signal_m=0, idler_m=-2)

    assert reduced.sidebands_summed == 3
    assert reduced.s_ss_abs == pytest.approx(3.0)


def _write_sweep_csv(path, rows: list[dict[str, str]], header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def test_sweep_without_qe_column_loads_as_absent(tmp_path) -> None:
    """Sweeps written before the QE column existed must still plot."""
    from scripts.plot_gain_map import _load_sweep

    path = tmp_path / "gain_sweep.csv"
    _write_sweep_csv(
        path,
        [{"status": "VALID_SOLVED", "signal_ghz": "7.8", "gain_db": "3.5"}],
        ["status", "signal_ghz", "gain_db"],
    )

    freq, gain, qe_ratio = _load_sweep(path)

    assert freq.tolist() == [7.8]
    assert gain.tolist() == [3.5]
    assert qe_ratio is None


def test_sweep_with_qe_column_loads_the_ratio(tmp_path) -> None:
    from scripts.plot_gain_map import _load_sweep

    path = tmp_path / "gain_sweep.csv"
    _write_sweep_csv(
        path,
        [
            {"status": "VALID_SOLVED", "signal_ghz": "7.8", "gain_db": "3.5",
             "qe_ratio": "1.032042"},
            {"status": "CHECK", "signal_ghz": "7.85", "gain_db": "3.6",
             "qe_ratio": "0.9"},
        ],
        ["status", "signal_ghz", "gain_db", "qe_ratio"],
    )

    freq, _, qe_ratio = _load_sweep(path)

    assert freq.tolist() == [7.8]
    assert qe_ratio is not None
    assert qe_ratio.tolist() == pytest.approx([1.032042])
