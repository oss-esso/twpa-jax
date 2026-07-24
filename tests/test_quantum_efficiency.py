from __future__ import annotations

import numpy as np
import pytest

from twpa_solver.signal import calc_qe, calc_qe_ideal


def test_calc_qe_real():
    s = np.array([[3 / 5, 4 / 5], [4 / 5, 3 / 5]])
    qe = calc_qe(s)
    np.testing.assert_allclose(qe, [[0.36, 0.64], [0.64, 0.36]])


def test_calc_qe_complex():
    s = np.array([[3 / 5, 4 / 5], [4 / 5, 3 / 5]], dtype=np.complex128)
    qe = calc_qe(s)
    np.testing.assert_allclose(qe, [[0.36, 0.64], [0.64, 0.36]])


def test_calc_qe_with_zero_noise_matches_no_noise():
    s = np.array([[3 / 5, 4 / 5], [4 / 5, 3 / 5]])
    s_noise = np.zeros((2, 2))
    qe = calc_qe(s, s_noise)
    np.testing.assert_allclose(qe, [[0.36, 0.64], [0.64, 0.36]])


def test_calc_qe_complex_with_zero_noise_matches_no_noise():
    s = np.array([[3 / 5, 4 / 5], [4 / 5, 3 / 5]], dtype=np.complex128)
    s_noise = np.zeros((2, 2), dtype=np.complex128)
    qe = calc_qe(s, s_noise)
    np.testing.assert_allclose(qe, [[0.36, 0.64], [0.64, 0.36]])


def test_calc_qe_with_nonzero_noise():
    s = np.array([[1.0, 2.0], [3.0, 4.0]])
    s_noise = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    qe = calc_qe(s, s_noise)
    np.testing.assert_allclose(
        qe,
        [[0.0526316, 0.210526], [0.0882353, 0.156863]],
        rtol=1e-5,
    )


def test_calc_qe_noise_dimension_mismatch_raises():
    s = np.array([[1.0, 2.0], [3.0, 4.0]])
    s_noise = np.zeros((3, 2))
    with pytest.raises(ValueError):
        calc_qe(s, s_noise)


def test_calc_qe_ideal_real():
    s = np.array([[3 / 5, 4 / 5], [4 / 5, 3 / 5]])
    qe_ideal = calc_qe_ideal(s)
    np.testing.assert_allclose(qe_ideal, [[1.0, 1.0], [1.0, 1.0]])


def test_calc_qe_ideal_complex():
    s = np.array([[3 / 5, 4 / 5], [4 / 5, 3 / 5]], dtype=np.complex128)
    qe_ideal = calc_qe_ideal(s)
    np.testing.assert_allclose(qe_ideal, [[1.0, 1.0], [1.0, 1.0]])


def test_calc_qe_ideal_above_unity():
    # |s|^2 = 4 > 1: 1 / (2 - 1/4) = 4/7
    s = np.array([[2.0]])
    qe_ideal = calc_qe_ideal(s)
    np.testing.assert_allclose(qe_ideal, [[4.0 / 7.0]])
