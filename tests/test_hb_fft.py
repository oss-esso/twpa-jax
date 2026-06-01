"""
Tests for twpa.core.hb_fft.

These tests define the expected Fourier-transform convention used by the
harmonic-balance stack.

Convention
----------
For harmonic orders k and fundamental angular frequency omega0,

    x(t_n) = sum_k X_k exp(+i k omega0 t_n)

and the inverse projection is

    X_k = (1/N) sum_n x(t_n) exp(-i k omega0 t_n)

The tests are API-tolerant in function names, but require the module to expose
a usable public interface for:

    - synthesis from harmonic coefficients to time samples,
    - projection from time samples to selected harmonic coefficients,
    - round trips on selected harmonic bases,
    - real-signal conjugate symmetry,
    - nonlinear products through time-domain multiplication.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import twpa.core.hb_fft as hb


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _get_attr_or_key(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _time_grid(n_time: int, omega0_rad_s: float = 2.0 * np.pi) -> np.ndarray:
    for name in [
        "make_time_grid",
        "time_grid",
        "periodic_time_grid",
        "hb_time_grid",
    ]:
        if hasattr(hb, name):
            return np.asarray(
                _call_with_supported_kwargs(
                    getattr(hb, name),
                    n_time=n_time,
                    n=n_time,
                    omega0_rad_s=omega0_rad_s,
                    omega0=omega0_rad_s,
                    fundamental_angular_frequency_rad_s=omega0_rad_s,
                    period_s=2.0 * np.pi / omega0_rad_s,
                ),
                dtype=float,
            )

    period = 2.0 * np.pi / omega0_rad_s
    return np.arange(n_time, dtype=float) * period / n_time


def _synthesize(
    coeffs: Any,
    orders: Sequence[int],
    *,
    n_time: int = 128,
    omega0_rad_s: float = 2.0 * np.pi,
) -> np.ndarray:
    t = _time_grid(n_time, omega0_rad_s)

    for name in [
        "coefficients_to_time",
        "coeffs_to_time",
        "harmonic_coefficients_to_time",
        "harmonics_to_time",
        "synthesize_time_signal",
        "synthesize",
        "ifft_selected_harmonics",
        "hb_ifft",
        "inverse_hb_fft",
    ]:
        if hasattr(hb, name):
            out = _call_with_supported_kwargs(
                getattr(hb, name),
                coeffs=coeffs,
                coefficients=coeffs,
                harmonic_coeffs=coeffs,
                X=coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                harmonics=np.asarray(orders),
                n_time=n_time,
                n=n_time,
                t=t,
                time_s=t,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
            )

            value = _get_attr_or_key(out, "time_signal", "x_t", "samples", "values", default=out)
            return np.asarray(value)

    raise AttributeError(
        "twpa.core.hb_fft must expose a synthesis helper such as "
        "coefficients_to_time, harmonics_to_time, or hb_ifft."
    )


def _analyze(
    samples: Any,
    orders: Sequence[int],
    *,
    omega0_rad_s: float = 2.0 * np.pi,
) -> np.ndarray:
    x = np.asarray(samples)
    n_time = x.shape[0]
    t = _time_grid(n_time, omega0_rad_s)

    for name in [
        "time_to_coefficients",
        "time_to_coeffs",
        "time_to_harmonic_coefficients",
        "time_to_harmonics",
        "analyze_time_signal",
        "project_harmonics",
        "fft_selected_harmonics",
        "hb_fft",
        "forward_hb_fft",
    ]:
        if hasattr(hb, name):
            out = _call_with_supported_kwargs(
                getattr(hb, name),
                samples=samples,
                values=samples,
                time_signal=samples,
                x_t=samples,
                x=samples,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                harmonics=np.asarray(orders),
                t=t,
                time_s=t,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
            )

            value = _get_attr_or_key(out, "coeffs", "coefficients", "harmonic_coeffs", "X", default=out)
            return np.asarray(value)

    raise AttributeError(
        "twpa.core.hb_fft must expose an analysis helper such as "
        "time_to_coefficients, project_harmonics, or hb_fft."
    )


def _round_trip(
    coeffs: Any,
    orders: Sequence[int],
    *,
    n_time: int = 128,
    omega0_rad_s: float = 2.0 * np.pi,
) -> np.ndarray:
    x = _synthesize(coeffs, orders, n_time=n_time, omega0_rad_s=omega0_rad_s)
    return _analyze(x, orders, omega0_rad_s=omega0_rad_s)


def _manual_synthesize(
    coeffs: np.ndarray,
    orders: Sequence[int],
    *,
    n_time: int,
    omega0_rad_s: float = 2.0 * np.pi,
) -> np.ndarray:
    t = _time_grid(n_time, omega0_rad_s)
    orders_arr = np.asarray(orders, dtype=int)
    coeffs_arr = np.asarray(coeffs, dtype=np.complex128)

    phase = np.exp(1j * np.outer(t * omega0_rad_s, orders_arr))
    return phase @ coeffs_arr


def _manual_analyze(
    samples: np.ndarray,
    orders: Sequence[int],
    *,
    omega0_rad_s: float = 2.0 * np.pi,
) -> np.ndarray:
    x = np.asarray(samples, dtype=np.complex128)
    n_time = x.shape[0]
    t = _time_grid(n_time, omega0_rad_s)
    orders_arr = np.asarray(orders, dtype=int)

    phase = np.exp(-1j * np.outer(t * omega0_rad_s, orders_arr))
    return phase.T @ x / n_time


def test_time_grid_is_periodic_and_uniform() -> None:
    n_time = 64
    omega0 = 2.0 * np.pi * 10e9
    t = _time_grid(n_time, omega0)

    assert t.shape == (n_time,)
    assert t[0] == pytest.approx(0.0, abs=1e-30)
    assert np.all(np.isfinite(t))
    assert np.all(np.diff(t) > 0.0)

    dt = np.diff(t)
    np.testing.assert_allclose(dt, dt[0], rtol=1e-13, atol=1e-30)

    period = 2.0 * np.pi / omega0
    assert t[-1] < period
    assert t[-1] + dt[0] == pytest.approx(period, rel=1e-13)


def test_single_positive_harmonic_synthesis_matches_exp_plus_iwt() -> None:
    orders = [1]
    coeffs = np.array([2.0 - 0.5j])
    n_time = 128

    x = _synthesize(coeffs, orders, n_time=n_time)
    expected = _manual_synthesize(coeffs, orders, n_time=n_time)

    assert x.shape[0] == n_time
    np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)


def test_single_negative_harmonic_synthesis_matches_exp_minus_iwt() -> None:
    orders = [-1]
    coeffs = np.array([1.5 + 0.25j])
    n_time = 128

    x = _synthesize(coeffs, orders, n_time=n_time)
    expected = _manual_synthesize(coeffs, orders, n_time=n_time)

    np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)


def test_multi_harmonic_synthesis_matches_manual_formula() -> None:
    orders = [-3, -1, 1, 3]
    coeffs = np.array([0.2 - 0.1j, 1.0 + 0.3j, 1.0 - 0.3j, 0.2 + 0.1j])
    n_time = 256

    x = _synthesize(coeffs, orders, n_time=n_time)
    expected = _manual_synthesize(coeffs, orders, n_time=n_time)

    np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)


def test_analysis_recovers_known_single_harmonic() -> None:
    orders = [-3, -1, 1, 3]
    coeffs = np.array([0.0, 0.0, 2.0 + 1.0j, 0.0])
    n_time = 256

    x = _manual_synthesize(coeffs, orders, n_time=n_time)
    recovered = _analyze(x, orders)

    np.testing.assert_allclose(recovered, coeffs, rtol=1e-12, atol=1e-12)


def test_analysis_recovers_manual_multi_harmonic_signal() -> None:
    orders = [-5, -3, -1, 1, 3, 5]
    coeffs = np.array(
        [
            -0.02 + 0.04j,
            0.10 - 0.20j,
            0.75 + 0.25j,
            0.75 - 0.25j,
            0.10 + 0.20j,
            -0.02 - 0.04j,
        ],
        dtype=np.complex128,
    )
    n_time = 512

    x = _manual_synthesize(coeffs, orders, n_time=n_time)
    recovered = _analyze(x, orders)

    np.testing.assert_allclose(recovered, coeffs, rtol=1e-12, atol=1e-12)


def test_round_trip_selected_harmonics() -> None:
    orders = [-3, -1, 1, 3]
    coeffs = np.array([0.1 - 0.2j, 1.0 + 0.5j, -0.3 + 0.8j, 0.05 + 0.01j])
    recovered = _round_trip(coeffs, orders, n_time=256)

    np.testing.assert_allclose(recovered, coeffs, rtol=1e-12, atol=1e-12)


def test_round_trip_with_nonunit_fundamental_frequency() -> None:
    orders = [-3, -1, 1, 3]
    coeffs = np.array([0.0 + 0.1j, 1.0 + 0.0j, 0.5 - 0.2j, -0.1 + 0.0j])
    omega0 = 2.0 * np.pi * 7.5e9

    recovered = _round_trip(coeffs, orders, n_time=256, omega0_rad_s=omega0)

    np.testing.assert_allclose(recovered, coeffs, rtol=1e-12, atol=1e-12)


def test_real_signal_conjugate_symmetric_coefficients_produce_real_samples() -> None:
    orders = [-3, -1, 1, 3]
    coeffs = np.array(
        [
            0.2 - 0.1j,
            1.0 + 0.4j,
            1.0 - 0.4j,
            0.2 + 0.1j,
        ],
        dtype=np.complex128,
    )

    x = _synthesize(coeffs, orders, n_time=256)

    np.testing.assert_allclose(np.imag(x), 0.0, rtol=0.0, atol=1e-12)


def test_real_cosine_coefficients_have_expected_amplitude() -> None:
    orders = [-1, 1]
    amplitude = 3.0
    phase = 0.37
    coeffs = np.array(
        [
            0.5 * amplitude * np.exp(-1j * phase),
            0.5 * amplitude * np.exp(1j * phase),
        ],
        dtype=np.complex128,
    )

    x = _synthesize(coeffs, orders, n_time=512)
    x_real = np.real(x)

    assert np.max(x_real) == pytest.approx(amplitude, rel=2e-4)
    assert np.min(x_real) == pytest.approx(-amplitude, rel=2e-4)
    np.testing.assert_allclose(np.imag(x), 0.0, atol=1e-12)


def test_projection_of_real_cosine_recovers_conjugate_pair() -> None:
    orders = [-1, 1]
    amplitude = 2.5
    phase = -0.2
    n_time = 512
    t = _time_grid(n_time, 2.0 * np.pi)

    x = amplitude * np.cos(2.0 * np.pi * t + phase)
    recovered = _analyze(x, orders)

    expected = np.array(
        [
            0.5 * amplitude * np.exp(-1j * phase),
            0.5 * amplitude * np.exp(1j * phase),
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(recovered, expected, rtol=1e-12, atol=1e-12)


def test_constant_dc_projection_if_zero_order_supported() -> None:
    orders = [0, 1]
    samples = np.ones(128) * 2.25

    try:
        recovered = _analyze(samples, orders)
    except (ValueError, AssertionError):
        pytest.skip("Zero-order/DC harmonics are intentionally unsupported by this package.")

    assert recovered[0] == pytest.approx(2.25 + 0.0j, rel=1e-13, abs=1e-13)
    assert recovered[1] == pytest.approx(0.0 + 0.0j, abs=1e-13)


def test_batch_synthesis_preserves_trailing_shape_if_supported() -> None:
    orders = [-1, 1]
    coeffs = np.array(
        [
            [1.0 + 0.0j, 0.5 + 0.2j],
            [1.0 - 0.0j, 0.5 - 0.2j],
        ],
        dtype=np.complex128,
    )

    try:
        x = _synthesize(coeffs, orders, n_time=128)
    except Exception as exc:
        pytest.skip(f"Batch coefficient synthesis is optional: {type(exc).__name__}: {exc}")

    assert x.shape[0] == 128
    assert x.shape[-1] == 2


def test_batch_analysis_preserves_trailing_shape_if_supported() -> None:
    orders = [-1, 1]
    coeffs = np.array(
        [
            [1.0 + 0.0j, 0.5 + 0.2j],
            [1.0 - 0.0j, 0.5 - 0.2j],
        ],
        dtype=np.complex128,
    )

    try:
        x = _synthesize(coeffs, orders, n_time=128)
        recovered = _analyze(x, orders)
    except Exception as exc:
        pytest.skip(f"Batch coefficient analysis is optional: {type(exc).__name__}: {exc}")

    assert recovered.shape == coeffs.shape
    np.testing.assert_allclose(recovered, coeffs, rtol=1e-12, atol=1e-12)


def test_product_in_time_matches_convolution_projection() -> None:
    orders = [-2, -1, 1, 2]
    coeffs_a = {
        -1: 0.5 + 0.2j,
        1: 0.5 - 0.2j,
    }
    coeffs_b = {
        -1: -0.3 + 0.1j,
        1: -0.3 - 0.1j,
    }

    a_vec = np.array([coeffs_a.get(k, 0.0) for k in orders], dtype=np.complex128)
    b_vec = np.array([coeffs_b.get(k, 0.0) for k in orders], dtype=np.complex128)

    x_a = _synthesize(a_vec, orders, n_time=512)
    x_b = _synthesize(b_vec, orders, n_time=512)

    product_coeffs = _analyze(x_a * x_b, orders)

    expected = []
    for k in orders:
        s = 0.0 + 0.0j
        for p, a_p in coeffs_a.items():
            q = k - p
            s += a_p * coeffs_b.get(q, 0.0)
        expected.append(s)

    np.testing.assert_allclose(product_coeffs, np.asarray(expected), rtol=1e-12, atol=1e-12)


def test_square_of_cosine_has_second_harmonic_and_dc_if_supported() -> None:
    n_time = 512
    t = _time_grid(n_time, 2.0 * np.pi)
    x = np.cos(2.0 * np.pi * t)

    orders = [0, -2, 2]
    try:
        coeffs = _analyze(x * x, orders)
    except (ValueError, AssertionError):
        pytest.skip("Zero-order/DC harmonics are intentionally unsupported by this package.")

    expected = np.array([0.5, 0.25, 0.25], dtype=np.complex128)
    np.testing.assert_allclose(coeffs, expected, rtol=1e-12, atol=1e-12)


def test_parseval_identity_for_selected_full_basis() -> None:
    orders = np.arange(-8, 9)
    coeffs = np.zeros(orders.shape, dtype=np.complex128)
    coeffs[orders == -3] = 0.2 - 0.1j
    coeffs[orders == -1] = 1.0 + 0.4j
    coeffs[orders == 1] = -0.3 + 0.7j
    coeffs[orders == 4] = 0.05 - 0.02j

    x = _synthesize(coeffs, orders, n_time=256)

    time_power = np.mean(np.abs(x) ** 2)
    coeff_power = np.sum(np.abs(coeffs) ** 2)

    assert time_power == pytest.approx(coeff_power, rel=1e-12, abs=1e-12)


def test_projection_is_orthogonal_for_distinct_harmonics() -> None:
    orders = [-5, -3, -1, 1, 3, 5]
    n_time = 512

    x = _manual_synthesize(np.array([1.0 + 0.0j]), [3], n_time=n_time)
    coeffs = _analyze(x, orders)

    expected = np.zeros(len(orders), dtype=np.complex128)
    expected[orders.index(3)] = 1.0

    np.testing.assert_allclose(coeffs, expected, rtol=1e-12, atol=1e-12)


def test_insufficient_time_samples_are_rejected_or_alias_consistently() -> None:
    orders = [-5, -3, -1, 1, 3, 5]
    coeffs = np.ones(len(orders), dtype=np.complex128)

    try:
        recovered = _round_trip(coeffs, orders, n_time=6)
    except (ValueError, AssertionError):
        return

    # If the implementation permits exactly N == number of harmonics, it must
    # still recover without aliasing for this selected basis.
    np.testing.assert_allclose(recovered, coeffs, rtol=1e-10, atol=1e-10)


def test_invalid_duplicate_orders_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _synthesize(np.array([1.0, 2.0]), [1, 1], n_time=64)

    with pytest.raises((ValueError, AssertionError)):
        _analyze(np.ones(64), [1, 1])


def test_coefficient_length_mismatch_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, IndexError)):
        _synthesize(np.array([1.0, 2.0, 3.0]), [-1, 1], n_time=64)


def test_noninteger_orders_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, TypeError)):
        _synthesize(np.array([1.0, 2.0]), [-1.5, 1.0], n_time=64)


def test_nonpositive_n_time_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _synthesize(np.array([1.0]), [1], n_time=0)


def test_nan_coefficients_propagate_to_time_samples() -> None:
    x = _synthesize(np.array([np.nan + 0.0j]), [1], n_time=64)

    assert np.any(~np.isfinite(np.asarray(x)))


def test_fft_matrix_helper_if_available() -> None:
    helper = None
    for name in [
        "harmonic_basis_matrix",
        "fourier_basis_matrix",
        "make_projection_matrix",
        "make_hb_fft_matrix",
    ]:
        if hasattr(hb, name):
            helper = getattr(hb, name)
            break

    if helper is None:
        pytest.skip("FFT/basis matrix helper is optional.")

    orders = np.array([-1, 1, 3])
    n_time = 128
    omega0 = 2.0 * np.pi
    t = _time_grid(n_time, omega0)

    B = np.asarray(
        _call_with_supported_kwargs(
            helper,
            orders=orders,
            harmonic_orders=orders,
            n_time=n_time,
            n=n_time,
            t=t,
            time_s=t,
            omega0_rad_s=omega0,
            omega0=omega0,
        )
    )

    assert B.shape[0] == n_time
    assert B.shape[1] == len(orders)
    np.testing.assert_allclose(np.abs(B), 1.0, rtol=1e-13, atol=1e-13)


def test_convolution_helper_if_available() -> None:
    helper = None
    for name in [
        "convolve_coefficients",
        "harmonic_convolution",
        "convolve_harmonics",
        "coefficient_convolution",
    ]:
        if hasattr(hb, name):
            helper = getattr(hb, name)
            break

    if helper is None:
        pytest.skip("Coefficient convolution helper is optional.")

    orders = np.array([-2, -1, 1, 2])
    a = np.array([0.0, 0.5 + 0.2j, 0.5 - 0.2j, 0.0])
    b = np.array([0.0, -0.3 + 0.1j, -0.3 - 0.1j, 0.0])

    conv = np.asarray(
        _call_with_supported_kwargs(
            helper,
            a=a,
            b=b,
            x=a,
            y=b,
            coeffs_a=a,
            coeffs_b=b,
            orders=orders,
            harmonic_orders=orders,
            output_orders=orders,
        )
    )

    x_a = _synthesize(a, orders, n_time=512)
    x_b = _synthesize(b, orders, n_time=512)
    expected = _analyze(x_a * x_b, orders)

    np.testing.assert_allclose(conv, expected, rtol=1e-12, atol=1e-12)


def test_multi_fundamental_round_trip_and_conjugates() -> None:
    frequencies = np.asarray([-10.0, -3.123, 3.123, 10.0])
    lattice = np.asarray([[-1, 0], [0, -1], [0, 1], [1, 0]])
    grid = hb.make_multi_fundamental_projection_grid(
        frequencies,
        fundamental_frequencies_hz=np.asarray([10.0, 3.123]),
        lattice_indices=lattice,
    )
    coeffs = np.asarray([0.2 - 0.1j, 0.5 + 0.3j, 0.5 - 0.3j, 0.2 + 0.1j])
    samples = grid.synthesize(coeffs)
    recovered = grid.project(samples)

    assert grid.mode == hb.ProjectionMode.MULTI_FUNDAMENTAL
    np.testing.assert_allclose(np.imag(samples), 0.0, atol=1e-12)
    np.testing.assert_allclose(recovered, coeffs, rtol=1e-12, atol=1e-12)


def test_multi_fundamental_projection_of_cubic_product() -> None:
    frequencies = np.asarray([-10.0, -3.123, 3.123, 10.0, 16.877])
    lattice = np.asarray([[-1, 0], [0, -1], [0, 1], [1, 0], [2, -1]])
    grid = hb.make_multi_fundamental_projection_grid(
        frequencies,
        fundamental_frequencies_hz=np.asarray([10.0, 3.123]),
        lattice_indices=lattice,
    )
    coeffs = np.zeros(frequencies.shape, dtype=np.complex128)
    coeffs[frequencies == 10.0] = 0.5
    coeffs[frequencies == -10.0] = 0.5
    coeffs[frequencies == 3.123] = 0.25
    coeffs[frequencies == -3.123] = 0.25
    projected = np.asarray(grid.project(grid.synthesize(coeffs) ** 3))

    # The 2 fp - fs product has three permutations in the cubic expansion.
    assert projected[-1] == pytest.approx(3.0 * 0.5 * 0.5 * 0.25, rel=1e-12)


def test_multi_fundamental_matches_single_period_for_commensurate_basis() -> None:
    frequencies = np.asarray([-2.0, -1.0, 1.0, 2.0])
    coeffs = np.asarray([0.1 - 0.2j, 0.5 + 0.3j, 0.5 - 0.3j, 0.1 + 0.2j])
    config = hb.HBProjectionConfig(
        n_time_samples=64,
        multi_fundamental_samples_per_axis=(64,),
    )
    single = hb.make_projection_grid(frequencies, fundamental_frequency_hz=1.0, config=config)
    torus = hb.make_multi_fundamental_projection_grid(
        frequencies,
        fundamental_frequencies_hz=np.asarray([1.0]),
        lattice_indices=np.asarray([[-2], [-1], [1], [2]]),
        config=config,
    )

    np.testing.assert_allclose(single.synthesize(coeffs), torus.synthesize(coeffs), atol=1e-12)
    np.testing.assert_allclose(single.project(single.synthesize(coeffs)), torus.project(torus.synthesize(coeffs)), atol=1e-12)


def test_multi_fundamental_projection_budget_is_enforced() -> None:
    with pytest.raises(ValueError, match="max_projection_samples"):
        hb.make_multi_fundamental_projection_grid(
            np.asarray([1.0, 2.0]),
            fundamental_frequencies_hz=np.asarray([1.0, 2.0]),
            lattice_indices=np.asarray([[1, 0], [0, 1]]),
            config=hb.HBProjectionConfig(
                multi_fundamental_samples_per_axis=(16, 16),
                max_projection_samples=128,
            ),
        )
