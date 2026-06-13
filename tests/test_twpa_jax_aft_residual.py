from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from twpa_solver.model.ipm import IPMConfig, build_ipm_jtwpa_reduced_marker
from twpa_solver.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver.residuals.jax_aft_hb import JaxPumpAFTResidual
from twpa_solver.solvers.jax_dense_newton import solve_jax_dense_newton


def test_jax_aft_residual_matches_numpy() -> None:
    model = build_ipm_jtwpa_reduced_marker(IPMConfig(cells_per_line=1))
    config = PumpAFTConfig(
        pump_frequency_hz=6e9,
        harmonics=1,
        source_current_peak_a=1e-9,
        residual_scale_a=1e-6,
    )
    numpy_residual = PumpAFTResidual(model, config)
    jax_residual = JaxPumpAFTResidual(model, config)
    x = np.linspace(-1e-20, 1e-20, numpy_residual.size)
    np.testing.assert_allclose(np.asarray(jax_residual(jnp.asarray(x))), numpy_residual(x), rtol=1e-9)
    compiled = jax.jit(lambda value: jax_residual(value))
    np.testing.assert_allclose(np.asarray(compiled(jnp.asarray(x))), numpy_residual(x), rtol=1e-9)


def test_jax_jvp_matches_finite_difference_on_twpa_residual() -> None:
    model = build_ipm_jtwpa_reduced_marker(IPMConfig(cells_per_line=1))
    residual = JaxPumpAFTResidual(
        model,
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=1, residual_scale_a=1e-6),
    )
    x = jnp.linspace(-1e-21, 1e-21, residual.size)
    tangent = jnp.ones_like(x) * 1e-22
    _, jvp = jax.jvp(residual, (x,), (tangent,))
    eps = 1e-3
    fd = (residual(x + eps * tangent) - residual(x - eps * tangent)) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(jvp), np.asarray(fd), rtol=1e-5, atol=1e-8)


def test_jax_dense_newton_solves_tiny_twpa_zero_source() -> None:
    model = build_ipm_jtwpa_reduced_marker(IPMConfig(cells_per_line=1))
    residual = JaxPumpAFTResidual(
        model,
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=1, residual_scale_a=1e-6),
    )
    result = solve_jax_dense_newton(residual, np.zeros(residual.size), tolerance=1e-10)
    assert result.success
    np.testing.assert_allclose(result.solution, 0.0, atol=1e-14)
