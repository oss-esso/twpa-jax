from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from twpa_solver.model.ipm import IPMConfig, build_ipm_jtwpa_reduced_marker
from twpa_solver.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver.residuals.jax_aft_hb import JaxPumpAFTResidual
from twpa_solver.solvers.jax_newton_krylov import solve_jax_newton_krylov
from twpa_solver.solvers.preconditioners import build_linear_passive_preconditioner


def test_linear_passive_preconditioner_solves_same_tiny_residual() -> None:
    model = build_ipm_jtwpa_reduced_marker(IPMConfig(cells_per_line=1))
    config = PumpAFTConfig(
        pump_frequency_hz=6e9,
        harmonics=1,
        source_current_peak_a=1e-10,
        residual_scale_a=1e-6,
    )
    numpy_residual = PumpAFTResidual(model, config)
    jax_residual = JaxPumpAFTResidual(model, config)
    preconditioner = build_linear_passive_preconditioner(numpy_residual)
    x0 = np.zeros(numpy_residual.size)

    without = solve_jax_newton_krylov(
        lambda x: jax_residual(x),
        x0,
        tolerance=1e-6,
        max_iterations=4,
    )
    with_preconditioner = solve_jax_newton_krylov(
        lambda x: jax_residual(x),
        x0,
        tolerance=1e-6,
        max_iterations=4,
        preconditioner=preconditioner,
    )
    assert with_preconditioner.residual_norm_inf <= without.residual_norm_inf + 1e-9
    np.testing.assert_allclose(
        jnp.asarray(with_preconditioner.solution),
        jnp.asarray(without.solution),
        rtol=1e-5,
        atol=1e-9,
    )
