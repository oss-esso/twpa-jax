from __future__ import annotations

import numpy as np

from twpa_solver_old.model.ipm import IPMConfig, build_ipm_jtwpa
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual


def test_zero_amplitude_zero_source_residual_is_zero() -> None:
    residual = PumpAFTResidual(
        build_ipm_jtwpa(IPMConfig(cells_per_line=1)),
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=2),
    )
    np.testing.assert_allclose(residual(np.zeros(residual.size)), 0.0, atol=1e-12)


def test_projection_recovers_cosine_coefficient() -> None:
    residual = PumpAFTResidual(
        build_ipm_jtwpa(IPMConfig(cells_per_line=1)),
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=2),
    )
    values = np.zeros((residual.time_samples, residual.num_nodes))
    values[:, 0] = 3.0 * np.cos(residual.theta)
    coeff = residual.project_time_to_coefficients(values).reshape(2, 2, residual.num_nodes)
    assert abs(coeff[0, 0, 0] - 3.0) < 1e-12
    assert abs(coeff[0, 1, 0]) < 1e-12
