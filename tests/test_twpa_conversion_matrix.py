from __future__ import annotations

import numpy as np

from twpa_solver.model.ipm import IPMConfig, build_ipm_jtwpa
from twpa_solver.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver.residuals.conversion import build_conversion_sparameters
from twpa_solver.residuals.linear import solve_linear_sparameters


def test_zero_pump_signal_s21_matches_linear_solver() -> None:
    model = build_ipm_jtwpa(IPMConfig(cells_per_line=1))
    residual = PumpAFTResidual(model, PumpAFTConfig(pump_frequency_hz=6e9, harmonics=1))
    conversion = build_conversion_sparameters(
        model,
        residual,
        np.zeros(residual.size),
        5e9,
        0,
        pump_success=True,
        pump_status="converged",
    )
    linear = solve_linear_sparameters(model, 5e9)
    np.testing.assert_allclose(conversion.s_conversion[1, 0], linear.s[1, 0], rtol=1e-8, atol=1e-8)
