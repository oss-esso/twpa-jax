from __future__ import annotations

import numpy as np

from twpa_solver_old.model.ipm import IPMConfig, build_ipm_jtwpa_reduced_marker
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver_old.residuals.conversion import build_conversion_sparameters
from twpa_solver_old.residuals.linear import solve_linear_sparameters


def test_zero_pump_conversion_admittance_is_sideband_block_diagonal() -> None:
    model = build_ipm_jtwpa_reduced_marker(IPMConfig(cells_per_line=1))
    residual = PumpAFTResidual(model, PumpAFTConfig(pump_frequency_hz=6e9, harmonics=2))
    conversion = build_conversion_sparameters(
        model,
        residual,
        np.zeros(residual.size),
        5e9,
        2,
        pump_success=True,
        pump_status="converged",
    )
    sideband_count = len(conversion.sidebands)
    port_count = len(model.ports)
    y = conversion.y_conversion_s.reshape(port_count, sideband_count, port_count, sideband_count)
    for in_sideband in range(sideband_count):
        for out_sideband in range(sideband_count):
            if in_sideband != out_sideband:
                np.testing.assert_allclose(y[:, out_sideband, :, in_sideband], 0.0, atol=1e-18)


def test_zero_pump_signal_s21_matches_linear_solver_with_sidebands() -> None:
    model = build_ipm_jtwpa_reduced_marker(IPMConfig(cells_per_line=1))
    residual = PumpAFTResidual(model, PumpAFTConfig(pump_frequency_hz=6e9, harmonics=2))
    conversion = build_conversion_sparameters(
        model,
        residual,
        np.zeros(residual.size),
        5e9,
        2,
        pump_success=True,
        pump_status="converged",
    )
    linear = solve_linear_sparameters(model, 5e9)
    sideband_index = conversion.sidebands.index(0)
    input_index = 0 * len(conversion.sidebands) + sideband_index
    output_index = 1 * len(conversion.sidebands) + sideband_index
    np.testing.assert_allclose(
        conversion.s_conversion[output_index, input_index],
        linear.s[1, 0],
        rtol=1e-8,
        atol=1e-8,
    )
