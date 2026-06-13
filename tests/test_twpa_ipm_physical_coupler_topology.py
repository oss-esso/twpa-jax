from __future__ import annotations

import numpy as np

from twpa_solver.model.ipm import IPMConfig, build_ipm_jtwpa_physical_coupler
from twpa_solver.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver.residuals.conversion import build_conversion_sparameters
from twpa_solver.residuals.linear import solve_linear_sparameters
from twpa_solver.solvers.scipy_least_squares import solve_least_squares


def test_physical_coupler_ipm_assembles_and_solves_linear_sparams() -> None:
    model = build_ipm_jtwpa_physical_coupler(IPMConfig(cells_per_line=1))
    assert len(model.ports) == 4
    assert model.metadata["coupler_model"] == "compact_coupled_inductor"
    assert model.pump_nodes == (8,)
    result = solve_linear_sparameters(model, 6e9)
    assert result.s.shape == (4, 4)
    assert np.all(np.isfinite(result.s))


def test_physical_coupler_tiny_pump_and_conversion_smoke() -> None:
    model = build_ipm_jtwpa_physical_coupler(IPMConfig(cells_per_line=1))
    residual = PumpAFTResidual(
        model,
        PumpAFTConfig(
            pump_frequency_hz=6e9,
            harmonics=1,
            source_current_peak_a=1e-9,
            residual_scale_a=1e-6,
        ),
    )
    pump = solve_least_squares(residual, np.zeros(residual.size), max_nfev=50)
    conversion = build_conversion_sparameters(
        model,
        residual,
        pump.solution,
        5e9,
        1,
        pump_success=pump.success,
        pump_status=pump.status,
    )
    assert conversion.s_conversion.shape == (12, 12)
    assert np.all(np.isfinite(conversion.s_conversion))
