from __future__ import annotations

import numpy as np

from twpa_solver_old.model.ipm import IPMConfig, build_ipm_jtwpa_reduced_marker
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver_old.residuals.conversion import build_conversion_sparameters


def _pump_solution_with_cos_amplitude(residual: PumpAFTResidual, amplitude_wb: float) -> np.ndarray:
    x = np.zeros(residual.size)
    coeff = x.reshape(residual.harmonics, 2, residual.num_nodes)
    coeff[0, 0, 0] = amplitude_wb
    return x


def test_small_pump_even_josephson_sideband_conversion_tends_to_zero() -> None:
    model = build_ipm_jtwpa_reduced_marker(IPMConfig(cells_per_line=1))
    residual = PumpAFTResidual(model, PumpAFTConfig(pump_frequency_hz=6e9, harmonics=3))
    small = build_conversion_sparameters(
        model,
        residual,
        _pump_solution_with_cos_amplitude(residual, 1e-19),
        5e9,
        2,
        pump_success=True,
        pump_status="converged",
    )
    smaller = build_conversion_sparameters(
        model,
        residual,
        _pump_solution_with_cos_amplitude(residual, 1e-21),
        5e9,
        2,
        pump_success=True,
        pump_status="converged",
    )
    sideband_0 = small.sidebands.index(0)
    sideband_2 = small.sidebands.index(2)
    in_signal = 0 * len(small.sidebands) + sideband_0
    out_idler = 1 * len(small.sidebands) + sideband_2
    assert abs(smaller.s_conversion[out_idler, in_signal]) < abs(
        small.s_conversion[out_idler, in_signal]
    )
