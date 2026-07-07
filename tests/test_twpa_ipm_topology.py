from __future__ import annotations

import numpy as np

from twpa_solver_old.model.ipm import IPMConfig, build_ipm_jtwpa
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver_old.solvers.scipy_least_squares import solve_least_squares


def test_ipm_topology_assembles_matrices_and_ports() -> None:
    model = build_ipm_jtwpa(IPMConfig(cells_per_line=2))
    assert model.capacitance_f.shape == (5, 5)
    assert len(model.ports) == 2
    assert model.josephson_incidence.shape[1] == 4
    assert model.pump_nodes == (1, 2)


def test_tiny_ipm_smoke_solve_returns_status_metadata() -> None:
    model = build_ipm_jtwpa(IPMConfig(cells_per_line=1))
    residual = PumpAFTResidual(
        model,
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=1, source_current_peak_a=1e-8),
    )
    result = solve_least_squares(residual, np.zeros(residual.size), max_nfev=40)
    assert result.status in {"converged", "diagnostic"}
    assert np.isfinite(result.residual_norm_inf)
