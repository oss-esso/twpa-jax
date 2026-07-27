from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.test_multitone_schur import _problems, _settings
from twpa_solver.multitone.preconditioners import resolve_multitone_preconditioner
from twpa_solver.multitone.preconditioners import FloquetSectorPreconditioner
from twpa_solver.multitone.resources import ResourceLimitExceeded, estimate, guard
from twpa_solver.pump import HarmonicNewtonKrylovSolver


def test_preconditioner_resolution_defaults_to_exact() -> None:
    assert resolve_multitone_preconditioner(None) == "real_coupled_fast"
    assert resolve_multitone_preconditioner("floquet_sector") == "floquet_sector"
    with pytest.raises(ValueError):
        resolve_multitone_preconditioner("unknown")


def test_floquet_sector_reaches_exact_root_and_has_real_telemetry() -> None:
    full, exact_problem = _problems()
    _, sector_problem = _problems()
    sector_problem.preconditioner = "floquet_sector"
    solver = HarmonicNewtonKrylovSolver(
        dataclasses.replace(_settings(), preconditioner="real_coupled_fast")
    )
    exact_state, exact_reports = solver.solve_continuation(
        exact_problem, continuation_steps=4
    )
    sector_state, sector_reports = solver.solve_continuation(
        sector_problem, continuation_steps=4
    )

    assert exact_reports[-1].converged
    assert sector_reports[-1].converged
    np.testing.assert_allclose(sector_state, exact_state, rtol=1e-8, atol=1e-18)
    assert all(report.gmres_iterations_total <= 2 for report in exact_reports)
    assert sector_reports[-1].gmres_iterations_total <= 3 * max(
        exact_reports[-1].gmres_iterations_total, 1
    )
    telemetry = FloquetSectorPreconditioner(sector_problem)
    telemetry.refactor(sector_problem.tangent_state(sector_state))
    assert telemetry.last_factor_backend == "superlu"
    assert telemetry.last_assembly_runtime_s > 0.0
    assert telemetry.last_factor_runtime_s > 0.0


def test_resource_guard_rejects_oversized_sector_request() -> None:
    full, _ = _problems()
    resource = estimate(full.basis, full.grid, 10_000, full.nb, "floquet_sector")
    with pytest.raises(ResourceLimitExceeded):
        guard(resource, 1e-12)


def test_resource_estimate_accepts_supported_spectral_preconditioner() -> None:
    full, _ = _problems()
    resource = estimate(full.basis, full.grid, 10, full.nb, "spectral_coupled")
    assert resource.preconditioner == "spectral_coupled"
