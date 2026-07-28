from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

from tests.test_multitone_schur import _problems, _settings
from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import build_lattice_basis
from twpa_solver.multitone.preconditioners import (
    FloquetSectorPreconditioner,
    resolve_multitone_preconditioner,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.resources import ResourceLimitExceeded, estimate, guard
from twpa_solver.multitone.schur import build_multitone_schur_problem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver


def test_preconditioner_resolution_defaults_to_exact() -> None:
    assert resolve_multitone_preconditioner(None) == "real_coupled_fast"
    assert resolve_multitone_preconditioner("floquet_sector") == "floquet_sector"
    with pytest.raises(ValueError):
        resolve_multitone_preconditioner("unknown")


def _q_problem(signal_order_max: int, preconditioner: str):
    circuit = CircuitMatrices(
        C=sp.eye(3, format="csr") * 1e-15,
        G=sp.eye(3, format="csr") * 1e-3,
        K=sp.csr_matrix(
            [[2e9, -1e9, 0], [-1e9, 2e9, -1e9], [0, -1e9, 2e9]]
        ),
        Bphi=sp.csr_matrix([[1.0], [0.0], [-1.0]]),
        Ic=np.array([1e-6]),
        port_to_index={1: 0, 2: 2},
    )
    basis = build_lattice_basis(
        [1], signal_order_max, 2.0e10, 1.0e9, 3.0e10
    )
    source = MultiToneDrive(basis.pump_tone, 0, 1e-9).to_coeffs(basis, 3)
    full = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(source)
    )
    return build_multitone_schur_problem(
        full, [0, 2], preconditioner=preconditioner
    )


@pytest.mark.parametrize("signal_order_max", (1, 2, 3))
def test_floquet_sector_tracks_exact_for_each_q(
    signal_order_max: int,
) -> None:
    exact_problem = _q_problem(signal_order_max, "real_coupled_fast")
    sector_problem = _q_problem(signal_order_max, "floquet_sector")
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
    np.testing.assert_allclose(
        sector_state, exact_state, rtol=1e-8, atol=1e-18
    )
    assert all(report.gmres_iterations_total <= 2 for report in exact_reports)
    for exact, sector in zip(exact_reports, sector_reports, strict=True):
        assert sector.gmres_iterations_total <= 3 * max(
            exact.gmres_iterations_total, 1
        )

    telemetry = FloquetSectorPreconditioner(sector_problem)
    telemetry.refactor(sector_problem.tangent_state(sector_state))
    assert telemetry.last_factor_backend == "superlu"
    assert telemetry.last_assembly_runtime_s > 0.0
    assert telemetry.last_factor_runtime_s > 0.0


def test_resource_guard_rejects_oversized_sector_request() -> None:
    full = _problems()[0]
    resource = estimate(full.basis, full.grid, 10_000, full.nb, "floquet_sector")
    with pytest.raises(ResourceLimitExceeded):
        guard(resource, 1e-12)


def test_resource_estimate_accepts_supported_spectral_preconditioner() -> None:
    full = _problems()[0]
    resource = estimate(full.basis, full.grid, 10, full.nb, "spectral_coupled")
    assert resource.preconditioner == "spectral_coupled"
