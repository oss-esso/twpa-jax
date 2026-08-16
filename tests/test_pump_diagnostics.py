from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.pump import hb
from twpa_solver.pump.diagnostics import (
    branch_current_profile,
    branch_stress_metrics,
    residual_spectrum_summary,
)


def _problem() -> hb.FullPumpProblem:
    grid = hb.HarmonicGrid(
        modes=np.asarray([1, 3]),
        nt=16,
        omega=2.0 * math.pi,
    )
    return hb.FullPumpProblem(
        C=sp.csr_matrix([[1.0 + 0.0j]]),
        G=sp.csr_matrix([[0.01 + 0.0j]]),
        K=sp.csr_matrix([[1.0 + 0.0j]]),
        Bphi=sp.csr_matrix([[1.0]]),
        branch=hb.JosephsonBranchArray(
            Ic=np.asarray([1.0]),
            phi0=1.0,
        ),
        grid=grid,
        pump_node_index=0,
        pump_current_a=0.2,
    )


def test_branch_stress_reports_utilization_and_tangent_margin() -> None:
    problem = _problem()
    X = np.zeros((2, 1), dtype=np.complex128)
    X[0, 0] = 0.25

    metrics = branch_stress_metrics(problem, X)

    assert metrics["strongest_branch_index"] == 0
    assert 0.45 < metrics["branch_current_max_over_ic"] < 0.5
    assert metrics["branch_min_cos_phase"] < 1.0
    assert metrics["branch_min_tangent_abs"] > 0.0


def test_residual_spectrum_identifies_modes_outside_retained_basis() -> None:
    problem = _problem()
    X = np.zeros((2, 1), dtype=np.complex128)

    summary = residual_spectrum_summary(problem, X, 1.0)

    assert summary["retained_modes"] == [1, 3]
    assert isinstance(summary["dominant_omitted_modes"], list)
    assert summary["max_omitted_mode_rel"] >= 0.0


def test_branch_current_profile_preserves_one_row_per_branch() -> None:
    problem = _problem()
    X = np.zeros((2, 1), dtype=np.complex128)
    X[0, 0] = 0.25

    profile = branch_current_profile(problem, X)

    assert profile["peak_abs_current_a"].shape == (1,)
    assert profile["critical_current_a"].shape == (1,)
    assert profile["peak_ratio_ic"][0] == profile["peak_abs_current_a"][0]


def test_utilization_uses_maximum_per_branch_ratio_for_nonuniform_ic() -> None:
    grid = hb.HarmonicGrid(
        modes=np.asarray([1]), nt=16, omega=2.0 * math.pi,
    )
    problem = hb.FullPumpProblem(
        C=sp.csr_matrix([[1.0 + 0.0j]]),
        G=sp.csr_matrix([[0.01 + 0.0j]]),
        K=sp.csr_matrix([[1.0 + 0.0j]]),
        Bphi=sp.csr_matrix([[0.9, 0.5]]),
        branch=hb.JosephsonBranchArray(
            Ic=np.asarray([1.0, 2.0]), phi0=1.0,
        ),
        grid=grid,
        pump_node_index=0,
        pump_current_a=0.2,
    )
    X = np.zeros((1, 1), dtype=np.complex128)
    X[0, 0] = 0.5

    metrics = branch_stress_metrics(problem, X)

    assert metrics["branch_current_max_over_ic"] == pytest.approx(
        np.sin(0.9), rel=1e-12,
    )
