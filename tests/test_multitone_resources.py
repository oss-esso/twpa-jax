from __future__ import annotations

import pytest

from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.resources import (
    ResourceLimitExceeded,
    available_memory_gb,
    estimate,
    fast_coupled_footprint,
    guard,
)

# Measured on jtwpa S=10 (n_tones=31, full backend n=2560) with pypardiso, via
# GetProcessMemoryInfo around one assemble_real_coupled_fast + refactor + solve.
JTWPA_S10_TONES = 31
JTWPA_S10_NODES = 2560
MEASURED_STEADY_GB = 2.80
MEASURED_PEAK_GB = 3.01
MEASURED_MATRIX_NNZ = 23_705_080


class _Grid:
    n = 20


def test_resource_guard_rejects_before_allocation() -> None:
    resource = estimate(build_three_tone_basis(10.0, 1.0), _Grid(), 100, 4, "real_coupled_fast")

    with pytest.raises(ResourceLimitExceeded):
        guard(resource, 1e-12)


def test_resource_estimate_reports_dimensions() -> None:
    resource = estimate(build_three_tone_basis(10.0, 1.0), _Grid(), 5, 2, "floquet_sector")

    assert resource.n_tones == 3
    assert resource.n_torus == 36
    assert resource.matrix_dimension == 2 * 2 * 5
    assert resource.total_bytes > resource.coefficient_state_bytes


def test_fast_coupled_matrix_nnz_matches_measured_assembly() -> None:
    footprint = fast_coupled_footprint(JTWPA_S10_TONES, JTWPA_S10_NODES)

    relative_error = abs(
        footprint.matrix_nnz - MEASURED_MATRIX_NNZ
    ) / MEASURED_MATRIX_NNZ
    assert relative_error < 0.02
    assert footprint.matrix_dimension == 2 * JTWPA_S10_TONES * JTWPA_S10_NODES


def test_fast_coupled_estimate_is_conservative_against_measured_peak() -> None:
    """The estimate must never fall below what the solve actually used.

    Underestimating overcommits workers and drives the box into swap; this is
    the regression that made a 4-worker jtwpa S=10 sweep exhaust 15 GB of RAM.
    """
    footprint = fast_coupled_footprint(JTWPA_S10_TONES, JTWPA_S10_NODES)

    assert footprint.peak_gb >= MEASURED_PEAK_GB
    assert footprint.steady_gb >= MEASURED_STEADY_GB
    assert footprint.peak_gb < 1.5 * MEASURED_PEAK_GB
    assert footprint.peak_bytes > footprint.steady_bytes


def test_fast_coupled_footprint_is_quadratic_in_tones_linear_in_nodes() -> None:
    tone_ratio = (
        fast_coupled_footprint(20, 2048).matrix_nnz
        / fast_coupled_footprint(10, 2048).matrix_nnz
    )
    assert 3.9 < tone_ratio < 4.1

    small = fast_coupled_footprint(31, 1000)
    large = fast_coupled_footprint(31, 2000)
    assert large.matrix_nnz == pytest.approx(2 * small.matrix_nnz, rel=1e-9)


@pytest.mark.parametrize("tones,nodes", [(0, 100), (10, 0), (-1, 10)])
def test_fast_coupled_footprint_rejects_invalid_dimensions(
    tones: int, nodes: int
) -> None:
    with pytest.raises(ValueError):
        fast_coupled_footprint(tones, nodes)


def test_available_memory_is_positive_or_unavailable() -> None:
    free_gb = available_memory_gb()

    assert free_gb is None or free_gb > 0.0
