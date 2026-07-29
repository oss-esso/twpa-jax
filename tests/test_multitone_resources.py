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

# Measured on jtwpa S=10 (n_tones=31, full backend n=2560) with pypardiso, as
# the peak working set of a whole run_compression run (3 signal power points).
JTWPA_S10_TONES = 31
JTWPA_S10_NODES = 2560
MEASURED_PEAK_GB = 2.51
MEASURED_BANDED_PEAK_GB = 1.84
MEASURED_MATRIX_NNZ = 23_705_080
# The machine these campaigns run on can spare about this much.
CAMPAIGN_MEMORY_CEILING_GB = 7.0


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
    assert footprint.peak_gb < 1.5 * MEASURED_PEAK_GB
    assert footprint.peak_bytes > footprint.steady_bytes


def test_banded_estimate_is_conservative_and_smaller_than_sparse() -> None:
    """Both backends must estimate at or above what they actually used.

    The banded band size is fixed by geometry, so an optimistic workspace term
    shows up here first -- and underestimating is what overcommits workers.
    """
    banded = fast_coupled_footprint(
        JTWPA_S10_TONES, JTWPA_S10_NODES, factor_backend="banded"
    )
    sparse = fast_coupled_footprint(JTWPA_S10_TONES, JTWPA_S10_NODES)

    assert banded.peak_gb >= MEASURED_BANDED_PEAK_GB
    assert banded.peak_gb < 1.5 * MEASURED_BANDED_PEAK_GB
    assert banded.peak_gb < sparse.peak_gb


def test_banded_backend_admits_a_third_worker_the_sparse_one_cannot() -> None:
    """The whole point of the banded backend is the extra worker.

    Throughput scales with worker count well past two, so on a machine that can
    spare only ~7 GB the banded footprint is what turns two workers into three.
    """
    def workers(footprint) -> int:
        headroom = max(0.75, 0.30 * footprint.peak_gb)
        return max(1, int((CAMPAIGN_MEMORY_CEILING_GB - headroom) // footprint.peak_gb))

    sparse = fast_coupled_footprint(JTWPA_S10_TONES, JTWPA_S10_NODES)
    banded = fast_coupled_footprint(
        JTWPA_S10_TONES, JTWPA_S10_NODES, factor_backend="banded"
    )

    assert workers(sparse) == 2
    assert workers(banded) == 3


def test_footprint_rejects_an_unknown_factor_backend() -> None:
    with pytest.raises(ValueError, match="unknown factor backend"):
        fast_coupled_footprint(10, 100, factor_backend="klu")


def test_production_basis_leaves_room_for_more_than_one_worker() -> None:
    """The campaign machine must fit at least two jtwpa S=10 workers.

    Throughput scales with worker count until memory bandwidth saturates, so a
    per-worker footprint that admits only one worker halves the campaign rate.
    This pins the footprint against the budget that actually exists rather than
    against an abstract ratio.
    """
    footprint = fast_coupled_footprint(JTWPA_S10_TONES, JTWPA_S10_NODES)

    assert footprint.peak_gb * 2 < CAMPAIGN_MEMORY_CEILING_GB


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
