"""Reproducibility, stream independence, and relative-to-nominal semantics."""

from __future__ import annotations

import numpy as np
import pytest

from twpa_solver.builders.scatter import (
    ScatterSpec,
    apply_scatter,
    component_rng,
    draw_factors,
)

N = 500


def factors(component: str, sigma: float, seed: int = 1, **kwargs: float) -> np.ndarray:
    return draw_factors(ScatterSpec(sigma, **kwargs), N, component_rng(seed, component))


def test_zero_sigma_leaves_every_value_untouched() -> None:
    nominal = np.linspace(1.0, 2.0, N)
    values, meta = apply_scatter(nominal, ScatterSpec(0.0), component_rng(1, "Lj"))
    np.testing.assert_array_equal(values, nominal)
    assert meta["factor_min"] == 1.0 and meta["factor_max"] == 1.0


def test_same_seed_reproduces_and_a_different_seed_does_not() -> None:
    np.testing.assert_array_equal(factors("Lj", 0.02, 3), factors("Lj", 0.02, 3))
    assert not np.allclose(factors("Lj", 0.02, 3), factors("Lj", 0.02, 4))


def test_lj_stream_matches_the_legacy_generator() -> None:
    # The pre-profile builder drew straight from default_rng(seed); keeping that
    # stream means an existing scattered design regenerates bit-for-bit.
    expected = np.clip(np.random.default_rng(5).normal(1.0, 0.01, N), 0.5, 1.5)
    np.testing.assert_array_equal(factors("Lj", 0.01, 5), expected)


def test_component_streams_are_mutually_independent() -> None:
    lj, cj, cg = (factors(name, 0.01, 11) for name in ("Lj", "Cj", "Cg"))
    assert not np.allclose(lj, cj)
    assert not np.allclose(lj, cg)
    assert not np.allclose(cj, cg)


def test_unknown_component_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scatter component"):
        component_rng(1, "Ll")


def test_clipping_is_enforced_and_counted() -> None:
    spec = ScatterSpec(1.0, clip_min=0.9, clip_max=1.1)
    values, meta = apply_scatter(np.ones(N), spec, component_rng(1, "Lj"))
    assert values.min() >= 0.9 and values.max() <= 1.1
    assert meta["clip_hits"] > 0


def test_sigma_is_relative_to_each_cell_nominal() -> None:
    # A steep nominal ramp makes the absolute spread large while the relative
    # spread must stay at sigma -- that is what "percentage of nominal" means.
    nominal = np.linspace(100e-12, 200e-12, N)
    sigma = 0.02
    values, _ = apply_scatter(nominal, ScatterSpec(sigma), component_rng(1, "Lj"))
    assert np.std(values / nominal) == pytest.approx(sigma, rel=0.15)
    # Cells at twice the nominal must absorb twice the absolute deviation; a
    # global sigma would instead give the same deviation everywhere.
    deviation = np.abs(values - nominal)
    assert deviation[-100:].mean() == pytest.approx(2.0 * deviation[:100].mean(), rel=0.3)


def test_uniform_distribution_matches_the_requested_standard_deviation() -> None:
    drawn = draw_factors(
        ScatterSpec(0.05, "uniform"), 200_000, component_rng(1, "Lj")
    )
    assert drawn.std() == pytest.approx(0.05, rel=0.05)


def test_unknown_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scatter distribution"):
        draw_factors(ScatterSpec(0.01, "cauchy"), 10, component_rng(1, "Lj"))


@pytest.mark.parametrize(
    "spec", [ScatterSpec(-0.1), ScatterSpec(0.1, clip_min=0.0), ScatterSpec(0.1, clip_min=2.0)]
)
def test_invalid_specs_are_rejected(spec: ScatterSpec) -> None:
    with pytest.raises(ValueError):
        draw_factors(spec, 10, component_rng(1, "Lj"))


def test_digest_tracks_the_realized_factors() -> None:
    _, first = apply_scatter(np.ones(N), ScatterSpec(0.01), component_rng(1, "Lj"))
    _, same = apply_scatter(np.ones(N), ScatterSpec(0.01), component_rng(1, "Lj"))
    _, other = apply_scatter(np.ones(N), ScatterSpec(0.01), component_rng(2, "Lj"))
    assert first["factor_digest"] == same["factor_digest"]
    assert first["factor_digest"] != other["factor_digest"]


def test_empty_array_reports_no_statistics() -> None:
    values, meta = apply_scatter(np.array([]), ScatterSpec(0.01), component_rng(1, "Lj"))
    assert values.size == 0
    assert meta["factor_min"] is None
