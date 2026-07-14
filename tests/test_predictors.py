"""Unit tests for the inter-cell state predictors (pure functions)."""

from __future__ import annotations

import numpy as np

from twpa_solver.pump.predictors import (
    axis_secant,
    copy_predictor,
    corner_predictor,
    plane_predictor,
    rank_candidates,
)


def _state(seed: int, shape: tuple[int, int] = (4, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_copy_returns_independent_copy() -> None:
    x = _state(0)
    out = copy_predictor(x)
    assert np.array_equal(out, x)
    out[0, 0] += 1.0
    assert out[0, 0] != x[0, 0]  # copy, not a view


def test_axis_secant_exact_on_linear_field() -> None:
    # X(t) = A + t*B is linear in t -> secant is exact.
    a, b = _state(1), _state(2)
    t0, t1, tt = 0.3, 0.5, 0.9
    x0, x1 = a + t0 * b, a + t1 * b
    pred = axis_secant(x0, x1, t0, t1, tt)
    assert pred is not None
    np.testing.assert_allclose(pred, a + tt * b, atol=1e-12)


def test_axis_secant_degenerate_returns_none() -> None:
    x = _state(3)
    assert axis_secant(x, x, 0.5, 0.5, 0.9) is None  # equal abscissa
    assert axis_secant(None, x, 0.1, 0.2, 0.3) is None  # missing neighbour


def test_corner_exact_on_bilinear_index_field() -> None:
    # X[i,j] affine in indices: base + i*di + j*dj -> corner is exact.
    base, di, dj = _state(4), _state(5), _state(6)

    def cell(i: int, j: int) -> np.ndarray:
        return base + i * di + j * dj

    pred = corner_predictor(cell(2, 1), cell(1, 2), cell(1, 1))
    assert pred is not None
    np.testing.assert_allclose(pred, cell(2, 2), atol=1e-12)


def test_corner_missing_neighbour_returns_none() -> None:
    x = _state(7)
    assert corner_predictor(x, None, x) is None


def test_plane_recovers_planted_plane() -> None:
    # X(P,f) = a0 + aP*P + af*f exactly; fit must recover value at target.
    a0, ap, af = _state(8), _state(9), _state(10)
    pts = [(-30.0, 7.5), (-28.0, 7.6), (-26.0, 7.4), (-24.0, 7.7)]
    samples = [(p, f, a0 + ap * p + af * f) for p, f in pts]
    p_t, f_t = -25.0, 7.55
    pred = plane_predictor(samples, p_t, f_t)
    assert pred is not None
    np.testing.assert_allclose(pred, a0 + ap * p_t + af * f_t, atol=1e-9)


def test_plane_too_few_or_degenerate_returns_none() -> None:
    x = _state(11)
    assert plane_predictor([(-30.0, 7.5, x), (-28.0, 7.6, x)], -25.0, 7.5) is None
    # All samples on one frequency line -> rank-deficient design.
    colinear = [(-30.0, 7.5, x), (-28.0, 7.5, x), (-26.0, 7.5, x)]
    assert plane_predictor(colinear, -25.0, 7.5) is None


def test_rank_candidates_orders_by_residual_and_drops_none() -> None:
    good, mid, bad = _state(12), _state(13), _state(14)
    target = good  # residual_fn = distance to `good`

    def residual_fn(x: np.ndarray) -> float:
        return float(np.linalg.norm(x - target))

    ranked = rank_candidates(
        {"bad": bad, "good": good, "none": None, "mid": mid},
        residual_fn,
    )
    names = [name for name, _, _ in ranked]
    assert names[0] == "good"
    assert "none" not in names
    assert ranked[0][2] == 0.0  # exact match has zero residual
    assert ranked[0][2] <= ranked[1][2] <= ranked[2][2]
