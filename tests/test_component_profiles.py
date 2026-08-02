"""Shape kernels, selectors, and the restricted custom-expression evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from twpa_solver.builders.profiles import (
    Segment,
    Selection,
    evaluate_profile,
    parse_profile_json,
    parse_profile_shorthand,
    segment_to_dict,
    segments_to_json,
)

CELLS = 60
PER_ROW = 20

ANCHORED_SHAPES = [
    ("linear", {}),
    ("power", {"exponent": 3.0}),
    ("parabola", {}),
    ("parabola", {"vertex": 0.2}),
    ("half_cosine", {}),
    ("tanh", {"sharpness": 2.0}),
]


def profile(
    segment: Segment,
    *,
    n_cells: int = CELLS,
    per_row: int | None = None,
    base: float = 1.0,
) -> np.ndarray:
    return evaluate_profile(
        [segment],
        n_cells=n_cells,
        cells_per_row=per_row if per_row is not None else PER_ROW,
        base_value=base,
    )


@pytest.mark.parametrize("shape,params", ANCHORED_SHAPES)
def test_anchored_shape_hits_both_endpoints(shape: str, params: dict[str, float]) -> None:
    values = profile(Segment(shape, 100.0, 200.0, params=params))
    assert values[0] == pytest.approx(100.0, rel=1e-12)
    assert values[-1] == pytest.approx(200.0, rel=1e-12)


def test_linear_endpoints_are_exact() -> None:
    values = profile(Segment("linear", 100e-12, 200e-12))
    assert values[0] == 100e-12
    assert values[-1] == 200e-12


def test_const_ignores_position_and_rejects_a_differing_end() -> None:
    assert np.all(profile(Segment("const", 5.0)) == 5.0)
    with pytest.raises(ValueError, match="const profiles require end"):
        profile(Segment("const", 5.0, 6.0))


def test_sine_uses_start_and_end_as_its_envelope() -> None:
    # Periodic shapes cannot be endpoint-anchored, so start/end bound the
    # oscillation instead. linspace does not land exactly on the extrema, so
    # the sampling error shrinks with cell count rather than vanishing.
    values = profile(Segment("sine", 1.0, 2.0, params={"periods": 1.0}), n_cells=2000, per_row=2000)
    assert values.min() == pytest.approx(1.0, abs=1e-5)
    assert values.max() == pytest.approx(2.0, abs=1e-5)
    assert values.mean() == pytest.approx(1.5, abs=1e-3)


def test_cosine_starts_at_its_upper_envelope() -> None:
    values = profile(Segment("cosine", 1.0, 2.0, params={"periods": 1.0}))
    assert values[0] == pytest.approx(2.0, rel=1e-12)


def test_single_cell_segment_returns_start() -> None:
    values = evaluate_profile(
        [Segment("linear", 7.0, 9.0, select=Selection(index=(3, 3)))],
        n_cells=CELLS, cells_per_row=PER_ROW, base_value=1.0,
    )
    assert values[3] == 7.0


def test_equivalent_selectors_resolve_to_the_same_cells() -> None:
    by_row = profile(Segment("const", 9.0, select=Selection(rows=(0, 0))))
    by_index = profile(Segment("const", 9.0, select=Selection(index=(0, PER_ROW - 1))))
    by_fraction = profile(
        Segment("const", 9.0, select=Selection(fraction=(0.0, PER_ROW / CELLS)))
    )
    np.testing.assert_array_equal(by_row, by_index)
    np.testing.assert_array_equal(by_row, by_fraction)
    assert np.count_nonzero(by_row == 9.0) == PER_ROW


def test_multiple_selectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="only one selector"):
        profile(Segment("const", 1.0, select=Selection(rows=(0, 0), index=(0, 1))))


@pytest.mark.parametrize(
    "selection",
    [Selection(rows=(0, 99)), Selection(index=(0, 999)), Selection(fraction=(0.0, 1.5))],
)
def test_out_of_range_selection_is_rejected(selection: Selection) -> None:
    with pytest.raises(ValueError):
        profile(Segment("const", 1.0, select=selection))


def test_per_row_domain_restarts_the_shape_in_each_row() -> None:
    segment = Segment("linear", 10.0, 20.0, domain="per_row")
    per_row = profile(segment)
    selection = profile(Segment("linear", 10.0, 20.0, domain="selection"))
    assert per_row[0] == pytest.approx(10.0)
    assert per_row[PER_ROW - 1] == pytest.approx(20.0)
    assert per_row[PER_ROW] == pytest.approx(10.0)
    assert not np.allclose(per_row, selection)


def test_per_row_and_selection_agree_for_a_constant() -> None:
    per_row = profile(Segment("const", 3.0, domain="per_row"))
    selection = profile(Segment("const", 3.0, domain="selection"))
    np.testing.assert_array_equal(per_row, selection)


def test_unknown_domain_is_rejected() -> None:
    with pytest.raises(ValueError, match="domain must be"):
        profile(Segment("const", 1.0, domain="sideways"))


def test_later_segments_overwrite_earlier_ones() -> None:
    values = evaluate_profile(
        [Segment("const", 2.0), Segment("const", 5.0, select=Selection(rows=(0, 0)))],
        n_cells=CELLS, cells_per_row=PER_ROW, base_value=1.0,
    )
    assert np.all(values[:PER_ROW] == 5.0)
    assert np.all(values[PER_ROW:] == 2.0)


def test_untouched_cells_keep_the_base_value() -> None:
    values = profile(Segment("const", 8.0, select=Selection(rows=(1, 1))), base=0.5)
    assert np.all(values[:PER_ROW] == 0.5)
    assert np.all(values[PER_ROW:2 * PER_ROW] == 8.0)


def test_non_positive_value_names_the_offending_cell() -> None:
    with pytest.raises(ValueError, match="non-positive value at cell"):
        profile(Segment("linear", 1.0, -1.0))


def test_parabola_defaults_to_the_plain_square() -> None:
    values = profile(Segment("parabola", 1.0, 2.0), n_cells=5, per_row=5)
    t = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(values, 1.0 + t**2)


def test_parabola_rejects_a_midpoint_vertex() -> None:
    with pytest.raises(ValueError, match="not 0.5"):
        profile(Segment("parabola", 1.0, 2.0, params={"vertex": 0.5}))


@pytest.mark.parametrize(
    "shape,params",
    [("power", {"exponent": 0.0}), ("tanh", {"sharpness": -1.0})],
)
def test_non_positive_shape_parameters_are_rejected(
    shape: str, params: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        profile(Segment(shape, 1.0, 2.0, params=params))


def test_unknown_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profile shape"):
        profile(Segment("spiral", 1.0, 2.0))


def test_custom_expression_supports_polynomials_and_math_calls() -> None:
    poly = profile(Segment("custom", 1.0, 2.0, expression="1-(1-t)**2"), n_cells=5, per_row=5)
    np.testing.assert_allclose(poly, [1.0, 1.4375, 1.75, 1.9375, 2.0])
    wave = profile(Segment("custom", 1.0, 2.0, expression="sin(pi*t)"), n_cells=5, per_row=5)
    assert wave[2] == pytest.approx(2.0)


def test_custom_expression_broadcasts_a_scalar() -> None:
    values = profile(Segment("custom", 1.0, 2.0, expression="0.5"), n_cells=4, per_row=4)
    np.testing.assert_allclose(values, 1.5)


@pytest.mark.parametrize(
    "expression",
    [
        "t.__class__",
        "open('x')",
        "__import__('os')",
        "[t]",
        "unknown_name * t",
        "t if t else t",
    ],
)
def test_custom_expression_rejects_escapes(expression: str) -> None:
    with pytest.raises(ValueError):
        profile(Segment("custom", 1.0, 2.0, expression=expression))


def test_custom_shape_requires_an_expression() -> None:
    with pytest.raises(ValueError, match="require expression"):
        profile(Segment("custom", 1.0, 2.0))


def test_shorthand_and_json_agree() -> None:
    shorthand = parse_profile_shorthand("rows=0-2:linear:120p->140p:periods=2")
    assert shorthand.select == Selection(rows=(0, 2))
    assert shorthand.start == pytest.approx(120e-12)
    assert shorthand.end == pytest.approx(140e-12)
    assert shorthand.params == {"periods": 2.0}
    assert shorthand == parse_profile_shorthand(
        "rows=0-2:linear:1.2e-10->1.4e-10:periods=2"
    )


def test_shorthand_reaches_domain_and_expression() -> None:
    segment = parse_profile_shorthand("all:custom:1p->2p:domain=per_row,expression=t")
    assert segment.domain == "per_row"
    assert segment.expression == "t"


def test_shorthand_rejects_a_malformed_selector() -> None:
    with pytest.raises(ValueError, match="unknown profile selector"):
        parse_profile_shorthand("layers=0:const:1p")


def test_segment_round_trips_through_its_serialized_form(tmp_path: Path) -> None:
    segments = [
        parse_profile_shorthand("rows=0-2:const:150p"),
        Segment("sine", 1.0, 2.0, select=Selection(fraction=(0.0, 0.5)),
                domain="per_row", params={"periods": 3.0}),
    ]
    spec = tmp_path / "profile.json"
    spec.write_text(json.dumps({"Lj": segments_to_json(segments), "Cg": []}),
                    encoding="utf-8")
    assert parse_profile_json(spec)["Lj"] == segments


def test_json_accepts_shorthand_strings(tmp_path: Path) -> None:
    spec = tmp_path / "profile.json"
    spec.write_text(json.dumps({"Lj": ["all:const:123.9p"]}), encoding="utf-8")
    assert parse_profile_json(spec)["Lj"] == [parse_profile_shorthand("all:const:123.9p")]


def test_segment_to_dict_omits_unset_selectors() -> None:
    assert segment_to_dict(Segment("const", 1.0))["select"] == {}
