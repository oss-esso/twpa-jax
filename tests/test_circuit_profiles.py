"""Phase 4 tests for object profiles and profiled JJ lines."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from twpa_solver.circuit import (
    Circuit,
    Constant,
    Cosine,
    Custom,
    HalfSine,
    Hann,
    Linear,
    Parabola,
    Power,
    Profile,
    Sine,
    Tanh,
)
from twpa_solver.builders.profiles import Segment, evaluate_profile


PROFILE_CASES = [
    (Constant(1.0), Segment("const", 1.0, 1.0)),
    (Linear(1.0, 3.0), Segment("linear", 1.0, 3.0)),
    (
        HalfSine(1.0, 3.0),
        Segment("custom", 1.0, 3.0, expression="sin(pi*t/2)"),
    ),
    (Hann(1.0, 3.0), Segment("half_cosine", 1.0, 3.0)),
    (
        Sine(1.0, 3.0, periods=2.0, phase=0.25),
        Segment("sine", 1.0, 3.0, params={"periods": 2.0, "phase": 0.25}),
    ),
    (
        Cosine(1.0, 3.0, periods=2.0, phase=0.25),
        Segment("cosine", 1.0, 3.0, params={"periods": 2.0, "phase": 0.25}),
    ),
    (
        Power(1.0, 3.0, exponent=2.0),
        Segment("power", 1.0, 3.0, params={"exponent": 2.0}),
    ),
    (
        Parabola(1.0, 3.0, vertex=0.2),
        Segment("parabola", 1.0, 3.0, params={"vertex": 0.2}),
    ),
    (
        Tanh(1.0, 3.0, sharpness=2.0),
        Segment("tanh", 1.0, 3.0, params={"sharpness": 2.0}),
    ),
    (
        Custom(1.0, 3.0, expression="1-(1-t)**2"),
        Segment("custom", 1.0, 3.0, expression="1-(1-t)**2"),
    ),
]


@pytest.mark.parametrize("profile, expected_segment", PROFILE_CASES)
def test_profile_objects_match_existing_segments(
    profile: Profile,
    expected_segment: Segment,
) -> None:
    actual_segment = profile.to_segment()
    assert actual_segment == expected_segment
    actual = evaluate_profile(
        [actual_segment],
        n_cells=12,
        cells_per_row=6,
        base_value=profile.start,
    )
    expected = evaluate_profile(
        [expected_segment],
        n_cells=12,
        cells_per_row=6,
        base_value=profile.start,
    )
    np.testing.assert_array_equal(actual, expected)


def test_hann_and_half_sine_are_distinct_on_the_same_inputs() -> None:
    t = np.linspace(0.0, 1.0, 9)
    half_sine = evaluate_profile(
        [HalfSine(1.0, 3.0).to_segment()],
        n_cells=9,
        cells_per_row=9,
        base_value=1.0,
    )
    hann = evaluate_profile(
        [Hann(1.0, 3.0).to_segment()],
        n_cells=9,
        cells_per_row=9,
        base_value=1.0,
    )
    np.testing.assert_array_equal(half_sine, 1.0 + 2.0 * np.sin(np.pi * t / 2.0))
    np.testing.assert_array_equal(hann, 1.0 + 2.0 * (1.0 - np.cos(np.pi * t)) / 2.0)
    assert not np.array_equal(half_sine, hann)


def test_profile_objects_are_frozen() -> None:
    profile = Linear(1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        profile.start = 3.0  # type: ignore[misc]


def test_jj_line_accepts_profile_objects() -> None:
    circuit = Circuit("profiled_line")
    path = circuit.path("signal")
    line = circuit.add_jj_line(
        path,
        cells=3,
        Lj=Linear(1e-12, 3e-12),
        Cj=Constant(5e-15),
        Cg=Linear(10e-15, 30e-15),
    )

    expected_lj = evaluate_profile(
        [Linear(1e-12, 3e-12).to_segment()],
        n_cells=3,
        cells_per_row=3,
        base_value=1e-12,
    )
    expected_cg = evaluate_profile(
        [Linear(10e-15, 30e-15).to_segment()],
        n_cells=3,
        cells_per_row=3,
        base_value=10e-15,
    )
    expected_cg[0] /= 2.0
    np.testing.assert_array_equal(
        [line.cell(index).Lj.value for index in range(3)], expected_lj
    )
    np.testing.assert_array_equal(
        [line.cell(index).Cj.value for index in range(3)], [5e-15] * 3
    )
    np.testing.assert_array_equal(
        [line.cell(index).Cg.value for index in range(3)], expected_cg
    )


def test_scalar_and_constant_profile_emit_identical_lines() -> None:
    def build(cg: float | Constant) -> list[dict[str, object]]:
        circuit = Circuit("constant_profile")
        path = circuit.path("signal")
        circuit.add_jj_line(
            path,
            cells=3,
            Lj=123.9e-12,
            Cj=145e-15,
            Cg=cg,
        )
        return [element.__dict__.copy() for element in circuit.compile().elements]

    assert build(66e-15) == build(Constant(66e-15))


def test_profile_selection_is_forwarded_to_existing_engine() -> None:
    from twpa_solver.builders.profiles import Selection

    profile = Constant(9.0, selection=Selection(index=(1, 2)))
    values = evaluate_profile(
        [profile.to_segment()],
        n_cells=4,
        cells_per_row=4,
        base_value=1.0,
    )
    np.testing.assert_array_equal(values, [1.0, 9.0, 9.0, 1.0])


def test_invalid_profile_domain_names_the_line_parameter() -> None:
    circuit = Circuit("invalid_profile")
    path = circuit.path("signal")
    with pytest.raises(ValueError, match="signal.Lj: profile domain"):
        circuit.add_jj_line(
            path,
            cells=3,
            Lj=Linear(1e-12, 2e-12, domain="invalid"),
            Cj=145e-15,
            Cg=66e-15,
        )


def test_non_positive_profile_value_names_the_line_parameter() -> None:
    circuit = Circuit("non_positive_profile")
    path = circuit.path("signal")
    with pytest.raises(ValueError, match="signal.Lj: profile produced a non-positive"):
        circuit.add_jj_line(
            path,
            cells=3,
            Lj=Linear(1e-12, -1e-12),
            Cj=145e-15,
            Cg=66e-15,
        )


def test_profiled_line_emission_is_deterministic() -> None:
    def build() -> list[dict[str, object]]:
        circuit = Circuit("deterministic_profile")
        path = circuit.path("signal")
        circuit.add_jj_line(
            path,
            cells=4,
            Lj=HalfSine(1e-12, 2e-12),
            Cj=Constant(145e-15),
            Cg=Hann(66e-15, 99e-15),
        )
        return [element.__dict__.copy() for element in circuit.compile().elements]

    assert build() == build()


def test_profile_classes_are_reexported_from_circuit() -> None:
    from twpa_solver import circuit

    assert circuit.Constant is Constant
    assert circuit.Hann is Hann
    assert circuit.HalfSine is HalfSine
