"""Frozen profile objects for deterministic per-cell circuit parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from twpa_solver.builders.profiles import Segment, Selection, evaluate_profile


@dataclass(frozen=True)
class Profile:
    """Base value range and domain shared by all circuit profiles."""

    start: float
    stop: float | None = None
    domain: str = "selection"
    selection: Selection | None = None
    _shape: ClassVar[str] = "const"

    def _parameters(self) -> dict[str, float]:
        """Return shape parameters for the wrapped segment."""

        return {}

    def _expression(self) -> str | None:
        """Return the custom expression for the wrapped segment."""

        return None

    def to_segment(self) -> Segment:
        """Convert this object to the existing profile-engine segment."""

        end = self.start if self.stop is None else self.stop
        return Segment(
            shape=self._shape,
            start=float(self.start),
            end=float(end),
            select=self.selection or Selection(),
            domain=self.domain,
            params=self._parameters(),
            expression=self._expression(),
        )


@dataclass(frozen=True)
class Constant(Profile):
    """A constant profile; ``Constant(value)`` is the scalar equivalent."""

    _shape: ClassVar[str] = "const"


@dataclass(frozen=True)
class Linear(Profile):
    """A linearly interpolated profile from ``start`` to ``stop``."""

    _shape: ClassVar[str] = "linear"


@dataclass(frozen=True)
class HalfSine(Profile):
    """The distinct ``sin(pi*t/2)`` profile used by existing YAML designs."""

    _shape: ClassVar[str] = "custom"

    def _expression(self) -> str:
        """Return the existing half-sine expression."""

        return "sin(pi*t/2)"


@dataclass(frozen=True)
class Hann(Profile):
    """The fabrication Hann profile mapped to the existing half-cosine shape.

    This is deliberately different from :class:`HalfSine`.
    """

    _shape: ClassVar[str] = "half_cosine"


@dataclass(frozen=True)
class Sine(Profile):
    """A periodic sine envelope using the existing profile engine."""

    periods: float = 1.0
    phase: float = 0.0
    _shape: ClassVar[str] = "sine"

    def _parameters(self) -> dict[str, float]:
        """Return periodic sine parameters."""

        return {"periods": self.periods, "phase": self.phase}


@dataclass(frozen=True)
class Cosine(Profile):
    """A periodic cosine envelope using the existing profile engine."""

    periods: float = 1.0
    phase: float = 0.0
    _shape: ClassVar[str] = "cosine"

    def _parameters(self) -> dict[str, float]:
        """Return periodic cosine parameters."""

        return {"periods": self.periods, "phase": self.phase}


@dataclass(frozen=True)
class Power(Profile):
    """A power-law profile with a positive exponent."""

    exponent: float = 1.0
    _shape: ClassVar[str] = "power"

    def _parameters(self) -> dict[str, float]:
        """Return the power exponent."""

        return {"exponent": self.exponent}


@dataclass(frozen=True)
class Parabola(Profile):
    """A parabola profile using the existing vertex convention."""

    vertex: float = 0.0
    _shape: ClassVar[str] = "parabola"

    def _parameters(self) -> dict[str, float]:
        """Return the parabola vertex."""

        return {"vertex": self.vertex}


@dataclass(frozen=True)
class Tanh(Profile):
    """A normalized hyperbolic-tangent profile."""

    sharpness: float = 1.0
    _shape: ClassVar[str] = "tanh"

    def _parameters(self) -> dict[str, float]:
        """Return the tanh sharpness."""

        return {"sharpness": self.sharpness}


@dataclass(frozen=True)
class Custom(Profile):
    """A profile backed by the existing restricted expression evaluator."""

    expression: str = ""
    _shape: ClassVar[str] = "custom"

    def _expression(self) -> str:
        """Return the configured custom expression."""

        return self.expression


__all__ = [
    "Constant",
    "Cosine",
    "Custom",
    "HalfSine",
    "Hann",
    "Linear",
    "Parabola",
    "Power",
    "Profile",
    "Selection",
    "Sine",
    "Tanh",
    "evaluate_profile",
]
