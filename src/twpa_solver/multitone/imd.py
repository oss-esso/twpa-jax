"""Intermodulation product tone bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex, canonicalize


@dataclass(frozen=True)
class ImProduct:
    """One intermodulation product and the tone that carries it."""

    order: int
    m: int
    n: int
    raw: ToneIndex
    tone: ToneIndex
    conjugated: bool
    family: str = "single_tone"
    q1: int | None = None
    q2: int | None = None
    ordering: str | None = None

    @property
    def label(self) -> str:
        if self.family == "two_tone":
            ordering = "w1w2" if self.ordering == "w1_minus_w2" else "w2w1"
            return f"imd2_o{self.order}_m{self.m}n{self.n}_{ordering}"
        return f"imd_o{self.order}_m{self.m}n{self.n}"


def enumerate_im_products(max_order: int, omega_p: float, delta: float) -> list[ImProduct]:
    """Return every observable odd-order IM product through ``max_order``."""
    if max_order < 3 or max_order % 2 == 0:
        raise ValueError(f"max_order must be an odd integer >= 3, got {max_order}")
    products: list[ImProduct] = []
    for order in range(3, max_order + 1, 2):
        for m in range(1, order):
            n = order - m
            raw = ToneIndex(m - n, -m)
            try:
                tone, conjugated = canonicalize(raw, omega_p, delta)
            except ValueError:
                continue
            products.append(ImProduct(order, m, n, raw, tone, conjugated))
    return products


def _validate_two_tone_placement(max_order: int, q1: int, q2: int) -> None:
    if q1 == q2:
        raise ValueError("two-tone indices must be distinct")
    if q1 == 0 or q2 == 0:
        raise ValueError("two-tone fundamentals must not occupy q=0")
    if q1 + q2 == 0:
        raise ValueError(
            "degenerate four-wave-mixing placement rejected: q1 + q2 must not be zero"
        )
    fundamentals = {
        ToneIndex(1, q1),
        ToneIndex(1, q2),
        ToneIndex(-1, -q1),
        ToneIndex(-1, -q2),
    }
    for order in range(3, max_order + 1, 2):
        m = (order + 1) // 2
        n = (order - 1) // 2
        for left, right in ((q1, q2), (q2, q1)):
            q = m * left - n * right
            if q == 0:
                raise ValueError(
                    f"two-tone product collides with the pump at q=0 at order {order}"
                )
            raw = ToneIndex(1, q)
            if q in {q1, q2, -q1, -q2} or raw in fundamentals:
                raise ValueError(
                    "two-tone product collides with a fundamental or its conjugate"
                )


def enumerate_two_tone_im_products(
    max_order: int, q1: int, q2: int
) -> list[ImProduct]:
    """Return odd-order products for two commensurate signal fundamentals.

    The returned raw tones use ``h=1`` and the supplied signal-sector index.
    The caller supplies ``delta`` when constructing the physical basis, so the
    same coordinates can be checked against both signal frequencies.
    """
    if max_order < 3 or max_order % 2 == 0:
        raise ValueError(f"max_order must be an odd integer >= 3, got {max_order}")
    _validate_two_tone_placement(max_order, int(q1), int(q2))
    products: list[ImProduct] = []
    for order in range(3, max_order + 1, 2):
        m = (order + 1) // 2
        n = (order - 1) // 2
        for ordering, left, right in (
            ("w1_minus_w2", int(q1), int(q2)),
            ("w2_minus_w1", int(q2), int(q1)),
        ):
            raw = ToneIndex(1, m * left - n * right)
            products.append(
                ImProduct(
                    order=order,
                    m=m,
                    n=n,
                    raw=raw,
                    tone=raw,
                    conjugated=False,
                    family="two_tone",
                    q1=int(q1),
                    q2=int(q2),
                    ordering=ordering,
                )
            )
    return products


def required_new_tones(products: list[ImProduct], basis: MultiToneBasis) -> list[ToneIndex]:
    """Return product tones absent from ``basis`` and its conjugates."""
    have = set(basis.tones)
    conjugates = {tone.conjugate() for tone in basis.tones}
    wanted = {p.tone for p in products if p.tone not in have and p.tone not in conjugates}
    return sorted(wanted, key=lambda tone: (tone.h, tone.q))


def extend_basis_with_im_tones(basis: MultiToneBasis, products: list[ImProduct]) -> MultiToneBasis:
    """Return ``basis`` extended with every unrepresented product tone."""
    new_tones = required_new_tones(products, basis)
    if not new_tones:
        return basis
    return MultiToneBasis(
        tones=list(basis.tones) + new_tones,
        omega_p=basis.omega_p,
        delta=basis.delta,
        pump_tone_index=basis.pump_tone_index,
    )
