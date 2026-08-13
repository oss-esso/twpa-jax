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

    @property
    def label(self) -> str:
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
