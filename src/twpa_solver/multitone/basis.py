"""Two-dimensional pump/signal tone bases and exact torus transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


REAL_RECONSTRUCTION_FACTOR = 2.0


@dataclass(frozen=True, order=True)
class ToneIndex:
    """Integer lattice coordinate for ``h * omega_p + q * delta``."""

    h: int
    q: int

    def omega(self, omega_p: float, delta: float) -> float:
        """Return this tone's physical angular frequency."""
        return self.h * omega_p + self.q * delta

    def conjugate(self) -> "ToneIndex":
        """Return the coordinate of the complex-conjugate tone."""
        return ToneIndex(-self.h, -self.q)

    def __add__(self, other: "ToneIndex") -> "ToneIndex":
        if not isinstance(other, ToneIndex):
            return NotImplemented
        return ToneIndex(self.h + other.h, self.q + other.q)

    def __sub__(self, other: "ToneIndex") -> "ToneIndex":
        if not isinstance(other, ToneIndex):
            return NotImplemented
        return ToneIndex(self.h - other.h, self.q - other.q)


def floquet_sideband_of(tone: ToneIndex) -> int | None:
    """Return the Floquet sideband represented by a signal-sector tone.

    The ``q=-1`` and ``q=+1`` sectors use the two conjugate frequency-folding
    conventions of the Floquet conversion problem.  Pump-only tones do not
    represent a signal sideband.
    """
    if tone.q == -1:
        return tone.h - 1
    if tone.q == 1:
        return -(tone.h + 1)
    return None


def covered_sidebands(basis: "MultiToneBasis") -> set[int]:
    """Return the Floquet sidebands represented by ``basis``."""
    return {
        sideband
        for tone in basis.tones
        if (sideband := floquet_sideband_of(tone)) is not None
    }


def canonicalize(
    tone: ToneIndex, omega_p: float, delta: float
) -> tuple[ToneIndex, bool]:
    """Return a positive-frequency representative and conjugation flag."""
    frequency = tone.omega(omega_p, delta)
    if frequency == 0.0:
        raise ValueError("DC tone (0, 0) or an exactly zero-frequency combination is out of scope")
    if frequency > 0.0:
        return tone, False
    return tone.conjugate(), True


def _next_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


@dataclass
class MultiToneBasis:
    """Positive-frequency tones and their finite, alias-safe torus grid."""

    tones: list[ToneIndex]
    omega_p: float
    delta: float
    n_p: int | None = None
    n_delta: int | None = None
    _index: dict[ToneIndex, int] = field(init=False, repr=False)
    _theta_flat: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tones = [
            tone if isinstance(tone, ToneIndex) else ToneIndex(*tone)
            for tone in self.tones
        ]
        if not self.tones:
            raise ValueError("multitone basis needs at least one tone")
        if len(set(self.tones)) != len(self.tones):
            raise ValueError("multitone basis contains duplicate tones")
        if any(tone.h == 0 and tone.q == 0 for tone in self.tones):
            raise ValueError("DC tone (0, 0) is out of scope")
        if any(tone.conjugate() in self.tones for tone in self.tones):
            raise ValueError("a tone and its conjugate cannot both be retained")
        if any(tone.omega(self.omega_p, self.delta) <= 0.0 for tone in self.tones):
            raise ValueError("all retained tones must have strictly positive frequency")

        required = {ToneIndex(1, 0), ToneIndex(1, -1), ToneIndex(1, 1)}
        missing = required.difference(self.tones)
        if missing:
            raise ValueError(f"basis is missing required tones: {sorted(missing)}")

        max_h = max(abs(tone.h + other.h) for tone in self.tones for other in self.tones)
        max_q = max(abs(tone.q + other.q) for tone in self.tones for other in self.tones)
        min_np = _next_even(2 * max_h + 1)
        min_nd = _next_even(2 * max_q + 1)
        self.n_p = min_np if self.n_p is None else int(self.n_p)
        self.n_delta = min_nd if self.n_delta is None else int(self.n_delta)
        if self.n_p < min_np or self.n_p % 2:
            raise ValueError(f"n_p must be even and >= {min_np}, got {self.n_p}")
        if self.n_delta < min_nd or self.n_delta % 2:
            raise ValueError(f"n_delta must be even and >= {min_nd}, got {self.n_delta}")

        self._index = {tone: i for i, tone in enumerate(self.tones)}
        r = np.arange(self.n_p, dtype=float)[:, None]
        s = np.arange(self.n_delta, dtype=float)[None, :]
        theta_p = 2.0 * np.pi * r / self.n_p
        theta_delta = 2.0 * np.pi * s / self.n_delta
        theta_p = np.broadcast_to(theta_p, (self.n_p, self.n_delta))
        theta_delta = np.broadcast_to(theta_delta, (self.n_p, self.n_delta))
        self._theta_flat = np.column_stack((theta_p.ravel(), theta_delta.ravel()))

    @property
    def n_tones(self) -> int:
        return len(self.tones)

    @property
    def pump_tone(self) -> ToneIndex:
        return ToneIndex(1, 0)

    @property
    def signal_tone(self) -> ToneIndex:
        return ToneIndex(1, -1)

    @property
    def idler_tone(self) -> ToneIndex:
        return ToneIndex(1, 1)

    @property
    def omegas(self) -> np.ndarray:
        return np.asarray([t.omega(self.omega_p, self.delta) for t in self.tones])

    @property
    def theta_flat(self) -> np.ndarray:
        return self._theta_flat.copy()

    def index_of(self, tone: ToneIndex) -> int:
        return self._index[tone]

    def signal_order(self, tone: ToneIndex) -> int:
        return abs(tone.q)

    def covered_sidebands(self) -> set[int]:
        """Return the Floquet sidebands represented by this basis."""
        return covered_sidebands(self)

    def to_metadata(self) -> dict[str, object]:
        return {
            "tones": [{"h": t.h, "q": t.q} for t in self.tones],
            "omega_p": self.omega_p,
            "delta": self.delta,
            "n_p": self.n_p,
            "n_delta": self.n_delta,
            "real_reconstruction_factor": REAL_RECONSTRUCTION_FACTOR,
            "phase_convention": "exp_plus_i_(h_theta_p_plus_q_theta_delta)",
        }

    def scatter_signed_spectrum(self, X: np.ndarray) -> np.ndarray:
        """Scatter positive coefficients and conjugates onto an FFT spectrum."""
        values = np.asarray(X, dtype=np.complex128)
        expected = (self.n_tones, values.shape[1] if values.ndim == 2 else -1)
        if values.ndim != 2 or values.shape[0] != self.n_tones:
            raise ValueError(f"X must have shape (n_tones, n_nodes), got {values.shape}")
        spectrum = np.zeros((self.n_p, self.n_delta, expected[1]), dtype=np.complex128)
        for row, tone in enumerate(self.tones):
            spectrum[tone.h % self.n_p, tone.q % self.n_delta] = values[row]
            conjugate = tone.conjugate()
            spectrum[conjugate.h % self.n_p, conjugate.q % self.n_delta] = np.conj(values[row])
        return spectrum

    def synthesize(self, X: np.ndarray) -> np.ndarray:
        """Synthesize ``2 Re sum(X_v exp(i theta_v))`` on the flat torus."""
        spectrum = self.scatter_signed_spectrum(X)
        waveform = (
            np.fft.ifft2(spectrum, axes=(0, 1))
            * (self.n_p * self.n_delta)
        )
        return np.asarray(waveform.real.reshape(self.n_p * self.n_delta, -1), dtype=float)

    def gather_positive_modes(self, spectrum: np.ndarray) -> np.ndarray:
        """Gather retained positive modes from a signed FFT spectrum."""
        return np.asarray(
            [spectrum[tone.h % self.n_p, tone.q % self.n_delta] for tone in self.tones],
            dtype=np.complex128,
        )

    def project(self, y: np.ndarray) -> np.ndarray:
        """Project a flat torus waveform onto the retained positive tones."""
        values = np.asarray(y)
        if values.ndim != 2 or values.shape[0] != self.n_p * self.n_delta:
            raise ValueError(
                f"y must have shape ({self.n_p * self.n_delta}, n_nodes), got {values.shape}"
            )
        spectrum = np.fft.fft2(
            values.reshape(self.n_p, self.n_delta, values.shape[1]), axes=(0, 1)
        ) / (self.n_p * self.n_delta)
        return self.gather_positive_modes(spectrum)


def build_three_tone_basis(omega_p: float, delta: float) -> MultiToneBasis:
    """Build the pump, signal, and idler basis."""
    return MultiToneBasis(
        tones=[ToneIndex(1, 0), ToneIndex(1, -1), ToneIndex(1, 1)],
        omega_p=omega_p,
        delta=delta,
    )


def build_lattice_basis(
    pump_modes: Iterable[int],
    signal_order_max: int,
    omega_p: float,
    delta: float,
    omega_max: float,
) -> MultiToneBasis:
    """Build the positive-frequency lattice clipped by ``omega_max``."""
    if signal_order_max < 1:
        raise ValueError("signal_order_max must be >= 1")
    candidates = []
    for h in pump_modes:
        for q in range(-signal_order_max, signal_order_max + 1):
            tone, _ = canonicalize(ToneIndex(int(h), q), omega_p, delta)
            if tone.omega(omega_p, delta) <= omega_max:
                candidates.append(tone)
    ordered = list(dict.fromkeys(sorted(candidates, key=lambda t: (t.h, t.q))))
    return MultiToneBasis(ordered, omega_p, delta)


def build_sideband_matched_basis(
    pump_modes: Iterable[int],
    sidebands: int,
    omega_p: float,
    delta: float,
    omega_max: float,
) -> MultiToneBasis:
    """Build a positive-frequency basis covering exactly ``[-S, S]``.

    Each signed Floquet sideband is first represented as ``(m + 1, -1)``.
    Negative physical frequencies are canonicalized to their positive
    conjugate, which naturally supplies the ``q=+1`` representatives.  Pump
    harmonics are retained in the ``q=0`` sector.

    Raises:
        ValueError: If a requested sideband is clipped or two requests fold
            onto the same positive-frequency tone.
    """
    if sidebands < 0:
        raise ValueError("sidebands must be nonnegative")
    if omega_max <= 0.0:
        raise ValueError("omega_max must be positive")

    candidates: list[tuple[ToneIndex, str]] = []
    seen_sources: dict[ToneIndex, str] = {}
    for mode in pump_modes:
        tone = ToneIndex(int(mode), 0)
        if tone.omega(omega_p, delta) <= 0.0:
            raise ValueError(f"pump tone {tone} is not positive-frequency")
        if tone.omega(omega_p, delta) > omega_max:
            raise ValueError(f"omega_max clips pump tone {tone}")
        candidates.append((tone, f"pump mode {mode}"))

    for sideband in range(-sidebands, sidebands + 1):
        signed = ToneIndex(sideband + 1, -1)
        tone, _ = canonicalize(signed, omega_p, delta)
        if tone.omega(omega_p, delta) > omega_max:
            raise ValueError(
                f"omega_max clips Floquet sideband {sideband}: {tone}"
            )
        source = f"Floquet sideband {sideband}"
        previous = seen_sources.get(tone)
        if previous is not None and previous != source:
            raise ValueError(f"sideband collision at {tone}: {previous}, {source}")
        seen_sources[tone] = source
        candidates.append((tone, source))

    ordered = list(dict.fromkeys(tone for tone, _ in candidates))
    basis = MultiToneBasis(ordered, omega_p, delta)
    expected = set(range(-sidebands, sidebands + 1))
    covered = basis.covered_sidebands()
    if covered != expected:
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        raise ValueError(
            f"sideband coverage mismatch: missing={missing}, extra={extra}"
        )
    return basis
