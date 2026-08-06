"""Frequency-dependent insertion-loss model for pump/signal attenuation.

The measured A10 line is the pump-line model and B1 is the signal-line model.
Both use

    att_dB(f) = c + a * sqrt(f) + b * f      (f in GHz, att in dB, positive = loss)

which is physically motivated: ``c`` is the fixed coupling/insertion loss,
``a * sqrt(f)`` is the conductor skin-effect loss, and ``b * f`` is the
dielectric loss. The default coefficients are a least-squares fit of
``docs/development/loss_A10.csv`` (RMS 0.37 dB, max 1.81 dB); evaluated in the pump band
(~8 GHz) the model returns ~35 dB, matching the old band-calibrated flat value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twpa_solver.ports import port_current_from_power_a

# Least-squares fit of docs/development/loss_A10.csv to att_dB(f) = c + a*sqrt(f) + b*f.
# f in GHz, att in dB. See tests/test_loss_model.py for the re-fit check.
LOSS_A10_C_DB = 27.3882157727
LOSS_A10_A_DB = 0.4579029666
LOSS_A10_B_DB = 0.8354288817

# Least-squares fit of docs/development/loss_B1.csv, same functional family
# (RMS 2.80e-5 dB, max 5.17e-5 dB -- this CSV is a closed form, not a noisy
# measurement). This is the SIGNAL line's calibration, distinct from A10 (the
# PUMP line). The pump line is fixed to A10 by an energy-conservation
# argument, not a fit choice: with loss_B1 on the signal line the measured
# P_sat is -64.43 dBm, so signal+idler power at compression is ~-61.4 dBm and
# the on-chip pump must exceed that -- which loss_A10 (34.54 dB @ 7.256 GHz,
# on-chip pump -55.54 dBm) satisfies and a ~45 dB or ~60 dB pump-line loss
# does not (see docs/development/psat_comparison_fix_plan.md).
LOSS_B1_C_DB = 50.0
LOSS_B1_A_DB = 3.3
LOSS_B1_B_DB = 0.14


@dataclass(frozen=True)
class InsertionLossModel:
    """Insertion loss ``att_dB(f) = c + a*sqrt(f) + b*f`` (f in GHz, att in dB)."""

    a_db: float
    b_db: float
    c_db: float = 0.0

    def attenuation_db(self, freq_ghz: float | np.ndarray) -> float | np.ndarray:
        """Attenuation in dB (positive = loss) at frequency ``freq_ghz`` (GHz)."""
        f = np.asarray(freq_ghz, dtype=float)
        if np.any(f < 0.0):
            raise ValueError("freq_ghz must be non-negative")
        att = self.c_db + self.a_db * np.sqrt(f) + self.b_db * f
        return float(att) if np.ndim(freq_ghz) == 0 else att

    def dbm_to_peak_current_a(
        self,
        power_dbm: float,
        freq_ghz: float,
        *,
        z0_ohm: float = 50.0,
        convention: str = "legacy_traveling_wave",
    ) -> float:
        """External pump power (dBm) -> on-chip peak current (A) at ``freq_ghz``.

        Applies the frequency-dependent attenuation, then inverts the matched
        wave-port available-power relation ``I_peak = sqrt(2 * P_W / Z0)``
        (see ``twpa_solver.ports``). For a fixed dBm the Norton convention's
        drive current is half this, so a map regenerated with the same dBm
        bounds under a different convention is a different physical sweep.
        """
        if z0_ohm <= 0.0:
            raise ValueError("z0_ohm must be positive")
        source_dbm = float(power_dbm) - float(self.attenuation_db(freq_ghz))
        power_w = 1.0e-3 * 10.0 ** (source_dbm / 10.0)
        return port_current_from_power_a(power_w, z0_ohm, convention=convention)

    @classmethod
    def fit_csv(
        cls,
        path: str | Path,
        *,
        freq_col: str = "Frequency_GHz",
        loss_col: str = "Insertion_Loss_dB",
    ) -> "InsertionLossModel":
        """Fit ``c + a*sqrt(f) + b*f`` to an insertion-loss CSV.

        The CSV stores insertion loss as a negative dB value (S21); attenuation
        is its magnitude. Header names default to the ``loss_A10.csv`` columns.
        """
        raw = np.genfromtxt(str(path), delimiter=",", names=True)
        freq_ghz = np.asarray(raw[freq_col], dtype=float)
        attenuation_db = -np.asarray(raw[loss_col], dtype=float)
        basis = np.column_stack(
            [np.ones_like(freq_ghz), np.sqrt(freq_ghz), freq_ghz]
        )
        coeffs, *_ = np.linalg.lstsq(basis, attenuation_db, rcond=None)
        c_db, a_db, b_db = (float(v) for v in coeffs)
        return cls(a_db=a_db, b_db=b_db, c_db=c_db)


def pump_loss_model() -> InsertionLossModel:
    """Return the measured A10 pump-line insertion-loss model."""
    return InsertionLossModel(
        a_db=LOSS_A10_A_DB, b_db=LOSS_A10_B_DB, c_db=LOSS_A10_C_DB
    )


def default_loss_model() -> InsertionLossModel:
    """Compatibility alias for :func:`pump_loss_model` (the pump line)."""
    return pump_loss_model()


def pump_line_loss_model() -> InsertionLossModel:
    """Compatibility alias for :func:`pump_loss_model`."""
    return pump_loss_model()


def signal_line_loss_model() -> InsertionLossModel:
    """The measured ``loss_B1`` insertion-loss model (frozen fit coefficients)."""
    return InsertionLossModel(
        a_db=LOSS_B1_A_DB, b_db=LOSS_B1_B_DB, c_db=LOSS_B1_C_DB
    )


def signal_loss_model() -> InsertionLossModel:
    """Return the measured B1 signal-line insertion-loss model."""
    return signal_line_loss_model()
