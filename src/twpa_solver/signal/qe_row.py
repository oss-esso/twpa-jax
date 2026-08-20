"""Full signal-output scattering row and its quantum-efficiency ratio.

``calc_qe`` normalises each output row by the total power arriving there from
every input mode, so it needs the whole row: fixing the output at the physical
signal sideband, every solved sideband must be excited in turn as the input.
This module builds that row for one signal frequency and reduces it to the
figure of merit ``qe_signal / qe_ideal_signal``.

``experiments/exp19_calcqe_validation.py`` established this construction; it
lives here so the production sweep and the experiment share one implementation.

Interpreting the ratio: ``calc_qe_ideal(S_ss)`` is exactly the quantum
efficiency of a row whose only other populated mode is an idler at the strength
unitarity requires, ``|S_si|^2 = |S_ss|^2 - 1``. A solved row that carries less
idler than that has too small a denominator and returns a ratio **above one**.
That is a unitarity diagnostic rather than an efficiency, so
:func:`signal_row_quantum_efficiency` also reports the unitarity residual
``|S_ss|^2 - |S_si|^2`` (exactly 1 for a lossless non-degenerate amplifier)
next to the ratio. Read them together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.core.linear import port_s_from_unit_current_response
from twpa_solver.signal.floquet import (
    build_signal_schur_partition,
    sideband_list,
    solve_gain_one_schur,
)
from twpa_solver.signal.quantum_efficiency import calc_qe, calc_qe_ideal


@dataclass(frozen=True)
class SignalRowQE:
    """Quantum-efficiency reduction of one signal-output scattering row."""

    sidebands_summed: int
    s_ss_abs: float
    qe_signal: float
    qe_ideal_signal: float
    qe_ratio: float
    unitarity_residual: float


def build_full_signal_row(
    *,
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    khat_off_0: sp.csr_matrix,
    omega_p: float,
    pump_freq_ghz: float,
    signal_ghz: float,
    sidebands: int,
    signal_m: int,
    source_index: int,
    out_index: int,
    source_port: int,
    out_port: int,
    z0_ohm: float,
    loss_model: str = "current_complex_c",
) -> tuple[list[int], np.ndarray]:
    """Return the sideband list and the signal-output row in the ladder basis.

    The output is fixed at ``signal_m`` and every sideband is excited as the
    input, reusing one Schur partition so each extra excitation re-solves only
    the small retained system. ``solve_gain_one_schur`` returns classical
    voltage-ratio S-parameters while ``calc_qe`` expects photon ladder
    operators, so the row is reweighted Manley-Rowe style relative to the
    signal frequency: ``S_ladder[n] = S_classical[n] * sqrt(freq[n]/freq_s)``.
    """
    ms = sideband_list(sidebands)
    if signal_m not in ms:
        raise ValueError(f"signal_m={signal_m} not in sideband set {ms}")

    schur_part = build_signal_schur_partition(
        circuit, omega_p, signal_ghz, sidebands,
        source_index, out_index, loss_model=loss_model,
    )
    other_m = next(m for m in ms if m != signal_m)

    s_classical = np.zeros(len(ms), dtype=np.complex128)
    for i, n in enumerate(ms):
        # Exciting the signal sideband reads the direct response; exciting any
        # other sideband reads that sideband's conversion into the signal
        # output, which the solver returns as the idler response.
        paired_m = other_m if n == signal_m else signal_m
        result = solve_gain_one_schur(
            circuit=circuit, khat=khat, khat_off_0=khat_off_0,
            omega_p=omega_p, signal_ghz=signal_ghz, sidebands=sidebands,
            signal_m=n, idler_m=paired_m,
            source_index=source_index, out_index=out_index,
            source_current_a=1.0, source_port=source_port,
            out_port=out_port, z0_ohm=z0_ohm, loss_model=loss_model,
            include_baselines=False, schur_part=schur_part,
        )
        response = result.vout_on if n == signal_m else result.vout_idler
        if response is None:
            raise RuntimeError(
                f"no response returned for sideband m={n} at {signal_ghz} GHz"
            )
        s_classical[i] = port_s_from_unit_current_response(
            response, source_port=source_port, out_port=out_port, z0_ohm=z0_ohm,
        )

    freqs_in = np.array([abs(signal_ghz + n * pump_freq_ghz) for n in ms])
    return ms, s_classical * np.sqrt(freqs_in / signal_ghz)


def reduce_signal_row(
    ms: list[int], s_row: np.ndarray, *, signal_m: int, idler_m: int,
) -> SignalRowQE:
    """Reduce a ladder-basis signal row to its quantum-efficiency figures."""
    sig_idx = ms.index(signal_m)
    qe_signal = float(calc_qe(s_row.reshape(1, -1))[0, sig_idx])
    qe_ideal_signal = float(calc_qe_ideal(np.array([[s_row[sig_idx]]]))[0, 0])
    power = np.abs(s_row) ** 2
    return SignalRowQE(
        sidebands_summed=len(ms),
        s_ss_abs=float(abs(s_row[sig_idx])),
        qe_signal=qe_signal,
        qe_ideal_signal=qe_ideal_signal,
        qe_ratio=(
            qe_signal / qe_ideal_signal if qe_ideal_signal > 0.0 else float("nan")
        ),
        unitarity_residual=float(power[sig_idx] - power[ms.index(idler_m)]),
    )


def signal_row_quantum_efficiency(
    *,
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    khat_off_0: sp.csr_matrix,
    omega_p: float,
    pump_freq_ghz: float,
    signal_ghz: float,
    sidebands: int,
    signal_m: int,
    idler_m: int,
    source_index: int,
    out_index: int,
    source_port: int,
    out_port: int,
    z0_ohm: float,
    loss_model: str = "current_complex_c",
) -> SignalRowQE:
    """Build the full signal row at one frequency and reduce it."""
    ms, s_row = build_full_signal_row(
        circuit=circuit, khat=khat, khat_off_0=khat_off_0, omega_p=omega_p,
        pump_freq_ghz=pump_freq_ghz, signal_ghz=signal_ghz, sidebands=sidebands,
        signal_m=signal_m, source_index=source_index, out_index=out_index,
        source_port=source_port, out_port=out_port, z0_ohm=z0_ohm,
        loss_model=loss_model,
    )
    return reduce_signal_row(ms, s_row, signal_m=signal_m, idler_m=idler_m)
