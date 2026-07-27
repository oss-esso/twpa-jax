"""Observable quantities extracted from finite-signal multitone states."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from twpa_solver.core.linear import (
    dynamic_block,
    port_s_from_unit_current_response,
    port_waves,
)
from twpa_solver.multitone.basis import REAL_RECONSTRUCTION_FACTOR


def _port_current_coefficients(X_full, basis, circuit):
    """Return physical port currents from the multitone KCL residual."""
    waveform = basis.synthesize(X_full)
    phase = (circuit.Bphi.T @ waveform.T).T
    nonlinear_time = circuit.Bphi @ (
        circuit.Ic[None, :] * np.sin(phase / circuit.phi0)
    ).T
    nonlinear = basis.project(nonlinear_time.T)
    currents = np.empty_like(X_full)
    for row, omega in enumerate(basis.omegas):
        linear = dynamic_block(circuit, float(omega)) @ X_full[row]
        currents[row] = REAL_RECONSTRUCTION_FACTOR * (linear + nonlinear[row])
    return currents


def extract_port_waves(
    X_full: np.ndarray,
    basis,
    circuit,
    ports: list[int] | tuple[int, ...],
    z0_ohm: float = 50.0,
) -> dict[str, Any]:
    """Extract voltage and power-wave values for every retained tone/port."""
    values: dict[str, Any] = {"a": {}, "b": {}, "a_power": {}, "b_power": {}}
    currents = _port_current_coefficients(X_full, basis, circuit)
    for row, tone in enumerate(basis.tones):
        omega = float(basis.omegas[row])
        for port in ports:
            node = circuit.port_to_index[int(port)]
            voltage = (
                REAL_RECONSTRUCTION_FACTOR
                * 1j
                * omega
                * X_full[row, node]
            )
            current = currents[row, node]
            a, b = port_waves(voltage, current, z0_ohm)
            key = (tone, int(port))
            values["a"][key] = complex(a)
            values["b"][key] = complex(b)
            values["a_power"][key] = float(abs(a) ** 2)
            values["b_power"][key] = float(abs(b) ** 2)
    return values


def tone_s21(
    X_full: np.ndarray,
    basis,
    circuit,
    *,
    signal_tone,
    source_port: int,
    out_port: int,
    source_current_a: float,
    z0_ohm: float = 50.0,
) -> complex:
    """Return paper-normalized S21 for one tone."""
    if source_current_a == 0.0:
        raise ValueError("source_current_a must be nonzero")
    row = basis.index_of(signal_tone)
    voltage = (
        REAL_RECONSTRUCTION_FACTOR
        * 1j
        * basis.omegas[row]
        * X_full[row, circuit.port_to_index[out_port]]
    )
    return port_s_from_unit_current_response(
        voltage / source_current_a,
        source_port=source_port,
        out_port=out_port,
        z0_ohm=z0_ohm,
    )


def junction_diagnostics(X_full: np.ndarray, basis, circuit) -> list[dict[str, float]]:
    """Return finite-torus phase and cosine diagnostics per junction."""
    waveform = basis.synthesize(X_full)
    phase = (circuit.Bphi.T @ waveform.T).T / circuit.phi0
    return [
        {
            "max_abs_phase": float(np.max(np.abs(phase[:, branch]))),
            "max_abs_sine": float(np.max(np.abs(np.sin(phase[:, branch])))),
            "min_cosine": float(np.min(np.cos(phase[:, branch]))),
        }
        for branch in range(phase.shape[1])
    ]


def power_balance(X_full: np.ndarray, basis, circuit) -> dict[str, float]:
    """Compute a conservative stored/dissipated power diagnostic."""
    waveform = basis.synthesize(X_full)
    dissipation = float(np.mean(np.sum(waveform * (waveform @ circuit.G.T), axis=1)))
    total = float(np.mean(np.sum(waveform * (waveform @ circuit.K.T), axis=1)))
    relative = abs(dissipation) / max(abs(total), 1e-30)
    return {
        "dissipated_power": dissipation,
        "stored_linear_energy_proxy": total,
        "power_balance_rel_err": relative,
    }


def reference_states(
    *,
    pump_off_signal_on: np.ndarray | None = None,
    pump_on_signal_infinitesimal: np.ndarray | None = None,
    pump_on_signal_finite: np.ndarray | None = None,
    pump_on_signal_off: np.ndarray | None = None,
) -> dict[str, np.ndarray | None]:
    """Package the four reference states used by normalized compression output."""
    return {
        "pump_off_signal_on": pump_off_signal_on,
        "pump_on_signal_infinitesimal": pump_on_signal_infinitesimal,
        "pump_on_signal_finite": pump_on_signal_finite,
        "pump_on_signal_off": pump_on_signal_off,
    }
