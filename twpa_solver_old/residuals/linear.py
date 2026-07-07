"""Linear frequency-domain admittance and S-parameter solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twpa_solver_old.model.topology import CircuitModel


@dataclass(frozen=True)
class LinearSParameters:
    frequency_hz: float
    y_port_s: np.ndarray
    s: np.ndarray
    metadata: dict[str, float | str]


def nodal_voltage_admittance(
    model: CircuitModel,
    frequency_hz: float,
    stiffness_h_inv: np.ndarray | None = None,
) -> np.ndarray:
    """Return nodal current/voltage admittance for exp(j omega t)."""
    omega = 2.0 * np.pi * float(frequency_hz)
    if omega <= 0.0:
        raise ValueError("frequency_hz must be positive")
    k = model.zero_signal_stiffness() if stiffness_h_inv is None else stiffness_h_inv
    return model.conductance_s + 1j * omega * model.capacitance_f + k / (1j * omega)


def port_admittance(
    model: CircuitModel,
    frequency_hz: float,
    stiffness_h_inv: np.ndarray | None = None,
) -> np.ndarray:
    """Eliminate internal nodes and return the port admittance matrix."""
    y = nodal_voltage_admittance(model, frequency_hz, stiffness_h_inv)
    ports = model.port_indices()
    internal = [idx for idx in range(model.num_nodes) if idx not in ports]
    ypp = y[np.ix_(ports, ports)]
    if not internal:
        return ypp
    ypi = y[np.ix_(ports, internal)]
    yip = y[np.ix_(internal, ports)]
    yii = y[np.ix_(internal, internal)]
    return ypp - ypi @ np.linalg.solve(yii, yip)


def y_to_s(y_port_s: np.ndarray, z0_ohm: float | np.ndarray = 50.0) -> np.ndarray:
    """Convert admittance parameters to power-wave S for real reference z0."""
    y = np.asarray(y_port_s, dtype=complex)
    if np.ndim(z0_ohm) == 0:
        z = np.eye(y.shape[0], dtype=complex) * float(z0_ohm)
    else:
        z = np.diag(np.asarray(z0_ohm, dtype=float))
    eye = np.eye(y.shape[0], dtype=complex)
    return (eye - z @ y) @ np.linalg.inv(eye + z @ y)


def solve_linear_sparameters(model: CircuitModel, frequency_hz: float) -> LinearSParameters:
    """Compute ordinary unpumped two-port S-parameters."""
    y = port_admittance(model, frequency_hz)
    z0 = np.asarray([port.z0_ohm for port in model.ports], dtype=float)
    return LinearSParameters(
        frequency_hz=float(frequency_hz),
        y_port_s=y,
        s=y_to_s(y, z0),
        metadata={"solver": "linear_frequency_domain"},
    )
