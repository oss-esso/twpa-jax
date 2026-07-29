"""Observable quantities extracted from finite-signal multitone states."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from twpa_solver.core import CircuitMatrices
from twpa_solver.core.linear import (
    dynamic_block,
    port_s_from_unit_current_response,
    port_waves,
)
from twpa_solver.multitone.basis import (
    REAL_RECONSTRUCTION_FACTOR,
    MultiToneBasis,
    ToneIndex,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath


def _port_current_coefficients(
    X_full: np.ndarray, basis: MultiToneBasis, circuit: CircuitMatrices
) -> np.ndarray:
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
    basis: MultiToneBasis,
    circuit: CircuitMatrices,
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
            # The KCL value is the Norton source current.  The port resistor
            # itself carries V/Z0, so the current entering the network is the
            # source current minus that resistor current.
            current = currents[row, node] - voltage / z0_ohm
            a, b = port_waves(voltage, current, z0_ohm)
            key = (tone, int(port))
            values["a"][key] = complex(a)
            values["b"][key] = complex(b)
            values["a_power"][key] = float(abs(a) ** 2)
            values["b_power"][key] = float(abs(b) ** 2)
    return values


def tone_s21(
    X_full: np.ndarray,
    basis: MultiToneBasis,
    circuit: CircuitMatrices,
    *,
    signal_tone: ToneIndex,
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


def junction_diagnostics(
    X_full: np.ndarray, basis: MultiToneBasis, circuit: CircuitMatrices
) -> list[dict[str, float]]:
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


def power_balance(
    X_full: np.ndarray, basis: MultiToneBasis, circuit: CircuitMatrices
) -> dict[str, float]:
    """Compute real-power and photon-flux balance diagnostics."""
    waveform = basis.synthesize(X_full)
    derivative = np.zeros_like(waveform)
    for row, tone in enumerate(basis.tones):
        coefficient = np.zeros_like(X_full)
        coefficient[row] = 1j * basis.omegas[row] * X_full[row]
        derivative += basis.synthesize(coefficient)
    acceleration = np.zeros_like(waveform)
    for row, tone in enumerate(basis.tones):
        coefficient = np.zeros_like(X_full)
        coefficient[row] = -(basis.omegas[row] ** 2) * X_full[row]
        acceleration += basis.synthesize(coefficient)
    phase = (circuit.Bphi.T @ waveform.T).T
    nonlinear = circuit.Bphi @ (
        circuit.Ic[None, :] * np.sin(phase / circuit.phi0)
    ).T
    internal = (
        (circuit.C @ acceleration.T).T
        + (circuit.G @ derivative.T).T
        + (circuit.K @ waveform.T).T
        + nonlinear.T
    )
    supplied_power = float(np.mean(np.sum(derivative * internal, axis=1)))
    dissipation = float(
        np.mean(np.sum(derivative * (circuit.G @ derivative.T).T, axis=1))
    )
    real_power_scale = max(abs(supplied_power), abs(dissipation))
    relative = (
        0.0
        if real_power_scale == 0.0
        else abs(supplied_power - dissipation) / real_power_scale
    )
    nonlinear_coeffs = basis.project(nonlinear.T)
    photon_terms = []
    for row, omega in enumerate(basis.omegas):
        power = 2.0 * np.real(
            (1j * omega * X_full[row]) @ np.conj(nonlinear_coeffs[row])
        )
        photon_terms.append(float(np.sum(power) / omega))
    photon_flux = float(np.sum(photon_terms))
    photon_scale = float(np.sum(np.abs(photon_terms)))
    return {
        "supplied_power": supplied_power,
        "dissipated_power": dissipation,
        "power_balance_rel_err": relative,
        "manley_rowe_photon_flux": photon_flux,
        "manley_rowe_rel_err": abs(photon_flux) / max(photon_scale, 1e-30),
    }


def _chain_branch_nodes(circuit: CircuitMatrices) -> tuple[np.ndarray, np.ndarray]:
    incidence = circuit.Bphi.tocsc()
    starts: list[int] = []
    stops: list[int] = []
    for branch in range(incidence.shape[1]):
        nodes = incidence.indices[
            incidence.indptr[branch] : incidence.indptr[branch + 1]
        ]
        if len(nodes) != 2:
            raise ValueError(
                "spatial_profiles requires a two-node chain incidence; "
                f"branch {branch} touches nodes {nodes.tolist()}"
            )
        starts.append(int(min(nodes)))
        stops.append(int(max(nodes)))
    starts_array = np.asarray(starts, dtype=int)
    stops_array = np.asarray(stops, dtype=int)
    monotone = np.all(np.diff(starts_array) >= 0) and np.all(
        np.diff(stops_array) > 0
    )
    if not monotone:
        raise ValueError(
            "spatial_profiles assumes Bphi branches are chain-monotone in "
            "cell order; observed endpoints="
            f"{list(zip(starts, stops))[:8]}"
        )
    return starts_array, stops_array


def spatial_profiles(
    X_full: np.ndarray,
    basis: MultiToneBasis,
    circuit: CircuitMatrices,
) -> list[dict[str, float | int]]:
    """Return branch-resolved three-wave amplitudes and effective mismatch.

    Branch index is treated as cell index only after validating that ``Bphi``
    is an ordered, connected two-node chain.
    """
    starts, stops = _chain_branch_nodes(circuit)
    branch_flux = np.asarray(circuit.Bphi.T @ np.asarray(X_full).T).T
    pump = branch_flux[basis.index_of(basis.pump_tone)]
    signal = branch_flux[basis.index_of(basis.signal_tone)]
    idler = branch_flux[basis.index_of(basis.idler_tone)]
    theta = np.unwrap(
        2.0 * np.angle(pump) - np.angle(signal) - np.angle(idler)
    )
    delta_k = np.gradient(theta) if theta.size > 1 else np.zeros_like(theta)
    return [
        {
            "branch_index": branch,
            "start_node": int(starts[branch]),
            "stop_node": int(stops[branch]),
            "pump_flux_abs": float(abs(pump[branch])),
            "signal_flux_abs": float(abs(signal[branch])),
            "idler_flux_abs": float(abs(idler[branch])),
            "theta_rad": float(theta[branch]),
            "delta_k_eff_rad_per_cell": float(delta_k[branch]),
        }
        for branch in range(circuit.branch_count)
    ]


def reference_states(
    *,
    problem: object | None = None,
    pump_source: np.ndarray | None = None,
    signal_source: np.ndarray | None = None,
    finite_signal_current_a: float | None = None,
    infinitesimal_signal_current_a: float = 1.0e-12,
    solver: object | None = None,
    pump_seed: np.ndarray | None = None,
    pump_off_signal_on: np.ndarray | None = None,
    pump_on_signal_infinitesimal: np.ndarray | None = None,
    pump_on_signal_finite: np.ndarray | None = None,
    pump_on_signal_off: np.ndarray | None = None,
) -> dict[str, np.ndarray | None]:
    """Solve or package the four states used by normalized compression output."""
    if problem is not None:
        if any(
            value is None
            for value in (
                pump_source,
                signal_source,
                finite_signal_current_a,
                solver,
                pump_seed,
            )
        ):
            raise ValueError(
                "solving reference states requires pump_source, signal_source, "
                "finite_signal_current_a, solver, and pump_seed"
            )

        def with_path(path: AffineSourcePath) -> object:
            if isinstance(problem, FullMultiToneProblem):
                return replace(problem, source_path=path)
            full = replace(problem.full, source_path=path)
            return type(problem)(
                full,
                problem.partition,
                linear_apply_mode=problem.linear_apply_mode,
            )

        zero = np.zeros_like(pump_seed)
        paths_and_seeds = {
            "pump_off_signal_on": (
                AffineSourcePath.signal_turn_on(
                    np.zeros_like(pump_source),
                    signal_source * float(finite_signal_current_a),
                ),
                zero,
            ),
            "pump_on_signal_off": (
                AffineSourcePath.pump_turn_on(pump_source),
                pump_seed,
            ),
            "pump_on_signal_infinitesimal": (
                AffineSourcePath.signal_turn_on(
                    pump_source,
                    signal_source * float(infinitesimal_signal_current_a),
                ),
                pump_seed,
            ),
        }
        solved: dict[str, np.ndarray] = {}
        for name, (path, seed) in paths_and_seeds.items():
            state, report = solver.solve_one(with_path(path), seed, 1.0)
            if not report.converged:
                raise RuntimeError(
                    f"reference state {name} failed: {report.failure_reason}"
                )
            solved[name] = state
        finite_path = AffineSourcePath.signal_turn_on(
            pump_source,
            signal_source * float(finite_signal_current_a),
        )
        finite_state, finite_report = solver.solve_one(
            with_path(finite_path),
            solved["pump_on_signal_infinitesimal"],
            1.0,
        )
        if not finite_report.converged:
            raise RuntimeError(
                "reference state pump_on_signal_finite failed: "
                f"{finite_report.failure_reason}"
            )
        solved["pump_on_signal_finite"] = finite_state
        return solved
    return {
        "pump_off_signal_on": pump_off_signal_on,
        "pump_on_signal_infinitesimal": pump_on_signal_infinitesimal,
        "pump_on_signal_finite": pump_on_signal_finite,
        "pump_on_signal_off": pump_on_signal_off,
    }


def reference_normalization(
    states: dict[str, np.ndarray],
    basis: MultiToneBasis,
    circuit: CircuitMatrices,
    *,
    source_port: int,
    out_port: int,
    signal_current_a: float,
    pump_current_a: float,
    z0_ohm: float = 50.0,
) -> dict[str, float]:
    """Return signal gain and pump depletion from the four reference states."""
    signal_on = tone_s21(
        states["pump_on_signal_finite"],
        basis,
        circuit,
        signal_tone=basis.signal_tone,
        source_port=source_port,
        out_port=out_port,
        source_current_a=signal_current_a,
        z0_ohm=z0_ohm,
    )
    signal_off = tone_s21(
        states["pump_off_signal_on"],
        basis,
        circuit,
        signal_tone=basis.signal_tone,
        source_port=source_port,
        out_port=out_port,
        source_current_a=signal_current_a,
        z0_ohm=z0_ohm,
    )
    pump_finite = tone_s21(
        states["pump_on_signal_finite"],
        basis,
        circuit,
        signal_tone=basis.pump_tone,
        source_port=source_port,
        out_port=out_port,
        source_current_a=pump_current_a,
        z0_ohm=z0_ohm,
    )
    pump_signal_off = tone_s21(
        states["pump_on_signal_off"],
        basis,
        circuit,
        signal_tone=basis.pump_tone,
        source_port=source_port,
        out_port=out_port,
        source_current_a=pump_current_a,
        z0_ohm=z0_ohm,
    )
    return {
        "gain_vs_off_db": float(
            20.0 * np.log10(max(abs(signal_on), 1e-300))
            - 20.0 * np.log10(max(abs(signal_off), 1e-300))
        ),
        "pump_depletion_db": float(
            20.0 * np.log10(max(abs(pump_finite), 1e-300))
            - 20.0 * np.log10(max(abs(pump_signal_off), 1e-300))
        ),
        "pump_off_signal_gain_db": float(
            20.0 * np.log10(max(abs(signal_off), 1e-300))
        ),
    }
