"""Observable quantities extracted from finite-signal multitone states."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from twpa_solver.core import CircuitMatrices
from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.core.linear import (
    default_loss_model_for,
    require_real,
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

if TYPE_CHECKING:
    from twpa_solver.multitone.imd import ImProduct


def _port_current_coefficients(
    X_full: np.ndarray,
    basis: MultiToneBasis,
    circuit: CircuitMatrices,
    dc_branch_flux: np.ndarray | None = None,
) -> np.ndarray:
    """Return physical port currents from the multitone KCL residual."""
    waveform = basis.synthesize(X_full)
    phase = (circuit.Bphi.T @ waveform.T).T
    law = make_branch_law(circuit)
    dc = np.zeros(circuit.Bphi.shape[1]) if dc_branch_flux is None else np.asarray(dc_branch_flux, dtype=float)
    if dc.shape != (circuit.Bphi.shape[1],):
        raise ValueError("dc_branch_flux must have one value per branch")
    nonlinear_time = circuit.Bphi @ (law.current(phase + dc[None, :]) - law.current(dc[None, :])).T
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
    dc_branch_flux: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract voltage and power-wave values for every retained tone/port."""
    values: dict[str, Any] = {"a": {}, "b": {}, "a_power": {}, "b_power": {}}
    currents = _port_current_coefficients(X_full, basis, circuit, dc_branch_flux)
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


def imd_products_dbc(
    X_full: np.ndarray,
    basis: MultiToneBasis,
    circuit: CircuitMatrices,
    products: list["ImProduct"],
    *,
    out_port: int,
    z0_ohm: float = 50.0,
    dc_branch_flux: np.ndarray | None = None,
) -> dict[str, float]:
    """Return each IM product's outgoing power relative to the signal tone."""
    waves = extract_port_waves(
        X_full, basis, circuit, ports=[int(out_port)], z0_ohm=z0_ohm,
        dc_branch_flux=dc_branch_flux,
    )
    b_power = waves["b_power"]
    signal_power = b_power.get((basis.signal_tone, int(out_port)))
    results: dict[str, float] = {}
    for product in products:
        power = b_power.get((product.tone, int(out_port)))
        if not signal_power or power is None or power <= 0.0:
            results[f"{product.label}_dbc"] = float("nan")
        else:
            results[f"{product.label}_dbc"] = float(10.0 * math.log10(power / signal_power))
    return results


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
    X_full: np.ndarray,
    basis: MultiToneBasis,
    circuit: CircuitMatrices,
    *,
    reference_X_full: np.ndarray | None = None,
    z0_ohm: float = 50.0,
) -> dict[str, Any]:
    """Compute real-power and incremental photon-flux diagnostics.

    ``reference_X_full`` is the pump-only operating point.  When supplied,
    the external Manley--Rowe quantity is formed from the change in port
    power caused by turning on the signal.  This removes pump-harmonic power
    conversion already present without a signal, which is not part of the
    signal/idler saturation invariant.  The returned ``pump_net_power_delta_w``
    is the all-port net pump-power change; ``pump_depletion_all_port_db`` is
    the corresponding all-port outgoing-pump power ratio in dB.  The two are
    emitted separately because the design contract is defined in net power,
    while dB depletion is conventionally reported from outgoing power.
    """
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
    nonlinear = circuit.Bphi @ make_branch_law(circuit).current(phase).T
    linear_coeffs = np.asarray([
        dynamic_block(
            circuit, omega, loss_model=default_loss_model_for(circuit)
        ) @ X_full[row]
        for row, omega in enumerate(basis.omegas)
    ])
    internal = require_real(
        basis.synthesize(linear_coeffs) + nonlinear.T,
        what="multitone internal residual",
    )
    internal_supplied_power = float(np.mean(np.sum(derivative * internal, axis=1)))
    dissipation = float(
        np.mean(np.sum(derivative * (circuit.G @ derivative.T).T, axis=1))
    )
    loss_dissipation = 0.0
    if circuit.has_loss:
        for row, omega in enumerate(basis.omegas):
            voltage = 1j * omega * X_full[row]
            loss_conductance = -abs(float(omega)) * circuit.C.imag
            loss_dissipation += 2.0 * float(
                np.real(np.vdot(voltage, loss_conductance @ voltage))
            )
    dissipation += loss_dissipation
    reference_dissipation = 0.0
    port_resistor_dissipation = 0.0
    reference_port_resistor_dissipation = 0.0
    if reference_X_full is not None:
        reference_waveform = basis.synthesize(reference_X_full)
        reference_derivative = np.zeros_like(reference_waveform)
        for row in range(len(basis.tones)):
            coefficient = np.zeros_like(reference_X_full)
            coefficient[row] = 1j * basis.omegas[row] * reference_X_full[row]
            reference_derivative += basis.synthesize(coefficient)
        reference_dissipation = float(
            np.mean(
                np.sum(
                    reference_derivative
                    * (circuit.G @ reference_derivative.T).T,
                    axis=1,
                )
            )
        )
    port_indices = np.asarray(
        [circuit.port_to_index[int(port)] for port in circuit.port_to_index],
        dtype=int,
    )
    if port_indices.size:
        port_resistor_dissipation = float(
            np.mean(np.sum(derivative[:, port_indices] ** 2, axis=1) / z0_ohm)
        )
        if reference_X_full is not None:
            reference_port_resistor_dissipation = float(
                np.mean(
                    np.sum(
                        reference_derivative[:, port_indices] ** 2,
                        axis=1,
                    )
                    / z0_ohm
                )
            )
    waves = extract_port_waves(
        X_full,
        basis,
        circuit,
        tuple(circuit.port_to_index),
        z0_ohm=z0_ohm,
    )
    reference_waves = (
        extract_port_waves(
            reference_X_full,
            basis,
            circuit,
            tuple(circuit.port_to_index),
            z0_ohm=z0_ohm,
        )
        if reference_X_full is not None
        else None
    )
    port_power_by_tone: dict[ToneIndex, float] = {}
    for row, tone in enumerate(basis.tones):
        power = float(
            0.5
            * sum(
                waves["a_power"][(tone, port)]
                - waves["b_power"][(tone, port)]
                for port in circuit.port_to_index
            )
        )
        if reference_waves is not None:
            power -= float(
                0.5
                * sum(
                    reference_waves["a_power"][(tone, port)]
                    - reference_waves["b_power"][(tone, port)]
                    for port in circuit.port_to_index
                )
            )
        port_power_by_tone[tone] = power
    external_supplied_power = float(sum(port_power_by_tone.values()))
    # ``extract_port_waves`` deliberately removes the explicit V/Z0 shunt
    # current before forming a/b.  The corresponding shunt loss must therefore
    # be removed from the dissipation side as well; otherwise the two sides
    # describe different boundaries and the balance can be O(1) wrong.
    external_dissipated_power = float(
        dissipation
        - reference_dissipation
        - port_resistor_dissipation
        + reference_port_resistor_dissipation
    )
    external_power_scale = max(
        abs(external_supplied_power), abs(external_dissipated_power)
    )
    external_power_balance_relative = (
        0.0
        if external_power_scale == 0.0
        else abs(external_supplied_power - external_dissipated_power)
        / external_power_scale
    )
    supplied_power = internal_supplied_power
    real_power_scale = max(abs(supplied_power), abs(dissipation))
    relative = (
        0.0
        if real_power_scale == 0.0
        else abs(supplied_power - dissipation) / real_power_scale
    )
    pump_tone = basis.pump_tone
    pump_net_power = float(
        0.5
        * sum(
            waves["a_power"][(pump_tone, port)]
            - waves["b_power"][(pump_tone, port)]
            for port in circuit.port_to_index
        )
    )
    pump_reference_net_power = None
    pump_net_power_delta = None
    pump_depletion_all_port_db = None
    pump_outgoing_power = float(
        0.5
        * sum(
            waves["b_power"][(pump_tone, port)]
            for port in circuit.port_to_index
        )
    )
    pump_reference_outgoing_power = None
    if reference_waves is not None:
        pump_reference_net_power = float(
            0.5
            * sum(
                reference_waves["a_power"][(pump_tone, port)]
                - reference_waves["b_power"][(pump_tone, port)]
                for port in circuit.port_to_index
            )
        )
        pump_net_power_delta = pump_net_power - pump_reference_net_power
        pump_reference_outgoing_power = float(
            0.5
            * sum(
                reference_waves["b_power"][(pump_tone, port)]
                for port in circuit.port_to_index
            )
        )
        pump_depletion_all_port_db = float(
            10.0
            * np.log10(
                max(pump_outgoing_power, 1e-300)
                / max(pump_reference_outgoing_power, 1e-300)
            )
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
    # Below this scale the quotient is dominated by cancellation/roundoff;
    # exposing it as a false 0.5--style physical error is misleading.
    manley_evaluable = photon_scale > 1e-28
    manley_relative = (
        abs(photon_flux) / photon_scale if manley_evaluable else 0.0
    )
    # The finite-signal Manley--Rowe invariant is the three-wave conversion
    # channel, not the raw photon sum over every retained pump harmonic.  A
    # pump-only solution can contain harmonic generation (for example the
    # h=1 -> h=3 conversion), which is conservative in energy but is not part
    # of signal/idler saturation and gives a spurious ~1/2 photon residual.
    conversion_tones = (basis.pump_tone, basis.signal_tone, basis.idler_tone)
    external_photon_terms = [
        float(port_power_by_tone[tone] / tone.omega(basis.omega_p, basis.delta))
        for tone in conversion_tones
    ]
    external_photon_flux = float(np.sum(external_photon_terms))
    external_photon_scale = float(np.sum(np.abs(external_photon_terms)))
    external_manley_evaluable = external_photon_scale > 1e-30
    return {
        "supplied_power": supplied_power,
        "internal_supplied_power": internal_supplied_power,
        "external_supplied_power": external_supplied_power,
        "external_dissipated_power": external_dissipated_power,
        "port_resistor_dissipated_power": port_resistor_dissipation,
        "reference_port_resistor_dissipated_power": reference_port_resistor_dissipation,
        "external_power_balance_rel_err": external_power_balance_relative,
        "dissipated_power": dissipation,
        "dielectric_dissipated_power": float(loss_dissipation),
        "power_balance_rel_err": relative,
        "manley_rowe_photon_flux": photon_flux,
        "manley_rowe_photon_scale": photon_scale,
        "manley_rowe_evaluable": float(manley_evaluable),
        "manley_rowe_rel_err": manley_relative,
        "external_manley_rowe_photon_flux": external_photon_flux,
        "external_manley_rowe_photon_scale": external_photon_scale,
        "external_manley_rowe_evaluable": float(external_manley_evaluable),
        "external_manley_rowe_rel_err": (
            abs(external_photon_flux) / external_photon_scale
            if external_manley_evaluable
            else 0.0
        ),
        # Explicit names for the two scopes.  The legacy keys above remain
        # readable by existing experiment consumers.
        "conversion_manley_rowe_photon_flux": external_photon_flux,
        "conversion_manley_rowe_photon_scale": external_photon_scale,
        "conversion_manley_rowe_evaluable": float(external_manley_evaluable),
        "conversion_manley_rowe_rel_err": (
            abs(external_photon_flux) / external_photon_scale
            if external_manley_evaluable
            else 0.0
        ),
        "all_tone_manley_rowe_photon_flux": photon_flux,
        "all_tone_manley_rowe_photon_scale": photon_scale,
        "all_tone_manley_rowe_evaluable": float(manley_evaluable),
        "all_tone_manley_rowe_rel_err": manley_relative,
        # The net pump quantity is the design-contract all-port observable.
        # The outgoing-power ratio is retained as a convenient dB presentation;
        # it is not substituted for the net-power field.
        "pump_net_power_w": pump_net_power,
        "pump_reference_net_power_w": pump_reference_net_power,
        "pump_net_power_delta_w": pump_net_power_delta,
        "pump_outgoing_power_w": pump_outgoing_power,
        "pump_reference_outgoing_power_w": pump_reference_outgoing_power,
        "pump_depletion_all_port_db": pump_depletion_all_port_db,
        "port_power_by_tone_w": {
            f"h{tone.h}_q{tone.q}": float(power)
            for tone, power in port_power_by_tone.items()
        },
    }


def spatial_depletion_null(
    spatial_rows: list[dict[str, Any]],
    small_signal_rows: list[dict[str, Any]],
) -> float:
    """Return a distributed depletion-only gain estimate in dB.

    The small-signal rows provide the local signal-growth increments and phase
    mismatch.  The operating-point pump amplitudes scale those increments,
    while the mismatch is deliberately kept at its small-signal value.  This
    is a diagnostic null, not a replacement for the nonlinear solve.

    Args:
        spatial_rows: Branch rows at the operating point.
        small_signal_rows: Branch rows at the zero-signal operating point.

    Raises:
        ValueError: If rows are empty, mismatched, or contain invalid fluxes.
    """
    if not spatial_rows or not small_signal_rows:
        raise ValueError("spatial depletion null requires non-empty profiles")
    if len(spatial_rows) != len(small_signal_rows):
        raise ValueError("operating and small-signal profiles must have equal length")
    ordered = sorted(spatial_rows, key=lambda row: int(row["branch_index"]))
    reference = sorted(
        small_signal_rows, key=lambda row: int(row["branch_index"])
    )
    signal_reference = np.asarray(
        [float(row["signal_flux_abs"]) for row in reference]
    )
    if np.any(signal_reference <= 0.0):
        raise ValueError("small-signal signal fluxes must be positive")
    local_growth = np.zeros(len(reference), dtype=float)
    local_growth[1:] = math.log(
        signal_reference[-1] / signal_reference[0]
    ) / max(len(reference) - 1, 1)
    reference_pump = np.asarray(
        [float(row["pump_flux_abs"]) for row in reference]
    )
    operating_pump = np.asarray(
        [float(row["pump_flux_abs"]) for row in ordered]
    )
    if np.any(reference_pump <= 0.0) or np.any(operating_pump < 0.0):
        raise ValueError("pump fluxes must be non-negative and nonzero at reference")
    reference_gain_db = float(
        20.0 * math.log10(signal_reference[-1] / signal_reference[0])
    )
    if np.allclose(operating_pump, reference_pump, rtol=1e-12, atol=0.0):
        return reference_gain_db
    delta_k = np.asarray(
        [float(row["delta_k_eff_rad_per_cell"]) for row in reference]
    )
    pump_scale = operating_pump / reference_pump
    phase_factor = np.cos(delta_k)
    log_amplitude_gain = float(
        np.sum(local_growth * pump_scale * phase_factor)
    )
    return float(20.0 * log_amplitude_gain / math.log(10.0))


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
            "pump_intensity_normalized": float(
                abs(pump[branch]) ** 2 / max(np.max(np.abs(pump) ** 2), 1e-300)
            ),
            "signal_intensity_normalized": float(
                abs(signal[branch]) ** 2 / max(np.max(np.abs(signal) ** 2), 1e-300)
            ),
            "theta_rad": float(theta[branch]),
            "delta_k_eff_rad_per_cell": float(delta_k[branch]),
        }
        for branch in range(circuit.branch_count)
    ]


def spatial_profile_summary(
    rows: list[dict[str, float | int]],
) -> dict[str, float | int | list[int] | None]:
    """Summarize pump/signal/idler spatial overlap for profile rows."""
    pump = np.asarray([float(row["pump_flux_abs"]) for row in rows])
    signal = np.asarray([float(row["signal_flux_abs"]) for row in rows])
    idler = np.asarray([float(row["idler_flux_abs"]) for row in rows])
    denominator = np.linalg.norm(pump) * np.linalg.norm(signal) * np.linalg.norm(idler)
    overlap = float(np.sum(pump * signal * idler) / max(denominator, 1e-300))
    threshold = 0.1 * float(np.max(pump))
    above = np.flatnonzero(pump >= threshold)
    branch_range = (
        [int(rows[int(above[0])]["branch_index"]), int(rows[int(above[-1])]["branch_index"])]
        if above.size
        else None
    )
    return {
        "overlap_integral": overlap,
        "pump_above_10pct_branch_range": branch_range,
        "pump_above_10pct_count": int(above.size),
        "pump_above_10pct_fraction": float(above.size / max(len(rows), 1)),
    }


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
