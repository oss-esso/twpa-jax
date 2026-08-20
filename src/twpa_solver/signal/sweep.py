"""Production signal-gain sweep orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import load_circuit
from twpa_solver.signal.gamma import compute_gamma_hat, load_dc_branch_flux
from twpa_solver.signal.gain import GainResult
from twpa_solver.signal.io import load_pump, write_outputs
from twpa_solver.signal.floquet import (
    assemble_khat_conversion_base,
    sideband_list,
    solve_gain_one,
)
from twpa_solver.signal.gamma import build_khat
from twpa_solver.signal.qe_row import signal_row_quantum_efficiency


def run_gain_sweep(
    circuit_dir: str | Path,
    pump_dir: str | Path,
    outdir: str | Path,
    *,
    signal_start_ghz: float,
    signal_stop_ghz: float,
    points: int,
    fallback_pump_freq_ghz: float,
    sidebands: int,
    gamma_nt: int,
    source_port: int = 1,
    out_port: int = 2,
    source_current_a: float = 1.0,
    z0_ohm: float = 50.0,
    signal_m: int = 0,
    idler_m: int = -2,
    loss_model: str = "current_complex_c",
    dc_solution: str | Path | None = None,
    pump_state_scale: float = 1.0,
    drop_gamma_tol: float = 0.0,
    include_baselines: bool = True,
    quantum_efficiency: bool = False,
) -> list[GainResult]:
    """Run and persist a production Floquet signal-gain sweep.

    This is the package equivalent of the historical experiment-09 sweep.
    It intentionally contains no subprocess or experiment-script dependency,
    so workflows and plotting tools can use the production solver directly.
    """
    circuit = load_circuit(circuit_dir)
    pump = load_pump(pump_dir, fallback_pump_freq_ghz)
    if abs(float(pump_state_scale) - 1.0) > 0.0:
        pump.X = pump.X * float(pump_state_scale)

    if source_port not in circuit.port_to_index:
        raise ValueError(f"source port {source_port} not in {circuit.port_to_index}")
    if out_port not in circuit.port_to_index:
        raise ValueError(f"out port {out_port} not in {circuit.port_to_index}")
    if points < 1:
        raise ValueError(f"points must be positive, got {points}")

    ms = sideband_list(sidebands)
    if signal_m not in ms:
        raise ValueError(f"signal_m={signal_m} not in sideband set {ms}")
    max_ell = max(abs(m - q) for m in ms for q in ms)

    dc_branch_flux = load_dc_branch_flux(dc_solution, circuit)
    gamma_hat = compute_gamma_hat(
        circuit=circuit,
        pump=pump,
        max_ell=max_ell,
        gamma_nt=gamma_nt,
        dc_branch_flux=dc_branch_flux,
    )
    khat = build_khat(
        Bphi=circuit.Bphi,
        gamma_hat=gamma_hat,
        drop_tol=drop_gamma_tol,
    )
    khat_big_base = assemble_khat_conversion_base(circuit, khat, ms)

    if dc_branch_flux is None:
        gamma_off = circuit.Ic / circuit.phi0
    else:
        gamma_off = (circuit.Ic / circuit.phi0) * np.cos(
            dc_branch_flux / circuit.phi0
        )
    khat_off_0 = (
        circuit.Bphi
        @ sp.diags(gamma_off, offsets=0, format="csr")
        @ circuit.Bphi.T
    ).astype(np.complex128).tocsr()

    freqs = np.linspace(signal_start_ghz, signal_stop_ghz, points)
    rows: list[GainResult] = []
    source_index = circuit.port_to_index[source_port]
    out_index = circuit.port_to_index[out_port]
    for signal_ghz in freqs:
        result = solve_gain_one(
            circuit=circuit,
            khat=khat,
            khat_off_0=khat_off_0,
            omega_p=pump.omega_p,
            signal_ghz=float(signal_ghz),
            sidebands=sidebands,
            signal_m=signal_m,
            idler_m=idler_m,
            source_index=source_index,
            out_index=out_index,
            source_current_a=source_current_a,
            source_port=source_port,
            out_port=out_port,
            z0_ohm=z0_ohm,
            loss_model=loss_model,
            khat_big_base=khat_big_base,
            include_baselines=include_baselines,
        )
        if quantum_efficiency and result.status == "VALID_SOLVED":
            row_qe = signal_row_quantum_efficiency(
                circuit=circuit,
                khat=khat,
                khat_off_0=khat_off_0,
                omega_p=pump.omega_p,
                pump_freq_ghz=pump.pump_freq_ghz,
                signal_ghz=float(signal_ghz),
                sidebands=sidebands,
                signal_m=signal_m,
                idler_m=idler_m,
                source_index=source_index,
                out_index=out_index,
                source_port=source_port,
                out_port=out_port,
                z0_ohm=z0_ohm,
                loss_model=loss_model,
            )
            result.qe_signal = row_qe.qe_signal
            result.qe_ideal_signal = row_qe.qe_ideal_signal
            result.qe_ratio = row_qe.qe_ratio
            result.qe_unitarity_residual = row_qe.unitarity_residual
            result.qe_sidebands_summed = row_qe.sidebands_summed
        rows.append(result)

    metadata = {
        "circuit_dir": str(circuit_dir),
        "pump_dir": str(pump_dir),
        "source_port": source_port,
        "out_port": out_port,
        "source_index": source_index,
        "out_index": out_index,
        "source_current_a": source_current_a,
        "z0_ohm": z0_ohm,
        "pump_harmonics": pump.harmonics,
        "pump_modes": list(pump.modes),
        "pump_basis": pump.basis.basis,
        "pump_mode_policy": pump.basis.policy,
        "pump_nt_original": pump.nt_original,
        "pump_freq_ghz": pump.pump_freq_ghz,
        "omega_p": pump.omega_p,
        "sidebands": sidebands,
        "sideband_set": ms,
        "signal_m": signal_m,
        "idler_m": idler_m,
        "gamma_nt": gamma_nt,
        "pump_state_scale": pump_state_scale,
        "loss_linearization_model": loss_model,
        "signal_start_ghz": signal_start_ghz,
        "signal_stop_ghz": signal_stop_ghz,
        "points": points,
        "include_baselines": include_baselines,
        "quantum_efficiency": quantum_efficiency,
    }
    write_outputs(outdir, rows, metadata)
    return rows
