"""Serial reduced/full effective-SNAIL HB campaign for the paper benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from references.le_gal_2025_gain_compression.cme import CMEParameters, integrate_cme
from references.le_gal_2025_gain_compression.cme import depletion_only_gain
from twpa_solver.builders.le_gal_2025 import build_effective_snail_line
from twpa_solver.multitone.basis import ToneIndex, build_sideband_matched_basis
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.observables import tone_s21
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicGrid, HarmonicNewtonKrylovSolver, NewtonKrylovSettings
from twpa_solver.pump.problem import FullPumpProblem


def _settings() -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=25, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=30, gmres_maxiter=50, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
    )


def _current_from_dbm(power_dbm: float, z0_ohm: float) -> float:
    return math.sqrt(2.0 * 10.0 ** ((power_dbm - 30.0) / 10.0) / z0_ohm)


def _run_point(cells: int, signal_ghz: float, power_dbm: float, sidebands: int, external_flux_on_small_junction: bool = False, pump_power_dbm: float = -78.4, pump_modes: tuple[int, ...] = (1, 3, 5)) -> dict[str, object]:
    started = time.perf_counter()
    circuit = build_effective_snail_line(
        cells=cells, port_impedance_ohm=62.4,
        external_flux_on_small_junction=external_flux_on_small_junction,
    )
    z0_ohm = float(circuit.metadata["port_impedance_ohm"])
    omega_p = 2.0 * math.pi * 7.5e9
    pump_current = _current_from_dbm(pump_power_dbm, z0_ohm)
    pump_problem = FullPumpProblem(
        C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
        branch=circuit.branch_law,
        grid=HarmonicGrid(np.array(pump_modes), nt=max(16, 2 * max(pump_modes) + 2), omega=omega_p),
        pump_node_index=circuit.port_to_index[1], pump_current_a=pump_current,
    )
    solver = HarmonicNewtonKrylovSolver(_settings())
    pump_state, pump_reports = solver.solve_continuation(pump_problem, continuation_steps=8)
    if not pump_reports[-1].converged:
        return {"cells": cells, "signal_GHz": signal_ghz, "signal_dBm": power_dbm,
                "status": "PUMP_FAILED", "runtime_s": time.perf_counter() - started}
    delta = omega_p - 2.0 * math.pi * signal_ghz * 1e9
    basis = build_sideband_matched_basis(list(pump_modes), sidebands, omega_p, delta, omega_p * 12.0)
    pump_source = np.zeros((basis.n_tones, circuit.node_count), dtype=np.complex128)
    pump_coeffs = pump_problem.source_coeffs(1.0)
    for row, mode in enumerate(pump_modes):
        pump_source[basis.index_of(ToneIndex(mode, 0))] = pump_coeffs[row]
    signal_current = _current_from_dbm(power_dbm, z0_ohm)
    signal_source = MultiToneDrive(
        basis.signal_tone, circuit.port_to_index[1],
        signal_current,
    ).to_coeffs(basis, circuit.node_count)
    problem = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.signal_turn_on(pump_source, signal_source)
    )
    seed = np.zeros_like(pump_source)
    for row, mode in enumerate(pump_modes):
        seed[basis.index_of(ToneIndex(mode, 0))] = pump_state[row]
    state, report = solver.solve_one(problem, seed, 1.0)
    residual = float(np.linalg.norm(problem.residual_coeffs(state, 1.0)))
    signal_row = basis.index_of(basis.signal_tone)
    pump_row = basis.index_of(basis.pump_tone)
    pump_off_problem = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.signal_turn_on(np.zeros_like(pump_source), signal_source)
    )
    pump_off, pump_off_report = solver.solve_one(
        pump_off_problem, np.zeros_like(seed), 1.0
    )
    s21 = tone_s21(
        state, basis, circuit, signal_tone=basis.signal_tone,
        source_port=1, out_port=2, source_current_a=signal_current, z0_ohm=z0_ohm,
    )
    s21_off = tone_s21(
        pump_off, basis, circuit, signal_tone=basis.signal_tone,
        source_port=1, out_port=2, source_current_a=signal_current, z0_ohm=z0_ohm,
    )
    pump_s21 = tone_s21(
        state, basis, circuit, signal_tone=basis.pump_tone,
        source_port=1, out_port=2, source_current_a=pump_current, z0_ohm=z0_ohm,
    )
    pump_s21_reference = tone_s21(
        seed, basis, circuit, signal_tone=basis.pump_tone,
        source_port=1, out_port=2, source_current_a=pump_current, z0_ohm=z0_ohm,
    )
    branch_flux = (circuit.Bphi.T @ state.T).T
    phase = np.unwrap(np.angle(branch_flux), axis=0)
    phase_mismatch = 2.0 * phase[pump_row] - phase[signal_row] - phase[basis.index_of(basis.idler_tone)]
    off_gain_linear = max(abs(s21_off) ** 2, 1e-300)
    return {
        "cells": cells, "signal_GHz": signal_ghz, "signal_dBm": power_dbm,
        "external_flux_on_small_junction": external_flux_on_small_junction,
        "pump_dBm": pump_power_dbm,
        "status": "SOLVED" if report.converged else "HB_FAILED",
        "residual_norm": residual,
        "s21_db": float(20.0 * np.log10(max(abs(s21), 1e-300))),
        "gain_vs_off_db": float(20.0 * np.log10(max(abs(s21) / max(abs(s21_off), 1e-300), 1e-300))),
        "hb_gain_db": float(20.0 * np.log10(max(abs(s21), 1e-300))),
        "pump_depletion_db": float(20.0 * np.log10(max(abs(pump_s21) / max(abs(pump_s21_reference), 1e-300), 1e-300))),
        "nonlinear_pump_phase_rad": float(np.angle(pump_s21 / pump_s21_reference)),
        "spatial_phase_mismatch_rad_min": float(np.nanmin(phase_mismatch)),
        "spatial_phase_mismatch_rad_max": float(np.nanmax(phase_mismatch)),
        "depletion_model_gain_dB": float(10.0 * np.log10(depletion_only_gain(off_gain_linear, 10.0 ** ((power_dbm - 30.0) / 10.0), 10.0 ** ((pump_power_dbm - 30.0) / 10.0)))),
        "pump_off_s21_db": float(20.0 * np.log10(max(abs(s21_off), 1e-300))),
        "pump_off_status": "SOLVED" if pump_off_report.converged else "HB_FAILED",
        "runtime_s": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[20, 50, 700])
    default_frequencies = sorted({
        round(float(value), 2) for value in np.arange(4.0, 11.0 + 0.125, 0.25)
    } | {6.4, 8.6})
    parser.add_argument("--frequencies", type=float, nargs="+", default=default_frequencies)
    parser.add_argument("--powers", type=float, nargs="+", default=[-115.0, -110.0, -105.0, -100.0, -94.0])
    parser.add_argument("--sidebands", type=int, default=3)
    parser.add_argument("--external-flux-on-small-junction", action="store_true")
    parser.add_argument("--pump-power-dbm", type=float, default=-78.4)
    parser.add_argument("--pump-modes", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for cells in args.cells:
        for frequency in args.frequencies:
            for power in args.powers:
                rows.append(_run_point(
                    cells, frequency, power, args.sidebands,
                    args.external_flux_on_small_junction,
                    args.pump_power_dbm,
                    tuple(args.pump_modes),
                ))
    grouped: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in rows:
        if row["status"] == "SOLVED":
            grouped.setdefault((int(row["cells"]), float(row["signal_GHz"])), []).append(row)
    for points in grouped.values():
        points.sort(key=lambda item: float(item["signal_dBm"]))
        reference_gain = float(points[0]["gain_vs_off_db"])
        if len(points) > 1 and abs(reference_gain - float(points[1]["gain_vs_off_db"])) > 0.05:
            for point in points:
                point["status"] = "NO_PLATEAU"
            continue
        for point in points:
            point["compression_db"] = reference_gain - float(point["gain_vs_off_db"])
        crossing = next((point for point in points if float(point["compression_db"]) >= 1.0), None)
        for point in points:
            point["p1db_dBm"] = float(crossing["signal_dBm"]) if crossing else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
