"""Run a KIMPA degenerate-3WM gain map and plot it against peak I/Ic.

The pump sweep is performed in increasing internal dBm order, but dBm is not
used as a plotted axis: the measured peak total branch current normalized by
Ic is the operating coordinate.  This keeps the map meaningful when the
pump-line attenuation is unknown.  For a pump at fp, the Floquet idler is the
negative-frequency partner at fp-fs; the reported physical idler frequency is
stored explicitly as ``fp - fs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_kimpa_gain
from twpa_solver.builders.kimpa import build_kimpa
from twpa_solver.core import kinetic_dc_branch_flux, PortEnvironment, save_circuit
from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.pump.basis import resolve_pump_basis
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.floquet import solve_gain_one
from twpa_solver.signal.io import PumpSolution
from twpa_solver.core.linear import port_s_from_unit_current_response
from twpa_solver.port_roles import resolve_mixing_order, resolve_port_roles
from twpa_solver.signal.passive import passive_network_matrices
from twpa_solver.signal.quantum_efficiency import calc_qe, calc_qe_ideal


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--fixture", choices=("kimpa_ideal_synthesis", "kimpa_fabricated_nominal", "kimpa_measured_seed", "kimpa_hung_2025"), default="kimpa_fabricated_nominal")
    p.add_argument("--pump-dbm-start", type=float, default=-35.0)
    p.add_argument("--pump-dbm-stop", type=float, default=-8.0)
    p.add_argument("--pump-points", type=int, default=15)
    p.add_argument("--pump-attenuation-db", type=float, default=0.0)
    p.add_argument("--pump-ghz", type=float, default=16.94)
    p.add_argument("--signal-start-ghz", type=float, default=7.8)
    p.add_argument("--signal-stop-ghz", type=float, default=9.1)
    p.add_argument("--signal-points", type=int, default=27)
    p.add_argument("--dc-current-a", type=float, default=550e-6)
    p.add_argument("--sidebands", type=int, default=5)
    p.add_argument("--max-ell", type=int, default=6)
    p.add_argument("--pump-nt", type=int, default=32)
    p.add_argument("--environment", choices=("ideal", "paper_standing_wave"), default="ideal")
    p.add_argument("--pump-port", type=int, default=None)
    p.add_argument("--source-port", type=int, default=None)
    p.add_argument("--out-port", type=int, default=None)
    p.add_argument("--mixing-order", choices=("auto", "3", "4"), default="auto")
    p.add_argument("--spectrum-start-ghz", type=float, default=7.5)
    p.add_argument("--spectrum-stop-ghz", type=float, default=9.5)
    p.add_argument("--spectrum-points", type=int, default=501)
    p.add_argument("--no-spectrum", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def _point_args(args: argparse.Namespace, outdir: Path, pump_dbm: float, signal_ghz: float) -> argparse.Namespace:
    values = [
        "--fixture", args.fixture, "--output-dir", str(outdir), "--pump-dbm", str(pump_dbm),
        "--pump-attenuation-db", str(args.pump_attenuation_db), "--pump-ghz", str(args.pump_ghz),
        "--signal-ghz", str(signal_ghz), "--dc-current-a", str(args.dc_current_a),
        "--sidebands", str(args.sidebands), "--max-ell", str(args.max_ell),
        "--pump-nt", str(args.pump_nt), "--environment", args.environment,
        "--no-waveforms",
    ]
    for flag, value in (("--pump-port", args.pump_port),
                        ("--source-port", args.source_port),
                        ("--out-port", args.out_port),
                        ("--mixing-order", args.mixing_order)):
        if value is not None:
            values.extend([flag, str(value)])
    return run_kimpa_gain.build_parser().parse_args(values)


def _write_plots(run_dir: Path, signal: np.ndarray, ratio: np.ndarray, gain: np.ndarray, idler: np.ndarray, status: np.ndarray) -> None:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    extent = [float(np.nanmin(ratio)), float(np.nanmax(ratio)), float(signal[0]), float(signal[-1])]
    for values, title, filename, cmap, label in (
        (gain, "KIMPA 3WM reflection gain", "gain_db_vs_I_over_Ic.png", "magma", "gain (dB)"),
        (idler, "KIMPA 3WM idler conversion", "idler_conversion_db_vs_I_over_Ic.png", "viridis", "idler power / signal-off (dB)"),
    ):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.axvline(1.0, color="cyan", linestyle="--", linewidth=1.2, label="I/Ic = 1")
        ax.axhline(signal[len(signal) // 2], color="white", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("peak total current / Ic")
        ax.set_ylabel("signal frequency (GHz)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label=label)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    peak = np.nanmax(gain, axis=0)
    peak_signal = signal[np.nanargmax(gain, axis=0)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ratio, peak, "o-", label="maximum over signal")
    ax.axvline(1.0, color="tab:red", linestyle="--", label="KI threshold")
    ax.axhline(17.0, color="tab:green", linestyle=":", label="17 dB paper target")
    ax.set_xlabel("peak total current / Ic")
    ax.set_ylabel("maximum reflection gain (dB)")
    ax.set_title("KIMPA gain envelope versus physical pump coordinate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "peak_gain_vs_I_over_Ic.png", dpi=160)
    plt.close(fig)
    np.savetxt(plot_dir / "peak_gain_envelope.csv", np.column_stack((ratio, peak, peak_signal)), delimiter=",", header="I_over_Ic,peak_gain_db,signal_ghz", comments="")


def _run_best_spectrum(args: argparse.Namespace, run_dir: Path, best: dict[str, object]) -> None:
    pump_dbm = float(best["pump_dbm_internal"])
    signal_best = float(best["signal_ghz"])
    point_dir = run_dir / "best_point"
    point_args = _point_args(args, point_dir, pump_dbm, signal_best)
    point_args.no_waveforms = False
    payload = run_kimpa_gain.run(point_args)
    (point_dir / "kimpa_gain.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    waveform = np.load(point_dir / "kimpa_gain_waveforms.npz")
    circuit = build_kimpa(args.fixture)
    roles = resolve_port_roles(
        circuit,
        pump_port=args.pump_port,
        source_port=args.source_port,
        out_port=args.out_port,
    )
    ports = tuple(sorted(circuit.port_to_index))
    source_port = roles["source_port"]
    out_port = roles["out_port"]
    dc_flux = kinetic_dc_branch_flux(circuit, args.dc_current_a)
    pump_freq = float(args.pump_ghz)
    omega_p = 2.0 * np.pi * pump_freq * 1e9
    modes = np.asarray(payload["pump_modes"], dtype=int)
    state = waveform["pump_state"]
    pump = PumpSolution(X=state, omega_p=omega_p, pump_freq_ghz=pump_freq,
                        harmonics=state.shape[0], nt_original=args.pump_nt,
                        metadata={}, modes=list(modes), basis=None)
    gamma_hat = compute_gamma_hat(circuit, pump, args.max_ell, args.pump_nt, dc_flux)
    khat = build_khat(circuit.Bphi, gamma_hat, 1e-30)
    gamma_off = make_branch_law(circuit).tangent(dc_flux[None, :])[0]
    khat_off = (circuit.Bphi @ sp.diags(gamma_off) @ circuit.Bphi.T).astype(np.complex128).tocsr()
    environment = PortEnvironment() if args.environment == "paper_standing_wave" else None
    freqs = np.linspace(args.spectrum_start_ghz, args.spectrum_stop_ghz, args.spectrum_points)
    rows = []
    signal_s = np.zeros((len(freqs), len(ports), len(ports)), dtype=np.complex128)
    idler_s = np.zeros_like(signal_s)
    for freq in freqs:
        row_index = len(rows)
        result_for_primary = None
        for source_index, source in enumerate(ports):
            for out_index, output in enumerate(ports):
                result = solve_gain_one(
                    circuit, khat, khat_off, omega_p, float(freq), args.sidebands,
                    signal_m=0, idler_m=-(int(payload["mixing_order"]) - 1),
                    source_index=circuit.port_to_index[source],
                    out_index=circuit.port_to_index[output], source_current_a=1.0,
                    source_port=source, out_port=output, z0_ohm=50.0,
                    environment=environment,
                )
                signal_s[row_index, out_index, source_index] = port_s_from_unit_current_response(
                    result.vout_on, source_port=source, out_port=output, z0_ohm=50.0
                )
                if result.vout_idler is not None:
                    idler_s[row_index, out_index, source_index] = 2.0 * result.vout_idler / 50.0
                if source == source_port and output == out_port:
                    result_for_primary = result
        if result_for_primary is None:
            raise RuntimeError("resolved KIMPA source/output ports were not found in spectrum matrix")
        primary_s = signal_s[row_index, ports.index(out_port), ports.index(source_port)]
        rows.append({
            "signal_ghz": float(freq), "idler_ghz": pump_freq - float(freq),
            "gain_db": result_for_primary.gain_db,
            "gain_vs_off_db": result_for_primary.gain_vs_off_db,
            "s11_real": float(np.real(primary_s)), "s11_imag": float(np.imag(primary_s)),
            "s11_phase_deg": float(np.angle(primary_s, deg=True)),
            "idler_power_rel_to_signal_off_db": result_for_primary.idler_power_rel_to_signal_off_db,
            "status": result_for_primary.status, "linear_rel_residual": result_for_primary.linear_rel_residual,
        })
    passive = passive_network_matrices(
        run_dir / "circuit", freqs * 1e9, ports=ports, z0_ohm=50.0,
        dc_branch_flux=dc_flux,
    )
    qe = np.asarray([calc_qe(s, s_noise=noise) for s, noise in zip(signal_s, idler_s)])
    qe_ideal = np.asarray([calc_qe_ideal(s) for s in signal_s])
    source_position = ports.index(source_port)
    out_position = ports.index(out_port)
    for row_index, row in enumerate(rows):
        row["s_primary_real"] = float(np.real(signal_s[row_index, out_position, source_position]))
        row["s_primary_imag"] = float(np.imag(signal_s[row_index, out_position, source_position]))
        row["s_primary_db"] = float(20.0 * np.log10(max(abs(signal_s[row_index, out_position, source_position]), 1e-300)))
        row["quantum_efficiency"] = float(qe[row_index, out_position, source_position])
        row["quantum_efficiency_ideal"] = float(qe_ideal[row_index, out_position, source_position])
        for output in ports:
            for source in ports:
                output_position = ports.index(output)
                input_position = ports.index(source)
                label = f"s{output}{source}"
                value = signal_s[row_index, output_position, input_position]
                row[f"{label}_real"] = float(np.real(value))
                row[f"{label}_imag"] = float(np.imag(value))
                row[f"{label}_db"] = float(20.0 * np.log10(max(abs(value), 1e-300)))
                row[f"{label}_qe"] = float(qe[row_index, output_position, input_position])
    np.savez(
        run_dir / "best_point_spectrum.npz",
        signal_frequency_ghz=freqs,
        idler_frequency_ghz=pump_freq - freqs,
        ports=np.asarray(ports, dtype=int),
        Z=passive["Z"], Y=passive["Y"], S=passive["S"],
        S_pumped=signal_s, S_idler=idler_s,
        quantum_efficiency=qe, quantum_efficiency_ideal=qe_ideal,
    )
    spectrum_path = run_dir / "best_point_spectrum.csv"
    with spectrum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gain_values = np.asarray([row["gain_db"] for row in rows], dtype=float)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(freqs, gain_values, label="signal gain")
    ax.axvline(signal_best, color="tab:red", linestyle="--", label=f"map optimum {signal_best:.4f} GHz")
    ax.axhline(17.0, color="tab:green", linestyle=":", label="17 dB paper target")
    ax.set_xlabel("signal frequency (GHz)")
    ax.set_ylabel("gain (dB)")
    ax.set_title(f"KIMPA 3WM spectrum; pump {pump_dbm:.3f} dBm, peak I/Ic={best['max_current_over_ic']:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "best_point_spectrum.png", dpi=160)
    plt.close(fig)
    summary = {
        "pump_dbm_internal": pump_dbm, "signal_best_ghz": signal_best,
        "idler_best_ghz": pump_freq - signal_best,
        "max_current_over_ic": best["max_current_over_ic"],
        "peak_spectrum_gain_db": float(np.nanmax(gain_values)),
        "peak_spectrum_signal_ghz": float(freqs[np.nanargmax(gain_values)]),
        "points": args.spectrum_points, "start_ghz": args.spectrum_start_ghz,
        "stop_ghz": args.spectrum_stop_ghz, "ports": list(ports),
        "matrix_file": "best_point_spectrum.npz",
        "quantum_efficiency_definition": "calc_qe(S_pumped, s_noise=S_idler)",
    }
    (run_dir / "best_point_spectrum_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote={spectrum_path}")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.pump_points < 1 or args.signal_points < 1:
        raise ValueError("pump-points and signal-points must be positive")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    circuit = build_kimpa(args.fixture)
    roles = resolve_port_roles(
        circuit,
        pump_port=args.pump_port,
        source_port=args.source_port,
        out_port=args.out_port,
    )
    for role, port in roles.items():
        setattr(args, role, port)
    args.mixing_order = resolve_mixing_order(
        args.mixing_order, dc_current_a=args.dc_current_a,
        design_meta=circuit.metadata,
    )
    circuit_dir = run_dir / "circuit"
    if not (circuit_dir / "C.npz").exists():
        save_circuit(circuit, circuit_dir)
    pumps = np.linspace(args.pump_dbm_start, args.pump_dbm_stop, args.pump_points)
    signals = np.linspace(args.signal_start_ghz, args.signal_stop_ghz, args.signal_points)
    rows: list[dict[str, object]] = []
    for pump_dbm in pumps:
        pump_dir = run_dir / f"pump_{pump_dbm:+.3f}dBm".replace("+", "p").replace("-", "m").replace(".", "p")
        # Keep each point independently reproducible; waveform arrays are
        # disabled to avoid multiplying map storage by the signal grid.
        for signal_ghz in signals:
            point_dir = pump_dir / f"signal_{signal_ghz:.6f}GHz"
            result = run_kimpa_gain.run(_point_args(args, point_dir, float(pump_dbm), float(signal_ghz)))
            rows.append({"pump_dbm_internal": float(pump_dbm), **result})
        print(f"completed pump={pump_dbm:.3f} dBm ({len(rows)}/{len(pumps) * len(signals)} points)")

    rows_path = run_dir / "kimpa_gain_map.csv"
    keys = ["pump_dbm_internal", "pump_dbm_on_chip", "signal_ghz", "idler_ghz", "max_current_over_ic", "gain_db", "gain_vs_off_db", "s11_real", "s11_imag", "s11_phase_deg", "idler_power_rel_to_signal_off_db", "pump_converged", "gain_status", "kinetic_status", "pump_coeff_rel", "linear_rel_residual"]
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ratios = np.asarray([float(row["max_current_over_ic"]) for row in rows], dtype=float).reshape(len(pumps), len(signals))
    gains = np.asarray([float(row["gain_db"]) for row in rows], dtype=float).reshape(len(pumps), len(signals))
    idler = np.asarray([[np.nan if row["idler_power_rel_to_signal_off_db"] is None else float(row["idler_power_rel_to_signal_off_db"]) for row in rows[i * len(signals):(i + 1) * len(signals)]] for i in range(len(pumps))])
    status = np.asarray([bool(row["pump_converged"]) for row in rows], dtype=bool).reshape(len(pumps), len(signals))
    # Plot columns sorted by their physical current coordinate, not by unknown dBm.
    order = np.argsort(ratios[:, 0])
    ratio_axis = ratios[order, 0]
    np.savez(run_dir / "kimpa_gain_map.npz", pump_dbm_internal=pumps[order], signal_ghz=signals, I_over_Ic=ratio_axis, gain_db=gains[order], idler_conversion_db=idler[order], pump_converged=status[order])
    metadata = {
        "fixture": args.fixture, "pump_ghz": args.pump_ghz,
        "degenerate_signal_ghz": args.pump_ghz / 2.0,
        "dc_current_a": args.dc_current_a, "environment": args.environment,
        "pump_port": args.pump_port, "source_port": args.source_port,
        "out_port": args.out_port, "ports": sorted(circuit.port_to_index),
        "mixing_order": args.mixing_order,
        "pump_axis_note": "Plots use peak total current / Ic; internal dBm is metadata only.",
        "idler_definition": "physical idler frequency = pump frequency - signal frequency; Floquet idler_m=-(mixing_order-1)",
    }
    (run_dir / "map_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not args.no_plots:
        _write_plots(run_dir, signals, ratio_axis, gains[order], idler[order], status[order])
    valid = [row for row in rows if bool(row["pump_converged"]) and row["gain_status"] == "VALID_SOLVED" and np.isfinite(float(row["gain_db"]))]
    if valid and not args.no_spectrum:
        _run_best_spectrum(args, run_dir, max(valid, key=lambda row: float(row["gain_db"])))
    print(f"wrote={rows_path}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    return 0 if run(build_parser().parse_args(argv)) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
