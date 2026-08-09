"""Calculate full-sideband quantum efficiency for campaign dissipation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from twpa_solver.core.circuit import load_circuit
from twpa_solver.core.linear import port_s_from_unit_current_response
from twpa_solver.signal.floquet import (
    build_signal_schur_partition,
    sideband_list,
    solve_gain_one_schur,
)
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.io import load_pump
from twpa_solver.signal.quantum_efficiency import calc_qe, calc_qe_ideal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "campaign_diss"
DEFAULT_OUTPUT = ROOT / "outputs" / "quantum_efficiency_campaign_diss.json"


def _valid_candidates(run_dir: Path) -> list[tuple[Path, dict, dict]]:
    candidates: list[tuple[Path, dict, dict]] = []
    for report_path in run_dir.rglob("gain_report.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metadata = report.get("metadata", {})
        for result in report.get("results", []):
            if result.get("status") in {"VALID_SOLVED", "VALID_CONVERGED"}:
                candidates.append((report_path, metadata, result))
    return candidates


def _best_candidate(run_dir: Path) -> tuple[Path, dict, dict]:
    candidates = _valid_candidates(run_dir)
    if not candidates:
        raise RuntimeError(f"no valid gain candidates found in {run_dir}")
    return max(candidates, key=lambda item: float(item[2].get("gain_db", -np.inf)))


def _signal_row(
    circuit_dir: Path,
    pump_dir: Path,
    metadata: dict,
    result: dict,
) -> tuple[list[int], np.ndarray]:
    circuit = load_circuit(circuit_dir)
    pump_freq = float(metadata["pump_freq_ghz"])
    pump = load_pump(pump_dir, fallback_pump_freq_ghz=pump_freq)
    signal_ghz = float(result["signal_ghz"])
    sidebands = int(metadata["sidebands"])
    source_port = int(metadata.get("source_port", 1))
    out_port = int(metadata.get("out_port", 2))
    source_index = circuit.port_to_index[source_port]
    out_index = circuit.port_to_index[out_port]
    modes = sideband_list(sidebands)
    max_ell = max(abs(m - q) for m in modes for q in modes)
    gamma_hat = compute_gamma_hat(
        circuit=circuit,
        pump=pump,
        max_ell=max_ell,
        gamma_nt=int(metadata.get("gamma_nt", 96)),
    )
    khat = build_khat(circuit.Bphi, gamma_hat, drop_tol=0.0)
    gamma_off = circuit.Ic / circuit.phi0
    khat_off_0 = (
        circuit.Bphi
        @ sp.diags(gamma_off, offsets=0, format="csr")
        @ circuit.Bphi.T
    ).astype(np.complex128).tocsr()
    schur_part = build_signal_schur_partition(
        circuit,
        pump.omega_p,
        signal_ghz,
        sidebands,
        source_index,
        out_index,
        loss_model="current_complex_c",
    )
    row = np.zeros(len(modes), dtype=np.complex128)
    signal_m = int(metadata.get("signal_m", 0))
    other_m = int(metadata.get("idler_m", -2))
    for i, input_m in enumerate(modes):
        if input_m == signal_m:
            idler_m = other_m
        else:
            idler_m = signal_m
        solved = solve_gain_one_schur(
            circuit=circuit,
            khat=khat,
            khat_off_0=khat_off_0,
            omega_p=pump.omega_p,
            signal_ghz=signal_ghz,
            sidebands=sidebands,
            signal_m=input_m,
            idler_m=idler_m,
            source_index=source_index,
            out_index=out_index,
            source_current_a=1.0,
            source_port=source_port,
            out_port=out_port,
            z0_ohm=float(metadata.get("z0_ohm", 50.0)),
            include_baselines=False,
            schur_part=schur_part,
        )
        voltage = solved.vout_on if input_m == signal_m else solved.vout_idler
        row[i] = port_s_from_unit_current_response(
            voltage,
            source_port=source_port,
            out_port=out_port,
            z0_ohm=float(metadata.get("z0_ohm", 50.0)),
        )
    freqs = np.abs(signal_ghz + np.asarray(modes) * pump_freq)
    return modes, row * np.sqrt(freqs / signal_ghz)


def calculate_run(run_dir: Path) -> dict[str, object]:
    report_path, metadata, result = _best_candidate(run_dir)
    design_dir = ROOT / metadata["ipm_dir"]
    pump_dir = ROOT / metadata["pump_dir"]
    modes, row = _signal_row(design_dir, pump_dir, metadata, result)
    signal_index = modes.index(int(metadata.get("signal_m", 0)))
    qe = calc_qe(row.reshape(1, -1))[0]
    qe_signal = float(qe[signal_index])
    qe_ideal = float(calc_qe_ideal(np.array([[row[signal_index]]]))[0, 0])
    return {
        "run": run_dir.name,
        "report": str(report_path.relative_to(ROOT)),
        "pump_dir": str(pump_dir.relative_to(ROOT)),
        "signal_ghz": float(result["signal_ghz"]),
        "pump_freq_ghz": float(metadata["pump_freq_ghz"]),
        "gain_db": float(result["gain_db"]),
        "sidebands": len(modes),
        "qe_signal": qe_signal,
        "qe_ideal_signal": qe_ideal,
        "efficiency_qe_over_ideal": qe_signal / qe_ideal,
        "qe_signal_leq_ideal": qe_signal <= qe_ideal + 1e-9,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = [calculate_run(run_dir) for run_dir in sorted(args.input.iterdir()) if run_dir.is_dir()]
    args.output.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['run']}: QE={row['qe_signal']:.6f} efficiency={row['efficiency_qe_over_ideal']:.6f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
