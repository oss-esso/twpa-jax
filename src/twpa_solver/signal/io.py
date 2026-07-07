from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver.pump.basis import PumpBasis, load_pump_basis_from_solution
from twpa_solver.signal.gain import GainResult, complex_to_pair

def infer_circuit_dir_from_pump_report(pump_dir: Path) -> str | None:
    report_path = pump_dir / "pump_report.json"
    if not report_path.exists():
        return None
    try:
        import json
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return None

    # Common report layouts.
    for container in (report, report.get("metadata", {}), report.get("settings", {})):
        if isinstance(container, dict):
            value = container.get("circuit_dir") or container.get("ipm_dir")
            if value:
                return str(value)

    return None


@dataclass
class PumpSolution:
    X: np.ndarray
    omega_p: float
    pump_freq_ghz: float
    harmonics: int
    nt_original: int
    metadata: dict[str, Any]
    modes: list[int]
    basis: PumpBasis


def load_pump(pump_dir: str | Path, fallback_pump_freq_ghz: float) -> PumpSolution:
    d = Path(pump_dir)

    sol_path = d / "pump_solution.npz"
    if not sol_path.exists():
        raise FileNotFoundError(f"missing pump solution: {sol_path}")

    report_path = d / "pump_report.json"
    metadata: dict[str, Any] = {}

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        metadata = rep.get("metadata", {})
    else:
        rep = {}

    pump_freq_ghz = float(metadata.get("pump_freq_ghz", fallback_pump_freq_ghz))
    fallback_omega_p = 2.0 * math.pi * pump_freq_ghz * 1e9

    # Mode-aware load: reconstruct the exact pump basis (dense or odd-phasor).
    X, basis = load_pump_basis_from_solution(d, fallback_omega_p=fallback_omega_p)
    omega_p = basis.omega_p if basis.omega_p > 0.0 else fallback_omega_p
    nt_original = int(metadata.get("nt", 0))

    return PumpSolution(
        X=X,
        omega_p=omega_p,
        pump_freq_ghz=pump_freq_ghz,
        harmonics=X.shape[0],
        nt_original=nt_original,
        metadata=metadata,
        modes=list(basis.modes),
        basis=basis,
    )


def csv_header() -> list[str]:
    return [
        "status",
        "signal_ghz",
        "gain_db",
        "s_param_abs",
        "gain_vs_off_db",
        "gain_vs_pumpdiag_db",
        "idler_power_rel_to_signal_off_db",
        "linear_rel_residual",
        "conversion_unknowns",
        "matrix_nnz",
        "assemble_runtime_s",
        "factor_solve_runtime_s",
        "vout_on_real",
        "vout_on_imag",
        "vout_off_real",
        "vout_off_imag",
    ]


def result_to_csv_row(r: GainResult) -> list[Any]:
    return [
        r.status,
        r.signal_ghz,
        r.gain_db,
        r.s_param_abs,
        r.gain_vs_off_db,
        r.gain_vs_pumpdiag_db,
        r.idler_power_rel_to_signal_off_db,
        r.linear_rel_residual,
        r.conversion_unknowns,
        r.matrix_nnz,
        r.assemble_runtime_s,
        r.factor_solve_runtime_s,
        r.vout_on.real,
        r.vout_on.imag,
        r.vout_off.real,
        r.vout_off.imag,
    ]


def print_sweep_row(r: GainResult) -> None:
    idb = "" if r.idler_power_rel_to_signal_off_db is None else f"{r.idler_power_rel_to_signal_off_db:.9g}"
    print(
        f"{r.signal_ghz:.12g},"
        f"{r.status},"
        f"{r.gain_db:.9g},"
        f"{r.s_param_abs:.9g},"
        f"{r.gain_vs_off_db:.9g},"
        f"{r.gain_vs_pumpdiag_db:.9g},"
        f"{idb},"
        f"{r.linear_rel_residual:.3e},"
        f"{r.factor_solve_runtime_s:.6f}"
    )


def write_outputs(outdir: str | Path, rows: list[GainResult], metadata: dict[str, Any]) -> None:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)

    csv_path = d / "gain_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_header())
        for r in rows:
            w.writerow(result_to_csv_row(r))

    js_path = d / "gain_report.json"
    payload = {
        "metadata": metadata,
        "results": [
            {
                **{
                    k: v
                    for k, v in r.__dict__.items()
                    if not isinstance(v, complex)
                },
                "vout_on": complex_to_pair(r.vout_on),
                "vout_off": complex_to_pair(r.vout_off),
                "vout_pumpdiag": complex_to_pair(r.vout_pumpdiag),
                "vout_idler": None if r.vout_idler is None else complex_to_pair(r.vout_idler),
            }
            for r in rows
        ],
    }

    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote_csv={csv_path}")
    print(f"wrote_report={js_path}")
