#!/usr/bin/env python3
"""Compile and linearly probe the previously untested rf-SQUID design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from twpa_solver.builders.ipm import LossSpec, build_matrices
from twpa_solver.core import CircuitMatrices, solve_linear_scattering
from twpa_solver.design import compile_design, load_design


def build_rf_squid(design_path: Path) -> tuple[CircuitMatrices, dict]:
    design = compile_design(load_design(design_path), strict=True)
    matrices = build_matrices(design.elements, LossSpec(0.0))
    circuit = CircuitMatrices(
        C=matrices["C"], G=matrices["G"], K=matrices["K"],
        Bphi=matrices["Bphi"], Ic=matrices["Ic"], Lj=matrices["Lj"],
        nodes=matrices["nodes"], port_to_index=matrices["port_vectors"],
        metadata={"design": design.name, "elements": len(design.elements)},
    )
    return circuit, {"name": design.name, "elements": len(design.elements),
                     "nodes": circuit.node_count, "branches": circuit.branch_count,
                     "ports": circuit.port_to_index}


def run_smoke(design_path: Path, frequencies_ghz: np.ndarray) -> dict:
    circuit, summary = build_rf_squid(design_path)
    if circuit.Bphi.nnz == 0:
        raise RuntimeError("rf-SQUID design has an empty Bphi/DC-flux path")
    rows = []
    for frequency in frequencies_ghz:
        result = solve_linear_scattering(
            circuit, frequency_hz=float(frequency) * 1e9,
            source_port=1, out_port=2, source_current_a=1.0,
        )
        rows.append({"frequency_ghz": float(frequency), "s21_abs": result.s_abs,
                     "s21_db": result.s_db})
    magnitudes = np.asarray([row["s21_abs"] for row in rows])
    summary.update({"bphi_nnz": int(circuit.Bphi.nnz), "frequency_rows": rows,
                    "s21_abs_min": float(np.min(magnitudes)),
                    "s21_abs_max": float(np.max(magnitudes)),
                    "s21_nonzero": bool(np.any(magnitudes > 1e-12)),
                    "dc_flux_path": True})
    if not summary["s21_nonzero"]:
        raise RuntimeError("linear rf-SQUID probe returned identically zero S21")
    return summary


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=root / "designs" / "rf_squid_2393_3wm.yaml")
    parser.add_argument("--start-ghz", type=float, default=4.0)
    parser.add_argument("--stop-ghz", type=float, default=12.0)
    parser.add_argument("--num", type=int, default=161)
    parser.add_argument("--output", type=Path, default=root / "outputs" / "chaos" / "phase0" / "rf_squid_smoke.json")
    args = parser.parse_args(argv)
    if args.num < 2:
        raise SystemExit("--num must be at least 2")
    result = run_smoke(args.design, np.linspace(args.start_ghz, args.stop_ghz, args.num))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "nodes": result["nodes"],
                      "branches": result["branches"], "s21_nonzero": result["s21_nonzero"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
