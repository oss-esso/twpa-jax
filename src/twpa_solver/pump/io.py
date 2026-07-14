from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver.pump.problem import FullPumpProblem
from twpa_solver.pump.solver import StepReport

def summarize_solution(problem: FullPumpProblem, X: np.ndarray) -> dict[str, float]:
    x_t = problem.grid.synthesize(X)
    psi_t = problem.branch_flux_time(X)
    i_t = problem.branch.current(psi_t)

    out = {
        "x_rms": float(np.sqrt(np.mean(x_t * x_t))),
        "x_max_abs": float(np.max(np.abs(x_t))),
        "branch_psi_rms": float(np.sqrt(np.mean(psi_t * psi_t))),
        "branch_psi_max_abs": float(np.max(np.abs(psi_t))),
        "branch_i_rms": float(np.sqrt(np.mean(i_t * i_t))),
        "branch_i_max_abs": float(np.max(np.abs(i_t))),
    }

    for h in range(problem.H):
        out[f"X_h{h + 1}_norm"] = float(np.linalg.norm(X[h]))

    return out


def write_results(
    outdir: str | Path,
    X: np.ndarray,
    reports: list[StepReport],
    solution_summary: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)

    pump_modes = np.asarray(
        metadata.get("pump_modes", list(range(1, X.shape[0] + 1))),
        dtype=np.int64,
    )
    # float32 halves the file; the pump phasors are ~1e-16 in magnitude so
    # float32's ~1e-7 relative precision is far below any downstream tolerance
    # (gain-map RMS targets ~1e-3 dB). savez_compressed then trims a little more.
    np.savez_compressed(
        d / "pump_solution.npz",
        X_real=X.real.astype(np.float32),
        X_imag=X.imag.astype(np.float32),
        harmonics=pump_modes,
        pump_modes=pump_modes,
    )

    report_json = {
        "metadata": metadata,
        "solution_summary": solution_summary,
        "reports": [asdict(r) for r in reports],
        "final_status": "VALID_CONVERGED"
        if reports and reports[-1].converged and abs(reports[-1].source_scale - 1.0) < 1e-12
        else "FAIL",
    }

    with open(d / "pump_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    print(f"wrote_solution={d / 'pump_solution.npz'}")
    print(f"wrote_report={d / 'pump_report.json'}")
