"""Audit all persisted complexity-ladder HB checkpoints without running TD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.pump.validation import validate_production_hb_state  # noqa: E402


def audit_checkpoint(run_dir: Path, frequency_hz: float = 7.9e9) -> dict[str, Any]:
    checkpoint = run_dir / "hb_checkpoint"
    report_path = checkpoint / "pump_report.json"
    solution_path = checkpoint / "pump_solution.npz"
    if not report_path.exists() or not solution_path.exists():
        return {"run": str(run_dir), "classification": "MISSING_CHECKPOINT"}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    data = np.load(solution_path)
    modes = np.asarray(data.get("pump_modes", data.get("harmonics")), dtype=int)
    state = np.asarray(data["X_real"], dtype=float) + 1j * np.asarray(data["X_imag"], dtype=float)
    circuit = load_circuit(run_dir / "circuit")
    branch = make_branch_law(circuit)
    current = float(report.get("metadata", {}).get("pump_current_a", float("nan")))
    validation = validate_production_hb_state(
        circuit, branch, frequency_hz=frequency_hz, pump_port=1,
        pump_current_a=current, modes=modes, state=state,
        nt=max(2 * int(modes.max()) + 1, 40), metadata=report.get("metadata", {}),
    )
    has_current_provenance = report.get("metadata", {}).get("hb_solver_family") == "production_pump"
    if has_current_provenance and validation["checkpoint_validated"]:
        classification = "VALID_CURRENT_PRODUCTION"
    elif validation["checkpoint_validated"]:
        classification = "VALIDATED_LEGACY"
    else:
        classification = "INVALID_LEGACY"
    return {
        "run": str(run_dir),
        "classification": classification,
        "production_hb_residual_rel": validation.get("production_hb_residual_rel"),
        "checkpoint_validated": validation["checkpoint_validated"],
        "hb_solver_family": validation["hb_solver_family"],
        "state_shape": validation["state_shape"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "complexity_ladder_provenance.json")
    args = parser.parse_args(argv)
    runs = sorted(path for path in args.root.glob("ladder_*") if (path / "hb_checkpoint").is_dir())
    result = [audit_checkpoint(path) for path in runs]
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
