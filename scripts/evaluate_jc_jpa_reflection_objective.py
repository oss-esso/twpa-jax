"""
Evaluate the first JosephsonCircuits-backed one-port reflection objective.

Default behavior:
    - load outputs/datasets/jc_jpa_reflection_v0/jc_jpa_reflection_dataset.npz
    - choose the sample with pump_current_a closest to 5.65e-9 A as target
    - compute objective loss for every sample
    - write objective_summary.json

Example
-------
python scripts/evaluate_jc_jpa_reflection_objective.py
python scripts/evaluate_jc_jpa_reflection_objective.py --target-pump-current-a 2e-9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpa.calibration.objectives import (
    OnePortReflectionObjectiveWeights,
    evaluate_jc_reflection_dataset_against_target,
)
from twpa.io.dataset_builder import load_jc_jpa_reflection_dataset
from twpa.io.simulation_schema import SCHEMA_VERSION, write_json


def _decode_names(names: np.ndarray) -> list[str]:
    out: list[str] = []
    for name in names:
        if isinstance(name, bytes):
            out.append(name.decode("utf-8"))
        else:
            out.append(str(name))
    return out


def build_jc_reflection_objective_summary(
    *,
    dataset_npz: Path,
    output_dir: Path,
    target_pump_current_a: float = 5.65e-9,
) -> dict[str, Any]:
    data = load_jc_jpa_reflection_dataset(dataset_npz)

    parameter_names = _decode_names(data["parameter_names"])
    parameters = np.asarray(data["parameters"], dtype=float)
    frequency_hz = np.asarray(data["frequency_hz"], dtype=float)
    s11_complex = np.asarray(data["s11_real"], dtype=float) + 1j * np.asarray(data["s11_imag"], dtype=float)
    reflection_db = np.asarray(data["reflection_db"], dtype=float)

    if "pump_current_a" not in parameter_names:
        raise ValueError(
            f"Dataset parameter_names does not include pump_current_a: {parameter_names}"
        )

    pump_idx = parameter_names.index("pump_current_a")
    pump_currents = parameters[:, pump_idx]

    target_index = int(np.argmin(np.abs(pump_currents - target_pump_current_a)))

    rows = evaluate_jc_reflection_dataset_against_target(
        parameters=parameters,
        frequency_hz=frequency_hz,
        s11_complex=s11_complex,
        reflection_db=reflection_db,
        target_index=target_index,
        weights=OnePortReflectionObjectiveWeights(),
    )

    ranked = sorted(rows, key=lambda row: row["total_loss"])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "objective_type": "jc_jpa_reflection_first_objective",
        "dataset_npz": str(dataset_npz),
        "output_dir": str(output_dir),
        "parameter_names": parameter_names,
        "n_samples": int(parameters.shape[0]),
        "n_frequency": int(frequency_hz.shape[0]),
        "target_pump_current_a": float(target_pump_current_a),
        "target_index": target_index,
        "target_parameters": parameters[target_index].tolist(),
        "target_peak_reflection_db": ranked[0]["target_peak_reflection_db"],
        "target_peak_frequency_hz": ranked[0]["target_peak_frequency_hz"],
        "ranked_losses": ranked,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "objective_summary.json", summary)

    return summary


def print_human_summary(summary: dict[str, Any]) -> None:
    print("JosephsonCircuits JPA reflection objective")
    print("==========================================")
    print(f"dataset_npz:             {summary['dataset_npz']}")
    print(f"n_samples:               {summary['n_samples']}")
    print(f"n_frequency:             {summary['n_frequency']}")
    print(f"target_index:            {summary['target_index']}")
    print(f"target_pump_current_a:   {summary['target_pump_current_a']}")
    print(f"target_peak_frequency:   {summary['target_peak_frequency_hz']}")
    print(f"target_peak_reflection:  {summary['target_peak_reflection_db']}")
    print()
    print("Ranked losses")
    print("-------------")

    for row in summary["ranked_losses"]:
        print(
            f"sample={row['sample_index']} "
            f"loss={row['total_loss']:.6e} "
            f"s11={row['s11_complex_loss']:.6e} "
            f"refl_db={row['reflection_db_loss']:.6e} "
            f"peak_f={row['peak_frequency_loss']:.6e} "
            f"peak_db={row['peak_reflection_db_loss']:.6e} "
            f"params={row['parameters']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_WORKSPACE_ROOT
        / "outputs"
        / "datasets"
        / "jc_jpa_reflection_v0"
        / "jc_jpa_reflection_dataset.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_WORKSPACE_ROOT
        / "outputs"
        / "objectives"
        / "jc_jpa_reflection_v0",
    )
    parser.add_argument("--target-pump-current-a", type=float, default=5.65e-9)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = build_jc_reflection_objective_summary(
        dataset_npz=args.dataset,
        output_dir=args.output_dir,
        target_pump_current_a=args.target_pump_current_a,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)

    best = summary["ranked_losses"][0]
    return 0 if best["sample_index"] == summary["target_index"] else 1


if __name__ == "__main__":
    raise SystemExit(main())