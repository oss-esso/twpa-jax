"""
Evaluate the first calibration objective on a linear S-parameter dataset.

Default behavior:
    - load outputs/datasets/linear_sparams_v0/linear_sparams_dataset.npz
    - choose the sample with z_line_ohm closest to 50 ohm as target
    - compute losses for every sample
    - write objective_summary.json

Example
-------
python scripts/evaluate_linear_sparams_objective.py
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
    SParameterObjectiveWeights,
    evaluate_dataset_against_target,
)
from twpa.io.dataset_builder import load_linear_sparams_dataset
from twpa.io.simulation_schema import SCHEMA_VERSION, write_json


def _decode_names(names: np.ndarray) -> list[str]:
    out = []
    for name in names:
        if isinstance(name, bytes):
            out.append(name.decode("utf-8"))
        else:
            out.append(str(name))
    return out


def build_objective_summary(
    *,
    dataset_npz: Path,
    output_dir: Path,
    target_z_line_ohm: float = 50.0,
) -> dict[str, Any]:
    data = load_linear_sparams_dataset(dataset_npz)

    parameter_names = _decode_names(data["parameter_names"])
    parameters = np.asarray(data["parameters"], dtype=float)
    frequency_hz = np.asarray(data["frequency_hz"], dtype=float)
    s_complex = np.asarray(data["s_real"], dtype=float) + 1j * np.asarray(data["s_imag"], dtype=float)
    gain_db = np.asarray(data["gain_db"], dtype=float)

    if "z_line_ohm" not in parameter_names:
        raise ValueError(f"Dataset parameter_names does not include z_line_ohm: {parameter_names}")

    z_idx = parameter_names.index("z_line_ohm")
    z_lines = parameters[:, z_idx]

    target_index = int(np.argmin(np.abs(z_lines - target_z_line_ohm)))

    rows = evaluate_dataset_against_target(
        parameters=parameters,
        frequency_hz=frequency_hz,
        s_complex=s_complex,
        gain_db=gain_db,
        target_index=target_index,
        weights=SParameterObjectiveWeights(),
    )

    ranked = sorted(rows, key=lambda row: row["total_loss"])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "objective_type": "linear_sparams_first_objective",
        "dataset_npz": str(dataset_npz),
        "output_dir": str(output_dir),
        "parameter_names": parameter_names,
        "n_samples": int(parameters.shape[0]),
        "n_frequency": int(frequency_hz.shape[0]),
        "target_z_line_ohm": float(target_z_line_ohm),
        "target_index": target_index,
        "target_parameters": parameters[target_index].tolist(),
        "ranked_losses": ranked,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "objective_summary.json", summary)

    return summary


def print_human_summary(summary: dict[str, Any]) -> None:
    print("Linear S-parameter objective")
    print("============================")
    print(f"dataset_npz:        {summary['dataset_npz']}")
    print(f"n_samples:          {summary['n_samples']}")
    print(f"n_frequency:        {summary['n_frequency']}")
    print(f"target_index:       {summary['target_index']}")
    print(f"target_parameters:  {summary['target_parameters']}")
    print()
    print("Ranked losses")
    print("-------------")

    for row in summary["ranked_losses"]:
        print(
            f"sample={row['sample_index']} "
            f"loss={row['total_loss']:.6e} "
            f"params={row['parameters']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_WORKSPACE_ROOT / "outputs" / "datasets" / "linear_sparams_v0" / "linear_sparams_dataset.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_WORKSPACE_ROOT / "outputs" / "objectives" / "linear_sparams_v0",
    )
    parser.add_argument("--target-z-line-ohm", type=float, default=50.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = build_objective_summary(
        dataset_npz=args.dataset,
        output_dir=args.output_dir,
        target_z_line_ohm=args.target_z_line_ohm,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)

    best = summary["ranked_losses"][0]
    return 0 if best["sample_index"] == summary["target_index"] else 1


if __name__ == "__main__":
    raise SystemExit(main())