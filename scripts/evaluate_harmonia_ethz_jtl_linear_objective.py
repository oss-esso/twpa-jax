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

from twpa.io.dataset_builder import load_harmonia_ethz_jtl_linear_dataset
from twpa.io.simulation_schema import SCHEMA_VERSION, write_json


def _as_str_list(values: np.ndarray) -> list[str]:
    return [str(x) for x in values.tolist()]


def _complex_s(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(data["s_real"], dtype=float) + 1j * np.asarray(data["s_imag"], dtype=float)


def _passivity_penalty(s: np.ndarray) -> float:
    penalties: list[float] = []

    for sample in s:
        for matrix in sample:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            excess = np.maximum(singular_values - 1.0, 0.0)
            penalties.append(float(np.mean(excess**2)))

    return float(np.mean(penalties)) if penalties else 0.0


def find_nearest_parameter_index(
    *,
    parameters: np.ndarray,
    parameter_names: list[str],
    parameter_name: str,
    target_value: float,
) -> int:
    if parameter_name not in parameter_names:
        raise ValueError(f"Parameter {parameter_name!r} not found in {parameter_names}")

    idx = parameter_names.index(parameter_name)
    values = np.asarray(parameters[:, idx], dtype=float)
    return int(np.argmin(np.abs(values - target_value)))


def evaluate_dataset_against_target(
    *,
    dataset_npz: str | Path,
    target_lj_h: float = 158.0e-12,
    target_index: int | None = None,
    weight_s21: float = 1.0,
    weight_match: float = 0.25,
    weight_gain: float = 0.05,
    weight_reciprocity: float = 0.01,
    weight_passivity: float = 0.01,
) -> dict[str, Any]:
    data = load_harmonia_ethz_jtl_linear_dataset(dataset_npz)

    parameter_names = _as_str_list(data["parameter_names"])
    parameters = np.asarray(data["parameters"], dtype=float)
    frequency_hz = np.asarray(data["frequency_hz"], dtype=float)
    s = _complex_s(data)
    gain_db = np.asarray(data["gain_db"], dtype=float)

    if s.ndim != 4 or s.shape[2:] != (2, 2):
        raise ValueError(f"Expected S shape (samples, frequency, 2, 2), got {s.shape}")

    n_samples = int(s.shape[0])

    if target_index is None:
        target_index = find_nearest_parameter_index(
            parameters=parameters,
            parameter_names=parameter_names,
            parameter_name="Lj_H",
            target_value=target_lj_h,
        )

    if not (0 <= target_index < n_samples):
        raise ValueError(f"target_index out of range: {target_index}")

    target_s = s[target_index]
    target_gain = gain_db[target_index]

    rows: list[dict[str, Any]] = []

    for sample_idx in range(n_samples):
        sample_s = s[sample_idx]
        sample_gain = gain_db[sample_idx]

        s21_error = float(np.mean(np.abs(sample_s[:, 1, 0] - target_s[:, 1, 0]) ** 2))

        match_error = float(
            0.5
            * (
                np.mean(np.abs(sample_s[:, 0, 0] - target_s[:, 0, 0]) ** 2)
                + np.mean(np.abs(sample_s[:, 1, 1] - target_s[:, 1, 1]) ** 2)
            )
        )

        gain_error = float(np.mean((sample_gain - target_gain) ** 2))

        reciprocity_error = float(np.mean(np.abs(sample_s[:, 1, 0] - sample_s[:, 0, 1]) ** 2))

        passivity_error = _passivity_penalty(sample_s[None, :, :, :])

        total_loss = (
            weight_s21 * s21_error
            + weight_match * match_error
            + weight_gain * gain_error
            + weight_reciprocity * reciprocity_error
            + weight_passivity * passivity_error
        )

        row = {
            "sample_index": int(sample_idx),
            "total_loss": float(total_loss),
            "s21_error": s21_error,
            "match_error": match_error,
            "gain_error": gain_error,
            "reciprocity_error": reciprocity_error,
            "passivity_error": passivity_error,
            "parameters": {
                name: float(parameters[sample_idx, param_idx])
                for param_idx, name in enumerate(parameter_names)
            },
        }

        rows.append(row)

    ranked = sorted(rows, key=lambda row: row["total_loss"])
    best = ranked[0]

    return {
        "schema_version": SCHEMA_VERSION,
        "objective_type": "harmonia_ethz_jtl_linear_target_match",
        "dataset_npz": str(dataset_npz),
        "n_samples": n_samples,
        "n_frequency": int(frequency_hz.shape[0]),
        "frequency_min_hz": float(np.min(frequency_hz)),
        "frequency_max_hz": float(np.max(frequency_hz)),
        "parameter_names": parameter_names,
        "target": {
            "selection": "nearest_Lj_H" if target_index is not None else "index",
            "target_lj_h": float(target_lj_h),
            "target_index": int(target_index),
            "target_parameters": {
                name: float(parameters[target_index, param_idx])
                for param_idx, name in enumerate(parameter_names)
            },
        },
        "weights": {
            "s21": float(weight_s21),
            "match": float(weight_match),
            "gain": float(weight_gain),
            "reciprocity": float(weight_reciprocity),
            "passivity": float(weight_passivity),
        },
        "best": best,
        "ranked": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_WORKSPACE_ROOT
        / "outputs"
        / "datasets"
        / "harmonia_ethz_jtl_linear_jc_v0"
        / "harmonia_ethz_jtl_linear_dataset.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_WORKSPACE_ROOT
        / "outputs"
        / "datasets"
        / "harmonia_ethz_jtl_linear_jc_v0"
        / "objective_summary.json",
    )
    parser.add_argument("--target-lj-h", type=float, default=158.0e-12)
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = evaluate_dataset_against_target(
        dataset_npz=args.dataset,
        target_lj_h=args.target_lj_h,
        target_index=args.target_index,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Evaluated Harmonia ETHZ JTL linear objective")
        print("===========================================")
        print(f"dataset:       {args.dataset}")
        print(f"output:        {args.output}")
        print(f"target_index:  {summary['target']['target_index']}")
        print(f"best_index:    {summary['best']['sample_index']}")
        print(f"best_loss:     {summary['best']['total_loss']}")

    best_index = int(summary["best"]["sample_index"])
    target_index = int(summary["target"]["target_index"])

    return 0 if best_index == target_index else 1


if __name__ == "__main__":
    raise SystemExit(main())