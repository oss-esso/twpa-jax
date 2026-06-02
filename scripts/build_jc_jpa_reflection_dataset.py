"""
Build an ML-ready dataset from a JosephsonCircuits JPA reflection campaign.

Example
-------
python scripts/build_jc_jpa_reflection_dataset.py ^
  --registry D:/Projects/Thesis/outputs/campaigns/jc_jpa_reflection_smoke/runs.csv ^
  --output-dir D:/Projects/Thesis/outputs/datasets/jc_jpa_reflection_v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpa.io.dataset_builder import build_jc_jpa_reflection_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=_WORKSPACE_ROOT
        / "outputs"
        / "campaigns"
        / "jc_jpa_reflection_smoke"
        / "runs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_WORKSPACE_ROOT
        / "outputs"
        / "datasets"
        / "jc_jpa_reflection_v0",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    built = build_jc_jpa_reflection_dataset(
        registry_csv=args.registry,
        output_dir=args.output_dir,
    )

    summary = {
        "dataset_npz": str(built.dataset_npz),
        "summary_json": str(built.summary_json),
        "n_samples": built.n_samples,
        "n_frequency": built.n_frequency,
        "parameter_names": built.parameter_names,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Built JosephsonCircuits JPA reflection dataset")
        print("==============================================")
        print(f"dataset_npz:     {built.dataset_npz}")
        print(f"summary_json:    {built.summary_json}")
        print(f"n_samples:       {built.n_samples}")
        print(f"n_frequency:     {built.n_frequency}")
        print(f"parameter_names: {built.parameter_names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())