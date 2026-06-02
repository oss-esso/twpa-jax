from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpa.calibration.dataset_objectives import evaluate_harmonia_jtl_linear_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_WORKSPACE_ROOT / "outputs" / "datasets" / "harmonia_jtl_linear_jc_v0" / "harmonia_jtl_linear_dataset.npz")
    parser.add_argument("--output", type=Path, default=_WORKSPACE_ROOT / "outputs" / "datasets" / "harmonia_jtl_linear_jc_v0" / "objective_summary.json")
    parser.add_argument("--target-lj-h", type=float, default=1.0e-9)
    args = parser.parse_args()
    summary = evaluate_harmonia_jtl_linear_dataset(dataset_npz=args.dataset, output_json=args.output, target_lj_h=args.target_lj_h)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
