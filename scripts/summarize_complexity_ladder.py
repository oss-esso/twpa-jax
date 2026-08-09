"""Create a compact comparison report from completed ladder/H3 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _summary(path: Path) -> dict:
    return json.loads((path / "ladder_summary.json").read_text(encoding="utf-8"))


def _row(label: str, path: Path, n_jj: int) -> dict[str, object]:
    summary = _summary(path)
    t = summary["transient"]
    obs = t.get("branch_transfer") or {}
    metrics = t["stroboscopic"]
    data = np.load(path / "transient" / "transient_observables.npz")
    return {
        "topology": label, "n_jj": n_jj,
        "target_current_a": t["target_current_a"],
        "classification": t["classification"],
        "final_status": t["final_status"],
        "r_j": float(np.max(data["max_abs_sin_phi"])),
        "phi_max": float(np.max(data["max_abs_phi"])),
        "min_cos": float(np.min(data["min_cos_phi"])),
        "strobe_tail_max": metrics["tail_max"],
        "phase_winding_cycles": t["mean_phase_winding_cycles"],
        "hb_transfer_converged": obs.get("hb_converged"),
        "hb_transfer_coeff_rel": obs.get("hb_coeff_rel"),
        "notes": "lossless JJ; production values; 50 ohm terminations",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "complexity_ladder_report")
    args = parser.parse_args()
    runs = [
        ("single JJ, 1 Ic", ROOT / "ladder_single_jj_h3", 1),
        ("single JJ, 2 Ic", ROOT / "ladder_single_jj_2ic", 1),
        ("single JJ, 3 Ic", ROOT / "ladder_single_jj_3ic", 1),
        ("uniform JTL N=8, 1.5 Ic", ROOT / "ladder_jtl8_h3", 8),
        ("uniform JTL N=8, 2 Ic", ROOT / "ladder_jtl8_2ic", 8),
        ("uniform JTL N=16, 1.5 Ic", ROOT / "ladder_jtl16_15ic", 16),
    ]
    rows = [_row(label, path, n_jj) for label, path, n_jj in runs]
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.outdir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([row["n_jj"] for row in rows], [row["r_j"] for row in rows], "o")
    ax.set(xlabel="number of nonlinear junctions", ylabel="max |sin(phi)|")
    fig.tight_layout(); fig.savefig(args.outdir / "rj_vs_complexity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([row["n_jj"] for row in rows], [row["target_current_a"] for row in rows], "o")
    ax.set(xlabel="number of nonlinear junctions", ylabel="source current (A)")
    fig.tight_layout(); fig.savefig(args.outdir / "source_drive_vs_complexity.png", dpi=150); plt.close(fig)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
