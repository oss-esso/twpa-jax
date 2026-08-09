"""Does fabrication disorder wash out the localized standing-wave depletion?

Follow-up to [[2c-spatial-depletion-localized-not-uniform]]: the ideal (zero
scatter, zero loss) live circuit shows local pump depletion reaching -8.65 dB
(86%) at one branch near the output coupler, while the lumped/port-averaged
depletion is only -0.32 dB (7%) -- interpreted as a standing-wave
interference null, sharp because the simulated device is perfectly uniform.

Real hardware has junction/geometry scatter (Lj, Cj, Cg vary cell-to-cell)
and finite dielectric loss, both of which should broaden and shallow any
coherent interference null. Tests this directly by rerunning
``exp51_spatial_depletion_profile`` across the disorder ladder already built
in ``designs/campaign_diss/`` (2c_base = zero scatter/loss control,
2c_sc{1,5,10}pct = Lj scatter sigma 1/5/10% with plasma_locked Cj, same
topology and junction count as designs/ipm_2c_fixed -- see
campaign_diss/campaign_manifest.json).

Hypothesis under test: if disorder is the reason hardware doesn't show this
behavior, the worst-branch local depletion should shrink toward the lumped
average as scatter sigma increases, and/or the worst-branch location should
stop being pinned to one fixed branch (since disorder randomizes the
standing-wave interference pattern run to run / cell to cell).
"""

from __future__ import annotations

import json
from pathlib import Path

import exp51_spatial_depletion_profile as spatial  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "outputs" / "exp52_disorder_ladder"

VARIANTS = [
    ("designs/campaign_diss/2c_base", "2c_base (control, 0% scatter, 0 loss)"),
    ("designs/campaign_diss/2c_sc1pct", "2c_sc1pct (1% Lj scatter)"),
    ("designs/campaign_diss/2c_sc5pct", "2c_sc5pct (5% Lj scatter)"),
    ("designs/campaign_diss/2c_sc10pct", "2c_sc10pct (10% Lj scatter)"),
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for circuit_dir, description in VARIANTS:
        label = Path(circuit_dir).name
        print(f"\n{'=' * 70}\n{description}\n{'=' * 70}")
        try:
            report = spatial.main(circuit_dir=circuit_dir, label=label)
        except RuntimeError as exc:
            print(f"FAILED: {exc}")
            report = {"circuit_dir": circuit_dir, "label": label, "failed": str(exc)}
        reports.append(report)

    (OUTPUT / "disorder_ladder_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )

    print(f"\n\n{'=' * 90}")
    print(f"{'label':<28} {'lumped_dB':>10} {'local_med':>10} {'local_worst':>12} "
          f"{'worst_branch':>13} {'actual_comp':>12}")
    for r in reports:
        if r.get("failed"):
            print(f"{r['label']:<28} FAILED: {r['failed']}")
            continue
        print(f"{r['label']:<28} {r['lumped_all_port_pump_depletion_db']:>10.4f} "
              f"{r['local_depletion_db_median']:>10.4f} "
              f"{r['local_depletion_db_min']:>12.4f} "
              f"{r['worst_branch']:>13d} "
              f"{r['actual_compression_db']:>12.4f}")
    print(f"\nwrote {OUTPUT / 'disorder_ladder_report.json'}")


if __name__ == "__main__":
    main()
