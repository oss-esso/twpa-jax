# experiments/exp14_diss_loss_study.py
"""Loss-convention study for jc_fqjtwpa_diss.

Separates where dielectric loss (complex C) enters the gain by toggling
--real-capacitance independently in the pump solve (exp08) and the linearized
gain solve (exp09), and compares each config's 21-pt curve to the JC reference.

Configs (all at the correct boosted pump current 2.475e-6 = 2 x 1.2375e-6):
    A  lossy pump  + lossy gain     (the physical "loss everywhere" config)
    B  lossless pump + lossy gain
    C  lossy pump  + lossless gain
    D  lossless pump + lossless gain

Finding (see REPORT.md): A is the correct scheme (loss in both); pump-loss
dominates (B/D explode). The A->JC residual (~0.8 dB) is a frequency-dependent
loss SHAPE difference, not amplitude (de-tuning current matches the peak but
worsens max_abs), pointing at JC's per-frequency lossy linearization.

This script only re-runs the comparison from existing artifacts; it does not
re-solve. Run the four configs with exp08/exp09 --real-capacitance first.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

JC_REF = "outputs/exp14_jc_refs/jc_fqjtwpa_diss_curve_21pt.csv"
CONFIGS = {
    "A_lossyP_lossyG": "outputs/_diss_A/gain_sweep.csv",
    "B_losslessP_lossyG": "outputs/_diss_B/gain_sweep.csv",
    "C_lossyP_losslessG": "outputs/_diss_C/gain_sweep.csv",
    "D_losslessP_losslessG": "outputs/_diss_D/gain_sweep.csv",
}
OUT = "outputs/exp14_diss_loss_study/diss_loss_study.csv"


def load_curve(p: str) -> dict[float, float]:
    d: dict[float, float] = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[round(float(r["signal_ghz"]), 3)] = float(r["gain_db"])
    return d


def main() -> None:
    jc = load_curve(JC_REF)
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in CONFIGS.items():
        if not Path(path).exists():
            rows.append((name, "MISSING", "", "", ""))
            continue
        py = load_curve(path)
        common = sorted(set(py) & set(jc))
        d = np.array([py[f] - jc[f] for f in common])
        rows.append((
            name,
            round(float(max(py.values())), 3),
            round(float(np.sqrt(np.mean(d**2))), 4),
            round(float(np.max(np.abs(d))), 4),
            round(float(max(jc.values())), 3),
        ))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "py_peak_db", "rms_err_db", "max_abs_err_db", "jc_peak_db"])
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
