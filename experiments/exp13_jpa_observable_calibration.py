from __future__ import annotations

import csv
import math
from pathlib import Path

CSV = Path("outputs/exp13_jpa_observable_calibration_501/gain_sweep.csv")

JC = {
    "gain_db_max": 13.300727259957004,
    "gain_db_mean": 0.9745846837647946,
    "gain_db_min": 0.0027159157639631156,
    "peak_frequency_ghz": 4.75,
}

rows = []
with CSV.open("r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

def f(x: str) -> float:
    return float(x)

def summarize(col: str) -> dict:
    vals = [(f(r["signal_ghz"]), f(r[col])) for r in rows]
    peak_f, vmax = max(vals, key=lambda t: t[1])
    vmin = min(v for _, v in vals)
    vmean = sum(v for _, v in vals) / len(vals)
    return {
        "column": col,
        "max": vmax,
        "mean": vmean,
        "min": vmin,
        "peak_frequency_ghz": peak_f,
        "err_max": vmax - JC["gain_db_max"],
        "err_mean": vmean - JC["gain_db_mean"],
        "err_min": vmin - JC["gain_db_min"],
        "err_peak_ghz": peak_f - JC["peak_frequency_ghz"],
    }

candidates = [
    "gain_db",
    "gain_vs_off_db",
    "gain_vs_pumpdiag_db",
]

print("JC reference:")
for k, v in JC.items():
    print(f"  {k}: {v}")
print()

for col in candidates:
    s = summarize(col)
    print(f"candidate={col}")
    print(f"  max={s['max']:.9g}  err={s['err_max']:.9g}")
    print(f"  mean={s['mean']:.9g} err={s['err_mean']:.9g}")
    print(f"  min={s['min']:.9g}  err={s['err_min']:.9g}")
    print(f"  peak={s['peak_frequency_ghz']:.12g} err={s['err_peak_ghz']:.9g}")
    print()
