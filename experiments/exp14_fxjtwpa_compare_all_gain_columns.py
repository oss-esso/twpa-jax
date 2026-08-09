from pathlib import Path
import csv
import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
PY = ROOT / "outputs" / "exp14_fxjtwpa_gain_from_pump_solved_6_10" / "gain_sweep.csv"
JC = ROOT / "outputs" / "exp14_jc_refs" / "jc_fxjtwpa_curve_21pt.csv"

CANDIDATE_COLUMNS = [
    "gain_db",
    "gain_vs_off_db",
    "gain_vs_pumpdiag_db",
    "idler_rel_db",
]

def load_py(path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            freq = round(float(r["signal_ghz"]), 6)
            rows[freq] = r
    return rows

def load_jc(path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            freq = round(float(r["signal_ghz"]), 6)
            rows[freq] = float(r["gain_db"])
    return rows

py = load_py(PY)
jc = load_jc(JC)
common = sorted(set(py) & set(jc))

print("n_common", len(common))
print("py frequencies", min(py), max(py), len(py))
print("jc frequencies", min(jc), max(jc), len(jc))
print()

summary = []

for col in CANDIDATE_COLUMNS:
    if col not in next(iter(py.values())):
        continue

    vals = []
    for f in common:
        try:
            vals.append(float(py[f][col]) - jc[f])
        except Exception:
            vals.append(float("nan"))

    diff = np.array(vals, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        continue

    py_curve = {f: float(py[f][col]) for f in common}
    row = {
        "column": col,
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_signed": float(np.mean(diff)),
        "py_peak": max(py_curve.values()),
        "py_peak_freq": max(py_curve, key=py_curve.get),
        "jc_peak": max(jc.values()),
        "jc_peak_freq": max(jc, key=jc.get),
    }
    summary.append(row)

for r in sorted(summary, key=lambda x: x["rms"]):
    print(
        f"{r['column']:22s} "
        f"rms={r['rms']:.6f} "
        f"max_abs={r['max_abs']:.6f} "
        f"mean={r['mean_signed']:+.6f} "
        f"py_peak={r['py_peak']:.6f}@{r['py_peak_freq']} "
        f"jc_peak={r['jc_peak']:.6f}@{r['jc_peak_freq']}"
    )

best = sorted(summary, key=lambda x: x["rms"])[0]
best_col = best["column"]

print()
print("BEST_COLUMN", best_col)
print("freq,py,jc,diff")
for f in common:
    p = float(py[f][best_col])
    j = jc[f]
    print(f"{f},{p:.9g},{j:.9g},{p-j:+.9g}")
