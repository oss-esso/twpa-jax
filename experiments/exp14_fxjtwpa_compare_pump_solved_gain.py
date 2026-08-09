from pathlib import Path
import csv
import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
PY = ROOT / "outputs" / "exp14_fxjtwpa_gain_from_pump_solved" / "gain_sweep.csv"

# Try likely reference names.
cands = list((ROOT / "outputs" / "exp14_jc_refs").glob("*fxjtwpa*curve*.csv"))
if not cands:
    cands = list((ROOT / "outputs").rglob("*fxjtwpa*curve*.csv"))

print("JC candidates:")
for i, p in enumerate(cands):
    print(i, p)

if not cands:
    raise SystemExit("No fxjtwpa JC reference curve found.")

JC = cands[0]
print("USING_JC", JC)

def load(path):
    d = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fghz = round(float(r["signal_ghz"]), 6)
            g = float(r["gain_db"])
            d[fghz] = g
    return d

py = load(PY)
jc = load(JC)
common = sorted(set(py) & set(jc))
if not common:
    raise SystemExit(f"No common frequencies. py={sorted(py)[:5]} jc={sorted(jc)[:5]}")

diff = np.array([py[f] - jc[f] for f in common])

print()
print("n_common", len(common))
print("py_peak", max(py.values()), "py_peak_freq", max(py, key=py.get))
print("jc_peak", max(jc.values()), "jc_peak_freq", max(jc, key=jc.get))
print("rms_db", float(np.sqrt(np.mean(diff * diff))))
print("max_abs_db", float(np.max(np.abs(diff))))
print("mean_signed_db", float(np.mean(diff)))
print()
print("freq,py,jc,diff")
for f in common:
    print(f"{f},{py[f]:.9g},{jc[f]:.9g},{py[f]-jc[f]:+.9g}")
