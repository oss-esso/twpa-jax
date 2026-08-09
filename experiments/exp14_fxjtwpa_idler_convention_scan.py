from __future__ import annotations

from pathlib import Path
import subprocess
import csv
import numpy as np
import sys

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"
JC = ROOT / "outputs" / "exp14_jc_refs" / "jc_fxjtwpa_curve_21pt.csv"
BASE_OUT = ROOT / "outputs" / "exp14_fxjtwpa_idler_convention_scan"

IPM = ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
PUMP = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "pump_solved"
DC = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "dc" / "dc_solution.npz"

COLUMNS = ["gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db", "idler_rel_db"]

def load_jc(path):
    d = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[round(float(r["signal_ghz"]), 6)] = float(r["gain_db"])
    return d

def load_py(path):
    d = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[round(float(r["signal_ghz"]), 6)] = r
    return d

def compare(py_csv):
    py = load_py(py_csv)
    jc = load_jc(JC)
    common = sorted(set(py) & set(jc))
    rows = []
    for col in COLUMNS:
        if col not in next(iter(py.values())):
            continue
        diff = np.array([float(py[f][col]) - jc[f] for f in common], dtype=float)
        rows.append({
            "column": col,
            "n_common": len(common),
            "rms": float(np.sqrt(np.mean(diff * diff))),
            "max_abs": float(np.max(np.abs(diff))),
            "mean_signed": float(np.mean(diff)),
            "py_peak": max(float(py[f][col]) for f in common),
            "py_peak_freq": max(common, key=lambda f: float(py[f][col])),
            "jc_peak": max(jc[f] for f in common),
            "jc_peak_freq": max(common, key=lambda f: jc[f]),
        })
    return sorted(rows, key=lambda r: r["rms"])

all_rows = []

for sidebands in [4, 6, 8]:
    for idler_m in [-1, 1, -2, 2]:
        outdir = BASE_OUT / f"sb{sidebands}_idler{idler_m:+d}".replace("+", "p").replace("-", "m")
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(EXP09),
            "--ipm-dir", str(IPM),
            "--pump-dir", str(PUMP),
            "--dc-solution", str(DC),
            "--sweep",
            "--signal-start-ghz", "6.0",
            "--signal-stop-ghz", "14.0",
            "--points", "21",
            "--sidebands", str(sidebands),
            "--signal-m", "0",
            "--idler-m", str(idler_m),
            "--source-port", "1",
            "--out-port", "2",
            "--source-current-a", "1.0",
            "--z0-ohm", "50.0",
            "--gamma-nt", "160",
            "--outdir", str(outdir),
        ]

        print("\nRUN sidebands", sidebands, "idler_m", idler_m)
        subprocess.run(cmd, check=True)

        bests = compare(outdir / "gain_sweep.csv")
        for r in bests:
            all_rows.append({"sidebands": sidebands, "idler_m": idler_m, **r})
        print("BEST", {"sidebands": sidebands, "idler_m": idler_m, **bests[0]})

summary = BASE_OUT / "summary.csv"
with summary.open("w", newline="", encoding="utf-8") as f:
    keys = list(all_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in all_rows:
        w.writerow(r)

print("\nSUMMARY SORTED")
for r in sorted(all_rows, key=lambda x: x["rms"]):
    print(
        f"sidebands={r['sidebands']:2d} "
        f"idler_m={r['idler_m']:+d} "
        f"col={r['column']:22s} "
        f"rms={r['rms']:.6f} "
        f"max_abs={r['max_abs']:.6f} "
        f"mean={r['mean_signed']:+.6f} "
        f"py_peak={r['py_peak']:.6f}@{r['py_peak_freq']} "
        f"jc_peak={r['jc_peak']:.6f}@{r['jc_peak_freq']}"
    )

print("\nWROTE", summary)
