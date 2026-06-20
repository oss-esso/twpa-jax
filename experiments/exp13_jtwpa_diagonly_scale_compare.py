from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"

CASE_DIR = ROOT / "outputs" / "jc_doc_python_designs" / "jc_jtwpa"
PUMP_ROOT = ROOT / "outputs" / "exp13_jtwpa_fast_scale_ladder"
OUT_ROOT = ROOT / "outputs" / "exp13_jtwpa_diagonly"
JC_CSV = OUT_ROOT / "jc_jtwpa_diagonly_curve_21pt.csv"

SCALES = [0.75, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]

def load_curve(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fk = "signal_ghz" if "signal_ghz" in rows[0] else "frequency_ghz"
    x = np.array([float(r[fk]) for r in rows], dtype=float)
    y = np.array([float(r["gain_db"]) for r in rows], dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]

def run(cmd, timeout=60):
    try:
        p = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return p.returncode
    except subprocess.TimeoutExpired:
        return 124

def compare(py_csv: Path, jc_f, jc_g):
    py_f, py_g = load_curve(py_csv)
    py_on_jc = py_g if np.array_equal(py_f, jc_f) else np.interp(jc_f, py_f, py_g)
    err = py_on_jc - jc_g
    return {
        "py_max": float(np.max(py_g)),
        "py_mean": float(np.mean(py_g)),
        "py_min": float(np.min(py_g)),
        "py_peak": float(py_f[int(np.argmax(py_g))]),
        "max_abs_err": float(np.max(np.abs(err))),
        "mean_abs_err": float(np.mean(np.abs(err))),
        "rms_err": float(np.sqrt(np.mean(err**2))),
    }

def main():
    jc_f, jc_g = load_curve(JC_CSV)

    print("JC diag-only max =", float(np.max(jc_g)), "peak =", float(jc_f[int(np.argmax(jc_g))]))
    print("scale py_max py_peak max_abs_err mean_abs_err rms_err status")

    rows = []

    for scale in SCALES:
        tag = f"scale_{scale:.3f}".replace(".", "p")
        pump_dir = PUMP_ROOT / tag / "pump"
        gain_dir = OUT_ROOT / tag / "python_diagonly_gain"
        py_csv = gain_dir / "gain_sweep.csv"

        if not (pump_dir / "pump_solution.npz").exists():
            print(f"{scale:.3f} nan nan nan nan nan MISSING_PUMP")
            rows.append({"scale": scale, "status": "MISSING_PUMP"})
            continue

        cmd = [
            sys.executable, str(EXP09),
            "--pump-dir", str(pump_dir),
            "--ipm-dir", str(CASE_DIR),
            "--sweep",
            "--signal-start-ghz", "4.0",
            "--signal-stop-ghz", "8.0",
            "--points", "21",
            "--sidebands", "0",
            "--gamma-nt", "80",
            "--source-port", "1",
            "--out-port", "2",
            "--outdir", str(gain_dir),
        ]

        rc = run(cmd, timeout=60)

        if rc != 0 or not py_csv.exists():
            print(f"{scale:.3f} nan nan nan nan nan GAIN_FAIL")
            rows.append({"scale": scale, "status": "GAIN_FAIL"})
            continue

        stats = compare(py_csv, jc_f, jc_g)
        row = {"scale": scale, "status": "VALID_RAN", **stats}
        rows.append(row)

        print(
            f"{scale:.3f} "
            f"{stats['py_max']:.6f} "
            f"{stats['py_peak']:.6f} "
            f"{stats['max_abs_err']:.6f} "
            f"{stats['mean_abs_err']:.6f} "
            f"{stats['rms_err']:.6f} "
            f"VALID_RAN"
        )

    out_json = OUT_ROOT / "jtwpa_diagonly_scale_comparison.json"
    out_csv = OUT_ROOT / "jtwpa_diagonly_scale_comparison.csv"

    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r["status"] == "VALID_RAN"]
    if valid:
        best = min(valid, key=lambda r: r["rms_err"])
        print()
        print("BEST_BY_RMS")
        print(json.dumps(best, indent=2))

    print("wrote_json =", out_json)
    print("wrote_csv =", out_csv)

if __name__ == "__main__":
    main()
