from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"

CASE_DIR = ROOT / "outputs" / "jc_doc_python_designs" / "jc_jtwpa"
JC_CSV = ROOT / "outputs" / "exp13_jtwpa_fast_scale2" / "jc_jtwpa_curve_21pt.csv"
OUT_ROOT = ROOT / "outputs" / "exp13_jtwpa_harmonic_ladder"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

summary = json.loads((CASE_DIR / "summary.json").read_text())
md = summary["metadata"]
src = md["pump_sources"][0]

pump_port = int(src["port"])
pump_current_a = float(src["current_a"])
pump_freq_ghz = float(md["pump_freqs_ghz"][0])
ic_median = float(summary["Ic_median"])

pump_ratio_scale2 = 2.0 * pump_current_a / ic_median
sidebands = int(md["Nmodulationharmonics"][0])  # should be 10

HARMONICS = [5, 7, 9, 10]
START_GHZ = 4.0
STOP_GHZ = 8.0
POINTS = 21

def load_curve(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fk = "signal_ghz" if "signal_ghz" in rows[0] else "frequency_ghz"
    x = np.array([float(r[fk]) for r in rows], dtype=float)
    y = np.array([float(r["gain_db"]) for r in rows], dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]

def run(cmd, timeout=60):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, time.time() - t0

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
    print("JC max =", float(np.max(jc_g)), "peak =", float(jc_f[int(np.argmax(jc_g))]))
    print("sidebands =", sidebands)
    print("H nt pump_s gain_s py_max py_peak max_abs_err mean_abs_err rms_err status")

    rows = []

    for H in HARMONICS:
        nt = max(96, 16 * H)
        tag = f"H{H}_nt{nt}"
        pump_dir = OUT_ROOT / tag / "pump"
        gain_dir = OUT_ROOT / tag / "gain"
        py_csv = gain_dir / "gain_sweep.csv"

        pump_cmd = [
            sys.executable, str(EXP08),
            "--ipm-dir", str(CASE_DIR),
            "--pump-port", str(pump_port),
            "--pump-freq-ghz", str(pump_freq_ghz),
            "--pump-current-ratio-ic", repr(pump_ratio_scale2),
            "--harmonics", str(H),
            "--nt", str(nt),
            "--continuation-steps", "6",
            "--continuation-predictor", "secant",
            "--newton-tol", "1e-5",
            "--gmres-rtol", "1e-3",
            "--jvp-mode", "aft",
            "--quiet",
            "--skip-time-residual",
            "--outdir", str(pump_dir),
        ]

        rc_p, pump_s = run(pump_cmd, timeout=60)

        if rc_p != 0 or not (pump_dir / "pump_solution.npz").exists():
            print(f"{H} {nt} {pump_s:.2f} nan nan nan nan nan nan PUMP_FAIL")
            rows.append({"H": H, "nt": nt, "status": "PUMP_FAIL", "pump_s": pump_s})
            continue

        gain_cmd = [
            sys.executable, str(EXP09),
            "--pump-dir", str(pump_dir),
            "--ipm-dir", str(CASE_DIR),
            "--sweep",
            "--signal-start-ghz", str(START_GHZ),
            "--signal-stop-ghz", str(STOP_GHZ),
            "--points", str(POINTS),
            "--sidebands", str(sidebands),
            "--gamma-nt", str(nt),
            "--source-port", "1",
            "--out-port", "2",
            "--outdir", str(gain_dir),
        ]

        rc_g, gain_s = run(gain_cmd, timeout=60)

        if rc_g != 0 or not py_csv.exists():
            print(f"{H} {nt} {pump_s:.2f} {gain_s:.2f} nan nan nan nan nan GAIN_FAIL")
            rows.append({"H": H, "nt": nt, "status": "GAIN_FAIL", "pump_s": pump_s, "gain_s": gain_s})
            continue

        stats = compare(py_csv, jc_f, jc_g)
        row = {"H": H, "nt": nt, "pump_s": pump_s, "gain_s": gain_s, "status": "VALID_RAN", **stats}
        rows.append(row)

        print(
            f"{H} {nt} "
            f"{pump_s:.2f} "
            f"{gain_s:.2f} "
            f"{stats['py_max']:.6f} "
            f"{stats['py_peak']:.6f} "
            f"{stats['max_abs_err']:.6f} "
            f"{stats['mean_abs_err']:.6f} "
            f"{stats['rms_err']:.6f} "
            f"VALID_RAN"
        )

    out_json = OUT_ROOT / "jtwpa_harmonic_ladder_summary.json"
    out_csv = OUT_ROOT / "jtwpa_harmonic_ladder_summary.csv"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r.get("status") == "VALID_RAN"]
    if valid:
        best = min(valid, key=lambda r: r["rms_err"])
        print()
        print("BEST_BY_RMS")
        print(json.dumps(best, indent=2))

    print("wrote_json =", out_json)
    print("wrote_csv =", out_csv)

if __name__ == "__main__":
    main()
