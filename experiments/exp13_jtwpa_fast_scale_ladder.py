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

CASE = "jc_jtwpa"
CASE_DIR = ROOT / "outputs" / "jc_doc_python_designs" / CASE
JC_CSV = ROOT / "outputs" / "exp13_jtwpa_fast_scale2" / "jc_jtwpa_curve_21pt.csv"
OUT_ROOT = ROOT / "outputs" / "exp13_jtwpa_fast_scale_ladder"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

summary = json.loads((CASE_DIR / "summary.json").read_text())
md = summary["metadata"]
src = md["pump_sources"][0]

pump_port = int(src["port"])
pump_current_a = float(src["current_a"])
pump_freq_ghz = float(md["pump_freqs_ghz"][0])
ic_median = float(summary["Ic_median"])

# Exact/fast JTWPA comparison settings.
pump_harmonics = 5
sidebands = 3
nt = 80
start_ghz = 4.0
stop_ghz = 8.0
points = 21

# Scale is relative to the JC builder pump current.
# scale=1.0 means Python pump current = JC current.
# scale=2.0 is the JPA-style correction that overpumped JTWPA.
SCALES = [0.75, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]

def load_curve(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fk = "signal_ghz" if "signal_ghz" in rows[0] else "frequency_ghz"
    x = np.array([float(r[fk]) for r in rows], dtype=float)
    y = np.array([float(r["gain_db"]) for r in rows], dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]

def run_quiet(cmd, timeout=60):
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return p.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, time.time() - t0

def summarize_gain_csv(py_csv: Path, jc_f, jc_g):
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
        "rms_err": float(np.sqrt(np.mean(err ** 2))),
    }

def main():
    jc_f, jc_g = load_curve(JC_CSV)
    jc_max = float(np.max(jc_g))
    jc_peak = float(jc_f[int(np.argmax(jc_g))])

    print(f"JC max={jc_max:.6f} peak={jc_peak:.6f}")
    print("scale pump_ratio pump_s gain_s py_max py_peak max_abs_err mean_abs_err rms_err status")

    rows = []

    for scale in SCALES:
        pump_ratio = scale * pump_current_a / ic_median
        tag = f"scale_{scale:.3f}".replace(".", "p")
        pump_dir = OUT_ROOT / tag / "pump"
        gain_dir = OUT_ROOT / tag / "gain"
        py_csv = gain_dir / "gain_sweep.csv"

        pump_cmd = [
            sys.executable, str(EXP08),
            "--ipm-dir", str(CASE_DIR),
            "--pump-port", str(pump_port),
            "--pump-freq-ghz", str(pump_freq_ghz),
            "--pump-current-ratio-ic", repr(pump_ratio),
            "--harmonics", str(pump_harmonics),
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

        rc_pump, pump_s = run_quiet(pump_cmd, timeout=60)

        if rc_pump != 0 or not (pump_dir / "pump_solution.npz").exists():
            print(f"{scale:.3f} {pump_ratio:.9e} {pump_s:.2f} nan nan nan nan nan nan PUMP_FAIL")
            rows.append({"scale": scale, "pump_ratio": pump_ratio, "status": "PUMP_FAIL"})
            continue

        gain_cmd = [
            sys.executable, str(EXP09),
            "--pump-dir", str(pump_dir),
            "--ipm-dir", str(CASE_DIR),
            "--sweep",
            "--signal-start-ghz", str(start_ghz),
            "--signal-stop-ghz", str(stop_ghz),
            "--points", str(points),
            "--sidebands", str(sidebands),
            "--gamma-nt", str(nt),
            "--source-port", "1",
            "--out-port", "2",
            "--outdir", str(gain_dir),
        ]

        rc_gain, gain_s = run_quiet(gain_cmd, timeout=60)

        if rc_gain != 0 or not py_csv.exists():
            print(f"{scale:.3f} {pump_ratio:.9e} {pump_s:.2f} {gain_s:.2f} nan nan nan nan nan GAIN_FAIL")
            rows.append({"scale": scale, "pump_ratio": pump_ratio, "status": "GAIN_FAIL"})
            continue

        stats = summarize_gain_csv(py_csv, jc_f, jc_g)
        row = {
            "scale": scale,
            "pump_ratio": pump_ratio,
            "pump_s": pump_s,
            "gain_s": gain_s,
            "status": "VALID_RAN",
            **stats,
        }
        rows.append(row)

        print(
            f"{scale:.3f} "
            f"{pump_ratio:.9e} "
            f"{pump_s:.2f} "
            f"{gain_s:.2f} "
            f"{stats['py_max']:.6f} "
            f"{stats['py_peak']:.6f} "
            f"{stats['max_abs_err']:.6f} "
            f"{stats['mean_abs_err']:.6f} "
            f"{stats['rms_err']:.6f} "
            f"VALID_RAN"
        )

    out_json = OUT_ROOT / "jtwpa_fast_scale_ladder_summary.json"
    out_csv = OUT_ROOT / "jtwpa_fast_scale_ladder_summary.csv"

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
