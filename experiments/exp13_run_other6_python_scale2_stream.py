from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"
EXP10 = ROOT / "experiments" / "exp10_jc_doc_python_design_builders.py"

DESIGN_ROOT = ROOT / "outputs" / "jc_doc_python_designs"
OUT_ROOT = ROOT / "outputs" / "exp13_other6_python_scale2_stream"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_CASES = [
    "jc_dpjpa",
    "jc_fxjpa",
    "jc_jtwpa",
    "jc_fqjtwpa",
    "jc_fqjtwpa_diss",
    "jc_fxjtwpa",
]

case_env = os.environ.get("EXP13_CASES", "").strip()
CASES = [x.strip() for x in case_env.split(",") if x.strip()] if case_env else DEFAULT_CASES

PORTS = {
    "jc_dpjpa": (1, 1),
    "jc_fxjpa": (1, 1),
    "jc_jtwpa": (1, 2),
    "jc_fqjtwpa": (1, 2),
    "jc_fqjtwpa_diss": (1, 2),
    "jc_fxjtwpa": (1, 2),
}

TIMEOUT_S = int(os.environ.get("EXP13_TIMEOUT_S", "1800"))

def run_stream(cmd, timeout_s=TIMEOUT_S):
    print("\n" + "=" * 120, flush=True)
    print(" ".join(str(x) for x in cmd), flush=True)
    print("=" * 120, flush=True)
    try:
        p = subprocess.run(cmd, text=True, timeout=timeout_s)
        return p.returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout_s} s", flush=True)
        return 124

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def signal_grid(md):
    sg = md["signal_ghz"]
    start = float(sg["start"])
    stop = float(sg["stop"])
    if "points" in sg:
        points = int(sg["points"])
    else:
        step = float(sg["step"])
        points = int(round((stop - start) / step)) + 1
    return start, stop, points

def summarize_gain_csv(path: Path):
    if not path.exists():
        return {}

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    freq_key = "signal_ghz" if "signal_ghz" in rows[0] else "frequency_ghz"
    gain_key = "gain_db"

    freq = np.array([float(r[freq_key]) for r in rows], dtype=float)
    gain = np.array([float(r[gain_key]) for r in rows], dtype=float)

    i = int(np.argmax(gain))
    return {
        "gain_db_max": float(np.max(gain)),
        "gain_db_mean": float(np.mean(gain)),
        "gain_db_min": float(np.min(gain)),
        "peak_frequency_ghz": float(freq[i]),
        "points": int(len(freq)),
    }

if not (DESIGN_ROOT / "build_manifest.json").exists():
    rc = run_stream([sys.executable, str(EXP10), "--outdir", str(DESIGN_ROOT)])
    if rc != 0:
        raise SystemExit("exp10 design build failed")

rows = []

for case in CASES:
    print(f"\n\n######## CASE {case} ########", flush=True)

    case_dir = DESIGN_ROOT / case
    summary_path = case_dir / "summary.json"

    row = {
        "case": case,
        "status": "",
        "reason": "",
        "python_curve_csv": "",
    }

    if not summary_path.exists():
        row["status"] = "MISSING_DESIGN"
        row["reason"] = str(summary_path)
        rows.append(row)
        continue

    summary = load_json(summary_path)
    md = summary["metadata"]
    features = md.get("features", {})
    sources = md.get("pump_sources", [])

    if len(sources) != 1:
        row["status"] = "SKIPPED_UNSUPPORTED"
        row["reason"] = f"current exp08 path is one-pump only; found {len(sources)} sources"
        rows.append(row)
        print(f"SKIP {case}: {row['reason']}", flush=True)
        continue

    if bool(features.get("needs_dc", False)):
        row["status"] = "SKIPPED_UNSUPPORTED"
        row["reason"] = "needs DC/flux operating point support"
        rows.append(row)
        print(f"SKIP {case}: {row['reason']}", flush=True)
        continue

    src = sources[0]
    pump_port = int(src["port"])
    pump_current_a = float(src["current_a"])
    pump_freq_ghz = float(md["pump_freqs_ghz"][0])
    ic_median = float(summary["Ic_median"])

    pump_ratio_scale2 = 2.0 * pump_current_a / ic_median

    npump = int(md["Npumpharmonics"][0])
    nmod = int(md["Nmodulationharmonics"][0])
    sidebands = nmod
    nt = max(96, 16 * npump)

    start, stop, points = signal_grid(md)
    source_port, out_port = PORTS[case]

    pump_dir = OUT_ROOT / case / "pump_scale2"
    gain_dir = OUT_ROOT / case / "gain_scale2"
    gain_csv = gain_dir / "gain_sweep.csv"

    row.update({
        "pump_port": pump_port,
        "pump_freq_ghz": pump_freq_ghz,
        "pump_current_a_jc": pump_current_a,
        "pump_current_ratio_ic_median_scale2": pump_ratio_scale2,
        "npump": npump,
        "nmod": nmod,
        "nt": nt,
        "signal_start_ghz": start,
        "signal_stop_ghz": stop,
        "source_port": source_port,
        "out_port": out_port,
        "python_curve_csv": str(gain_csv),
    })

    pump_cmd = [
        sys.executable, str(EXP08),
        "--ipm-dir", str(case_dir),
        "--pump-port", str(pump_port),
        "--pump-freq-ghz", str(pump_freq_ghz),
        "--pump-current-ratio-ic", repr(pump_ratio_scale2),
        "--harmonics", str(npump),
        "--nt", str(nt),
        "--continuation-steps", "10",
        "--continuation-predictor", "secant",
        "--newton-tol", "3e-6",
        "--gmres-rtol", "3e-4",
        "--jvp-mode", "aft",
        "--quiet",
        "--skip-time-residual",
        "--outdir", str(pump_dir),
    ]

    rc = run_stream(pump_cmd)
    row["pump_returncode"] = rc

    if rc != 0 or not (pump_dir / "pump_solution.npz").exists():
        row["status"] = "PUMP_FAILED"
        row["reason"] = "pump command failed or did not write pump_solution.npz"
        rows.append(row)
        continue

    gain_cmd = [
        sys.executable, str(EXP09),
        "--pump-dir", str(pump_dir),
        "--ipm-dir", str(case_dir),
        "--sweep",
        "--signal-start-ghz", str(start),
        "--signal-stop-ghz", str(stop),
        "--points", str(points),
        "--sidebands", str(sidebands),
        "--gamma-nt", str(nt),
        "--source-port", str(source_port),
        "--out-port", str(out_port),
        "--outdir", str(gain_dir),
    ]

    rc = run_stream(gain_cmd)
    row["gain_returncode"] = rc

    if rc != 0 or not gain_csv.exists():
        row["status"] = "GAIN_FAILED"
        row["reason"] = "gain command failed or did not write gain_sweep.csv"
        rows.append(row)
        continue

    row.update(summarize_gain_csv(gain_csv))
    row["status"] = "VALID_RAN"
    rows.append(row)

out_json = OUT_ROOT / "other6_python_scale2_summary.json"
out_csv = OUT_ROOT / "other6_python_scale2_summary.csv"

out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

fieldnames = sorted({k for r in rows for k in r.keys()})
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("\n=== SUMMARY ===")
for r in rows:
    print(
        f"{r['case']:18s} {r['status']:22s} "
        f"gain_max={r.get('gain_db_max')} peak={r.get('peak_frequency_ghz')} reason={r.get('reason','')}",
        flush=True,
    )

print("wrote_json=", out_json)
print("wrote_csv=", out_csv)
