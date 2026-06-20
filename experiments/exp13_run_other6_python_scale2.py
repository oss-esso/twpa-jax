from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"
EXP10 = ROOT / "experiments" / "exp10_jc_doc_python_design_builders.py"

DESIGN_ROOT = ROOT / "outputs" / "jc_doc_python_designs"
OUT_ROOT = ROOT / "outputs" / "exp13_other6_python_scale2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CASES = [
    "jc_dpjpa",
    "jc_fxjpa",
    "jc_jtwpa",
    "jc_fqjtwpa",
    "jc_fqjtwpa_diss",
    "jc_fxjtwpa",
]

PORTS = {
    "jc_dpjpa": (1, 1),
    "jc_fxjpa": (1, 1),
    "jc_jtwpa": (1, 2),
    "jc_fqjtwpa": (1, 2),
    "jc_fqjtwpa_diss": (1, 2),
    "jc_fxjtwpa": (1, 2),
}

def run(cmd):
    print("\n" + "=" * 120)
    print(" ".join(str(x) for x in cmd))
    print("=" * 120)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    return p.returncode, p.stdout

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

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

def parse_stdout_float(out, key):
    prefix = key + "="
    for line in out.splitlines():
        if line.startswith(prefix):
            try:
                return float(line.split("=", 1)[1].strip())
            except Exception:
                return None
    return None

def parse_stdout_str(out, key):
    prefix = key + "="
    for line in out.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None

# Build designs if missing.
if not (DESIGN_ROOT / "build_manifest.json").exists():
    code, _ = run([sys.executable, str(EXP10), "--outdir", str(DESIGN_ROOT)])
    if code != 0:
        raise SystemExit("exp10 design build failed")

rows = []

for case in CASES:
    case_dir = DESIGN_ROOT / case
    summary_path = case_dir / "summary.json"

    row = {
        "case": case,
        "attempted": False,
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
        row["reason"] = "current exp08/exp09 path is one-pump only; case has multiple pump/DC sources"
        rows.append(row)
        continue

    if features.get("needs_dc", False):
        row["status"] = "SKIPPED_UNSUPPORTED"
        row["reason"] = "needs DC/flux operating point support"
        rows.append(row)
        continue

    src = sources[0]
    pump_port = int(src["port"])
    pump_current_a = float(src["current_a"])
    pump_freq_ghz = float(md["pump_freqs_ghz"][0])
    ic_median = float(summary["Ic_median"])

    pump_ratio_scale2 = 2.0 * pump_current_a / ic_median

    npump = int(md["Npumpharmonics"][0])
    nmod = int(md["Nmodulationharmonics"][0])

    # Keep exact JC modulation sidebands for the comparison.
    sidebands = nmod

    # Use enough time samples for AFT. This mirrors the JPA H16/nt256 relation.
    nt = max(96, 16 * npump)

    start, stop, points = signal_grid(md)
    source_port, out_port = PORTS[case]

    pump_dir = OUT_ROOT / case / "pump_scale2"
    gain_dir = OUT_ROOT / case / "gain_scale2"

    row.update({
        "attempted": True,
        "pump_port": pump_port,
        "pump_freq_ghz": pump_freq_ghz,
        "pump_current_a_jc": pump_current_a,
        "pump_current_ratio_ic_median_scale2": pump_ratio_scale2,
        "npump": npump,
        "nmod": nmod,
        "nt": nt,
        "signal_start_ghz": start,
        "signal_stop_ghz": stop,
        "points": points,
        "source_port": source_port,
        "out_port": out_port,
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

    pump_code, pump_out = run(pump_cmd)
    row["pump_returncode"] = pump_code
    row["pump_status"] = parse_stdout_str(pump_out, "status")
    row["pump_runtime_s"] = parse_stdout_float(pump_out, "total_runtime_s")
    row["pump_final_coeff_rel"] = parse_stdout_float(pump_out, "final_coeff_rel")

    if pump_code != 0 or row["pump_status"] != "VALID_CONVERGED":
        row["status"] = "PUMP_FAILED"
        row["reason"] = "pump did not converge or script returned nonzero"
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

    gain_code, gain_out = run(gain_cmd)
    row["gain_returncode"] = gain_code
    row["all_status_valid"] = parse_stdout_str(gain_out, "all_status_valid")
    row["gain_db_max"] = parse_stdout_float(gain_out, "gain_db_max")
    row["gain_db_mean"] = parse_stdout_float(gain_out, "gain_db_mean")
    row["gain_db_min"] = parse_stdout_float(gain_out, "gain_db_min")
    row["peak_frequency_ghz"] = parse_stdout_float(gain_out, "peak_frequency_ghz")
    row["gain_runtime_s"] = parse_stdout_float(gain_out, "total_runtime_s")
    row["python_curve_csv"] = str(gain_dir / "gain_sweep.csv")

    if gain_code == 0:
        row["status"] = "VALID_RAN"
    else:
        row["status"] = "GAIN_FAILED"
        row["reason"] = "gain script returned nonzero"

    rows.append(row)

out_json = OUT_ROOT / "other6_python_scale2_summary.json"
out_csv = OUT_ROOT / "other6_python_scale2_summary.csv"

out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

fieldnames = sorted({k for r in rows for k in r.keys()})
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("\n=== PYTHON OTHER6 SCALE2 SUMMARY ===")
for r in rows:
    print(
        f"{r['case']:18s} {r['status']:22s} "
        f"gain_max={r.get('gain_db_max')} peak={r.get('peak_frequency_ghz')} reason={r.get('reason','')}"
    )

print("wrote_json=", out_json)
print("wrote_csv=", out_csv)
