from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
OUTROOT = ROOT / "outputs" / "exp08_tolerance_ladder"
OUTROOT.mkdir(parents=True, exist_ok=True)

EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"

CASES = [
    ("nt1e-8_gm1e-5", "1e-8", "1e-5"),
    ("nt1e-7_gm1e-5", "1e-7", "1e-5"),
    ("nt1e-7_gm3e-5", "1e-7", "3e-5"),
    ("nt1e-6_gm1e-4", "1e-6", "1e-4"),
    ("nt3e-6_gm3e-4", "3e-6", "3e-4"),
]

def grab_float(text: str, key: str):
    m = re.search(rf"{re.escape(key)}=([-+0-9.eE]+)", text)
    return None if m is None else float(m.group(1))

def grab_str(text: str, key: str):
    m = re.search(rf"{re.escape(key)}=([^\s]+)", text)
    return None if m is None else m.group(1)

def run(cmd: list[str]) -> tuple[int, str]:
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(p.stdout)
    return p.returncode, p.stdout

rows = []

for name, newton_tol, gmres_rtol in CASES:
    pump_dir = OUTROOT / f"pump_{name}"
    gain_dir = OUTROOT / f"gain_{name}"

    pump_cmd = [
        sys.executable, str(EXP08),
        "--harmonics", "7",
        "--nt", "96",
        "--pump-current-ratio-ic", "3.0",
        "--continuation-steps", "10",
        "--continuation-predictor", "secant",
        "--newton-tol", newton_tol,
        "--gmres-rtol", gmres_rtol,
        "--jvp-mode", "aft",
        "--skip-time-residual",
        "--quiet",
        "--outdir", str(pump_dir),
    ]

    pump_code, pump_out = run(pump_cmd)

    row = {
        "case": name,
        "newton_tol": newton_tol,
        "gmres_rtol": gmres_rtol,
        "pump_returncode": pump_code,
        "pump_status": grab_str(pump_out, "status"),
        "pump_runtime_s": grab_float(pump_out, "total_runtime_s"),
        "final_coeff_rel": grab_float(pump_out, "final_coeff_rel"),
        "final_newton": grab_float(pump_out, "final_newton_iterations_last_step"),
        "final_gmres": grab_float(pump_out, "final_gmres_iterations_last_step"),
        "branch_i_max_abs": grab_float(pump_out, "branch_i_max_abs"),
        "branch_psi_max_abs": grab_float(pump_out, "branch_psi_max_abs"),
        "gain_returncode": None,
        "max_gain_vs_off_db": None,
        "max_gain_vs_pumpdiag_db": None,
        "gain_all_status_valid": None,
        "gain_runtime_s": None,
    }

    if pump_code == 0 and "status=VALID_CONVERGED" in pump_out:
        gain_cmd = [
            sys.executable, str(EXP09),
            "--pump-dir", str(pump_dir),
            "--sweep",
            "--signal-start-ghz", "4.0",
            "--signal-stop-ghz", "8.0",
            "--points", "21",
            "--sidebands", "2",
            "--gamma-nt", "96",
            "--outdir", str(gain_dir),
        ]

        gain_code, gain_out = run(gain_cmd)

        row["gain_returncode"] = gain_code
        row["max_gain_vs_off_db"] = grab_float(gain_out, "max_gain_vs_off_db")
        row["max_gain_vs_pumpdiag_db"] = grab_float(gain_out, "max_gain_vs_pumpdiag_db")
        row["gain_runtime_s"] = grab_float(gain_out, "total_runtime_s")
        row["gain_all_status_valid"] = grab_str(gain_out, "all_status_valid")

    rows.append(row)

    summary_path = OUTROOT / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

for r in rows:
    print(
        f"{r['case']:16s} "
        f"pump_s={r['pump_runtime_s']} "
        f"coeff={r['final_coeff_rel']} "
        f"gmres={r['final_gmres']} "
        f"gain_diag={r['max_gain_vs_pumpdiag_db']} "
        f"branch_i={r['branch_i_max_abs']} "
        f"branch_psi={r['branch_psi_max_abs']}"
    )

print(f"\nwrote_summary={OUTROOT / 'summary.csv'}")
