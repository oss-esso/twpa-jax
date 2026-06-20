from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"

case = "jc_jtwpa"
case_dir = ROOT / "outputs" / "jc_doc_python_designs" / case
out_root = ROOT / "outputs" / "exp13_jtwpa_fast_scale2"
pump_dir = out_root / "pump_h5_fast"
gain_dir = out_root / "gain_h5_21pt"

summary = json.loads((case_dir / "summary.json").read_text())
md = summary["metadata"]

src = md["pump_sources"][0]
pump_port = int(src["port"])
pump_current_a = float(src["current_a"])
pump_freq_ghz = float(md["pump_freqs_ghz"][0])
ic_median = float(summary["Ic_median"])

# JC-source convention found on JPA.
pump_ratio_scale2 = 2.0 * pump_current_a / ic_median

# Fast diagnostic, not exact JC parity.
pump_harmonics = 5
sidebands = 3
nt = 80

def run(cmd, timeout=60):
    print("\n" + "=" * 100, flush=True)
    print(" ".join(str(x) for x in cmd), flush=True)
    print("=" * 100, flush=True)
    try:
        p = subprocess.run(cmd, text=True, timeout=timeout)
        print("returncode =", p.returncode, flush=True)
        return p.returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s", flush=True)
        return 124

pump_cmd = [
    sys.executable, str(EXP08),
    "--ipm-dir", str(case_dir),
    "--pump-port", str(pump_port),
    "--pump-freq-ghz", str(pump_freq_ghz),
    "--pump-current-ratio-ic", repr(pump_ratio_scale2),
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

rc = run(pump_cmd, timeout=60)
if rc != 0:
    raise SystemExit("FAST_PUMP_FAILED_OR_TIMEOUT")

if not (pump_dir / "pump_solution.npz").exists():
    raise SystemExit("FAST_PUMP_NO_SOLUTION")

# Very cheap curve shape check. Exact JC has 131 points; here only 21.
gain_cmd = [
    sys.executable, str(EXP09),
    "--pump-dir", str(pump_dir),
    "--ipm-dir", str(case_dir),
    "--sweep",
    "--signal-start-ghz", "4.0",
    "--signal-stop-ghz", "8.0",
    "--points", "21",
    "--sidebands", str(sidebands),
    "--gamma-nt", str(nt),
    "--source-port", "1",
    "--out-port", "2",
    "--outdir", str(gain_dir),
]

rc = run(gain_cmd, timeout=60)
if rc != 0:
    raise SystemExit("FAST_GAIN_FAILED_OR_TIMEOUT")

print("\nFAST_JTWPA_DONE")
print("pump_dir =", pump_dir)
print("gain_dir =", gain_dir)
