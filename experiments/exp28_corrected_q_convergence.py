"""Run representative Q=2/Q=3 corrected-termination compression points."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT = ROOT / "outputs" / "exp28_2c_termination_43ohm_q_convergence"
CIRCUIT = Path("D:/tmp/exp28_2c_termination_43ohm/ipm_python_design_terminated")


def main() -> None:
    common = [
        sys.executable, "scripts/run_compression.py",
        "--circuit-dir", str(CIRCUIT),
        "--pump-freq-ghz", "7.540816326531111",
        "--signal-ghz", "7.4",
        "--pump-current-a", "7.231074707853736e-06",
        "--pump-mode-policy", "dense_real", "--pump-harmonics", "6", "--pump-nt", "40",
        "--multitone-basis", "lattice",
        "--source-port", "1", "--pump-port", "4", "--out-port", "2", "--diagnostic-port", "2",
        "--attenuation-db", "0", "--factor-backend", "pardiso",
        "--n-signal-power", "25", "--signal-current-min-a", "1e-10", "--signal-current-max-a", "1e-6",
        "--recovery", "ladder", "--signal-continuation-deadline-s", "600", "--allow-memory-overcommit",
    ]
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for q in (2, 3):
        output = BASE_OUTPUT / f"q{q}"
        command = common + ["--output-dir", str(output), "--multitone-sidebands", str(q)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        (output / "run_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (output / "run_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        summary_path = output / "compression_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"status": "SUBPROCESS_FAILED", "returncode": completed.returncode}
        results.append({"q": q, **summary})
    (BASE_OUTPUT / "q_convergence_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
