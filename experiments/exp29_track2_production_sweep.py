"""Extend the production 2c fixed-frequency pump sweep into low gain."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "exp29_track2_production_wide"


def main() -> None:
    command = [
        sys.executable,
        "scripts/run_compression.py",
        "--circuit-dir",
        "designs/ipm_2c_fixed",
        "--pump-freq-ghz",
        "7.540816326531111",
        "--signal-ghz",
        "7.4",
        "--pump-current-list",
        "4.2e-6",
        "4.8e-6",
        "5.2e-6",
        "5.6e-6",
        "5.8e-6",
        "--pump-mode-policy",
        "dense_real",
        "--pump-harmonics",
        "6",
        "--pump-nt",
        "40",
        "--multitone-basis",
        "lattice",
        "--multitone-sidebands",
        "10",
        "--source-port",
        "1",
        "--pump-port",
        "4",
        "--out-port",
        "2",
        "--diagnostic-port",
        "2",
        "--attenuation-db",
        "0",
        "--factor-backend",
        "pardiso",
        "--n-signal-power",
        "25",
        "--signal-current-min-a",
        "1e-10",
        "--signal-current-max-a",
        "3e-6",
        "--recovery",
        "ladder",
        "--signal-continuation-deadline-s",
        "600",
        "--allow-memory-overcommit",
        "--output-dir",
        str(OUTPUT),
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT, text=True, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
