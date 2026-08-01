"""Run the exp26 7.629 GHz q<=1 profile set with overlap observables."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twpa_solver.multitone.observables import spatial_profile_summary

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "exp27_track3_overlap_7p629_q01"


def main() -> None:
    command = [
        sys.executable,
        "scripts/run_compression.py",
        "--output-dir", str(OUTPUT),
        "--circuit-dir", "outputs/ipm_python_design",
        "--pump-freq-ghz", "7.540816326531111",
        "--signal-ghz", "7.629",
        "--pump-current-a", "7.231074707853736e-06",
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "dense_real",
        "--pump-harmonics", "6",
        "--pump-nt", "40",
        "--multitone-basis", "lattice",
        "--multitone-sidebands", "1",
        "--source-port", "1",
        "--pump-port", "4",
        "--out-port", "2",
        "--attenuation-db", "0",
        "--factor-backend", "pardiso",
        "--n-signal-power", "16",
        "--signal-current-min-a", "1e-10",
        "--signal-current-max-a", "1e-6",
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", "600",
        "--allow-memory-overcommit",
        "--save-states", "all",
        "--spatial-profiles",
        "--spatial-profiles-all",
    ]
    if not (OUTPUT / "compression_summary.json").exists():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "run_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (OUTPUT / "run_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary_path = OUTPUT / "compression_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"compression run failed with return code {completed.returncode}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[int, dict[str, float | int | str]]] = {}
    with (OUTPUT / "spatial_profiles.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            label = str(raw["operating_point"])
            row = {
                key: value if key == "operating_point" else float(value)
                for key, value in raw.items()
                if value != ""
            }
            grouped.setdefault(label, {})[int(row["branch_index"])] = row
    summary["spatial_profile_summary"] = {
        label: spatial_profile_summary(list(rows.values()))
        for label, rows in grouped.items()
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = {
        "status": summary.get("status"),
        "signal_ghz": summary.get("signal_ghz"),
        "max_power_balance_rel_err": summary.get("max_power_balance_rel_err"),
        "spatial_profile_summary": summary.get("spatial_profile_summary", {}),
    }
    (OUTPUT / "overlap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
