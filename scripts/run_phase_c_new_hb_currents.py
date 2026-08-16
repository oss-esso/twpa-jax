"""Run Phase C multitone or pump-diagnostic points at FDTD currents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_compression  # noqa: E402

DEVICE = {
    "jc_jtwpa": {
        "circuit": ROOT / "outputs" / "jc_doc_python_designs" / "jc_jtwpa",
        "pump": 7.12, "signal": 6.62, "pump_port": 1, "modes": 10, "nt": 40,
        "signal_current": 3.0e-8,
    },
    "jc_fqjtwpa": {
        "circuit": ROOT / "outputs" / "jc_doc_python_designs" / "jc_fqjtwpa",
        "pump": 7.9, "signal": 7.4, "pump_port": 1, "modes": 10, "nt": 40,
        "signal_current": 3.0e-8,
    },
    "ipm_2c_fixed": {
        "circuit": ROOT / "designs" / "ipm_2c_fixed",
        "pump": 7.9, "signal": 7.4, "pump_port": 4, "modes": 10, "nt": 40,
        "signal_current": 3.0e-8,
    },
    "guarcello": {
        "circuit": ROOT / ".hybrid_outputs" / "phase_c_guarcello_hb" / "circuit",
        "pump": 7.0, "signal": 6.42, "pump_port": 1, "modes": 19, "nt": 80,
        "signal_current": 2.0e-7,
    },
}


def _fdtd_currents(device: str) -> list[float]:
    rows = []
    for path in (ROOT / "outputs" / "chaos" / "phaseB_signal" / device).rglob("result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        current = row.get("pump_current_peak_a_achieved")
        if current is None and device == "guarcello":
            power_w = 1.0e-3 * 10.0 ** (float(row["pump_power_dbm"]) / 10.0)
            current = math.sqrt(2.0 * power_w / 50.0)
        if current is not None:
            rows.append(float(current))
    return sorted(rows)


def _common_converged_currents(device: str, currents: list[float]) -> list[float]:
    single_path = ROOT / ".hybrid_outputs" / "phase_c_single_tone_exact" / device / "map_points.csv"
    with single_path.open(newline="", encoding="utf-8") as handle:
        single = list(csv.DictReader(handle))
    new_root = ROOT / ".hybrid_outputs" / "phase_c_new_hb" / device
    new = []
    for path in new_root.rglob("compression_summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        gain = row.get("small_signal_gain_vs_off_db")
        if gain is not None and math.isfinite(float(gain)) and int(row.get("n_failed_power_points", 0)) == 0:
            new.append(float(row["pump_current_a"]))
    accepted = []
    for current in currents:
        single_ok = any(
            row.get("status") == "PASS"
            and math.isclose(float(row["pump_current_peak_a"]), current, rel_tol=2e-9, abs_tol=1e-18)
            for row in single
        )
        new_ok = any(math.isclose(value, current, rel_tol=2e-9, abs_tol=1e-18) for value in new)
        if single_ok and new_ok:
            accepted.append(current)
    return accepted


def run(device: str, mode: str, output: Path) -> int:
    config = DEVICE[device]
    currents = _fdtd_currents(device)
    if mode == "timing":
        currents = _common_converged_currents(device, currents)
    cli = [
        "--output-dir", str(output),
        "--circuit-dir", str(config["circuit"]),
        "--pump-freq-ghz", str(config["pump"]),
        "--signal-ghz", str(config["signal"]),
        "--pump-current-list", *[format(value, ".17g") for value in currents],
        "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", str(config["modes"]),
        "--pump-nt", str(config["nt"]),
        "--source-port", "1", "--pump-port", str(config["pump_port"]),
        "--out-port", "2", "--diagnostic-port", "2",
        "--attenuation-db", "0", "--pump-attenuation-db", "0",
        "--signal-attenuation-db", "0", "--power-convention", "legacy_traveling_wave",
        "--multitone-basis", "matched", "--multitone-sidebands", "6",
        "--n-signal-power", "1",
        "--signal-current-min-a", str(config["signal_current"]),
        "--signal-current-max-a", str(config["signal_current"]),
        "--recovery", "ladder", "--save-states", "none",
    ]
    if mode == "diagnostic":
        cli.append("--force-single-tone")
    return run_compression.main(cli)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=tuple(DEVICE), required=True)
    parser.add_argument("--mode", choices=("new", "timing", "diagnostic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.device, args.mode, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
