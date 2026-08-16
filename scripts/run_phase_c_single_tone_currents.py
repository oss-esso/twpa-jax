"""Run pump-only HB plus Floquet gain at Phase C FDTD pump currents."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map as gain_map  # noqa: E402


DEVICE = {
    "jc_jtwpa": {
        "circuit": ROOT / "outputs" / "jc_doc_python_designs" / "jc_jtwpa",
        "pump_ghz": 7.12, "signal_ghz": 6.62, "pump_port": 1,
        "source_port": 1, "out_port": 2, "mode_count": 10, "nt": 40,
    },
    "jc_fqjtwpa": {
        "circuit": ROOT / "outputs" / "jc_doc_python_designs" / "jc_fqjtwpa",
        "pump_ghz": 7.9, "signal_ghz": 7.4, "pump_port": 1,
        "source_port": 1, "out_port": 2, "mode_count": 10, "nt": 40,
    },
    "ipm_2c_fixed": {
        "circuit": ROOT / "designs" / "ipm_2c_fixed",
        "pump_ghz": 7.9, "signal_ghz": 7.4, "pump_port": 4,
        "source_port": 1, "out_port": 2, "mode_count": 10, "nt": 40,
    },
    "guarcello": {
        "circuit": ROOT / ".hybrid_outputs" / "phase_c_guarcello_hb" / "circuit",
        "pump_ghz": 7.0, "signal_ghz": 6.42, "pump_port": 1,
        "source_port": 1, "out_port": 2, "mode_count": 19, "nt": 80,
    },
}


def _fdtd_rows(device: str) -> list[dict[str, object]]:
    root = ROOT / "outputs" / "chaos" / "phaseB_signal" / device
    rows = []
    for path in root.rglob("result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        current = row.get("pump_current_peak_a_achieved")
        if current is None and device == "guarcello":
            power_w = 1.0e-3 * 10.0 ** (float(row["pump_power_dbm"]) / 10.0)
            current = math.sqrt(2.0 * power_w / 50.0)
        if current is None:
            continue
        row["pump_current_peak_a_achieved"] = float(current)
        row["_source_path"] = str(path)
        rows.append(row)
    return sorted(rows, key=lambda row: float(row["pump_current_peak_a_achieved"]))


def run(device: str, output: Path, sidebands: int) -> dict[str, object]:
    config = DEVICE[device]
    fdtd = _fdtd_rows(device)
    cli = [
        "--mode", "warmstart", "--executor", "inprocess",
        "--circuit-dir", str(config["circuit"]), "--outdir", str(output),
        "--n-power", str(len(fdtd)), "--n-frequency", "1",
        "--pump-power-min-dbm", "0", "--pump-power-max-dbm", "1",
        "--pump-freq-min-ghz", str(config["pump_ghz"]),
        "--pump-freq-max-ghz", str(config["pump_ghz"]),
        "--signal-ghz", str(config["signal_ghz"]),
        "--signal-attenuation-db", "0", "--attenuation-db", "0",
        "--no-signal-spectrum", "--power-convention", "legacy_traveling_wave",
        "--pump-port", str(config["pump_port"]),
        "--source-port", str(config["source_port"]),
        "--out-port", str(config["out_port"]),
        "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", str(config["mode_count"]),
        "--harmonics", str(2 * int(config["mode_count"]) - 1),
        "--nt", str(config["nt"]),
        "--sidebands", str(sidebands), "--continuation-steps", "4",
    ]
    args = gain_map.parse_args(cli)
    points = [
        gain_map.GridPoint(
            index=index, i_power=index, j_freq=0,
            power_dbm=float(row.get("pump_power_dbm", row.get("control_value", index))),
            pump_freq_ghz=float(config["pump_ghz"]),
            current_a=float(row["pump_current_peak_a_achieved"]),
        )
        for index, row in enumerate(fdtd)
    ]
    output.mkdir(parents=True, exist_ok=True)
    engine = gain_map.InProcessEngine(args)
    rows = gain_map.run_warm_pass_inprocess(points, output / "warm", engine)
    for row, source in zip(rows, fdtd):
        row["control_axis"] = source.get("control_axis")
        row["control_value"] = source.get("control_value")
        row["fdtd_source_path"] = source["_source_path"]
        row["sidebands"] = sidebands
    gain_map.write_points_csv(output / "map_points.csv", rows)
    summary = {
        "device": device, "sidebands": sidebands, "requested": len(points),
        "converged": sum(row.get("status") == "PASS" for row in rows),
        "source_path": str(output / "map_points.csv"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=tuple(DEVICE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidebands", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.output, args.sidebands), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
