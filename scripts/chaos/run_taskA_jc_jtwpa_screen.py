"""Run the bounded jc_jtwpa timestep screen at one signal-driven point."""
from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.chaos import measure_ansatz_validity as ansatz
from scripts.chaos import run_guarcello_jc_phase5 as phase5
from scripts.chaos import run_phaseB_pump_only as phaseb


ROOT = Path(__file__).resolve().parents[2]
DEVICE = "jc_jtwpa"
POWER_DBM = -27.8
DT_NORM = 0.005
SIGNAL_CURRENT_A = 3.0e-8
PERIODS = 2100.0
OUTPUT = ROOT / "outputs" / "chaos" / "phaseB_signal" / DEVICE / "dense_m27p8000_dt005"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(phaseb._json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs" / "jc_doc_python_designs" / DEVICE
    spec = phase5.derive_device_spec(source)
    pump_hz = phase5.resolve_pump_frequency(spec)
    tmax_norm = PERIODS * spec.omega_plasma / pump_hz
    device = phase5.load_jc_device(source)

    gate = phase5._measure_linear_limit(
        device, spec, pump_hz, DT_NORM, retain_linear_inductance=True,
    )
    _atomic_json(OUTPUT / "linear_gate.json", {
        "device": DEVICE,
        "dt_norm": DT_NORM,
        "pump_hz": pump_hz,
        **gate,
    })

    started = time.perf_counter()
    row = phaseb._run_point(
        DEVICE, POWER_DBM, OUTPUT, DT_NORM, tmax_norm,
        signal_current_a=SIGNAL_CURRENT_A,
        pump_off_output=None,
    )
    runtime = time.perf_counter() - started
    reduced = ansatz.analyse_point(OUTPUT, 400)
    _atomic_json(OUTPUT / "taskA_summary.json", {
        "device": DEVICE,
        "pump_power_dbm": POWER_DBM,
        "dt_norm": DT_NORM,
        "signal_current_a": SIGNAL_CURRENT_A,
        "signal_hz": spec.signal_ghz * 1e9,
        "pump_hz": pump_hz,
        "pump_periods_requested": PERIODS,
        "tmax_norm": tmax_norm,
        "linear_gate": gate,
        "ansatz": reduced,
        "result": row,
        "wrapper_runtime_s": runtime,
    })
    print(json.dumps({"linear_gate": gate, "ansatz": reduced, "result": row}, indent=2))
    return 0 if row.get("status") != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
