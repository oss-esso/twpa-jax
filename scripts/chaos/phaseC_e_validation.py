"""Run the short Phase C E1/E2 linear-limit diagnostics."""

from __future__ import annotations

import json
import math
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.chaos.run_guarcello_jc_phase5 import (
    _json_safe,
    _measure_linear_limit,
    derive_device_spec,
    load_jc_device,
    resolve_pump_frequency,
)


def measure_jc_finite_linear_gate(
    device_name: str,
    *,
    dt_norms: tuple[float, ...] = (0.01, 0.005, 0.0025),
) -> dict[str, dict]:
    """Measure the non-degenerate linear-limit error for a JC fixture."""
    source = ROOT / "outputs" / "jc_doc_python_designs" / device_name
    spec = derive_device_spec(source)
    device = load_jc_device(Path(spec.circuit_dir))
    pump_hz = resolve_pump_frequency(spec)
    result: dict[str, dict] = {}
    for dt_norm in dt_norms:
        row = _measure_linear_limit(
            device, spec, pump_hz, float(dt_norm),
            implicit_linear_stiffness=True,
            retain_linear_inductance=True,
        )
        row.update({"device": device_name, "dt_norm": float(dt_norm)})
        result[f"dt_norm_{str(dt_norm).replace('.', 'p')}"] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["ipm_2c_fixed", "rf_squid_2393_3wm"])
    parser.add_argument("--dt-norm", type=float)
    parser.add_argument(
        "--retain-linear-inductance", action="store_true",
        help="keep finite Josephson small-signal inductance when Ic is disabled",
    )
    args = parser.parse_args()
    output = ROOT / "outputs" / "chaos" / "phaseC"
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "ipm_2c_fixed": ROOT / "designs" / "ipm_2c_fixed",
        "rf_squid_2393_3wm": ROOT / "designs" / "rf_squid_2393_3wm.yaml",
    }
    path = output / "E_validation.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("E1_explicit_2c", {})
        payload.setdefault("E2_dt_scaling", {})
    else:
        payload = {"E1_explicit_2c": {}, "E2_dt_scaling": {}}

    e1_dts = (args.dt_norm,) if args.dt_norm is not None else (0.01, 0.005, 0.0025)
    for dt_norm in e1_dts:
        if args.device not in (None, "ipm_2c_fixed"):
            break
        if str(dt_norm) in payload["E1_explicit_2c"]:
            continue
        spec = derive_device_spec(sources["ipm_2c_fixed"])
        device = load_jc_device(Path(spec.circuit_dir))
        pump_hz = resolve_pump_frequency(spec)
        result = _measure_linear_limit(
            device, spec, pump_hz, dt_norm,
            implicit_linear_stiffness=False,
            retain_linear_inductance=args.retain_linear_inductance,
        )
        result.update({
            "device": "ipm_2c_fixed",
            "dt_norm": dt_norm,
            "omega_pump_dt": 2.0 * math.pi * pump_hz * dt_norm / spec.omega_plasma,
            "stiffness_path": "explicit_K_override",
        })
        payload["E1_explicit_2c"][str(dt_norm)] = result
        path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(json.dumps(_json_safe(result), sort_keys=True))

    e2_dts = (args.dt_norm,) if args.dt_norm is not None else (0.01, 0.005, 0.0025)
    selected_sources = (
        {args.device: sources[args.device]} if args.device is not None else sources
    )
    for dt_norm in e2_dts:
        for name, source in selected_sources.items():
            key = f"{name}:{dt_norm}"
            if key in payload["E2_dt_scaling"]:
                continue
            spec = derive_device_spec(source)
            device = load_jc_device(Path(spec.circuit_dir))
            pump_hz = resolve_pump_frequency(spec)
            result = _measure_linear_limit(
                device, spec, pump_hz, dt_norm,
                implicit_linear_stiffness=True,
                retain_linear_inductance=args.retain_linear_inductance,
            )
            result.update({
                "device": name,
                "dt_norm": dt_norm,
                "omega_pump_dt": 2.0 * math.pi * pump_hz * dt_norm / spec.omega_plasma,
                "stiffness_path": "implicit_K_default",
            })
            payload["E2_dt_scaling"][f"{name}:{dt_norm}"] = result
            path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
            print(json.dumps(_json_safe(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
