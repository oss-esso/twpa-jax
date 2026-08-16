"""Validate the RF-SQUID phase-offset transient fix against HB and dt halving."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.chaos.run_guarcello_jc_phase5 import (
    _run_point,
    _tone_amplitude,
    derive_device_spec,
)


PUMP_HZ = 12.080e9
HB_CURRENT_A = 6.324555320336759e-06
HB_FUNDAMENTAL_V = 1.0681e-04
HB_RJ = 0.8616


def _tmax_norm_for_periods(spec, periods: float, dt_norm: float) -> float:
    steps_per_period = spec.omega_plasma / PUMP_HZ / dt_norm
    return periods * steps_per_period * dt_norm


def _run_case(
    spec,
    output: Path,
    *,
    label: str,
    current_a: float,
    dt_norm: float,
    periods: float,
    phi_dc_rad: float | None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    row, _, runtime, final_q, trace_t, trace_v = _run_point(
        spec,
        current_a,
        dt_norm=dt_norm,
        tmax_norm=_tmax_norm_for_periods(spec, periods, dt_norm),
        signal_current_a=0.0,
        pump_off_output=None,
        initial_state=None,
        phi_dc_rad=phi_dc_rad,
    )
    np.savez_compressed(output / "trace.npz", t=trace_t, v_out=trace_v)
    late = np.arange(trace_t.size) >= max(1, trace_t.size // 2)
    amplitude = _tone_amplitude(trace_t[late], trace_v[late], PUMP_HZ)
    payload = {
        "label": label,
        "device": spec.name,
        "pump_hz": PUMP_HZ,
        "pump_current_a": current_a,
        "dt_norm": dt_norm,
        "requested_pump_periods": periods,
        "actual_pump_periods": float(trace_t[-1] * PUMP_HZ),
        "phi_dc_rad": float(row["phi_dc_rad"]),
        "bias_applied": bool(row["bias_applied"]),
        "fundamental_output_amplitude_v": amplitude,
        "max_abs_output_voltage_v": float(np.max(np.abs(trace_v))),
        "r_j": float(row["r_j"]),
        "runtime_s": runtime,
        "row": row,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "chaos" / "rf_squid_td_fix",
    )
    parser.add_argument("--stage1-periods", type=float, default=600.0)
    parser.add_argument("--dt-check-periods", type=float, default=150.0)
    args = parser.parse_args()

    source = ROOT / "designs" / "rf_squid_2393_3wm.yaml"
    spec = derive_device_spec(source)
    args.output.mkdir(parents=True, exist_ok=True)

    stage1 = _run_case(
        spec, args.output / "stage1_unbiased",
        label="stage1_unbiased_against_hb",
        current_a=HB_CURRENT_A, dt_norm=0.01,
        periods=args.stage1_periods, phi_dc_rad=0.0,
    )
    stage1["hb_reference"] = {
        "fundamental_output_amplitude_v": HB_FUNDAMENTAL_V,
        "r_j": HB_RJ,
        "fundamental_ratio_td_over_hb": (
            stage1["fundamental_output_amplitude_v"] / HB_FUNDAMENTAL_V
        ),
        "r_j_ratio_td_over_hb": stage1["r_j"] / HB_RJ,
    }
    (args.output / "stage1_unbiased" / "result.json").write_text(
        json.dumps(stage1, indent=2), encoding="utf-8",
    )

    dt_cases = []
    for dt_norm in (0.01, 0.005):
        dt_cases.append(_run_case(
            spec, args.output / f"dt_check_{dt_norm:.4f}",
            label="biased_zero_seed_dt_check",
            current_a=1.0e-7, dt_norm=dt_norm,
            periods=args.dt_check_periods, phi_dc_rad=spec.phi_dc_rad,
        ))
    dt_ratio = (
        dt_cases[1]["max_abs_output_voltage_v"]
        / dt_cases[0]["max_abs_output_voltage_v"]
    )
    summary = {
        "status": "COMPLETE",
        "change_1": "static continuous equilibrium seed removed; runs start from zero unless warm-started",
        "change_2": "dc port current removed; RF-SQUID bias is a uniform branch phase offset",
        "stage1": stage1,
        "dt_halving_check": {
            "cases": dt_cases,
            "max_voltage_ratio_dt_half_over_coarse": dt_ratio,
            "interpretation": "ratio near one supports timestep convergence; ratio near four would indicate the removed seed artifact",
        },
    }
    (args.output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "stage1_fundamental_ratio": stage1["hb_reference"]["fundamental_ratio_td_over_hb"],
        "stage1_rj_ratio": stage1["hb_reference"]["r_j_ratio_td_over_hb"],
        "dt_half_max_voltage_ratio": dt_ratio,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
