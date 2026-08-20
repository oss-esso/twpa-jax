"""Measure JAX float32/float64 divergence for the fixed IPM chaos point.

The probe deliberately uses the JAX backend for both precisions and both
banded-solve implementations.  It does not modify campaign artifacts unless
``--output`` is supplied.  The pump-only observable is reduced with the same
``measure_ansatz_validity`` implementation used by the stored corpus.

PAUSED 2026-08-20: this is the physics gate of the GPU-session toolchain, which
is built and CPU-smoke-tested but has never been run on a GPU.  Its GO/NO-GO
verdict has therefore never been produced on hardware, and no result in this
repository depends on it.  Read ``docs/development/gpu_session_runbook.md``
before reviving it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.chaos.measure_ansatz_validity import (
    BAND_HIGH_MULTIPLE,
    BAND_LOW_FRACTION,
    _reduce_lattice,
)
from scripts.chaos.run_guarcello_jc_phase5 import (
    JAX_SOLVE_KINDS,
    PHI0_REDUCED,
    derive_device_spec,
    integrate_jc_banded_batch,
    load_jc_device,
    resolve_pump_frequency,
)


ROOT = Path(__file__).resolve().parents[2]
CONTROL_VALUE = 0.5975
CONTROL_CURRENT_SCALE_A = 1.1628e-5
PUMP_HZ = 7.9e9
SIGNAL_HZ = 7.4e9
SIGNAL_CURRENT_A = 0.0
DEFAULT_TRAJECTORY_STEPS = (600, 6_000, 60_000)
DEFAULT_OBSERVABLE_STEPS = 6_271_516

def _spectrum(
    t: np.ndarray, v: np.ndarray, pump_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the windowed spectrum normalized to the pump bin."""
    centered = v - np.mean(v)
    window = np.hanning(v.size)
    amplitude = (
        2.0 * np.abs(np.fft.rfft(centered * window))
        / max(np.sum(window), 1.0e-30)
    )
    frequencies = np.fft.rfftfreq(v.size, np.mean(np.diff(t)))
    pump_index = int(np.argmin(np.abs(frequencies - pump_hz)))
    pump_amplitude = max(float(amplitude[pump_index]), np.finfo(float).tiny)
    spectrum_db = 20.0 * np.log10(
        np.maximum(amplitude, np.finfo(float).tiny) / pump_amplitude
    )
    return frequencies, spectrum_db, pump_amplitude

def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Return a norm-relative difference, handling two zero arrays."""
    numerator = float(np.linalg.norm(np.asarray(first) - np.asarray(second)))
    denominator = float(np.linalg.norm(np.asarray(second)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _growth_summary(
    steps: list[int], errors: list[float],
) -> dict[str, Any]:
    """Summarize bounded/power-law/exponential error growth without tuning."""
    if len(steps) < 2:
        return {
            "classification": "insufficient_steps",
            "steps": steps,
            "errors": errors,
        }
    finite = np.asarray(
        [max(float(error), np.finfo(float).tiny) for error in errors],
        dtype=float,
    )
    step_values = np.asarray(steps, dtype=float)
    log_steps = np.log(step_values)
    log_errors = np.log(finite)
    power_slope = float(np.polyfit(log_steps, log_errors, 1)[0])
    exponential_slope = float(np.polyfit(step_values, log_errors, 1)[0])
    if float(np.max(finite)) <= max(1.0e-12, 10.0 * float(finite[0])):
        classification = "bounded"
    elif exponential_slope > 0.0 and power_slope > 2.0:
        classification = "exponential"
    else:
        classification = "linear_or_power_law"
    return {
        "classification": classification,
        "steps": steps,
        "errors": errors,
        "power_law_slope": power_slope,
        "exponential_log_slope_per_step": exponential_slope,
    }


def _pump_only_fraction(
    times: np.ndarray,
    voltage: np.ndarray,
    pump_hz: float,
) -> float:
    """Reduce one recorded voltage trace with the corpus pump-only lattice."""
    frequency, spectrum_db, _pump_amplitude = _spectrum(times, voltage, pump_hz)
    band = (
        (frequency > BAND_LOW_FRACTION * pump_hz)
        & (frequency < BAND_HIGH_MULTIPLE * pump_hz)
        & np.isfinite(spectrum_db)
    )
    frequency = frequency[band]
    power = 10.0 ** (spectrum_db[band] / 10.0)
    if frequency.size < 2:
        raise ValueError("precision probe produced too few spectral bins")
    window_hz = 3.0 * float(np.median(np.diff(frequency)))
    reduction = _reduce_lattice(
        frequency,
        power,
        pump_hz,
        SIGNAL_HZ,
        window_hz,
        400,
        pump_only=True,
    )
    return float(reduction["off_lattice"])


def _run_case(
    device: Any,
    *,
    current_a: float,
    dt_s: float,
    n_steps: int,
    record_stride: int,
    dtype: str,
    solve_kind: str,
    jax_device: str,
) -> dict[str, Any]:
    """Run one precision/solver combination and retain its small outputs."""
    started = time.perf_counter()
    result = integrate_jc_banded_batch(
        device,
        pump_currents_a=np.asarray([current_a]),
        pump_hz=PUMP_HZ,
        signal_current_a=SIGNAL_CURRENT_A,
        signal_hz=SIGNAL_HZ,
        dt_s=dt_s,
        n_steps=n_steps,
        record_stride=record_stride,
        backend="jax",
        jax_device=jax_device,
        solve_kind=solve_kind,
        dtype=dtype,
    )
    return {
        "dtype": dtype,
        "solve_kind": solve_kind,
        "n_steps": n_steps,
        "runtime_s": float(result[3]),
        "wall_s": time.perf_counter() - started,
        "times": np.asarray(result[0][0]),
        "voltage": np.asarray(result[1][0]),
        "branch_r": np.asarray(result[2][0]),
        "q_final": np.asarray(result[4][0]),
    }


def measure_precision(
    *,
    trajectory_steps: tuple[int, ...] = DEFAULT_TRAJECTORY_STEPS,
    observable_steps: int = DEFAULT_OBSERVABLE_STEPS,
    record_stride: int = 20,
    observable_solve_kinds: tuple[str, ...] = JAX_SOLVE_KINDS,
    jax_device: str = "cpu",
) -> dict[str, Any]:
    """Run the precision matrix and return a JSON-safe report."""
    source = ROOT / "designs" / "ipm_2c_fixed"
    spec = derive_device_spec(source)
    device = load_jc_device(Path(spec.circuit_dir))
    if device.natural_bandwidth != 4558:
        raise RuntimeError(
            "precision probe requires the live 6096-node build with "
            f"natural_bandwidth=4558, got {device.natural_bandwidth}"
        )
    if not math.isclose(resolve_pump_frequency(spec), PUMP_HZ, rel_tol=0.0, abs_tol=1.0):
        raise RuntimeError("precision probe pump frequency is not 7.9 GHz")
    if any(step <= 0 for step in trajectory_steps) or observable_steps <= 0:
        raise ValueError("all probe step counts must be positive")
    if not observable_solve_kinds or any(
        solve_kind not in JAX_SOLVE_KINDS for solve_kind in observable_solve_kinds
    ):
        raise ValueError("observable_solve_kinds contains an unsupported solve kind")
    dt_norm = 0.01
    dt_s = dt_norm / spec.omega_plasma
    current_a = CONTROL_VALUE * CONTROL_CURRENT_SCALE_A
    runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for solve_kind in JAX_SOLVE_KINDS:
        steps = list(trajectory_steps)
        if solve_kind in observable_solve_kinds:
            steps.append(observable_steps)
        for n_steps in sorted(set(steps)):
            for dtype in ("float64", "float32"):
                runs[(solve_kind, dtype, n_steps)] = _run_case(
                    device,
                    current_a=current_a,
                    dt_s=dt_s,
                    n_steps=n_steps,
                    record_stride=record_stride,
                    dtype=dtype,
                    solve_kind=solve_kind,
                    jax_device=jax_device,
                )

    trajectory: list[dict[str, Any]] = []
    for solve_kind in JAX_SOLVE_KINDS:
        q_errors: list[float] = []
        voltage_errors: list[float] = []
        for n_steps in sorted(trajectory_steps):
            reference = runs[(solve_kind, "float64", n_steps)]
            candidate = runs[(solve_kind, "float32", n_steps)]
            q_error = _relative_difference(candidate["q_final"], reference["q_final"])
            voltage_error = _relative_difference(candidate["voltage"], reference["voltage"])
            q_errors.append(q_error)
            voltage_errors.append(voltage_error)
            trajectory.append({
                "solve_kind": solve_kind,
                "n_steps": n_steps,
                "q_final_relative_error": q_error,
                "voltage_relative_error": voltage_error,
                "float64_q_norm": float(np.linalg.norm(reference["q_final"])),
                "float64_voltage_norm": float(np.linalg.norm(reference["voltage"])),
            })
        growth = {
            "solve_kind": solve_kind,
            "q_final": _growth_summary(list(sorted(trajectory_steps)), q_errors),
            "voltage": _growth_summary(list(sorted(trajectory_steps)), voltage_errors),
        }
        trajectory.append({"growth_summary": growth})

    observable: list[dict[str, Any]] = []
    for solve_kind in observable_solve_kinds:
        reference = runs[(solve_kind, "float64", observable_steps)]
        candidate = runs[(solve_kind, "float32", observable_steps)]
        reference_fraction = _pump_only_fraction(
            reference["times"], reference["voltage"], PUMP_HZ,
        )
        candidate_fraction = _pump_only_fraction(
            candidate["times"], candidate["voltage"], PUMP_HZ,
        )
        absolute_difference = abs(candidate_fraction - reference_fraction)
        observable.append({
            "solve_kind": solve_kind,
            "n_steps": observable_steps,
            "record_stride": record_stride,
            "total_pump_periods": float(reference["times"][-1] * PUMP_HZ),
            "float64_off_lattice": reference_fraction,
            "float32_off_lattice": candidate_fraction,
            "absolute_difference": absolute_difference,
            "passes_tolerance": absolute_difference < 1.0e-4,
        })
    return {
        "device": "ipm_2c_fixed",
        "jax_device": jax_device,
        "natural_bandwidth": device.natural_bandwidth,
        "selected_bandwidth": device.selected_bandwidth,
        "n_nodes": device.n_nodes,
        "pump_hz": PUMP_HZ,
        "control_value_I_over_I_bound": CONTROL_VALUE,
        "pump_current_a": current_a,
        "dt_norm": dt_norm,
        "dt_s": dt_s,
        "trajectory": trajectory,
        "observable": observable,
        "observable_tolerance_absolute": 1.0e-4,
        "observable_solve_kinds": list(observable_solve_kinds),
        "verdict": (
            "GO"
            if all(item["passes_tolerance"] for item in observable)
            else "NO_GO"
        ),
        "runs": [
            {
                "dtype": run["dtype"],
                "solve_kind": run["solve_kind"],
                "n_steps": run["n_steps"],
                "runtime_s": run["runtime_s"],
                "wall_s": run["wall_s"],
            }
            for run in runs.values()
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-steps", type=int, nargs="+",
        default=list(DEFAULT_TRAJECTORY_STEPS),
    )
    parser.add_argument("--observable-steps", type=int, default=DEFAULT_OBSERVABLE_STEPS)
    parser.add_argument(
        "--observable-solve-kinds", nargs="+", choices=JAX_SOLVE_KINDS,
        default=list(JAX_SOLVE_KINDS),
    )
    parser.add_argument("--record-stride", type=int, default=20)
    parser.add_argument(
        "--jax-device", choices=("cpu", "gpu"), default="cpu",
        help="device for the JAX backend; GPU sessions should select gpu",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = measure_precision(
        trajectory_steps=tuple(args.trajectory_steps),
        observable_steps=args.observable_steps,
        record_stride=args.record_stride,
        observable_solve_kinds=tuple(args.observable_solve_kinds),
        jax_device=args.jax_device,
    )
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
