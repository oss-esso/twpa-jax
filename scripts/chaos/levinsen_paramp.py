#!/usr/bin/env python3
"""Four-state Levinsen--Tornes Josephson parametric-amplifier diagnostic.

All quantities use the junction plasma frequency as the time scale.  The
McCumber parameter therefore appears as the inverse damping factor
``1/sqrt(beta_c)``; using ``beta_c`` as the inertial coefficient would instead
normalize by the characteristic (not plasma) frequency and misses the stated
265 Hz small-signal resonance.
The tuned-circuit state is the current delivered by the series resonator; the
load current is its sum with the injected signal current, as in Fig. 5.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LevinsenParameters:
    beta_c: float = 25.0
    junction_frequency_hz: float = 318.0
    dc_bias: float = 0.4
    pump_frequency_hz: float = 480.0
    signal_frequency_hz: float = 265.0
    load_resistance_ratio: float = 4.0
    tuned_q: float = 10.0
    tuned_circuit_frequency_hz: float | None = None
    pump_amplitude: float = 0.230
    josephson_current: float = 1.0

    @property
    def pump_omega(self) -> float:
        return self.pump_frequency_hz / self.junction_frequency_hz

    @property
    def signal_omega(self) -> float:
        return self.signal_frequency_hz / self.junction_frequency_hz

    @property
    def tuned_omega(self) -> float:
        if self.tuned_circuit_frequency_hz is None:
            raise ValueError("tuned_circuit_frequency_hz must be measured or supplied explicitly")
        return self.tuned_circuit_frequency_hz / self.junction_frequency_hz


Signal = Callable[[float], float]


def levinsen_rhs(
    time: float, state: np.ndarray, parameters: LevinsenParameters,
    signal_current: float | Signal = 0.0,
) -> np.ndarray:
    """Evaluate the normalized four-state Levinsen equations."""
    phi, velocity, charge, tuned_current = np.asarray(state, dtype=float)
    signal = signal_current(time) if callable(signal_current) else float(signal_current)
    drive = (parameters.dc_bias
             + parameters.pump_amplitude * math.cos(parameters.pump_omega * time)
             + signal)
    acceleration = (
        drive - parameters.josephson_current * math.sin(phi)
        - tuned_current - velocity / math.sqrt(parameters.beta_c)
    )
    omega_0 = parameters.tuned_omega
    resonator_acceleration = (
        omega_0 / (parameters.load_resistance_ratio * parameters.tuned_q) * velocity
        - omega_0 / parameters.tuned_q * tuned_current
        - omega_0 * omega_0 * charge
    )
    return np.array([velocity, acceleration, tuned_current,
                     resonator_acceleration], dtype=float)


def rk4_step(
    time: float, state: np.ndarray, step: float,
    rhs: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Advance one deterministic fourth-order Runge--Kutta step."""
    k1 = rhs(time, state)
    k2 = rhs(time + step / 2.0, state + step * k1 / 2.0)
    k3 = rhs(time + step / 2.0, state + step * k2 / 2.0)
    k4 = rhs(time + step, state + step * k3)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_levinsen(
    parameters: LevinsenParameters, initial_state: np.ndarray | None = None,
    *, duration: float = 80.0, step: float = 0.002,
    signal_current: float | Signal = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate and return normalized time and the four-state trace."""
    if duration <= 0.0 or step <= 0.0:
        raise ValueError("duration and step must be positive")
    count = int(round(duration / step)) + 1
    times = np.arange(count, dtype=float) * step
    states = np.empty((count, 4), dtype=float)
    states[0] = np.zeros(4) if initial_state is None else np.asarray(initial_state, dtype=float)
    if states[0].shape != (4,):
        raise ValueError("initial_state must have four components")
    rhs = lambda time, state: levinsen_rhs(time, state, parameters, signal_current)
    for index in range(count - 1):
        states[index + 1] = rk4_step(times[index], states[index], step, rhs)
    return times, states


def phasor(signal: np.ndarray, times: np.ndarray, frequency: float) -> complex:
    """Extract a single-frequency complex phasor by orthogonal projection."""
    phase = 2.0 * math.pi * frequency * times
    return complex(2.0 * np.mean(signal * np.cos(phase)),
                  -2.0 * np.mean(signal * np.sin(phase)))


def gamma_from_phasors(signal_phasor: complex, tuned_phasor: complex) -> float:
    """Return ``Gamma = |i_out / i_in|^2`` for load-current phasors."""
    if signal_phasor == 0.0:
        raise ValueError("signal phasor must be nonzero")
    return float(abs((signal_phasor + tuned_phasor) / signal_phasor) ** 2)


def measure_ring_resonance(
    parameters: LevinsenParameters, *, duration: float = 40.0,
    step: float = 0.002, initial_phase: float = 1.0e-3,
) -> dict[str, float | str]:
    """Measure the small-signal ring resonance in physical hertz."""
    equilibrium = (
        math.asin(np.clip(parameters.dc_bias / parameters.josephson_current, -1.0, 1.0))
        if parameters.josephson_current else 0.0
    )
    times, states = integrate_levinsen(
        parameters, np.array([equilibrium + initial_phase, 0.0, 0.0, 0.0]),
        duration=duration, step=step,
    )
    tail = states[:, 1] - np.mean(states[:, 1])
    spectrum = np.abs(np.fft.rfft(tail * np.hanning(tail.size)))
    frequencies = np.fft.rfftfreq(tail.size, step) * (
        2.0 * math.pi * parameters.junction_frequency_hz
    )
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    return {
        "measured_resonance_hz": float(frequencies[index]),
        "requested_signal_hz": float(parameters.signal_frequency_hz),
        "normalization": "time is normalized by angular plasma frequency 2*pi*junction_frequency_hz",
        "tuned_frequency_assumption_hz": float(parameters.tuned_circuit_frequency_hz),
    }


def run_gain_point(
    parameters: LevinsenParameters, *, signal_amplitude: float = 1.0,
    noise_amplitude: float = 0.0, duration: float = 80.0, step: float = 0.002,
) -> dict[str, float | str]:
    """Run one pump point and measure signal and white-noise gain."""
    rng = np.random.default_rng(12345)
    noise = rng.normal(0.0, noise_amplitude, int(round(duration / step)) + 1)
    signal = signal_amplitude * np.sin(parameters.signal_omega * np.arange(noise.size) * step)
    input_current = signal + noise
    callback = lambda time: float(np.interp(time, np.arange(noise.size) * step, input_current))
    times, states = integrate_levinsen(
        parameters, duration=duration, step=step, signal_current=callback,
    )
    discard = times.size // 2
    signal_phasor = phasor(signal[discard:], times[discard:], parameters.signal_omega / (2.0 * math.pi))
    tuned_phasor = phasor(states[discard:, 3], times[discard:], parameters.signal_omega / (2.0 * math.pi))
    gain = gamma_from_phasors(signal_phasor, tuned_phasor)
    output = input_current[discard:] + states[discard:, 3]
    spectrum_signal = np.abs(np.fft.rfft(
        (states[discard:, 1] - np.mean(states[discard:, 1]))
        * np.hanning(states[discard:, 1].size)
    ))
    spectrum_frequency_hz = (
        np.fft.rfftfreq(states[discard:, 1].size, step)
        * (2.0 * math.pi * parameters.junction_frequency_hz)
    )
    half_index = int(np.argmin(
        np.abs(spectrum_frequency_hz - parameters.pump_frequency_hz / 2.0)
    ))
    gain_available = bool(gain > 1.0 and parameters.pump_amplitude > 0.0)
    noise_gain = (
        float(np.std(output - signal[discard:]) / max(np.std(noise[discard:]), 1e-30))
        if noise_amplitude and gain_available else float("nan")
    )
    return {"pump_amplitude": parameters.pump_amplitude, "gain_linear": gain,
            "gain_db": 10.0 * math.log10(max(gain, 1e-300)),
            "noise_gain_amplitude": noise_gain,
            "noise_temperature_ratio": noise_gain * noise_gain if math.isfinite(noise_gain) else float("nan"),
            "half_harmonic_frequency_hz": float(spectrum_frequency_hz[half_index]),
            "half_harmonic_amplitude": float(spectrum_signal[half_index]),
            "noise_result": (
                "constant_noise_temperature" if noise_amplitude and gain_available
                else "not_evaluable_no_gain" if noise_amplitude else "not_run"
            )}


def run_pump_sweep(
    output: Path, *, amplitudes: np.ndarray, noise_amplitude: float = 0.0,
    tuned_circuit_frequency_hz: float, duration: float = 80.0, step: float = 0.002,
) -> dict[str, object]:
    """Run a resumable pump sweep, rewriting the JSON after every point."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    base_parameters = LevinsenParameters(
        tuned_circuit_frequency_hz=tuned_circuit_frequency_hz,
        pump_amplitude=0.0,
    )
    ring = measure_ring_resonance(base_parameters)
    output.write_text(json.dumps({"status": "IN_PROGRESS", "ring_measurement": ring,
                                  "rows": rows}, indent=2), encoding="utf-8")
    for amplitude in amplitudes:
        parameters = LevinsenParameters(
            pump_amplitude=float(amplitude),
            tuned_circuit_frequency_hz=tuned_circuit_frequency_hz,
        )
        rows.append(run_gain_point(parameters, noise_amplitude=noise_amplitude,
                                   duration=duration, step=step))
        output.write_text(json.dumps({"status": "IN_PROGRESS", "parameters": asdict(parameters),
                                      "rows": rows}, indent=2), encoding="utf-8")
    payload = {"status": "COMPLETE", "amplitudes": amplitudes.tolist(), "rows": rows,
               "ring_measurement": ring,
               "target": {"gain_db": ">30 at half-harmonic threshold",
                          "post_threshold": "gain collapse and no noise rise"}}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=0.20)
    parser.add_argument("--stop", type=float, default=0.27)
    parser.add_argument("--num", type=int, default=29)
    parser.add_argument("--noise-amplitude", type=float, default=0.02)
    parser.add_argument("--tuned-frequency-hz", type=float, required=True,
                        help="explicit tuned-circuit frequency; 240 Hz is the paper-figure assumption")
    parser.add_argument("--duration", type=float, default=80.0)
    parser.add_argument("--step", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=Path("outputs/chaos/phase1/levinsen_pump_sweep.json"))
    parser.add_argument("--ring-only", action="store_true")
    args = parser.parse_args(argv)
    parameters = LevinsenParameters(
        tuned_circuit_frequency_hz=args.tuned_frequency_hz, pump_amplitude=0.0,
    )
    if args.ring_only:
        payload = measure_ring_resonance(parameters)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    payload = run_pump_sweep(args.output, amplitudes=np.linspace(args.start, args.stop, args.num),
                              noise_amplitude=args.noise_amplitude,
                              tuned_circuit_frequency_hz=args.tuned_frequency_hz,
                              duration=args.duration,
                              step=args.step)
    print(json.dumps({"status": payload["status"], "output": str(args.output),
                      "points": len(payload["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
