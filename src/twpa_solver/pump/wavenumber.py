"""Standing-wave-safe pump wavenumber measurements."""

from __future__ import annotations

from typing import Any

import numpy as np

from twpa_solver.pump.problem import FullPumpProblem


def measure_pump_nonlinear_wavenumber(
    problem: FullPumpProblem,
    linear_state: np.ndarray,
    pumped_state: np.ndarray,
    *,
    cell_length_m: float,
    end_mask_cells: int = 35,
) -> dict[str, Any]:
    """Measure the pump-induced wavenumber reduction from two HB states.

    The interior node phasor obeys the lossless uniform-ladder recurrence
    ``x[n-1] + x[n+1] = 2 cos(q) x[n]`` for any mixture of forward and
    backward Bloch waves.  Fitting this recurrence therefore removes the
    standing-wave bias that contaminates a straight fit to ``angle(x[n])``.

    The returned nonlinear shift is ``k_linear - k_pumped``.  It is positive
    when a positive cubic branch coefficient stiffens the branch and reduces
    its propagation wavenumber.
    """
    modes = np.rint(problem.grid.k).astype(int)
    matches = np.flatnonzero(modes == int(problem.source_mode))
    if matches.size != 1:
        raise ValueError(
            f"source mode {problem.source_mode} occurs {matches.size} times in "
            f"pump modes {modes.tolist()}"
        )
    source_row = int(matches[0])
    linear = np.asarray(linear_state, dtype=np.complex128)
    pumped = np.asarray(pumped_state, dtype=np.complex128)
    if linear.shape != pumped.shape or linear.ndim != 2:
        raise ValueError("linear and pumped states must have the same 2-D shape")
    if linear.shape[0] != modes.size:
        raise ValueError("state rows do not match the harmonic grid")
    node_count = linear.shape[1]
    if end_mask_cells < 1 or 2 * end_mask_cells + 5 >= node_count:
        raise ValueError("end_mask_cells leaves too few interior nodes")
    node_indices = np.arange(end_mask_cells, node_count - end_mask_cells)

    def fit_state(state: np.ndarray) -> dict[str, float]:
        values = state[source_row, node_indices]
        center = values[1:-1]
        neighbors = values[:-2] + values[2:]
        denominator = float(np.vdot(2.0 * center, 2.0 * center).real)
        cosine = float(
            np.vdot(2.0 * center, neighbors).real / max(denominator, 1e-300)
        )
        q_rad_per_cell = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        recurrence_fit = 2.0 * cosine * center
        recurrence_residual = float(
            np.linalg.norm(neighbors - recurrence_fit)
            / max(np.linalg.norm(neighbors), 1e-300)
        )

        phase = np.unwrap(np.angle(values))
        phase_slope = float(np.polyfit(node_indices * cell_length_m, phase, 1)[0])
        naive_wavenumber = abs(phase_slope)

        forward = np.exp(-1j * q_rad_per_cell * node_indices)
        backward = np.exp(+1j * q_rad_per_cell * node_indices)
        design = np.column_stack((forward, backward))
        amplitudes, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ amplitudes
        fit_residual = float(
            np.linalg.norm(values - fitted) / max(np.linalg.norm(values), 1e-300)
        )
        larger = float(np.max(np.abs(amplitudes)))
        smaller = float(np.min(np.abs(amplitudes)))
        envelope = np.abs(values)
        ripple = float(
            (np.max(envelope) - np.min(envelope))
            / max(np.mean(envelope), 1e-300)
        )
        return {
            "wavenumber_rad_per_m": q_rad_per_cell / cell_length_m,
            "q_rad_per_cell": q_rad_per_cell,
            "recurrence_relative_residual": recurrence_residual,
            "forward_backward_fit_relative_residual": fit_residual,
            "backward_forward_amplitude_ratio": smaller / max(larger, 1e-300),
            "amplitude_envelope_peak_to_peak_over_mean": ripple,
            "naive_phase_slope_wavenumber_rad_per_m": naive_wavenumber,
        }

    linear_fit = fit_state(linear)
    pumped_fit = fit_state(pumped)
    nonlinear = (
        linear_fit["wavenumber_rad_per_m"]
        - pumped_fit["wavenumber_rad_per_m"]
    )
    return {
        "source_mode": int(problem.source_mode),
        "source_row": source_row,
        "pump_modes": modes.tolist(),
        "end_mask_cells": int(end_mask_cells),
        "linear": linear_fit,
        "pumped": pumped_fit,
        "dk_nl_rad_per_m": float(nonlinear),
        "nonlinear_phase_rad": float(nonlinear * cell_length_m * (node_count - 1)),
    }
