from __future__ import annotations

import math

import numpy as np
import pytest

from tests.test_multitone_physics import _jpa, _pump
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.compression import solve_signal_power_point
from twpa_solver.multitone.observables import tone_s21
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver
from twpa_solver.multitone.compression_curve import (
    build_compression_curve,
    depletion_only_model,
    refine_p1db,
)


def test_refine_p1db_and_depletion_model() -> None:
    crossing = refine_p1db(lambda power: power + 40.0, (-40.0, -30.0))
    assert crossing == pytest.approx(-39.0, abs=0.01)
    assert depletion_only_model(10.0, 1.0, 100.0) == pytest.approx(10.0 / 1.2)


def test_curve_reports_nonmonotonic_compression() -> None:
    curve = build_compression_curve([-40, -39, -38, -37], [20, 19, 19.5, 18], 20)
    assert curve.first_1db_crossing_dbm == -39
    assert curve.number_of_crossings == 2
    assert curve.nonmonotonic_compression


@pytest.mark.slow
def test_real_multitone_compression_sweep_warm_starts_nearest_state() -> None:
    circuit, metadata = _jpa()
    pump_problem, pump_state, _solution, _khat = _pump(circuit, metadata)
    basis = build_three_tone_basis(
        2.0 * math.pi * 4.75001e9, 2.0 * math.pi * 0.25001e9
    )
    pump_source = MultiToneDrive(
        basis.pump_tone,
        circuit.port_to_index[1],
        pump_problem.pump_current_a,
    ).to_coeffs(basis, circuit.C.shape[0])
    signal_source = MultiToneDrive(
        basis.signal_tone, circuit.port_to_index[1], 1.0
    ).to_coeffs(basis, circuit.C.shape[0])
    base = FullMultiToneProblem(
        circuit, basis, AffineSourcePath.pump_turn_on(pump_source)
    )
    initial = np.zeros((basis.n_tones, circuit.C.shape[0]), dtype=np.complex128)
    initial[basis.index_of(basis.pump_tone)] = pump_state[0]
    solver = HarmonicNewtonKrylovSolver(_pump.__globals__["_settings"]())
    currents = np.geomspace(1e-12, 1e-10, 5)
    previous = initial
    previous_previous = None
    gains = []
    for current in currents:
        point = solve_signal_power_point(
            base,
            previous,
            previous_previous,
            float(current),
            pump_source=pump_source,
            signal_source=signal_source,
            solver=solver,
        )
        assert point.status == "VALID_SOLVED"
        gains.append(
            20.0
            * math.log10(
                abs(
                    tone_s21(
                        point.state,
                        basis,
                        circuit,
                        signal_tone=basis.signal_tone,
                        source_port=1,
                        out_port=1,
                        source_current_a=float(current),
                    )
                )
            )
        )
        previous_previous = previous
        previous = point.state
    assert len(gains) == len(currents)
    assert all(np.isfinite(gains))
