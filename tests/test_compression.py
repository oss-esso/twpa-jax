from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from tests.test_multitone_physics import _jpa, _pump, _settings
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.compression import solve_signal_power_point
from twpa_solver.multitone.observables import (
    reference_states,
    spatial_profiles,
    tone_s21,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver
from twpa_solver.multitone.compression_curve import (
    build_compression_curve,
    depletion_only_model,
    refine_p1db,
)
from twpa_solver.core import CircuitMatrices


def test_refine_p1db_and_depletion_model() -> None:
    crossing = refine_p1db(lambda power: power + 40.0, (-40.0, -30.0))
    assert crossing == pytest.approx(-39.0, abs=0.01)
    assert depletion_only_model(10.0, 1.0, 100.0) == pytest.approx(10.0 / 1.2)


def test_curve_reports_nonmonotonic_compression() -> None:
    curve = build_compression_curve([-40, -39, -38, -37], [20, 19, 19.5, 18], 20)
    assert curve.first_1db_crossing_dbm == -39
    assert curve.number_of_crossings == 2
    assert curve.nonmonotonic_compression


def test_spatial_profiles_validate_chain_and_unwrap_phase() -> None:
    circuit, _metadata = _jpa()
    basis = build_three_tone_basis(10.0, 1.0)
    state = np.ones((basis.n_tones, circuit.node_count), dtype=np.complex128)
    with pytest.raises(ValueError, match="two-node chain"):
        spatial_profiles(state, basis, circuit)

    incidence = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, -1.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
    )
    chain = CircuitMatrices(
        C=sp.eye(4, format="csr"),
        G=sp.csr_matrix((4, 4)),
        K=sp.eye(4, format="csr"),
        Bphi=incidence,
        Ic=np.ones(3),
    )
    chain_state = np.zeros((basis.n_tones, 4), dtype=np.complex128)
    chain_state[:, 0] = 1.0
    chain_state[:, 1] = np.exp(0.25j)
    chain_state[:, 2] = np.exp(0.5j)
    chain_state[:, 3] = np.exp(0.75j)
    profiles = spatial_profiles(chain_state, basis, chain)
    assert [row["branch_index"] for row in profiles] == [0, 1, 2]
    assert all(np.isfinite(row["delta_k_eff_rad_per_cell"]) for row in profiles)


def test_reference_states_execute_all_four_solve_paths() -> None:
    circuit, _metadata = _jpa()
    basis = build_three_tone_basis(10.0, 1.0)
    shape = (basis.n_tones, circuit.node_count)
    pump_source = np.zeros(shape, dtype=np.complex128)
    signal_source = np.zeros(shape, dtype=np.complex128)
    problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.pump_turn_on(pump_source),
    )

    class RecordingSolver:
        def __init__(self) -> None:
            self.calls = 0

        def solve_one(self, candidate, seed, scale):
            del candidate, scale
            self.calls += 1
            return np.asarray(seed), SimpleNamespace(converged=True, failure_reason="")

    solver = RecordingSolver()
    states = reference_states(
        problem=problem,
        pump_source=pump_source,
        signal_source=signal_source,
        finite_signal_current_a=1.0e-9,
        solver=solver,
        pump_seed=np.zeros(shape, dtype=np.complex128),
    )
    assert solver.calls == 4
    assert set(states) == {
        "pump_off_signal_on",
        "pump_on_signal_infinitesimal",
        "pump_on_signal_finite",
        "pump_on_signal_off",
    }


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
    solver = HarmonicNewtonKrylovSolver(_settings())
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
