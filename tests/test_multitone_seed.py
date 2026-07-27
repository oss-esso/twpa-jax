from __future__ import annotations

import numpy as np
import math

from tests.test_multitone_physics import _basis, _jpa, _pump, _pump_source
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.multitone.seed import promote_pump_solution, seed_from_floquet
from twpa_solver.pump import HarmonicNewtonKrylovSolver
from twpa_solver.pump.basis import PumpBasis
from twpa_solver.signal.floquet import solve_gain_one_schur


def test_promote_pump_solution_maps_q_zero_sector() -> None:
    basis = build_three_tone_basis(10.0, 1.0)
    pump_basis = PumpBasis([1], "dense_real", 10.0)
    pump = np.array([[2.0 + 1.0j, 3.0]], dtype=np.complex128)

    promoted = promote_pump_solution(pump, pump_basis, basis)

    np.testing.assert_array_equal(promoted[basis.index_of(basis.pump_tone)], pump[0])
    assert np.count_nonzero(promoted) == 2


def test_seed_from_floquet_maps_sideband_to_signal_tone() -> None:
    basis = build_three_tone_basis(10.0, 1.0)
    response = {0: np.array([2.0 + 1.0j])}

    seed = seed_from_floquet(
        basis,
        response,
        omega_p=10.0,
        omega_s=9.0,
        signal_current_a=1e-6,
    )

    np.testing.assert_allclose(
        seed[basis.index_of(basis.signal_tone), 0], (2.0 + 1.0j) * 1e-6
    )


def test_real_floquet_seed_warm_starts_finite_signal_sweep() -> None:
    circuit, metadata = _jpa()
    pump_problem, pump_state, _solution, khat = _pump(circuit, metadata)
    omega_p = 2.0 * math.pi * 4.75001e9
    basis = _basis(4.5)
    pump_source = _pump_source(pump_problem, basis)
    floquet_response = {}
    for node in range(circuit.C.shape[0]):
        result = solve_gain_one_schur(
            circuit=circuit,
            khat=khat,
            khat_off_0=khat[0],
            omega_p=omega_p,
            signal_ghz=4.5,
            sidebands=1,
            signal_m=0,
            idler_m=1,
            source_index=circuit.port_to_index[1],
            out_index=node,
            source_current_a=1.0,
            source_port=1,
            out_port=1,
            z0_ohm=50.0,
            include_baselines=False,
        )
        floquet_response.setdefault(0, np.zeros(circuit.C.shape[0], complex))
        floquet_response[0][node] = result.vout_on / (1j * 2.0 * math.pi * 4.5e9)
    currents = np.geomspace(1e-12, 1e-10, 15)
    seed = promote_pump_solution(
        pump_state, PumpBasis([1, 3, 5], "dense_real", omega_p), basis
    ) + seed_from_floquet(
        basis,
        floquet_response,
        omega_p=omega_p,
        omega_s=2.0 * math.pi * 4.5e9,
        signal_current_a=float(currents[0]),
    )
    settings = _pump.__globals__["_settings"]()
    solver = HarmonicNewtonKrylovSolver(settings)
    previous = seed
    cold_solves = 0
    iterations = []
    for current in currents:
        source_signal = MultiToneDrive(
            basis.signal_tone, circuit.port_to_index[1], float(current)
        ).to_coeffs(basis, circuit.C.shape[0])
        problem = FullMultiToneProblem(
            circuit,
            basis,
            AffineSourcePath.signal_turn_on(pump_source, source_signal),
        )
        state, report = solver.solve_one(problem, previous, 1.0)
        assert report.converged
        iterations.append(report.newton_iterations)
        previous = state
    assert iterations[0] <= 3
    assert len(iterations) == 15
    assert cold_solves == 0
