from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices, load_circuit
from twpa_solver.multitone.basis import (
    MultiToneBasis,
    ToneIndex,
    build_sideband_matched_basis,
)
from twpa_solver.multitone.observables import tone_s21
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.schur import build_multitone_schur_problem
from twpa_solver.multitone.seed import promote_pump_solution
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import HarmonicNewtonKrylovSolver, NewtonKrylovSettings
from twpa_solver.pump.backends.schur_partition import restrict
from twpa_solver.signal.floquet import GainResult, solve_gain_one
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.io import PumpSolution, load_pump


ROOT = Path(__file__).resolve().parents[1]
SIGNAL_CURRENT_A = 1e-12


@dataclass(frozen=True)
class DistributedCase:
    name: str
    artifact: str
    pump_ghz: float
    signal_ghz: float
    expected_gain_vs_off_db: float


@dataclass(frozen=True)
class ParityMeasurement:
    sidebands: int
    multitone_gain_db: float
    multitone_gain_vs_off_db: float
    floquet_gain_db: float
    floquet_gain_vs_off_db: float
    pump_only_max_abs_err: float
    pump_only_off_q_max_abs: float
    runtime_s: float


CASES = (
    DistributedCase(
        "jtwpa",
        "exp14_jtwpa_odd10_scale2",
        7.12,
        6.6,
        27.542559891831555,
    ),
    DistributedCase(
        "fqjtwpa",
        "exp14_fqjtwpa_odd10_scale2",
        7.9,
        7.4,
        28.536893889606535,
    ),
)


def _settings() -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10,
        max_newton=20,
        gmres_rtol=1e-8,
        gmres_atol=0.0,
        gmres_restart=40,
        gmres_maxiter=60,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled_fast",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )


def _pump_off_khat(circuit: CircuitMatrices) -> sp.csr_matrix:
    return (
        circuit.Bphi
        @ sp.diags(circuit.Ic / circuit.phi0)
        @ circuit.Bphi.T
    ).tocsr()


def _pump_source(
    circuit: CircuitMatrices,
    pump: PumpSolution,
    basis: MultiToneBasis,
) -> np.ndarray:
    source = np.zeros(
        (basis.n_tones, circuit.node_count), dtype=np.complex128
    )
    row = basis.index_of(ToneIndex(1, 0))
    source[row, circuit.port_to_index[1]] = 0.5 * float(
        pump.metadata["pump_current_a"]
    )
    return source


def _signal_source(
    circuit: CircuitMatrices, basis: MultiToneBasis
) -> np.ndarray:
    return MultiToneDrive(
        basis.signal_tone,
        circuit.port_to_index[1],
        SIGNAL_CURRENT_A,
    ).to_coeffs(basis, circuit.node_count)


def _solve(
    circuit: CircuitMatrices,
    basis: MultiToneBasis,
    source_path: AffineSourcePath,
    initial_full: np.ndarray,
) -> tuple[np.ndarray, object]:
    full = FullMultiToneProblem(circuit, basis, source_path)
    problem = build_multitone_schur_problem(
        full,
        [circuit.port_to_index[1], circuit.port_to_index[2]],
    )
    state, report = HarmonicNewtonKrylovSolver(_settings()).solve_one(
        problem, restrict(initial_full, problem.part), 1.0
    )
    assert report.converged, report.failure_reason
    return problem.reconstruct_full(state), report


def measure_distributed_parity(
    case: DistributedCase,
    sidebands: int,
    *,
    idler_m: int = -2,
) -> ParityMeasurement:
    started = time.perf_counter()
    artifact = ROOT / "outputs" / case.artifact
    pump = load_pump(artifact / "pump", case.pump_ghz)
    circuit = load_circuit(ROOT / str(pump.metadata["ipm_dir"]))
    delta = pump.omega_p - 2.0 * math.pi * case.signal_ghz * 1e9
    basis = build_sideband_matched_basis(
        pump.modes,
        sidebands,
        pump.omega_p,
        delta,
        pump.omega_p * 22.0,
    )
    pump_source = _pump_source(circuit, pump, basis)
    signal_source = _signal_source(circuit, basis)
    pump_seed = promote_pump_solution(pump.X, pump.basis, basis)

    pump_only, _ = _solve(
        circuit,
        basis,
        AffineSourcePath.pump_turn_on(pump_source),
        pump_seed,
    )
    on, _ = _solve(
        circuit,
        basis,
        AffineSourcePath.signal_turn_on(pump_source, signal_source),
        pump_only,
    )
    off, _ = _solve(
        circuit,
        basis,
        AffineSourcePath.signal_turn_on(
            np.zeros_like(pump_source), signal_source
        ),
        np.zeros_like(pump_seed),
    )

    signal_row = basis.index_of(basis.signal_tone)
    out_node = circuit.port_to_index[2]
    gain_vs_off_db = 20.0 * math.log10(
        abs(on[signal_row, out_node] / off[signal_row, out_node])
    )
    gain_db = 20.0 * math.log10(
        abs(
            tone_s21(
                on,
                basis,
                circuit,
                signal_tone=basis.signal_tone,
                source_port=1,
                out_port=2,
                source_current_a=SIGNAL_CURRENT_A,
            )
        )
    )

    khat = build_khat(
        circuit.Bphi,
        compute_gamma_hat(circuit, pump, 2 * sidebands, pump.nt_original),
        1e-30,
    )
    floquet: GainResult = solve_gain_one(
        circuit=circuit,
        khat=khat,
        khat_off_0=_pump_off_khat(circuit),
        omega_p=pump.omega_p,
        signal_ghz=case.signal_ghz,
        sidebands=sidebands,
        signal_m=0,
        idler_m=idler_m,
        source_index=circuit.port_to_index[1],
        out_index=circuit.port_to_index[2],
        source_current_a=SIGNAL_CURRENT_A,
        source_port=1,
        out_port=2,
        z0_ohm=50.0,
    )

    pump_rows = [basis.index_of(ToneIndex(mode, 0)) for mode in pump.modes]
    pump_error = float(np.max(np.abs(pump_only[pump_rows] - pump.X)))
    off_q = [row for row, tone in enumerate(basis.tones) if tone.q != 0]
    return ParityMeasurement(
        sidebands=sidebands,
        multitone_gain_db=gain_db,
        multitone_gain_vs_off_db=gain_vs_off_db,
        floquet_gain_db=floquet.gain_db,
        floquet_gain_vs_off_db=floquet.gain_vs_off_db,
        pump_only_max_abs_err=pump_error,
        pump_only_off_q_max_abs=float(np.max(np.abs(pump_only[off_q]))),
        runtime_s=time.perf_counter() - started,
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_distributed_small_signal_parity(case: DistributedCase) -> None:
    measurement = measure_distributed_parity(case, sidebands=10)
    assert measurement.multitone_gain_vs_off_db > 3.0
    assert abs(
        measurement.multitone_gain_vs_off_db - case.expected_gain_vs_off_db
    ) < 0.05
    assert abs(
        measurement.multitone_gain_vs_off_db
        - measurement.floquet_gain_vs_off_db
    ) < 0.05
    assert abs(
        measurement.multitone_gain_db - measurement.floquet_gain_db
    ) < 0.05
    assert measurement.pump_only_max_abs_err < 1e-10
    assert measurement.pump_only_off_q_max_abs < 1e-12
