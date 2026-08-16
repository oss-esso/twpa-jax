from __future__ import annotations

import numpy as np

from twpa_solver.pump.basis import PumpBasis
from twpa_solver.pump.floquet import (
    build_period_doubled_seed,
    period_doubled_basis,
)
from twpa_solver.pump.periodic_branch import build_period_doubled_problem
from twpa_solver.signal.period_doubled import period_doubled_idler_sideband
from twpa_solver.core.nonlinear import JosephsonBranchLaw
import scipy.sparse as sp


class _Circuit:
    C = sp.eye(2, format="csr", dtype=np.complex128)
    G = sp.csr_matrix((2, 2), dtype=np.complex128)
    K = sp.eye(2, format="csr", dtype=np.complex128)
    Bphi = sp.csr_matrix(np.array([[1.0], [0.0]]))
    port_to_index = {1: 0}
    branch_count = 1
    has_loss = False


def test_period_doubled_problem_uses_mode_two_source() -> None:
    basis = period_doubled_basis(PumpBasis([0, 1, 2], "dense_real", 10.0))
    problem = build_period_doubled_problem(
        _Circuit(),
        JosephsonBranchLaw(np.array([1.0]), 1.0),
        basis,
        pump_current_a=0.1,
        pump_port=1,
    )
    assert problem.source_mode == 2
    assert problem.source_row == basis.modes.index(2)


def test_period_doubled_three_wave_mixing_idler_sideband_is_minus_two() -> None:
    assert period_doubled_idler_sideband() == -2


def test_period_doubled_basis_places_physical_pump_at_mode_two() -> None:
    basis = PumpBasis([0, 1, 2, 3], "dense_real", 10.0)

    doubled = period_doubled_basis(basis)

    assert doubled.omega_p == 5.0
    assert doubled.source_mode == 2
    assert doubled.modes == list(range(7))


def test_period_doubled_seed_maps_negative_hill_frequency_by_conjugation() -> None:
    basis = PumpBasis([0, 1, 2], "dense_real", 10.0)
    doubled = period_doubled_basis(basis)
    pump = np.zeros((3, 2), dtype=np.complex128)
    pump[1, 0] = 2.0 + 1.0j
    vector = np.array([1.0 + 2.0j, 3.0 + 4.0j])

    seed = build_period_doubled_seed(
        pump,
        basis,
        vector,
        [-1],
        doubled,
        perturbation_amplitude=1.0e-3,
    )

    # The period-1 mode-one pump is mode two in the half-pump lattice.
    assert seed[2, 0] == 2.0 + 1.0j
    # m=-1 gives half-pump index -1, mapped to positive mode 1 by conjugation.
    assert seed[1, 0] != 0.0j
    assert seed[1, 1] != 0.0j
