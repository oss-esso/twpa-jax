import numpy as np

from twpa_solver.multitone.basis import (
    ToneIndex,
    build_half_pump_basis,
    build_lattice_basis,
)
from twpa_solver.multitone.seed import promote_pump_solution
from twpa_solver.pump.basis import PumpBasis


def test_half_pump_basis_places_physical_pump_at_h2():
    basis = build_half_pump_basis([1, 2, 3], 2, 10.0, 1.0)
    assert basis.pump_tone == ToneIndex(2, 0)
    assert basis.pump_tone.omega(basis.omega_p, basis.delta) == 10.0
    assert basis.signal_tone.omega(basis.omega_p, basis.delta) == 4.0
    assert basis.idler_tone.omega(basis.omega_p, basis.delta) == 6.0


def test_half_pump_pump_modes_are_even_and_torus_round_trips():
    basis = build_half_pump_basis([1, 2, 3], 2, 10.0, 1.0)
    assert all(tone.h % 2 == 0 for tone in basis.tones if tone.q == 0 and tone != ToneIndex(1, 0))
    rng = np.random.default_rng(4)
    values = rng.normal(size=(basis.n_tones, 2)) + 1j * rng.normal(size=(basis.n_tones, 2))
    np.testing.assert_allclose(basis.project(basis.synthesize(values)), values, rtol=1e-14, atol=1e-14)


def test_promote_pump_solution_scales_by_pump_tone():
    basis = build_half_pump_basis([1, 2, 3], 2, 10.0, 1.0)
    pump_basis = PumpBasis([1, 2, 3], "dense_real", 20.0)
    source = np.arange(6, dtype=float).reshape(3, 2).astype(complex)
    promoted = promote_pump_solution(source, pump_basis, basis)
    for mode, row in zip((1, 2, 3), source):
        np.testing.assert_array_equal(promoted[basis.index_of(ToneIndex(2 * mode, 0))], row)


def test_default_lattice_keeps_default_pump_tone():
    basis = build_lattice_basis([1, 2, 3], 2, 10.0, 1.0, 50.0)
    assert basis.pump_tone == ToneIndex(1, 0)
