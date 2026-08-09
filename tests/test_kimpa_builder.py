import numpy as np
import pytest

from twpa_solver.builders.kimpa import KIMPA_FIXTURES, add_transmission_line_ladder, build_kimpa
from twpa_solver.builders.jc_doc import CircuitBuilder
from twpa_solver.core.kinetic import KineticInductorBranchLaw
from twpa_solver.core.kinetic import kinetic_dc_branch_flux
from twpa_solver.core import save_circuit
from twpa_solver.signal.passive import passive_s_matrix


def test_ladder_stamps_symmetric_pi_totals():
    builder = CircuitBuilder("line")
    values = add_transmission_line_ladder(builder, "tl", "a", "b", 80.0, "quarter", 8e9, 10)
    assert values["L_line_h"] == 2.5e-9
    assert values["C_line_f"] == 1.0 / (4 * 8e9 * 80.0)
    assert sum(float(cap[3]) for cap in builder.capacitors) == pytest.approx(values["C_line_f"])
    assert len(builder.linear_inductors) == 10


def test_kinetic_branch_is_not_stamped_into_k_and_lg_is():
    circuit = build_kimpa("kimpa_fabricated_nominal", cells=(2, 2, 2))
    assert circuit.branch_law.metadata["type"] == "kinetic_inductor"
    assert isinstance(circuit.branch_law, KineticInductorBranchLaw)
    assert circuit.branch_law.differential_inductance(np.zeros((1, 1)))[0, 0] == circuit.metadata["Lk_h"]
    # The geometric inductor is a real K stamp; the kinetic branch is represented only in Bphi.
    assert circuit.K.nnz > 0
    assert circuit.Bphi.shape[1] == 1


def test_three_fixtures_have_documented_kinetic_scales():
    circuits = [build_kimpa(name, cells=(2, 2, 2)) for name in KIMPA_FIXTURES]
    assert [c.metadata["Lk_h"] for c in circuits] == [0.999e-9, 56.0**2 * 330e-15 - 200e-12, 0.633e-9, 835e-12]


def test_hung_fixture_has_explicit_scales_and_round_trip_values():
    circuit = build_kimpa("kimpa_hung_2025", cells=(1, 1, 1))
    assert circuit.metadata["Lk_h"] == 835e-12
    assert circuit.metadata["Istar2_a"] == 3.25e-3
    assert circuit.metadata["Istar4_a"] == 1.70e-3
    assert circuit.branch_law.istar2_a[0] == 3.25e-3
    assert circuit.branch_law.istar4_a[0] == 1.70e-3


def test_dc_helper_inverts_kinetic_branch_and_zeros_other_branches():
    circuit = build_kimpa("kimpa_measured_seed", cells=(2, 2, 2))
    flux = kinetic_dc_branch_flux(circuit, 550e-6)
    assert np.count_nonzero(flux) == 1
    np.testing.assert_allclose(circuit.branch_law.current(flux[None, :])[0], [550e-6], rtol=1e-14)


def test_mixed_builder_returns_composite_law():
    builder = CircuitBuilder("mixed")
    builder.josephson_inductor("jj", "a", "0", 1e-9)
    builder.kinetic_inductor("ki", "b", "0", 1e-9, 1e-3)
    assembled = builder.assemble()
    assert assembled["branch_law"].metadata["type"] == "composite"
    assert assembled["Bphi"].shape[1] == 2


def test_legacy_and_explicit_kinetic_inputs_and_invalid_fields():
    legacy = CircuitBuilder("legacy")
    legacy.kinetic_inductor("ki", "a", "0", 835e-12, 1.15e-3)
    assert legacy.assemble()["branch_law"].istar2_a[0] == pytest.approx(3.25e-3)
    explicit = CircuitBuilder("explicit")
    explicit.kinetic_inductor("ki", "a", "0", 835e-12, 1.15e-3, "hung_2025", 3.25e-3, 1.70e-3)
    law = explicit.assemble()["branch_law"]
    assert law.istar2_a[0] == 3.25e-3 and law.istar4_a[0] == 1.70e-3
    with pytest.raises(TypeError):
        explicit.kinetic_inductor("bad", "b", "0", 835e-12, 1.15e-3, Istar22=3.25e-3)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        explicit.kinetic_inductor("bad_lk", "c", "0", 0.0, 1.15e-3)
    with pytest.raises(ValueError):
        explicit.kinetic_inductor("bad_i2", "d", "0", 835e-12, 1.15e-3, "hung_2025", 0.0, 1.70e-3)


def test_one_port_passive_readout_accepts_single_rhs(tmp_path):
    circuit = build_kimpa("kimpa_measured_seed", cells=(1, 1, 1))
    save_circuit(circuit, tmp_path)
    result = passive_s_matrix(
        tmp_path, np.array([8.0e9]), ports=(1,),
        dc_branch_flux=kinetic_dc_branch_flux(circuit, 550e-6),
    )
    assert result.shape == (1, 1, 1)
    assert np.isfinite(result[0, 0, 0].real)
