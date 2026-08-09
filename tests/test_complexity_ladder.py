from __future__ import annotations

import numpy as np

from twpa_solver.builders.complexity_ladder import (
    LadderParameters,
    build_linear_fixture,
    build_single_jj,
    build_uniform_jtl,
    build_ipm_single_nonlinear_section,
)
from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.pump.validation import validate_production_hb_state


def test_rung_zero_is_linear_and_matches_lj_relation() -> None:
    params = LadderParameters()
    circuit = build_linear_fixture(params)

    assert circuit.branch_count == 0
    assert circuit.node_count == 2
    assert np.isclose(params.ic_a, circuit.phi0 / params.lj_h)
    assert circuit.metadata["nonlinear"] is False


def test_single_jj_retains_nonlinear_branch_and_cj() -> None:
    params = LadderParameters()
    circuit = build_single_jj(params)

    assert circuit.branch_count == 1
    assert np.isclose(circuit.Ic[0], params.ic_a)
    assert circuit.metadata["nonlinear"] is True
    assert circuit.C.nnz > 0


def test_linear_fixture_matches_single_jj_zero_phase_tangent() -> None:
    linear = build_linear_fixture()
    nonlinear = build_single_jj()
    tangent = nonlinear.Bphi @ np.diag(nonlinear.Ic / nonlinear.phi0) @ nonlinear.Bphi.T

    assert np.allclose(linear.K.toarray(), np.asarray(tangent))


def test_uniform_jtl_has_requested_number_of_junctions_and_metadata() -> None:
    circuit = build_uniform_jtl(8)

    assert circuit.branch_count == 8
    assert circuit.metadata["n_cells"] == 8
    assert circuit.metadata["topology"] == "uniform_nonlinear_jtl"
    assert circuit.port_to_index[1] != circuit.port_to_index[2]


def test_ipm_single_section_matches_one_production_section() -> None:
    circuit = build_ipm_single_nonlinear_section(418)

    assert circuit.branch_count == 418
    assert circuit.node_count == 419
    assert circuit.metadata["topology"] == "IPM_SINGLE_NONLINEAR_SECTION"
    assert circuit.metadata["n_cells"] == 418


def test_production_hb_validation_rejects_wrong_state_shape() -> None:
    circuit = build_uniform_jtl(8)
    validation = validate_production_hb_state(
        circuit,
        make_branch_law(circuit),
        frequency_hz=7.9e9,
        pump_port=1,
        pump_current_a=0.0,
        modes=np.arange(1, 3),
        state=np.zeros((2, circuit.node_count - 1), dtype=complex),
        nt=8,
    )

    assert validation["hb_solver_family"] == "production_pump"
    assert not validation["checkpoint_validated"]
    assert not validation["state_shape_ok"]
