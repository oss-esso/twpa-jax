from __future__ import annotations

from pathlib import Path

from scripts.h1_transient_branch_transfer import (
    audit_circuit,
    classify_state,
    load_hb_initial,
    build_system,
)


ROOT = Path(__file__).resolve().parents[1]


def test_h1_audit_detects_the_index_one_output_port() -> None:
    audit = audit_circuit(ROOT / "designs" / "ipm_2c_fixed")

    assert audit.algebraic_nodes == [4576]
    assert audit.differential_nodes == 6135
    assert audit.c_factorable
    assert audit.algebraic_g_factorable


def test_h1_classifier_separates_periodic_and_nonperiodic_states() -> None:
    periodic = {"tail_max": 1e-4}
    nonperiodic = {"tail_max": 1e-2}

    assert classify_state(periodic, 0.0, True) == "PERIOD_1"
    assert classify_state(nonperiodic, 0.0, True) == "BROADBAND_OR_CHAOTIC"
    assert classify_state(periodic, 1e9, True, 0.2) == "RUNNING_PHASE"
    assert classify_state(periodic, 0.0, False) == "TRANSIENT_NUMERICAL_FAILURE"


def test_h1_classifier_recognizes_period_two_from_stroboscopic_distance() -> None:
    period_two = {"tail_max": 2e-3, "tail_d2_max": 2e-4, "tail_d3_max": 2e-3}

    assert classify_state(period_two, 0.0, True) == "PERIOD_2"


def test_h1_hb_checkpoint_reconstructs_a_phase_zero_state() -> None:
    system = build_system(ROOT / "designs" / "ipm_2c_fixed", 7.9, 4)
    checkpoint = (
        ROOT / "g1_current_79" / "pass" / "points"
        / "point_0012_p_m19p6842dbm_fp_7p9ghz" / "pump"
    )
    x0, w0, current, report = load_hb_initial(
        checkpoint, system.circuit, system.omega
    )

    assert x0.shape == (system.n,)
    assert w0.shape == (system.n,)
    assert current > 0.0
    assert report["final_status"] == "VALID_CONVERGED"
