from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.h1_transient_branch_transfer import (
    audit_circuit,
    classify_state,
    classify_td_result,
    _strobe_summary,
    checkpoint_stroboscopic_diagnostics,
    load_hb_initial,
    build_system,
    node_velocity,
    parse_args,
    max_abs_phi_envelope_classification,
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


def test_h1_period_two_uses_late_d2_closure_not_early_transient_maximum() -> None:
    periods = np.arange(31, dtype=float)
    distances = {
        "d1": np.full(30, 2e-3),
        "d2": np.array([2e-2, *([2e-4] * 28)]),
        "d3": np.full(28, 2e-3),
    }
    summary = _strobe_summary(periods, distances)

    assert summary["tail_d2_max"] == 2e-4
    assert classify_state(summary, 0.0, True) == "PERIOD_2"


def test_h1_checkpoint_diagnostics_are_recorded_at_specified_hold_lengths() -> None:
    periods = np.arange(441, dtype=float)
    distances = {f"d{n}": np.full(441 - n, 1e-4) for n in (1, 2, 3, 4, 6, 8)}
    strobe = _strobe_summary(periods, distances)

    checkpoints = checkpoint_stroboscopic_diagnostics(strobe)

    assert [item["hold_periods"] for item in checkpoints] == [40, 90, 140, 250, 440]
    assert checkpoints[-1]["stroboscopic"]["periods"][-1] == 440.0


def test_decay_aware_adapter_does_not_promote_broadband_hold() -> None:
    assert classify_td_result({
        "classification": "BROADBAND_OR_CHAOTIC",
        "decay_aware": {
            "class": "UNRESOLVED_SLOW_RELAXATION",
            "tau_periods": 1690.0,
        },
    }) == "UNRESOLVED_SLOW_RELAXATION"


def test_decay_aware_adapter_preserves_persistent_broadband_transition() -> None:
    assert classify_td_result({
        "classification": "BROADBAND_OR_CHAOTIC",
        "decay_aware": {"class": "PERSISTENT_NONPERIODIC"},
    }) == "BROADBAND_OR_CHAOTIC"


def test_h1_compact_storage_limits_are_explicit() -> None:
    args = parse_args([
        "--compact-output",
        "--compact-sample-count", "64",
        "--compact-history-states", "128",
    ])

    assert args.compact_output
    assert args.compact_sample_count == 64
    assert args.compact_history_states == 128


def test_h1_adaptive_state_relative_atol_is_an_explicit_option() -> None:
    args = parse_args(["--method", "BDF", "--atol-mode", "state_relative"])

    assert args.atol_mode == "state_relative"
    assert args.atol_floor == pytest.approx(1e-12)


def test_h1_l_stable_alias_selects_bdf() -> None:
    args = parse_args(["--l-stable"])
    assert args.method == "BDF"


def test_h1_node_velocity_reconstructs_algebraic_port() -> None:
    system = build_system(ROOT / "designs" / "ipm_2c_fixed", 7.9, 4)
    q = np.zeros(system.n)
    p = np.zeros(system.n)
    source = system.source(0.0, 0.0, 1e-6, 2.0 * np.pi)

    expected = system.algebraic_velocity(q, p[system.differential], source)[0]

    assert node_velocity(system, q, p, 4576, source) == pytest.approx(expected)


def test_max_abs_phi_envelope_is_the_primary_growth_discriminant() -> None:
    periods = np.arange(20, dtype=float)
    result = max_abs_phi_envelope_classification(
        {
            "theta": periods * 2.0 * np.pi,
            "max_abs_phi": 0.2 + 2.0e-5 * periods,
        },
        ramp_periods=5,
    )

    assert result["class"] == "GROWING_MAX_ABS_PHI"
    assert result["slope_per_period"] == pytest.approx(2.0e-5)


def test_max_abs_phi_envelope_accepts_bounded_slope() -> None:
    periods = np.arange(20, dtype=float)
    result = max_abs_phi_envelope_classification(
        {
            "theta": periods * 2.0 * np.pi,
            "max_abs_phi": 0.2 + 5.0e-6 * periods,
        },
        ramp_periods=5,
    )

    assert result["class"] == "NON_GROWING_MAX_ABS_PHI"


def test_h1_hb_checkpoint_reconstructs_a_phase_zero_state() -> None:
    system = build_system(ROOT / "designs" / "ipm_2c_fixed", 7.9, 4)
    checkpoint = (
        ROOT / "outputs" / "g1_current_79" / "pass" / "points"
        / "point_0012_p_m19p6842dbm_fp_7p9ghz" / "pump"
    )
    x0, w0, current, report = load_hb_initial(
        checkpoint, system.circuit, system.omega
    )

    assert x0.shape == (system.n,)
    assert w0.shape == (system.n,)
    assert current > 0.0
    assert report["final_status"] == "VALID_CONVERGED"
