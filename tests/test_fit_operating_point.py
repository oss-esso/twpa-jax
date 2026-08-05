from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from twpa_solver.builders.jc_doc import build_jpa
from twpa_solver.core import CircuitMatrices, load_circuit, save_circuit

from scripts.fit_model_operating_point import (
    ROI_WEIGHT_FLOOR_DB,
    _ripple_weights,
    build_pump_engine,
    fit_operating_point,
    reference_signal_current_a,
    solve_finite_signal_gain_db,
    solve_pump_point_robust,
    sweep_model_g0_db,
)


def _save_jpa_circuit(tmp_path: Path) -> Path:
    builder, metadata = build_jpa()
    arrays = builder.assemble()
    circuit = CircuitMatrices(
        C=arrays["C"], G=arrays["G"], K=arrays["K"], Bphi=arrays["Bphi"],
        Ic=arrays["Ic"], port_to_index=arrays["ports"], metadata=metadata,
    )
    circuit_dir = tmp_path / "jpa_circuit"
    save_circuit(circuit, circuit_dir)
    return circuit_dir


# JPA design frequency and its known-compressing operating current, same
# point tests/test_run_compression_cli.py's _jpa_gain_args uses ("the exp20
# jpa operating point, which really does compress"). The finite-signal
# reference current used here is chosen far below that (see
# _TINY_REFERENCE_CURRENT_A) so it sits deep in the near-linear regime.
_JPA_TRUE_PUMP_FREQ_GHZ = 4.75001
_JPA_TRUE_PUMP_CURRENT_A = 1.13e-08
_JPA_PORTS = dict(pump_port=1, source_port=1, out_port=1)
_JPA_ENGINE_KWARGS = dict(
    # mode_count=10 matches tests/test_run_compression_cli.py's proven-working
    # _jpa_gain_args default; a smaller count (e.g. 4) hit a degenerate
    # zero-frequency tone combination in build_sideband_matched_basis at this
    # near-degenerate JPA operating point (signal ~ pump).
    pump_mode_policy="positive_odd_jc", pump_mode_count=10, pump_harmonics=10, pump_nt=40,
)
_TINY_REFERENCE_CURRENT_A = 1.0e-10


def _jpa_engine(tmp_path: Path, circuit_dir: Path):
    return build_pump_engine(
        circuit_dir, **_JPA_PORTS, **_JPA_ENGINE_KWARGS, outdir=tmp_path / "engine_scratch",
    )


def test_ripple_weights_floors_negative_and_scales_with_gain() -> None:
    target = np.array([-10.0, -1.0, 0.0, 5.0, 20.0])
    weight = _ripple_weights(target, floor=ROI_WEIGHT_FLOOR_DB)
    assert weight[0] == pytest.approx(ROI_WEIGHT_FLOOR_DB)  # deep negative -> floored
    assert weight[1] == pytest.approx(ROI_WEIGHT_FLOOR_DB)  # small negative -> floored
    assert weight[2] == pytest.approx(ROI_WEIGHT_FLOOR_DB)  # zero -> floored
    assert weight[3] == pytest.approx(5.0)
    assert weight[4] == pytest.approx(20.0)
    assert np.all(np.diff(weight) >= 0.0)  # monotone non-decreasing with gain


def test_reference_signal_current_a_scales_with_attenuation() -> None:
    """Higher frequency -> more loss_B1 attenuation -> smaller on-chip current,
    at fixed instrument power -- and the current must be well below the
    device's compressing current (this is meant to be a near-linear
    reference point, not a saturation point)."""
    i_low_freq = reference_signal_current_a(-60.0, 4.0)
    i_high_freq = reference_signal_current_a(-60.0, 12.0)
    assert i_low_freq > i_high_freq > 0.0

    i_higher_power = reference_signal_current_a(-40.0, 7.0)
    i_lower_power = reference_signal_current_a(-60.0, 7.0)
    assert i_higher_power > i_lower_power > 0.0


def test_solve_pump_point_robust_warm_start_reaches_same_state_as_cold(tmp_path: Path) -> None:
    """A warm-started solve (single Newton, then power-substep bridge if
    needed) must reach the same physical solution a cold solve reaches on
    its own -- warm-starting is a convergence-speed device, not a different
    physics path.
    """
    circuit_dir = _save_jpa_circuit(tmp_path)
    engine = _jpa_engine(tmp_path, circuit_dir)
    pass_dir = tmp_path / "pass"

    current_a = _JPA_TRUE_PUMP_CURRENT_A
    current_b = current_a * 1.05  # a nearby current, not identical

    state_a, _basis = solve_pump_point_robust(
        engine, pass_dir, 1, _JPA_TRUE_PUMP_FREQ_GHZ, current_a,
        warm_state=None, warm_current_a=None,
    )
    assert state_a is not None

    warm_state, _basis = solve_pump_point_robust(
        engine, pass_dir, 2, _JPA_TRUE_PUMP_FREQ_GHZ, current_b,
        warm_state=state_a, warm_current_a=current_a,
    )
    cold_state, _basis = solve_pump_point_robust(
        engine, pass_dir, 3, _JPA_TRUE_PUMP_FREQ_GHZ, current_b,
        warm_state=None, warm_current_a=None,
    )
    assert warm_state is not None
    assert cold_state is not None
    np.testing.assert_allclose(warm_state, cold_state, rtol=1e-5, atol=1e-12)


def test_solve_finite_signal_gain_handles_unreachable_pump_current_gracefully(
    tmp_path: Path,
) -> None:
    """An absurd pump current must not crash the sweep -- either the pump
    solve fails outright (None) or the resulting operating point is
    numerically degenerate and the signal point comes back None.
    """
    circuit_dir = _save_jpa_circuit(tmp_path)
    engine = _jpa_engine(tmp_path, circuit_dir)
    pass_dir = tmp_path / "pass"

    X, pump_basis = solve_pump_point_robust(
        engine, pass_dir, 1, _JPA_TRUE_PUMP_FREQ_GHZ, 5.0,
        warm_state=None, warm_current_a=None,
    )
    if X is None:
        return  # pump itself already failed -- the graceful case
    omega_p = 2.0 * math.pi * _JPA_TRUE_PUMP_FREQ_GHZ * 1e9
    circuit = load_circuit(circuit_dir)
    gain = solve_finite_signal_gain_db(
        circuit,
        X, pump_basis, 5.0, omega_p, _JPA_TRUE_PUMP_FREQ_GHZ, _TINY_REFERENCE_CURRENT_A,
        pump_port=_JPA_PORTS["pump_port"], source_port=_JPA_PORTS["source_port"],
        out_port=_JPA_PORTS["out_port"], z0_ohm=50.0, multitone_sidebands=2,
    )
    assert gain is None or math.isfinite(gain)


def test_fit_operating_point_recovers_synthetic_target_within_grid_step(
    tmp_path: Path,
) -> None:
    """Self-recovery: build the target from the model at a KNOWN (f_p, I_p)
    using the SAME finite-signal machinery fit_operating_point uses
    internally, then confirm the coarse+fine grid search finds it back
    within the fine grid's own step size, and that the objective is finite
    at the optimum.
    """
    circuit_dir = _save_jpa_circuit(tmp_path)
    circuit = load_circuit(circuit_dir)
    engine = _jpa_engine(tmp_path, circuit_dir)
    pass_dir = tmp_path / "truth_pass"

    fit_freq_bounds_ghz = (4.70, 4.80)
    signal_freq_step_mhz = 10.0
    freq_grid = np.arange(
        fit_freq_bounds_ghz[0], fit_freq_bounds_ghz[1] + 1e-9,
        signal_freq_step_mhz / 1000.0,
    )
    reference_currents = np.full(freq_grid.shape, _TINY_REFERENCE_CURRENT_A)

    truth, truth_state = sweep_model_g0_db(
        circuit, engine, pass_dir, 0, _JPA_TRUE_PUMP_FREQ_GHZ, _JPA_TRUE_PUMP_CURRENT_A,
        freq_grid, reference_currents,
        pump_port=_JPA_PORTS["pump_port"], source_port=_JPA_PORTS["source_port"],
        out_port=_JPA_PORTS["out_port"], multitone_sidebands=2, z0_ohm=50.0,
    )
    assert truth is not None
    assert truth_state is not None
    assert np.isfinite(truth).sum() >= 8, "synthetic target must have enough valid points to fit"

    # smallest_instrument_dbm chosen so reference_signal_current_a reproduces
    # _TINY_REFERENCE_CURRENT_A at these frequencies (within loss_B1's small
    # variation across a 0.1 GHz window, negligible here).
    from twpa_solver.loss import signal_line_loss_model
    att_db = float(signal_line_loss_model().attenuation_db(_JPA_TRUE_PUMP_FREQ_GHZ))
    onchip_dbm = 10.0 * math.log10(
        (_TINY_REFERENCE_CURRENT_A**2 * 50.0 / 8.0) / 1.0e-3
    )
    smallest_instrument_dbm = onchip_dbm + att_db

    result = fit_operating_point(
        circuit, circuit_dir, freq_grid, truth, smallest_instrument_dbm,
        freq_bounds_ghz=(4.74, 4.76),
        current_bounds_a=(0.8e-8, 1.6e-8),
        fit_freq_bounds_ghz=fit_freq_bounds_ghz,
        coarse_freq_points=3, coarse_current_points=3, fine_points=3,
        signal_freq_step_mhz=signal_freq_step_mhz,
        pump_port=_JPA_PORTS["pump_port"], source_port=_JPA_PORTS["source_port"],
        out_port=_JPA_PORTS["out_port"], pump_mode_policy=_JPA_ENGINE_KWARGS["pump_mode_policy"],
        pump_mode_count=_JPA_ENGINE_KWARGS["pump_mode_count"],
        pump_harmonics=_JPA_ENGINE_KWARGS["pump_harmonics"], pump_nt=_JPA_ENGINE_KWARGS["pump_nt"],
        multitone_sidebands=2, z0_ohm=50.0, outdir=tmp_path / "fit_out", progress=False,
    )

    fine_freq_step = float(result["fine_freq_ghz"][1] - result["fine_freq_ghz"][0])
    fine_logi_step = float(result["fine_log_current"][1] - result["fine_log_current"][0])

    assert np.isfinite(result["fine_surface"]).any(), "objective must be finite somewhere on the fine grid"
    assert abs(result["pump_freq_ghz"] - _JPA_TRUE_PUMP_FREQ_GHZ) <= fine_freq_step + 1e-9
    assert abs(
        math.log10(result["pump_current_a"]) - math.log10(_JPA_TRUE_PUMP_CURRENT_A)
    ) <= fine_logi_step + 1e-9

    min_fine_cost = float(np.min(result["fine_surface"]))
    assert min_fine_cost < 1.0  # dB^2 -- a tight match, not just "not infinite"
