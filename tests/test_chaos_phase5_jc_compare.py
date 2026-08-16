from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.chaos.run_guarcello_jc_phase5 import (
    derive_device_spec,
    integrate_jc_banded_numba,
    load_jc_device,
    derive_time_budget,
    integrate_jc_banded,
    _integrate_jc_compiled,
    power_labels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_jtwpa_descriptor_identifies_plain_jtwpa_and_resonator_profile() -> None:
    spec = derive_device_spec(ROOT / "outputs/jc_doc_python_designs/jc_jtwpa")

    assert spec.name == "jc_jtwpa"
    assert spec.branch_count == 2047
    assert spec.ic_min_a == spec.ic_max_a == 3.4e-6
    assert spec.has_parallel_geometric_inductor is False
    assert spec.resonator_period == 4


def test_fqjtwpa_descriptor_retains_nonuniform_junction_profile() -> None:
    spec = derive_device_spec(ROOT / "outputs/jc_doc_python_designs/jc_fqjtwpa")

    assert spec.name == "jc_fqjtwpa"
    assert spec.branch_count == 1999
    assert spec.ic_min_a < spec.ic_max_a
    assert spec.resonator_period == 8
    assert spec.profile_is_nonuniform is True
    assert spec.pump_ghz == 7.9
    assert spec.signal_ghz == 7.4


def test_time_budget_reports_period_and_retained_window() -> None:
    spec = derive_device_spec(ROOT / "outputs/jc_doc_python_designs/jc_jtwpa")
    budget = derive_time_budget(spec, dt_norm=0.01, tmax_norm=20_000.0)

    assert budget.steps_per_pump_period >= 100.0
    assert budget.retained_pump_periods >= 300.0


def test_guarcello_jc_device_is_natural_order_banded() -> None:
    device = load_jc_device(ROOT / "outputs/jc_doc_python_designs/jc_jtwpa")

    assert device.n_nodes == 2560
    assert device.natural_bandwidth == 2
    assert device.ic_uniform is True


def test_compiled_kernel_is_available_for_the_phase5_time_loop() -> None:
    assert integrate_jc_banded_numba is not None


def test_compiled_kernel_matches_interpreted_short_trajectory() -> None:
    spec = derive_device_spec(ROOT / "outputs/jc_doc_python_designs/jc_jtwpa")
    device = load_jc_device(ROOT / "outputs/jc_doc_python_designs/jc_jtwpa")
    settings = dict(
        pump_current_a=1.0e-6,
        pump_hz=spec.pump_ghz * 1e9,
        signal_current_a=1.0e-9,
        signal_hz=spec.signal_ghz * 1e9,
        dt_s=0.01 / spec.omega_plasma,
        n_steps=2000,
        record_stride=20,
        initial_q=None,
    )
    interpreted = integrate_jc_banded(device, **settings)
    compiled = _integrate_jc_compiled(device, **settings)
    np.testing.assert_allclose(compiled[4], interpreted[4], rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(compiled[2], interpreted[2], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(compiled[0], interpreted[0], rtol=0.0, atol=0.0)


def test_phase_c_2c_selects_measured_rcm_order_and_pump_port() -> None:
    spec = derive_device_spec(ROOT / "designs/ipm_2c_fixed")
    device = load_jc_device(ROOT / "designs/ipm_2c_fixed")

    assert spec.natural_bandwidth == 4578
    assert spec.rcm_bandwidth == 5
    assert spec.selected_ordering == "rcm"
    assert spec.selected_bandwidth == 5
    assert spec.pump_port == 4
    assert spec.signal_source_port == 1
    assert spec.signal_output_port == 2
    assert device.pump_node != device.signal_node
    assert device.pump_output_node != device.pump_node


def test_phase_c_rf_squid_build_has_profile_and_measured_bandwidth() -> None:
    spec = derive_device_spec(ROOT / "designs/rf_squid_2393_3wm.yaml")
    device = load_jc_device(ROOT / spec.circuit_dir)

    assert spec.node_count == 7180
    assert spec.branch_count == 2393
    assert spec.natural_bandwidth == 2
    assert spec.rcm_bandwidth == 2
    assert device.Cg.size == 2393
    assert np.unique(device.Cg).size == 3
    assert spec.profile_is_nonuniform is True
    assert spec.has_parallel_geometric_inductor is True


def test_phase_c_power_labels_preserve_named_conventions() -> None:
    labels = power_labels(1.0e-6, 7.9e9)

    assert labels["power_convention"] == "legacy_traveling_wave"
    assert labels["loss_model"] == "pump_line_loss_model_A10"
    assert labels["pump_power_instrument_dbm"] > labels["pump_power_onchip_dbm"]
