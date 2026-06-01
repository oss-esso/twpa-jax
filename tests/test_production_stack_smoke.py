"""
Smoke tests for the production TWPA simulator stack.

These tests are intentionally small. They verify that the production-facing
modules import, construct objects, run tiny linear/nonlinear cases, and produce
serializable reports.

They are not accuracy benchmarks. They are regression sentinels for:

    - linear layout/cascade/dispersion stack
    - nonlinear one-node HB
    - distributed pump-HB stack
    - workflow wrappers
    - calibration parameter transforms
    - synthetic benchmark wrapper

Run:

    pytest tests/test_production_stack_smoke.py -q

For slower nonlinear smoke tests:

    pytest tests/test_production_stack_smoke.py -q --run-slow
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Pytest options
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow dense-HB smoke tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow dense-HB smoke test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_json_serializable(payload: Any) -> None:
    json.dumps(payload, default=str)


def tiny_nonlinear_params():
    from twpa.core.params import NonlinearParams

    return NonlinearParams(
        I_star_A=1.0e-3,
        beta_nl=1.0,
        quartic_coefficient=0.0,
        dc_bias_A=0.0,
    )


def tiny_uniform_layout(n_cells: int = 4):
    from twpa.workflows.synthetic_benchmarks import (
        SyntheticLayoutKind,
        SyntheticLayoutSpec,
        build_synthetic_layout,
    )

    spec = SyntheticLayoutSpec(
        kind=SyntheticLayoutKind.UNIFORM,
        n_cells=n_cells,
        length_m=2.0e-4,
        z0_ohm=50.0,
        phase_velocity_m_per_s=1.2e8,
        name=f"pytest_tiny_uniform_{n_cells}",
    )
    return build_synthetic_layout(spec)


def tiny_pump_plan():
    from twpa.core.frequency_plan import make_pump_only_plan

    return make_pump_only_plan(
        6.0e9,
        n_harmonics=1,
        include_negative=True,
        include_dc=False,
        sort="frequency",
    )


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

def test_production_modules_import() -> None:
    import twpa.nonlinear.one_node
    import twpa.nonlinear.distributed_hb
    import twpa.nonlinear.pump_hb_ladder
    import twpa.nonlinear.linearization
    import twpa.nonlinear.gain
    import twpa.workflows.industrial_100mm
    import twpa.workflows.calibration
    import twpa.workflows.synthetic_benchmarks

    assert twpa.nonlinear.one_node is not None
    assert twpa.nonlinear.distributed_hb is not None
    assert twpa.nonlinear.pump_hb_ladder is not None
    assert twpa.nonlinear.linearization is not None
    assert twpa.nonlinear.gain is not None
    assert twpa.workflows.industrial_100mm is not None
    assert twpa.workflows.calibration is not None
    assert twpa.workflows.synthetic_benchmarks is not None


def test_public_exports_are_present() -> None:
    import twpa.nonlinear.pump_hb_ladder as pump
    import twpa.nonlinear.gain as gain
    import twpa.workflows.industrial_100mm as industrial
    import twpa.workflows.calibration as calibration

    for name in [
        "PumpDriveConfig",
        "PumpHBLadderConfig",
        "solve_pump_hb_ladder",
        "dbm_to_watt",
        "watt_to_dbm",
    ]:
        assert hasattr(pump, name), name

    for name in [
        "GainSolveConfig",
        "GainSweepConfig",
        "solve_gain_point",
        "solve_gain_sweep_from_pump",
    ]:
        assert hasattr(gain, name), name

    for name in [
        "IndustrialLayoutSpec",
        "build_industrial_layout",
        "run_industrial_100mm_workflow",
    ]:
        assert hasattr(industrial, name), name

    for name in [
        "CalibrationParameterSpec",
        "CalibrationVectorSpec",
        "calibrate",
    ]:
        assert hasattr(calibration, name), name


# ---------------------------------------------------------------------------
# Scalar / config tests
# ---------------------------------------------------------------------------

def test_pump_power_current_conversions_round_trip() -> None:
    from twpa.nonlinear.pump_hb_ladder import (
        available_power_from_norton_current_rms,
        dbm_to_norton_current_rms,
        dbm_to_watt,
        norton_current_rms_to_dbm,
        watt_to_dbm,
    )

    p_dbm = -80.0
    p_w = dbm_to_watt(p_dbm)
    assert float(p_w) > 0.0

    p_dbm_back = watt_to_dbm(p_w)
    assert float(p_dbm_back) == pytest.approx(p_dbm, abs=1e-10)

    i_rms = dbm_to_norton_current_rms(p_dbm, source_impedance_ohm=50.0)
    assert float(i_rms) > 0.0

    p_from_i = available_power_from_norton_current_rms(i_rms, source_impedance_ohm=50.0)
    assert float(p_from_i) == pytest.approx(float(p_w), rel=1e-12)

    p_dbm_from_i = norton_current_rms_to_dbm(i_rms, source_impedance_ohm=50.0)
    assert float(p_dbm_from_i) == pytest.approx(p_dbm, abs=1e-10)


def test_pump_drive_config_serializable() -> None:
    from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig

    drive = PumpDriveConfig.from_available_power_dbm(
        pump_frequency_hz=8.0e9,
        power_dbm=-90.0,
        source_impedance_ohm=50.0,
    )

    assert drive.current_rms_A > 0.0
    assert drive.available_power_dbm == pytest.approx(-90.0)
    assert_json_serializable(drive.to_dict())


def test_calibration_parameter_transform_round_trip() -> None:
    from twpa.workflows.calibration import (
        CalibrationParameterSpec,
        CalibrationVectorSpec,
        ParameterTransform,
    )

    spec = CalibrationParameterSpec(
        name="L_scale",
        initial=1.0,
        lower=0.5,
        upper=2.0,
        transform=ParameterTransform.LOG,
    )

    encoded = spec.encode(1.25)
    decoded = spec.decode(encoded)
    assert decoded == pytest.approx(1.25)

    vector_spec = CalibrationVectorSpec((spec,))
    vec = vector_spec.initial_vector()
    params = vector_spec.decode_vector(vec)

    assert params["L_scale"] == pytest.approx(1.0)
    assert_json_serializable(vector_spec.to_dict())


# ---------------------------------------------------------------------------
# Layout / linear stack tests
# ---------------------------------------------------------------------------

def test_industrial_layout_builds_100mm_20000_cells() -> None:
    from twpa.workflows.industrial_100mm import (
        IndustrialLayoutSpec,
        build_industrial_layout,
    )

    spec = IndustrialLayoutSpec(
        length_m=0.100,
        n_cells=20_000,
        z0_ohm=50.0,
        phase_velocity_m_per_s=1.2e8,
        name="pytest_100mm_20000",
    )
    layout = build_industrial_layout(spec)

    assert layout.n_cells == 20_000
    assert float(layout.total_length_m) == pytest.approx(0.100)
    assert_json_serializable(layout.summary())


def test_tiny_linear_stage_runs() -> None:
    from twpa.workflows.industrial_100mm import (
        IndustrialLinearStageConfig,
        run_linear_stage,
    )

    layout = tiny_uniform_layout(n_cells=8)

    cfg = IndustrialLinearStageConfig(
        frequency_min_hz=1.0e9,
        frequency_max_hz=4.0e9,
        n_frequency_points=9,
    )

    result = run_linear_stage(layout, cfg)

    assert result.frequency_hz.shape == (9,)
    assert result.scan.s.shape[0] == 9
    assert result.status.value in {"pass", "fail"}
    assert_json_serializable(result.to_dict())


def test_synthetic_layout_builds_all_default_specs() -> None:
    from twpa.workflows.synthetic_benchmarks import (
        build_synthetic_layout,
        default_synthetic_layout_specs,
    )

    specs = default_synthetic_layout_specs()
    assert specs

    for spec in specs:
        layout = build_synthetic_layout(spec)
        assert layout.n_cells == spec.n_cells
        assert float(layout.total_length_m) == pytest.approx(spec.length_m)
        assert_json_serializable(layout.summary())


# ---------------------------------------------------------------------------
# Nonlinear one-node / distributed tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_one_node_self_checks_run() -> None:
    from twpa.nonlinear.one_node import run_one_node_self_checks

    plan = tiny_pump_plan()
    report = run_one_node_self_checks(plan)

    assert "passed" in report
    assert_json_serializable(report)


@pytest.mark.slow
def test_distributed_hb_linear_initial_guess_shapes() -> None:
    from twpa.nonlinear.distributed_hb import (
        DistributedHBConfig,
        make_input_pump_current_injection,
        make_kinetic_model_from_layout,
        make_distributed_linear_initial_guess,
    )

    layout = tiny_uniform_layout(n_cells=4)
    plan = tiny_pump_plan()
    cfg = DistributedHBConfig()
    nonlinear = tiny_nonlinear_params()
    model = make_kinetic_model_from_layout(layout, nonlinear)

    injection = make_input_pump_current_injection(
        plan,
        layout,
        cfg,
        pump_label="pump",
        pump_current_rms_A=1e-10 + 0j,
    )

    x0 = make_distributed_linear_initial_guess(plan, layout, cfg, injection)

    assert x0.node_voltage_coeffs_V.shape == (plan.n_tones, layout.n_cells + 1)
    assert x0.branch_current_coeffs_A.shape == (plan.n_tones, layout.n_cells)
    assert model.L0_H.shape == (layout.n_cells,)


@pytest.mark.slow
def test_pump_hb_tiny_solve_runs() -> None:
    from twpa.nonlinear.distributed_hb import DistributedHBConfig
    from twpa.nonlinear.pump_hb_ladder import (
        PumpDriveConfig,
        PumpHBLadderConfig,
        solve_pump_hb_ladder,
    )
    from twpa.solvers.hb_solver import DenseNewtonConfig

    layout = tiny_uniform_layout(n_cells=3)
    nonlinear = tiny_nonlinear_params()

    drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=6.0e9,
        current_rms_A=1e-10,
    )

    cfg = PumpHBLadderConfig(
        n_pump_harmonics=1,
        distributed=DistributedHBConfig(
            source_conductance_S=1.0 / 50.0,
            load_conductance_S=1.0 / 50.0,
        ),
        solver=DenseNewtonConfig(
            max_iter=12,
            abs_tol=1e-8,
            rel_tol=1e-8,
            fail_on_nonconvergence=False,
            verbose=False,
        ),
    )

    result = solve_pump_hb_ladder(
        layout,
        nonlinear,
        drive=drive,
        pump_config=cfg,
    )

    assert result.state.node_voltage_coeffs_V.shape[1] == layout.n_cells + 1
    assert result.state.branch_current_coeffs_A.shape[1] == layout.n_cells
    assert result.status.value in {"converged", "failed"}
    assert_json_serializable(result.to_dict())


# ---------------------------------------------------------------------------
# Linearization tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_build_linearization_from_tiny_pump_result() -> None:
    from twpa.nonlinear.distributed_hb import DistributedHBConfig
    from twpa.nonlinear.linearization import (
        SmallSignalLinearizationConfig,
        build_linearization_from_pump_result,
    )
    from twpa.nonlinear.pump_hb_ladder import (
        PumpDriveConfig,
        PumpHBLadderConfig,
        solve_pump_hb_ladder,
    )
    from twpa.solvers.hb_solver import DenseNewtonConfig

    layout = tiny_uniform_layout(n_cells=2)
    nonlinear = tiny_nonlinear_params()

    drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=6.0e9,
        current_rms_A=1e-10,
    )

    pump_cfg = PumpHBLadderConfig(
        n_pump_harmonics=1,
        distributed=DistributedHBConfig(),
        solver=DenseNewtonConfig(
            max_iter=10,
            abs_tol=1e-8,
            rel_tol=1e-8,
            fail_on_nonconvergence=False,
            verbose=False,
        ),
    )

    pump = solve_pump_hb_ladder(
        layout,
        nonlinear,
        drive=drive,
        pump_config=pump_cfg,
    )

    lin = build_linearization_from_pump_result(
        pump.distributed_result,
        config=SmallSignalLinearizationConfig(),
    )

    assert lin.unknown_size > 0
    assert lin.residual_size > 0
    assert lin.operating_state.node_voltage_coeffs_V.shape == pump.state.node_voltage_coeffs_V.shape
    assert_json_serializable(lin.to_dict(include_matrix=False))


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------

def test_linear_only_industrial_workflow_runs_tiny_layout(tmp_path: Path) -> None:
    from twpa.workflows.industrial_100mm import (
        Industrial100mmWorkflowConfig,
        IndustrialLayoutSpec,
        IndustrialLinearStageConfig,
        IndustrialRunMode,
        run_industrial_100mm_workflow,
    )

    cfg = Industrial100mmWorkflowConfig(
        mode=IndustrialRunMode.LINEAR_ONLY,
        layout=IndustrialLayoutSpec(
            length_m=1.0e-3,
            n_cells=16,
            z0_ohm=50.0,
            phase_velocity_m_per_s=1.2e8,
            name="pytest_workflow_linear",
        ),
        linear=IndustrialLinearStageConfig(
            frequency_min_hz=1.0e9,
            frequency_max_hz=4.0e9,
            n_frequency_points=9,
        ),
        output_dir=str(tmp_path),
        save_artifacts=True,
        name="pytest_linear_workflow",
    )

    result = run_industrial_100mm_workflow(cfg)

    assert result.layout.n_cells == 16
    assert result.linear.frequency_hz.shape == (9,)
    assert result.status.value in {"pass", "fail", "partial"}
    assert "summary_json" in result.artifact_paths
    assert Path(result.artifact_paths["summary_json"]).exists()
    assert_json_serializable(result.to_dict())


def test_synthetic_fast_linear_benchmarks_run(tmp_path: Path) -> None:
    from twpa.workflows.synthetic_benchmarks import (
        make_fast_linear_synthetic_config,
        run_synthetic_benchmarks,
    )

    cfg = make_fast_linear_synthetic_config(
        output_dir=str(tmp_path),
        save_artifacts=True,
    )

    result = run_synthetic_benchmarks(cfg)

    assert len(result.layout_results) > 0
    assert result.status.value in {"pass", "fail", "skip", "error"}
    assert "summary_json" in result.artifact_paths
    assert Path(result.artifact_paths["summary_json"]).exists()
    assert_json_serializable(result.to_dict())


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

def test_linear_calibration_objective_zero_for_identical_synthetic_data() -> None:
    from twpa.workflows.calibration import (
        CalibrationTarget,
        SParameterCalibrationData,
        evaluate_calibration_objective,
        sparameter_residual_vector,
    )
    from twpa.linear.cascade import run_linear_scan

    layout = tiny_uniform_layout(n_cells=6)
    frequency_hz = jnp.linspace(1.0e9, 4.0e9, 7, dtype=jnp.float64)

    scan = run_linear_scan(frequency_hz, layout)

    data = SParameterCalibrationData(
        frequency_hz=frequency_hz,
        s=scan.s,
        s21_db=scan.s21_db,
        weight_complex=1.0,
        weight_s21_db=1.0,
    )

    target = CalibrationTarget(base_layout=layout)

    params = {
        "L_scale": 1.0,
        "C_scale": 1.0,
        "C_stub_scale": 1.0,
        "R_scale": 1.0,
        "G_scale": 1.0,
    }

    residual = sparameter_residual_vector(target, params, data)
    evaluation = evaluate_calibration_objective(
        target,
        params,
        sparameter_data=data,
    )

    assert float(jnp.linalg.norm(residual)) < 1e-8
    assert evaluation.loss < 1e-16
    assert_json_serializable(evaluation.to_dict())


def test_calibration_vector_spec_decodes_disabled_parameters() -> None:
    from twpa.workflows.calibration import (
        CalibrationParameterSpec,
        CalibrationVectorSpec,
        ParameterTransform,
    )

    spec = CalibrationVectorSpec(
        (
            CalibrationParameterSpec(
                name="L_scale",
                initial=1.0,
                lower=0.5,
                upper=2.0,
                transform=ParameterTransform.LOG,
                enabled=True,
            ),
            CalibrationParameterSpec(
                name="C_scale",
                initial=1.0,
                lower=0.5,
                upper=2.0,
                transform=ParameterTransform.LOG,
                enabled=False,
            ),
        )
    )

    vec = spec.initial_vector()
    decoded = spec.decode_vector(vec)

    assert tuple(spec.enabled_names) == ("L_scale",)
    assert decoded["L_scale"] == pytest.approx(1.0)
    assert decoded["C_scale"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Script import smoke tests
# ---------------------------------------------------------------------------

def test_scripts_import_without_running() -> None:
    import importlib

    modules = [
        "scripts.run_linear_validation",
        "scripts.run_pump_hb",
        "scripts.run_gain_map",
        "scripts.run_industrial_100mm",
        "scripts.run_calibration",
        "scripts.run_synthetic_benchmarks",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert hasattr(module, "build_parser")
        assert hasattr(module, "main")