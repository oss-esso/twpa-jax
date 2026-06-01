from __future__ import annotations

import warnings

import jax.numpy as jnp


def test_cli_parser_exposes_expected_commands() -> None:
    from twpa.cli import COMMAND_MODULES, build_parser

    parser = build_parser()
    help_text = parser.format_help()

    for command in COMMAND_MODULES:
        assert command in help_text


def test_harmonics_selected_plan_wrapper_warns() -> None:
    import twpa.core.harmonics as hm

    assert not hasattr(hm, "FrequencyPlan")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = hm.make_frequency_plan(10.0e9, 6.0e9, (-3, -1, 1, 3))

    assert any(item.category is DeprecationWarning for item in caught)
    assert plan.idler_frequency_hz == 14.0e9


def test_selected_harmonic_one_node_wrapper_warns() -> None:
    from twpa.nonlinear.one_node import solve_one_node_hb

    drive = jnp.asarray([1.0e-9 + 0.0j, 1.0e-9 + 0.0j])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solve_one_node_hb(
            drive,
            jnp.asarray([-1, 1]),
            C_F=100e-15,
            L0_H=1e-9,
            I_star_A=1e-3,
            beta=0.0,
            omega0_rad_s=2.0 * jnp.pi * 5.0e9,
            n_time=64,
        )

    assert any(item.category is DeprecationWarning for item in caught)
    assert result.success


def test_selected_harmonic_distributed_wrapper_warns() -> None:
    from twpa.nonlinear.distributed_hb import solve_distributed_hb

    drive = jnp.zeros((2, 3), dtype=jnp.complex128).at[:, 0].set(1.0e-9 + 0.0j)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solve_distributed_hb(
            drive,
            jnp.asarray([-1, 1]),
            L_series_H=1e-12,
            C_shunt_F=2e-15,
            I_star_A=1e-3,
            beta=0.0,
            omega0_rad_s=2.0 * jnp.pi * 5.0e9,
            n_time=64,
        )

    assert any(item.category is DeprecationWarning for item in caught)
    assert result.success


def test_cell_local_block_jacobi_factory_builds_ready_preconditioner() -> None:
    from twpa.core.frequency_plan import make_pump_only_plan
    from twpa.core.params import NonlinearParams
    from twpa.nonlinear.distributed_hb import (
        DistributedHBConfig,
        make_cell_local_block_jacobi_preconditioner_factory,
        make_distributed_linear_initial_guess,
        make_input_pump_current_injection,
        make_kinetic_model_from_layout,
    )
    from twpa.core.hb_fft import HBProjectionConfig, make_projection_grid_from_plan
    from twpa.workflows.synthetic_benchmarks import (
        SyntheticLayoutKind,
        SyntheticLayoutSpec,
        build_synthetic_layout,
    )

    layout = build_synthetic_layout(
        SyntheticLayoutSpec(
            kind=SyntheticLayoutKind.UNIFORM,
            n_cells=2,
            length_m=4.0e-4,
            z0_ohm=50.0,
            phase_velocity_m_per_s=1.2e8,
            name="pytest_block_jacobi_factory",
        )
    )
    plan = make_pump_only_plan(
        pump_frequency_hz=6.0e9,
        n_harmonics=1,
        include_negative=True,
        include_dc=False,
    )
    config = DistributedHBConfig()
    ki_model = make_kinetic_model_from_layout(layout, NonlinearParams(I_star_A=1.0e-3))
    injection = make_input_pump_current_injection(
        plan,
        layout,
        config,
        pump_label="pump",
        pump_current_rms_A=1.0e-10 + 0.0j,
    )
    x0 = make_distributed_linear_initial_guess(plan, layout, config, injection)
    projection_config = HBProjectionConfig(n_time_samples=64)
    projection_grid = make_projection_grid_from_plan(
        plan,
        fundamental_frequency_hz=plan.reference_pump_hz,
        config=projection_config,
    )
    factory = make_cell_local_block_jacobi_preconditioner_factory(
        plan,
        layout,
        config,
        ki_model,
        projection_grid=projection_grid,
        projection_config=projection_config,
    )
    preconditioner = factory(jnp.zeros((2 * plan.n_tones * (2 * layout.n_cells + 1),)), jnp.zeros((2 * plan.n_tones * (2 * layout.n_cells + 1),)))

    assert preconditioner.ready
    assert preconditioner.metadata["source"] == "make_cell_local_block_jacobi_preconditioner_factory"
