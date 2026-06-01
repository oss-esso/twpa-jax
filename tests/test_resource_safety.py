from types import SimpleNamespace

import jax.numpy as jnp
import pytest


def test_dense_pump_guard_refuses_large_reference_problem_before_allocation() -> None:
    from twpa.core.params import NonlinearParams
    from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig, solve_pump_hb_ladder

    layout = SimpleNamespace(n_cells=2000)
    drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=10e9,
        current_rms_A=1e-6,
    )

    with pytest.raises(RuntimeError, match="refused unsafe problem size"):
        solve_pump_hb_ladder(layout, NonlinearParams(), drive=drive)


def test_dense_gain_guard_refuses_large_reference_problem_before_allocation() -> None:
    from twpa.nonlinear.linearization import (
        LinearizationBackend,
        SmallSignalLinearizationConfig,
        solve_linearized_small_signal,
    )

    linearization = SimpleNamespace(
        plan=SimpleNamespace(n_tones=100),
        layout=SimpleNamespace(n_cells=100),
        config=SmallSignalLinearizationConfig(
            backend=LinearizationBackend.DENSE_JACOBIAN,
        ),
    )

    with pytest.raises(RuntimeError, match="refused unsafe problem size"):
        solve_linearized_small_signal(linearization, source=None)


def test_dense_reference_resource_estimator_matches_unknown_formula() -> None:
    from scripts.estimate_hb_resources import estimate

    result = estimate(n_cells=8, n_tones=6)
    assert result["real_unknowns"] == 2 * 6 * (2 * 8 + 1)
    assert result["one_float64_jacobian_mib"] > 0.0


def test_dense_finite_signal_guard_refuses_large_reference_problem_before_allocation() -> None:
    from twpa.core.params import NonlinearParams
    from twpa.nonlinear.finite_signal_hb import (
        SignalDriveConfig,
        solve_finite_signal_hb,
    )
    from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig

    with pytest.raises(RuntimeError, match="refused unsafe problem size"):
        solve_finite_signal_hb(
            SimpleNamespace(n_cells=100),
            NonlinearParams(),
            pump_drive=PumpDriveConfig.from_current_rms(
                pump_frequency_hz=6.0e9,
                current_rms_A=1.0e-10,
            ),
            signal_drive=SignalDriveConfig(
                signal_frequency_hz=3.0e9,
                current_rms_A=1.0e-12,
            ),
        )


def test_hb_dispatcher_runs_matrix_free_newton_krylov() -> None:
    from twpa.core.params import SolverBackend, SolverConfig
    from twpa.solvers.hb_solver import solve_hb

    result = solve_hb(
        lambda x: x - jnp.asarray([1.0, -2.0]),
        jnp.asarray([0.0, 0.0]),
        config=SolverConfig(
            backend=SolverBackend.NEWTON_KRYLOV,
            max_iter=4,
            verbose=False,
        ),
    )

    assert result.converged
    assert result.report.config["backend"] == "newton_krylov"
    assert result.report.metadata["matrix_free"] is True
    assert jnp.allclose(result.x, jnp.asarray([1.0, -2.0]))


@pytest.mark.slow
def test_tiny_pump_newton_krylov_matches_dense_reference() -> None:
    from twpa.core.params import NonlinearParams, SolverBackend, SolverConfig
    from twpa.nonlinear.distributed_hb import DistributedHBConfig
    from twpa.nonlinear.pump_hb_ladder import (
        PumpDriveConfig,
        PumpHBLadderConfig,
        solve_pump_hb_ladder,
    )
    from twpa.solvers.hb_solver import DenseNewtonConfig
    from twpa.workflows.synthetic_benchmarks import (
        SyntheticLayoutKind,
        SyntheticLayoutSpec,
        build_synthetic_layout,
    )

    layout = build_synthetic_layout(
        SyntheticLayoutSpec(
            kind=SyntheticLayoutKind.UNIFORM,
            n_cells=1,
            length_m=2.0e-4,
            z0_ohm=50.0,
            phase_velocity_m_per_s=1.2e8,
            name="pytest_tiny_pump_equivalence",
        )
    )
    nonlinear = NonlinearParams(I_star_A=1.0e-3)
    drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=6.0e9,
        current_rms_A=1.0e-10,
    )
    common = dict(
        n_pump_harmonics=1,
        distributed=DistributedHBConfig(),
        max_dense_real_unknowns=64,
    )

    dense = solve_pump_hb_ladder(
        layout,
        nonlinear,
        drive=drive,
        pump_config=PumpHBLadderConfig(
            **common,
            solver=DenseNewtonConfig(
                max_iter=8,
                abs_tol=1e-12,
                rel_tol=1e-12,
                verbose=False,
            ),
        ),
    )
    matrix_free = solve_pump_hb_ladder(
        layout,
        nonlinear,
        drive=drive,
        pump_config=PumpHBLadderConfig(
            **common,
            solver=SolverConfig(
                backend=SolverBackend.NEWTON_KRYLOV,
                max_iter=8,
                abs_tol=1e-12,
                rel_tol=1e-12,
                verbose=False,
            ),
        ),
    )

    assert dense.converged
    assert matrix_free.converged
    assert matrix_free.solver_result.report.metadata["matrix_free"] is True
    assert matrix_free.solver_result.report.config["solver"]["use_preconditioner"] is True
    assert (
        matrix_free.solver_result.report.metadata["preconditioner_summary"]["source_history"][
            0
        ]
        == "make_cell_local_block_jacobi_preconditioner_factory"
    )
    assert jnp.allclose(
        matrix_free.state.node_voltage_coeffs_V,
        dense.state.node_voltage_coeffs_V,
        rtol=1e-8,
        atol=1e-14,
    )
    assert jnp.allclose(
        matrix_free.state.branch_current_coeffs_A,
        dense.state.branch_current_coeffs_A,
        rtol=1e-8,
        atol=1e-14,
    )


@pytest.mark.slow
def test_tiny_gain_matrix_free_matches_dense_reference() -> None:
    from twpa.core.params import NonlinearParams
    from twpa.nonlinear.distributed_hb import DistributedHBConfig
    from twpa.nonlinear.linearization import (
        LinearizationBackend,
        SmallSignalLinearizationConfig,
        SmallSignalSource,
        build_linearization_from_pump_result,
        solve_linearized_small_signal,
    )
    from twpa.nonlinear.pump_hb_ladder import (
        PumpDriveConfig,
        PumpHBLadderConfig,
        solve_pump_hb_ladder,
    )
    from twpa.solvers.hb_solver import DenseNewtonConfig
    from twpa.workflows.synthetic_benchmarks import (
        SyntheticLayoutKind,
        SyntheticLayoutSpec,
        build_synthetic_layout,
    )

    layout = build_synthetic_layout(
        SyntheticLayoutSpec(
            kind=SyntheticLayoutKind.UNIFORM,
            n_cells=1,
            length_m=2.0e-4,
            z0_ohm=50.0,
            phase_velocity_m_per_s=1.2e8,
            name="pytest_tiny_gain_equivalence",
        )
    )
    pump = solve_pump_hb_ladder(
        layout,
        NonlinearParams(I_star_A=1.0e-3),
        drive=PumpDriveConfig.from_current_rms(
            pump_frequency_hz=6.0e9,
            current_rms_A=1.0e-10,
        ),
        pump_config=PumpHBLadderConfig(
            n_pump_harmonics=1,
            distributed=DistributedHBConfig(),
            solver=DenseNewtonConfig(max_iter=8, abs_tol=1e-12, rel_tol=1e-12),
        ),
    )
    source = SmallSignalSource.current_phasor_at_node(
        plan=pump.frequency_plan,
        layout=layout,
        node=0,
        label="pump",
        rms_current_A=1.0e-12,
    )
    dense = solve_linearized_small_signal(
        build_linearization_from_pump_result(
            pump.distributed_result,
            config=SmallSignalLinearizationConfig(
                backend=LinearizationBackend.DENSE_JACOBIAN,
            ),
        ),
        source,
    )
    matrix_free = solve_linearized_small_signal(
        build_linearization_from_pump_result(
            pump.distributed_result,
            config=SmallSignalLinearizationConfig(
                backend=LinearizationBackend.JAX_LINEARIZE,
            ),
        ),
        source,
    )

    assert dense.converged
    assert matrix_free.converged
    assert matrix_free.metadata["matrix_free"] is True
    assert jnp.allclose(
        matrix_free.node_voltage_coeffs_V,
        dense.node_voltage_coeffs_V,
        rtol=1e-6,
        atol=1e-14,
    )
    assert jnp.allclose(
        matrix_free.branch_current_coeffs_A,
        dense.branch_current_coeffs_A,
        rtol=1e-6,
        atol=1e-14,
    )


@pytest.mark.slow
def test_tiny_finite_signal_matrix_free_hb_converges() -> None:
    from twpa.core.params import NonlinearParams, SolverBackend, SolverConfig
    from twpa.nonlinear.finite_signal_hb import (
        FiniteSignalHBConfig,
        SignalDriveConfig,
        solve_finite_signal_hb,
    )
    from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig
    from twpa.workflows.synthetic_benchmarks import (
        SyntheticLayoutKind,
        SyntheticLayoutSpec,
        build_synthetic_layout,
    )

    layout = build_synthetic_layout(
        SyntheticLayoutSpec(
            kind=SyntheticLayoutKind.UNIFORM,
            n_cells=1,
            length_m=2.0e-4,
            z0_ohm=50.0,
            phase_velocity_m_per_s=1.2e8,
            name="pytest_tiny_finite_signal",
        )
    )
    result = solve_finite_signal_hb(
        layout,
        NonlinearParams(I_star_A=1.0e-3),
        pump_drive=PumpDriveConfig.from_current_rms(
            pump_frequency_hz=6.0e9,
            current_rms_A=1.0e-10,
        ),
        signal_drive=SignalDriveConfig(
            signal_frequency_hz=3.0e9,
            current_rms_A=1.0e-12,
        ),
        finite_config=FiniteSignalHBConfig(
            n_pump_harmonics=1,
            include_second_order_sidebands=False,
            solver=SolverConfig(
                backend=SolverBackend.NEWTON_KRYLOV,
                max_iter=8,
                abs_tol=1e-10,
                rel_tol=1e-10,
                verbose=False,
            ),
        ),
    )

    assert result.converged
    assert result.frequency_plan.n_tones == 6
    assert result.distributed_result.residual.norm < 1e-12
