from __future__ import annotations

from pathlib import Path


def _config(tmp_path: Path, **overrides: object):
    from scripts.compression_sweep import CompressionSweepConfig

    base = dict(
        n_cells=1,
        length_mm=0.2,
        z0_ohm=50.0,
        phase_velocity_m_per_s=1.2e8,
        layout_csv=None,
        pump_npz=None,
        pump_frequency_ghz=6.0,
        pump_current_ratio=1.0e-7,
        pump_phase_rad=0.0,
        i_star_a=1.0e-3,
        l0_scale=1.0,
        nonlinear_beta=1.0,
        signal_f_min_ghz=2.5,
        signal_f_max_ghz=3.5,
        n_signal=3,
        target_signal_frequency_ghz=3.0,
        signal_current_rms_a_values=(1.0e-12,),
        signal_power_dbm_values=(-120.0,),
        compression_db=1.0,
        reference_mode="first",
        harmonic_orders=(-1, 1),
        n_time=32,
        max_iter=8,
        tolerance=1e-10,
        damping=1.0,
        continuation_steps=2,
        pump_solver_mode="auto",
        pump_numerical_backend="newton_krylov",
        gain_solver_mode="auto",
        point_solver_mode="native_finite_signal",
        require_package_gain_solver=False,
        allow_partial_pump_fallback=True,
        layout_kind="uniform",
        include_resonators=False,
        disorder_std=0.0,
        seed=1234,
        output_dir=str(tmp_path),
        name="pytest_compression_native",
        quick=False,
        fail_fast=False,
        keep_per_point_plots=False,
        keep_per_point_checkpoints=False,
        export_profile_csv=False,
        make_summary_plots=False,
        full_pump_script="scripts/full_pump_hb_100mm.py",
        gain_script="scripts/gain_from_pumped_solution.py",
        python_executable="python",
    )
    base.update(overrides)
    return CompressionSweepConfig(**base)


def test_native_point_solver_auto_decision(tmp_path: Path) -> None:
    from scripts.compression_sweep import _native_point_solver_allowed

    cfg = _config(tmp_path, point_solver_mode="auto", n_cells=32)
    assert _native_point_solver_allowed(cfg) == (True, "auto native finite-signal point solver")

    cfg_no_target = _config(tmp_path, point_solver_mode="auto", target_signal_frequency_ghz=None)
    assert _native_point_solver_allowed(cfg_no_target) == (
        True,
        "auto native finite-signal point solver",
    )

    cfg_large = _config(tmp_path, point_solver_mode="auto", n_cells=512)
    assert _native_point_solver_allowed(cfg_large) == (
        True,
        "auto native finite-signal point solver",
    )

    cfg_legacy = _config(tmp_path, point_solver_mode="gain_script")
    allowed, reason = _native_point_solver_allowed(cfg_legacy)
    assert not allowed
    assert "gain_script" in reason


def test_native_finite_signal_compression_point_runs(tmp_path: Path) -> None:
    from scripts.compression_sweep import RunStatus, run_native_finite_signal_point

    cfg = _config(tmp_path)
    point = run_native_finite_signal_point(
        config=cfg,
        point_index=0,
        signal_current_rms_a=1.0e-12,
        signal_power_dbm=-120.0,
        output_dir=tmp_path,
    )

    assert point.status == RunStatus.PASS
    assert point.gain_status == RunStatus.PASS
    assert point.gain_returncode == 0
    assert point.target_signal_frequency_ghz == 3.0
    assert point.target_gain_db is not None
    assert point.gain_arrays_npz is not None
    assert point.gain_summary_path is not None
    assert Path(point.gain_arrays_npz).exists()
    assert Path(point.gain_summary_path).exists()
    assert point.gain_summary["driver"] == "native_finite_signal_hb"
    assert point.gain_summary["converged"] is True


def test_native_wideband_compression_writes_gain_matrix(tmp_path: Path) -> None:
    import numpy as np

    from scripts.compression_sweep import RunStatus, run_native_wideband_compression

    cfg = _config(
        tmp_path,
        target_signal_frequency_ghz=None,
        n_signal=2,
        signal_current_rms_a_values=(1.0e-12,),
        signal_power_dbm_values=(-120.0,),
    )
    summary = run_native_wideband_compression(config=cfg, output_dir=tmp_path)

    assert summary["status"] in {RunStatus.PASS.value, RunStatus.PARTIAL.value}
    assert summary["driver"] == "native_finite_signal_hb_wideband"
    with np.load(summary["artifact_paths"]["arrays_npz"]) as data:
        assert data["signal_gain_db"].shape == (2, 1)
        assert data["residual_norm"].shape == (2, 1)
