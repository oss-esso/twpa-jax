from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np


def test_native_gain_map_workflow_exports_compact_cube(tmp_path: Path) -> None:
    from twpa.core.hb_fft import HBProjectionConfig
    from twpa.core.params import NonlinearParams, SolverBackend, SolverConfig
    from twpa.nonlinear.pump_hb_ladder import PumpHBLadderConfig
    from twpa.workflows.gain_map import export_native_gain_map_artifacts, solve_native_gain_map
    from twpa.workflows.synthetic_benchmarks import (
        SyntheticLayoutKind,
        SyntheticLayoutSpec,
        build_synthetic_layout,
    )

    layout = build_synthetic_layout(
        SyntheticLayoutSpec(kind=SyntheticLayoutKind.UNIFORM, n_cells=1, length_m=2e-4)
    )
    config = PumpHBLadderConfig(
        n_pump_harmonics=1,
        projection=HBProjectionConfig(n_time_samples=16),
        solver=SolverConfig(backend=SolverBackend.NEWTON_KRYLOV, max_iter=6),
    )
    result = solve_native_gain_map(
        layout,
        NonlinearParams(I_star_A=1e-3),
        pump_frequencies_hz=[6e9],
        pump_current_ratios=[1e-7],
        signal_frequencies_hz=[3e9],
        pump_config=config,
    )
    paths = export_native_gain_map_artifacts(result, tmp_path)

    assert result.passed
    assert result.signal_gain_db.shape == (1, 1, 1)
    assert Path(paths["summary_json"]).exists()
    with np.load(paths["arrays_npz"]) as arrays:
        assert arrays["signal_gain_db"].shape == (1, 1, 1)


def test_recovery_gain_residual_uses_nominal_target_and_factories(monkeypatch) -> None:
    import jax.numpy as jnp

    from twpa.core.layout import make_uniform_layout
    from twpa.core.params import LineParams, NonlinearParams
    from twpa.inference.priors import make_default_twpa_scale_prior_set
    from twpa.inference.recovery import build_default_recovery_residual_factory
    from twpa.inference.synthetic import (
        SyntheticCombinedDataset,
        SyntheticGainDataset,
        SyntheticNoiseConfig,
    )
    from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig, PumpHBLadderConfig
    import twpa.workflows.calibration as calibration

    layout = make_uniform_layout(
        LineParams.from_z0_vp(
            length_m=2e-4,
            n_cells=1,
            z0_ohm=50.0,
            phase_velocity_m_per_s=1.2e8,
        )
    )
    nonlinear = NonlinearParams(I_star_A=1e-3)
    pump_drive = PumpDriveConfig.from_current_rms(pump_frequency_hz=6e9, current_rms_A=1e-7)
    pump_config = PumpHBLadderConfig(n_pump_harmonics=1)
    gain = SyntheticGainDataset(
        signal_frequency_hz=jnp.asarray([3e9]),
        idler_frequency_hz=jnp.asarray([9e9]),
        signal_gain_db_clean=jnp.asarray([0.0]),
        signal_gain_db_noisy=jnp.asarray([0.0]),
        idler_conversion_db_clean=None,
        idler_conversion_db_noisy=None,
        signal_labels=("signal_0",),
        idler_labels=("idler_0",),
        noise=SyntheticNoiseConfig(),
    )
    dataset = SyntheticCombinedDataset(gain=gain)
    captured = {}

    def fake_evaluate(target, params, *, gain_data):
        captured["target"] = target
        captured["params"] = dict(params)
        return SimpleNamespace(residual=jnp.asarray([0.0]))

    monkeypatch.setattr(calibration, "evaluate_calibration_objective", fake_evaluate)
    factory = build_default_recovery_residual_factory(
        layout=layout,
        nonlinear_params=nonlinear,
        pump_drive=pump_drive,
        pump_config=pump_config,
    )
    params = {"I_star_scale": 1.1, "pump_current_scale": 0.9}
    residual = factory(dataset, make_default_twpa_scale_prior_set())(params)

    assert np.asarray(residual).tolist() == [0.0]
    assert captured["params"] == params
    assert captured["target"].base_layout is layout
    assert captured["target"].base_nonlinear_params is nonlinear
    assert captured["target"].pump_drive is pump_drive
    assert captured["target"].target_plan_factory is not None
    assert captured["target"].sweep_config_factory is not None


def test_calibration_diagnostics_report_parameter_correlation() -> None:
    from twpa.core.layout import make_uniform_layout
    from twpa.core.params import LineParams
    from twpa.workflows.calibration import (
        CalibrationParameterSpec,
        CalibrationTarget,
        CalibrationVectorSpec,
        finite_difference_residual_jacobian,
    )

    layout = make_uniform_layout(
        LineParams.from_z0_vp(
            length_m=2e-4,
            n_cells=1,
            z0_ohm=50.0,
            phase_velocity_m_per_s=1.2e8,
        )
    )
    spec = CalibrationVectorSpec(
        (
            CalibrationParameterSpec("I_star_scale", 1.0, 0.5, 1.5),
            CalibrationParameterSpec("pump_current_scale", 1.0, 0.5, 1.5),
        )
    )
    target = CalibrationTarget(
        base_layout=layout,
        residual_hooks=(lambda params: [params["I_star_scale"] + params["pump_current_scale"]],),
    )
    diagnostics = finite_difference_residual_jacobian(target, spec, spec.initial_vector())

    assert diagnostics["parameter_names"] == ["I_star_scale", "pump_current_scale"]
    assert diagnostics["strongly_correlated_pairs"][0]["correlation"] == 1.0


def test_run_report_excludes_generated_reports_and_active_output(tmp_path: Path) -> None:
    from scripts.make_run_report import build_parser, discover_summary_jsons, resolve_config

    root = tmp_path / "outputs"
    root.mkdir()
    workflow = root / "workflow_summary.json"
    workflow.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    prior = root / "old_report" / "make_run_report_summary.json"
    prior.parent.mkdir()
    prior.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    current = root / "new_report" / "nested_summary.json"
    current.parent.mkdir()
    current.write_text(json.dumps({"status": "pass"}), encoding="utf-8")

    parser = build_parser()
    config = resolve_config(
        parser.parse_args(["--scan-root", str(root), "--output-dir", str(current.parent), "--overwrite"])
    )
    assert discover_summary_jsons(config) == [workflow]

    included = resolve_config(
        parser.parse_args(
            [
                "--scan-root",
                str(root),
                "--output-dir",
                str(current.parent),
                "--include-report-summaries",
                "--overwrite",
            ]
        )
    )
    assert discover_summary_jsons(included) == [prior, workflow]


def test_bridge_manifest_deduplicates_and_compacts_json(tmp_path: Path) -> None:
    from scripts.export_bridge_dataset import (
        build_parser,
        load_sources_into_bridge,
        materialize_sources,
        resolve_config,
    )

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"payload": "x" * 1000}), encoding="utf-8")
    parser = build_parser()
    config = resolve_config(
        parser.parse_args(
            [
                "--output-dir",
                str(tmp_path / "bridge"),
                "--extra-json",
                str(source),
                str(source),
                "--max-manifest-chars-per-source",
                "80",
            ]
        )
    )
    sources = materialize_sources(config)
    _, manifest = load_sources_into_bridge(sources, config)
    item = manifest["extra_json"]

    assert len(sources) == 1
    assert "json" not in item
    assert item["truncated"] is True
    assert len(item["json_preview"]) == 80
