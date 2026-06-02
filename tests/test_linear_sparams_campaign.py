from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_linear_sparams_campaign import (
    campaign_paths,
    compute_2port_metrics,
    make_linear_sparams_config,
    run_campaign,
)


def test_make_linear_sparams_config() -> None:
    cfg = make_linear_sparams_config(index=0, z_line_ohm=50.0, n_frequency=101)

    assert cfg["schema_version"] == "0.1.0"
    assert cfg["simulation_type"] == "linear_sparams"
    assert cfg["circuit_template"] == "matched_transmission_line"
    assert cfg["parameters"]["z_ref_ohm"] == 50.0
    assert cfg["parameters"]["z_line_ohm"] == 50.0
    assert cfg["axes"]["frequency_hz"]["points"] == 101


def test_campaign_paths(tmp_path: Path) -> None:
    paths = campaign_paths(tmp_path)

    assert paths["configs"] == tmp_path / "configs"
    assert paths["runs"] == tmp_path / "runs"
    assert paths["registry"] == tmp_path / "runs.csv"
    assert paths["summary"] == tmp_path / "campaign_summary.json"


def test_actual_linear_sparams_campaign_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    summary = run_campaign(
        z_lines_ohm=[45.0, 50.0, 55.0],
        harmonia_root=harmonia_root,
        campaign_dir=tmp_path / "campaign",
        force=True,
        timeout_s=120.0,
        n_frequency=21,
    )

    assert summary["n_requested"] == 3
    assert summary["n_launched"] == 3
    assert summary["registry"]["by_status"] == {"PASS": 3}
    assert summary["registry"]["by_simulation_type"] == {"linear_sparams": 3}

    runs = {run["z_line_ohm"]: run for run in summary["runs"]}

    m45 = runs[45.0]["metrics"]
    m50 = runs[50.0]["metrics"]
    m55 = runs[55.0]["metrics"]

    assert m45["all_arrays_finite"]
    assert m50["all_arrays_finite"]
    assert m55["all_arrays_finite"]

    assert m50["max_abs_s11"] < 1e-10
    assert m50["max_abs_s22"] < 1e-10

    assert m45["max_abs_s11"] > m50["max_abs_s11"]
    assert m55["max_abs_s11"] > m50["max_abs_s11"]

    assert m45["reciprocal_error_max_abs"] < 1e-10
    assert m50["reciprocal_error_max_abs"] < 1e-10
    assert m55["reciprocal_error_max_abs"] < 1e-10

    assert m45["passivity_max_singular_value"] <= 1.0 + 1e-10
    assert m50["passivity_max_singular_value"] <= 1.0 + 1e-10
    assert m55["passivity_max_singular_value"] <= 1.0 + 1e-10

    assert abs(m50["gain_db_max"]) < 1e-9


def test_compute_2port_metrics_on_existing_linear_run_if_available() -> None:
    run_dir = Path(r"D:\Projects\Thesis\outputs\julia_engine_linear\matched_tl_001")

    if not (run_dir / "simulation.h5").exists():
        pytest.skip("Local linear S-parameter run not available.")

    metrics = compute_2port_metrics(run_dir)

    assert metrics["all_arrays_finite"]
    assert metrics["s_shape"][1:] == [2, 2]
    assert metrics["reciprocal_error_max_abs"] < 1e-10
    assert metrics["passivity_max_singular_value"] <= 1.0 + 1e-10