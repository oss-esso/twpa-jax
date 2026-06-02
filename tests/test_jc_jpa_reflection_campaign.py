from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_jc_jpa_reflection_campaign import (
    campaign_paths,
    compute_one_port_reflection_metrics,
    make_jc_jpa_reflection_config,
    run_campaign,
)


def test_make_jc_jpa_reflection_config() -> None:
    cfg = make_jc_jpa_reflection_config(
        index=0,
        pump_current_a=2.0e-9,
        n_frequency=7,
    )

    assert cfg["schema_version"] == "0.1.0"
    assert cfg["simulation_type"] == "jc_jpa_reflection_smoke"
    assert cfg["circuit_template"] == "one_port_jpa_reflection"
    assert cfg["parameters"]["pump_current_a"] == 2.0e-9
    assert cfg["axes"]["frequency_hz"]["points"] == 7


def test_campaign_paths(tmp_path: Path) -> None:
    paths = campaign_paths(tmp_path)

    assert paths["configs"] == tmp_path / "configs"
    assert paths["runs"] == tmp_path / "runs"
    assert paths["registry"] == tmp_path / "runs.csv"
    assert paths["summary"] == tmp_path / "campaign_summary.json"


def test_compute_metrics_on_existing_jc_smoke_if_available() -> None:
    run_dir = Path(r"D:\Projects\Thesis\outputs\jc_jpa_reflection_smoke\run_001")

    if not (run_dir / "simulation.h5").exists():
        pytest.skip("Local JC JPA reflection smoke output not available.")

    metrics = compute_one_port_reflection_metrics(run_dir)

    assert metrics["frequency_points"] == 11
    assert metrics["s_shape"] == [11, 1, 1]
    assert metrics["all_arrays_finite"]
    assert metrics["reflection_db_max"] >= metrics["reflection_db_min"]


def test_actual_jc_jpa_reflection_campaign_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    summary = run_campaign(
        pump_currents_a=[0.0, 2.0e-9, 5.65e-9],
        harmonia_root=harmonia_root,
        campaign_dir=tmp_path / "campaign",
        force=True,
        timeout_s=180.0,
        n_frequency=7,
    )

    assert summary["n_requested"] == 3
    assert summary["n_launched"] == 3
    assert summary["registry"]["by_status"] == {"PASS": 3}
    assert summary["registry"]["by_simulation_type"] == {"jc_jpa_reflection_smoke": 3}

    for run in summary["runs"]:
        assert run["ok"]
        assert run["status"] == "PASS"
        assert run["metrics"] is not None
        assert run["metrics"]["all_arrays_finite"]
        assert run["metrics"]["frequency_points"] == 7
        assert run["metrics"]["s_shape"] == [7, 1, 1]

    paths = campaign_paths(tmp_path / "campaign")
    assert paths["registry"].exists()
    assert paths["summary"].exists()