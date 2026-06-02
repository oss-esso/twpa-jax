from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_harmonia_jtl_linear_campaign import (
    campaign_paths,
    compute_jtl_linear_metrics,
    make_harmonia_jtl_linear_config,
    run_campaign,
)


def test_make_harmonia_jtl_linear_config() -> None:
    cfg = make_harmonia_jtl_linear_config(
        index=0,
        Lj_H=1.0e-9,
        n_cell=4,
        n_frequency=7,
    )

    assert cfg["schema_version"] == "0.1.0"
    assert cfg["simulation_type"] == "harmonia_jtl_linear_jc_smoke"
    assert cfg["circuit_template"] == "circuit_ir_jtl_chain_linear_jc"
    assert cfg["parameters"]["Lj_H"] == 1.0e-9
    assert cfg["parameters"]["N_cell"] == 4
    assert cfg["parameters"]["port_impedance_ohm"] == 50.0
    assert cfg["axes"]["frequency_hz"]["points"] == 7


def test_campaign_paths(tmp_path: Path) -> None:
    paths = campaign_paths(tmp_path)

    assert paths["configs"] == tmp_path / "configs"
    assert paths["runs"] == tmp_path / "runs"
    assert paths["registry"] == tmp_path / "runs.csv"
    assert paths["summary"] == tmp_path / "campaign_summary.json"


def test_compute_metrics_on_existing_harmonia_jtl_linear_if_available() -> None:
    run_dir = Path(r"D:\Projects\Thesis\outputs\harmonia_jtl_linear_jc_smoke\run_001")

    if not (run_dir / "simulation.h5").exists():
        pytest.skip("Local Harmonia JTL linear smoke output not available.")

    metrics = compute_jtl_linear_metrics(run_dir)

    assert metrics["frequency_points"] == 11
    assert metrics["s_shape"] == [11, 2, 2]
    assert metrics["all_arrays_finite"]
    assert metrics["gain_db_max"] >= metrics["gain_db_min"]


def test_actual_harmonia_jtl_linear_campaign_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    summary = run_campaign(
        lj_values_h=[0.8e-9, 1.0e-9, 1.2e-9],
        harmonia_root=harmonia_root,
        campaign_dir=tmp_path / "campaign",
        force=True,
        timeout_s=240.0,
        n_cell=4,
        n_frequency=7,
    )

    assert summary["n_requested"] == 3
    assert summary["n_launched"] == 3
    assert summary["registry"]["by_status"] == {"PASS": 3}
    assert summary["registry"]["by_simulation_type"] == {
        "harmonia_jtl_linear_jc_smoke": 3
    }

    for run in summary["runs"]:
        assert run["ok"]
        assert run["status"] == "PASS"
        assert run["metrics"] is not None
        assert run["metrics"]["all_arrays_finite"]
        assert run["metrics"]["frequency_points"] == 7
        assert run["metrics"]["s_shape"] == [7, 2, 2]

    paths = campaign_paths(tmp_path / "campaign")
    assert paths["registry"].exists()
    assert paths["summary"].exists()