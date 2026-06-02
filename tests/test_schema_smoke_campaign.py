from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_schema_smoke_campaign import (
    campaign_paths,
    make_schema_smoke_config,
    run_campaign,
)


def test_make_schema_smoke_config_is_deterministic() -> None:
    cfg0 = make_schema_smoke_config(0, n_frequency=5)
    cfg1 = make_schema_smoke_config(1, n_frequency=5)

    assert cfg0["schema_version"] == "0.1.0"
    assert cfg0["simulation_type"] == "schema_smoke"
    assert cfg0["circuit_template"] == "matched_through_2port"
    assert cfg0["seed"] == 1234
    assert cfg1["seed"] == 1235

    assert cfg0["axes"]["frequency_hz"]["points"] == 5
    assert cfg0["axes"]["frequency_hz"]["start"] == 4.0e9
    assert cfg1["axes"]["frequency_hz"]["start"] == 4.1e9


def test_campaign_paths(tmp_path: Path) -> None:
    paths = campaign_paths(tmp_path)

    assert paths["configs"] == tmp_path / "configs"
    assert paths["runs"] == tmp_path / "runs"
    assert paths["registry"] == tmp_path / "runs.csv"
    assert paths["summary"] == tmp_path / "campaign_summary.json"


def test_actual_schema_smoke_campaign_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    summary = run_campaign(
        n=2,
        harmonia_root=harmonia_root,
        campaign_dir=tmp_path / "campaign",
        force=True,
        timeout_s=120.0,
    )

    assert summary["n_requested"] == 2
    assert summary["n_launched"] == 2
    assert summary["registry"]["by_status"] == {"PASS": 2}
    assert summary["registry"]["by_simulation_type"] == {"schema_smoke": 2}

    paths = campaign_paths(tmp_path / "campaign")
    assert paths["registry"].exists()
    assert paths["summary"].exists()
    assert (paths["runs"] / "run_000" / "simulation.h5").exists()
    assert (paths["runs"] / "run_001" / "simulation.h5").exists()