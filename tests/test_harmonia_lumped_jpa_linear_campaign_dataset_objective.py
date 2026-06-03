from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_harmonia_lumped_jpa_linear_objective import (
    evaluate_dataset_against_target,
    find_nearest_parameter_index,
)
from scripts.run_harmonia_lumped_jpa_linear_campaign import (
    campaign_paths,
    compute_lumped_jpa_linear_metrics,
    make_harmonia_lumped_jpa_linear_config,
    run_campaign,
)
from twpa.io.dataset_builder import (
    DEFAULT_HARMONIA_LUMPED_JPA_LINEAR_PARAMETER_NAMES,
    build_harmonia_lumped_jpa_linear_dataset,
    extract_parameter_vector,
    load_harmonia_lumped_jpa_linear_dataset,
)


def test_make_harmonia_lumped_jpa_linear_config() -> None:
    cfg = make_harmonia_lumped_jpa_linear_config(
        index=0,
        Lj_H=1000.0e-12,
        n_frequency=7,
    )

    assert cfg["schema_version"] == "0.1.0"
    assert cfg["simulation_type"] == "harmonia_lumped_jpa_linear_jc_smoke"
    assert cfg["circuit_template"] == "circuit_ir_lumped_jpa_reflection_linear_jc"
    assert cfg["parameters"]["Lj_H"] == 1000.0e-12
    assert cfg["parameters"]["port_impedance_ohm"] == 50.0
    assert cfg["axes"]["frequency_hz"]["points"] == 7


def test_campaign_paths(tmp_path: Path) -> None:
    paths = campaign_paths(tmp_path)

    assert paths["configs"] == tmp_path / "configs"
    assert paths["runs"] == tmp_path / "runs"
    assert paths["registry"] == tmp_path / "runs.csv"
    assert paths["summary"] == tmp_path / "campaign_summary.json"


def test_extract_lumped_jpa_parameter_vector() -> None:
    config = {
        "parameters": {
            "Cc_F": 100.0e-15,
            "Lj_H": 1000.0e-12,
            "Cj_F": 1000.0e-15,
            "port_impedance_ohm": 50.0,
            "pump_frequency_hz": 6.0e9,
            "pump_current_a": 0.0,
            "n_pump_harmonics": 1,
            "n_modulation_harmonics": 1,
        }
    }

    x = extract_parameter_vector(
        config,
        parameter_names=DEFAULT_HARMONIA_LUMPED_JPA_LINEAR_PARAMETER_NAMES,
    )

    assert x.shape == (len(DEFAULT_HARMONIA_LUMPED_JPA_LINEAR_PARAMETER_NAMES),)
    np.testing.assert_allclose(
        x,
        [
            100.0e-15,
            1000.0e-12,
            1000.0e-15,
            50.0,
            6.0e9,
            0.0,
            1.0,
            1.0,
        ],
    )


def test_find_nearest_parameter_index() -> None:
    parameters = np.asarray(
        [
            [800.0e-12],
            [1000.0e-12],
            [1200.0e-12],
        ]
    )

    idx = find_nearest_parameter_index(
        parameters=parameters,
        parameter_names=["Lj_H"],
        parameter_name="Lj_H",
        target_value=990.0e-12,
    )

    assert idx == 1


def test_actual_lumped_jpa_campaign_dataset_objective_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    campaign_dir = tmp_path / "campaign"
    dataset_dir = tmp_path / "dataset"

    summary = run_campaign(
        lj_values_h=[800.0e-12, 1000.0e-12, 1200.0e-12],
        harmonia_root=harmonia_root,
        campaign_dir=campaign_dir,
        force=True,
        timeout_s=240.0,
        n_frequency=7,
    )

    assert summary["n_requested"] == 3
    assert summary["n_launched"] == 3
    assert summary["registry"]["by_status"] == {"PASS": 3}
    assert summary["registry"]["by_simulation_type"] == {
        "harmonia_lumped_jpa_linear_jc_smoke": 3
    }

    for run in summary["runs"]:
        assert run["ok"]
        assert run["status"] == "PASS"
        assert run["metrics"] is not None
        assert run["metrics"]["all_arrays_finite"]
        assert run["metrics"]["frequency_points"] == 7
        assert run["metrics"]["s_shape"] == [7, 1, 1]

    built = build_harmonia_lumped_jpa_linear_dataset(
        registry_csv=campaign_dir / "runs.csv",
        output_dir=dataset_dir,
    )

    assert built.n_samples == 3
    assert built.n_frequency == 7
    assert built.dataset_npz.exists()
    assert built.summary_json.exists()

    data = load_harmonia_lumped_jpa_linear_dataset(built.dataset_npz)

    assert data["parameters"].shape == (
        3,
        len(DEFAULT_HARMONIA_LUMPED_JPA_LINEAR_PARAMETER_NAMES),
    )
    assert data["frequency_hz"].shape == (7,)
    assert data["s_real"].shape == (3, 7, 1, 1)
    assert data["s_imag"].shape == (3, 7, 1, 1)
    assert data["reflection_db"].shape == (3, 7)

    parameter_names = [str(x) for x in data["parameter_names"]]
    lj_idx = parameter_names.index("Lj_H")
    np.testing.assert_allclose(
        data["parameters"][:, lj_idx],
        [800.0e-12, 1000.0e-12, 1200.0e-12],
    )

    s = data["s_real"] + 1j * data["s_imag"]

    assert np.all(np.isfinite(s.real))
    assert np.all(np.isfinite(s.imag))
    assert np.all(np.isfinite(data["reflection_db"]))

    assert not np.allclose(data["s_real"][0], data["s_real"][-1])

    objective = evaluate_dataset_against_target(
        dataset_npz=built.dataset_npz,
        target_lj_h=1000.0e-12,
    )

    assert objective["n_samples"] == 3
    assert objective["target"]["target_index"] == 1
    assert objective["best"]["sample_index"] == 1
    assert objective["best"]["total_loss"] < 1e-20

    losses = [row["total_loss"] for row in objective["ranked"]]
    assert losses == sorted(losses)


def test_compute_metrics_on_existing_lumped_jpa_linear_if_available() -> None:
    run_dir = Path(r"D:\Projects\Thesis\outputs\harmonia_lumped_jpa_linear_jc_smoke\run_001")

    if not (run_dir / "simulation.h5").exists():
        pytest.skip("Local lumped JPA linear smoke output not available.")

    metrics = compute_lumped_jpa_linear_metrics(run_dir)

    assert metrics["frequency_points"] == 11
    assert metrics["s_shape"] == [11, 1, 1]
    assert metrics["all_arrays_finite"]
    assert metrics["reflection_db_max"] >= metrics["reflection_db_min"]