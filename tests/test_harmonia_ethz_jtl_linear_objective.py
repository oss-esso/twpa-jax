from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_harmonia_ethz_jtl_linear_objective import (
    evaluate_dataset_against_target,
    find_nearest_parameter_index,
)
from scripts.run_harmonia_ethz_jtl_linear_campaign import run_campaign
from twpa.io.dataset_builder import (
    build_harmonia_ethz_jtl_linear_dataset,
    load_harmonia_ethz_jtl_linear_dataset,
)


def test_find_nearest_parameter_index() -> None:
    import numpy as np

    parameters = np.asarray(
        [
            [10.0, 140.0e-12],
            [10.0, 158.0e-12],
            [10.0, 180.0e-12],
        ]
    )

    idx = find_nearest_parameter_index(
        parameters=parameters,
        parameter_names=["n_cells", "Lj_H"],
        parameter_name="Lj_H",
        target_value=160.0e-12,
    )

    assert idx == 1


def test_actual_harmonia_ethz_jtl_linear_objective_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    campaign_dir = tmp_path / "campaign"
    dataset_dir = tmp_path / "dataset"

    summary = run_campaign(
        lj_values_h=[140.0e-12, 158.0e-12, 180.0e-12],
        harmonia_root=harmonia_root,
        campaign_dir=campaign_dir,
        force=True,
        timeout_s=300.0,
        n_frequency=5,
        n_cells=10,
    )

    assert summary["registry"]["by_status"] == {"PASS": 3}

    built = build_harmonia_ethz_jtl_linear_dataset(
        registry_csv=campaign_dir / "runs.csv",
        output_dir=dataset_dir,
    )

    data = load_harmonia_ethz_jtl_linear_dataset(built.dataset_npz)
    assert data["parameters"].shape[0] == 3

    objective = evaluate_dataset_against_target(
        dataset_npz=built.dataset_npz,
        target_lj_h=158.0e-12,
    )

    assert objective["n_samples"] == 3
    assert objective["target"]["target_index"] == 1
    assert objective["best"]["sample_index"] == 1
    assert objective["best"]["total_loss"] < 1e-20

    ranked_indices = [row["sample_index"] for row in objective["ranked"]]
    assert ranked_indices[0] == 1

    losses = [row["total_loss"] for row in objective["ranked"]]
    assert losses == sorted(losses)