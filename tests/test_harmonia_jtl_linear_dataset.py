from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_harmonia_jtl_linear_campaign import run_campaign
from twpa.io.dataset_builder import (
    DEFAULT_HARMONIA_JTL_LINEAR_PARAMETER_NAMES,
    build_harmonia_jtl_linear_dataset,
    extract_parameter_vector,
    load_harmonia_jtl_linear_dataset,
)


def test_extract_harmonia_jtl_linear_parameter_vector() -> None:
    config = {
        "parameters": {
            "N_cell": 4,
            "Cg_F": 50.0e-15,
            "Lj_H": 1.0e-9,
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
        parameter_names=DEFAULT_HARMONIA_JTL_LINEAR_PARAMETER_NAMES,
    )

    assert x.shape == (len(DEFAULT_HARMONIA_JTL_LINEAR_PARAMETER_NAMES),)
    np.testing.assert_allclose(
        x,
        [
            4.0,
            50.0e-15,
            1.0e-9,
            1000.0e-15,
            50.0,
            6.0e9,
            0.0,
            1.0,
            1.0,
        ],
    )


def test_actual_harmonia_jtl_linear_dataset_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    campaign_dir = tmp_path / "campaign"
    dataset_dir = tmp_path / "dataset"

    summary = run_campaign(
        lj_values_h=[0.8e-9, 1.0e-9, 1.2e-9],
        harmonia_root=harmonia_root,
        campaign_dir=campaign_dir,
        force=True,
        timeout_s=240.0,
        n_cell=4,
        n_frequency=7,
    )

    assert summary["registry"]["by_status"] == {"PASS": 3}

    built = build_harmonia_jtl_linear_dataset(
        registry_csv=campaign_dir / "runs.csv",
        output_dir=dataset_dir,
    )

    assert built.n_samples == 3
    assert built.n_frequency == 7
    assert built.dataset_npz.exists()
    assert built.summary_json.exists()

    data = load_harmonia_jtl_linear_dataset(built.dataset_npz)

    assert data["parameters"].shape == (
        3,
        len(DEFAULT_HARMONIA_JTL_LINEAR_PARAMETER_NAMES),
    )
    assert data["frequency_hz"].shape == (7,)
    assert data["s_real"].shape == (3, 7, 2, 2)
    assert data["s_imag"].shape == (3, 7, 2, 2)
    assert data["gain_db"].shape == (3, 7)

    lj_idx = list(data["parameter_names"]).index("Lj_H")
    lj_values = data["parameters"][:, lj_idx]
    np.testing.assert_allclose(lj_values, [0.8e-9, 1.0e-9, 1.2e-9])

    s = data["s_real"] + 1j * data["s_imag"]

    assert np.all(np.isfinite(s.real))
    assert np.all(np.isfinite(s.imag))
    assert np.all(np.isfinite(data["gain_db"]))

    # The sweep should not collapse to identical S-parameter tensors.
    assert not np.allclose(data["s_real"][0], data["s_real"][-1])