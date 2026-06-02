from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_jc_jpa_reflection_campaign import run_campaign
from twpa.io.dataset_builder import (
    DEFAULT_JC_JPA_REFLECTION_PARAMETER_NAMES,
    build_jc_jpa_reflection_dataset,
    extract_parameter_vector,
    load_jc_jpa_reflection_dataset,
)


def test_extract_jc_jpa_parameter_vector() -> None:
    config = {
        "parameters": {
            "R_ohm": 50.0,
            "Cc_F": 100.0e-15,
            "Lj_H": 1000.0e-12,
            "Cj_F": 1000.0e-15,
            "pump_frequency_hz": 4.75001e9,
            "pump_current_a": 2.0e-9,
            "n_pump_harmonics": 4,
            "n_modulation_harmonics": 4,
        }
    }

    x = extract_parameter_vector(
        config,
        parameter_names=DEFAULT_JC_JPA_REFLECTION_PARAMETER_NAMES,
    )

    assert x.shape == (len(DEFAULT_JC_JPA_REFLECTION_PARAMETER_NAMES),)
    np.testing.assert_allclose(
        x,
        [
            50.0,
            100.0e-15,
            1000.0e-12,
            1000.0e-15,
            4.75001e9,
            2.0e-9,
            4.0,
            4.0,
        ],
    )


def test_actual_jc_jpa_reflection_dataset_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    campaign_dir = tmp_path / "campaign"
    dataset_dir = tmp_path / "dataset"

    summary = run_campaign(
        pump_currents_a=[0.0, 2.0e-9, 5.65e-9],
        harmonia_root=harmonia_root,
        campaign_dir=campaign_dir,
        force=True,
        timeout_s=180.0,
        n_frequency=7,
    )

    assert summary["registry"]["by_status"] == {"PASS": 3}

    built = build_jc_jpa_reflection_dataset(
        registry_csv=campaign_dir / "runs.csv",
        output_dir=dataset_dir,
    )

    assert built.n_samples == 3
    assert built.n_frequency == 7
    assert built.dataset_npz.exists()
    assert built.summary_json.exists()

    data = load_jc_jpa_reflection_dataset(built.dataset_npz)

    assert data["parameters"].shape == (
        3,
        len(DEFAULT_JC_JPA_REFLECTION_PARAMETER_NAMES),
    )
    assert data["frequency_hz"].shape == (7,)
    assert data["s11_real"].shape == (3, 7)
    assert data["s11_imag"].shape == (3, 7)
    assert data["reflection_db"].shape == (3, 7)

    pump_idx = list(data["parameter_names"]).index("pump_current_a")
    pump_currents = data["parameters"][:, pump_idx]
    np.testing.assert_allclose(pump_currents, [0.0, 2.0e-9, 5.65e-9])

    s11 = data["s11_real"] + 1j * data["s11_imag"]
    assert np.all(np.isfinite(s11.real))
    assert np.all(np.isfinite(s11.imag))
    assert np.all(np.isfinite(data["reflection_db"]))

    # The sweep should not collapse to identical curves.
    assert not np.allclose(data["reflection_db"][0], data["reflection_db"][-1])