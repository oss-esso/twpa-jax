from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_harmonia_ethz_jtl_linear_campaign import run_campaign
from twpa.io.dataset_builder import (
    DEFAULT_HARMONIA_ETHZ_JTL_LINEAR_PARAMETER_NAMES,
    build_harmonia_ethz_jtl_linear_dataset,
    extract_parameter_vector,
    load_harmonia_ethz_jtl_linear_dataset,
)


def test_extract_harmonia_ethz_jtl_linear_parameter_vector() -> None:
    config = {
        "parameters": {
            "n_cells": 10,
            "section_pitch": 4,
            "section_phase": 0,
            "short_section_segments": 2,
            "long_section_segments": 3,
            "long_section_every": 2,
            "Cg_F": 66.0e-15,
            "Lj_H": 158.0e-12,
            "Cj_F": 147.0e-15,
            "Cl_F": 1.73e-15,
            "Ll_H": 4.13e-12,
            "port_impedance_ohm": 50.0,
            "pump_frequency_hz": 6.0e9,
            "pump_current_a": 0.0,
            "n_pump_harmonics": 1,
            "n_modulation_harmonics": 1,
        }
    }

    x = extract_parameter_vector(
        config,
        parameter_names=DEFAULT_HARMONIA_ETHZ_JTL_LINEAR_PARAMETER_NAMES,
    )

    assert x.shape == (len(DEFAULT_HARMONIA_ETHZ_JTL_LINEAR_PARAMETER_NAMES),)
    np.testing.assert_allclose(
        x,
        [
            10.0,
            4.0,
            0.0,
            2.0,
            3.0,
            2.0,
            66.0e-15,
            158.0e-12,
            147.0e-15,
            1.73e-15,
            4.13e-12,
            50.0,
            6.0e9,
            0.0,
            1.0,
            1.0,
        ],
    )


def test_actual_harmonia_ethz_jtl_linear_dataset_if_available(tmp_path: Path) -> None:
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

    assert built.n_samples == 3
    assert built.n_frequency == 5
    assert built.dataset_npz.exists()
    assert built.summary_json.exists()

    data = load_harmonia_ethz_jtl_linear_dataset(built.dataset_npz)

    assert data["parameters"].shape == (
        3,
        len(DEFAULT_HARMONIA_ETHZ_JTL_LINEAR_PARAMETER_NAMES),
    )
    assert data["frequency_hz"].shape == (5,)
    assert data["s_real"].shape == (3, 5, 2, 2)
    assert data["s_imag"].shape == (3, 5, 2, 2)
    assert data["gain_db"].shape == (3, 5)

    parameter_names = [str(x) for x in data["parameter_names"]]
    lj_idx = parameter_names.index("Lj_H")
    lj_values = data["parameters"][:, lj_idx]

    np.testing.assert_allclose(lj_values, [140.0e-12, 158.0e-12, 180.0e-12])

    s = data["s_real"] + 1j * data["s_imag"]

    assert np.all(np.isfinite(s.real))
    assert np.all(np.isfinite(s.imag))
    assert np.all(np.isfinite(data["gain_db"]))

    assert not np.allclose(data["s_real"][0], data["s_real"][-1])