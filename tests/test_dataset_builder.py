from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_linear_sparams_campaign import run_campaign
from twpa.io.dataset_builder import (
    DEFAULT_LINEAR_PARAMETER_NAMES,
    build_linear_sparams_dataset,
    extract_parameter_vector,
    load_linear_sparams_dataset,
)


def test_extract_parameter_vector() -> None:
    config = {
        "parameters": {
            "z_ref_ohm": 50.0,
            "z_line_ohm": 45.0,
            "length_m": 0.1,
            "phase_velocity_m_per_s": 1.2e8,
            "attenuation_np_per_m": 0.0,
        }
    }

    x = extract_parameter_vector(config)

    assert x.shape == (len(DEFAULT_LINEAR_PARAMETER_NAMES),)
    np.testing.assert_allclose(x, [50.0, 45.0, 0.1, 1.2e8, 0.0])


def test_extract_parameter_vector_rejects_missing_parameter() -> None:
    config = {"parameters": {"z_ref_ohm": 50.0}}

    with pytest.raises(ValueError, match="Missing required parameter"):
        extract_parameter_vector(config)


def test_actual_linear_campaign_dataset_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists():
        pytest.skip("Local Harmonia.jl runner not available.")

    campaign_dir = tmp_path / "campaign"
    dataset_dir = tmp_path / "dataset"

    summary = run_campaign(
        z_lines_ohm=[45.0, 50.0, 55.0],
        harmonia_root=harmonia_root,
        campaign_dir=campaign_dir,
        force=True,
        timeout_s=120.0,
        n_frequency=11,
    )

    assert summary["registry"]["by_status"] == {"PASS": 3}

    built = build_linear_sparams_dataset(
        registry_csv=campaign_dir / "runs.csv",
        output_dir=dataset_dir,
    )

    assert built.n_samples == 3
    assert built.n_frequency == 11
    assert built.dataset_npz.exists()
    assert built.summary_json.exists()

    data = load_linear_sparams_dataset(built.dataset_npz)

    assert data["parameters"].shape == (3, len(DEFAULT_LINEAR_PARAMETER_NAMES))
    assert data["frequency_hz"].shape == (11,)
    assert data["s_real"].shape == (3, 11, 2, 2)
    assert data["s_imag"].shape == (3, 11, 2, 2)
    assert data["gain_db"].shape == (3, 11)

    z_lines = data["parameters"][:, 1]
    np.testing.assert_allclose(z_lines, [45.0, 50.0, 55.0])

    s = data["s_real"] + 1j * data["s_imag"]
    s11_max = np.max(np.abs(s[:, :, 0, 0]), axis=1)

    assert s11_max[1] < s11_max[0]
    assert s11_max[1] < s11_max[2]
    assert s11_max[1] < 1e-10