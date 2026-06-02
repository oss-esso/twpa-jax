from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twpa.io.julia_bridge import load_julia_simulation


def test_actual_jc_jpa_reflection_smoke_if_available() -> None:
    run_dir = Path(r"D:\Projects\Thesis\outputs\jc_jpa_reflection_smoke\run_001")

    if not (run_dir / "simulation.h5").exists():
        pytest.skip("Local JosephsonCircuits JPA smoke run not available.")

    data = load_julia_simulation(run_dir)

    assert data.status.status == "PASS"
    assert data.status.simulation_type == "jc_jpa_reflection_smoke"
    assert data.frequency_hz is not None
    assert data.s_parameters is not None
    assert data.gain_db is not None

    assert data.frequency_hz.shape == (11,)
    assert data.s_parameters.shape == (11, 1, 1)
    assert data.gain_db.shape == (11,)

    assert np.all(np.isfinite(data.frequency_hz))
    assert np.all(np.isfinite(data.s_parameters.real))
    assert np.all(np.isfinite(data.s_parameters.imag))
    assert np.all(np.isfinite(data.gain_db))