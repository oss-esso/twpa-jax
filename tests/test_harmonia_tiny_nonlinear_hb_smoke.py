from pathlib import Path

import numpy as np
import h5py

from twpa.io.julia_bridge import load_julia_simulation
from twpa.io.julia_runner import run_harmonia_simulation


def test_actual_harmonia_tiny_nonlinear_hb_smoke(tmp_path: Path) -> None:
    root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    result = run_harmonia_simulation(config_path=root / "examples/configs/harmonia_tiny_nonlinear_hb_smoke.json", output_dir=tmp_path / "run", harmonia_jl_root=root, force=True, timeout_s=180.0)
    assert result.ok
    assert result.status is not None
    assert result.status.simulation_type == "harmonia_tiny_nonlinear_hb_smoke"
    data = load_julia_simulation(tmp_path / "run")
    assert data.frequency_hz is not None
    assert data.s_parameters is not None
    assert np.all(np.isfinite(data.s_parameters))
    assert np.max(np.abs(data.s_parameters)) > 0
    with h5py.File(tmp_path / "run/simulation.h5") as h5:
        assert float(h5["metadata/pump_current_a"][()]) > 0
        assert not bool(h5["metadata/residual_available"][()])
        assert "status_basis" in h5["metadata"]
        assert "topology" in h5
