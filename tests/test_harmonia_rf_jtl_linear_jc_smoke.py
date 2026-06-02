from pathlib import Path

import numpy as np

from twpa.io.julia_bridge import load_julia_simulation
from twpa.io.julia_runner import run_harmonia_simulation


def test_actual_harmonia_rf_jtl_linear_jc_smoke(tmp_path: Path) -> None:
    root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    result = run_harmonia_simulation(
        config_path=root / "examples/configs/harmonia_rf_jtl_linear_jc_smoke.json",
        output_dir=tmp_path / "run",
        harmonia_jl_root=root,
        force=True,
    )
    assert result.ok
    data = load_julia_simulation(tmp_path / "run")
    assert data.s_parameters is not None
    assert data.s_parameters.shape == (5, 2, 2)
    assert np.all(np.isfinite(data.s_parameters))
