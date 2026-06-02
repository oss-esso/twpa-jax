from __future__ import annotations

import json
from pathlib import Path
import h5py
import numpy as np
import pytest

from twpa.io.julia_bridge import load_julia_simulation
from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.hdf5_utils import decode_h5_string


def test_actual_harmonia_jtl_linear_jc_smoke_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    config_path = harmonia_root / "examples" / "configs" / "harmonia_jtl_linear_jc_smoke.json"

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists() or not config_path.exists():
        pytest.skip("Local Harmonia.jl JTL linear JC smoke setup not available.")

    output_dir = tmp_path / "harmonia_jtl_linear_jc_smoke"

    result = run_harmonia_simulation(
        config_path=config_path,
        output_dir=output_dir,
        harmonia_jl_root=harmonia_root,
        force=True,
        timeout_s=180.0,
    )

    assert result.returncode == 0
    assert result.ok
    assert result.status is not None
    assert result.status.status == "PASS"
    assert result.status.simulation_type == "harmonia_jtl_linear_jc_smoke"

    data = load_julia_simulation(output_dir)

    assert data.status.status == "PASS"
    assert data.status.simulation_type == "harmonia_jtl_linear_jc_smoke"
    assert data.status.circuit_template == "circuit_ir_jtl_chain_linear_jc"

    assert data.frequency_hz is not None
    assert data.s_parameters is not None
    assert data.gain_db is not None

    assert data.frequency_hz.shape == (11,)
    assert data.s_parameters.shape == (11, 2, 2)
    assert data.gain_db.shape == (11,)

    assert np.all(np.isfinite(data.frequency_hz))
    assert np.all(np.isfinite(data.s_parameters.real))
    assert np.all(np.isfinite(data.s_parameters.imag))
    assert np.all(np.isfinite(data.gain_db))

    # Not a physics claim yet; just make sure the solver did not emit all zeros.
    assert np.max(np.abs(data.s_parameters)) > 0.0

    h5_path = output_dir / "simulation.h5"

    with h5py.File(h5_path, "r") as h5:
        assert decode_h5_string(h5.attrs["simulation_type"]) == "harmonia_jtl_linear_jc_smoke"
        assert decode_h5_string(h5.attrs["backend"]) == "Harmonia.CircuitIR + JosephsonCircuits.hbsolve"
        assert bool(h5.attrs["topology_only"]) is False
        assert int(h5.attrs["n_ports"]) == 2

        topology = json.loads(decode_h5_string(h5["topology"]["topology_json"][()]))

    names = topology["solver_export_names"]

    assert "R_P1" in names
    assert "R_P2" in names
    assert "C_jtl_Cg_1" in names
    assert "Lj_jtl_Lj_1" in names
    assert "C_jtl_Lj_1" in names
