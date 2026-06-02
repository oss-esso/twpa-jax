import json
from pathlib import Path

import h5py

from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.hdf5_utils import decode_h5_string


def test_actual_harmonia_rf_jtl_topology_smoke(tmp_path: Path) -> None:
    root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    result = run_harmonia_simulation(config_path=root / "examples/configs/harmonia_rf_jtl_topology_smoke.json", output_dir=tmp_path / "run", harmonia_jl_root=root, force=True)
    assert result.ok
    with h5py.File(tmp_path / "run/simulation.h5") as h5:
        assert bool(h5.attrs["topology_only"])
        topology = json.loads(decode_h5_string(h5["topology"]["topology_json"][()]))
    assert topology["expected_ir_elements"] == 10
    assert topology["expected_jc_tuples"] == 12
    assert topology["ir_element_count_match"]
    assert topology["jc_tuple_count_match"]
    assert all(name[0] in {"P", "R", "C", "L"} for name in topology["solver_export_names"])
