from pathlib import Path
from twpa.io.julia_runner import run_harmonia_simulation
import h5py,json
from twpa.io.hdf5_utils import decode_h5_string
def test_coupler(tmp_path):
 root=Path(r"D:\Projects\Thesis\Harmonia.jl"); r=run_harmonia_simulation(config_path=root/"examples/configs/harmonia_coupler_topology_smoke.json",output_dir=tmp_path/"r",harmonia_jl_root=root,force=True); assert r.ok
 with h5py.File(tmp_path/"r/simulation.h5") as h: t=json.loads(decode_h5_string(h["topology/topology_json"][()]))
 assert not t["solver_export_supported"]; assert "mutual-inductor" in t["solver_export_blocker"]
