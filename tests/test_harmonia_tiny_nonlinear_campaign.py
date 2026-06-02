from pathlib import Path
from scripts.run_harmonia_tiny_nonlinear_campaign import run_campaign
def test_nonlinear_campaign(tmp_path):
 s=run_campaign(pump_currents_a=[0,3e-9,5.65e-9],harmonia_root=Path(r"D:\Projects\Thesis\Harmonia.jl"),campaign_dir=tmp_path/"c",force=True)
 assert s["registry"]["by_status"]=={"PASS":3}; assert s["response_variation_max_abs"] >= 0
