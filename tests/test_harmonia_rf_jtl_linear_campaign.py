from pathlib import Path
from scripts.run_harmonia_rf_jtl_linear_campaign import run_campaign
def test_rf_campaign(tmp_path):
 s=run_campaign(lrf_values_h=[80e-9,100e-9,120e-9],harmonia_root=Path(r"D:\Projects\Thesis\Harmonia.jl"),campaign_dir=tmp_path/"c",force=True)
 assert s["registry"]["by_status"]=={"PASS":3}
 assert s["runs"][0]["metrics"] != s["runs"][-1]["metrics"]
