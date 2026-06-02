from pathlib import Path
from scripts.run_harmonia_rf_jtl_linear_campaign import run_campaign
from twpa.io.dataset_builder import build_harmonia_rf_jtl_linear_dataset
from twpa.calibration.dataset_objectives import evaluate_two_port_dataset
def test_rf_objective(tmp_path):
 run_campaign(lrf_values_h=[80e-9,100e-9,120e-9],harmonia_root=Path(r"D:\Projects\Thesis\Harmonia.jl"),campaign_dir=tmp_path/"c",force=True)
 b=build_harmonia_rf_jtl_linear_dataset(registry_csv=tmp_path/"c/runs.csv",output_dir=tmp_path/"d")
 s=evaluate_two_port_dataset(dataset_npz=b.dataset_npz,output_json=tmp_path/"o.json",parameter_name="Lrf_H",target_value=100e-9)
 assert s["ranking"][0]["sample_index"]==1; assert s["ranking"][0]["total_loss"] < 1e-20
