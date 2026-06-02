from pathlib import Path
from scripts.run_harmonia_rf_jtl_linear_campaign import run_campaign
from twpa.io.dataset_builder import build_harmonia_rf_jtl_linear_dataset
import numpy as np
def test_rf_dataset(tmp_path):
 run_campaign(lrf_values_h=[80e-9,100e-9,120e-9],harmonia_root=Path(r"D:\Projects\Thesis\Harmonia.jl"),campaign_dir=tmp_path/"c",force=True)
 b=build_harmonia_rf_jtl_linear_dataset(registry_csv=tmp_path/"c/runs.csv",output_dir=tmp_path/"d"); d=np.load(b.dataset_npz)
 assert d["parameters"].shape==(3,11); assert d["s_real"].shape==(3,5,2,2)
