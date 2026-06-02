from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from twpa.io.dataset_builder import build_harmonia_rf_jtl_linear_dataset
ROOT=Path(__file__).resolve().parents[2]
if __name__=="__main__": print(build_harmonia_rf_jtl_linear_dataset(registry_csv=ROOT/"outputs/campaigns/harmonia_rf_jtl_linear_jc/runs.csv",output_dir=ROOT/"outputs/datasets/harmonia_rf_jtl_linear_jc_v0"))
