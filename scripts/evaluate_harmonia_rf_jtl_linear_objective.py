from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from twpa.calibration.dataset_objectives import evaluate_two_port_dataset
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs/datasets/harmonia_rf_jtl_linear_jc_v0"
if __name__=="__main__": print(evaluate_two_port_dataset(dataset_npz=OUT/"harmonia_rf_jtl_linear_dataset.npz",output_json=OUT/"objective_summary.json",parameter_name="Lrf_H",target_value=100e-9))
