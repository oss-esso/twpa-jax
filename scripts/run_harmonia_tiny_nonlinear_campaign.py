from pathlib import Path
import argparse, numpy as np, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from twpa.io.campaigns import run_parameter_campaign, compute_one_port_run_metrics
from twpa.io.simulation_schema import SCHEMA_VERSION, write_json
from twpa.io.julia_bridge import load_julia_simulation
ROOT=Path(__file__).resolve().parents[2]
def config(i,v): return {"schema_version":SCHEMA_VERSION,"simulation_type":"harmonia_tiny_nonlinear_hb_smoke","circuit_template":"circuit_ir_tiny_nonlinear_jpa_reflection","seed":4700+i,"parameters":{"R_ohm":50.0,"Cc_F":100e-15,"Lj_H":1e-9,"Cj_F":1e-12,"pump_frequency_hz":4.75001e9,"pump_current_a":v,"n_pump_harmonics":2,"n_modulation_harmonics":2},"axes":{"frequency_hz":{"start":4.7e9,"stop":4.8e9,"points":3}}}
def run_campaign(*,pump_currents_a,harmonia_root,campaign_dir,force=False):
 s=run_parameter_campaign(values=pump_currents_a,parameter_name="pump_current_a",campaign_type="harmonia_tiny_nonlinear_pump_sweep",harmonia_root=harmonia_root,campaign_dir=campaign_dir,make_config=config,run_name=lambda v:f"pump_{v:.3e}".replace(".","p").replace("-","m"),force=force,compute_metrics=compute_one_port_run_metrics)
 arr=[load_julia_simulation(Path(r["output_dir"])).s_parameters for r in s["runs"] if r["ok"]]; s["response_variation_max_abs"]=float(max(np.max(np.abs(a-arr[0])) for a in arr)); s["pump_dependence_observed"]=s["response_variation_max_abs"]>1e-12; write_json(campaign_dir/"campaign_summary.json",s); return s
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--force",action="store_true"); a=p.parse_args(); print(run_campaign(pump_currents_a=[0,3e-9,5.65e-9],harmonia_root=ROOT/"Harmonia.jl",campaign_dir=ROOT/"outputs/campaigns/harmonia_tiny_nonlinear",force=a.force))
