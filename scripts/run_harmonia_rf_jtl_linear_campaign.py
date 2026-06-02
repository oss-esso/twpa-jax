from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from twpa.io.campaigns import run_parameter_campaign
from twpa.io.simulation_schema import SCHEMA_VERSION
ROOT=Path(__file__).resolve().parents[2]
def make_config(i,v): return {"schema_version":SCHEMA_VERSION,"simulation_type":"harmonia_rf_jtl_linear_jc_smoke","circuit_template":"circuit_ir_rf_jtl_chain_linear_jc","seed":4600+i,"parameters":{"N_cell":2,"prefix":"rf_jtl","start_node":"n1","ground":"0","Cg_F":50e-15,"Lj_H":1e-9,"Cj_F":1e-12,"Lrf_H":v,"Lp_H":10e-12,"port_impedance_ohm":50.0,"pump_frequency_hz":6e9,"pump_current_a":0.0,"n_pump_harmonics":1,"n_modulation_harmonics":1},"axes":{"frequency_hz":{"start":4e9,"stop":8e9,"points":5}}}
def run_campaign(*,lrf_values_h,harmonia_root,campaign_dir,force=False,timeout_s=300): return run_parameter_campaign(values=lrf_values_h,parameter_name="Lrf_H",campaign_type="harmonia_rf_jtl_lrf_sweep",harmonia_root=harmonia_root,campaign_dir=campaign_dir,make_config=make_config,run_name=lambda v:f"Lrf_{v:.3e}".replace(".","p").replace("-","m"),force=force,timeout_s=timeout_s)
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--force",action="store_true"); a=p.parse_args(); print(run_campaign(lrf_values_h=[80e-9,100e-9,120e-9],harmonia_root=ROOT/"Harmonia.jl",campaign_dir=ROOT/"outputs/campaigns/harmonia_rf_jtl_linear_jc",force=a.force))
