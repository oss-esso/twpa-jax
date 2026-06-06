"""Reusable helpers for Julia/Harmonia simulation campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from types import SimpleNamespace
import shutil

from twpa.io.julia_bridge import load_julia_simulation, read_status_json
from twpa.io.run_registry import register_run_dir
from twpa.io.simulation_schema import compute_two_port_metrics
from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.julia_batch_runner import run_harmonia_simulation_batch
from twpa.io.run_registry import registry_summary
from twpa.io.simulation_schema import write_json


def campaign_paths(campaign_dir: Path) -> dict[str, Path]:
    return {
        "configs": campaign_dir / "configs",
        "runs": campaign_dir / "runs",
        "registry": campaign_dir / "runs.csv",
        "summary": campaign_dir / "campaign_summary.json",
    }


def compute_two_port_run_metrics(run_dir: Path) -> dict[str, Any]:
    data = load_julia_simulation(run_dir)
    if data.frequency_hz is None:
        raise ValueError(f"Missing frequency axis: {run_dir}")
    if data.s_parameters is None:
        raise ValueError(f"Missing S-parameters: {run_dir}")
    if data.gain_db is None:
        raise ValueError(f"Missing gain_db: {run_dir}")
    return compute_two_port_metrics(
        frequency_hz=data.frequency_hz,
        s_parameters=data.s_parameters,
        gain_db=data.gain_db,
    ).to_dict()


def register_completed_run(
    *,
    registry_csv: Path,
    run_dir: Path,
    result: Any,
    compute_metrics=compute_two_port_run_metrics,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "returncode": result.returncode,
        "ok": result.ok,
        "output_dir": str(run_dir),
        "status": None if result.status is None else result.status.status,
        "run_id": None if result.status is None else result.status.run_id,
    }
    if result.status is not None:
        registered = register_run_dir(registry_csv, run_dir)
        record["registered_status"] = registered.status
    if result.ok:
        record["metrics"] = compute_metrics(run_dir)
    else:
        record["metrics"] = None
        record["failure_reason"] = None if result.status is None else result.status.failure_reason
    return record

def compute_one_port_run_metrics(run_dir: Path) -> dict[str, Any]:
    data=load_julia_simulation(run_dir)
    if data.frequency_hz is None or data.s_parameters is None or data.gain_db is None:
        raise ValueError(f"Missing one-port arrays: {run_dir}")
    s11=data.s_parameters[:,0,0]
    return {"frequency_points":len(data.frequency_hz),"s_shape":list(data.s_parameters.shape),
        "max_abs_s11":float(abs(s11).max()),"reflection_db_min":float(data.gain_db.min()),
        "reflection_db_max":float(data.gain_db.max()),"all_arrays_finite":bool(
        __import__("numpy").all(__import__("numpy").isfinite(data.s_parameters)))}

def run_parameter_campaign(*, values, parameter_name, campaign_type, harmonia_root, campaign_dir,
    make_config, run_name, timeout_s=300.0, force=False, compute_metrics=compute_two_port_run_metrics,
    use_batch_runner=False, julia_executable="julia"):
    if force and campaign_dir.exists(): shutil.rmtree(campaign_dir)
    paths=campaign_paths(campaign_dir); paths["configs"].mkdir(parents=True,exist_ok=True); paths["runs"].mkdir(parents=True,exist_ok=True)
    runs=[]
    prepared=[]

    for index,value in enumerate(values):
        name=run_name(value); config_path=paths["configs"]/f"{name}.json"; output_dir=paths["runs"]/name
        write_json(config_path,make_config(index,float(value)))
        prepared.append((index, float(value), name, config_path, output_dir))

    if use_batch_runner:
        batch_result=run_harmonia_simulation_batch(
            items=[(config_path, output_dir) for _,_,_,config_path,output_dir in prepared],
            harmonia_jl_root=harmonia_root,
            julia_executable=julia_executable,
            timeout_s=timeout_s,
            force=force,
            use_cache=not force,
            batch_work_dir=paths["runs"]/"_julia_batch_runner",
        )
        record_by_output={Path(r.output_dir).resolve(): r for r in batch_result.records}

        for _,value,name,config_path,output_dir in prepared:
            status_path=output_dir/"status.json"
            status=read_status_json(status_path) if status_path.exists() else None
            batch_record=record_by_output.get(output_dir.resolve())
            returncode=batch_record.returncode if batch_record is not None else batch_result.returncode
            ok=(returncode == 0 and status is not None and status.status == "PASS")
            result=SimpleNamespace(returncode=returncode,ok=ok,output_dir=output_dir,status=status)
            record={"run_name":name,parameter_name:float(value),"batch_runner":True}
            record.update(register_completed_run(registry_csv=paths["registry"],run_dir=output_dir,result=result,compute_metrics=compute_metrics)); runs.append(record)
    else:
        for _,value,name,config_path,output_dir in prepared:
            result=run_harmonia_simulation(config_path=config_path,output_dir=output_dir,harmonia_jl_root=harmonia_root,
                julia_executable=julia_executable,force=force,timeout_s=timeout_s,use_cache=not force)
            record={"run_name":name,parameter_name:float(value),"batch_runner":False}
            record.update(register_completed_run(registry_csv=paths["registry"],run_dir=output_dir,result=result,compute_metrics=compute_metrics)); runs.append(record)

    summary={"campaign_type":campaign_type,"campaign_dir":str(campaign_dir),"swept_parameter":parameter_name,
        "swept_values":[float(v) for v in values],"n_requested":len(values),"n_launched":len(runs),
        "use_batch_runner":bool(use_batch_runner),"registry":registry_summary(paths["registry"]),"runs":runs}
    write_json(paths["summary"],summary); return summary
