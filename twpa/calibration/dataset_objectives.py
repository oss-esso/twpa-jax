from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from twpa.calibration.objectives import evaluate_sparameter_objective
from twpa.io.dataset_builder import load_harmonia_jtl_linear_dataset
from twpa.io.simulation_schema import write_json
from twpa.calibration.objectives import evaluate_dataset_against_target

def select_target_index(parameters,parameter_names,parameter_name,target_value):
    idx=list(parameter_names).index(parameter_name); values=np.asarray(parameters[:,idx],dtype=float)
    return int(np.argmin(np.abs(values-float(target_value)))), values

def evaluate_two_port_dataset(*,dataset_npz,output_json,parameter_name,target_value):
    with np.load(dataset_npz,allow_pickle=False) as data: d={k:data[k] for k in data.files}
    names=[str(x) for x in d["parameter_names"]]; target,values=select_target_index(d["parameters"],names,parameter_name,target_value)
    rows=evaluate_dataset_against_target(parameters=d["parameters"],frequency_hz=d["frequency_hz"],s_complex=d["s_real"]+1j*d["s_imag"],gain_db=d["gain_db"],target_index=target)
    for row in rows: row["index"]=row["sample_index"]
    ranking=sorted(rows,key=lambda x:(x["total_loss"],x["sample_index"]))
    out={"dataset_npz":str(dataset_npz),"parameter_name":parameter_name,"target_requested":float(target_value),"target_index":target,"target_value":float(values[target]),"ranking":ranking}
    write_json(output_json,out); return out


def evaluate_harmonia_jtl_linear_dataset(
    *,
    dataset_npz: str | Path,
    output_json: str | Path,
    target_lj_h: float,
) -> dict[str, Any]:
    return evaluate_two_port_dataset(dataset_npz=dataset_npz,output_json=output_json,parameter_name="Lj_H",target_value=target_lj_h)
    data = load_harmonia_jtl_linear_dataset(dataset_npz)
    parameter_names = [str(name) for name in data["parameter_names"]]
    lj_index = parameter_names.index("Lj_H")
    lj_values = np.asarray(data["parameters"][:, lj_index], dtype=float)
    target_index = int(np.argmin(np.abs(lj_values - float(target_lj_h))))
    frequency_hz = np.asarray(data["frequency_hz"], dtype=float)
    s = np.asarray(data["s_real"] + 1j * data["s_imag"], dtype=np.complex128)
    gain_db = np.asarray(data["gain_db"], dtype=float)

    evaluations = []
    for index in range(s.shape[0]):
        result = evaluate_sparameter_objective(
            frequency_hz=frequency_hz,
            candidate_s=s[index],
            target_s=s[target_index],
            candidate_gain_db=gain_db[index],
            target_gain_db=gain_db[target_index],
        )
        evaluations.append({"index": index, "Lj_H": float(lj_values[index]), **result.to_dict()})

    ranking = sorted(evaluations, key=lambda item: (item["total_loss"], item["index"]))
    summary = {
        "dataset_npz": str(Path(dataset_npz)),
        "target_requested_Lj_H": float(target_lj_h),
        "target_index": target_index,
        "target_Lj_H": float(lj_values[target_index]),
        "ranking": ranking,
    }
    write_json(Path(output_json), summary)
    return summary
