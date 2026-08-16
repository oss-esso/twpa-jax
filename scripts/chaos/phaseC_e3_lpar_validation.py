"""Measure whether the RF-SQUID Lpar stamp affects the C-G4 observable."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from twpa_solver.builders.ipm import Element, LossSpec, build_matrices  # noqa: E402
from scripts.chaos.run_guarcello_jc_phase5 import (  # noqa: E402
    _built_element_records,
    _measure_linear_limit,
    derive_device_spec,
    load_jc_device,
    phase_c_source_path,
    resolve_pump_frequency,
)


OUT = ROOT / "outputs" / "chaos" / "phaseC" / "E3_lpar_mutation.json"


def _write(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mutated_device(device, factor: float):
    source = resolve_device_directory_for_records(phase_c_source_path(device.name))
    records = _built_element_records(source)
    elements = []
    lpar_count = 0
    for item in records:
        value = item["value"]
        if item.get("role") == "rf_squid_lpar":
            value = float(value) * factor
            lpar_count += 1
        elements.append(
            Element(
                name=item["name"], n1=item["n1"], n2=item["n2"],
                value=value, kind=item["kind"], role=item.get("role", ""),
                cell_index=item.get("cell_index"),
            )
        )
    matrices = build_matrices(elements, LossSpec(0.0))
    K = matrices["K"].tocsr()
    if device.selected_ordering == "rcm":
        K = K[device.permutation][:, device.permutation].tocsr()
    mutated = replace(device, K=K)
    return mutated, lpar_count


def resolve_device_directory_for_records(source: Path) -> Path:
    from scripts.chaos.run_guarcello_jc_phase5 import resolve_device_directory

    return resolve_device_directory(source)


def main() -> None:
    source = phase_c_source_path("rf_squid_2393_3wm")
    device = load_jc_device(source)
    spec = derive_device_spec(source)
    pump_hz = resolve_pump_frequency(spec)
    records = _built_element_records(resolve_device_directory_for_records(source))
    role_counts: dict[str, int] = {}
    for item in records:
        role = str(item.get("role", ""))
        role_counts[role] = role_counts.get(role, 0) + 1
    mass = (device.C + device.G).tocsr()
    zero_rows = np.flatnonzero(np.asarray(mass.getnnz(axis=1)).reshape(-1) == 0)
    payload = {
        "device": device.name,
        "pump_hz": pump_hz,
        "dt_norm": 0.01,
        "selected_ordering": device.selected_ordering,
        "selected_bandwidth": device.selected_bandwidth,
        "element_role_counts": role_counts,
        "lpar_values_ohm_s": sorted({float(item["value"]) for item in records if item.get("role") == "rf_squid_lpar"}),
        "mass_matrix_shape": list(mass.shape),
        "mass_zero_rows": int(zero_rows.size),
        "mass_zero_row_indices_first_last": [int(zero_rows[0]), int(zero_rows[-1])] if zero_rows.size else [],
        "results": {},
        "loop_topology": (
            "Each cell has Lw (left-to-wire), Lm (wire-to-right), Lpar "
            "(wire-to-branch), and JJ (branch-to-right). The built RF branch "
            "therefore contains Lpar in series with the Josephson branch and "
            "is parallel to Lm; it is not a standalone JJ-to-ground shunt."
        ),
    }
    _write(payload)

    baseline = _measure_linear_limit(device, spec, pump_hz, 0.01, implicit_linear_stiffness=True)
    payload["results"]["Lpar_x1"] = baseline
    _write(payload)

    mutated, count = _mutated_device(device, 10.0)
    payload["mutated_lpar_count"] = int(count)
    payload["results"]["Lpar_x10"] = _measure_linear_limit(
        mutated, spec, pump_hz, 0.01, implicit_linear_stiffness=True,
    )
    payload["relative_error_change"] = (
        payload["results"]["Lpar_x10"]["relative_error"]
        - payload["results"]["Lpar_x1"]["relative_error"]
    )
    _write(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
