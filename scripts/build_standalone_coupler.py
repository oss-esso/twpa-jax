#!/usr/bin/env python3
"""Build an isolated four-port coupler from an existing IPM design.

The source IPM supplies the realized coupler geometry. The generated circuit
contains only that coupler, with ports 1/3 on the left terminals and ports 2/4
on the right terminals, each terminated by the source design's 50-ohm port
resistance.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from exp07_python_ipm_design_builder import (  # noqa: E402
    CouplerDiscrete,
    CouplerGeometry,
    Element,
    IPMParams,
    add,
    add_edge_coupled_directional_coupler,
    build_matrices,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-ipm-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.source_ipm_dir / "ipm_summary.json").read_text())
    source_params = summary["params"]
    param_names = {f.name for f in fields(IPMParams)}
    params = replace(IPMParams(), **{k: v for k, v in source_params.items() if k in param_names})

    geometry = CouplerGeometry(**summary["coupler"]["geometry"])
    coupler = CouplerDiscrete(
        L_cell=summary["coupler"]["L_cell"],
        Cc_cell=summary["coupler"]["Cc_cell"],
        C_gnd_cell=summary["coupler"]["C_gnd_cell"],
        K_ind=summary["coupler"]["K_ind"],
        N_coupled=summary["coupler"]["N_coupled"],
        N_uncoupled=summary["coupler"]["N_uncoupled"],
        geometry=geometry,
    )

    ground = params.ground
    top_left = params.start_node_top
    bottom_left = params.start_node_bot
    circuit: list[Element] = []
    for port, node, label in ((1, top_left, "top_left"), (3, bottom_left, "bottom_left")):
        add(circuit, f"P{label}", node, ground, port, "port")
        add(circuit, f"R{label}", node, ground, params.Z0, "resistor")

    top_right, bottom_right = add_edge_coupled_directional_coupler(
        circuit, top_left, bottom_left, ground, coupler
    )
    for port, node, label in ((2, top_right, "top_right"), (4, bottom_right, "bottom_right")):
        add(circuit, f"P{label}", node, ground, port, "port")
        add(circuit, f"R{label}", node, ground, params.Z0, "resistor")

    mats = build_matrices(circuit)
    ends = {
        "top_left_node": top_left,
        "bottom_left_node": bottom_left,
        "top_right_node": top_right,
        "bottom_right_node": bottom_right,
        "source_ipm_dir": str(args.source_ipm_dir),
    }
    extra = {
        "standalone_coupler": True,
        "source_ipm_dir": str(args.source_ipm_dir),
    }
    write_outputs(str(args.outdir), circuit, params, coupler, ends, mats, extra_summary=extra)
    print(
        f"wrote={args.outdir} K_dB={geometry.k_db:.9f} "
        f"Zin_ohm={geometry.z_input_ohm:.9f} N_coupled={coupler.N_coupled}"
    )


if __name__ == "__main__":
    main()
