from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from twpa_solver.builders.ipm import (IPMParams, LossSpec, build_component_plan,
                                       build_matrices)
from twpa_solver.builders.profiles import parse_profile_json, parse_profile_shorthand
from twpa_solver.builders.scatter import ScatterSpec
from twpa_solver.design import compile_design, load_design
from twpa_solver.design.parameters import resolve_parameters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile a declarative circuit design")
    parser.add_argument("--design", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--write-matrices", action="store_true")
    parser.add_argument("--coupler-mode", default=None,
                        choices=("auto", "cached", "ideal", "optimize"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--profile-json")
    parser.add_argument("--lj-profile", action="append", default=[])
    parser.add_argument("--cg-profile", action="append", default=[])
    parser.add_argument("--lj-scatter-sigma", type=float, default=0.0)
    parser.add_argument("--cj-scatter-sigma", type=float, default=0.0)
    parser.add_argument("--cj-scatter-mode", choices=("independent", "plasma_locked"), default="independent")
    parser.add_argument("--cg-scatter-sigma", type=float, default=0.0)
    parser.add_argument("--scatter-seed", type=int, default=1)
    parser.add_argument("--tan-delta", type=float, default=0.0)
    parser.add_argument("--tan-delta-role", action="append", default=[])
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {outdir}; use --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)
    source = load_design(args.design)
    parameters = dict(source.get("parameters", {}))
    parameters = resolve_parameters(parameters, parameters, "parameters")
    plan = None
    if (args.profile_json or args.lj_profile or args.cg_profile or
            args.lj_scatter_sigma or args.cj_scatter_sigma or args.cg_scatter_sigma):
        json_segments = (parse_profile_json(Path(args.profile_json))
                          if args.profile_json else {"Lj": [], "Cg": []})
        params = IPMParams(**{key: value for key, value in parameters.items()
                              if key in IPMParams.__dataclass_fields__})
        plan = build_component_plan(
            params,
            lj_segments=json_segments["Lj"] + [parse_profile_shorthand(text)
                                                for text in args.lj_profile],
            cg_segments=json_segments["Cg"] + [parse_profile_shorthand(text)
                                                for text in args.cg_profile],
            lj_scatter=ScatterSpec(args.lj_scatter_sigma),
            cj_scatter=ScatterSpec(args.cj_scatter_sigma, mode=args.cj_scatter_mode),
            cg_scatter=ScatterSpec(args.cg_scatter_sigma),
            seed=args.scatter_seed,
        )
    design = compile_design(source, coupler_mode=args.coupler_mode,
                            strict=args.strict, plan=plan)
    effective_plan = design.component_plan
    with (outdir / "elements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("idx", "name", "node1", "node2", "value", "kind", "role", "cell_index"))
        for index, element in enumerate(design.elements):
            writer.writerow((index, element.name, element.n1, element.n2, element.value,
                             element.kind, element.role, element.cell_index))
    with (outdir / "ports.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("port", "node"))
        writer.writerows((number, record.node) for number, record in design.ports.items())
    role_overrides: dict[str, float] = {}
    for item in args.tan_delta_role:
        role, separator, value = item.partition("=")
        if not separator or not role:
            raise ValueError(f"invalid --tan-delta-role {item!r}; expected ROLE=VALUE")
        role_overrides[role] = float(value)
    summary = {
        "name": design.name, "elements": len(design.elements),
        "cursors": design.cursors, "ports": {str(k): v.node for k, v in design.ports.items()},
        "loss": {"default": args.tan_delta, "by_role": role_overrides},
    }
    if effective_plan is not None:
        summary["component_plan"] = effective_plan.metadata
    (outdir / "design_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    resolved = {
        **design.metadata, "name": design.name, "cursors": design.cursors,
        "blocks": [record.__dict__ for record in design.blocks],
        "named_nodes": design.named_nodes,
        "named_elements": design.named_elements,
        "elements": [element.__dict__ for element in design.elements],
    }
    if effective_plan is not None:
        resolved["component_plan"] = effective_plan.metadata
    (outdir / "design_resolved.json").write_text(
        json.dumps(resolved, indent=2, default=str), encoding="utf-8")
    if args.write_matrices:
        matrices = build_matrices(design.elements,
                                  LossSpec(args.tan_delta, role_overrides))
        for name in ("C", "G", "K", "Bphi"):
            sp.save_npz(outdir / f"{name}.npz", matrices[name])
        arrays = {key: matrices[key] for key in ("nodes", "Ic", "Lj")}
        if effective_plan is not None:
            arrays.update({"Cj": effective_plan.cj, "Cg": effective_plan.cg,
                           "Lj_nominal": effective_plan.lj_nominal,
                           "Cj_nominal": effective_plan.cj_nominal,
                           "Cg_nominal": effective_plan.cg_nominal,
                           "cell_index": np.arange(effective_plan.lj.size,
                                                    dtype=np.int64)})
        arrays.update({"phi0_reduced": np.array([2.067833848e-15 / (2 * np.pi)]),
                       "port_numbers": np.array(sorted(matrices["ports"])),
                       "port_indices": np.array([matrices["port_vectors"][p]
                                                   for p in sorted(matrices["ports"])]),})
        np.savez(outdir / "arrays.npz", **arrays)
        np.savez(outdir / "ipm_arrays.npz", **arrays)


if __name__ == "__main__":
    main()
