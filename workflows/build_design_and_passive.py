"""Build an IPM design and generate its pump-off S-parameter plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from twpa_solver.circuit import Circuit
from twpa_solver.builders import ipm
from twpa_solver.core.circuit import CircuitMatrices, load_circuit
from twpa_solver.design.__main__ import main as compile_design_main
from twpa_solver.signal.passive import db20, passive_s_matrix, passive_s_matrix_from_circuit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="All unrecognised options are forwarded to the IPM builder.",
    )
    parser.add_argument(
        "--output-dir", "--design-dir", dest="design_dir", type=Path,
        help="Output root; batched designs are written to one subdirectory each",
    )
    parser.add_argument(
        "--design", nargs="+", type=Path,
        help="One or more declarative YAML design inputs",
    )
    parser.add_argument("--passive-start-ghz", type=float, default=4.0)
    parser.add_argument("--passive-stop-ghz", type=float, default=11.0)
    parser.add_argument("--passive-points", type=int, default=1401)
    parser.add_argument("--passive-z0-ohm", type=float, default=50.0)
    return parser


def _write_passive(design_dir: Path, args: argparse.Namespace) -> None:
    if args.passive_points < 2 or args.passive_stop_ghz <= args.passive_start_ghz:
        raise ValueError("passive frequency range must be increasing and have at least 2 points")
    freq_ghz = np.linspace(
        args.passive_start_ghz, args.passive_stop_ghz, args.passive_points
    )
    ports = tuple(sorted(load_circuit(design_dir).port_to_index))
    matrix = passive_s_matrix(
        design_dir, freq_ghz * 1e9, ports=ports, z0_ohm=args.passive_z0_ohm
    )
    if ports == (1, 2, 3, 4):
        names = ("s11", "s21", "s31", "s41", "s14", "s24", "s34")
        pairs = ((1, 1), (2, 1), (3, 1), (4, 1), (1, 4), (2, 4), (3, 4))
    else:
        pairs = [(out, source) for source in ports for out in ports]
        names = tuple(f"s{out}{source}" for out, source in pairs)
    traces = {
        name: db20(matrix[:, out - 1, source - 1])
        for name, (out, source) in zip(names, pairs)
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np.savez_compressed(design_dir / "passive_sparameters.npz", freq_ghz=freq_ghz, **{
        f"{name}_db": values for name, values in traces.items()
    })

    def save(names_to_plot: tuple[str, ...], stem: str, title: str) -> None:
        fig, axes = plt.subplots(len(names_to_plot), 1, figsize=(11, 3.2 * len(names_to_plot)), sharex=True)
        for axis, name in zip(np.atleast_1d(axes), names_to_plot):
            axis.plot(freq_ghz, traces[name], lw=1.2)
            axis.set_ylabel(f"|{name.upper()}| (dB)")
            axis.grid(True, alpha=0.3)
        np.atleast_1d(axes)[-1].set_xlabel("Signal frequency (GHz)")
        fig.suptitle(title)
        fig.tight_layout()
        for suffix, kwargs in (("png", {"dpi": 200}), ("pdf", {}), ("svg", {})):
            fig.savefig(design_dir / f"{stem}.{suffix}", **kwargs)
        plt.close(fig)

    preferred = tuple(name for name in ("s21", "s24") if name in traces)
    all_input = tuple(name for name in ("s11", "s21", "s31", "s41") if name in traces)
    save(preferred or names[:1], "passive_s21_s24", f"Pump-off response: {design_dir.name}")
    save(all_input or names[:1], "passive_s11_s21_s31_s41",
         f"Pump-off response from port 1: {design_dir.name}")


def _coupler_settings(
    design_dir: Path,
    mode_override: str | None = None,
) -> dict[str, object]:
    """Read the coupler settings emitted by either supported build path."""

    summary_path = design_dir / "ipm_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        params = dict(summary.get("params", {}))
        geometry = dict(summary.get("coupler", {}).get("geometry", {}))
        if params:
            inferred_mode = "ideal" if float(geometry.get("width_um", 1.0)) == 0.0 else "auto"
            return {
                "coupling_db": float(params.get("coupling_dB", -14.0)),
                "frequency_hz": float(params.get("coupler_freq_hz", 8.0e9)),
                "z0_ohm": float(params.get("Z0", 50.0)),
                "cell_length_um": float(params.get("cell_length_um", 10.0)),
                "mode": mode_override or inferred_mode,
                "model": str(geometry.get("model", "auto")),
            }

    resolved_path = design_dir / "design_resolved.json"
    if resolved_path.exists():
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        params = dict(resolved.get("parameters", {}))
        if params:
            coupler_settings = dict(resolved.get("coupler_settings", {}))
            return {
                "coupling_db": float(coupler_settings.get(
                    "coupling_dB", params.get("coupling_dB", -14.0)
                )),
                "frequency_hz": float(coupler_settings.get(
                    "coupler_freq_hz", params.get("coupler_freq_hz", 8.0e9)
                )),
                "z0_ohm": float(coupler_settings.get(
                    "Z0", params.get("Z0", 50.0)
                )),
                "cell_length_um": float(coupler_settings.get(
                    "cell_length_um", params.get("cell_length_um", 10.0)
                )),
                "mode": str(resolved.get("coupler_mode", "auto")),
                "model": str(resolved.get("coupler_geometry", {}).get("model", "auto")),
            }

    raise FileNotFoundError(
        f"cannot determine coupler settings from {design_dir}; "
        "expected ipm_summary.json or design_resolved.json"
    )


def _standalone_coupler_matrices(settings: dict[str, object]) -> CircuitMatrices:
    """Build the isolated four-port coupler using the normal circuit builder."""

    circuit = Circuit("standalone_coupler")
    signal = circuit.path("signal")
    pump = circuit.path("pump")
    handle = circuit.add_directional_coupler(
        signal,
        pump,
        coupling_db=float(settings["coupling_db"]),
        frequency=float(settings["frequency_hz"]),
        z0=float(settings["z0_ohm"]),
        mode=str(settings["mode"]),
        cell_length_um=float(settings["cell_length_um"]),
        name="coupler",
    )
    circuit.add_port(handle.signal_in, number=1, impedance=float(settings["z0_ohm"]))
    circuit.add_port(handle.signal_out, number=2, impedance=float(settings["z0_ohm"]))
    circuit.add_port(handle.pump_in, number=3, impedance=float(settings["z0_ohm"]))
    circuit.add_port(handle.pump_out, number=4, impedance=float(settings["z0_ohm"]))
    for index, node in enumerate(
        (handle.signal_in, handle.signal_out, handle.pump_in, handle.pump_out),
        start=1,
    ):
        circuit.add_resistor(
            node,
            circuit.ground,
            float(settings["z0_ohm"]),
            name=f"Rport{index}",
        )

    compiled = circuit.compile()
    matrices = compiled.matrices()
    return CircuitMatrices(
        C=matrices["C"],
        G=matrices["G"],
        K=matrices["K"],
        Bphi=matrices["Bphi"],
        Ic=matrices["Ic"],
        Lj=matrices["Lj"],
        nodes=matrices["nodes"],
        port_to_index=matrices["port_vectors"],
        metadata={"name": "standalone_coupler", "settings": settings},
    )


def _write_coupler_passive(
    design_dir: Path,
    args: argparse.Namespace,
    mode_override: str | None = None,
) -> None:
    """Write the isolated coupler four-port S-parameter plot and data."""

    settings = _coupler_settings(design_dir, mode_override=mode_override)
    freq_ghz = np.linspace(
        args.passive_start_ghz, args.passive_stop_ghz, args.passive_points
    )
    matrix = passive_s_matrix_from_circuit(
        _standalone_coupler_matrices(settings),
        freq_ghz * 1e9,
        ports=(1, 2, 3, 4),
        z0_ohm=float(settings["z0_ohm"]),
    )
    names = ("s11", "s21", "s31", "s41")
    traces = {
        name: db20(matrix[:, output - 1, 0])
        for name, output in zip(names, (1, 2, 3, 4))
    }
    np.savez_compressed(
        design_dir / "coupler_passive_sparameters.npz",
        freq_ghz=freq_ghz,
        **{f"{name}_db": values for name, values in traces.items()},
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 12.8), sharex=True)
    for axis, name in zip(np.atleast_1d(axes), names):
        axis.plot(freq_ghz, traces[name], lw=1.2)
        axis.set_ylabel(f"|{name.upper()}| (dB)")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Signal frequency (GHz)")
    fig.suptitle(f"Pump-off isolated coupler response: {design_dir.name}")
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 200}), ("pdf", {}), ("svg", {})):
        fig.savefig(design_dir / f"coupler_passive_s11_s21_s31_s41.{suffix}", **kwargs)
    plt.close(fig)


def _design_contains_coupler(design_dir: Path) -> bool:
    """Return whether the resolved design contains a directional coupler."""

    resolved_path = design_dir / "design_resolved.json"
    if not resolved_path.exists():
        # Legacy builder output does not carry the declarative block list.
        return True
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    blocks = resolved.get("blocks")
    if not isinstance(blocks, list):
        return True
    return any(
        isinstance(block, dict)
        and block.get("type") in {"coupler", "directional_coupler"}
        for block in blocks
    )


def _job_directories(
    design_paths: list[Path], design_root: Path,
) -> list[tuple[Path, Path]]:
    """Return stable, non-overlapping output directories for a design batch."""

    names = [path.stem for path in design_paths]
    if len(set(names)) != len(names):
        raise ValueError(
            "multiple designs have the same filename stem; use unique YAML names"
        )
    return [(path, design_root / name) for path, name in zip(design_paths, names)]


def _build_one(
    design: Path | None,
    design_dir: Path,
    workflow_args: argparse.Namespace,
    builder_args: list[str],
) -> None:
    """Build one design and generate all passive artifacts before returning."""

    job_builder_args = list(builder_args)
    if design is not None:
        job_builder_args.extend([
            "--design", str(design),
            "--outdir", str(design_dir),
            "--write-matrices",
        ])
        compile_design_main(job_builder_args)
    else:
        job_builder_args.extend([
            "--outdir", str(design_dir),
            "--write-matrices",
        ])
        ipm.main(job_builder_args)

    forwarded_coupler_mode = None
    for index, argument in enumerate(job_builder_args):
        if argument == "--coupler-mode" and index + 1 < len(job_builder_args):
            forwarded_coupler_mode = job_builder_args[index + 1]
        elif argument.startswith("--coupler-mode="):
            forwarded_coupler_mode = argument.split("=", 1)[1]
    _write_passive(design_dir, workflow_args)
    if _design_contains_coupler(design_dir):
        _write_coupler_passive(
            design_dir,
            workflow_args,
            mode_override=forwarded_coupler_mode,
        )
    print(f"wrote design and passive plots to {design_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    workflow_args, builder_args = parser.parse_known_args(argv)
    if workflow_args.design is None and workflow_args.design_dir is None:
        parser.error("one of --design or --design-dir is required")
    if workflow_args.design is not None and workflow_args.design_dir is None:
        parser.error("--design-dir is required with --design")

    if workflow_args.design is not None:
        design_paths = list(workflow_args.design)
        if len(design_paths) == 1:
            jobs = [(design_paths[0], workflow_args.design_dir)]
        else:
            try:
                jobs = _job_directories(design_paths, workflow_args.design_dir)
            except ValueError as exc:
                parser.error(str(exc))
    else:
        jobs = [(None, workflow_args.design_dir)]

    for design, design_dir in jobs:
        _build_one(design, design_dir, workflow_args, builder_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
