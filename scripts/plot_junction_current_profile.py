"""Plot the instantaneous current in every Josephson junction of a pump state.

The input pump directory must contain the current production
``pump_solution.npz`` and ``pump_report.json``.  The circuit is rebuilt from
the persisted harmonic metadata before the current profile is evaluated.
This makes the plot an independent reconstruction of the stored HB state,
not a plot of the scalar maximum recorded by the map runner.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twpa_solver.core import default_loss_model_for, load_circuit
from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.pump.diagnostics import branch_current_profile
from twpa_solver.pump.problem import FullPumpProblem, HarmonicGrid


def _read_metadata(pump_dir: Path) -> dict:
    with (pump_dir / "pump_report.json").open("r", encoding="utf-8") as f:
        report = json.load(f)
    metadata = report.get("metadata", {})
    if report.get("final_status") != "VALID_CONVERGED":
        raise ValueError(
            f"pump state is not a validated production solution: "
            f"{report.get('final_status')!r}"
        )
    return metadata


def _element_labels(circuit_dir: Path, count: int) -> list[dict[str, str]]:
    path = circuit_dir / "elements.csv"
    if not path.exists():
        return [
            {"name": f"JJ_{i}", "node1": "", "node2": ""}
            for i in range(count)
        ]
    labels: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("kind", "").lower() == "josephson_inductor":
                labels.append({
                    "name": row.get("name", f"JJ_{len(labels)}"),
                    "node1": row.get("node1", ""),
                    "node2": row.get("node2", ""),
                })
    if len(labels) != count:
        raise ValueError(
            f"elements.csv contains {len(labels)} Josephson branches, "
            f"but the solution contains {count} branches"
        )
    return labels


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pump-dir", type=Path, required=True)
    p.add_argument("--circuit-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pump_dir = args.pump_dir
    metadata = _read_metadata(pump_dir)
    circuit_raw = metadata.get("circuit_dir")
    circuit_dir = args.circuit_dir or (Path(circuit_raw) if circuit_raw else None)
    if circuit_dir is None:
        raise ValueError("--circuit-dir is required when report metadata has no circuit_dir")
    circuit = load_circuit(circuit_dir)

    with np.load(pump_dir / "pump_solution.npz") as data:
        X = np.asarray(data["X_real"], dtype=np.float64) + 1j * np.asarray(
            data["X_imag"], dtype=np.float64
        )
        modes = np.asarray(
            metadata.get("pump_modes", data.get("pump_modes")), dtype=int
        ).reshape(-1)
    if modes.size != X.shape[0]:
        raise ValueError(f"solution rows {X.shape[0]} do not match pump modes {modes.size}")

    frequency_ghz = float(metadata["pump_freq_ghz"])
    omega = float(metadata.get("omega_p", 2.0 * np.pi * frequency_ghz * 1e9))
    nt = int(metadata.get("nt", max(40, 2 * int(np.max(modes)) + 4)))
    grid = HarmonicGrid(modes=modes, nt=nt, omega=omega)
    dc_flux = np.asarray(
        metadata.get("dc_branch_flux", np.zeros(circuit.branch_count)),
        dtype=np.float64,
    )
    pump_port = int(metadata.get("pump_port", 4))
    pump_node = circuit.port_to_index[pump_port]
    problem = FullPumpProblem(
        C=circuit.C,
        G=circuit.G,
        K=circuit.K,
        Bphi=circuit.Bphi,
        branch=make_branch_law(circuit),
        grid=grid,
        pump_node_index=pump_node,
        pump_current_a=float(metadata["pump_current_a"]),
        source_mode=1,
        loss_model=default_loss_model_for(circuit),
        dc_branch_flux=dc_flux,
    )
    profile = branch_current_profile(problem, X)
    # Reconstruct phase from the same HB waveform and persisted DC branch
    # flux.  Omitting the DC term would report the dynamic phase only and can
    # materially under-report the biased junction phase.
    dynamic_flux = np.asarray(problem.branch_flux_time(X), dtype=np.float64)
    dc_flux = np.asarray(problem.dc_branch_flux, dtype=np.float64).reshape(1, -1)
    phase_rad = (dynamic_flux + dc_flux) / float(problem.branch.phi0)
    peak_abs_phase_rad = np.max(np.abs(phase_rad), axis=0)
    rms_phase_rad = np.sqrt(np.mean(phase_rad**2, axis=0))
    mean_phase_rad = np.mean(phase_rad, axis=0)
    labels = _element_labels(circuit_dir, circuit.branch_count)
    outdir = args.outdir or pump_dir / "junction_current_profile"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "junction_current_profile.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "junction_index", "element_name", "node1", "node2",
            "peak_abs_current_a", "rms_current_a", "mean_current_a",
            "critical_current_a", "peak_ratio_ic", "peak_time_index",
            "peak_abs_phase_rad", "rms_phase_rad", "mean_phase_rad",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(circuit.branch_count):
            writer.writerow({
                "junction_index": i,
                "element_name": labels[i]["name"],
                "node1": labels[i]["node1"],
                "node2": labels[i]["node2"],
                **{key: float(profile[key][i]) for key in (
                    "peak_abs_current_a", "rms_current_a", "mean_current_a",
                    "critical_current_a", "peak_ratio_ic",
                )},
                "peak_time_index": int(profile["peak_time_index"][i]),
                "peak_abs_phase_rad": float(peak_abs_phase_rad[i]),
                "rms_phase_rad": float(rms_phase_rad[i]),
                "mean_phase_rad": float(mean_phase_rad[i]),
            })

    ratio = profile["peak_ratio_ic"]
    current_ua = profile["peak_abs_current_a"] * 1e6
    x = np.arange(circuit.branch_count)
    strongest = int(np.nanargmax(ratio))
    phase_strongest = int(np.nanargmax(peak_abs_phase_rad))
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    axes[0].plot(x, current_ua, linewidth=0.65, color="tab:blue")
    axes[0].scatter([strongest], [current_ua[strongest]], color="tab:red", zorder=3)
    axes[0].set_ylabel("Peak |junction current| (µA)")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(x, ratio, linewidth=0.65, color="tab:purple")
    axes[1].axhline(1.0, color="tab:red", linestyle="--", linewidth=1.0, label="Ic")
    axes[1].scatter([strongest], [ratio[strongest]], color="tab:red", zorder=3)
    axes[1].set_xlabel("Josephson junction index along stored branch order")
    axes[1].set_ylabel("Peak |I| / Ic")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[2].plot(
        x, peak_abs_phase_rad, linewidth=0.65, color="tab:green",
        label="peak |phase|",
    )
    axes[2].plot(
        x, rms_phase_rad, linewidth=0.55, color="tab:orange",
        alpha=0.85, label="RMS phase",
    )
    axes[2].scatter(
        [phase_strongest], [peak_abs_phase_rad[phase_strongest]],
        color="tab:red", zorder=3,
    )
    axes[2].set_xlabel("Josephson junction index along stored branch order")
    axes[2].set_ylabel("Junction phase (rad)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")
    fig.suptitle(
        f"2c pump junction current and phase profile: {frequency_ghz:.6g} GHz, "
        f"{metadata.get('pump_power_dbm_requested', '?')} dBm; "
        f"max I/Ic branch {strongest} ({ratio[strongest]:.6g}), "
        f"max |phase| branch {phase_strongest} "
        f"({peak_abs_phase_rad[phase_strongest]:.6g} rad)"
    )
    plot_path = outdir / "junction_current_profile.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "pump_dir": str(pump_dir),
        "circuit_dir": str(circuit_dir),
        "pump_freq_ghz": frequency_ghz,
        "pump_power_dbm": metadata.get("pump_power_dbm_requested"),
        "junction_count": circuit.branch_count,
        "strongest_junction_index": strongest,
        "strongest_junction_name": labels[strongest]["name"],
        "max_peak_current_a": float(np.max(profile["peak_abs_current_a"])),
        "max_peak_ratio_ic": float(ratio[strongest]),
        "max_peak_abs_phase_rad": float(peak_abs_phase_rad[phase_strongest]),
        "max_rms_phase_rad": float(np.max(rms_phase_rad)),
        "phase_definition": "(dynamic branch flux + persisted dc branch flux) / phi0",
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    with (outdir / "junction_current_profile.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
