"""Measure pump-off standing-wave content and map the 2c topology."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core.circuit import load_circuit
from twpa_solver.core.linear import dynamic_block


OUTPUT = ROOT / "outputs" / "exp27_track1_standing_wave"
FREQUENCY_GHZ = 7.629
OMEGA = 2.0 * math.pi * FREQUENCY_GHZ * 1e9
Z_LINE = math.sqrt(123.9e-12 / 17.3e-15)


def branch_endpoints(circuit: object) -> tuple[np.ndarray, np.ndarray]:
    incidence = circuit.Bphi.tocsc()
    starts = []
    stops = []
    for branch in range(circuit.branch_count):
        nodes = incidence.indices[incidence.indptr[branch] : incidence.indptr[branch + 1]]
        if len(nodes) != 2:
            raise ValueError(f"branch {branch} has {len(nodes)} endpoints")
        starts.append(int(min(nodes)))
        stops.append(int(max(nodes)))
    return np.asarray(starts), np.asarray(stops)


def solve_pump_off(circuit: object) -> np.ndarray:
    gamma = circuit.Ic / circuit.phi0
    extra_k = (circuit.Bphi @ sp.diags(gamma) @ circuit.Bphi.T).astype(np.complex128).tocsr()
    rhs = np.zeros(circuit.node_count, dtype=np.complex128)
    rhs[circuit.port_to_index[1]] = 1.0
    return spla.spsolve(dynamic_block(circuit, OMEGA, extra_K=extra_k), rhs)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    circuit = load_circuit(ROOT / "outputs" / "ipm_python_design")
    starts, stops = branch_endpoints(circuit)
    state = solve_pump_off(circuit)
    voltage = 1j * OMEGA * state
    branch_flux = np.asarray(circuit.Bphi.T @ state).reshape(-1)
    branch_current = (circuit.Ic / circuit.phi0) * branch_flux
    branch_voltage = 0.5 * (voltage[starts] + voltage[stops])
    v_plus = 0.5 * (branch_voltage + Z_LINE * branch_current)
    v_minus = 0.5 * (branch_voltage - Z_LINE * branch_current)
    ratio = np.abs(v_minus) / np.maximum(np.abs(v_plus), 1e-300)

    row_length = int(circuit.metadata.get("params", {}).get("array_length", 418))
    row_index = np.arange(circuit.branch_count) // row_length
    row_position = np.arange(circuit.branch_count) % row_length
    port_map = {str(port): int(index) for port, index in circuit.port_to_index.items()}
    port_branch_distance = {
        str(port): {
            "node_index": int(index),
            "nearest_branch_start": int(np.argmin(np.abs(starts - index))),
            "nearest_branch_stop": int(np.argmin(np.abs(stops - index))),
            "is_josephson_branch_node": bool(np.any(starts == index) or np.any(stops == index)),
        }
        for port, index in circuit.port_to_index.items()
    }

    lj = float(np.median(circuit.Lj)) if circuit.Lj is not None else 123.9e-12
    cg_diag = np.asarray(circuit.C.diagonal()).real
    cg_median = float(np.median(cg_diag))
    branch_degree = np.asarray(np.abs(circuit.Bphi).sum(axis=1)).reshape(-1)
    junction_cap = float(circuit.metadata.get("params", {}).get("Cj", 145e-15))
    ground_cap_on_branch_nodes = cg_diag - branch_degree * junction_cap
    ground_cap_on_branch_nodes = ground_cap_on_branch_nodes[branch_degree > 0.0]
    path_cg_median = float(np.median(ground_cap_on_branch_nodes))
    tau_cell = math.sqrt(lj * cg_median)
    tau_line = circuit.branch_count * tau_cell
    path_z_line = math.sqrt(lj / path_cg_median)
    path_tau_line = circuit.branch_count * math.sqrt(lj * path_cg_median)
    delay = {
        "lj_median_h": lj,
        "cg_diagonal_median_f": cg_median,
        "junction_capacitance_f": junction_cap,
        "ground_capacitance_on_branch_nodes_median_f": path_cg_median,
        "z_line_from_branch_node_ground_cap_ohm": path_z_line,
        "n_branches": circuit.branch_count,
        "tau_cell_s": tau_cell,
        "tau_one_way_s": tau_line,
        "fsr_one_over_tau_ghz": 1.0 / tau_line / 1e9,
        "fsr_one_over_2tau_ghz": 1.0 / (2.0 * tau_line) / 1e9,
        "path_tau_one_way_s": path_tau_line,
        "path_fsr_one_over_tau_ghz": 1.0 / path_tau_line / 1e9,
        "path_fsr_one_over_2tau_ghz": 1.0 / (2.0 * path_tau_line) / 1e9,
        "measured_ripple_period_ghz": 0.1843157894736639,
    }

    rows = []
    for index in range(circuit.branch_count):
        rows.append(
            {
                "branch_index": int(index),
                "row_index": int(row_index[index]),
                "row_position": int(row_position[index]),
                "start_node": int(starts[index]),
                "stop_node": int(stops[index]),
                "v_plus_abs": float(abs(v_plus[index])),
                "v_minus_abs": float(abs(v_minus[index])),
                "backward_forward_ratio": float(ratio[index]),
            }
        )
    with (OUTPUT / "standing_wave_branches.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    report = {
        "frequency_ghz": FREQUENCY_GHZ,
        "z_line_ohm": Z_LINE,
        "backward_forward_ratio_min": float(np.min(ratio)),
        "backward_forward_ratio_median": float(np.median(ratio)),
        "backward_forward_ratio_mean": float(np.mean(ratio)),
        "backward_forward_ratio_max": float(np.max(ratio)),
        "backward_forward_ratio_quantiles": [float(value) for value in np.quantile(ratio, [0.1, 0.5, 0.9])],
        "branch_rows": int(np.max(row_index) + 1),
        "row_length": row_length,
        "branch_1254": rows[1254],
        "delay": delay,
        "port_map": port_map,
        "port_branch_distance": port_branch_distance,
        "chain_start_node": int(starts[0]),
        "chain_end_node": int(stops[-1]),
    }
    with (OUTPUT / "standing_wave_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(np.arange(circuit.branch_count), ratio, linewidth=0.6)
    axes[0].axvline(1254, color="tab:red", linestyle="--", label="branch 1254")
    axes[0].set_ylabel(r"$|V_-|/|V_+|$")
    axes[0].legend()
    axes[1].plot(row_position, ratio, linewidth=0.5, alpha=0.7)
    axes[1].set_xlabel("Position within each 418-branch row")
    axes[1].set_ylabel(r"$|V_-|/|V_+|$")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle("2c pump-off forward/backward wave decomposition at 7.629 GHz")
    figure.tight_layout()
    figure.savefig(OUTPUT / "standing_wave_decomposition.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
