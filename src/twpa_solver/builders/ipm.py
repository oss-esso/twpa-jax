
# experiments/exp07_python_ipm_design_builder.py
"""
Experiment 07: standalone Python IPM design builder.

No Julia. No JosephsonCircuits.

This ports the uploaded Julia topology generator into Python:

    Transmission_line_block.jl
    directional_coupler_block.jl
    CPW_Theory.jl
    IPM.jl
    IPM_JTWPA.jl

Outputs:
    outputs/ipm_python_design/ipm_elements.csv
    outputs/ipm_python_design/ipm_ports.csv
    outputs/ipm_python_design/ipm_summary.json

Optional matrix outputs:
    outputs/ipm_python_design/C.npz
    outputs/ipm_python_design/G.npz
    outputs/ipm_python_design/K.npz
    outputs/ipm_python_design/Bphi.npz
    outputs/ipm_python_design/ipm_arrays.npz

Matrix convention:
    C xddot + G xdot + K_lin x + Bphi i_J(Bphi.T x) = i_src

    Josephson branches:
        Lj elements are not stamped into K_lin.
        They become nonlinear Bphi branches with Ic = phi0_reduced / Lj.
        Their Cj capacitance is stamped into C.

    Linear inductors:
        ordinary L branches stamp 1/L into K_lin.

    Coupled inductors:
        K elements reference two L branches.
        The pair stamps B Lpair^{-1} B.T into K_lin instead of independent 1/L stamps.

Run:
    python experiments/exp07_python_ipm_design_builder.py --write-matrices --draw

Fast default:
    --coupler-mode cached

More literal CPW reverse optimization:
    --coupler-mode optimize
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


C_LIGHT = 299_792_458.0
EPS0 = 8.8541878128e-12
PHI0_REDUCED = 2.067833848e-15 / (2.0 * math.pi)


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class Element:
    name: str
    n1: Any
    n2: Any
    value: float | int | str
    kind: str


@dataclass
class CouplerGeometry:
    width_um: float
    gap_between_lines_um: float
    gap_to_ground_um: float
    length_um: float
    k_db: float
    z_input_ohm: float


@dataclass
class CouplerDiscrete:
    L_cell: float
    Cc_cell: float
    C_gnd_cell: float
    K_ind: float
    N_coupled: int
    N_uncoupled: int
    geometry: CouplerGeometry


@dataclass
class IPMParams:
    start_node_top: int = 1
    start_node_bot: int = 10000
    ground: int = 0

    array_length: int = 418
    num_rows: int = 6
    arrays_per_dc: int = 3

    coupling_dB: float = -14.0
    Z0: float = 50.0
    coupler_freq_hz: float = 8.0e9

    length_of_long_TL: int = 250
    length_of_short_TL: int = 30
    coupler_section_length: int = 1500
    len1: int = 100
    len2: int = 50
    len3: int = 100
    len4: int = 300

    Lj: float = 123.9e-12
    Cj: float = 145.0e-15
    Cg: float = 66.0e-15
    Cl: float = 10 * 1.73e-15
    Ll: float = 10 * 4.13e-12

    Rleft: float = 50.0
    Rright: float = 50.0
    Rm: float = 50.0

    cell_length_um: float = 10.0

    # Cached geometry close to the previous optimized design printout.
    # Use --coupler-mode optimize to recompute with scipy L-BFGS-B.
    cached_coupler_width_um: float = 39.897
    cached_coupler_gap_um: float = 44.762
    cached_coupler_gap_to_ground_um: float = 10.5973385055
    cached_coupler_length_um: float = 3787.7


# =============================================================================
# CPW / edge-coupled coupler theory
# =============================================================================

def elliptical_integral(k: float) -> float:
    if k < 0.0 or k >= 1.0:
        k = min(max(k, 0.0), 1.0 - 1e-15)

    a = 1.0
    b = math.sqrt(max(0.0, 1.0 - k * k))
    c_val = k
    K_int = math.pi / 2.0 / a

    while c_val > 1e-10:
        aN = (a + b) / 2.0
        bN = math.sqrt(a * b)
        cN = (a - b) / 2.0
        K_int = math.pi / 2.0 / aN
        a, b, c_val = aN, bN, cN

    return K_int


def elliptical_integral_prime(k: float) -> float:
    return elliptical_integral(math.sqrt(max(0.0, 1.0 - k * k)))


def edge_coupled_cpw(
    width: float,
    gap_to_ground: float,
    gap_between_lines: float,
    height: float = 525.0,
    eps: float = 11.9,
) -> dict[str, float]:
    S = width * 1e-6
    W = gap_to_ground * 1e-6
    d = gap_between_lines * 1e-6
    h = height * 1e-6

    r = d / (d + 2.0 * S)
    k1 = (d + 2.0 * S) / (d + 2.0 * S + 2.0 * W)
    delta = math.sqrt((1.0 - r * r) / (1.0 - k1 * k1 * r * r))

    C_ae = (
        2.0
        * EPS0
        * elliptical_integral(delta * k1)
        / elliptical_integral_prime(delta * k1)
    )

    r1 = math.sinh(math.pi * d / (4.0 * h)) / math.sinh(
        (math.pi / (2.0 * h)) * (d / 2.0 + S)
    )
    k2 = math.sinh((math.pi / (2.0 * h)) * (d / 2.0 + S)) / math.sinh(
        (math.pi / (2.0 * h)) * (d / 2.0 + S + W)
    )
    psi = math.sqrt((1.0 - r1 * r1) / (1.0 - k2 * k2 * r1 * r1))

    C_de = (
        EPS0
        * (eps - 1.0)
        * elliptical_integral(psi * k2)
        / elliptical_integral_prime(psi * k2)
    )

    C_e = C_ae + C_de
    eps_e = C_e / C_ae
    v_e = C_LIGHT / math.sqrt(eps_e)
    Z_e = 1.0 / (C_LIGHT * C_ae * math.sqrt(eps_e))

    C_ao = 2.0 * EPS0 * elliptical_integral(delta) / elliptical_integral_prime(delta)

    C_15 = math.sinh(math.pi / (2.0 * h) * (d / 2.0 + S + W)) ** 2
    C_14 = math.sinh(math.pi * d / (4.0 * h)) ** 2
    C_13 = math.sinh(math.pi / (2.0 * h) * (d / 2.0 + S)) ** 2

    C_12 = 0.5 * (
        math.sqrt(1.0 + C_15) / ((1.0 + C_13) * (1.0 + C_14)) ** 0.25
        - ((1.0 + C_13) * (1.0 + C_14)) ** 0.25 / math.sqrt(1.0 + C_15)
    )
    C_11 = 0.5 * (
        ((1.0 + C_13) / (1.0 + C_14)) ** 0.25
        - ((1.0 + C_14) / (1.0 + C_13)) ** 0.25
    )
    chi = -0.5 * (
        ((1.0 + C_13) * (1.0 + C_14)) ** 0.25
        - ((1.0 + C_13) * (1.0 + C_14)) ** (-0.25)
    )

    rad = (C_12 * C_12 / (C_11 * C_11) - 1.0) * (
        chi * chi / (C_11 * C_11) - 1.0
    )
    rad = max(rad, 0.0)

    kappa = 1.0 / (C_12 - chi) * (
        -1.0 - C_12 * chi / (C_11 * C_11) - math.sqrt(rad)
    )
    k3 = (C_11 * (1.0 + kappa * C_12)) / (
        C_12 + kappa * C_11 * C_11
    )

    C_do = (
        2.0
        * EPS0
        * (eps - 1.0)
        * elliptical_integral(k3)
        / elliptical_integral_prime(k3)
    )

    C_o = C_ao + C_do
    eps_o = C_o / C_ao
    v_o = C_LIGHT / math.sqrt(eps_o)
    Z_o = 1.0 / (C_LIGHT * C_ao * math.sqrt(eps_o))

    Z_input = math.sqrt(Z_e * Z_o)

    C_mutual = (C_o - C_e) / 2.0
    C_self = C_e

    L_e = Z_e * Z_e * C_e
    L_o = Z_o * Z_o * C_o
    L_mutual = (L_e - L_o) / 2.0
    L_self = (L_e + L_o) / 2.0

    return {
        "C_e": C_e,
        "C_o": C_o,
        "v_e": v_e,
        "v_o": v_o,
        "Z_e": Z_e,
        "Z_o": Z_o,
        "L_e": L_e,
        "L_o": L_o,
        "Z": Z_input,
        "C_self": C_self,
        "C_mutual": C_mutual,
        "L_self": L_self,
        "L_mutual": L_mutual,
    }


def estimate_edge_coupled_directional_coupler(
    width: float,
    gap_to_ground: float,
    gap_between_lines: float,
    freq: float = 8e9,
) -> dict[str, float]:
    res = edge_coupled_cpw(width, gap_to_ground, gap_between_lines)

    K_voltage = (res["Z_o"] - res["Z_e"]) / (res["Z_o"] + res["Z_e"])
    K_dB = 20.0 * math.log10(abs(K_voltage))

    beta_e = 2.0 * math.pi * freq / res["v_e"]
    beta_o = 2.0 * math.pi * freq / res["v_o"]

    L_meters = math.pi / (beta_e + beta_o)

    return {
        "K_dB": K_dB,
        "Length_um": L_meters * 1e6,
        "Z_input": res["Z"],
    }


def optimize_coupler_geometry(
    coupling_dB: float = -14.0,
    freq: float = 8e9,
    Z0: float = 50.0,
    gap_to_ground: float | None = None,
) -> CouplerGeometry:
    if gap_to_ground is None:
        bounds = [(5.0, 50.0), (5.0, 500.0), (5.0, 100.0)]
        x0 = np.array([20.0, 50.0, 10.0], dtype=float)

        def cost(x: np.ndarray) -> float:
            try:
                r = estimate_edge_coupled_directional_coupler(
                    x[0], x[2], x[1], freq=freq
                )
                return (r["K_dB"] - coupling_dB) ** 2 + (r["Z_input"] - Z0) ** 2
            except Exception:
                return 1e30

        opt = minimize(
            cost,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        width, gap_lines, gap_gnd = opt.x

    else:
        bounds = [(5.0, 50.0), (5.0, 500.0)]
        x0 = np.array([20.0, 50.0], dtype=float)

        def cost(x: np.ndarray) -> float:
            try:
                r = estimate_edge_coupled_directional_coupler(
                    x[0], gap_to_ground, x[1], freq=freq
                )
                return (r["K_dB"] - coupling_dB) ** 2 + (r["Z_input"] - Z0) ** 2
            except Exception:
                return 1e30

        opt = minimize(
            cost,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        width, gap_lines = opt.x
        gap_gnd = float(gap_to_ground)

    r = estimate_edge_coupled_directional_coupler(width, gap_gnd, gap_lines, freq=freq)

    return CouplerGeometry(
        width_um=float(width),
        gap_between_lines_um=float(gap_lines),
        gap_to_ground_um=float(gap_gnd),
        length_um=float(r["Length_um"]),
        k_db=float(r["K_dB"]),
        z_input_ohm=float(r["Z_input"]),
    )


def calculate_discrete_params(
    width: float,
    gap_gnd: float,
    gap_lines: float,
    target_length_um: float,
    cell_length_um: float = 10.0,
) -> CouplerDiscrete:
    cell_length_m = cell_length_um * 1e-6
    N_coupled = int(round(target_length_um / cell_length_um))

    unit_params = edge_coupled_cpw(width, gap_gnd, gap_lines)

    L_cell = unit_params["L_self"] * cell_length_m
    Cc_cell = unit_params["C_mutual"] * cell_length_m
    C_gnd_cell = unit_params["C_self"] * cell_length_m
    K_ind = unit_params["L_mutual"] / unit_params["L_self"]

    est = estimate_edge_coupled_directional_coupler(width, gap_gnd, gap_lines)

    geom = CouplerGeometry(
        width_um=width,
        gap_between_lines_um=gap_lines,
        gap_to_ground_um=gap_gnd,
        length_um=target_length_um,
        k_db=est["K_dB"],
        z_input_ohm=est["Z_input"],
    )

    return CouplerDiscrete(
        L_cell=float(L_cell),
        Cc_cell=float(Cc_cell),
        C_gnd_cell=float(C_gnd_cell),
        K_ind=float(K_ind),
        N_coupled=N_coupled,
        N_uncoupled=30,
        geometry=geom,
    )


# =============================================================================
# Netlist construction
# =============================================================================

def add(circuit: list[Element], name: str, n1: Any, n2: Any, value: Any, kind: str) -> None:
    circuit.append(Element(name=name, n1=n1, n2=n2, value=value, kind=kind))


def add_jj(
    circuit: list[Element],
    node_j: int,
    node_jj: int,
    Lj: float,
    Cj: float,
    mod_factor: float = 1.0,
) -> None:
    Lj_mod = Lj / mod_factor
    Cj_mod = Cj * mod_factor
    add(circuit, f"Lj{node_j}_{node_jj}", node_j, node_jj, Lj_mod, "josephson_inductor")
    add(circuit, f"C{node_j}_{node_jj}", node_j, node_jj, Cj_mod, "capacitor")


def add_tl_element(
    circuit: list[Element],
    node_j: int,
    ground: int,
    Ll: float,
    Cl: float,
) -> None:
    add(circuit, f"C{node_j}_{ground}", node_j, ground, Cl, "capacitor")
    add(circuit, f"L{node_j}_{node_j + 1}", node_j, node_j + 1, Ll, "linear_inductor")


def add_jtl_element(
    circuit: list[Element],
    node_j: int,
    ground: int,
    Cg: float,
    Lj: float,
    Cj: float,
    mod_factor: float = 1.0,
) -> None:
    add(circuit, f"C{node_j}_{ground}", node_j, ground, Cg, "capacitor")
    add_jj(circuit, node_j, node_j + 1, Lj, Cj, mod_factor=mod_factor)


def add_tl(
    circuit: list[Element],
    n_start: int,
    ground: int,
    L_cell: float,
    C_gnd_cell: float,
    N_cell: int,
) -> int:
    n_curr = n_start
    for _ in range(N_cell):
        add_tl_element(circuit, n_curr, ground, L_cell, C_gnd_cell)
        n_curr += 1
    return n_curr


def add_jtl(
    circuit: list[Element],
    n_start: int,
    ground: int,
    Cg: float,
    Lj: float,
    Cj: float,
    N_cell: int,
    mod_array: np.ndarray | None = None,
    mod_start_idx: int = 0,
) -> tuple[int, int]:
    n_curr = n_start
    curr_mod_idx = mod_start_idx

    for _ in range(N_cell):
        mf = float(mod_array[curr_mod_idx]) if mod_array is not None else 1.0
        add_jtl_element(circuit, n_curr, ground, Cg, Lj, Cj, mod_factor=mf)
        n_curr += 1

        if mod_array is not None:
            curr_mod_idx += 1

    return n_curr, curr_mod_idx


def add_coupling(
    circuit: list[Element],
    n_t: int,
    n_b: int,
    Cc_cell: float,
    K_ind: float,
) -> None:
    add(circuit, f"Cc{n_t}_{n_b}", n_t, n_b, Cc_cell, "coupling_capacitor")
    add(
        circuit,
        f"K{n_t}_{n_b}",
        f"L{n_t}_{n_t + 1}",
        f"L{n_b}_{n_b + 1}",
        K_ind,
        "mutual_inductor_k",
    )


def add_edge_coupled_directional_coupler(
    circuit: list[Element],
    n_t_start: int,
    n_b_start: int,
    ground: int,
    p: CouplerDiscrete,
) -> tuple[int, int]:
    n_t = n_t_start
    n_b = n_b_start

    add_tl_element(circuit, n_t, ground, p.L_cell, p.C_gnd_cell / 2.0)
    add_tl_element(circuit, n_b, ground, p.L_cell, p.C_gnd_cell / 2.0)
    add_coupling(circuit, n_t, n_b, p.Cc_cell / 2.0, p.K_ind)

    n_t += 1
    n_b += 1

    for _ in range(2, p.N_coupled + 1):
        add_tl_element(circuit, n_t, ground, p.L_cell, p.C_gnd_cell)
        add_tl_element(circuit, n_b, ground, p.L_cell, p.C_gnd_cell)
        add_coupling(circuit, n_t, n_b, p.Cc_cell, p.K_ind)
        n_t += 1
        n_b += 1

    add(circuit, f"C{n_t}_{ground}_end", n_t, ground, p.C_gnd_cell / 2.0, "capacitor")
    add(circuit, f"C{n_b}_{ground}_end", n_b, ground, p.C_gnd_cell / 2.0, "capacitor")
    add(circuit, f"Cc{n_t}_{n_b}_end", n_t, n_b, p.Cc_cell / 2.0, "coupling_capacitor")

    return n_t, n_b


def make_coupler_discrete(params: IPMParams, mode: str) -> CouplerDiscrete:
    if mode == "cached":
        return calculate_discrete_params(
            params.cached_coupler_width_um,
            params.cached_coupler_gap_to_ground_um,
            params.cached_coupler_gap_um,
            params.cached_coupler_length_um,
            params.cell_length_um,
        )

    if mode == "optimize":
        geom = optimize_coupler_geometry(
            coupling_dB=params.coupling_dB,
            freq=params.coupler_freq_hz,
            Z0=params.Z0,
        )
        return calculate_discrete_params(
            geom.width_um,
            geom.gap_to_ground_um,
            geom.gap_between_lines_um,
            geom.length_um,
            params.cell_length_um,
        )

    raise ValueError(f"unknown coupler mode {mode}")


def generate_and_append_coupler(
    circuit: list[Element],
    n_t: int,
    n_b: int,
    ground: int,
    discrete: CouplerDiscrete,
) -> tuple[int, int]:
    return add_edge_coupled_directional_coupler(circuit, n_t, n_b, ground, discrete)


def make_ipm(
    params: IPMParams,
    coupler: CouplerDiscrete,
    mod_array: np.ndarray | None = None,
) -> tuple[list[Element], dict[str, int]]:
    circuit: list[Element] = []

    start_node_top = params.start_node_top
    start_node_bot = params.start_node_bot
    ground = params.ground

    add(circuit, f"P{start_node_top}_{ground}", start_node_top, ground, 1, "port")
    add(circuit, f"R{start_node_top}_{ground}", start_node_top, ground, params.Rleft, "resistor")
    start_node_top = add_tl(circuit, start_node_top, ground, params.Ll, params.Cl, params.len1)

    add(circuit, f"P{start_node_bot}_{ground}", start_node_bot, ground, 3, "port")
    add(circuit, f"R{start_node_bot}_{ground}", start_node_bot, ground, params.Rm, "resistor")
    start_node_bot = add_tl(circuit, start_node_bot, ground, params.Ll, params.Cl, params.len3)

    n_t_end, n_b_end = generate_and_append_coupler(
        circuit,
        start_node_top,
        start_node_bot,
        ground,
        coupler,
    )

    j_top = n_t_end
    j_bottom = n_b_end

    curr_mod_idx = 0
    ll = params.length_of_short_TL
    ll2 = params.length_of_long_TL

    if mod_array is None:
        mod_array = np.ones(params.num_rows * params.array_length, dtype=float)

    for i in range(1, params.num_rows):
        mf = float(mod_array[curr_mod_idx])
        add_jtl_element(
            circuit,
            j_top,
            ground,
            params.Cg / 2.0,
            params.Lj,
            params.Cj,
            mod_factor=mf,
        )
        j_top += 1
        curr_mod_idx += 1

        j_top, curr_mod_idx = add_jtl(
            circuit,
            j_top,
            ground,
            params.Cg,
            params.Lj,
            params.Cj,
            params.array_length - 1,
            mod_array=mod_array,
            mod_start_idx=curr_mod_idx,
        )

        add(circuit, f"C{j_top}_{ground}_JTL_end", j_top, ground, params.Cg / 2.0, "capacitor")

        if i % params.arrays_per_dc == 0:
            j_top = add_tl(circuit, j_top, ground, params.Ll, params.Cl, ll2)
            j_bottom = add_tl(
                circuit,
                j_bottom,
                ground,
                params.Ll,
                params.Cl,
                params.coupler_section_length,
            )
            j_top, j_bottom = generate_and_append_coupler(
                circuit,
                j_top,
                j_bottom,
                ground,
                coupler,
            )
        else:
            j_top = add_tl(circuit, j_top, ground, params.Ll, params.Cl, ll)

    mf = float(mod_array[curr_mod_idx])
    add_jtl_element(
        circuit,
        j_top,
        ground,
        params.Cg / 2.0,
        params.Lj,
        params.Cj,
        mod_factor=mf,
    )
    j_top += 1
    curr_mod_idx += 1

    j_top, curr_mod_idx = add_jtl(
        circuit,
        j_top,
        ground,
        params.Cg,
        params.Lj,
        params.Cj,
        params.array_length - 1,
        mod_array=mod_array,
        mod_start_idx=curr_mod_idx,
    )

    add(circuit, f"C{j_top}_{ground}_JTL_end", j_top, ground, params.Cg / 2.0, "capacitor")

    j_top = add_tl(circuit, j_top, ground, params.Ll, params.Cl, params.len2)
    add(circuit, f"R{j_top}_{ground}", j_top, ground, params.Rright, "resistor")
    add(circuit, f"P{j_top}_{ground}", j_top, ground, 2, "port")

    j_bottom = add_tl(circuit, j_bottom, ground, params.Ll, params.Cl, params.len4)
    add(circuit, f"R{j_bottom}_{ground}", j_bottom, ground, params.Rm, "resistor")
    add(circuit, f"P{j_bottom}_{ground}", j_bottom, ground, 4, "port")

    ends = {
        "top_end_node": j_top,
        "bottom_end_node": j_bottom,
        "jj_mod_used": curr_mod_idx,
    }

    return circuit, ends


# =============================================================================
# Matrix assembly
# =============================================================================

def add_stamp_2node(
    rows: list[int],
    cols: list[int],
    data: list[float],
    node_to_idx: dict[int, int],
    n1: int,
    n2: int,
    value: float,
) -> None:
    idxs: list[int] = []
    coefs: list[float] = []

    if n1 != 0:
        idxs.append(node_to_idx[n1])
        coefs.append(1.0)
    if n2 != 0:
        idxs.append(node_to_idx[n2])
        coefs.append(-1.0)

    for a, ca in zip(idxs, coefs):
        for b, cb in zip(idxs, coefs):
            rows.append(a)
            cols.append(b)
            data.append(value * ca * cb)


def build_matrices(circuit: list[Element]) -> dict[str, Any]:
    nodes = sorted(
        {int(e.n1) for e in circuit if isinstance(e.n1, int) and int(e.n1) != 0}
        | {int(e.n2) for e in circuit if isinstance(e.n2, int) and int(e.n2) != 0}
    )
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    C_r: list[int] = []
    C_c: list[int] = []
    C_d: list[float] = []

    G_r: list[int] = []
    G_c: list[int] = []
    G_d: list[float] = []

    K_r: list[int] = []
    K_c: list[int] = []
    K_d: list[float] = []

    linear_L: dict[str, tuple[int, int, float]] = {}
    coupled_names: set[str] = set()
    k_elems: list[Element] = []
    jj_branches: list[tuple[int, int, float]] = []
    ports: dict[int, tuple[int, int]] = {}

    for e in circuit:
        if e.kind in ("capacitor", "coupling_capacitor"):
            add_stamp_2node(C_r, C_c, C_d, node_to_idx, int(e.n1), int(e.n2), float(e.value))

        elif e.kind == "resistor":
            add_stamp_2node(G_r, G_c, G_d, node_to_idx, int(e.n1), int(e.n2), 1.0 / float(e.value))

        elif e.kind == "linear_inductor":
            linear_L[e.name] = (int(e.n1), int(e.n2), float(e.value))

        elif e.kind == "mutual_inductor_k":
            k_elems.append(e)
            coupled_names.add(str(e.n1))
            coupled_names.add(str(e.n2))

        elif e.kind == "josephson_inductor":
            jj_branches.append((int(e.n1), int(e.n2), float(e.value)))

        elif e.kind == "port":
            ports[int(e.value)] = (int(e.n1), int(e.n2))

    for lname, (n1, n2, L) in linear_L.items():
        if lname not in coupled_names:
            add_stamp_2node(K_r, K_c, K_d, node_to_idx, n1, n2, 1.0 / L)

    for ke in k_elems:
        l1 = str(ke.n1)
        l2 = str(ke.n2)
        k = float(ke.value)

        if l1 not in linear_L or l2 not in linear_L:
            raise ValueError(f"K element references missing L: {ke}")

        n1a, n1b, L1 = linear_L[l1]
        n2a, n2b, L2 = linear_L[l2]

        M = k * math.sqrt(L1 * L2)
        det = L1 * L2 - M * M
        invL = np.array(
            [
                [L2 / det, -M / det],
                [-M / det, L1 / det],
            ],
            dtype=float,
        )

        branches = [(n1a, n1b), (n2a, n2b)]

        for bi, (a1, a2) in enumerate(branches):
            idxs_i: list[int] = []
            coefs_i: list[float] = []

            if a1 != 0:
                idxs_i.append(node_to_idx[a1])
                coefs_i.append(1.0)
            if a2 != 0:
                idxs_i.append(node_to_idx[a2])
                coefs_i.append(-1.0)

            for bj, (b1, b2) in enumerate(branches):
                idxs_j: list[int] = []
                coefs_j: list[float] = []

                if b1 != 0:
                    idxs_j.append(node_to_idx[b1])
                    coefs_j.append(1.0)
                if b2 != 0:
                    idxs_j.append(node_to_idx[b2])
                    coefs_j.append(-1.0)

                val = invL[bi, bj]

                for ia, ca in zip(idxs_i, coefs_i):
                    for jb, cb in zip(idxs_j, coefs_j):
                        K_r.append(ia)
                        K_c.append(jb)
                        K_d.append(val * ca * cb)

    b_rows: list[int] = []
    b_cols: list[int] = []
    b_data: list[float] = []
    Ic: list[float] = []
    Lj: list[float] = []

    for j, (n1, n2, L) in enumerate(jj_branches):
        if n1 != 0:
            b_rows.append(node_to_idx[n1])
            b_cols.append(j)
            b_data.append(1.0)
        if n2 != 0:
            b_rows.append(node_to_idx[n2])
            b_cols.append(j)
            b_data.append(-1.0)

        Lj.append(L)
        Ic.append(PHI0_REDUCED / L)

    C = sp.coo_matrix((C_d, (C_r, C_c)), shape=(n, n)).tocsr()
    G = sp.coo_matrix((G_d, (G_r, G_c)), shape=(n, n)).tocsr()
    K = sp.coo_matrix((K_d, (K_r, K_c)), shape=(n, n)).tocsr()
    Bphi = sp.coo_matrix((b_data, (b_rows, b_cols)), shape=(n, len(jj_branches))).tocsr()

    port_vectors = {
        p: node_to_idx[n1]
        for p, (n1, n2) in ports.items()
        if n2 == 0 and n1 != 0
    }

    return {
        "nodes": np.array(nodes, dtype=np.int64),
        "C": C,
        "G": G,
        "K": K,
        "Bphi": Bphi,
        "Ic": np.array(Ic, dtype=float),
        "Lj": np.array(Lj, dtype=float),
        "ports": ports,
        "port_vectors": port_vectors,
    }


def apply_lj_scatter(
    circuit: list[Element],
    *,
    sigma: float,
    seed: int,
    clip_min: float = 0.5,
    clip_max: float = 1.5,
) -> dict[str, Any]:
    """Apply deterministic multiplicative Gaussian scatter to Josephson Lj."""
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("lj scatter sigma must be non-negative")
    if clip_min <= 0.0 or clip_max <= 0.0 or clip_min > clip_max:
        raise ValueError("lj scatter clip bounds must be positive and ordered")

    jj = [e for e in circuit if e.kind == "josephson_inductor"]
    meta: dict[str, Any] = {
        "lj_scatter_enabled": bool(sigma > 0.0),
        "lj_scatter_sigma": sigma,
        "lj_scatter_seed": int(seed),
        "lj_scatter_count": len(jj),
        "lj_scatter_clip_min": float(clip_min),
        "lj_scatter_clip_max": float(clip_max),
    }
    if not jj or sigma == 0.0:
        return meta

    rng = np.random.default_rng(int(seed))
    factors = rng.normal(loc=1.0, scale=sigma, size=len(jj))
    factors = np.clip(factors, clip_min, clip_max)
    base_lj = np.array([float(e.value) for e in jj], dtype=float)
    new_lj = base_lj * factors
    for e, value in zip(jj, new_lj):
        e.value = float(value)

    meta.update(
        {
            "lj_scatter_factor_min": float(np.min(factors)),
            "lj_scatter_factor_max": float(np.max(factors)),
            "lj_scatter_factor_mean": float(np.mean(factors)),
            "lj_scatter_factor_std": float(np.std(factors, ddof=0)),
            "lj_base_min_h": float(np.min(base_lj)),
            "lj_base_max_h": float(np.max(base_lj)),
            "lj_scattered_min_h": float(np.min(new_lj)),
            "lj_scattered_max_h": float(np.max(new_lj)),
        }
    )
    return meta


# =============================================================================
# Outputs / drawing
# =============================================================================

def count_kinds(circuit: list[Element]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in circuit:
        out[e.kind] = out.get(e.kind, 0) + 1
    return out


def write_outputs(
    outdir: str,
    circuit: list[Element],
    params: IPMParams,
    coupler: CouplerDiscrete,
    ends: dict[str, int],
    mats: dict[str, Any] | None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "ipm_elements.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "name", "node1", "node2", "value", "kind"])
        for i, e in enumerate(circuit, 1):
            w.writerow([i, e.name, e.n1, e.n2, e.value, e.kind])

    ports = [e for e in circuit if e.kind == "port"]
    with open(os.path.join(outdir, "ipm_ports.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["port", "name", "node1", "node2"])
        for e in sorted(ports, key=lambda x: int(x.value)):
            w.writerow([e.value, e.name, e.n1, e.n2])

    nodes = sorted(
        {int(e.n1) for e in circuit if isinstance(e.n1, int) and int(e.n1) != 0}
        | {int(e.n2) for e in circuit if isinstance(e.n2, int) and int(e.n2) != 0}
    )

    summary: dict[str, Any] = {
        "status": "PYTHON_IPM_BUILD_OK",
        "total_elements": len(circuit),
        "non_ground_nodes": len(nodes),
        "min_node": nodes[0],
        "max_node": nodes[-1],
        "counts": count_kinds(circuit),
        "params": asdict(params),
        "coupler": asdict(coupler),
        "ends": ends,
        "interpretation": {
            "port1": "top signal input",
            "port2": "top signal output",
            "port3": "bottom pump rail left/reference side",
            "port4": "bottom pump source side; counter-propagating relative to signal 1->2",
        },
    }
    if extra_summary:
        summary.update(extra_summary)

    if mats is not None:
        summary["matrices"] = {
            "C_shape": list(mats["C"].shape),
            "C_nnz": mats["C"].nnz,
            "G_shape": list(mats["G"].shape),
            "G_nnz": mats["G"].nnz,
            "K_shape": list(mats["K"].shape),
            "K_nnz": mats["K"].nnz,
            "Bphi_shape": list(mats["Bphi"].shape),
            "Bphi_nnz": mats["Bphi"].nnz,
            "Ic_min": float(mats["Ic"].min()) if mats["Ic"].size else None,
            "Ic_max": float(mats["Ic"].max()) if mats["Ic"].size else None,
            "ports": mats["ports"],
            "port_vectors": mats["port_vectors"],
        }

        sp.save_npz(os.path.join(outdir, "C.npz"), mats["C"])
        sp.save_npz(os.path.join(outdir, "G.npz"), mats["G"])
        sp.save_npz(os.path.join(outdir, "K.npz"), mats["K"])
        sp.save_npz(os.path.join(outdir, "Bphi.npz"), mats["Bphi"])

        ordered_ports = sorted(mats["port_vectors"].keys())
        np.savez(
            os.path.join(outdir, "ipm_arrays.npz"),
            nodes=mats["nodes"],
            Ic=mats["Ic"],
            Lj=mats["Lj"],
            phi0_reduced=np.array([PHI0_REDUCED]),
            port_numbers=np.array(ordered_ports, dtype=np.int64),
            port_indices=np.array([mats["port_vectors"][p] for p in ordered_ports], dtype=np.int64),
        )

    with open(os.path.join(outdir, "ipm_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("PYTHON_IPM_BUILD_OK")
    print(f"total_elements={summary['total_elements']}")
    print(f"non_ground_nodes={summary['non_ground_nodes']}")

    for k in sorted(summary["counts"]):
        print(f"{k}={summary['counts'][k]}")

    coupler = summary["coupler"]
    print(f"coupler_N_coupled={coupler['N_coupled']}")
    print(f"coupler_L_cell_pH={coupler['L_cell'] * 1e12:.6g}")
    print(f"coupler_Cc_cell_fF={coupler['Cc_cell'] * 1e15:.6g}")
    print(f"coupler_C_gnd_cell_fF={coupler['C_gnd_cell'] * 1e15:.6g}")
    print(f"coupler_K_ind={coupler['K_ind']:.9g}")
    print(f"coupler_width_um={coupler['geometry']['width_um']:.6g}")
    print(f"coupler_gap_um={coupler['geometry']['gap_between_lines_um']:.6g}")
    print(f"coupler_gap_to_ground_um={coupler['geometry']['gap_to_ground_um']:.6g}")
    print(f"coupler_length_um={coupler['geometry']['length_um']:.6g}")

    if "matrices" in summary:
        m = summary["matrices"]
        print(
            f"C_nnz={m['C_nnz']} "
            f"G_nnz={m['G_nnz']} "
            f"K_nnz={m['K_nnz']} "
            f"Bphi_shape={m['Bphi_shape']} "
            f"Bphi_nnz={m['Bphi_nnz']}"
        )
        print(f"Ic_min={m['Ic_min']:.12e}")
        print(f"Ic_max={m['Ic_max']:.12e}")
        print(f"ports={m['ports']}")
        print(f"port_vectors={m['port_vectors']}")


def draw_schematics(outdir: str) -> None:
    try:
        import schemdraw
        import schemdraw.elements as elm
        import schemdraw.flow as flow
    except Exception as exc:
        print(f"DRAW_SKIPPED schemdraw_import_error={exc}")
        print("Install with: pip install schemdraw")
        return

    os.makedirs(outdir, exist_ok=True)

    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.0)

        d += elm.Dot().label("P1\nsignal in", loc="left")
        d += elm.Line().right().length(1.0).label("TL\nlen1=100", loc="top")
        d += flow.Box(w=1.4, h=0.7).label("Coupler 1")
        d += elm.Line().right().length(0.8)
        d += flow.Box(w=1.5, h=0.7).label("JTL rows\n1-3")
        d += elm.Line().right().length(0.8).label("long TL\n250", loc="top")
        d += flow.Box(w=1.4, h=0.7).label("Coupler 2")
        d += elm.Line().right().length(0.8)
        d += flow.Box(w=1.5, h=0.7).label("JTL rows\n4-6")
        d += elm.Line().right().length(1.0).label("TL\nlen2=50", loc="top")
        d += elm.Dot().label("P2\nsignal out", loc="right")

        d += elm.Dot().at((2.0, -2.0)).label("P3\npump rail left", loc="left")
        d += elm.Line().right().length(7.0).label("bottom pump rail / coupler separation", loc="bottom")
        d += elm.Dot().label("P4\npump source", loc="right")
        d += elm.Arrow().at((8.5, -2.7)).left().length(3.0).label(
            "pump direction 4 ? 3\ncounter-propagating",
            loc="bottom",
        )
        d += elm.Arrow().at((2.0, 1.0)).right().length(3.0).label(
            "signal direction 1 ? 2",
            loc="top",
        )

        d.save(os.path.join(outdir, "ipm_block_diagram.svg"))

    print(f"wrote={os.path.join(outdir, 'ipm_block_diagram.svg')}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(conflict_handler="resolve")
    p.add_argument("--outdir", default=os.path.join("outputs", "ipm_python_design"))
    p.add_argument("--coupler-mode", choices=["cached", "optimize"], default="cached")
    p.add_argument("--write-matrices", action="store_true")
    p.add_argument("--draw", action="store_true")
    p.add_argument("--lj-scatter-sigma", type=float, default=0.0)
    p.add_argument("--lj-scatter-seed", type=int, default=1)
    p.add_argument("--lj-scatter-clip-min", type=float, default=0.5)
    p.add_argument("--lj-scatter-clip-max", type=float, default=1.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    params = IPMParams()
    coupler = make_coupler_discrete(params, args.coupler_mode)
    circuit, ends = make_ipm(params, coupler)
    scatter_meta = apply_lj_scatter(
        circuit,
        sigma=args.lj_scatter_sigma,
        seed=args.lj_scatter_seed,
        clip_min=args.lj_scatter_clip_min,
        clip_max=args.lj_scatter_clip_max,
    )

    mats = build_matrices(circuit) if args.write_matrices else None

    summary = write_outputs(
        outdir=args.outdir,
        circuit=circuit,
        params=params,
        coupler=coupler,
        ends=ends,
        mats=mats,
        extra_summary=scatter_meta,
    )
    print_summary(summary)

    if args.draw:
        draw_schematics(args.outdir)


if __name__ == "__main__":
    main()
