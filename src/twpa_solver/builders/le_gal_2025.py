"""Effective-SNAIL line builder for the Le Gal benchmark."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

from twpa_solver.core import CircuitMatrices
from twpa_solver.core.constants import PHI0_REDUCED
from twpa_solver.core.nonlinear import EffectiveSnailBranchLaw


def ladder_dispersion(
    omega_rad_per_s: float | np.ndarray,
    *,
    inductance_h: float,
    snail_capacitance_f: float,
    ground_capacitance_f: float,
    cell_length_m: float,
) -> np.ndarray:
    """Return discrete-LC-ladder wavenumber in rad/m.

    The exact cell relation is
    ``sin(k dx / 2)^2 = omega^2 L Cg / (4 (1 - omega^2 L C))``.
    """
    omega = np.asarray(omega_rad_per_s, dtype=float)
    argument = omega * np.sqrt(inductance_h * ground_capacitance_f)
    argument /= 2.0 * np.sqrt(
        1.0 - omega**2 * inductance_h * snail_capacitance_f
    )
    return 2.0 * np.arcsin(argument) / cell_length_m


def build_effective_snail_line(
    *,
    cells: int = 700,
    cell_length_m: float = 8.7e-6,
    critical_current_a: float = 1.4e-6,
    ratio: float = 0.062,
    snail_capacitance_f: float = 31e-15,
    ground_capacitance_f: float = 223.5e-15,
    flux_over_flux0: float = 0.5,
    port_impedance_ohm: float = 62.4,
    shunt_conductance_s: float = 0.0,
    external_flux_on_small_junction: bool = False,
) -> CircuitMatrices:
    """Build a serial effective-SNAIL line with explicit matched ports.

    ``shunt_conductance_s`` is an SI conductance stamp.  A dielectric loss
    tangent is frequency dependent and must be converted by the caller with
    :func:`conductance_from_loss_tangent` at the chosen reference frequency.
    """
    if cells < 1:
        raise ValueError("cells must be positive")
    n = int(cells)
    node_count = n + 1
    phi_ext = np.full(n, 2.0 * math.pi * PHI0_REDUCED * flux_over_flux0)
    ic = np.full(n, critical_current_a)
    ratios = np.full(n, ratio)
    unshifted = EffectiveSnailBranchLaw(
        ic, ratios, phi_ext, PHI0_REDUCED,
        external_flux_on_small_junction=external_flux_on_small_junction,
    )
    equilibrium_values = []
    for index in range(n):
        def residual(value: float, idx: int = index) -> float:
            if external_flux_on_small_junction:
                normalized = ratio * math.sin((value - phi_ext[idx]) / PHI0_REDUCED)
                normalized += math.sin(value / (3.0 * PHI0_REDUCED))
            else:
                normalized = ratio * math.sin(value / PHI0_REDUCED)
                normalized += math.sin((value - phi_ext[idx]) / (3.0 * PHI0_REDUCED))
            return float(critical_current_a * normalized)
        center = phi_ext[index]
        root = brentq(
            residual,
            center - math.pi * PHI0_REDUCED,
            center + math.pi * PHI0_REDUCED,
            xtol=1e-30,
            rtol=1e-14,
        )
        equilibrium_values.append(root)
    equilibrium_values = np.asarray(equilibrium_values)
    equilibrium = equilibrium_values[None, :]
    law = EffectiveSnailBranchLaw(
        ic, ratios, phi_ext, PHI0_REDUCED, equilibrium_flux=equilibrium_values,
        external_flux_on_small_junction=external_flux_on_small_junction,
    )
    equilibrium_residual = np.max(np.abs(law.current(np.zeros((1, n)))))
    # Double precision evaluation of the two cancelling sine terms bottoms
    # out around 1e-17 A at the published half-flux point.
    if equilibrium_residual >= 1e-16 * critical_current_a:
        raise ValueError(
            f"SNAIL equilibrium residual too large: {equilibrium_residual}"
        )
    if np.any(law.tangent(np.zeros((1, n))) <= 0.0):
        raise ValueError("SNAIL equilibrium is not on a stable branch")
    linear_inductance = 1.0 / law.tangent(np.zeros((1, n)))[0]
    bphi = sp.lil_matrix((node_count, n), dtype=float)
    for index in range(n):
        bphi[index, index] = 1.0
        bphi[index + 1, index] = -1.0
    bphi = bphi.tocsr()
    c = bphi @ sp.diags(np.full(n, snail_capacitance_f)) @ bphi.T
    c = c + sp.eye(node_count, format="csr") * ground_capacitance_f
    # K contains only linear branches. The effective SNAIL stiffness enters
    # through Bphi @ i_branch in every nonlinear residual and tangent. Stamping
    # its small-signal slope here as well would count it twice.
    k = sp.csr_matrix((node_count, node_count), dtype=float)
    g = sp.eye(node_count, format="csr") * shunt_conductance_s
    g = g.tolil()
    g[0, 0] += 1.0 / port_impedance_ohm
    g[-1, -1] += 1.0 / port_impedance_ohm
    metadata = {
        "model": "effective_snail_line",
        "cells": n,
        "cell_length_m": cell_length_m,
        "snail_capacitance_f": snail_capacitance_f,
        "ground_capacitance_f": ground_capacitance_f,
        "port_impedance_ohm": port_impedance_ohm,
        "branch_law": law.metadata,
        "equilibrium_flux_rad": (equilibrium_values / PHI0_REDUCED).tolist(),
        "linear_inductance_h": float(np.mean(linear_inductance)),
        "shunt_conductance_s": shunt_conductance_s,
        "external_flux_on_small_junction": external_flux_on_small_junction,
    }
    return CircuitMatrices(
        C=c.tocsr(),
        G=g.tocsr(),
        K=k,
        Bphi=bphi,
        Ic=ic,
        phi0=PHI0_REDUCED,
        port_to_index={1: 0, 2: n},
        metadata=metadata,
        branch_law=law,
    )


def conductance_from_loss_tangent(
    tan_delta: float, capacitance_f: float, omega_rad_s: float
) -> float:
    """Convert capacitor loss tangent to a conductance at ``omega_rad_s``."""
    return float(omega_rad_s * capacitance_f * tan_delta)
