"""Small-signal gain around a validated half-pump PERIOD2 state."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.sparse as sp

from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.signal.floquet import solve_gain_one
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.io import PumpSolution


def period_doubled_idler_sideband() -> int:
    """Return the signed sideband for ``f_i=f_p-f_s`` on the half-pump grid."""
    return -2


def solve_period_doubled_gain(
    circuit: Any,
    state: np.ndarray,
    basis: Any,
    *,
    signal_ghz: float,
    source_port: int,
    out_port: int,
    source_current_a: float = 1.0,
    z0_ohm: float = 50.0,
    sidebands: int = 6,
    gamma_nt: int = 512,
    dc_branch_flux: np.ndarray | None = None,
    loss_model: str = "current_complex_c",
    linear_solver: str = "superlu",
    environment: Any | None = None,
):
    """Solve gain using the half-pump Floquet lattice.

    The physical pump is mode two of ``basis``.  For the standard 3WM
    relation ``f_p=f_s+f_i``, the signed idler is at sideband ``m=-2`` when
    the Floquet spacing is ``f_p/2``.  The pump state must already have passed
    PERIOD2 HB and physical-state validation; this function does not accept a
    failed or unresolved transient as a pump.
    """
    if int(basis.source_mode) != 2:
        raise ValueError("basis must place the physical pump at mode two")
    state = np.asarray(state, dtype=np.complex128)
    modes = list(int(mode) for mode in basis.modes)
    if state.ndim != 2 or state.shape[0] != len(modes):
        raise ValueError("state shape does not match the half-pump basis")
    if sidebands < 2:
        raise ValueError("sidebands must be >= 2 for the 3WM idler")
    dc = np.zeros(circuit.branch_count, dtype=float) if dc_branch_flux is None else np.asarray(dc_branch_flux, dtype=float).reshape(-1)
    if dc.size != circuit.branch_count:
        raise ValueError("dc_branch_flux has incompatible length")

    pump = PumpSolution(
        X=state,
        omega_p=float(basis.omega_p),
        pump_freq_ghz=float(basis.omega_p / (2.0 * math.pi * 1.0e9)),
        harmonics=len(modes),
        nt_original=0,
        metadata=basis.to_metadata(),
        modes=modes,
        basis=basis,
    )
    ms = list(range(-int(sidebands), int(sidebands) + 1))
    max_ell = max(abs(m - q) for m in ms for q in ms)
    gamma_hat = compute_gamma_hat(
        circuit,
        pump,
        max_ell=max_ell,
        gamma_nt=max(int(gamma_nt), 2 * max(modes) + 4),
        dc_branch_flux=dc,
    )
    khat = build_khat(circuit.Bphi, gamma_hat, drop_tol=0.0)
    branch = make_branch_law(circuit)
    gamma_off = np.asarray(branch.tangent(dc[None, :]))[0]
    khat_off_0 = (
        circuit.Bphi
        @ sp.diags(gamma_off, offsets=0, format="csr")
        @ circuit.Bphi.T
    ).astype(np.complex128).tocsr()
    return solve_gain_one(
        circuit=circuit,
        khat=khat,
        khat_off_0=khat_off_0,
        omega_p=float(basis.omega_p),
        signal_ghz=float(signal_ghz),
        sidebands=int(sidebands),
        signal_m=0,
        idler_m=period_doubled_idler_sideband(),
        source_index=circuit.port_to_index[int(source_port)],
        out_index=circuit.port_to_index[int(out_port)],
        source_current_a=float(source_current_a),
        source_port=int(source_port),
        out_port=int(out_port),
        z0_ohm=float(z0_ohm),
        loss_model=loss_model,
        linear_solver=linear_solver,
        environment=environment,
    )

