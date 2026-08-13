"""Milestone H1: pump-only transient branch-transfer experiment.

This is deliberately a diagnostic, not a production map path.  It reuses the
persisted circuit matrices and the production HB Fourier convention to ramp a
7.9 GHz pump through the low-drive HB obstruction, classify the held state, and
optionally project a periodic attractor back into the HB solver.

The circuit matrices use node flux as the state variable.  A zero row of C is
handled as an index-one algebraic velocity constraint; no dense inverse of C is
formed.  The resulting first-order system is integrated in pump phase theta,
which keeps the numerical state near the scale of phi0.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

try:
    import matplotlib.pyplot as plt
except ImportError:  # compact numerical runs do not require plotting
    plt = None

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

STROBE_MULTIPLES = (1, 2, 3, 4, 5, 6, 8, 12, 16)


def _finite_tail(values: Any, fraction: float = 0.5) -> np.ndarray:
    """Return a finite late-time slice used by recurrence classification."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return array
    start = max(0, array.size - max(6, int(math.ceil(array.size * fraction))))
    return array[start:]


def _strobe_summary(
    periods: np.ndarray,
    distances: dict[str, np.ndarray],
    *,
    max_period: float | None = None,
) -> dict[str, Any]:
    """Build aligned, late-window recurrence metrics for a stroboscopic run."""
    if max_period is not None:
        periods = periods[periods <= float(max_period) + 1e-12]
    payload: dict[str, Any] = {
        "periods": periods.tolist(),
        "periods_by_n": {},
    }
    for key, values in distances.items():
        n = int(key[1:])
        count = max(0, periods.size - n)
        values = np.asarray(values, dtype=float)[:count]
        value_periods = periods[n:n + values.size]
        payload[key] = values.tolist()
        payload["periods_by_n"][key] = value_periods.tolist()
        tail = _finite_tail(values)
        payload.setdefault("tail_median_by_n", {})[key] = (
            float(np.median(tail)) if tail.size else float("inf")
        )
        payload.setdefault("tail_max_by_n", {})[key] = (
            float(np.max(tail)) if tail.size else float("inf")
        )
    d1_tail = _finite_tail(distances.get("d1", []))
    payload["tail_median"] = float(np.median(d1_tail)) if d1_tail.size else float("inf")
    payload["tail_max"] = float(np.max(d1_tail)) if d1_tail.size else float("inf")
    payload["tail_d2_max"] = payload.get("tail_max_by_n", {}).get("d2")
    payload["tail_d3_max"] = payload.get("tail_max_by_n", {}).get("d3")
    return payload

from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.core.constants import PHI0_REDUCED  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.pump import basis as pump_basis  # noqa: E402
from twpa_solver.pump.hb import FullPumpProblem, HarmonicGrid  # noqa: E402
from twpa_solver.pump.io import summarize_solution  # noqa: E402
from twpa_solver.pump.solver import HarmonicNewtonKrylovSolver, NewtonKrylovSettings  # noqa: E402
from twpa_solver.pump.validation import validate_production_hb_state  # noqa: E402


@dataclass(frozen=True)
class TransientAudit:
    circuit_dir: str
    node_count: int
    branch_count: int
    c_nnz: int
    g_nnz: int
    k_nnz: int
    algebraic_nodes: list[int]
    differential_nodes: int
    c_factorable: bool
    algebraic_g_factorable: bool
    transient_integrator: str
    source_convention: str
    hb_convention: str


@dataclass(frozen=True)
class ShiftedBranchLaw:
    """Branch current about a fixed DC flux, matching production HB."""

    base: Any
    dc_flux: np.ndarray

    def current(self, flux: np.ndarray) -> np.ndarray:
        dc = self.dc_flux[None, :]
        return self.base.current(flux + dc) - self.base.current(dc)

    def tangent(self, flux: np.ndarray) -> np.ndarray:
        return self.base.tangent(flux + self.dc_flux[None, :])

    def gamma(self, flux: np.ndarray) -> np.ndarray:
        return self.tangent(flux)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"type": "dc_shifted", "base": self.base.metadata}


@dataclass
class TransientSystem:
    circuit: Any
    branch: Any
    omega: float
    pump_node: int
    differential: np.ndarray
    algebraic: np.ndarray
    c_factor: Any
    g_alg_factor: Any
    full_state: bool = False
    phi0: float = PHI0_REDUCED

    def __post_init__(self) -> None:
        self.circuit.C = self.circuit.C.tocsr()
        self.circuit.G = self.circuit.G.tocsr()
        self.circuit.K = self.circuit.K.tocsr()
        self.circuit.Bphi = self.circuit.Bphi.tocsr()
        self._g_ad = self.circuit.G[self.algebraic][:, self.differential]
        self._k_a = self.circuit.K[self.algebraic]
        self._b_a = self.circuit.Bphi[self.algebraic]
        self._g_d = self.circuit.G[self.differential]
        self._k_d = self.circuit.K[self.differential]
        self._b_d = self.circuit.Bphi[self.differential]

    @property
    def n(self) -> int:
        return self.circuit.node_count

    def pack(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        if self.full_state:
            return np.concatenate((q, p))
        return np.concatenate((q[self.differential], p[self.differential], q[self.algebraic]))

    def unpack(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.full_state:
            return np.asarray(y[:self.n]), np.asarray(y[self.n:2 * self.n])
        nd = self.differential.size
        q = np.zeros(self.n, dtype=float)
        p = np.zeros(self.n, dtype=float)
        q[self.differential] = y[:nd]
        p[self.differential] = y[nd:2 * nd]
        q[self.algebraic] = y[2 * nd:]
        return q, p

    def algebraic_velocity(self, q: np.ndarray, p_d: np.ndarray, source: np.ndarray) -> np.ndarray:
        if self.g_alg_factor is None:
            flux = self.phi0 * (self.circuit.Bphi.T @ q)
            tangent = np.asarray(self.branch.tangent(flux[None, :]))[0]
            constraint_jac = self.circuit.K[self.algebraic] + (
                self._b_a @ sp.diags(tangent) @ self.circuit.Bphi.T
            )
            aa = constraint_jac[:, self.algebraic].tocsc()
            ad = constraint_jac[:, self.differential]
            rhs = self.source_derivative(0.0, source)[self.algebraic] / self.phi0
            rhs = rhs - ad @ p_d
            return np.asarray(spla.spsolve(aa, rhs), dtype=float)
        flux = self.phi0 * (self.circuit.Bphi.T @ q)
        currents = np.asarray(self.branch.current(flux[None, :]))[0]
        rhs = source[self.algebraic] - self._g_ad @ (self.omega * self.phi0 * p_d)
        rhs = rhs - self._k_a @ (self.phi0 * q) - self._b_a @ currents
        return np.asarray(self.g_alg_factor.solve(np.asarray(rhs).reshape(-1)), dtype=float) / (
            self.omega * self.phi0
        )

    def source_derivative(self, theta: float, source: np.ndarray) -> np.ndarray:
        """Derivative of the source with respect to pump phase."""
        # The source vector already contains the instantaneous amplitude.  The
        # only nonzero source is the pump entry; use the current ramp policy's
        # derivative supplied by the caller where needed.  For the lossless
        # rf-SQUID algebraic rows this is identically zero, so returning zero
        # is exact for the relevant constraint block.
        return np.zeros_like(source)

    def project_algebraic_state(self, q: np.ndarray, p: np.ndarray, source: np.ndarray) -> None:
        """Project q onto static algebraic constraints and recover p_a.

        This is used only when G_aa is singular.  It solves the actual
        nonlinear constraint Jacobian K_aa + B_a diag(I') B_a^T and does not
        insert conductance or capacitance.
        """
        if not self.algebraic.size:
            return
        for _ in range(4):
            flux = self.phi0 * (self.circuit.Bphi.T @ q)
            currents = np.asarray(self.branch.current(flux[None, :]))[0]
            residual = self._k_a @ (self.phi0 * q) + self._b_a @ currents - source[self.algebraic]
            if float(np.linalg.norm(residual)) < 1e-12:
                break
            tangent = np.asarray(self.branch.tangent(flux[None, :]))[0]
            jac = self.circuit.K[self.algebraic] + (
                self._b_a @ sp.diags(tangent) @ self.circuit.Bphi.T
            )
            delta = spla.spsolve(jac[:, self.algebraic].tocsc(), -residual / self.phi0)
            q[self.algebraic] += delta
        p[self.algebraic] = self.algebraic_velocity(q, p[self.differential], source)

    def rhs(self, theta: float, y: np.ndarray, start_current: float, target_current: float, ramp_theta: float) -> np.ndarray:
        q, p = self.unpack(y)
        if self.algebraic.size:
            p_a = self.algebraic_velocity(q, p[self.differential], self.source(theta, start_current, target_current, ramp_theta))
            p[self.algebraic] = p_a
        flux = self.phi0 * (self.circuit.Bphi.T @ q)
        currents = np.asarray(self.branch.current(flux[None, :]))[0]
        source = self.source(theta, start_current, target_current, ramp_theta)
        force = source[self.differential]
        force = force - self._g_d @ (self.omega * self.phi0 * p)
        force = force - self._k_d @ (self.phi0 * q) - self._b_d @ currents
        p_dot_d = self.c_factor.solve(np.asarray(force).reshape(-1)) / (self.omega**2 * self.phi0)
        return np.concatenate((p[self.differential], p_dot_d, p[self.algebraic]))

    def source(self, theta: float, start_current: float, target_current: float, ramp_theta: float) -> np.ndarray:
        if ramp_theta <= 0.0:
            amplitude = target_current
        else:
            u = min(max(theta / ramp_theta, 0.0), 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            amplitude = start_current + (target_current - start_current) * smooth
        source = np.zeros(self.n, dtype=float)
        source[self.pump_node] = amplitude * math.cos(theta)
        return source

    def jacobian_sparsity(self) -> sp.csr_matrix:
        """Return a sparse local-dependence pattern for BDF finite differences.

        The exact reduced vector field contains the factored inverse of C, but
        this structural pattern preserves the circuit-local dependencies that
        drive the nonlinear Jacobian and avoids one finite-difference solve per
        state coordinate.
        """
        nd = self.differential.size
        na = self.algebraic.size
        nstate = 2 * nd + na
        q_pattern = (self._k_d != 0) + (self._b_d @ self.circuit.Bphi.T != 0)
        qa_pattern = (self.circuit.K[self.differential][:, self.algebraic] != 0)
        p_pattern = self._g_d[:, self.differential] != 0
        rows: list[int] = []
        cols: list[int] = []
        for row in range(nd):
            rows.append(row); cols.append(nd + row)
        q_rows, q_cols = q_pattern.nonzero()
        rows.extend((nd + q_rows).tolist()); cols.extend(q_cols.tolist())
        qa_rows, qa_cols = qa_pattern.nonzero()
        rows.extend((nd + qa_rows).tolist()); cols.extend((2 * nd + qa_cols).tolist())
        p_rows, p_cols = p_pattern.nonzero()
        rows.extend((nd + p_rows).tolist()); cols.extend((nd + p_cols).tolist())
        if na:
            ga_rows, ga_cols = self._g_ad.nonzero()
            rows.extend((2 * nd + ga_rows).tolist()); cols.extend(ga_cols.tolist())
            ka_rows, ka_cols = self._k_a[:, self.differential].nonzero()
            rows.extend((2 * nd + ka_rows).tolist()); cols.extend(ka_cols.tolist())
            kaa_rows, kaa_cols = self._k_a[:, self.algebraic].nonzero()
            rows.extend((2 * nd + kaa_rows).tolist()); cols.extend((2 * nd + kaa_cols).tolist())
            rows.extend((2 * nd + ga_rows).tolist()); cols.extend((nd + ga_cols).tolist())
        pattern = sp.coo_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(nstate, nstate),
        ).tocsr()
        # BDF's Newton matrix needs a diagonal structural entry even where the
        # vector field has an exact zero derivative (notably the algebraic
        # coordinate row in this index-one reduction).
        return (pattern + sp.eye(nstate, format="csr", dtype=bool)).astype(bool)


def audit_circuit(circuit_dir: Path) -> TransientAudit:
    circuit = load_circuit(circuit_dir)
    zero_rows = np.flatnonzero(np.diff(circuit.C.indptr) == 0)
    zero_cols = np.flatnonzero(np.diff(circuit.C.tocsc().indptr) == 0)
    if not np.array_equal(zero_rows, zero_cols):
        raise ValueError("C has unmatched zero rows/columns; MVP cannot eliminate it safely")
    differential = np.setdiff1d(np.arange(circuit.node_count), zero_rows)
    c_factorable = True
    try:
        spla.splu(circuit.C[differential][:, differential].tocsc())
    except RuntimeError:
        c_factorable = False
    algebraic_g_factorable = True
    if zero_rows.size:
        try:
            spla.splu(circuit.G[zero_rows][:, zero_rows].tocsc())
        except RuntimeError:
            algebraic_g_factorable = False
    return TransientAudit(
        circuit_dir=str(circuit_dir), node_count=circuit.node_count,
        branch_count=circuit.branch_count, c_nnz=circuit.C.nnz,
        g_nnz=circuit.G.nnz, k_nnz=circuit.K.nnz,
        algebraic_nodes=zero_rows.astype(int).tolist(),
        differential_nodes=int(differential.size), c_factorable=c_factorable,
        algebraic_g_factorable=algebraic_g_factorable,
        transient_integrator=(
            "scipy.integrate.solve_ivp available; H3 uses sparse implicit trapezoid"
        ),
        source_convention="peak Norton/current-source amplitude: I_p(t)=I_p cos(theta)",
        hb_convention="X_k positive phasor, x(t)=2 Re sum X_k exp(+i k theta)",
    )


def build_system(
    circuit_dir: Path, freq_ghz: float, pump_port: int,
    dc_branch_flux: np.ndarray | None = None,
) -> TransientSystem:
    circuit = load_circuit(circuit_dir)
    algebraic = np.flatnonzero(np.diff(circuit.C.indptr) == 0)
    differential = np.setdiff1d(np.arange(circuit.node_count), algebraic)
    c_factor = spla.splu(circuit.C[differential][:, differential].tocsc())
    g_alg_factor = None
    full_state = False
    if algebraic.size:
        try:
            g_alg_factor = spla.splu(circuit.G[algebraic][:, algebraic].tocsc())
        except RuntimeError:
            # Use the exact index-one reduction below: p_a is recovered from
            # the differentiated nonlinear algebraic constraint for explicit
            # backends.  The validated trapezoid path retains the full
            # (q,p) Newton system and projects the lossless constraints after
            # each endpoint; no Gmin or artificial circuit element is used.
            full_state = True
    base_branch = make_branch_law(circuit)
    dc = (np.zeros(circuit.branch_count, dtype=float)
          if dc_branch_flux is None else np.asarray(dc_branch_flux, dtype=float).reshape(-1))
    if dc.size != circuit.branch_count:
        raise ValueError("dc_branch_flux must match the branch count")
    branch = ShiftedBranchLaw(base_branch, dc) if np.any(dc) else base_branch
    return TransientSystem(
        circuit=circuit, branch=branch,
        omega=2.0 * math.pi * freq_ghz * 1e9,
        pump_node=circuit.port_to_index[pump_port], differential=differential,
        algebraic=algebraic, c_factor=c_factor, g_alg_factor=g_alg_factor,
        full_state=full_state,
    )


def dc_flux_from_external_fraction(circuit_dir: Path, fraction: float, phi0: float) -> np.ndarray:
    """Convert external flux fraction to the rf-SQUID DC branch flux.

    Compiled design metadata supplies Lm and Ic.  For generic circuits, a
    direct reduced phase is retained as the backward-compatible fallback.
    """
    if fraction == 0.0:
        circuit = load_circuit(circuit_dir)
        return np.zeros(circuit.branch_count, dtype=float)
    params_path = circuit_dir / "design_resolved.json"
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8")).get("parameters", {})
        if "Ic" in params and "Lm" in params:
            beta_l = float(params["Lm"]) * float(params["Ic"]) / phi0
            external_phase = 2.0 * math.pi * fraction
            phase = brentq(
                lambda value: value - external_phase + beta_l * math.sin(value),
                external_phase - beta_l - 0.5,
                external_phase + beta_l + 0.5,
            )
            circuit = load_circuit(circuit_dir)
            return np.full(circuit.branch_count, phase * phi0, dtype=float)
    circuit = load_circuit(circuit_dir)
    return np.full(circuit.branch_count, fraction * 2.0 * math.pi * phi0, dtype=float)


def load_hb_initial(
    checkpoint: Path,
    circuit: Any,
    omega: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    report = json.loads((checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    if report.get("final_status") != "VALID_CONVERGED":
        raise ValueError(f"checkpoint is not converged: {checkpoint}")
    data = np.load(checkpoint / "pump_solution.npz")
    X = np.asarray(data["X_real"], dtype=float) + 1j * np.asarray(data["X_imag"], dtype=float)
    modes = np.asarray(data["pump_modes"], dtype=int)
    grid = HarmonicGrid(modes=modes, nt=max(2 * int(modes.max()) + 1, 40), omega=omega)
    x = grid.synthesize(X)
    w = grid.synthesize_derivative(X, 1) / omega
    current = float(report["metadata"]["pump_current_a"])
    return x[0], w[0], current, report


def checkpoint_dc_flux(
    report: dict[str, Any],
    circuit: Any,
    fallback: np.ndarray | None = None,
    checkpoint: Path | None = None,
) -> np.ndarray:
    """Return the exact DC branch-flux vector used by the saved HB fixture.

    Production HB checkpoints are authoritative.  Re-solving a nominal external
    flux fraction here can select a different convention/operating point and
    invalidate an otherwise correct HB state at the TD handoff.
    """
    metadata = report.get("metadata", {})
    value = metadata.get("dc_branch_flux")
    if value is None:
        value = metadata.get("dc_branch_flux_wb")
    if value is None and checkpoint is not None:
        sidecar = checkpoint / "hybrid_fixture_config.json"
        if sidecar.exists():
            value = json.loads(sidecar.read_text(encoding="utf-8")).get(
                "dc_branch_flux"
            )
    if value is None:
        if fallback is None:
            return np.zeros(circuit.branch_count, dtype=float)
        value = fallback
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(circuit.branch_count, float(array[0]), dtype=float)
    if array.size != circuit.branch_count:
        raise ValueError("checkpoint DC flux has incompatible branch count")
    return array


def make_observables(
    system: TransientSystem,
    theta: np.ndarray,
    states: np.ndarray,
    *,
    out_port: int = 2,
    start_current: float = 0.0,
    target_current: float = 0.0,
    ramp_theta: float = 0.0,
) -> dict[str, np.ndarray]:
    """Collect transient observables, including the selected port voltage."""
    out: dict[str, list[float]] = {
        "theta": [], "mu": [], "source_current_a": [], "max_abs_sin_phi": [],
        "max_abs_phi": [], "min_cos_phi": [], "strongest_branch": [],
        "state_norm": [], "pump_node_flux": [], "output_voltage_v": [],
        "shunt_power_w": [],
    }
    rcsj = system.circuit.metadata.get("metadata", {}).get("rcsj", {})
    if not rcsj:
        rcsj = system.circuit.metadata.get("rcsj", {})
    resistance = np.asarray(rcsj.get("resistance_ohm", []), dtype=float).reshape(-1)
    if resistance.size not in (0, system.circuit.branch_count):
        raise ValueError("RCSJ resistance metadata does not match branch count")
    output_node = system.circuit.port_to_index[out_port]
    dc_flux = np.asarray(
        getattr(system.branch, "dc_flux", np.zeros(system.circuit.branch_count)),
        dtype=float,
    ).reshape(-1)
    for angle, y in zip(theta, states.T):
        q, p = system.unpack(y)
        if output_node in system.algebraic:
            algebraic_index = int(np.flatnonzero(system.algebraic == output_node)[0])
            p[output_node] = system.algebraic_velocity(
                q,
                p[system.differential],
                system.source(
                    float(angle), start_current, target_current, ramp_theta
                ),
            )[algebraic_index]
        phi = (
            np.asarray(system.circuit.Bphi.T @ (system.phi0 * q)) + dc_flux
        ) / system.phi0
        sin_phi = np.sin(phi)
        idx = int(np.argmax(np.abs(sin_phi)))
        out["theta"].append(float(angle))
        out["mu"].append(float("nan"))
        out["source_current_a"].append(float(system.source(angle, 1.0, 1.0, 0.0)[system.pump_node]))
        out["max_abs_sin_phi"].append(float(np.max(np.abs(sin_phi))))
        out["max_abs_phi"].append(float(np.max(np.abs(phi))))
        out["min_cos_phi"].append(float(np.min(np.cos(phi))))
        out["strongest_branch"].append(idx)
        out["state_norm"].append(float(np.linalg.norm(q) / math.sqrt(q.size)))
        out["pump_node_flux"].append(float(system.phi0 * q[system.pump_node]))
        out["output_voltage_v"].append(
            float(system.omega * system.phi0 * p[output_node])
        )
        if resistance.size:
            branch_voltage = system.omega * system.phi0 * np.asarray(
                system.circuit.Bphi.T @ p, dtype=float
            )
            out["shunt_power_w"].append(float(np.sum(branch_voltage**2 / resistance)))
        else:
            out["shunt_power_w"].append(0.0)
    return {key: np.asarray(value) for key, value in out.items()}


def node_velocity(
    system: TransientSystem,
    q: np.ndarray,
    p: np.ndarray,
    node: int,
    source: np.ndarray,
) -> float:
    """Return a node velocity, including an index-one algebraic node."""
    node = int(node)
    if node not in system.algebraic:
        return float(p[node])
    algebraic_index = int(np.flatnonzero(system.algebraic == node)[0])
    return float(
        system.algebraic_velocity(q, p[system.differential], source)[algebraic_index]
    )


def stroboscopic_diagnostics(system: TransientSystem, theta: np.ndarray, states: np.ndarray, periods: int) -> dict[str, Any]:
    points = np.arange(0.0, float(periods) + 1.0)
    origin = float(theta[0]) if theta.size else 0.0
    selected = np.asarray([int(np.argmin(np.abs(theta - (origin + 2.0 * math.pi * x)))) for x in points])
    distances: dict[str, np.ndarray] = {}
    for period_multiple in STROBE_MULTIPLES:
        if selected.size > period_multiple:
            reference = states[:, selected[:-period_multiple]]
            pair_scales = np.maximum(np.linalg.norm(reference, axis=0), 1.0)
            distances[f"d{period_multiple}"] = (
                np.linalg.norm(
                    states[:, selected[period_multiple:]] - reference, axis=0
                ) / pair_scales
            )
    return _strobe_summary(points, distances)


def classify_state(
    strobe: dict[str, Any], mean_phase_velocity: float, integrator_success: bool,
    mean_phase_winding_cycles: float = 0.0,
) -> str:
    if not integrator_success:
        return "TRANSIENT_NUMERICAL_FAILURE"
    # A nonzero fitted slope is expected from bounded numerical drift.  Require
    # a substantial unwrapped phase winding before calling this running phase.
    if abs(mean_phase_winding_cycles) > 0.1:
        return "RUNNING_PHASE"
    if strobe["tail_max"] < 5e-4:
        return "PERIOD_1"
    if strobe.get("tail_d2_max") is not None and strobe["tail_d2_max"] < 5e-4:
        return "PERIOD_2"
    if strobe.get("tail_d3_max") is not None and strobe["tail_d3_max"] < 5e-4:
        return "PERIOD_3"
    if strobe["tail_max"] < 1e-2:
        return "QUASIPERIODIC_OR_PERIOD_N"
    return "BROADBAND_OR_CHAOTIC"


def decay_aware_stroboscopic_classification(strobe: dict[str, Any]) -> dict[str, Any]:
    """Estimate whether a non-periodic-looking hold is still relaxing.

    The decision uses late-window magnitude and trend for d1/d2/d3.  A finite
    tail with a downward envelope is therefore not promoted to a physical
    non-periodic state merely because its early transient was large.
    """
    series: dict[str, np.ndarray] = {}
    for key in ("d1", "d2", "d3"):
        values = np.asarray(strobe.get(key, []), dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            series[key] = values
    d1 = series.get("d1", np.empty(0))
    if d1.size < 6:
        return {"class": "UNRESOLVED_SLOW_RELAXATION", "trend_b": None, "tau_periods": None}

    def fit_slopes(values: np.ndarray) -> list[float]:
        window = max(6, values.size // 3)
        if values.size < window:
            return []
        index = np.arange(values.size, dtype=float)
        return [
            float(np.polyfit(index[i:i + window], np.log(values[i:i + window]), 1)[0])
            for i in range(0, values.size - window + 1, max(1, window // 2))
        ]

    slopes_by_key = {key: fit_slopes(values) for key, values in series.items()}
    slopes = slopes_by_key["d1"]
    b = slopes[-1] if slopes else 0.0
    tau = float(-1.0 / b) if b < 0.0 else None
    tail_max_by_n = {
        key: float(np.max(_finite_tail(values))) for key, values in series.items()
    }
    periodic_support = all(
        tail_max_by_n.get(key, 0.0) < 5e-4 for key in ("d1", "d2", "d3")
    )
    if periodic_support:
        cls = "PERIOD_1"
    else:
        decaying = b < -1e-3 and len(slopes) >= 2 and all(s < 0.0 for s in slopes[-2:])
        for key, key_slopes in slopes_by_key.items():
            if key != "d1" and key_slopes and key_slopes[-1] > 2e-4:
                decaying = False
        cls = "RELAXING_TO_PERIOD1" if decaying else (
            "PERSISTENT_NONPERIODIC" if abs(b) < 2e-4 else "UNRESOLVED_SLOW_RELAXATION"
        )
    return {
        "class": cls,
        "trend_b": b,
        "tau_periods": tau,
        "window_slopes": slopes,
        "window_slopes_by_n": slopes_by_key,
        "late_max_by_n": tail_max_by_n,
    }


def classify_td_result(result: dict[str, Any]) -> str:
    """Map H1 diagnostics to a conservative hybrid-column classification.

    A broadband-looking finite hold is not evidence of a physical transition
    when the decay-aware test still sees relaxation.  Keep that distinction at
    the adapter boundary so the column policy cannot accidentally promote a
    slow transient to a physical boundary.
    """
    raw = str(result.get("classification", ""))
    decay_class = str((result.get("decay_aware") or {}).get("class", ""))
    if raw in {"QUASIPERIODIC_OR_PERIOD_N", "BROADBAND_OR_CHAOTIC"}:
        if decay_class in {"UNRESOLVED_SLOW_RELAXATION", "RELAXING_TO_PERIOD1"}:
            return "UNRESOLVED_SLOW_RELAXATION"
    return raw


def max_abs_phi_envelope_classification(
    data: dict[str, np.ndarray],
    ramp_periods: int,
    *,
    growth_threshold_per_period: float = 1.0e-5,
) -> dict[str, Any]:
    """Classify the post-ramp phase envelope by its fitted slope.

    The phase-envelope slope is the primary transient discriminant.  It is a
    level-independent quantity, so it does not require a same-protocol floor
    calibration.  Recurrence distances remain available as secondary
    diagnostics, but cannot change this label.
    """
    periods = np.asarray(data.get("theta", []), dtype=float) / (2.0 * math.pi)
    envelope = np.asarray(data.get("max_abs_phi", []), dtype=float)
    mask = (
        np.isfinite(periods)
        & np.isfinite(envelope)
        & (periods >= float(ramp_periods))
    )
    periods = periods[mask]
    envelope = envelope[mask]
    if periods.size < 2 or np.ptp(periods) <= 0.0:
        return {
            "class": "ENVELOPE_SLOPE_UNRESOLVED",
            "slope_per_period": None,
            "threshold_per_period": growth_threshold_per_period,
            "post_ramp_samples": int(periods.size),
        }
    slope = float(np.polyfit(periods, envelope, 1)[0])
    label = (
        "GROWING_MAX_ABS_PHI"
        if slope > growth_threshold_per_period
        else "NON_GROWING_MAX_ABS_PHI"
    )
    return {
        "class": label,
        "slope_per_period": slope,
        "threshold_per_period": growth_threshold_per_period,
        "post_ramp_samples": int(periods.size),
    }


def checkpoint_stroboscopic_diagnostics(
    strobe: dict[str, Any],
    checkpoints: tuple[int, ...] = (40, 90, 140, 250, 440),
) -> list[dict[str, Any]]:
    """Evaluate recurrence and decay-aware classification at hold checkpoints."""
    periods = np.asarray(strobe.get("periods", []), dtype=float)
    distances = {
        key: np.asarray(strobe.get(key, []), dtype=float)
        for key in (f"d{n}" for n in STROBE_MULTIPLES)
        if strobe.get(key)
    }
    observations: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        sliced = _strobe_summary(periods, distances, max_period=checkpoint)
        raw = classify_state(sliced, 0.0, True)
        decay = decay_aware_stroboscopic_classification(sliced)
        observations.append({
            "hold_periods": int(checkpoint),
            "classification": raw,
            "decay_aware": decay,
            "stroboscopic": sliced,
        })
    return observations


def phase_winding_series(system: TransientSystem, theta: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Return mean unwrapped junction-phase winding in pump cycles."""
    if states.size == 0:
        return np.empty(0, dtype=float)
    phase = np.asarray([
        system.circuit.Bphi.T @ system.unpack(states[:, i])[0]
        for i in range(states.shape[1])
    ])
    return np.mean(np.unwrap(phase, axis=0) - np.unwrap(phase, axis=0)[0], axis=1) / (2.0 * math.pi)


def project_periodic_state(
    system: TransientSystem, dense_state: Any, final_theta: float,
    modes: np.ndarray, current: float, *,
    projection_periods: int = 5,
    samples_per_period: int = 64,
    solve_hb: bool = True,
    preconditioner: str = "real_coupled",
) -> dict[str, Any]:
    """Project several settled periods and diagnose a fixed-drive HB solve.

    Averaging the Fourier coefficients over several periods suppresses TD
    integration noise without storing the full transient.  The projection
    sampling is independent of the transient output sampling and is enlarged
    automatically to resolve the highest retained mode.
    """
    if projection_periods < 1 or samples_per_period < 2:
        raise ValueError("projection periods and samples must be positive")
    nt = max(int(samples_per_period), 2 * int(np.max(modes)) + 4)
    grid = HarmonicGrid(modes=modes, nt=nt, omega=system.omega)
    projected: list[np.ndarray] = []
    waveforms: list[np.ndarray] = []
    for period in range(projection_periods):
        theta = (
            final_theta - (period + 1) * 2.0 * math.pi
            + 2.0 * math.pi * np.arange(nt) / nt
        )
        states = dense_state(theta)
        x_t = np.asarray([
            system.phi0 * system.unpack(states[:, i])[0]
            for i in range(states.shape[1])
        ])
        waveforms.append(x_t)
        projected.append(grid.project_positive(x_t))
    X_seed = np.mean(projected, axis=0)
    x_reprojected = grid.synthesize(X_seed)
    projection_error = float(np.mean([
        np.linalg.norm(x_reprojected - waveform)
        / max(np.linalg.norm(waveform), 1e-300)
        for waveform in waveforms
    ]))
    x_t = waveforms[-1]
    problem = FullPumpProblem(
        system.circuit.C, system.circuit.G, system.circuit.K,
        system.circuit.Bphi, system.branch, grid, system.pump_node, current,
    )
    projected_norms = problem.norms(X_seed, 1.0, True)
    if not solve_hb:
        return {
            "projection_error_rms": projection_error,
            "projection_periods": int(projection_periods),
            "projection_samples_per_period": int(nt),
            "projected_hb_coeff_rel": float(projected_norms["coeff_rel"]),
            "projected_hb_time_rel": float(projected_norms["time_rel"]),
            "hb_converged": False,
            "hb_skipped": True,
            "hb_state": X_seed,
            "hb_state_is_td_projection": True,
        }
    solver = HarmonicNewtonKrylovSolver(NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=12, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=60, gmres_maxiter=80, min_alpha=1.0 / 1024.0,
        preconditioner=preconditioner, compute_time_residual=True,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
        stall_ratio=0.8, stall_patience=4, solve_deadline_s=180.0,
    ))
    X_hb, report = solver.solve_one(problem, X_seed, 1.0)
    transient_phi = system.circuit.Bphi.T @ x_t.T
    hb_phi = problem.branch_flux_time(X_hb).T
    return {
        "projection_error_rms": projection_error,
        "projection_periods": int(projection_periods),
        "projection_samples_per_period": int(nt),
        "projected_hb_coeff_rel": float(projected_norms["coeff_rel"]),
        "projected_hb_time_rel": float(projected_norms["time_rel"]),
        "hb_converged": bool(report.converged),
        "hb_coeff_rel": float(report.coeff_rel),
        "hb_time_rel": report.time_rel,
        "hb_newton_iterations": report.newton_iterations,
        "hb_runtime_s": report.runtime_s,
        "transient_rj": float(np.max(np.abs(np.sin(transient_phi / system.phi0)))),
        "hb_rj": float(np.max(np.abs(np.sin(hb_phi / system.phi0)))),
        "hb_solution_summary": summarize_solution(problem, X_hb),
        # The TD projection is the authoritative handoff seed.  The failed
        # Newton iterate above is diagnostic only; returning it here can move
        # the later residual homotopy away from the physical TD orbit.
        "hb_state": X_seed,
        "hb_state_is_td_projection": True,
    }


def implicit_euler_ramp(
    system: TransientSystem, y0: np.ndarray, start_current: float,
    target_current: float, total_theta: float, ramp_theta: float,
    step_theta: float, newton_tol: float = 1e-3, max_newton: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate the index-one circuit with sparse backward Euler steps."""
    q, p = system.unpack(y0)
    theta_values = [0.0]
    state_values = [np.array(y0, copy=True)]
    theta = 0.0
    steps = 0
    newton_total = 0
    message = "success"
    while theta < total_theta - 1e-12:
        h = min(step_theta, total_theta - theta)
        q_prev = q.copy()
        p_prev = p.copy()
        source = system.source(theta + h, start_current, target_current, ramp_theta)
        q_trial = q_prev + h * p_prev
        converged = False
        for _ in range(max_newton):
            p_trial = (q_trial - q_prev) / h
            qdd = (p_trial - p_prev) / h
            flux = system.phi0 * (system.circuit.Bphi.T @ q_trial)
            currents = np.asarray(system.branch.current(flux[None, :]))[0]
            residual = system.circuit.C @ (system.omega**2 * qdd)
            residual = residual + system.circuit.G @ (system.omega * p_trial)
            residual = residual + system.circuit.K @ q_trial
            residual = residual + system.circuit.Bphi @ currents / system.phi0
            residual = residual - source / system.phi0
            scaled_norm = float(np.linalg.norm(residual) / math.sqrt(residual.size))
            if scaled_norm < newton_tol:
                converged = True
                break
            tangent = np.asarray(system.branch.tangent(flux[None, :]))[0]
            jacobian = (
                system.circuit.C * (system.omega**2 / h**2)
                + system.circuit.G * (system.omega / h)
                + system.circuit.K
                + system.circuit.Bphi @ sp.diags(tangent) @ system.circuit.Bphi.T
            ).tocsc()
            try:
                delta = spla.spsolve(jacobian, -residual)
            except RuntimeError as exc:
                message = f"sparse Newton factorization failed: {exc}"
                break
            q_trial = q_trial + delta
            newton_total += 1
        if not converged:
            if total_theta - theta < max(step_theta, 0.1):
                theta_values.append(total_theta)
                state_values.append(system.pack(q_prev, p_prev))
                return np.asarray(theta_values), np.asarray(state_values).T, {
                    "success": True,
                    "message": "success (terminal Newton residual floor)",
                    "steps": steps, "newton_iterations": newton_total,
                }
            message = message if message != "success" else f"implicit Newton failed at theta={theta:.6g}"
            return np.asarray(theta_values), np.asarray(state_values).T, {
                "success": False, "message": message, "steps": steps,
                "newton_iterations": newton_total,
            }
        q = q_trial
        p = (q - q_prev) / h
        theta += h
        theta_values.append(theta)
        state_values.append(system.pack(q, p))
        steps += 1
    return np.asarray(theta_values), np.asarray(state_values).T, {
        "success": True, "message": message, "steps": steps,
        "newton_iterations": newton_total,
    }


def implicit_trapezoid_ramp(
    system: TransientSystem, y0: np.ndarray, start_current: float,
    target_current: float, total_theta: float, ramp_theta: float,
    step_theta: float, newton_tol: float = 1e-6, max_newton: int = 10,
    checkpoint_dir: Path | None = None, checkpoint_periods: int = 10,
    initial_theta: float = 0.0,
    min_step_theta: float = 1.0 / 32.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate the full index-one DAE with adaptive trapezoid steps.

    The physical DAE is unchanged.  If Newton cannot correct one step, the
    step is retried from the last accepted state at half the step size.  This
    is essential near a junction tangent singularity, where a fixed large
    step can fail even though the trajectory remains finite and below
    ``|I_J| = I_c``.
    """
    if step_theta <= 0.0 or min_step_theta <= 0.0:
        raise ValueError("step sizes must be positive")
    if min_step_theta > step_theta:
        min_step_theta = step_theta
    q, p = system.unpack(y0)
    if system.algebraic.size and not system.full_state:
        p[system.algebraic] = system.algebraic_velocity(
            q, p[system.differential], system.source(0.0, start_current, target_current, ramp_theta)
        )
    elif system.full_state:
        system.project_algebraic_state(
            q, p, system.source(0.0, start_current, target_current, ramp_theta)
        )
    n = system.n
    theta_values = [initial_theta]
    state_values = [system.pack(q, p)]
    theta = initial_theta
    newton_total = 0
    step_reductions = 0
    min_step_used = step_theta
    next_step_theta = step_theta
    next_checkpoint = (initial_theta + float(checkpoint_periods) * 2.0 * math.pi
                       if checkpoint_dir else math.inf)
    while theta < total_theta - 1e-12:
        q_old, p_old = q.copy(), p.copy()
        h = min(next_step_theta, total_theta - theta)
        while True:
            q_trial, p_trial = q_old + h * p_old, p_old.copy()
            source_mid = system.source(theta + 0.5 * h, start_current, target_current, ramp_theta)
            converged = False
            for _ in range(max_newton):
                flux = system.phi0 * (system.circuit.Bphi.T @ q_trial)
                current_new = np.asarray(system.branch.current(flux[None, :]))[0]
                flux_old = system.phi0 * (system.circuit.Bphi.T @ q_old)
                current_old = np.asarray(system.branch.current(flux_old[None, :]))[0]
                dynamic = system.circuit.C @ (system.omega**2 * (p_trial - p_old) / h)
                dynamic = dynamic + system.circuit.G @ (system.omega * (p_trial + p_old) / 2.0)
                dynamic = dynamic + system.circuit.K @ ((q_trial + q_old) / 2.0)
                dynamic = dynamic + system.circuit.Bphi @ (current_new + current_old) / (2.0 * system.phi0)
                dynamic = dynamic - source_mid / system.phi0
                kinematic = q_trial - q_old - h * (p_trial + p_old) / 2.0
                scale_dynamic = max(abs(target_current / system.phi0), 1.0)
                norm = max(
                    float(np.linalg.norm(dynamic) / math.sqrt(n) / scale_dynamic),
                    float(np.linalg.norm(kinematic) / math.sqrt(n)),
                )
                if norm < newton_tol:
                    converged = True
                    break
                tangent = np.asarray(system.branch.tangent(flux[None, :]))[0]
                dynamic_q = (
                    system.circuit.K / 2.0
                    + system.circuit.Bphi @ sp.diags(tangent) @ system.circuit.Bphi.T / 2.0
                )
                dynamic_p = system.circuit.C * (system.omega**2 / h) + system.circuit.G * (system.omega / 2.0)
                jac = sp.bmat(
                    [[dynamic_q, dynamic_p], [sp.eye(n, format="csr"), -h * sp.eye(n, format="csr") / 2.0]],
                    format="csc",
                )
                factor = spla.splu(jac)
                delta = factor.solve(-np.concatenate((dynamic, kinematic)))
                q_trial += delta[:n]
                p_trial += delta[n:]
                newton_total += 1
                # The long-hold path performs thousands of sparse factorizations.
                # Release the temporary block matrix and solve buffers before
                # the next accepted step; otherwise Python/SciPy allocator
                # retention can make RSS grow across an otherwise bounded run.
                del delta, factor, jac, dynamic_q, dynamic_p, tangent
            if converged:
                # Retain a reduced step after entering the stiff near-Ic
                # regime instead of retrying the original large step at every
                # subsequent interval.
                next_step_theta = h
                break
            if h <= min_step_theta * (1.0 + 1e-12):
                return np.asarray(theta_values), np.asarray(state_values).T, {
                    "success": False,
                    "message": (
                        f"implicit trapezoid Newton failed at theta={theta:.6g} "
                        f"at minimum step h={h:.6g}"
                    ),
                    "steps": len(theta_values) - 1,
                    "newton_iterations": newton_total,
                    "step_reductions": step_reductions,
                    "min_step_used": min_step_used,
                }
            h = max(min_step_theta, 0.5 * h)
            next_step_theta = h
            step_reductions += 1
            min_step_used = min(min_step_used, h)
        q, p = q_trial, p_trial
        if system.full_state:
            system.project_algebraic_state(q, p, system.source(theta + h, start_current, target_current, ramp_theta))
        theta += h
        theta_values.append(theta)
        state_values.append(system.pack(q, p))
        if checkpoint_dir is not None and theta + 1e-10 >= next_checkpoint:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                checkpoint_dir / "transient_restart.npz",
                theta=np.asarray(theta), y=state_values[-1],
                start_current=np.asarray(start_current), target_current=np.asarray(target_current),
                ramp_theta=np.asarray(ramp_theta), step_theta=np.asarray(next_step_theta),
            )
            (checkpoint_dir / "transient_restart.json").write_text(json.dumps({
                "theta": theta, "period": theta / (2.0 * math.pi),
                "steps": len(theta_values) - 1, "checkpoint_periods": checkpoint_periods,
            }, indent=2), encoding="utf-8")
            next_checkpoint += float(checkpoint_periods) * 2.0 * math.pi
    return np.asarray(theta_values), np.asarray(state_values).T, {
        "success": True, "message": "success", "steps": len(theta_values) - 1,
        "newton_iterations": newton_total, "step_reductions": step_reductions,
        "min_step_used": min_step_used,
    }


def implicit_trapezoid_ramp_bounded(
    system: TransientSystem, y0: np.ndarray, start_current: float,
    target_current: float, total_theta: float, ramp_theta: float,
    step_theta: float, newton_tol: float = 1e-6, max_newton: int = 10,
    checkpoint_dir: Path | None = None, checkpoint_periods: int = 10,
    initial_theta: float = 0.0, min_step_theta: float = 1.0 / 32.0,
    sample_count: int = 256, history_states: int = 1024, out_port: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Memory-bounded implicit trapezoid integration for long compact holds.

    The legacy integrator stores every full state.  For this circuit that is
    approximately 100 kB per accepted step, so a several-hundred-period hold
    consumes gigabytes before post-processing starts.  This path stores only:

    * uniformly sampled states for compact observables;
    * a bounded stroboscopic state ring for d1/d2/d3/.../d16;
    * a bounded final state ring for projection and phase diagnostics; and
    * scalar pump-flux history for the late spectrum.

    The physical equations, Newton solve, step control, and restart format are
    identical to ``implicit_trapezoid_ramp``.
    """
    if step_theta <= 0.0 or min_step_theta <= 0.0:
        raise ValueError("step sizes must be positive")
    if min_step_theta > step_theta:
        min_step_theta = step_theta
    if sample_count < 2 or history_states < 8:
        raise ValueError("bounded storage limits are too small")

    q, p = system.unpack(y0)
    if system.algebraic.size and not system.full_state:
        p[system.algebraic] = system.algebraic_velocity(
            q, p[system.differential],
            system.source(0.0, start_current, target_current, ramp_theta),
        )
    elif system.full_state:
        system.project_algebraic_state(
            q, p, system.source(0.0, start_current, target_current, ramp_theta)
        )

    initial_state = system.pack(q, p)
    sample_targets = np.linspace(
        initial_theta, total_theta, int(sample_count) + 1
    )
    sampled_theta = [float(initial_theta)]
    sampled_states = [np.array(initial_state, copy=True)]
    next_sample = 1
    # Fixed phase grids avoid allocating a full state on every accepted
    # Newton step.  The state ring spans the final 20 pump periods and is
    # sufficient for late-time phase diagnostics and HB projection.
    history_step = 2.0 * math.pi / 32.0
    history_start = max(float(initial_theta), total_theta - 20.0 * 2.0 * math.pi)
    history = deque(maxlen=int(history_states))
    next_history = history_start
    if next_history <= initial_theta + 1e-12:
        history.append((float(initial_theta), np.array(initial_state, copy=True)))
        next_history += history_step
    scalar_step = 2.0 * math.pi / 128.0
    scalar_start = max(float(initial_theta), total_theta - 512.0 * 2.0 * math.pi)
    scalar_history = deque(maxlen=max(65536, int(sample_count) * 8))
    scalar_output_voltage = deque(maxlen=max(65536, int(sample_count) * 8))
    output_node = system.circuit.port_to_index[int(out_port)]
    next_scalar = scalar_start
    if next_scalar <= initial_theta + 1e-12:
        scalar_history.append((
            float(initial_theta), float(system.phi0 * q[system.pump_node])
        ))
        scalar_output_voltage.append((
            float(initial_theta),
            float(
                system.omega
                * system.phi0
                * node_velocity(
                    system,
                    q,
                    p,
                    output_node,
                    system.source(
                        initial_theta, start_current, target_current, ramp_theta
                    ),
                )
            ),
        ))
        next_scalar += scalar_step

    strobe_theta: list[float] = []
    strobe_multiples = STROBE_MULTIPLES
    strobe_values: dict[str, list[float]] = {
        f"d{multiple}": [] for multiple in strobe_multiples
    }
    strobe_pump_flux: list[float] = []
    strobe_state_norm: list[float] = []
    strobe_history: deque[np.ndarray] = deque(maxlen=max(strobe_multiples))
    next_strobe = max(float(ramp_theta), float(initial_theta))
    if next_strobe <= initial_theta + 1e-12:
        strobe_theta.append(float(initial_theta))
        strobe_history.append(np.array(initial_state, copy=True))
        next_strobe += 2.0 * math.pi

    def payload() -> dict[str, Any]:
        d = {key: np.asarray(value, dtype=float) for key, value in strobe_values.items()}
        strobe = _strobe_summary(np.asarray(strobe_theta, dtype=float) / (2.0 * math.pi), d)
        return {
            "_bounded_sample_theta": np.asarray(sampled_theta, dtype=float),
            "_bounded_sample_states": np.asarray(sampled_states, dtype=float).T,
            "_bounded_strobe_theta": np.asarray(strobe_theta, dtype=float),
            "_bounded_strobe": {
                **strobe,
                "pump_flux": strobe_pump_flux,
                "state_norm": strobe_state_norm,
            },
            "_bounded_history_theta": np.asarray([x[0] for x in history], dtype=float),
            "_bounded_history_states": np.column_stack([x[1] for x in history]),
            "_bounded_scalar_theta": np.asarray([x[0] for x in scalar_history], dtype=float),
            "_bounded_scalar_pump_flux": np.asarray([x[1] for x in scalar_history], dtype=float),
            "_bounded_scalar_output_voltage": np.asarray(
                [x[1] for x in scalar_output_voltage], dtype=float
            ),
        }

    def finish(success: bool, message: str, steps: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        result: dict[str, Any] = {
            "success": success, "message": message, "steps": steps,
            "newton_iterations": newton_total, "step_reductions": step_reductions,
            "min_step_used": min_step_used,
        }
        result.update(payload())
        return (
            result["_bounded_sample_theta"],
            result["_bounded_sample_states"],
            result,
        )

    n = system.n
    theta = float(initial_theta)
    steps = 0
    newton_total = 0
    step_reductions = 0
    min_step_used = step_theta
    next_step_theta = step_theta
    cached_factor = None
    cached_factor_h = None
    cached_factor_step = -10**9
    next_checkpoint = (
        initial_theta + float(checkpoint_periods) * 2.0 * math.pi
        if checkpoint_dir else math.inf
    )

    while theta < total_theta - 1e-12:
        q_old, p_old = q.copy(), p.copy()
        previous_theta = theta
        previous_state = system.pack(q_old, p_old)
        # These quantities depend only on the accepted state at the beginning
        # of the step. Recomputing them inside every Newton iteration was a
        # dominant avoidable cost for the long 7.9 GHz holds.
        flux_old = system.phi0 * (system.circuit.Bphi.T @ q_old)
        current_old = np.asarray(system.branch.current(flux_old[None, :]))[0]
        scale_dynamic = max(abs(target_current / system.phi0), 1.0)
        h = min(next_step_theta, total_theta - theta)
        while True:
            q_trial, p_trial = q_old + h * p_old, p_old.copy()
            source_mid = system.source(
                theta + 0.5 * h, start_current, target_current, ramp_theta
            )
            converged = False
            factor = None
            if (
                cached_factor is not None
                and cached_factor_h == h
                and steps - cached_factor_step < 2
            ):
                factor = cached_factor
            for _ in range(max_newton):
                flux = system.phi0 * (system.circuit.Bphi.T @ q_trial)
                current_new = np.asarray(system.branch.current(flux[None, :]))[0]
                dynamic = system.circuit.C @ (system.omega**2 * (p_trial - p_old) / h)
                dynamic = dynamic + system.circuit.G @ (system.omega * (p_trial + p_old) / 2.0)
                dynamic = dynamic + system.circuit.K @ ((q_trial + q_old) / 2.0)
                dynamic = dynamic + system.circuit.Bphi @ (current_new + current_old) / (2.0 * system.phi0)
                dynamic = dynamic - source_mid / system.phi0
                kinematic = q_trial - q_old - h * (p_trial + p_old) / 2.0
                norm = max(
                    float(np.linalg.norm(dynamic) / math.sqrt(n) / scale_dynamic),
                    float(np.linalg.norm(kinematic) / math.sqrt(n)),
                )
                if norm < newton_tol:
                    converged = True
                    break
                if factor is None:
                    tangent = np.asarray(system.branch.tangent(flux[None, :]))[0]
                    dynamic_q = (
                        system.circuit.K / 2.0
                        + system.circuit.Bphi @ sp.diags(tangent) @ system.circuit.Bphi.T / 2.0
                    )
                    dynamic_p = system.circuit.C * (system.omega**2 / h) + system.circuit.G * (system.omega / 2.0)
                    jac = sp.bmat(
                        [[dynamic_q, dynamic_p],
                         [sp.eye(n, format="csr"), -h * sp.eye(n, format="csr") / 2.0]],
                        format="csc",
                    )
                    factor = spla.splu(jac)
                    cached_factor = factor
                    cached_factor_h = h
                    cached_factor_step = steps
                rhs = np.empty(2 * n, dtype=float)
                rhs[:n] = dynamic
                rhs[n:] = kinematic
                delta = factor.solve(-rhs)
                q_trial += delta[:n]
                p_trial += delta[n:]
                newton_total += 1
                # A reused factor is a two-step chord-Newton predictor. If it
                # does not satisfy the residual on the next iteration, the
                # factor is discarded and rebuilt at the updated state.
                factor = None
                del delta
            if converged:
                next_step_theta = h
                break
            if h <= min_step_theta * (1.0 + 1e-12):
                return finish(
                    False,
                    f"implicit trapezoid Newton failed at theta={theta:.6g} "
                    f"at minimum step h={h:.6g}",
                    steps,
                )
            h = max(min_step_theta, 0.5 * h)
            next_step_theta = h
            step_reductions += 1
            min_step_used = min(min_step_used, h)

        q, p = q_trial, p_trial
        if system.full_state:
            system.project_algebraic_state(
                q, p, system.source(theta + h, start_current, target_current, ramp_theta)
            )
        theta += h
        steps += 1
        # Frequent full-generation collection costs more than it saves here:
        # the bounded path already limits retained arrays. Keep a safety
        # collection interval, but do not interrupt every 32 accepted steps.
        if steps % 1024 == 0:
            gc.collect()
        current_state = system.pack(q, p)
        while next_sample < sample_targets.size and sample_targets[next_sample] <= theta + 1e-12:
            target_theta = float(sample_targets[next_sample])
            alpha = (target_theta - previous_theta) / max(theta - previous_theta, 1e-300)
            sampled_theta.append(target_theta)
            sampled_states.append(np.asarray(
                previous_state + alpha * (current_state - previous_state), dtype=float
            ))
            next_sample += 1

        while next_history <= theta + 1e-12:
            alpha = (next_history - previous_theta) / max(theta - previous_theta, 1e-300)
            sampled = np.asarray(
                previous_state + alpha * (current_state - previous_state), dtype=float
            )
            history.append((float(next_history), sampled))
            next_history += history_step

        while next_scalar <= theta + 1e-12:
            alpha = (next_scalar - previous_theta) / max(theta - previous_theta, 1e-300)
            sampled = np.asarray(
                previous_state + alpha * (current_state - previous_state), dtype=float
            )
            sampled_q, sampled_p = system.unpack(sampled)
            scalar_history.append((
                float(next_scalar), float(system.phi0 * sampled_q[system.pump_node])
            ))
            scalar_output_voltage.append((
                float(next_scalar),
                float(
                    system.omega
                    * system.phi0
                    * node_velocity(
                        system,
                        sampled_q,
                        sampled_p,
                        output_node,
                        system.source(
                            next_scalar,
                            start_current,
                            target_current,
                            ramp_theta,
                        ),
                    )
                ),
            ))
            next_scalar += scalar_step

        while next_strobe <= theta + 1e-12 and next_strobe <= total_theta + 1e-12:
            alpha = (next_strobe - previous_theta) / max(theta - previous_theta, 1e-300)
            sampled = np.asarray(
                previous_state + alpha * (current_state - previous_state), dtype=float
            )
            strobe_theta.append(float(next_strobe))
            sampled_q, _ = system.unpack(sampled)
            strobe_pump_flux.append(float(system.phi0 * sampled_q[system.pump_node]))
            strobe_state_norm.append(float(np.linalg.norm(sampled_q) / math.sqrt(n)))
            for multiple in strobe_multiples:
                key = f"d{multiple}"
                if len(strobe_history) >= multiple:
                    reference = strobe_history[-multiple]
                    scale = max(float(np.linalg.norm(reference)), 1.0)
                    strobe_values[key].append(
                        float(np.linalg.norm(sampled - reference) / scale)
                    )
            strobe_history.append(np.array(sampled, copy=True))
            next_strobe += 2.0 * math.pi

        if checkpoint_dir is not None and theta + 1e-10 >= next_checkpoint:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            voltage_checkpoint = checkpoint_dir / "transient_voltage_observables.npz"
            voltage_tmp = checkpoint_dir / "transient_voltage_observables.tmp.npz"
            np.savez(
                voltage_tmp,
                theta=np.asarray([x[0] for x in scalar_output_voltage], dtype=float),
                output_voltage_v=np.asarray([x[1] for x in scalar_output_voltage], dtype=float),
            )
            voltage_tmp.replace(voltage_checkpoint)
            np.savez_compressed(
                checkpoint_dir / "transient_restart.npz",
                theta=np.asarray(theta), y=current_state,
                start_current=np.asarray(start_current), target_current=np.asarray(target_current),
                ramp_theta=np.asarray(ramp_theta), step_theta=np.asarray(next_step_theta),
            )
            (checkpoint_dir / "transient_restart.json").write_text(json.dumps({
                "theta": theta, "period": theta / (2.0 * math.pi),
                "steps": steps, "checkpoint_periods": checkpoint_periods,
            }, indent=2), encoding="utf-8")
            next_checkpoint += float(checkpoint_periods) * 2.0 * math.pi

    return finish(True, "success", steps)


def plot_results(outdir: Path, data: dict[str, np.ndarray], spectrum: dict[str, np.ndarray]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required only for non-compact plotting output")
    theta = data["theta"]
    time_ns = theta / (2.0 * math.pi * 7.9e9) * 1e9
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ns, data["source_current_a"] * 1e6)
    ax.set(xlabel="time (ns)", ylabel="pump source current (µA)")
    fig.tight_layout(); fig.savefig(outdir / "pump_drive_vs_time.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ns, data["max_abs_sin_phi"])
    ax.set(xlabel="time (ns)", ylabel="max |sin(phi)|")
    fig.tight_layout(); fig.savefig(outdir / "junction_utilization_vs_time.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ns, data["max_abs_phi"], label="max |phi|")
    ax.plot(time_ns, data["min_cos_phi"], label="min cos(phi)")
    ax.legend(); ax.set(xlabel="time (ns)")
    fig.tight_layout(); fig.savefig(outdir / "phase_metrics_vs_time.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ns, data["state_norm"], label="state RMS")
    ax.plot(time_ns, data["pump_node_flux"] / PHI0_REDUCED, label="pump flux / phi0")
    ax.legend(); ax.set(xlabel="time (ns)")
    fig.tight_layout(); fig.savefig(outdir / "state_observables_vs_time.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(spectrum["frequency_ghz"], spectrum["amplitude"], "o-")
    ax.set(xlabel="frequency (GHz)", ylabel="pump-node flux spectrum")
    fig.tight_layout(); fig.savefig(outdir / "late_time_spectrum.png", dpi=150); plt.close(fig)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    """Replace a compact numerical artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _segmented_ivp(
    system: TransientSystem,
    y0: np.ndarray,
    start_current: float,
    target_current: float,
    ramp_theta: float,
    total_theta: float,
        args: argparse.Namespace,
        outdir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate BDF/Radau in short segments and publish progress artifacts."""
    # One-period segments keep restart and observable artifacts bounded even
    # when the stiff BDF solve takes a long time on a high-Q ladder rung.
    segment_theta = 2.0 * math.pi
    theta = 0.0
    state = np.asarray(y0, dtype=float)
    theta_parts = [np.asarray([theta], dtype=float)]
    state_parts = [state[:, None]]
    nfev = njev = nlu = 0
    success = True
    message = "success"
    segment_index = 0
    while theta < total_theta - 1e-12:
        endpoint = min(total_theta, theta + segment_theta)
        solve_options: dict[str, Any] = {
            "method": args.method, "rtol": args.rtol,
            "atol": (
                args.atol
                * np.maximum(np.abs(state), args.atol_floor)
                if args.atol_mode == "state_relative"
                else args.atol
            ),
            "max_step": args.max_step, "dense_output": True,
        }
        if args.method in ("BDF", "Radau"):
            solve_options["jac_sparsity"] = system.jacobian_sparsity()
        sol = solve_ivp(
            lambda angle, value: system.rhs(
                angle, value, start_current, target_current, ramp_theta
            ),
            (theta, endpoint), state, **solve_options,
        )
        nfev += int(sol.nfev); njev += int(sol.njev or 0); nlu += int(sol.nlu or 0)
        if not sol.success or sol.sol is None:
            success = False
            message = str(sol.message)
            break
        samples = max(2, int(round((endpoint - theta) / (2.0 * math.pi) * args.samples_per_period)) + 1)
        segment_angles = np.linspace(theta, endpoint, samples)
        segment_states = np.asarray(sol.sol(segment_angles), dtype=float)
        theta_parts.append(segment_angles[1:])
        state_parts.append(segment_states[:, 1:])
        theta = endpoint
        state = segment_states[:, -1]
        segment_index += 1
        _atomic_npz(
            outdir / "transient_progress.npz",
            theta=np.concatenate(theta_parts),
            states=np.concatenate(state_parts, axis=1),
            segment=np.asarray(segment_index),
            completed_periods=np.asarray(theta / (2.0 * math.pi)),
        )
        _atomic_npz(
            outdir / "transient_restart.npz",
            theta=np.asarray(theta), y=state,
            start_current=np.asarray(start_current), target_current=np.asarray(target_current),
            ramp_theta=np.asarray(ramp_theta), step_theta=np.asarray(args.max_step),
        )
        segment_data = make_observables(
            system,
            segment_angles,
            segment_states,
            out_port=args.out_port,
            start_current=start_current,
            target_current=target_current,
            ramp_theta=ramp_theta,
        )
        _atomic_npz(outdir / "transient_progress_observables.npz", **segment_data)
    angles = np.concatenate(theta_parts)
    states = np.concatenate(state_parts, axis=1)
    return angles, states, {
        "success": success, "message": message,
        "nfev": nfev, "njev": njev, "nlu": nlu,
        "steps": nfev, "newton_iterations": None,
        "step_reductions": None,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    audit = audit_circuit(args.circuit_dir)
    if not audit.c_factorable:
        raise RuntimeError(f"DAE audit failed: {audit}")
    # Load the production checkpoint before constructing the TD fixture.  Its
    # persisted DC branch vector is part of the physical problem definition and
    # must take precedence over a nominal CLI flux fraction.
    checkpoint_circuit = load_circuit(args.circuit_dir)
    checkpoint_report = json.loads(
        (args.checkpoint / "pump_report.json").read_text(encoding="utf-8")
    )
    fixture_config: dict[str, Any] = {}
    fixture_config_path = args.checkpoint / "hybrid_fixture_config.json"
    if fixture_config_path.exists():
        fixture_config = json.loads(fixture_config_path.read_text(encoding="utf-8"))
    fallback_dc = dc_flux_from_external_fraction(
        args.circuit_dir, float(getattr(args, "dc_flux_over_phi0", 0.0)), PHI0_REDUCED
    )
    dc_flux = checkpoint_dc_flux(
        checkpoint_report, checkpoint_circuit, fallback_dc, args.checkpoint
    )
    checkpoint_pump_port = checkpoint_report.get("metadata", {}).get("pump_port")
    if checkpoint_pump_port is None:
        checkpoint_pump_port = fixture_config.get("pump_port")
    if checkpoint_pump_port is not None and int(checkpoint_pump_port) != int(args.pump_port):
        raise ValueError(
            f"checkpoint pump_port={checkpoint_pump_port} differs from TD pump_port={args.pump_port}"
        )
    system = build_system(args.circuit_dir, args.freq_ghz, args.pump_port, dc_flux)
    if system.full_state and args.method != "implicit_trapezoid":
        raise RuntimeError(
            "RF-SQUID lossless algebraic constraints require implicit_trapezoid"
        )
    X0, w0, checkpoint_current, hb_report = load_hb_initial(
        args.checkpoint, system.circuit, system.omega
    )
    checkpoint_data = np.load(args.checkpoint / "pump_solution.npz")
    checkpoint_X = np.asarray(checkpoint_data["X_real"], dtype=float) + 1j * np.asarray(checkpoint_data["X_imag"], dtype=float)
    checkpoint_modes = np.asarray(
        hb_report["metadata"].get("pump_modes", checkpoint_data["pump_modes"]), dtype=int
    )
    checkpoint_dc = checkpoint_dc_flux(hb_report, system.circuit, dc_flux)
    validation = validate_production_hb_state(
        system.circuit, make_branch_law(system.circuit), frequency_hz=args.freq_ghz * 1e9,
        pump_port=args.pump_port, pump_current_a=checkpoint_current,
        modes=checkpoint_modes, state=checkpoint_X,
        nt=max(2 * int(checkpoint_modes.max()) + 1, 40),
        metadata=hb_report.get("metadata", {}),
        dc_branch_flux=checkpoint_dc,
    )
    if not validation["checkpoint_validated"]:
        result = {
            "checkpoint": str(args.checkpoint),
            "classification": "INVALID_HB_FIXTURE",
            "final_status": "INVALID_HB_FIXTURE",
            "hb_validation": validation,
            "integrator": None,
        }
        (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    initialization_mode = getattr(args, "initialization_mode", "hb_periodic")
    restart_path = getattr(args, "transient_restart", None)
    if restart_path is not None and initialization_mode != "hb_periodic":
        raise ValueError("--transient-restart cannot be combined with zero-pump initialization")
    if restart_path is not None:
        restart_data = np.load(restart_path)
        y0 = np.asarray(restart_data["y"], dtype=float)
        # A resumed bridge starts at the already reached physical drive.  The
        # continuation ramp is local to this call, so its phase coordinate is
        # intentionally reset to zero.
        start_current = float(restart_data["target_current"])
    elif initialization_mode == "zero_pump_equilibrium":
        # The unbiased 2c circuit has zero DC branch flux.  Start from the
        # actual zero-pump equilibrium and project only the algebraic rows.
        # No state from another target is allowed in this mode.
        start_current = 0.0
        q0 = np.zeros(system.n, dtype=float)
        p0 = np.zeros(system.n, dtype=float)
        system.project_algebraic_state(q0, p0, system.source(0.0, 0.0, 0.0, 0.0))
        y0 = system.pack(q0, p0)
    elif initialization_mode == "hb_periodic":
        q0 = X0 / system.phi0; p0 = w0 / system.phi0
        start_current = checkpoint_current
        if system.g_alg_factor is None:
            system.project_algebraic_state(
                q0, p0, system.source(0.0, start_current, args.target_current_a, 0.0)
            )
        y0 = system.pack(q0, p0)
    else:
        raise ValueError(f"unknown initialization mode: {initialization_mode}")
    ramp_theta = 2.0 * math.pi * args.ramp_periods
    total_theta = 2.0 * math.pi * (args.ramp_periods + args.hold_periods)
    compact_output = bool(getattr(args, "compact_output", False))
    if args.method == "implicit_euler":
        sample_theta, states, integrator = implicit_euler_ramp(
            system, y0, start_current, args.target_current_a, total_theta,
            ramp_theta, args.max_step, args.atol, args.max_newton,
        )
        dense_state = lambda query: np.vstack([np.interp(query, sample_theta, row) for row in states])
    elif args.method == "implicit_trapezoid":
        if compact_output:
            sample_theta, states, integrator = implicit_trapezoid_ramp_bounded(
                system, y0, start_current, args.target_current_a, total_theta,
                ramp_theta, args.max_step, args.atol, args.max_newton,
                checkpoint_dir=outdir / "restart_checkpoints",
                checkpoint_periods=args.checkpoint_periods,
                min_step_theta=getattr(args, "min_step_theta", 1.0 / 32.0),
                sample_count=int(getattr(args, "compact_sample_count", 256)),
                history_states=int(getattr(args, "compact_history_states", 1024)),
                out_port=args.out_port,
            )
            integrator.pop("_bounded_sample_theta")
            integrator.pop("_bounded_sample_states")
            bounded_strobe = integrator.pop("_bounded_strobe")
            bounded_strobe_theta = integrator.pop("_bounded_strobe_theta")
            bounded_history_theta = integrator.pop("_bounded_history_theta")
            bounded_history_states = integrator.pop("_bounded_history_states")
            bounded_scalar_theta = integrator.pop("_bounded_scalar_theta")
            bounded_scalar_pump_flux = integrator.pop("_bounded_scalar_pump_flux")
            bounded_scalar_output_voltage = integrator.pop("_bounded_scalar_output_voltage")
            dense_state = lambda query: np.vstack([
                np.interp(query, bounded_history_theta, row)
                for row in bounded_history_states
            ])
        else:
            sample_theta, states, integrator = implicit_trapezoid_ramp(
                system, y0, start_current, args.target_current_a, total_theta,
                ramp_theta, args.max_step, args.atol, args.max_newton,
                checkpoint_dir=outdir / "restart_checkpoints",
                checkpoint_periods=args.checkpoint_periods,
                min_step_theta=getattr(args, "min_step_theta", 1.0 / 32.0),
            )
            dense_state = lambda query: np.vstack([np.interp(query, sample_theta, row) for row in states])
    else:
        if compact_output:
            sample_theta, states, integrator = _segmented_ivp(
                system, y0, start_current, args.target_current_a, ramp_theta,
                total_theta, args, outdir,
            )
            dense_state = lambda query: np.vstack([
                np.interp(query, sample_theta, row) for row in states
            ])
        else:
            solve_options: dict[str, Any] = {
                "method": args.method, "rtol": args.rtol,
                "atol": (
                    args.atol
                    * np.maximum(np.abs(y0), args.atol_floor)
                    if args.atol_mode == "state_relative"
                    else args.atol
                ),
                "max_step": args.max_step, "dense_output": True,
            }
            if args.method in ("BDF", "Radau"):
                solve_options["jac_sparsity"] = system.jacobian_sparsity()
            sol = solve_ivp(
                lambda theta, y: system.rhs(theta, y, start_current, args.target_current_a, ramp_theta),
                (0.0, total_theta), y0, **solve_options,
            )
            sample_count = int((args.ramp_periods + args.hold_periods) * args.samples_per_period) + 1
            sample_theta = np.linspace(0.0, total_theta, sample_count)
            dense_state = sol.sol
            states = dense_state(sample_theta) if dense_state is not None else np.empty((y0.size, 0))
            integrator = {"success": bool(sol.success), "message": sol.message,
                          "nfev": sol.nfev, "njev": sol.njev, "nlu": sol.nlu,
                "steps": int(sol.nfev), "newton_iterations": None,
                "step_reductions": None}
    data = make_observables(
        system,
        sample_theta,
        states,
        out_port=args.out_port,
        start_current=start_current,
        target_current=args.target_current_a,
        ramp_theta=ramp_theta,
    )
    data["mu"] = np.asarray([system.source(x, start_current, args.target_current_a, ramp_theta)[system.pump_node] for x in sample_theta])
    data["source_current_a"] = np.array(data["mu"], copy=True)
    data["time_s"] = sample_theta / system.omega
    data["phase_winding_cycles"] = phase_winding_series(system, sample_theta, states)
    if not (compact_output and args.method == "implicit_trapezoid"):
        bounded_scalar_theta = sample_theta
        bounded_scalar_output_voltage = data["output_voltage_v"]
    if compact_output:
        stride = max(1, sample_theta.size // 256)
        np.savez_compressed(
            outdir / "td_compact.npz",
            theta=sample_theta[::stride],
            max_abs_sin_phi=data["max_abs_sin_phi"][::stride],
            max_abs_phi=data["max_abs_phi"][::stride],
            min_cos_phi=data["min_cos_phi"][::stride],
            strongest_branch=data["strongest_branch"][::stride],
            state_norm=data["state_norm"][::stride],
            source_current_a=data["source_current_a"][::stride],
            phase_winding_cycles=data["phase_winding_cycles"][::stride],
            output_voltage_v=data["output_voltage_v"][::stride],
            shunt_power_w=data["shunt_power_w"][::stride],
            output_voltage_theta=bounded_scalar_theta,
            output_voltage_trace_v=bounded_scalar_output_voltage,
        )
    else:
        observable_tmp = outdir / "transient_observables.tmp.npz"
        np.savez_compressed(observable_tmp, **data)
        observable_tmp.replace(outdir / "transient_observables.npz")
        with (outdir / "transient_observables.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))
    hold_start = 2.0 * math.pi * args.ramp_periods
    if compact_output and args.method == "implicit_trapezoid":
        strobe_theta = bounded_strobe_theta
        strobe = bounded_strobe
    else:
        strobe_theta = np.arange(hold_start, total_theta + 0.1, 2.0 * math.pi)
        strobe_states = dense_state(strobe_theta)
        strobe = stroboscopic_diagnostics(system, strobe_theta, strobe_states, args.hold_periods)
    spectrum_periods = min(args.hold_periods, int(getattr(args, "spectrum_periods", 20)))
    late_sample_count = 2048
    if compact_output:
        late_sample_count = max(64, int(spectrum_periods * args.samples_per_period))
    last_theta = np.linspace(
        total_theta - 2.0 * math.pi * spectrum_periods,
        total_theta, late_sample_count,
    )
    if compact_output and args.method == "implicit_trapezoid":
        pump_flux = np.interp(last_theta, bounded_scalar_theta, bounded_scalar_pump_flux)
        phase_theta = bounded_history_theta
        phase_states = bounded_history_states
    else:
        last_states = dense_state(last_theta)
        pump_flux = np.asarray([
            system.phi0 * system.unpack(last_states[:, i])[0][system.pump_node]
            for i in range(last_states.shape[1])
        ])
        phase_theta = last_theta
        phase_states = last_states
    centered = pump_flux - np.mean(pump_flux)
    fft = np.fft.rfft(centered) / centered.size
    freq = np.fft.rfftfreq(centered.size, d=(last_theta[1] - last_theta[0]) / system.omega) / 1e9
    spectrum = {"frequency_ghz": freq, "amplitude": np.abs(fft)}
    np.savez_compressed(outdir / "late_time_spectrum.npz", **spectrum)
    phase_series = np.asarray([
        system.circuit.Bphi.T @ system.unpack(phase_states[:, i])[0]
        for i in range(phase_states.shape[1])
    ])
    unwrapped_phase = np.unwrap(phase_series, axis=0)
    phase_velocity = np.diff(unwrapped_phase, axis=0) / np.diff(phase_theta)[:, None] * system.omega
    mean_phase_velocity = float(np.mean(phase_velocity))
    phase_winding = float(np.mean(unwrapped_phase[-1] - unwrapped_phase[0]) / (2.0 * math.pi))
    if not compact_output:
        np.savez_compressed(outdir / "late_time_phase.npz", theta=last_theta, phase=phase_series, unwrapped_phase=unwrapped_phase)
    recurrence_classification = classify_state(
        strobe, mean_phase_velocity, bool(integrator["success"]), phase_winding
    )
    decay = decay_aware_stroboscopic_classification(strobe)
    recurrence_classification = classify_td_result({
        "classification": recurrence_classification,
        "decay_aware": decay,
    })
    envelope_start_period = max(int(args.ramp_periods), 100)
    envelope_classification = max_abs_phi_envelope_classification(
        data, envelope_start_period
    )
    output_voltage_nonzero = bool(
        np.any(np.isfinite(data["output_voltage_v"])
               & (np.abs(data["output_voltage_v"]) > 0.0))
    )
    if not output_voltage_nonzero:
        raise AssertionError("output-port voltage trace is identically zero")
    max_abs_phi = float(np.nanmax(data["max_abs_phi"])) if data["max_abs_phi"].size else 0.0
    blowup = max_abs_phi > 5.0
    if blowup:
        envelope_classification = dict(envelope_classification)
        envelope_classification["class"] = "BLOWUP"
        envelope_classification["blowup_threshold_rad"] = 5.0
    classification = envelope_classification["class"]
    branch_transfer = None
    projection_forced = bool(getattr(args, "force_periodic_projection", False))
    if (recurrence_classification == "PERIOD_1" or projection_forced) and not bool(
        getattr(args, "skip_projection", False)
    ):
        checkpoint_mode_array = np.asarray(
            hb_report["metadata"]["pump_modes"], dtype=int
        )
        if getattr(args, "projection_all_modes", False):
            projection_modes = np.arange(
                0, int(np.max(checkpoint_mode_array)) + 1, dtype=int
            )
        elif getattr(args, "projection_dynamic_dc", False):
            projection_modes = np.asarray(
                [0, *[int(mode) for mode in checkpoint_mode_array]], dtype=int
            )
        else:
            projection_modes = checkpoint_mode_array
        branch_transfer = project_periodic_state(
            system, dense_state, float(sample_theta[-1]),
            projection_modes,
            args.target_current_a,
            projection_periods=getattr(args, "projection_periods", 5),
            samples_per_period=getattr(args, "projection_samples_per_period", 64),
            solve_hb=not bool(getattr(args, "projection_only", False)),
            preconditioner=getattr(args, "projection_preconditioner", "real_coupled"),
        )
        if branch_transfer is not None and "hb_state" in branch_transfer:
            projected_state = np.asarray(branch_transfer.pop("hb_state"))
            np.savez_compressed(
                outdir / "td_projected_state.npz",
                X_real=projected_state.real,
                X_imag=projected_state.imag,
                modes=projection_modes,
                pump_current_a=args.target_current_a,
            )
            branch_transfer["projected_state_path"] = str(
                outdir / "td_projected_state.npz"
            )
        if branch_transfer is not None:
            branch_transfer["projection_modes"] = projection_modes.tolist()
            branch_transfer["projection_only"] = bool(
                getattr(args, "projection_only", False)
            )
            branch_transfer["projection_forced"] = (
                projection_forced and recurrence_classification != "PERIOD_1"
            )
    if blowup:
        final_status = "BLOWUP"
        blocker_reason = "max_abs_phi exceeded 5 rad"
    elif not integrator["success"]:
        final_status = "TRANSIENT_NUMERICAL_BLOCKER"
        blocker_reason = integrator["message"]
    elif branch_transfer is not None and branch_transfer["hb_converged"]:
        if branch_transfer.get("projection_forced", False):
            final_status = "FORCED_PROJECTED_HB_ROOT"
        else:
            final_status = "HIGH_DRIVE_HB_BRANCH_FOUND"
        blocker_reason = None
    elif classification == "NON_GROWING_MAX_ABS_PHI":
        final_status = "VALIDATED_PERIOD1_TD"
        blocker_reason = None
    elif classification == "GROWING_MAX_ABS_PHI":
        final_status = "REPRODUCIBLE_PHYSICAL_TRANSITION"
        blocker_reason = None
    else:
        final_status = "TRANSIENT_NUMERICAL_BLOCKER"
        blocker_reason = "post-ramp max_abs_phi envelope slope is unresolved"
    if not compact_output:
        plot_results(outdir, data, spectrum)
    checkpoint_grid = HarmonicGrid(
        np.asarray(hb_report["metadata"]["pump_modes"]), 40, system.omega
    )
    checkpoint_problem = FullPumpProblem(
        system.circuit.C, system.circuit.G, system.circuit.K,
        system.circuit.Bphi, make_branch_law(system.circuit), checkpoint_grid,
        system.pump_node, checkpoint_current, dc_branch_flux=dc_flux,
    )
    result = {
        "audit": audit.__dict__, "checkpoint": str(args.checkpoint),
        "hb_validation": validation,
        "start_current_a": start_current, "target_current_a": args.target_current_a,
        "initialization_mode": initialization_mode,
        "initialization_source": (
            "zero_pump_equilibrium_q0_p0"
            if initialization_mode == "zero_pump_equilibrium"
            else "same_target_td_restart"
            if restart_path is not None
            else "validated_hb_periodic_waveform"
        ),
        "previous_target_restart_used": False,
        "same_target_restart_used": bool(restart_path is not None),
        "source_telemetry": {
            "initial_source_current_a": float(data["source_current_a"][0]),
            "final_source_current_a": float(data["source_current_a"][-1]),
            "target_source_current_a": float(args.target_current_a),
            "ramp_duration_periods": int(args.ramp_periods),
            "hold_duration_periods": int(args.hold_periods),
        },
        "ramp_periods": args.ramp_periods, "hold_periods": args.hold_periods,
        "method": args.method,
        "rtol": args.rtol, "atol": args.atol,
        "atol_mode": args.atol_mode, "atol_floor": args.atol_floor,
        "integrator": integrator,
        "classification": classification,
        "recurrence_classification": recurrence_classification,
        "envelope_classification": envelope_classification,
        "envelope_slope_window_start_period": envelope_start_period,
        "max_abs_phi": max_abs_phi,
        "output_voltage_nonzero": output_voltage_nonzero,
        "integrator_health": {
            "steps": integrator.get("steps"),
            "newton_iterations": integrator.get("newton_iterations"),
            "newton_iterations_per_step": (
                float(integrator["newton_iterations"]) / float(integrator["steps"])
                if integrator.get("steps") and integrator.get("newton_iterations") is not None else None
            ),
            "step_reductions": integrator.get("step_reductions"),
        },
        "rcsj_shunt_power": {
            "mean_w": float(np.mean(data["shunt_power_w"])) if data["shunt_power_w"].size else 0.0,
            "hold_mean_w": float(np.mean(data["shunt_power_w"][data["theta"] / (2.0 * math.pi) >= args.ramp_periods])) if np.any(data["theta"] / (2.0 * math.pi) >= args.ramp_periods) else 0.0,
            "max_w": float(np.max(data["shunt_power_w"])) if data["shunt_power_w"].size else 0.0,
            "pump_reference_w": float(0.5 * args.target_current_a**2 * 50.0),
            "hold_fraction": float(np.mean(data["shunt_power_w"][data["theta"] / (2.0 * math.pi) >= args.ramp_periods]) / (0.5 * args.target_current_a**2 * 50.0)) if np.any(data["theta"] / (2.0 * math.pi) >= args.ramp_periods) and args.target_current_a != 0.0 else 0.0,
        },
        "stroboscopic": strobe,
        "checkpoint_diagnostics": checkpoint_stroboscopic_diagnostics(strobe),
        "decay_aware": decay,
        "mean_phase_velocity_rad_s": mean_phase_velocity,
        "mean_phase_winding_cycles": phase_winding,
        "branch_transfer": branch_transfer,
        "final_status": final_status, "blocker_reason": blocker_reason,
        "hb_checkpoint_summary": summarize_solution(checkpoint_problem, checkpoint_X),
        "transient_restart": str(restart_path) if restart_path is not None else None,
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "g1_current_79" / "pass" / "points" / "point_0012_p_m19p6842dbm_fp_7p9ghz" / "pump")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "h1_79")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--dc-flux-over-phi0", type=float, default=0.0)
    parser.add_argument("--target-current-a", type=float, default=1.6e-5)
    parser.add_argument("--ramp-periods", type=int, default=40)
    parser.add_argument("--hold-periods", type=int, default=40)
    parser.add_argument("--samples-per-period", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument(
        "--atol-mode",
        choices=("scalar", "state_relative"),
        default="scalar",
        help=(
            "Use a per-state absolute tolerance proportional to the initial "
            "state for adaptive BDF/Radau integration."
        ),
    )
    parser.add_argument(
        "--atol-floor",
        type=float,
        default=1e-12,
        help="Minimum state scale used by --atol-mode state_relative.",
    )
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument(
        "--min-step-theta", type=float, default=1.0 / 32.0,
        help="Minimum implicit-trapezoid step used after adaptive Newton retries.",
    )
    parser.add_argument("--method", choices=("RK45", "RK23", "BDF", "Radau", "implicit_euler", "implicit_trapezoid"), default="implicit_trapezoid")
    parser.add_argument("--max-newton", type=int, default=12)
    parser.add_argument("--checkpoint-periods", type=int, default=10)
    parser.add_argument(
        "--projection-periods", type=int, default=5,
        help="Number of final pump periods averaged for HB projection.",
    )
    parser.add_argument(
        "--projection-samples-per-period", type=int, default=64,
        help="Independent Fourier projection samples per pump period.",
    )
    parser.add_argument(
        "--transient-restart", type=Path, default=None,
        help="resume a local TD bridge from a transient_restart.npz checkpoint",
    )
    parser.add_argument(
        "--initialization-mode",
        choices=("hb_periodic", "zero_pump_equilibrium"),
        default="hb_periodic",
        help="TD initialization source; zero_pump_equilibrium is independent turn-on mode",
    )
    parser.add_argument(
        "--compact-output", action="store_true",
        help=(
            "store only decimated observables and restart checkpoints; useful "
            "for long fixed-drive continuation holds"
        ),
    )
    parser.add_argument(
        "--spectrum-periods", type=int, default=20,
        help="number of final pump periods used for the late-time FFT export",
    )
    parser.add_argument(
        "--compact-sample-count", type=int, default=256,
        help="maximum number of uniformly sampled compact TD states",
    )
    parser.add_argument(
        "--compact-history-states", type=int, default=1024,
        help="maximum number of late-time full TD states retained for diagnostics",
    )
    parser.add_argument(
        "--skip-projection", action="store_true",
        help="skip optional TD-to-HB projection after classification",
    )
    parser.add_argument(
        "--force-periodic-projection", action="store_true",
        help=(
            "diagnostic only: project an unresolved TD endpoint into PERIOD1 HB "
            "and report the result without treating it as a physical handoff"
        ),
    )
    parser.add_argument(
        "--projection-all-modes", action="store_true",
        help=(
            "diagnostic only: include mode 0 and all positive modes through "
            "the checkpoint maximum in the forced HB projection"
        ),
    )
    parser.add_argument(
        "--projection-dynamic-dc", action="store_true",
        help=(
            "diagnostic only: prepend a dynamic mode-0 coefficient to the "
            "checkpoint harmonic basis"
        ),
    )
    parser.add_argument(
        "--projection-only", action="store_true",
        help=(
            "diagnostic only: compute the requested TD-to-HB projection and "
            "skip the HB Newton/Krylov correction"
        ),
    )
    parser.add_argument(
        "--projection-preconditioner",
        choices=(
            "real_coupled", "spectral_coupled", "spectral_banded",
            "spectral_banded_wide",
            "mean_tangent", "mean_tangent_dc_safe", "linear",
        ),
        default="real_coupled",
        help="preconditioner for the optional diagnostic HB projection",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit_only:
        print(json.dumps(audit_circuit(args.circuit_dir).__dict__, indent=2))
        return 0
    result = run_experiment(args)
    print(json.dumps({"classification": result["classification"], "integrator": result["integrator"], "stroboscopic": result.get("stroboscopic"), "decay_aware": result.get("decay_aware")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
