"""Run Guarcello's known-time-level banded algorithm on JC circuits.

The Josephson current is evaluated at the known time level, so the node matrix
is constant and factored once in natural-order banded storage. The JC
topologies are loaded from their exported matrices and are not projected onto
an unrelated sparse implicit engine.

python scripts/chaos/run_guarcello_jc_phase5.py `
    --device both `
    --dt-norm 0.01 `
    --tmax-norm 20000 `
    --output outputs/chaos/phase5 `
    --per-point-budget-s 900
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import eigsh
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.h1_transient_branch_transfer import (
    build_system,
    implicit_trapezoid_ramp_bounded,
    make_observables,
    stroboscopic_diagnostics,
)
from twpa_solver.builders.ipm import LossSpec, build_matrices
from twpa_solver.core import load_circuit
from twpa_solver.core.constants import PHI0_REDUCED
from twpa_solver.core.linear import solve_linear_scattering
from twpa_solver.design import compile_design, load_design
from twpa_solver.loss import pump_line_loss_model
from twpa_solver.ports import port_available_power_w
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
WIDEBAND_HALF_BANDWIDTH_HZ = 25.0e6
PUMP_NOTCH_HALF_WIDTH_HZ = 2.0e6


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    circuit_dir: str
    node_count: int
    branch_count: int
    ic_min_a: float
    ic_max_a: float
    ic_median_a: float
    cj_nominal_f: float
    cg_nominal_f: float
    resonator_period: int
    has_parallel_geometric_inductor: bool
    profile_is_nonuniform: bool
    pump_ghz: float
    signal_ghz: float
    port_network: str
    source_path: str
    natural_bandwidth: int
    rcm_bandwidth: int
    selected_bandwidth: int
    selected_ordering: str
    pump_port: int
    pump_output_port: int
    signal_source_port: int
    signal_output_port: int
    cj_min_f: float
    cj_max_f: float
    cg_min_f: float
    cg_max_f: float
    dc_flux_bias_present: bool
    dc_flux_bias_source: str
    dc_bias_current_a: float
    external_flux_fraction: float
    beta_l: float
    phi_ext_rad: float
    phi_dc_rad: float
    dc_bias_convention: str

    @property
    def omega_plasma(self) -> float:
        return math.sqrt(self.ic_median_a / (PHI0_REDUCED * self.cj_nominal_f))


@dataclass(frozen=True)
class JcDevice:
    name: str
    n_nodes: int
    C: sp.csr_matrix
    G: sp.csr_matrix
    K: sp.csr_matrix
    Bphi: sp.csr_matrix
    Ic: np.ndarray
    Lj: np.ndarray
    Cj: np.ndarray
    Cg: np.ndarray
    phi0: float
    natural_bandwidth: int
    rcm_bandwidth: int
    selected_bandwidth: int
    selected_ordering: str
    permutation: np.ndarray
    ic_uniform: bool
    pump_node: int
    pump_output_node: int
    signal_node: int
    output_node: int
    has_parallel_geometric_inductor: bool
    implicit_linear_stiffness: bool
    dc_bias_current_a: float
    phi_dc_rad: float


def _load_sparse(path: Path) -> sp.csr_matrix:
    values = np.load(path, allow_pickle=True)
    return sp.csr_matrix(
        (values["data"], values["indices"], values["indptr"]),
        shape=tuple(values["shape"]),
    )


def _array_path(circuit_dir: Path) -> Path:
    for name in ("ipm_arrays.npz", "arrays.npz"):
        path = circuit_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no persisted array file in {circuit_dir}")


def _load_device_arrays(circuit_dir: Path) -> dict[str, np.ndarray]:
    with np.load(_array_path(circuit_dir), allow_pickle=True) as arrays:
        return {name: np.asarray(arrays[name]) for name in arrays.files}


def _assembled_pattern(
    C: sp.csr_matrix, G: sp.csr_matrix, K: sp.csr_matrix, Bphi: sp.csr_matrix,
) -> sp.csr_matrix:
    pattern = (K != 0).astype(np.int8)
    pattern = pattern + (C != 0).astype(np.int8)
    pattern = pattern + (G != 0).astype(np.int8)
    pattern = pattern + ((Bphi @ Bphi.T) != 0).astype(np.int8)
    return pattern.tocsr()


def _matrix_bandwidth(matrix: sp.spmatrix) -> int:
    coo = matrix.tocoo()
    if coo.nnz == 0:
        return 0
    return int(np.max(np.abs(coo.row - coo.col)))


def _ordering_for_pattern(
    pattern: sp.csr_matrix,
) -> tuple[int, int, str, np.ndarray, int]:
    natural = _matrix_bandwidth(pattern)
    permutation = reverse_cuthill_mckee(pattern, symmetric_mode=True)
    rcm_pattern = pattern[permutation][:, permutation]
    rcm = _matrix_bandwidth(rcm_pattern)
    if natural <= rcm:
        return natural, rcm, "natural", np.arange(pattern.shape[0], dtype=np.int64), natural
    return natural, rcm, "rcm", permutation.astype(np.int64), rcm


def _profile_from_source(
    name: str, n_branches: int, arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    cj = np.asarray(arrays.get("Cj", np.empty(0)), dtype=float).reshape(-1)
    cg = np.asarray(arrays.get("Cg", np.empty(0)), dtype=float).reshape(-1)
    if cj.size == n_branches and cg.size == n_branches:
        return cj, cg
    if name != "rf_squid_2393_3wm":
        defaults = {
            "jc_jtwpa": (55.0e-15, 45.0e-15),
            "jc_fqjtwpa": (40.0e-15, 76.6e-15),
        }
        if name == "ipm_2c_fixed":
            metadata_path = ROOT / "designs" / name / "design_resolved.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                parameters = metadata.get("parameters", {})
                cj_value = float(parameters["Cj"])
                cg_value = float(parameters["Cg"])
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                raise ValueError(
                    "ipm_2c_fixed requires Cj/Cg arrays or scalar values in "
                    "design_resolved.json"
                ) from error
            return np.full(n_branches, cj_value), np.full(n_branches, cg_value)
        if name not in defaults:
            raise ValueError(f"missing per-cell Cj/Cg arrays for {name}")
        cj_value, cg_value = defaults[name]
        return np.full(n_branches, cj_value), np.full(n_branches, cg_value)
    source = ROOT / "designs" / "rf_squid_2393_3wm.yaml"
    text = source.read_text(encoding="utf-8")

    def scalar(name: str) -> float:
        match = re.search(rf"^\s*{name}:\s*([^#\s]+)", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing {name} in {source}")
        return float(match.group(1))

    cj = np.full(n_branches, scalar("Cj"), dtype=float)
    pattern = np.asarray(
        [scalar("C1"), scalar("C2"), scalar("C1"), scalar("C3")],
        dtype=float,
    )
    counts = np.asarray([6, 6, 6, 6], dtype=int)
    period = np.repeat(pattern, counts)
    cg = np.resize(period, n_branches)
    return cj, cg


def _built_element_records(circuit_dir: Path) -> list[dict[str, Any]]:
    path = circuit_dir / "design_resolved.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in payload.get("elements", []) if isinstance(item, dict)]


def _has_parallel_geometric_inductor(circuit_dir: Path) -> bool:
    """Detect an RF-SQUID shunt from the built circuit, not its device name."""
    return any(
        item.get("role") == "rf_squid_lpar" and float(item.get("value", 0.0)) > 0.0
        for item in _built_element_records(circuit_dir)
    )


def _dc_flux_bias_metadata(circuit_dir: Path) -> tuple[bool, str]:
    """Report whether the built source contains an explicit DC flux/bias term."""
    if circuit_dir.name == "rf_squid_2393_3wm":
        return True, "runtime DC current source at port 1"
    records = _built_element_records(circuit_dir)
    bias_roles = {"dc_flux", "flux_bias", "bias_current", "dc_bias"}
    if any(item.get("role") in bias_roles for item in records):
        return True, "built element role"
    source = circuit_dir / "design_resolved.json"
    if source.exists():
        text = source.read_text(encoding="utf-8").lower()
        if "flux_bias" in text or "dc_flux" in text or "bias_current" in text:
            return True, "built design metadata"
    return False, "absent from YAML and built element list"


RF_SQUID_EXTERNAL_FLUX_FRACTION = 0.33


def rf_squid_bias_metadata(circuit_dir: Path) -> dict[str, Any]:
    """Return the runtime port bias and self-consistent RF-SQUID phase."""
    resolved = resolve_device_directory(circuit_dir)
    if resolved.name != "rf_squid_2393_3wm":
        return {
            "dc_bias_current_a": 0.0,
            "external_flux_fraction": 0.0,
            "beta_l": 0.0,
            "phi_ext_rad": 0.0,
            "phi_dc_rad": 0.0,
            "dc_bias_convention": "none",
        }
    parameters = json.loads(
        (resolved / "design_resolved.json").read_text(encoding="utf-8")
    ).get("parameters", {})
    lm = float(parameters["Lm"])
    ic = float(parameters["Ic"])
    phi_ext = RF_SQUID_EXTERNAL_FLUX_FRACTION * 2.0 * math.pi
    beta_l = lm * ic / PHI0_REDUCED
    phi_dc = brentq(
        lambda phase: phase - phi_ext + beta_l * math.sin(phase),
        phi_ext - beta_l - 0.5,
        phi_ext + beta_l + 0.5,
    )
    # Keep the transient path tied to the production HB flux convention.
    # The helper returns branch flux in webers; convert it back to reduced
    # phase and assert that both implementations use the same fixed point.
    from scripts.run_gain_map import rf_squid_dc_branch_flux_from_external_fraction

    circuit = load_circuit(resolved)
    hb_branch_flux = rf_squid_dc_branch_flux_from_external_fraction(
        resolved, circuit, RF_SQUID_EXTERNAL_FLUX_FRACTION,
    )
    hb_phi_dc = float(hb_branch_flux[0] / circuit.phi0)
    if not math.isclose(phi_dc, hb_phi_dc, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(
            "RF-SQUID phase convention mismatch: "
            f"time-domain={phi_dc:.16g}, HB={hb_phi_dc:.16g}"
        )
    # The applied source current represents Phi_ext = Lm * Idc. The internal
    # junction phase is the self-consistent solution above, not phi_ext.
    idc = phi_ext * PHI0_REDUCED / lm
    return {
        "dc_bias_current_a": float(idc),
        "external_flux_fraction": RF_SQUID_EXTERNAL_FLUX_FRACTION,
        "beta_l": float(beta_l),
        "phi_ext_rad": float(phi_ext),
        "phi_dc_rad": float(phi_dc),
        "dc_bias_convention": (
            "self_consistent_uniform_branch_phase_offset=phi_ext-"
            "beta_L*sin(phi_dc); HB helper verified; "
            "legacy Idc=Phi_ext/Lm retained as metadata only"
        ),
    }


def _build_yaml_design(source_path: Path) -> Path:
    output = ROOT / "outputs" / "chaos" / "phaseC" / "build" / source_path.stem
    if (output / "C.npz").exists() and _array_path(output).exists():
        arrays = _load_device_arrays(output)
        if "Cj" not in arrays or "Cg" not in arrays:
            cj, cg = _profile_from_source(source_path.stem, arrays["Ic"].size, arrays)
            arrays.update({"Cj": cj, "Cg": cg})
            np.savez(output / "ipm_arrays.npz", **arrays)
        return output
    output.mkdir(parents=True, exist_ok=True)
    design = compile_design(load_design(source_path), coupler_mode="auto", strict=True)
    matrices = build_matrices(design.elements, LossSpec(0.0))
    for name in ("C", "G", "K", "Bphi"):
        sp.save_npz(output / f"{name}.npz", matrices[name].tocsr())
    arrays = {
        "nodes": np.asarray(matrices["nodes"]),
        "Ic": np.asarray(matrices["Ic"], dtype=float),
        "Lj": np.asarray(matrices["Lj"], dtype=float),
        "phi0_reduced": np.asarray([PHI0_REDUCED]),
        "port_numbers": np.asarray(sorted(matrices["port_vectors"]), dtype=np.int64),
        "port_indices": np.asarray(
            [matrices["port_vectors"][port] for port in sorted(matrices["port_vectors"])]
        ),
    }
    cj, cg = _profile_from_source(source_path.stem, arrays["Ic"].size, arrays)
    arrays.update({"Cj": cj, "Cg": cg})
    np.savez(
        output / "ipm_arrays.npz", **arrays,
    )
    (output / "design_summary.json").write_text(
        json.dumps({"name": design.name, "nodes": int(matrices["C"].shape[0]),
                    "branches": int(matrices["Bphi"].shape[1]),
                    "ports": {str(k): int(v) for k, v in matrices["ports"].items()}}, indent=2),
        encoding="utf-8",
    )
    return output


def resolve_device_directory(circuit_dir: Path) -> Path:
    """Return a matrix directory, building a YAML design under phaseC output."""
    if circuit_dir.is_file() and circuit_dir.suffix.lower() in {".yaml", ".yml"}:
        return _build_yaml_design(circuit_dir)
    if not circuit_dir.is_dir():
        raise FileNotFoundError(f"device source does not exist: {circuit_dir}")
    return circuit_dir


def phase_c_source_path(name: str) -> Path:
    """Return the checked-in source for a Phase C device."""
    if name == "rf_squid_2393_3wm":
        return ROOT / "designs" / f"{name}.yaml"
    if name == "ipm_2c_fixed":
        return ROOT / "designs" / name
    raise ValueError(f"not a Phase C device: {name}")


def load_jc_device(circuit_dir: Path) -> JcDevice:
    """Load one topology and select the measured narrowest band ordering."""
    circuit_dir = resolve_device_directory(circuit_dir)
    C = _load_sparse(circuit_dir / "C.npz")
    G = _load_sparse(circuit_dir / "G.npz")
    K = _load_sparse(circuit_dir / "K.npz")
    Bphi = _load_sparse(circuit_dir / "Bphi.npz")
    pattern = _assembled_pattern(C, G, K, Bphi)
    natural_bandwidth, rcm_bandwidth, ordering, permutation, selected_bandwidth = _ordering_for_pattern(pattern)
    arrays = _load_device_arrays(circuit_dir)
    Ic = np.asarray(arrays["Ic"], dtype=float).reshape(-1)
    Lj = np.asarray(arrays.get("Lj", np.empty(0)), dtype=float).reshape(-1)
    Cj, Cg = _profile_from_source(circuit_dir.name, Ic.size, arrays)
    if C.shape[0] != C.shape[1] or Bphi.shape[0] != C.shape[0]:
        raise ValueError("JC matrices have incompatible node dimensions")
    if ordering == "rcm":
        C = C[permutation][:, permutation].tocsr()
        G = G[permutation][:, permutation].tocsr()
        K = K[permutation][:, permutation].tocsr()
        Bphi = Bphi[permutation, :].tocsr()
    selected_actual = _matrix_bandwidth(_assembled_pattern(C, G, K, Bphi))
    if selected_actual != selected_bandwidth:
        raise RuntimeError(
            f"{circuit_dir.name}: selected bandwidth changed from "
            f"{selected_bandwidth} to {selected_actual} after ordering"
        )
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=np.int64)
    port_numbers = np.asarray(arrays["port_numbers"], dtype=int)
    port_indices = np.asarray(arrays["port_indices"], dtype=int)
    ports = {int(number): int(inverse[index]) for number, index in zip(port_numbers, port_indices)}
    name = circuit_dir.name
    if name == "ipm_2c_fixed":
        required = {1, 2, 3, 4}
        if not required.issubset(ports):
            raise ValueError(f"ipm_2c_fixed requires ports 1,2,3,4; found {ports}")
        pump_port, pump_output_port, signal_port, output_port = 4, 3, 1, 2
    else:
        if not {1, 2}.issubset(ports):
            raise ValueError(f"{name} requires ports 1 and 2; found {ports}")
        pump_port, pump_output_port, signal_port, output_port = 1, 2, 1, 2
    bias = rf_squid_bias_metadata(circuit_dir)
    return JcDevice(
        name=circuit_dir.name,
        n_nodes=C.shape[0], C=C, G=G, K=K, Bphi=Bphi,
        Ic=Ic, Lj=Lj, Cj=Cj, Cg=Cg, phi0=float(np.asarray(arrays["phi0_reduced"]).reshape(-1)[0]),
        natural_bandwidth=natural_bandwidth, rcm_bandwidth=rcm_bandwidth,
        selected_bandwidth=selected_bandwidth, selected_ordering=ordering,
        permutation=permutation,
        ic_uniform=bool(np.ptp(Ic) == 0.0),
        pump_node=ports[pump_port], pump_output_node=ports[pump_output_port],
        signal_node=ports[signal_port],
        output_node=ports[output_port],
        has_parallel_geometric_inductor=_has_parallel_geometric_inductor(circuit_dir),
        # K is linear and belongs in the constant factored matrix. The old
        # explicit treatment remains available only through an explicit
        # override in the integration helpers for regression comparisons.
        implicit_linear_stiffness=True,
        dc_bias_current_a=float(bias["dc_bias_current_a"]),
        phi_dc_rad=float(bias["phi_dc_rad"]),
    )


def _jc_band_matrix(matrix: sp.spmatrix, bandwidth: int) -> np.ndarray:
    coo = matrix.tocoo()
    measured = int(np.max(np.abs(coo.row - coo.col)))
    if measured > bandwidth:
        raise RuntimeError(f"assembled bandwidth {measured} exceeds {bandwidth}")
    band = np.zeros((2 * bandwidth + 1, matrix.shape[0]), dtype=float)
    for row, col, value in zip(coo.row, coo.col, coo.data):
        band[bandwidth + row - col, col] += value
    return band


def _factor_banded_lu(matrix: sp.spmatrix, bandwidth: int) -> np.ndarray:
    """Factor a natural-order band matrix once, retaining its band storage."""
    coo = matrix.tocoo()
    band = np.zeros((2 * bandwidth + 1, matrix.shape[0]), dtype=np.float64)
    for row, col, value in zip(coo.row, coo.col, coo.data):
        band[bandwidth + row - col, col] += value
    n = matrix.shape[0]
    for k in range(n):
        pivot = band[bandwidth, k]
        if abs(pivot) <= np.finfo(float).tiny:
            raise np.linalg.LinAlgError(f"zero pivot in band factorization at {k}")
        i_stop = min(n, k + bandwidth + 1)
        j_stop = min(n, k + bandwidth + 1)
        for i in range(k + 1, i_stop):
            multiplier = band[bandwidth + i - k, k] / pivot
            band[bandwidth + i - k, k] = multiplier
            for j in range(k + 1, j_stop):
                if abs(i - j) <= bandwidth:
                    band[bandwidth + i - j, j] -= multiplier * band[bandwidth + k - j, j]
    return band


def _unpack_banded_factor(
    lu: np.ndarray, bandwidth: int,
) -> tuple[int, tuple[np.ndarray, ...]]:
    """Copy LU diagonals into contiguous arrays for fixed-width solves."""
    n = lu.shape[1]
    diagonal = np.array(lu[bandwidth, :], copy=True)
    zeros = np.zeros(n, dtype=np.float64)
    lower = tuple(
        np.array(lu[bandwidth + offset, :], copy=True)
        if offset <= bandwidth else zeros.copy()
        for offset in range(1, 6)
    )
    upper = tuple(
        np.array(lu[bandwidth - offset, :], copy=True)
        if offset <= bandwidth else zeros.copy()
        for offset in range(1, 6)
    )
    kind = bandwidth if bandwidth in (2, 5) else 0
    return kind, (diagonal, *lower, *upper)


@njit(cache=True, fastmath=True, nogil=True)
def _csr_matvec_into(indptr, indices, data, vector, result):
    for row in range(result.size):
        total = 0.0
        for pos in range(indptr[row], indptr[row + 1]):
            total += data[pos] * vector[indices[pos]]
        result[row] = total


@njit(cache=True, fastmath=True, nogil=True)
def _csr_transpose_matvec_into(indptr, indices, data, vector, result):
    result[:] = 0.0
    for row in range(vector.size):
        value = vector[row]
        for pos in range(indptr[row], indptr[row + 1]):
            result[indices[pos]] += data[pos] * value


@njit(cache=True, fastmath=False, nogil=True)
def _solve_banded_lu_into(lu, rhs, solution):
    bandwidth = (lu.shape[0] - 1) // 2
    diagonal = bandwidth
    n = rhs.size
    for i in range(n):
        value = rhs[i]
        for k in range(max(0, i - bandwidth), i):
            value -= lu[diagonal + i - k, k] * solution[k]
        solution[i] = value
    for i in range(n - 1, -1, -1):
        value = solution[i]
        for k in range(i + 1, min(n, i + bandwidth + 1)):
            value -= lu[diagonal + i - k, k] * solution[k]
        solution[i] = value / lu[diagonal, i]


@njit(cache=True, fastmath=False, nogil=True)
def _solve_banded_lu_bw2_into(
    diagonal, lower1, lower2, upper1, upper2, rhs, solution,
):
    n = rhs.size
    for i in range(n):
        value = rhs[i]
        if i >= 2:
            value -= lower2[i - 2] * solution[i - 2]
        if i >= 1:
            value -= lower1[i - 1] * solution[i - 1]
        solution[i] = value
    for i in range(n - 1, -1, -1):
        value = solution[i]
        if i + 1 < n:
            value -= upper1[i + 1] * solution[i + 1]
        if i + 2 < n:
            value -= upper2[i + 2] * solution[i + 2]
        solution[i] = value / diagonal[i]


@njit(cache=True, fastmath=False, nogil=True)
def _solve_banded_lu_bw5_into(
    diagonal,
    lower1, lower2, lower3, lower4, lower5,
    upper1, upper2, upper3, upper4, upper5,
    rhs, solution,
):
    n = rhs.size
    for i in range(n):
        value = rhs[i]
        if i >= 5:
            value -= lower5[i - 5] * solution[i - 5]
        if i >= 4:
            value -= lower4[i - 4] * solution[i - 4]
        if i >= 3:
            value -= lower3[i - 3] * solution[i - 3]
        if i >= 2:
            value -= lower2[i - 2] * solution[i - 2]
        if i >= 1:
            value -= lower1[i - 1] * solution[i - 1]
        solution[i] = value
    for i in range(n - 1, -1, -1):
        value = solution[i]
        if i + 1 < n:
            value -= upper1[i + 1] * solution[i + 1]
        if i + 2 < n:
            value -= upper2[i + 2] * solution[i + 2]
        if i + 3 < n:
            value -= upper3[i + 3] * solution[i + 3]
        if i + 4 < n:
            value -= upper4[i + 4] * solution[i + 4]
        if i + 5 < n:
            value -= upper5[i + 5] * solution[i + 5]
        solution[i] = value / diagonal[i]


@njit(cache=True, fastmath=False, nogil=True)
def _solve_banded_lu_dispatch(
    lu, factor_kind, diagonal,
    lower1, lower2, lower3, lower4, lower5,
    upper1, upper2, upper3, upper4, upper5,
    rhs, solution,
):
    if factor_kind == 2:
        _solve_banded_lu_bw2_into(
            diagonal, lower1, lower2, upper1, upper2, rhs, solution,
        )
    elif factor_kind == 5:
        _solve_banded_lu_bw5_into(
            diagonal,
            lower1, lower2, lower3, lower4, lower5,
            upper1, upper2, upper3, upper4, upper5,
            rhs, solution,
        )
    else:
        _solve_banded_lu_into(lu, rhs, solution)


@njit(cache=True, fastmath=True, nogil=True)
def _solve_banded_lu(lu, rhs):
    """Compatibility wrapper retaining the historical allocating signature."""
    solution = np.empty(rhs.size, dtype=np.float64)
    _solve_banded_lu_into(lu, rhs, solution)
    return solution


def _incidence_endpoints(matrix: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Extract the positive and negative endpoint of every incidence column."""
    csc = matrix.tocsc()
    plus = np.empty(csc.shape[1], dtype=np.int64)
    minus = np.empty(csc.shape[1], dtype=np.int64)
    for branch in range(csc.shape[1]):
        start, stop = csc.indptr[branch], csc.indptr[branch + 1]
        if stop - start != 2:
            raise ValueError("Bphi must have exactly two entries per branch")
        values = csc.data[start:stop]
        if not np.all(np.abs(values) == 1.0):
            raise ValueError("Bphi contains an incidence value other than +/-1")
        if values[0] == 1.0 and values[1] == -1.0:
            plus[branch], minus[branch] = csc.indices[start], csc.indices[start + 1]
        elif values[1] == 1.0 and values[0] == -1.0:
            plus[branch], minus[branch] = csc.indices[start + 1], csc.indices[start]
        else:
            raise ValueError("Bphi branch must contain one +1 and one -1")
    return plus, minus


def _incidence_row_parts(
    node_plus: np.ndarray, node_minus: np.ndarray, n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build row-wise incidence indices from endpoint arrays without matrix data."""
    rows: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for branch, (plus, minus) in enumerate(zip(node_plus, node_minus)):
        rows[int(plus)].append((branch, 1.0))
        rows[int(minus)].append((branch, -1.0))
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for row in range(n_nodes):
        indptr[row + 1] = indptr[row] + len(rows[row])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    signs = np.empty(int(indptr[-1]), dtype=np.float64)
    for row, entries in enumerate(rows):
        start = int(indptr[row])
        for offset, (branch, sign) in enumerate(entries):
            indices[start + offset] = branch
            signs[start + offset] = sign
    return indptr, indices, signs


@njit(cache=True, fastmath=True, nogil=True)
def _integrate_jc_banded_numba_stage_a(
    c_indptr, c_indices, c_data,
    g_indptr, g_indices, g_data,
    k_indptr, k_indices, k_data,
    node_plus, node_minus, row_indptr, row_indices, row_signs, n_branches,
    ic, lower_factor, factor_kind, factor_diagonal,
    lower1, lower2, lower3, lower4, lower5,
    upper1, upper2, upper3, upper4, upper5,
    phi0, pump_current_a, phi_dc_rad, pump_hz,
    signal_current_a, signal_hz, dt_s, n_steps, record_stride,
    pump_node, signal_node, output_node, implicit_linear_stiffness,
    q_previous_initial, q_current_initial,
):
    """Stage-A loop with preallocated solver, derivative, and incidence data."""
    n = c_indptr.size - 1
    q_prev = np.zeros(n, dtype=np.float64)
    q_cur = np.zeros(n, dtype=np.float64)
    q_next = np.empty(n, dtype=np.float64)
    if q_previous_initial.size == n:
        q_prev[:] = q_previous_initial
    if q_current_initial.size == n:
        q_cur[:] = q_current_initial
    out_count = n_steps // record_stride + 1
    times = np.empty(out_count, dtype=np.float64)
    voltage = np.empty(out_count, dtype=np.float64)
    branch_r = np.empty(out_count, dtype=np.float64)
    rhs = np.empty(n, dtype=np.float64)
    work = np.empty(n, dtype=np.float64)
    q_difference = np.empty(n, dtype=np.float64)
    derivative = np.empty(n, dtype=np.float64)
    phase = np.empty(n_branches, dtype=np.float64)
    current = np.empty(n_branches, dtype=np.float64)
    rec = 0
    times[rec] = 0.0
    voltage[rec] = 0.0
    branch_r[rec] = 0.0
    rec += 1
    pump_omega = 2.0 * math.pi * pump_hz
    signal_omega = 2.0 * math.pi * signal_hz
    for step in range(1, n_steps + 1):
        t = step * dt_s
        for j in range(n_branches):
            phase[j] = 0.0
            phase[j] += q_cur[node_plus[j]]
            phase[j] -= q_cur[node_minus[j]]
        for j in range(n_branches):
            phase[j] /= phi0
            current[j] = ic[j] * math.sin(phase[j] + phi_dc_rad)
        rhs[:] = 0.0
        rhs[pump_node] = pump_current_a * math.cos(pump_omega * t)
        if signal_node == pump_node:
            rhs[pump_node] += signal_current_a * math.cos(signal_omega * t)
        else:
            rhs[signal_node] = signal_current_a * math.cos(signal_omega * t)
        if not implicit_linear_stiffness:
            _csr_matvec_into(k_indptr, k_indices, k_data, q_cur, work)
            rhs -= work
        for row in range(n):
            total = 0.0
            for pos in range(row_indptr[row], row_indptr[row + 1]):
                total += row_signs[pos] * current[row_indices[pos]]
            work[row] = total
        rhs -= work
        q_difference[:] = 2.0 * q_cur - q_prev
        _csr_matvec_into(c_indptr, c_indices, c_data, q_difference, work)
        rhs += work / (dt_s * dt_s)
        _csr_matvec_into(g_indptr, g_indices, g_data, q_prev, work)
        rhs += work / (2.0 * dt_s)
        _solve_banded_lu_dispatch(
            lower_factor, factor_kind, factor_diagonal,
            lower1, lower2, lower3, lower4, lower5,
            upper1, upper2, upper3, upper4, upper5,
            rhs, q_next,
        )
        if step % record_stride == 0:
            derivative[:] = (q_next - q_prev) / (2.0 * dt_s)
            times[rec] = t
            voltage[rec] = derivative[output_node]
            maximum = 0.0
            for j in range(n_branches):
                value = abs(math.sin(phase[j] + phi_dc_rad))
                if value > maximum:
                    maximum = value
            branch_r[rec] = maximum
            rec += 1
        q_prev, q_cur, q_next = q_cur, q_next, q_prev
    return times[:rec], voltage[:rec], branch_r[:rec], q_cur


@njit(cache=True, fastmath=True, nogil=True)
def integrate_jc_banded_numba(
    c_indptr, c_indices, c_data,
    g_indptr, g_indices, g_data,
    k_indptr, k_indices, k_data,
    b_indptr, b_indices, b_data, n_branches,
    ic, lower_factor, phi0, pump_current_a, phi_dc_rad, pump_hz, signal_current_a,
    signal_hz, dt_s, n_steps, record_stride, pump_node, signal_node, output_node,
    implicit_linear_stiffness,
    q_initial,
):
    """Compiled Guarcello known-time-level loop using precomputed band factors."""
    n = c_indptr.size - 1
    q_prev = np.zeros(n, dtype=np.float64)
    q_cur = np.zeros(n, dtype=np.float64)
    if q_initial.size == n:
        q_prev[:] = q_initial
        q_cur[:] = q_initial
    out_count = n_steps // record_stride + 1
    times = np.empty(out_count, dtype=np.float64)
    voltage = np.empty(out_count, dtype=np.float64)
    branch_r = np.empty(out_count, dtype=np.float64)
    source = np.zeros(n, dtype=np.float64)
    work = np.zeros(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    q_difference = np.zeros(n, dtype=np.float64)
    phase = np.empty(n_branches, dtype=np.float64)
    current = np.empty(n_branches, dtype=np.float64)
    rec = 0
    times[rec] = 0.0
    voltage[rec] = 0.0
    branch_r[rec] = 0.0
    rec += 1
    pump_omega = 2.0 * math.pi * pump_hz
    signal_omega = 2.0 * math.pi * signal_hz
    for step in range(1, n_steps + 1):
        t = step * dt_s
        _csr_transpose_matvec_into(b_indptr, b_indices, b_data, q_cur, phase)
        phase /= phi0
        for j in range(n_branches):
            current[j] = ic[j] * math.sin(phase[j] + phi_dc_rad)
        source[:] = 0.0
        source[pump_node] = pump_current_a * math.cos(pump_omega * t)
        if signal_node == pump_node:
            source[pump_node] += signal_current_a * math.cos(signal_omega * t)
        else:
            source[signal_node] += signal_current_a * math.cos(signal_omega * t)
        rhs[:] = source
        if not implicit_linear_stiffness:
            _csr_matvec_into(k_indptr, k_indices, k_data, q_cur, work)
            rhs -= work
        _csr_matvec_into(b_indptr, b_indices, b_data, current, work)
        rhs -= work
        q_difference[:] = 2.0 * q_cur - q_prev
        _csr_matvec_into(c_indptr, c_indices, c_data, q_difference, work)
        rhs += work / (dt_s * dt_s)
        _csr_matvec_into(g_indptr, g_indices, g_data, q_prev, work)
        rhs += work / (2.0 * dt_s)
        q_next = _solve_banded_lu(lower_factor, rhs)
        if step % record_stride == 0:
            derivative = (q_next - q_prev) / (2.0 * dt_s)
            times[rec] = t
            voltage[rec] = derivative[output_node]
            maximum = 0.0
            for j in range(n_branches):
                value = abs(math.sin(phase[j] + phi_dc_rad))
                if value > maximum:
                    maximum = value
            branch_r[rec] = maximum
            rec += 1
        q_prev = q_cur
        q_cur = q_next
    return times[:rec], voltage[:rec], branch_r[:rec], q_cur


def _csr_parts(matrix: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = matrix.tocsr()
    return matrix.indptr.astype(np.int64), matrix.indices.astype(np.int64), matrix.data.astype(np.float64)


def _integrate_jc_compiled_stage_a(
    device: JcDevice, *, pump_current_a: float, pump_hz: float,
    signal_current_a: float, signal_hz: float, dt_s: float, n_steps: int,
    record_stride: int, initial_q: np.ndarray | None,
    implicit_linear_stiffness: bool | None = None,
    phi_dc_rad: float | None = None,
    initial_q_previous: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    n = device.n_nodes
    use_implicit_linear_stiffness = (
        device.implicit_linear_stiffness
        if implicit_linear_stiffness is None else bool(implicit_linear_stiffness)
    )
    branch_phase_offset = device.phi_dc_rad if phi_dc_rad is None else float(phi_dc_rad)
    constant = device.C / dt_s**2 + device.G / (2.0 * dt_s)
    if use_implicit_linear_stiffness:
        constant = constant + device.K
    lower = _factor_banded_lu(constant, device.selected_bandwidth)
    factor_kind, factor_arrays = _unpack_banded_factor(
        lower, device.selected_bandwidth,
    )
    parts = [_csr_parts(matrix) for matrix in (device.C, device.G, device.K)]
    node_plus, node_minus = _incidence_endpoints(device.Bphi)
    row_indptr, row_indices, row_signs = _incidence_row_parts(
        node_plus, node_minus, device.n_nodes,
    )
    started = time.perf_counter()
    result = _integrate_jc_banded_numba_stage_a(
        *parts[0], *parts[1], *parts[2], node_plus, node_minus,
        row_indptr, row_indices, row_signs, device.Ic.size,
        device.Ic, lower, factor_kind, *factor_arrays, PHI0_REDUCED,
        pump_current_a, branch_phase_offset,
        pump_hz, signal_current_a,
        signal_hz, dt_s, n_steps, record_stride, device.pump_node,
        device.signal_node, device.output_node,
        use_implicit_linear_stiffness,
        (
            np.zeros(n)
            if initial_q_previous is None
            else np.asarray(initial_q_previous, dtype=np.float64)
        ),
        np.zeros(n) if initial_q is None else np.asarray(initial_q, dtype=np.float64),
    )
    return (*result[:3], time.perf_counter() - started, result[3])


def _compact_banded_matrix(
    matrix: sp.spmatrix, bandwidth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack only populated diagonals, in CSR row-summation order."""
    coo = matrix.tocoo()
    if coo.nnz and int(np.max(np.abs(coo.row - coo.col))) > bandwidth:
        raise ValueError("matrix exceeds the selected runtime bandwidth")
    offsets = np.unique(coo.row - coo.col)[::-1].astype(np.int64)
    values = np.zeros((offsets.size, matrix.shape[0]), dtype=np.float64)
    offset_to_row = {int(offset): index for index, offset in enumerate(offsets)}
    for row, col, value in zip(coo.row, coo.col, coo.data):
        values[offset_to_row[int(row - col)], col] += value
    return offsets, values


def _compact_banded_interior(
    offsets: np.ndarray, n: int,
) -> tuple[int, int]:
    """Return the half-open row span where every compact diagonal is valid."""
    if offsets.size == 0:
        return 0, n
    return max(0, int(np.max(offsets))), min(n, n + int(np.min(offsets)))


@njit(cache=True, fastmath=True, nogil=True)
def _compact_banded_sum_interior(offsets, values, vector, row):
    total = 0.0
    for diagonal in range(offsets.size):
        col = row - offsets[diagonal]
        coefficient = values[diagonal, col]
        if coefficient != 0.0:
            total += coefficient * vector[col]
    return total


@njit(cache=True, fastmath=True, nogil=True)
def _compact_banded_sum_edge(offsets, values, vector, row, n):
    total = 0.0
    for diagonal in range(offsets.size):
        col = row - offsets[diagonal]
        if 0 <= col < n:
            coefficient = values[diagonal, col]
            if coefficient != 0.0:
                total += coefficient * vector[col]
    return total


@njit(cache=True, fastmath=True, nogil=True)
def _compact_banded_sum_difference_interior(
    offsets, values, current, previous, row,
):
    total = 0.0
    for diagonal in range(offsets.size):
        col = row - offsets[diagonal]
        coefficient = values[diagonal, col]
        if coefficient != 0.0:
            total += coefficient * (2.0 * current[col] - previous[col])
    return total


@njit(cache=True, fastmath=True, nogil=True)
def _compact_banded_sum_difference_edge(
    offsets, values, current, previous, row, n,
):
    total = 0.0
    for diagonal in range(offsets.size):
        col = row - offsets[diagonal]
        if 0 <= col < n:
            coefficient = values[diagonal, col]
            if coefficient != 0.0:
                total += coefficient * (2.0 * current[col] - previous[col])
    return total


@njit(cache=True, fastmath=True, nogil=True)
def _integrate_jc_banded_numba_stage_b(
    c_offsets, c_values, c_interior_start, c_interior_stop,
    g_offsets, g_values, g_interior_start, g_interior_stop,
    k_offsets, k_values, k_interior_start, k_interior_stop,
    node_plus, node_minus, row_indptr, row_indices, row_signs, n_branches,
    ic, lower_factor, factor_kind, factor_diagonal,
    lower1, lower2, lower3, lower4, lower5,
    upper1, upper2, upper3, upper4, upper5,
    phi0, pump_current_a, phi_dc_rad, pump_hz,
    signal_current_a, signal_hz, dt_s, inv_dt_sq, inv_two_dt, n_steps,
    record_stride, pump_node, signal_node, output_node,
    implicit_linear_stiffness, q_initial,
):
    """Stage-B loop with fused RHS assembly and banded matrix products."""
    n = c_values.shape[1]
    q_prev = np.zeros(n, dtype=np.float64)
    q_cur = np.zeros(n, dtype=np.float64)
    q_next = np.empty(n, dtype=np.float64)
    if q_initial.size == n:
        q_prev[:] = q_initial
        q_cur[:] = q_initial
    out_count = n_steps // record_stride + 1
    times = np.empty(out_count, dtype=np.float64)
    voltage = np.empty(out_count, dtype=np.float64)
    branch_r = np.empty(out_count, dtype=np.float64)
    rhs = np.empty(n, dtype=np.float64)
    branch_force = np.empty(n, dtype=np.float64)
    q_difference = np.empty(n, dtype=np.float64)
    work = np.empty(n, dtype=np.float64)
    derivative = np.empty(n, dtype=np.float64)
    phase = np.empty(n_branches, dtype=np.float64)
    current = np.empty(n_branches, dtype=np.float64)
    rec = 0
    times[rec] = 0.0
    voltage[rec] = 0.0
    branch_r[rec] = 0.0
    rec += 1
    pump_omega = 2.0 * math.pi * pump_hz
    signal_omega = 2.0 * math.pi * signal_hz
    for step in range(1, n_steps + 1):
        t = step * dt_s
        for j in range(n_branches):
            phase[j] = 0.0
            phase[j] += q_cur[node_plus[j]]
            phase[j] -= q_cur[node_minus[j]]
            phase[j] /= phi0
        for j in range(n_branches):
            current[j] = ic[j] * math.sin(phase[j] + phi_dc_rad)
        for row in range(n):
            total = 0.0
            for pos in range(row_indptr[row], row_indptr[row + 1]):
                total += row_signs[pos] * current[row_indices[pos]]
            branch_force[row] = total
        rhs[:] = 0.0
        rhs[pump_node] = pump_current_a * math.cos(pump_omega * t)
        if signal_node == pump_node:
            rhs[pump_node] += signal_current_a * math.cos(signal_omega * t)
        else:
            rhs[signal_node] = signal_current_a * math.cos(signal_omega * t)
        if not implicit_linear_stiffness:
            for i in range(n):
                if k_interior_start <= i < k_interior_stop:
                    work[i] = _compact_banded_sum_interior(
                        k_offsets, k_values, q_cur, i,
                    )
                else:
                    work[i] = _compact_banded_sum_edge(
                        k_offsets, k_values, q_cur, i, n,
                    )
            rhs -= work
        rhs -= branch_force
        q_difference[:] = 2.0 * q_cur - q_prev
        for i in range(n):
            if c_interior_start <= i < c_interior_stop:
                work[i] = _compact_banded_sum_interior(
                    c_offsets, c_values, q_difference, i,
                )
            else:
                work[i] = _compact_banded_sum_edge(
                    c_offsets, c_values, q_difference, i, n,
                )
        rhs += work * inv_dt_sq
        for i in range(n):
            if g_interior_start <= i < g_interior_stop:
                work[i] = _compact_banded_sum_interior(
                    g_offsets, g_values, q_prev, i,
                )
            else:
                work[i] = _compact_banded_sum_edge(
                    g_offsets, g_values, q_prev, i, n,
                )
        rhs += work * inv_two_dt
        _solve_banded_lu_dispatch(
            lower_factor, factor_kind, factor_diagonal,
            lower1, lower2, lower3, lower4, lower5,
            upper1, upper2, upper3, upper4, upper5,
            rhs, q_next,
        )
        if step % record_stride == 0:
            derivative[:] = (q_next - q_prev) / (2.0 * dt_s)
            times[rec] = t
            voltage[rec] = derivative[output_node]
            maximum = 0.0
            for j in range(n_branches):
                value = abs(math.sin(phase[j] + phi_dc_rad))
                if value > maximum:
                    maximum = value
            branch_r[rec] = maximum
            rec += 1
        q_prev, q_cur, q_next = q_cur, q_next, q_prev
    return times[:rec], voltage[:rec], branch_r[:rec], q_cur


def _integrate_jc_compiled(
    device: JcDevice, *, pump_current_a: float, pump_hz: float,
    signal_current_a: float, signal_hz: float, dt_s: float, n_steps: int,
    record_stride: int, initial_q: np.ndarray | None,
    implicit_linear_stiffness: bool | None = None,
    phi_dc_rad: float | None = None,
    initial_q_previous: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Run the production known-time-level integration path.

    This delegates to the Stage-A loop, which keeps the matrix products in CSR.
    The Stage-B loop packs C, G and K into compact diagonals and fuses the
    right-hand side into one pass; that was expected to be faster and measures
    slower, on both devices and at both bandwidths present in this repository:

        device            original    Stage A    Stage B
        rf_squid (bw 2)      5380       6885       4179  steps/s
        ipm_2c_fixed (bw 5)  4743       6900       3977  steps/s

    A compact diagonal walks one stream per diagonal, each strided by the node
    count, where CSR holds the same few nonzeros per row contiguously in a
    single array, so the packed form trades a cheap indirection for several
    times the number of open memory streams.  Stage B is retained because its
    representation is a reasonable starting point for a blocked or vectorized
    rewrite, but nothing should call it until it is measured faster than this.
    """
    return _integrate_jc_compiled_stage_a(
        device, pump_current_a=pump_current_a, pump_hz=pump_hz,
        signal_current_a=signal_current_a, signal_hz=signal_hz, dt_s=dt_s,
        n_steps=n_steps, record_stride=record_stride, initial_q=initial_q,
        implicit_linear_stiffness=implicit_linear_stiffness,
        phi_dc_rad=phi_dc_rad,
        initial_q_previous=initial_q_previous,
    )


def integrate_jc_banded(
    device: JcDevice,
    *,
    pump_current_a: float,
    pump_hz: float,
    signal_current_a: float,
    signal_hz: float,
    dt_s: float,
    n_steps: int,
    record_stride: int = 20,
    initial_q: np.ndarray | None = None,
    implicit_linear_stiffness: bool | None = None,
    phi_dc_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Integrate JC node fluxes with known-time-level Josephson currents."""
    n = device.n_nodes
    dt = float(dt_s)
    omega = 2.0 * math.pi * pump_hz
    use_implicit_linear_stiffness = (
        device.implicit_linear_stiffness
        if implicit_linear_stiffness is None else bool(implicit_linear_stiffness)
    )
    A = device.C / dt**2 + device.G / (2.0 * dt)
    if use_implicit_linear_stiffness:
        A = A + device.K
    band = _jc_band_matrix(A, device.selected_bandwidth)
    q_prev = np.zeros(n, dtype=float) if initial_q is None else np.array(initial_q, copy=True)
    q_cur = np.array(q_prev, copy=True)
    source = np.zeros(n, dtype=float)
    out_count = n_steps // record_stride + 1
    times = np.empty(out_count, dtype=float)
    voltage = np.empty(out_count, dtype=float)
    branch_r = np.empty(out_count, dtype=float)
    t0 = time.perf_counter()
    rec = 0
    times[rec] = 0.0; voltage[rec] = 0.0; branch_r[rec] = 0.0; rec += 1
    pump_node = device.pump_node
    output_node = device.output_node
    for step in range(1, n_steps + 1):
        t = step * dt
        phase = device.Bphi.T @ q_cur / PHI0_REDUCED
        current = device.Ic * np.sin(phase + phi_dc_rad)
        source.fill(0.0)
        source[pump_node] = pump_current_a * math.cos(omega * t)
        if device.signal_node == pump_node:
            source[pump_node] += signal_current_a * math.cos(2.0 * math.pi * signal_hz * t)
        else:
            source[device.signal_node] += signal_current_a * math.cos(2.0 * math.pi * signal_hz * t)
        rhs = source - device.Bphi @ current
        if not use_implicit_linear_stiffness:
            rhs -= device.K @ q_cur
        rhs += device.C @ (2.0 * q_cur - q_prev) / dt**2
        rhs += device.G @ q_prev / (2.0 * dt)
        q_next = solve_banded(
            (device.selected_bandwidth, device.selected_bandwidth),
            band, np.asarray(rhs).reshape(-1), check_finite=False,
        )
        if step % record_stride == 0:
            derivative = (q_next - q_prev) / (2.0 * dt)
            times[rec] = t
            voltage[rec] = derivative[output_node]
            branch_r[rec] = float(np.max(np.abs(
                np.sin(device.Bphi.T @ q_cur / PHI0_REDUCED + phi_dc_rad)
            )))
            rec += 1
        q_prev, q_cur = q_cur, q_next
    return times[:rec], voltage[:rec], branch_r[:rec], time.perf_counter() - t0, q_cur


@dataclass(frozen=True)
class TimeBudget:
    dt_norm: float
    tmax_norm: float
    dt_physical_ps: float
    steps_per_pump_period: float
    retained_pump_periods: float
    measured_steps_per_second: float
    measured_seconds_per_point: float


def _load_arrays(circuit_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = _load_device_arrays(resolve_device_directory(circuit_dir))
    return np.asarray(arrays["Ic"], dtype=float), np.asarray(
        arrays.get("Lj", np.empty(0)), dtype=float
    )


def _summary_for_device(circuit_dir: Path) -> dict[str, Any]:
    for name in ("ipm_summary.json", "summary.json", "design_summary.json"):
        path = circuit_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"name": circuit_dir.name}


def _device_name(summary: dict[str, Any], circuit_dir: Path) -> str:
    return str(summary.get("case") or summary.get("name") or circuit_dir.name)


def _cached_rf_pump_frequency(circuit_dir: Path) -> float | None:
    cache = ROOT / "outputs" / "chaos" / "phaseC" / "rf_squid_2393_3wm" / "transmitting_band.json"
    if not cache.exists():
        return None
    payload = json.loads(cache.read_text(encoding="utf-8"))
    return float(payload["selected_pump_ghz"])


def measure_transmitting_band(
    device: JcDevice, *, start_ghz: float = 4.0, stop_ghz: float = 12.0,
    points: int = 33,
) -> dict[str, Any]:
    """Measure the RF-SQUID signal band and select its strongest point."""
    frequencies = np.linspace(start_ghz, stop_ghz, points)
    rows = [
        solve_linear_scattering(
            device_to_circuit(device), frequency_hz=float(frequency) * 1e9,
            source_port=1, out_port=2, source_current_a=1.0,
        )
        for frequency in frequencies
    ]
    s_db = np.asarray([row.s_db for row in rows], dtype=float)
    index = int(np.argmax(s_db))
    selected = float(frequencies[index])
    payload = {
        "frequency_ghz": frequencies.tolist(), "s21_db": s_db.tolist(),
        "selected_pump_ghz": selected,
        "selection": "maximum measured linear S21 within 4-12 GHz scan",
    }
    cache = ROOT / "outputs" / "chaos" / "phaseC" / "rf_squid_2393_3wm" / "transmitting_band.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def device_to_circuit(device: JcDevice) -> Any:
    """Convert the ordered kernel representation to the linear solver type."""
    from twpa_solver.core.circuit import CircuitMatrices

    ports = {1: device.signal_node, 2: device.output_node}
    if device.name == "ipm_2c_fixed":
        ports.update({3: device.pump_output_node, 4: device.pump_node})
    return CircuitMatrices(
        C=device.C, G=device.G, K=device.K, Bphi=device.Bphi,
        Ic=device.Ic, Lj=device.Lj, phi0=device.phi0,
        port_to_index=ports,
    )


def derive_device_spec(circuit_dir: Path) -> DeviceSpec:
    """Extract the circuit facts needed by the phase-5 campaign."""
    resolved_dir = resolve_device_directory(circuit_dir)
    summary = _summary_for_device(resolved_dir)
    ic, lj = _load_arrays(resolved_dir)
    arrays = _load_device_arrays(resolved_dir)
    name = _device_name(summary, resolved_dir)
    cj, cg = _profile_from_source(name, ic.size, arrays)
    if name == "jc_jtwpa":
        period = 4
        pump_ghz, signal_ghz = 7.12, 6.62
    elif name == "jc_fqjtwpa":
        period = 8
        pump_ghz, signal_ghz = 7.90, 7.40
    elif name == "ipm_2c_fixed":
        period = 1
        pump_ghz, signal_ghz = 7.90, 7.40
    elif name == "rf_squid_2393_3wm":
        period = 4
        pump_ghz = 12.080
        signal_ghz = float("nan")
    else:
        raise ValueError(f"unsupported Phase C circuit: {name}")
    device = load_jc_device(resolved_dir)
    dc_flux_bias_present, dc_flux_bias_source = _dc_flux_bias_metadata(resolved_dir)
    bias = rf_squid_bias_metadata(resolved_dir)
    return DeviceSpec(
        name=name,
        circuit_dir=str(resolved_dir),
        node_count=int(device.n_nodes),
        branch_count=int(ic.size),
        ic_min_a=float(np.min(ic)),
        ic_max_a=float(np.max(ic)),
        ic_median_a=float(np.median(ic)),
        cj_nominal_f=float(np.median(cj)),
        cg_nominal_f=float(np.median(cg)),
        resonator_period=period,
            has_parallel_geometric_inductor=device.has_parallel_geometric_inductor,
        profile_is_nonuniform=bool(np.ptp(ic) > 0.0 or np.ptp(lj) > 0.0 or np.ptp(cg) > 0.0),
        pump_ghz=pump_ghz,
        signal_ghz=signal_ghz,
        port_network=("50 ohm source/load from G.npz; pump port 4, signal 1 -> 2"
                      if name == "ipm_2c_fixed" else
                      "50 ohm source/load from G.npz; ports 1 -> 2"),
        source_path=str(circuit_dir),
        natural_bandwidth=device.natural_bandwidth,
        rcm_bandwidth=device.rcm_bandwidth,
        selected_bandwidth=device.selected_bandwidth,
        selected_ordering=device.selected_ordering,
        pump_port=4 if name == "ipm_2c_fixed" else 1,
        pump_output_port=3 if name == "ipm_2c_fixed" else 2,
        signal_source_port=1,
        signal_output_port=2,
            cj_min_f=float(np.min(cj)), cj_max_f=float(np.max(cj)),
            cg_min_f=float(np.min(cg)), cg_max_f=float(np.max(cg)),
            dc_flux_bias_present=dc_flux_bias_present,
            dc_flux_bias_source=dc_flux_bias_source,
            dc_bias_current_a=float(bias["dc_bias_current_a"]),
            external_flux_fraction=float(bias["external_flux_fraction"]),
            beta_l=float(bias["beta_l"]),
            phi_ext_rad=float(bias["phi_ext_rad"]),
            phi_dc_rad=float(bias["phi_dc_rad"]),
            dc_bias_convention=str(bias["dc_bias_convention"]),
        )


def derive_time_budget(
    spec: DeviceSpec,
    *,
    dt_norm: float,
    tmax_norm: float,
) -> TimeBudget:
    """Derive sampling guarantees and a transparent cost estimate."""
    dt_s = dt_norm / spec.omega_plasma
    pump_period_s = 1.0 / (spec.pump_ghz * 1e9)
    steps_per_period = pump_period_s / dt_s
    retained = tmax_norm * (1.0 - 0.5) / (spec.omega_plasma * pump_period_s)
    if retained < 300.0:
        tmax_norm = 600.0 * spec.omega_plasma * pump_period_s
        retained = tmax_norm * (1.0 - 0.5) / (spec.omega_plasma * pump_period_s)
    return TimeBudget(
        dt_norm=dt_norm,
        tmax_norm=tmax_norm,
        dt_physical_ps=dt_s * 1e12,
        steps_per_pump_period=steps_per_period,
        retained_pump_periods=retained,
        measured_steps_per_second=float("nan"),
        measured_seconds_per_point=float("nan"),
    )


def _install_two_tone_source(system: Any, signal_ghz: float, signal_current_a: float) -> None:
    ratio = signal_ghz / (system.omega / (2.0 * math.pi * 1e9))
    system.signal_frequency_ratio = ratio
    system.signal_current_a = signal_current_a
    original_source = system.source

    def source(self: Any, theta: float, start: float, target: float, ramp: float) -> np.ndarray:
        value = original_source(theta, start, target, ramp)
        value[self.pump_node] += self.signal_current_a * math.cos(
            self.signal_frequency_ratio * theta
        )
        return value

    system.source = MethodType(source, system)


def _tone_amplitude(t: np.ndarray, voltage: np.ndarray, frequency_hz: float) -> float:
    w = 2.0 * math.pi * frequency_hz
    matrix = np.column_stack((np.ones_like(t), np.cos(w * t), np.sin(w * t)))
    coeff, *_ = np.linalg.lstsq(matrix, voltage, rcond=None)
    return float(math.hypot(coeff[1], coeff[2]))


def _spectrum(t: np.ndarray, voltage: np.ndarray, resistance: float) -> tuple[np.ndarray, np.ndarray]:
    centered = voltage - np.mean(voltage)
    window = np.hanning(centered.size)
    spectrum = np.fft.rfft(centered * window)
    amp = 2.0 * np.abs(spectrum) / (centered.size * np.mean(window))
    power_dbm = 10.0 * np.log10(np.maximum((amp / math.sqrt(2.0)) ** 2 / resistance, 1e-300) / 1e-3)
    freq = np.fft.rfftfreq(centered.size, t[1] - t[0])
    return freq, power_dbm


def _wideband_gain(
    t: np.ndarray,
    voltage: np.ndarray,
    signal_hz: float,
    pump_hz: float,
    signal_current_a: float,
) -> float:
    freq, dbm = _spectrum(t, voltage, 50.0)
    mask = np.abs(freq - signal_hz) <= WIDEBAND_HALF_BANDWIDTH_HZ
    for harmonic in range(1, 6):
        mask &= np.abs(freq - harmonic * pump_hz) > PUMP_NOTCH_HALF_WIDTH_HZ
    output_w = float(np.sum(1e-3 * 10.0 ** (dbm[mask] / 10.0)))
    input_w = signal_current_a**2 * 50.0 / 2.0
    return 10.0 * math.log10(max(output_w, 1e-300) / input_w)


def _safe_wideband_gain(
    t: np.ndarray, voltage: np.ndarray, signal_hz: float, pump_hz: float,
    signal_current_a: float,
) -> float | None:
    """``_wideband_gain`` that reports None on a diverged trace, never raises."""
    try:
        return _wideband_gain(t, voltage, signal_hz, pump_hz, signal_current_a)
    except (OverflowError, FloatingPointError, ValueError):
        return None


def _run_point(
    spec: DeviceSpec,
    pump_current_a: float,
    *,
    dt_norm: float,
    tmax_norm: float,
    signal_current_a: float,
    pump_off_output: float | None,
    method: str = "guarcello_banded",
    initial_state: np.ndarray | None = None,
    start_current_a: float = 0.0,
    phi_dc_rad: float | None = None,
) -> tuple[dict[str, Any], float, float, np.ndarray]:
    if method != "guarcello_banded":
        raise ValueError("phase 5 uses Guarcello's known-time-level integrator only")
    device = load_jc_device(Path(spec.circuit_dir))
    dt_s = dt_norm / spec.omega_plasma
    n_steps = int(round(tmax_norm / dt_norm))
    pump_hz = resolve_pump_frequency(spec)
    signal_hz = (
        spec.signal_ghz * 1e9
        if math.isfinite(spec.signal_ghz) and spec.signal_ghz > 0.0
        else pump_hz
    )
    branch_phase_offset = spec.phi_dc_rad if phi_dc_rad is None else float(phi_dc_rad)
    theta, voltage, branch_r, runtime, final_q = _integrate_jc_compiled(
        device,
        pump_current_a=pump_current_a, pump_hz=pump_hz,
        signal_current_a=signal_current_a,
        signal_hz=signal_hz,
        dt_s=dt_s,
        n_steps=n_steps,
        record_stride=20,
        initial_q=initial_state,
        phi_dc_rad=branch_phase_offset,
    )
    late = np.arange(theta.size) >= max(0, theta.size - max(10, theta.size // 2))
    trace_t = theta.copy()
    trace_v = voltage.copy()
    time_s = trace_t[late]
    voltage_ss = trace_v[late]
    signal_installed = bool(signal_current_a > 0.0)
    # A solution that runs away past the transition drives these expressions to
    # overflow.  That is a property of the state, not a driver fault, and it
    # must not cost the caller its trace: four points of the 2026-08-15 signal
    # campaign lost 805-868 s of completed integration to an OverflowError
    # raised after the stepping had finished.  Degrade the gain to None instead.
    try:
        amplitude = (
            _tone_amplitude(time_s, voltage_ss, signal_hz) if signal_installed else None
        )
        gain_absolute = (
            None if amplitude is None else
            20.0 * math.log10(max(amplitude, 1e-300) / (signal_current_a * 50.0))
        )
        gain_vs_off = (
            None if amplitude is None or pump_off_output is None else
            20.0 * math.log10(max(amplitude / pump_off_output, 1e-300))
        )
        gain_status = "OK" if signal_installed else "NO_SIGNAL"
    except (OverflowError, FloatingPointError, ValueError) as error:
        amplitude, gain_absolute, gain_vs_off = None, None, None
        gain_status = f"DIVERGED {error!r}"
    total_periods = theta[-1] * pump_hz
    row = {
        "method_attribution": "Guarcello known-time-level banded FDTD algorithm",
        "device_source_path": spec.source_path,
        "selected_ordering": spec.selected_ordering,
        "natural_bandwidth": spec.natural_bandwidth,
        "rcm_bandwidth": spec.rcm_bandwidth,
        "selected_bandwidth": spec.selected_bandwidth,
        "pump_port": spec.pump_port,
        "signal_source_port": spec.signal_source_port,
        "signal_output_port": spec.signal_output_port,
        "dc_bias_current_a": spec.dc_bias_current_a,
        "external_flux_fraction": spec.external_flux_fraction,
        "beta_l": spec.beta_l,
        "phi_ext_rad": spec.phi_ext_rad,
        "phi_dc_rad": branch_phase_offset,
        "bias_applied": bool(abs(branch_phase_offset) > 0.0),
        "dc_bias_convention": spec.dc_bias_convention,
        "bias_phase_is_uniform": True,
        "bias_phase_model": "uniform_branch_phase_offset",
        "pump_current_peak_a_requested": pump_current_a,
        "pump_current_peak_a_achieved": pump_current_a,
        **power_labels(pump_current_a, pump_hz),
        "gain_absolute_db": gain_absolute,
        "gain_vs_off_db": gain_vs_off,
        "gain_wideband_db": _safe_wideband_gain(
            time_s, voltage_ss, signal_hz, pump_hz, signal_current_a,
        ) if signal_installed else None,
        "gain_status": gain_status,
        # Recorded so downstream plots can mark the signal and idler lines
        # without re-deriving them from the device spec.
        "signal_hz": float(signal_hz),
        "idler_hz": float(abs(pump_hz - signal_hz)),
        "signal_installed": signal_installed,
        "r_j": float(np.max(branch_r[late])),
        "pump_branch_current_peak_a_achieved": float(
            np.max(branch_r[late]) * spec.ic_max_a
        ),
        "min_cos_phi": float(np.min(np.cos(
            device.Bphi.T @ final_q / PHI0_REDUCED + branch_phase_offset
        ))),
        "argmax_cell_index": int(np.argmax(np.abs(np.sin(
            device.Bphi.T @ final_q / PHI0_REDUCED + branch_phase_offset
        )))),
        "integrator_success": True,
        "integrator_message": "known-time-level constant banded solve",
        "strobe_d1_tail": float("nan"),
        "runtime_s": runtime,
        "integrator": "guarcello_banded",
        "total_pump_periods": total_periods,
        "steady_window_pump_periods": min(100.0, total_periods),
        "effective_transient_fraction": (
            max(0.0, 1.0 - min(100.0, total_periods) / total_periods)
            if total_periods > 0.0 else 0.0
        ),
        "record_stride": 20,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "steady_state_start_index": int(np.flatnonzero(late)[0]) if np.any(late) else 0,
    }
    return row, amplitude, runtime, final_q, trace_t, trace_v


def resolve_pump_frequency(spec: DeviceSpec) -> float:
    """Return a measured or fixed pump frequency in Hz for a device."""
    if math.isfinite(spec.pump_ghz) and spec.pump_ghz > 0.0:
        return spec.pump_ghz * 1e9
    device = load_jc_device(Path(spec.circuit_dir))
    payload = measure_transmitting_band(device)
    return float(payload["selected_pump_ghz"]) * 1e9


def power_labels(
    current_a: float, pump_hz: float, *, z0_ohm: float = 50.0,
    convention: str = "legacy_traveling_wave",
) -> dict[str, Any]:
    """Label one applied current with both on-chip and instrument powers."""
    on_chip_w = port_available_power_w(current_a, z0_ohm, convention=convention)
    on_chip_dbm = 10.0 * math.log10(max(on_chip_w, 1e-300) / 1e-3)
    loss = pump_line_loss_model()
    attenuation_db = float(loss.attenuation_db(pump_hz / 1e9))
    return {
        "pump_power_onchip_dbm": on_chip_dbm,
        "pump_power_instrument_dbm": on_chip_dbm + attenuation_db,
        "power_convention": convention,
        "loss_model": "pump_line_loss_model_A10",
        "loss_attenuation_db": attenuation_db,
    }


def finite_linear_inductance_device(device: JcDevice) -> JcDevice:
    """Replace Josephson nonlinearity by its finite small-signal stiffness.

    Setting ``Ic`` to zero alone removes the only inductive path in the JC
    fixtures.  This variant keeps ``Bphi diag(Ic / phi0) Bphi.T`` in ``K``
    while disabling the nonlinear current, so the time-domain kernel has a
    non-degenerate linear reference.
    """
    josephson_stiffness = (
        device.Bphi @ sp.diags(device.Ic / device.phi0) @ device.Bphi.T
    ).tocsr()
    return replace(
        device,
        K=(device.K + josephson_stiffness).tocsr(),
        Ic=np.zeros_like(device.Ic),
        phi_dc_rad=0.0,
        implicit_linear_stiffness=True,
    )


def _discrete_linear_steady_state(
    device: JcDevice, *, frequency_hz: float, source_current_a: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact previous and current states of the discrete recurrence."""
    omega = 2.0 * math.pi * frequency_hz
    z = np.exp(1j * omega * dt_s)
    c_dt2 = device.C / dt_s**2
    g_2dt = device.G / (2.0 * dt_s)
    discrete = (
        z * (c_dt2 + g_2dt + device.K)
        - 2.0 * c_dt2
        + (c_dt2 - g_2dt) / z
    ).tocsc()
    source = np.zeros(device.n_nodes, dtype=np.complex128)
    source[device.pump_node] = source_current_a * z
    phasor = spla.spsolve(discrete, source)
    return np.real(phasor / z), np.real(phasor)


def _measure_linear_limit(
    device: JcDevice, spec: DeviceSpec, pump_hz: float, dt_norm: float,
    implicit_linear_stiffness: bool | None = None,
    retain_linear_inductance: bool = False,
) -> dict[str, Any]:
    """Compare a linearized kernel run with the continuous linear S21 solve."""
    if retain_linear_inductance:
        device = finite_linear_inductance_device(device)
    linear_device = replace(
        device, Ic=np.zeros_like(device.Ic), output_node=device.pump_output_node,
        phi_dc_rad=0.0,
    )
    dt_s = dt_norm / spec.omega_plasma
    steps_per_period = 1.0 / pump_hz / dt_s
    n_steps = max(200, int(math.ceil(20.0 * steps_per_period)))
    source_current = 1.0e-8
    q_previous, q_current = _discrete_linear_steady_state(
        linear_device, frequency_hz=pump_hz,
        source_current_a=source_current, dt_s=dt_s,
    )
    times, voltage, _, runtime, _ = _integrate_jc_compiled(
        linear_device, pump_current_a=source_current, pump_hz=pump_hz,
        signal_current_a=0.0, signal_hz=pump_hz, dt_s=dt_s, n_steps=n_steps,
        record_stride=20, initial_q=q_current,
        initial_q_previous=q_previous,
        implicit_linear_stiffness=implicit_linear_stiffness,
        phi_dc_rad=0.0,
    )
    late = times >= times[-1] * 0.5
    amplitude = _tone_amplitude(times[late], voltage[late], pump_hz)
    kernel_s = 2.0 * amplitude / source_current / 50.0
    reference = solve_linear_scattering(
        device_to_circuit(linear_device), frequency_hz=pump_hz,
        source_port=spec.pump_port, out_port=spec.pump_output_port,
        source_current_a=1.0,
    )
    finite = bool(math.isfinite(kernel_s) and math.isfinite(reference.s_abs))
    relative_error = (
        abs(abs(kernel_s) - reference.s_abs) / max(reference.s_abs, 1e-300)
        if finite and reference.s_abs > 0.0 else float("nan")
    )
    return {
        "kernel_s_abs": float(abs(kernel_s)) if finite else None,
        "linear_solve_s_abs": float(reference.s_abs),
        "relative_error": float(relative_error) if math.isfinite(relative_error) else None,
        "kernel_finite": finite,
        "linear_reference": (
            "finite_junction_inductance" if retain_linear_inductance
            else "zero_Ic_degenerate"
        ),
        "initial_state": "exact_discrete_single_frequency_steady_state",
        "measurement_periods": 20.0,
        "runtime_s": float(runtime), "n_steps": n_steps,
        "pass_rtol_1e-9": bool(finite and relative_error <= 1e-9),
    }


def _measure_zero_drive(
    device: JcDevice, spec: DeviceSpec, pump_hz: float, dt_norm: float,
) -> dict[str, Any]:
    """Run the zero-drive equilibrium for 100 pump periods."""
    dt_s = dt_norm / spec.omega_plasma
    steps_per_period = 1.0 / pump_hz / dt_s
    n_steps = max(1, int(math.ceil(100.0 * steps_per_period)))
    _, _, _, runtime_first, settled_q = _integrate_jc_compiled(
        device, pump_current_a=0.0, pump_hz=pump_hz, signal_current_a=0.0,
        signal_hz=pump_hz, dt_s=dt_s, n_steps=n_steps,
        record_stride=n_steps, initial_q=np.zeros(device.n_nodes),
        phi_dc_rad=device.phi_dc_rad,
    )
    _, _, _, runtime_second, final_q = _integrate_jc_compiled(
        device, pump_current_a=0.0, pump_hz=pump_hz, signal_current_a=0.0,
        signal_hz=pump_hz, dt_s=dt_s, n_steps=n_steps,
        record_stride=n_steps, initial_q=settled_q,
        phi_dc_rad=device.phi_dc_rad,
    )
    maximum = float(np.max(np.abs(final_q - settled_q)))
    return {
        "n_steps": n_steps, "periods": 200.0,
        "runtime_s": float(runtime_first + runtime_second),
        "reference": "second 100-period hold from the first empirically settled state",
        "max_abs_final_q_minus_empirical_settled_state": maximum,
        "pass_at_1e-12": bool(maximum <= 1e-12),
        "dc_bias_current_a": device.dc_bias_current_a,
        "phi_dc_rad": device.phi_dc_rad,
    }


def _explicit_stability_bound(device: JcDevice, spec: DeviceSpec, dt_norm: float) -> dict[str, Any]:
    """Measure the finite explicit bound of the linearized generalized pencil."""
    stiffness = (
        device.K
        + device.Bphi @ sp.diags(device.Ic / device.phi0) @ device.Bphi.T
    ).tocsr()
    mass = (device.C + device.G).tocsr()
    zero_diagonal = np.flatnonzero(np.abs(mass.diagonal()) == 0.0)
    dynamic = np.flatnonzero(np.abs(mass.diagonal()) > 0.0)
    if zero_diagonal.size and mass[zero_diagonal].nnz:
        raise RuntimeError(
            f"{device.name}: zero-mass rows are coupled in C+G; "
            "finite-subspace reduction is not valid"
        )
    pencil_stiffness = stiffness[dynamic][:, dynamic] if zero_diagonal.size else stiffness
    pencil_mass = mass[dynamic][:, dynamic] if zero_diagonal.size else mass
    eigenvalue = float(eigsh(
        pencil_stiffness, M=pencil_mass, k=1, which="LA",
        return_eigenvectors=False, tol=1e-8, maxiter=20000,
    )[0])
    explicit_limit = 2.0 / math.sqrt(eigenvalue)
    dt_s = dt_norm / spec.omega_plasma
    return {
        "lambda_max_s_minus_2": eigenvalue,
        "explicit_limit_s": explicit_limit,
        "dt_s": dt_s,
        "dt_over_limit": dt_s / explicit_limit,
        "explicitly_stable": bool(dt_s < explicit_limit),
        "mass_matrix_singular": bool(zero_diagonal.size),
        "mass_zero_diagonal_nodes": int(zero_diagonal.size),
        "dynamic_nodes_used": int(dynamic.size if zero_diagonal.size else mass.shape[0]),
        "bound_method": (
            "eigsh on dynamic subspace after removing uncoupled algebraic rows"
            if zero_diagonal.size else "generalized eigsh on full pencil"
        ),
    }


def _stiffness_path_regression(
    spec: DeviceSpec, *, dt_norm: float, pump_current_a: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare the retained explicit Phase B path with the implicit default."""
    if spec.name not in {"jc_jtwpa", "jc_fqjtwpa"}:
        return {"status": "NOT_APPLICABLE"}
    device = load_jc_device(Path(spec.circuit_dir))
    settings = {
        "pump_current_a": pump_current_a,
        "pump_hz": spec.pump_ghz * 1.0e9,
        "signal_current_a": 0.0,
        "signal_hz": spec.signal_ghz * 1.0e9,
        "dt_s": dt_norm / spec.omega_plasma,
        "n_steps": 2000,
        "record_stride": 20,
        "initial_q": None,
    }
    explicit = _integrate_jc_compiled(
        device, **settings, implicit_linear_stiffness=False,
    )
    implicit = _integrate_jc_compiled(device, **settings, implicit_linear_stiffness=True)
    q_scale = max(float(np.max(np.abs(explicit[4]))), 1.0e-300)
    v_scale = max(float(np.max(np.abs(explicit[1]))), 1.0e-300)
    q_difference = float(np.max(np.abs(explicit[4] - implicit[4])) / q_scale)
    v_difference = float(np.max(np.abs(explicit[1] - implicit[1])) / v_scale)
    return {
        "status": "COMPLETE",
        "explicit_path": "Phase B known-time-level path; K on RHS",
        "implicit_path": "new default; K in constant factored matrix",
        "n_steps": settings["n_steps"],
        "max_relative_voltage_difference": v_difference,
        "max_relative_final_state_difference": q_difference,
        "explicit_runtime_s": float(explicit[3]),
        "implicit_runtime_s": float(implicit[3]),
        "explicit_integrator_success": True,
        "implicit_integrator_success": True,
    }


def _profile_kernel_components(device: JcDevice, dt_s: float, repetitions: int = 200) -> dict[str, Any]:
    """Time the compiled per-step primitives after their signatures are warm."""
    constant = device.C / dt_s**2 + device.G / (2.0 * dt_s) + device.K
    factor_started = time.perf_counter()
    factor = _factor_banded_lu(constant, device.selected_bandwidth)
    factor_time = time.perf_counter() - factor_started
    c_parts = _csr_parts(device.C)
    g_parts = _csr_parts(device.G)
    k_parts = _csr_parts(device.K)
    b_parts = _csr_parts(device.Bphi)
    n = device.n_nodes
    branch_count = device.Ic.size
    vector = np.ones(n, dtype=np.float64)
    branch_vector = np.ones(branch_count, dtype=np.float64)
    result = np.empty(n, dtype=np.float64)
    branch_result = np.empty(branch_count, dtype=np.float64)
    rhs = np.ones(n, dtype=np.float64)
    for fn, args in (
        (_csr_matvec_into, (*c_parts, vector, result)),
        (_csr_matvec_into, (*g_parts, vector, result)),
        (_csr_matvec_into, (*k_parts, vector, result)),
        (_csr_matvec_into, (*b_parts, branch_vector, result)),
        (_csr_transpose_matvec_into, (*b_parts, vector, branch_result)),
        (_solve_banded_lu, (factor, rhs)),
    ):
        fn(*args)

    def timed(fn: Any, args: tuple[Any, ...]) -> float:
        started = time.perf_counter()
        for _ in range(repetitions):
            fn(*args)
        return (time.perf_counter() - started) / repetitions

    components = {
        "Bphi_transpose_phase_s": timed(
            _csr_transpose_matvec_into, (*b_parts, vector, branch_result),
        ),
        "C_matvec_s": timed(_csr_matvec_into, (*c_parts, vector, result)),
        "G_matvec_s": timed(_csr_matvec_into, (*g_parts, vector, result)),
        "K_matvec_s": timed(_csr_matvec_into, (*k_parts, vector, result)),
        "Bphi_matvec_s": timed(_csr_matvec_into, (*b_parts, branch_vector, result)),
        "band_solve_s": timed(_solve_banded_lu, (factor, rhs)),
    }
    return {
        "repetitions": repetitions,
        "selected_bandwidth": device.selected_bandwidth,
        "node_count": device.n_nodes,
        "branch_count": int(branch_count),
        "factor_setup_s": factor_time,
        "components": components,
        "measured_component_sum_s": float(sum(components.values())),
    }


def _matrix_digest(matrix: sp.spmatrix) -> str:
    csr = matrix.tocsr()
    digest = hashlib.sha256()
    for values in (csr.indptr, csr.indices, csr.data):
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ordering_regression(name: str) -> dict[str, Any]:
    source = (ROOT / "outputs" / "jc_doc_python_designs" / name)
    device = load_jc_device(source)
    raw = {key: _load_sparse(source / f"{key}.npz") for key in ("C", "G", "K", "Bphi")}
    identity = np.array_equal(device.permutation, np.arange(device.n_nodes))
    unchanged = identity and all(_matrix_digest(raw[key]) == _matrix_digest(getattr(device, key)) for key in raw)
    return {
        "device": name, "selected_ordering": device.selected_ordering,
        "permutation_identity": bool(identity), "matrix_bytes_unchanged": bool(unchanged),
        "natural_bandwidth": device.natural_bandwidth,
        "rcm_bandwidth": device.rcm_bandwidth,
        "pass": bool(unchanged),
    }


def run_phase_c_preflight(
    output: Path, *, dt_norm: float = 0.01,
) -> dict[str, Any]:
    """Run C-G1 through C-G5 without launching a sweep."""
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "ipm_2c_fixed": ROOT / "designs" / "ipm_2c_fixed",
        "rf_squid_2393_3wm": ROOT / "designs" / "rf_squid_2393_3wm.yaml",
    }
    stability_sources = {
        "jc_jtwpa": ROOT / "outputs" / "jc_doc_python_designs" / "jc_jtwpa",
        "jc_fqjtwpa": ROOT / "outputs" / "jc_doc_python_designs" / "jc_fqjtwpa",
        **sources,
    }
    stability: dict[str, Any] = {}
    for name, source in stability_sources.items():
        stability_device = load_jc_device(Path(derive_device_spec(source).circuit_dir))
        stability[name] = _explicit_stability_bound(
            stability_device, derive_device_spec(source), dt_norm,
        )
    (output / "eigenvalue_table.json").write_text(
        json.dumps(_json_safe({"dt_norm": dt_norm, "results": stability}), indent=2),
        encoding="utf-8",
    )
    results: dict[str, Any] = {}
    for name, source in sources.items():
        spec = derive_device_spec(source)
        device = load_jc_device(Path(spec.circuit_dir))
        if name == "rf_squid_2393_3wm" and spec.pump_ghz <= 0.0:
            band = measure_transmitting_band(device)
            spec = derive_device_spec(source)
        else:
            band = None
        pump_hz = resolve_pump_frequency(spec)
        measured = measure_device_rate(
            spec, dt_norm=dt_norm, signal_current_a=0.0,
            output=output / name,
        )
        linear = _measure_linear_limit(device, spec, pump_hz, dt_norm)
        zero_drive = _measure_zero_drive(device, spec, pump_hz, dt_norm)
        result = {
            "device": name, "source_path": str(source),
            "resolved_circuit_dir": spec.circuit_dir,
            "device_spec": asdict(spec),
            "ordering": {
                "natural_bandwidth": device.natural_bandwidth,
                "rcm_bandwidth": device.rcm_bandwidth,
                "selected_bandwidth": device.selected_bandwidth,
                "selected_ordering": device.selected_ordering,
                "asserted": device.selected_bandwidth in {
                    device.natural_bandwidth, device.rcm_bandwidth
                },
            },
            "ports": {
                "pump_port": spec.pump_port,
                "pump_output_port": spec.pump_output_port,
                "signal_source_port": spec.signal_source_port,
                "signal_output_port": spec.signal_output_port,
                "selected_pump_frequency_ghz": pump_hz / 1e9,
            },
            "per_cell_profiles": {
                "Ic_min_a": float(np.min(device.Ic)), "Ic_max_a": float(np.max(device.Ic)),
                "Cj_min_f": float(np.min(device.Cj)), "Cj_max_f": float(np.max(device.Cj)),
                "Cg_min_f": float(np.min(device.Cg)), "Cg_max_f": float(np.max(device.Cg)),
                "Cj_array_length": int(device.Cj.size), "Cg_array_length": int(device.Cg.size),
            },
            "parallel_geometric_inductor": {
                "present": device.has_parallel_geometric_inductor,
                "branch_law": (
                    "K includes Lpar^-1 and Bphi carries Ic*sin(phi/phi0)"
                    if device.has_parallel_geometric_inductor else
                    "Bphi carries Ic*sin(phi/phi0); no Lpar branch detected"
                ),
            },
            "dc_flux_bias": {
                "present": spec.dc_flux_bias_present,
                "source": spec.dc_flux_bias_source,
            },
            "D1_explicit_stability": stability[name],
            "C-G3_rate": measured,
            "C-G4_linear_limit": linear,
            "C-G5_zero_drive": zero_drive,
            "C-G2_stiffness_path": _stiffness_path_regression(
                spec, dt_norm=dt_norm,
            ),
            "rf_transmitting_band": band,
        }
        (output / name / "result.json").write_text(
            json.dumps(_json_safe(result), indent=2), encoding="utf-8",
        )
        results[name] = result
    results["C-G2_ordering_regression"] = {
        name: _ordering_regression(name)
        for name in ("jc_jtwpa", "jc_fqjtwpa")
    }
    results["C-G2_stiffness_paths"] = {}
    for name in ("jc_jtwpa", "jc_fqjtwpa"):
        old_spec = derive_device_spec(stability_sources[name])
        results["C-G2_stiffness_paths"][name] = _stiffness_path_regression(
            old_spec, dt_norm=dt_norm,
        )
    payload = {
        "status": "PREFLIGHT_COMPLETE", "dt_norm": dt_norm,
        "D1_explicit_stability_table": stability,
        "devices": results,
    }
    (output / "preflight.json").write_text(
        json.dumps(_json_safe(payload), indent=2), encoding="utf-8",
    )
    return payload


def _read_hb_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_current_points(hb_rows: list[dict[str, str]], ic_scale: float) -> list[float]:
    values = [float(row["pump_current_peak_a"]) for row in hb_rows]
    last_r = float(hb_rows[-1]["pump_branch_current_max_over_ic"])
    target = values[-1] / last_r * ic_scale
    if target <= values[-1]:
        target = values[-1] * 1.5
    return values + np.linspace(values[-1], target, 7, dtype=float).tolist()[1:]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_phase5_device(name: str, output: Path) -> list[Path]:
    """Plot Guarcello FDTD points together with the HB reference column."""
    import matplotlib.pyplot as plt

    summary_path = output / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"phase-5 summary is missing: {summary_path}")
    rows = _read_hb_rows(summary_path)
    complete = [row for row in rows if row.get("status") == "COMPLETE"]
    if not complete:
        raise ValueError(f"no completed phase-5 rows in {summary_path}")
    if any(not row.get("pump_power_dbm", "").strip() for row in complete):
        raise ValueError("summary.csv is missing pump_power_dbm")

    fdtd_power = np.array([float(row["pump_power_dbm"]) for row in complete])
    fdtd_gain = np.array([float(row["gain_vs_off_db"]) for row in complete])
    fdtd_rj = np.array([float(row["r_j"]) for row in complete])
    fdtd_order = np.argsort(fdtd_power)
    fdtd_power, fdtd_gain, fdtd_rj = fdtd_power[fdtd_order], fdtd_gain[fdtd_order], fdtd_rj[fdtd_order]

    hb_name = name.removeprefix("jc_")
    hb_path = ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811" / hb_name / "hb_up_to_failure.csv"
    hb_rows = _read_hb_rows(hb_path)
    hb_valid = [row for row in hb_rows if row.get("status") == "PASS" and row.get("pump_status") in {"VALID_CONVERGED", "VALID_SOLVED"} and row.get("gain_vs_off_db", "").strip()]
    hb_power = np.array([float(row["pump_power_dbm"]) for row in hb_valid])
    hb_gain = np.array([float(row["gain_vs_off_db"]) for row in hb_valid])
    hb_rj = np.array([float(row["pump_branch_current_max_over_ic"]) for row in hb_valid])
    hb_order = np.argsort(hb_power)
    hb_power, hb_gain, hb_rj = hb_power[hb_order], hb_gain[hb_order], hb_rj[hb_order]
    hb_failure = next((float(row["pump_power_dbm"]) for row in hb_rows if row.get("pump_status") not in {"VALID_CONVERGED", "VALID_SOLVED"}), None)

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(hb_power, hb_gain, "k--", alpha=0.75, label="HB reference")
    axis.plot(fdtd_power, fdtd_gain, "o-", color="tab:blue", label="Guarcello FDTD")
    if hb_failure is not None:
        axis.axvline(hb_failure, color="tab:red", linestyle=":", label="HB failure")
    axis.set(xlabel="Pump power (dBm)", ylabel="Gain vs pump-off (dB)", title=f"Phase 5 {name}: gain")
    axis.grid(alpha=0.25); axis.legend()
    gain_path = output / "phase5_gain_vs_power_dbm.png"
    fig.savefig(gain_path, dpi=180); plt.close(fig); paths.append(gain_path)

    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(hb_power, hb_rj, "k--", alpha=0.75, label="HB reference")
    axis.plot(fdtd_power, fdtd_rj, "o-", color="tab:orange", label="Guarcello FDTD")
    axis.axhline(1.0, color="tab:purple", linestyle="--", label="$r_j=1$")
    if hb_failure is not None:
        axis.axvline(hb_failure, color="tab:red", linestyle=":", label="HB failure")
    axis.set(xlabel="Pump power (dBm)", ylabel=r"$r_j=I_{J,\max}/I_c$", title=f"Phase 5 {name}: junction drive")
    axis.grid(alpha=0.25); axis.legend()
    drive_path = output / "phase5_rj_vs_power_dbm.png"
    fig.savefig(drive_path, dpi=180); plt.close(fig); paths.append(drive_path)
    return paths
def _worker(args: argparse.Namespace) -> int:
    spec = derive_device_spec(Path(args.circuit_dir))
    initial = None if not args.state_in else np.load(args.state_in)["state"]
    row, amplitude, runtime, final_state, trace_t, trace_v = _run_point(
        spec,
        args.pump_current_a,
        dt_norm=args.dt_norm,
        tmax_norm=args.tmax_norm,
        signal_current_a=args.signal_current_a,
        pump_off_output=args.pump_off_output,
        method=args.method,
        initial_state=initial,
        start_current_a=args.start_current_a,
    )
    np.savez_compressed(args.state_out, state=final_state)
    np.savez_compressed(args.trace_out, t=trace_t, v_out=trace_v)
    Path(args.result_json).write_text(
        json.dumps({"row": row, "amplitude": amplitude, "runtime_s": runtime}, indent=2),
        encoding="utf-8",
    )
    return 0


def _run_subprocess_point(
    spec: DeviceSpec,
    *,
    pump_current_a: float,
    start_current_a: float,
    initial_state: np.ndarray | None,
    method: str,
    dt_norm: float,
    tmax_norm: float,
    signal_current_a: float,
    pump_off_output: float | None,
    output: Path,
    budget_s: float,
) -> tuple[dict[str, Any], float | None, np.ndarray | None]:
    output.mkdir(parents=True, exist_ok=True)
    state_in = output / "state_in.npz"
    state_out = output / "state_out.npz"
    result_json = output / "result.json"
    trace_out = output / "trace.npz"
    if initial_state is not None:
        np.savez_compressed(state_in, state=initial_state)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--circuit-dir", spec.circuit_dir,
        "--pump-current-a", str(pump_current_a),
        "--start-current-a", str(start_current_a),
        "--dt-norm", str(dt_norm), "--tmax-norm", str(tmax_norm),
        "--signal-current-a", str(signal_current_a), "--method", method,
        "--state-out", str(state_out), "--result-json", str(result_json),
        "--trace-out", str(trace_out),
    ]
    if pump_off_output is not None:
        command.extend(["--pump-off-output", str(pump_off_output)])
    if initial_state is not None:
        command.extend(["--state-in", str(state_in)])
    started = time.perf_counter()
    try:
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=budget_s, check=False,
            )
    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT", "integrator": method,
            "pump_current_peak_a_requested": pump_current_a,
            "start_current_a": start_current_a,
            "wall_time_budget_s": budget_s,
            "runtime_s": time.perf_counter() - started,
        }, None, None
    if completed.returncode != 0 or not result_json.exists():
        return {
            "status": "FAILED", "integrator": method,
            "pump_current_peak_a_requested": pump_current_a,
            "start_current_a": start_current_a,
            "wall_time_budget_s": budget_s,
            "runtime_s": time.perf_counter() - started,
            "return_code": completed.returncode,
        }, None, None
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    row = payload["row"]
    row["status"] = "COMPLETE"
    row["wall_time_budget_s"] = budget_s
    row["trace_path"] = str(trace_out)
    state = np.load(state_out)["state"] if state_out.exists() else None
    amplitude = payload.get("amplitude")
    return row, (None if amplitude is None else float(amplitude)), state


def measure_device_rate(
    spec: DeviceSpec,
    *,
    dt_norm: float,
    signal_current_a: float,
    output: Path,
) -> dict[str, Any]:
    """Measure the bounded solver on exactly 200 normalized steps."""
    output.mkdir(parents=True, exist_ok=True)
    # Compile the device-specific Numba signature before timing the requested
    # 200-step measurement. The compilation cost is not an integration rate.
    _run_point(
        spec, 0.0, dt_norm=dt_norm, tmax_norm=dt_norm,
        signal_current_a=0.0, pump_off_output=None,
        method="guarcello_banded",
    )
    started = time.perf_counter()
    row, _, runtime, _, _, _ = _run_point(
        spec, 0.0, dt_norm=dt_norm, tmax_norm=200.0 * dt_norm,
        signal_current_a=signal_current_a, pump_off_output=None,
        method="guarcello_banded",
    )
    steps = 200
    rate = steps / max(runtime, 1e-12)
    measurement = {
        "method_attribution": "Guarcello known-time-level banded FDTD algorithm",
        "benchmark_steps": steps,
        "benchmark_runtime_s": runtime,
        "measured_steps_per_second": rate,
        "benchmark_row": row,
        "measured_wall_time_s": time.perf_counter() - started,
        "component_breakdown": _profile_kernel_components(
            load_jc_device(Path(spec.circuit_dir)),
            dt_norm / spec.omega_plasma,
        ),
    }
    (output / "measured_rate_200_steps.json").write_text(
        json.dumps(measurement, indent=2), encoding="utf-8",
    )
    return measurement


def run_device(name: str, *, dt_norm: float, tmax_norm: float, output: Path,
               per_point_budget_s: float | None = None,
               pump_power_min_dbm: float = -30.0,
               pump_power_max_dbm: float = -28.0,
               pump_power_points: int = 10,
               pump_power_values: tuple[float, ...] | None = None) -> dict[str, Any]:
    if name == "ipm_2c_fixed":
        circuit_dir = ROOT / "designs" / name
    elif name == "rf_squid_2393_3wm":
        circuit_dir = ROOT / "designs" / f"{name}.yaml"
    else:
        circuit_dir = ROOT / "outputs" / "jc_doc_python_designs" / name
    hb_name = name.removeprefix("jc_")
    hb_dir = ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811" / hb_name
    spec = derive_device_spec(circuit_dir)
    budget = derive_time_budget(spec, dt_norm=dt_norm, tmax_norm=tmax_norm)
    output.mkdir(parents=True, exist_ok=True)
    hb_rows = _read_hb_rows(hb_dir / "hb_up_to_failure.csv")
    valid_hb = [r for r in hb_rows if r.get("status") == "PASS" and r.get("pump_status") in {"VALID_CONVERGED", "VALID_SOLVED"}]
    if len(valid_hb) < 2:
        raise RuntimeError("at least two valid HB rows are required for pump-power mapping")
    hb_power = np.array([float(r["pump_power_dbm"]) for r in valid_hb])
    hb_current = np.array([float(r["pump_current_peak_a"]) for r in valid_hb])
    order = np.argsort(hb_power); hb_power, hb_current = hb_power[order], hb_current[order]
    requested_powers = (np.asarray(pump_power_values, dtype=float)
                        if pump_power_values is not None
                        else np.linspace(pump_power_min_dbm, pump_power_max_dbm, pump_power_points))
    edge = np.polyfit(hb_power[-2:], np.log(hb_current[-2:]), 1)
    currents = np.interp(requested_powers, hb_power, hb_current)
    high = requested_powers > hb_power[-1]
    currents[high] = np.exp(np.polyval(edge, requested_powers[high]))
    signal_current = math.sqrt(2.0 * 1e-3 * 10.0 ** (-100.0 / 10.0) / 50.0) / 50.0
    measurement = measure_device_rate(spec, dt_norm=dt_norm, signal_current_a=signal_current, output=output)
    budget_payload = asdict(budget)
    budget_payload["measured_steps_per_second"] = measurement["measured_steps_per_second"]
    budget_payload["measured_seconds_per_point"] = budget.tmax_norm / dt_norm / measurement["measured_steps_per_second"]
    (output / "cost_estimate.json").write_text(json.dumps({"method_attribution": "Guarcello known-time-level banded FDTD algorithm", "device": asdict(spec), "time_budget": budget_payload}, indent=2), encoding="utf-8")
    point_budget = per_point_budget_s or min(900.0, budget_payload["measured_seconds_per_point"] * 1.5)

    pump_off_dir = output / "pump_off"
    off_summary = pump_off_dir / "summary.json"
    off_state_path = pump_off_dir / "state_out.npz"
    if off_summary.exists() and off_state_path.exists():
        off_row = json.loads(off_summary.read_text(encoding="utf-8"))
        off_payload = pump_off_dir / "result.json"
        if off_payload.exists():
            off_amplitude = float(json.loads(off_payload.read_text(encoding="utf-8"))["amplitude"])
        else:
            off_amplitude = float(off_row.get("amplitude", 0.0))
        off_state = np.load(off_state_path)["state"]
    else:
        off_row, off_amplitude, off_state = _run_subprocess_point(
            spec, pump_current_a=0.0, start_current_a=0.0, initial_state=None,
            method="guarcello_banded", dt_norm=dt_norm, tmax_norm=tmax_norm,
            signal_current_a=signal_current, pump_off_output=None,
            output=pump_off_dir, budget_s=point_budget)
        (pump_off_dir / "summary.json").write_text(json.dumps({**off_row, "amplitude": off_amplitude}, indent=2), encoding="utf-8")
    if off_amplitude is None or off_state is None:
        raise RuntimeError("pump-off reference did not complete")

    existing: list[dict[str, Any]] = []
    summary_path = output / "summary.csv"
    if summary_path.exists():
        existing = _read_hb_rows(summary_path)
    for row in existing:
        if not row.get("pump_power_dbm", "").strip():
            current = float(row["pump_current_peak_a_requested"])
            row["pump_power_dbm"] = str(float(hb_power[np.argmin(abs(hb_current-current))]))
            row["pump_power_mapping"] = "nearest_valid_hb_row"
    unique_existing: dict[float, dict[str, Any]] = {}
    for row in existing:
        key = round(float(row["pump_current_peak_a_requested"]), 15)
        unique_existing[key] = row
    existing = list(unique_existing.values())
    rows: list[dict[str, Any]] = []
    used_existing: set[int] = set()
    previous_state = off_state; previous_current = 0.0
    for index, (power_dbm, current) in enumerate(zip(requested_powers, currents)):
        match_index = next((j for j, r in enumerate(existing)
                            if j not in used_existing
                            and abs(float(r.get("pump_power_dbm", "nan")) - power_dbm) <= 0.15
                            and r.get("status") == "COMPLETE"), None)
        match = existing[match_index] if match_index is not None else None
        if match_index is not None:
            used_existing.add(match_index)
        if match is not None:
            row = dict(match)
            state_path = output / f"point_{int(row.get('point_index', index)):03d}" / "state_out.npz"
            if not state_path.exists():
                state_path = output / f"point_new_{index:03d}" / "state_out.npz"
            if state_path.exists():
                previous_state = np.load(state_path)["state"]
                previous_current = float(row["pump_current_peak_a_requested"])
            rows.append(row)
            continue
        point_dir = output / f"point_new_{index:03d}"
        row, _, state = _run_subprocess_point(
            spec, pump_current_a=float(current), start_current_a=previous_current,
            initial_state=previous_state, method="guarcello_banded", dt_norm=dt_norm,
            tmax_norm=tmax_norm, signal_current_a=signal_current,
            pump_off_output=off_amplitude, output=point_dir, budget_s=point_budget)
        row["point_index"] = index
        row["pump_power_dbm"] = str(float(power_dbm))
        row["pump_power_mapping"] = "hb_interpolation" if power_dbm <= hb_power[-1] else "hb_edge_extrapolation"
        rows.append(row)
        if state is not None:
            previous_state, previous_current = state, float(current)
        _write_csv(summary_path, rows)
    for j, row in enumerate(existing):
        if j not in used_existing:
            rows.append(dict(row))
    _write_csv(summary_path, rows)
    result = {"device": asdict(spec), "time_budget": asdict(budget), "pump_off": off_row,
              "points": len(rows), "method_attribution": "Guarcello known-time-level banded FDTD algorithm",
              "chosen_integrator": "guarcello_banded", "per_point_wall_time_budget_s": point_budget,
              "requested_pump_powers_dbm": requested_powers.tolist(), "status": "COMPLETE"}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--circuit-dir", type=Path)
    parser.add_argument("--pump-current-a", type=float)
    parser.add_argument("--start-current-a", type=float, default=0.0)
    parser.add_argument("--signal-current-a", type=float)
    parser.add_argument("--pump-off-output", type=float, default=None)
    parser.add_argument("--method", choices=["guarcello_banded"], default="guarcello_banded")
    parser.add_argument("--state-in", type=Path, default=None)
    parser.add_argument("--state-out", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--trace-out", type=Path)
    parser.add_argument("--device", choices=["jc_jtwpa", "jc_fqjtwpa", "ipm_2c_fixed", "rf_squid_2393_3wm", "both"], default="both")
    parser.add_argument("--dt-norm", type=float, default=0.01)
    parser.add_argument("--tmax-norm", type=float, default=20_000.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chaos/phase5")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--phase-c-preflight", action="store_true")
    parser.add_argument("--per-point-budget-s", type=float, default=None)
    parser.add_argument("--pump-power-min-dbm", type=float, default=-30.0)
    parser.add_argument("--pump-power-max-dbm", type=float, default=-28.0)
    parser.add_argument("--pump-power-points", type=int, default=10)
    parser.add_argument("--pump-power-values", type=str, default=None,
                        help="comma-separated explicit pump powers in dBm")
    args = parser.parse_args()
    if args.phase_c_preflight:
        run_phase_c_preflight(args.output, dt_norm=args.dt_norm)
        return 0
    if args.worker:
        return _worker(args)
    names = ["jc_jtwpa", "jc_fqjtwpa"] if args.device == "both" else [args.device]
    if args.plot_only:
        for name in names:
            paths = plot_phase5_device(name, args.output / name)
            print(json.dumps({"device": name, "plots": [str(path) for path in paths]}))
        return 0
    if args.benchmark_only:
        args.output.mkdir(parents=True, exist_ok=True)
        signal_current = math.sqrt(2.0 * 1e-3 * 10.0 ** (-100.0 / 10.0) / 50.0) / 50.0
        for name in names:
            spec_source = (
                phase_c_source_path(name)
                if name in {"ipm_2c_fixed", "rf_squid_2393_3wm"}
                else ROOT / "outputs/jc_doc_python_designs" / name
            )
            spec = derive_device_spec(spec_source)
            measure_device_rate(
                spec, dt_norm=args.dt_norm, signal_current_a=signal_current,
                output=args.output / name,
            )
        return 0
    if args.estimate_only:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in names:
            spec = derive_device_spec(ROOT / "outputs/jc_doc_python_designs" / name)
            budget = derive_time_budget(spec, dt_norm=args.dt_norm, tmax_norm=args.tmax_norm)
            (args.output / f"{name}_cost_estimate.json").write_text(
                json.dumps({
                    "method_attribution": "Guarcello known-time-level banded FDTD algorithm",
                    "device": asdict(spec),
                    "time_budget": asdict(budget),
                }, indent=2),
                encoding="utf-8",
            )
            print(json.dumps({"device": name, **asdict(budget)}, sort_keys=True))
        return 0
    for name in names:
        run_device(
            name, dt_norm=args.dt_norm, tmax_norm=args.tmax_norm,
            output=args.output / name, per_point_budget_s=args.per_point_budget_s,
            pump_power_min_dbm=args.pump_power_min_dbm, pump_power_max_dbm=args.pump_power_max_dbm,
            pump_power_points=args.pump_power_points,
            pump_power_values=(tuple(float(value) for value in args.pump_power_values.split(","))
                               if args.pump_power_values else None),
        )
        paths = plot_phase5_device(name, args.output / name)
        print(json.dumps({"device": name, "plots": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
