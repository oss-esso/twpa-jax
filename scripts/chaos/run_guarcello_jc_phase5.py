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
import json
import math
import sys
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.h1_transient_branch_transfer import (
    build_system,
    implicit_trapezoid_ramp_bounded,
    make_observables,
    stroboscopic_diagnostics,
)
from twpa_solver.core import load_circuit
from twpa_solver.core.constants import PHI0_REDUCED
from scipy.integrate import solve_ivp


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
    natural_bandwidth: int
    ic_uniform: bool
    pump_node: int
    output_node: int


def _load_sparse(path: Path) -> sp.csr_matrix:
    values = np.load(path, allow_pickle=True)
    return sp.csr_matrix(
        (values["data"], values["indices"], values["indptr"]),
        shape=tuple(values["shape"]),
    )


def load_jc_device(circuit_dir: Path) -> JcDevice:
    """Load one JC topology for the known-time-level Guarcello scheme."""
    C = _load_sparse(circuit_dir / "C.npz")
    G = _load_sparse(circuit_dir / "G.npz")
    K = _load_sparse(circuit_dir / "K.npz")
    Bphi = _load_sparse(circuit_dir / "Bphi.npz")
    matrix = (K + C + G + Bphi @ Bphi.T).tocoo()
    bandwidth = int(np.max(np.abs(matrix.row - matrix.col)))
    if bandwidth > 2:
        raise RuntimeError(
            f"{circuit_dir.name}: natural bandwidth {bandwidth} exceeds 2"
        )
    arrays = np.load(circuit_dir / "ipm_arrays.npz", allow_pickle=True)
    Ic = np.asarray(arrays["Ic"], dtype=float)
    Lj = np.asarray(arrays["Lj"], dtype=float)
    if C.shape[0] != C.shape[1] or Bphi.shape[0] != C.shape[0]:
        raise ValueError("JC matrices have incompatible node dimensions")
    return JcDevice(
        name=circuit_dir.name,
        n_nodes=C.shape[0], C=C, G=G, K=K, Bphi=Bphi,
        Ic=Ic, Lj=Lj, natural_bandwidth=bandwidth,
        ic_uniform=bool(np.ptp(Ic) == 0.0),
        pump_node=int(np.asarray(arrays["port_indices"])[0]),
        output_node=int(np.asarray(arrays["port_indices"])[1]),
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


@njit(cache=True, fastmath=True, nogil=True)
def _solve_banded_lu(lu, rhs):
    bandwidth = (lu.shape[0] - 1) // 2
    diagonal = bandwidth
    n = rhs.size
    solution = rhs.copy()
    for i in range(n):
        value = solution[i]
        for k in range(max(0, i - bandwidth), i):
            value -= lu[diagonal + i - k, k] * solution[k]
        solution[i] = value
    for i in range(n - 1, -1, -1):
        value = solution[i]
        for k in range(i + 1, min(n, i + bandwidth + 1)):
            value -= lu[diagonal + i - k, k] * solution[k]
        solution[i] = value / lu[diagonal, i]
    return solution


@njit(cache=True, fastmath=True, nogil=True)
def integrate_jc_banded_numba(
    c_indptr, c_indices, c_data,
    g_indptr, g_indices, g_data,
    k_indptr, k_indices, k_data,
    b_indptr, b_indices, b_data, n_branches,
    ic, lower_factor, phi0, pump_current_a, pump_hz, signal_current_a,
    signal_hz, dt_s, n_steps, record_stride, pump_node, output_node,
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
            current[j] = ic[j] * math.sin(phase[j])
        source[:] = 0.0
        source[pump_node] = pump_current_a * math.cos(pump_omega * t)
        source[pump_node] += signal_current_a * math.cos(signal_omega * t)
        rhs[:] = source
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
                value = abs(math.sin(phase[j]))
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


def _integrate_jc_compiled(
    device: JcDevice, *, pump_current_a: float, pump_hz: float,
    signal_current_a: float, signal_hz: float, dt_s: float, n_steps: int,
    record_stride: int, initial_q: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    n = device.n_nodes
    constant = device.C / dt_s**2 + device.G / (2.0 * dt_s)
    lower = _factor_banded_lu(constant, device.natural_bandwidth)
    parts = [_csr_parts(matrix) for matrix in (device.C, device.G, device.K, device.Bphi)]
    started = time.perf_counter()
    result = integrate_jc_banded_numba(
        *parts[0], *parts[1], *parts[2], *parts[3], device.Ic.size,
        device.Ic, lower, PHI0_REDUCED, pump_current_a, pump_hz, signal_current_a,
        signal_hz, dt_s, n_steps, record_stride, device.pump_node,
        device.output_node, np.zeros(n) if initial_q is None else np.asarray(initial_q, dtype=np.float64),
    )
    return (*result[:3], time.perf_counter() - started, result[3])


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Integrate JC node fluxes with known-time-level Josephson currents."""
    n = device.n_nodes
    dt = float(dt_s)
    omega = 2.0 * math.pi * pump_hz
    A = device.C / dt**2 + device.G / (2.0 * dt)
    band = _jc_band_matrix(A, device.natural_bandwidth)
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
        current = device.Ic * np.sin(phase)
        source.fill(0.0)
        source[pump_node] = pump_current_a * math.cos(omega * t)
        source[pump_node] += signal_current_a * math.cos(2.0 * math.pi * signal_hz * t)
        rhs = source - device.K @ q_cur - device.Bphi @ current
        rhs += device.C @ (2.0 * q_cur - q_prev) / dt**2
        rhs += device.G @ q_prev / (2.0 * dt)
        q_next = solve_banded(
            (device.natural_bandwidth, device.natural_bandwidth),
            band, np.asarray(rhs).reshape(-1), check_finite=False,
        )
        if step % record_stride == 0:
            derivative = (q_next - q_prev) / (2.0 * dt)
            times[rec] = t
            voltage[rec] = derivative[output_node]
            branch_r[rec] = float(np.max(np.abs(np.sin(device.Bphi.T @ q_cur / PHI0_REDUCED))))
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
    arrays = np.load(circuit_dir / "ipm_arrays.npz", allow_pickle=True)
    return np.asarray(arrays["Ic"], dtype=float), np.asarray(arrays["Lj"], dtype=float)


def derive_device_spec(circuit_dir: Path) -> DeviceSpec:
    """Extract the circuit facts needed by the phase-5 campaign."""
    summary = json.loads((circuit_dir / "ipm_summary.json").read_text(encoding="utf-8"))
    ic, lj = _load_arrays(circuit_dir)
    name = str(summary["case"])
    if name == "jc_jtwpa":
        cj_nominal_f, cg_nominal_f, period = 55e-15, 45e-15, 4
        pump_ghz, signal_ghz = 7.12, 6.62
    elif name == "jc_fqjtwpa":
        cj_nominal_f, cg_nominal_f, period = 40e-15, 76.6e-15, 8
        pump_ghz, signal_ghz = 7.90, 7.40
    else:
        raise ValueError(f"unsupported phase-5 circuit: {name}")
    k = np.load(circuit_dir / "K.npz", allow_pickle=True)
    has_linear_inductor = bool(np.any(np.asarray(k["data"], dtype=float) != 0.0))
    return DeviceSpec(
        name=name,
        circuit_dir=str(circuit_dir),
        node_count=int(summary["nodes"]),
        branch_count=int(ic.size),
        ic_min_a=float(np.min(ic)),
        ic_max_a=float(np.max(ic)),
        ic_median_a=float(np.median(ic)),
        cj_nominal_f=cj_nominal_f,
        cg_nominal_f=cg_nominal_f,
        resonator_period=period,
        has_parallel_geometric_inductor=False,
        profile_is_nonuniform=bool(np.ptp(ic) > 0.0 or np.ptp(lj) > 0.0),
        pump_ghz=pump_ghz,
        signal_ghz=signal_ghz,
        port_network="50 ohm source and load from G.npz; ports 1 and 2",
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
) -> tuple[dict[str, Any], float, float, np.ndarray]:
    if method != "guarcello_banded":
        raise ValueError("phase 5 uses Guarcello's known-time-level integrator only")
    device = load_jc_device(Path(spec.circuit_dir))
    dt_s = dt_norm / spec.omega_plasma
    n_steps = int(round(tmax_norm / dt_norm))
    theta, voltage, branch_r, runtime, final_q = _integrate_jc_compiled(
        device,
        pump_current_a=pump_current_a,
        pump_hz=spec.pump_ghz * 1e9,
        signal_current_a=signal_current_a,
        signal_hz=spec.signal_ghz * 1e9,
        dt_s=dt_s,
        n_steps=n_steps,
        record_stride=20,
        initial_q=initial_state,
    )
    signal_hz = spec.signal_ghz * 1e9
    pump_hz = spec.pump_ghz * 1e9
    late = np.arange(theta.size) >= max(0, theta.size - max(10, theta.size // 2))
    trace_t = theta.copy()
    trace_v = voltage.copy()
    time_s = trace_t[late]
    voltage_ss = trace_v[late]
    amplitude = _tone_amplitude(time_s, voltage_ss, signal_hz)
    gain_absolute = 20.0 * math.log10(max(amplitude, 1e-300) / (signal_current_a * 50.0))
    gain_vs_off = float("nan") if pump_off_output is None else 20.0 * math.log10(max(amplitude / pump_off_output, 1e-300))
    total_periods = theta[-1] * spec.pump_ghz * 1e9
    row = {
        "method_attribution": "Guarcello known-time-level banded FDTD algorithm",
        "pump_current_peak_a_requested": pump_current_a,
        "pump_current_peak_a_achieved": pump_current_a,
        "gain_absolute_db": gain_absolute,
        "gain_vs_off_db": gain_vs_off,
        "gain_wideband_db": _wideband_gain(time_s, voltage_ss, signal_hz, pump_hz, signal_current_a),
        "r_j": float(np.max(branch_r[late])),
        "pump_branch_current_peak_a_achieved": float(
            np.max(branch_r[late]) * spec.ic_max_a
        ),
        "min_cos_phi": float(np.min(np.cos(device.Bphi.T @ final_q / PHI0_REDUCED))),
        "argmax_cell_index": int(np.argmax(np.abs(np.sin(device.Bphi.T @ final_q / PHI0_REDUCED)))),
        "integrator_success": True,
        "integrator_message": "known-time-level constant banded solve",
        "strobe_d1_tail": float("nan"),
        "runtime_s": runtime,
        "integrator": "guarcello_banded",
        "total_pump_periods": total_periods,
        "steady_window_pump_periods": min(100.0, total_periods),
        "effective_transient_fraction": max(0.0, 1.0 - min(100.0, total_periods) / total_periods),
        "record_stride": 20,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "steady_state_start_index": int(np.flatnonzero(late)[0]) if np.any(late) else 0,
    }
    return row, amplitude, runtime, final_q, trace_t, trace_v


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
    """Plot completed phase-5 gain and junction-drive comparisons."""
    import matplotlib.pyplot as plt

    summary_path = output / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"phase-5 summary is missing: {summary_path}")
    rows = _read_hb_rows(summary_path)
    complete = [row for row in rows if row.get("status") == "COMPLETE"]
    if not complete:
        raise ValueError(f"no completed phase-5 rows in {summary_path}")
    hb_name = name.removeprefix("jc_")
    hb_path = ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811" / hb_name / "hb_up_to_failure.csv"
    hb_rows = _read_hb_rows(hb_path)

    fdtd_current = np.array([float(row["pump_current_peak_a_requested"]) for row in complete])
    fdtd_gain = np.array([float(row["gain_vs_off_db"]) for row in complete])
    fdtd_r = np.array([float(row["r_j"]) for row in complete])
    hb_current = np.array([float(row["pump_current_peak_a"]) for row in hb_rows])
    hb_gain = np.array([float(row["gain_vs_off_db"]) for row in hb_rows])
    hb_r = np.array([float(row["pump_branch_current_max_over_ic"]) for row in hb_rows])
    failure = next(
        (current for current, row in zip(hb_current, hb_rows)
         if row.get("pump_status") not in {"VALID_CONVERGED", "VALID_SOLVED"}),
        None,
    )

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(hb_current, hb_gain, "k--", alpha=0.7, label="HB gain vs off")
    axis.plot(fdtd_current, fdtd_gain, "o-", label="transient gain vs off")
    if failure is not None:
        axis.axvline(failure, color="tab:red", linestyle=":", label="HB failure current")
    axis.set(xlabel="On-chip peak pump current (A)", ylabel="Gain vs pump-off (dB)",
             title=f"Phase 5 {name}: gain comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    gain_path = output / "phase5_gain_vs_current.png"
    fig.savefig(gain_path, dpi=180)
    plt.close(fig)
    paths.append(gain_path)

    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(hb_current, hb_r, "k--", alpha=0.7, label="HB $r_j$")
    axis.plot(fdtd_current, fdtd_r, "o-", label="transient $r_j$")
    axis.axhline(1.0, color="tab:purple", linestyle="--", label="$r_j=1$")
    if failure is not None:
        axis.axvline(failure, color="tab:red", linestyle=":", label="HB failure current")
    axis.set(xlabel="On-chip peak pump current (A)", ylabel=r"$r_j=I_{J,\max}/I_c$",
             title=f"Phase 5 {name}: junction drive comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    drive_path = output / "phase5_rj_vs_current.png"
    fig.savefig(drive_path, dpi=180)
    plt.close(fig)
    paths.append(drive_path)
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
    return row, float(payload["amplitude"]), state


def measure_device_rate(
    spec: DeviceSpec,
    *,
    dt_norm: float,
    signal_current_a: float,
    output: Path,
) -> dict[str, Any]:
    """Measure the bounded solver on exactly 200 normalized steps."""
    output.mkdir(parents=True, exist_ok=True)
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
    }
    (output / "measured_rate_200_steps.json").write_text(
        json.dumps(measurement, indent=2), encoding="utf-8",
    )
    return measurement


def run_device(name: str, *, dt_norm: float, tmax_norm: float, output: Path, per_point_budget_s: float | None = None) -> dict[str, Any]:
    circuit_dir = ROOT / "outputs" / "jc_doc_python_designs" / name
    hb_name = name.removeprefix("jc_")
    hb_dir = ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811" / hb_name
    spec = derive_device_spec(circuit_dir)
    budget = derive_time_budget(spec, dt_norm=dt_norm, tmax_norm=tmax_norm)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cost_estimate.json").write_text(
        json.dumps({"method_attribution": "our sparse transient engine vs our harmonic-balance solver", "device": asdict(spec), "time_budget": asdict(budget)}, indent=2),
        encoding="utf-8",
    )
    hb_rows = _read_hb_rows(hb_dir / "hb_up_to_failure.csv")
    requested_powers = (-32.5, -29.684, -29.053)
    selected_rows = [
        min(hb_rows, key=lambda row: abs(float(row["pump_power_dbm"]) - power))
        for power in requested_powers
    ]
    points = [float(row["pump_current_peak_a"]) for row in selected_rows]
    signal_current = math.sqrt(2.0 * 1e-3 * 10.0 ** (-100.0 / 10.0) / 50.0) / 50.0
    effective_tmax_norm = budget.tmax_norm
    measurement = measure_device_rate(
        spec, dt_norm=dt_norm, signal_current_a=signal_current, output=output,
    )
    budget_payload = asdict(budget)
    budget_payload["measured_steps_per_second"] = measurement["measured_steps_per_second"]
    budget_payload["measured_seconds_per_point"] = (
        budget.tmax_norm / dt_norm / measurement["measured_steps_per_second"]
    )
    (output / "cost_estimate.json").write_text(
        json.dumps({
            "method_attribution": "Guarcello known-time-level banded FDTD algorithm",
            "device": asdict(spec),
            "time_budget": budget_payload,
        }, indent=2),
        encoding="utf-8",
    )
    point_budget = per_point_budget_s or min(900.0, budget_payload["measured_seconds_per_point"] * 1.5)
    off_row, off_amplitude, off_state = _run_subprocess_point(
        spec, pump_current_a=0.0, start_current_a=0.0, initial_state=None,
        method="guarcello_banded", dt_norm=dt_norm, tmax_norm=effective_tmax_norm,
        signal_current_a=signal_current, pump_off_output=None,
        output=output / "pump_off", budget_s=point_budget,
    )
    (output / "pump_off" / "summary.json").write_text(
        json.dumps(off_row, indent=2), encoding="utf-8",
    )
    if off_amplitude is None or off_state is None:
        raise RuntimeError("pump-off reference did not complete")
    chosen = "guarcello_banded"
    rows: list[dict[str, Any]] = []
    previous_state = off_state
    previous_current = 0.0
    for index, current in enumerate(points):
        row, _, state = _run_subprocess_point(
            spec, pump_current_a=current, start_current_a=previous_current,
            initial_state=previous_state, method=chosen, dt_norm=dt_norm,
            tmax_norm=effective_tmax_norm, signal_current_a=signal_current,
            pump_off_output=off_amplitude, output=output / f"point_{index:03d}",
            budget_s=point_budget,
        )
        row["point_index"] = index
        rows.append(row)
        _write_csv(output / "summary.csv", rows)
        if state is not None:
            previous_state = state
            previous_current = current
    result = {
        "device": asdict(spec),
        "time_budget": asdict(budget),
        "pump_off": off_row,
        "points": len(rows),
        "method_attribution": "Guarcello known-time-level banded FDTD algorithm vs our harmonic-balance solver",
        "chosen_integrator": chosen,
        "integrator_screen": None,
        "per_point_wall_time_budget_s": point_budget,
        "status": "COMPLETE",
    }
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
    parser.add_argument("--device", choices=["jc_jtwpa", "jc_fqjtwpa", "both"], default="both")
    parser.add_argument("--dt-norm", type=float, default=0.01)
    parser.add_argument("--tmax-norm", type=float, default=20_000.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chaos/phase5")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--per-point-budget-s", type=float, default=None)
    args = parser.parse_args()
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
            spec = derive_device_spec(ROOT / "outputs/jc_doc_python_designs" / name)
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
        )
        paths = plot_phase5_device(name, args.output / name)
        print(json.dumps({"device": name, "plots": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
