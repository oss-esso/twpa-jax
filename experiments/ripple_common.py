# experiments/ripple_common.py
"""Shared helpers for the S42-ripple pump-placement workflow (twpa_jax port).

This is the Python/JAX-pipeline analogue of the Harmonia.jl ripple workflow. It
reuses the existing experiment stack:

* design build   -> ``exp07_python_ipm_design_builder`` (2- or 3-coupler IPM)
* passive S      -> ``exp09_full_ipm_gain_from_pump`` linear blocks (pump off)
* pump solve     -> ``exp08_full_ipm_pump_solve`` (subprocess)
* gain sweep     -> ``exp09_full_ipm_gain_from_pump`` (subprocess)

The passive 4-port S-matrix (including the coupler-transmission ripple ``S42``)
is obtained with the pump *off*: the exp09 linear block ``D(omega) + Khat0``
already carries the Josephson inductance through ``gamma_hat[0] = Ic/phi0``, so a
single sparse solve per (frequency, source-port) yields a genuine passive
S-parameter via the same Norton-port normalisation exp09 uses for gain.

Ports (IPM JTWPA): ``1 = signal in``, ``2 = signal out``, ``3 = pump rail``,
``4 = pump source``. ``S42`` is the passive coupler transmission whose periodic
ripple sets the +120 degrees pump placement.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from exp07_python_ipm_design_builder import (  # noqa: E402
    IPMParams,
    apply_lj_scatter,
    build_matrices,
    make_coupler_discrete,
    make_ipm,
    write_outputs,
)
from exp09_full_ipm_gain_from_pump import (  # noqa: E402
    dynamic_block,
    load_ipm,
    port_s_from_unit_current_response,
)

ROOT = _HERE.parent
EXP08 = "experiments/exp08_full_ipm_pump_solve.py"
EXP09 = "experiments/exp09_full_ipm_gain_from_pump.py"

# Dimensionless node-flux bound (psi/phi0) separating physical solves from
# past-fold runaways. Physical IPM pump states sit at O(0.1-10); a runaway
# blows up far above this.
FLUX_MAX = 1.0e3

# The two IPM variants. ``2c`` reuses the cached coupler geometry baked into
# exp07; ``3c`` re-optimises the coupler for -20 dB at 10 GHz and doubles the
# JTL array so a coupler lands every two arrays.
DESIGNS: dict[str, dict[str, Any]] = {
    "2c": {
        "overrides": {},
        "coupler_mode": "cached",
    },
    "3c": {
        "overrides": {
            "array_length": 648,
            "arrays_per_dc": 2,
            "coupling_dB": -20.0,
            "coupler_freq_hz": 10.0e9,
        },
        "coupler_mode": "optimize",
    },
}


# =============================================================================
# Design build
# =============================================================================

def build_design(
    design: str,
    ipm_dir: str | Path,
    *,
    lj_scatter_sigma: float = 0.0,
    lj_scatter_seed: int = 1,
) -> Path:
    """Build a 2- or 3-coupler IPM design and write its matrices.

    Args:
        design: ``"2c"`` or ``"3c"``.
        ipm_dir: Output directory for the exp07 design artifacts (C/G/K/Bphi
            .npz, ipm_arrays.npz, ipm_summary.json).
        lj_scatter_sigma: Multiplicative Gaussian sigma for Josephson Lj values.
            Use ``0.01`` for 1 percent junction scatter.
        lj_scatter_seed: RNG seed for deterministic scatter.

    Returns:
        The resolved ``ipm_dir`` path.

    Raises:
        ValueError: If ``design`` is not a known key of :data:`DESIGNS`.
    """
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}; choose from {list(DESIGNS)}")

    spec = DESIGNS[design]
    params = replace(IPMParams(), **spec["overrides"])
    coupler = make_coupler_discrete(params, spec["coupler_mode"])
    circuit, ends = make_ipm(params, coupler)
    scatter_meta = apply_lj_scatter(
        circuit,
        sigma=lj_scatter_sigma,
        seed=lj_scatter_seed,
    )
    mats = build_matrices(circuit)

    out = Path(ipm_dir)
    write_outputs(str(out), circuit, params, coupler, ends, mats, extra_summary=scatter_meta)
    return out


def ic_reference_a(ipm_dir: str | Path) -> float:
    """Return the median junction critical current ``Ic`` (A) for the design."""
    ipm = load_ipm(ipm_dir)
    return float(np.median(ipm.Ic))


# =============================================================================
# Passive 4-port S-matrix (pump off)
# =============================================================================

def passive_s_matrix(
    ipm_dir: str | Path,
    freqs_hz: np.ndarray,
    *,
    ports: tuple[int, ...] = (1, 2, 3, 4),
    z0_ohm: float = 50.0,
) -> np.ndarray:
    """Passive (pump-off) 4-port S-matrix on a frequency grid.

    A single LU per frequency solves every source column, giving the full
    port-to-port S at Norton-port normalisation ``S_ij = 2 V_i / (I_j Z0)``
    (minus 1 on the diagonal), matching exp09's gain convention.

    Args:
        ipm_dir: Directory holding the exp07 design matrices.
        freqs_hz: 1-D array of signal frequencies (Hz).
        ports: Port numbers to include, in output index order.
        z0_ohm: Reference port impedance.

    Returns:
        Complex array ``S`` of shape ``(F, P, P)`` indexed ``[freq, out, in]``
        over ``ports``.
    """
    ipm = load_ipm(ipm_dir)
    for p in ports:
        if p not in ipm.port_to_index:
            raise ValueError(f"port {p} not in design ports {ipm.port_to_index}")

    # Pump-off Josephson stiffness: gamma(t)=Ic/phi0 (cos 0 = 1).
    gamma_off = ipm.Ic / ipm.phi0
    khat_off_0 = (
        ipm.Bphi @ sp.diags(gamma_off, offsets=0, format="csr") @ ipm.Bphi.T
    ).astype(np.complex128).tocsr()

    n = ipm.C.shape[0]
    idx = [ipm.port_to_index[p] for p in ports]
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    s = np.zeros((freqs.size, len(ports), len(ports)), dtype=np.complex128)

    rhs = np.zeros((n, len(ports)), dtype=np.complex128)
    for col, src_index in enumerate(idx):
        rhs[src_index, col] = 1.0  # unit current at each source port

    for fi, f in enumerate(freqs):
        omega = 2.0 * math.pi * float(f)
        A = (dynamic_block(ipm, omega) + khat_off_0).tocsc()
        lu = spla.splu(A)
        y = lu.solve(rhs)  # (n, P): node fluxes for each source excitation
        for ci, src in enumerate(ports):
            for ri, out in enumerate(ports):
                v_out = 1j * omega * y[idx[ri], ci]
                s[fi, ri, ci] = port_s_from_unit_current_response(
                    v_out, source_port=src, out_port=out, z0_ohm=z0_ohm
                )
    return s


def db20(x: np.ndarray) -> np.ndarray:
    """20*log10|x| with a small floor to avoid log(0)."""
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


# =============================================================================
# S42 ripple: peaks, local period, +120 degrees placement
# =============================================================================

@dataclass(frozen=True)
class Placement:
    """A +120-degrees pump placement referenced to one S42 peak."""

    ref_peak_ghz: float
    period_ghz: float
    fp_ghz: float
    offset_mhz: float
    offset_deg: float


def find_s42_peaks(
    freq_ghz: np.ndarray,
    s42_db: np.ndarray,
    band_ghz: tuple[float, float],
) -> np.ndarray:
    """Local maxima of the passive |S42| ripple inside a band (GHz)."""
    lo, hi = band_ghz
    mask = (freq_ghz >= lo) & (freq_ghz <= hi)
    fb, sb = freq_ghz[mask], s42_db[mask]
    peaks = [
        fb[i]
        for i in range(1, len(sb) - 1)
        if sb[i] > sb[i - 1] and sb[i] >= sb[i + 1]
    ]
    return np.asarray(peaks, dtype=float)


def local_period_ghz(peak_ghz: float, peaks_ghz: np.ndarray) -> float:
    """Local ripple period at a peak: mean gap to its neighbouring peaks."""
    peaks = np.sort(np.asarray(peaks_ghz, dtype=float))
    i = int(np.argmin(np.abs(peaks - peak_ghz)))
    gaps = []
    if i > 0:
        gaps.append(peaks[i] - peaks[i - 1])
    if i < len(peaks) - 1:
        gaps.append(peaks[i + 1] - peaks[i])
    if not gaps:
        raise ValueError("need at least two S42 peaks to estimate a period")
    return float(np.mean(gaps))


def place_120(peak_ghz: float, peaks_ghz: np.ndarray) -> Placement:
    """Place fp one third of a ripple period (+120 degrees) above a peak."""
    period = local_period_ghz(peak_ghz, peaks_ghz)
    fp = peak_ghz + period / 3.0
    return Placement(
        ref_peak_ghz=float(peak_ghz),
        period_ghz=period,
        fp_ghz=float(fp),
        offset_mhz=(fp - peak_ghz) * 1e3,
        offset_deg=(fp - peak_ghz) / period * 360.0,
    )


def snap_to_120(fp_ghz: float, peaks_ghz: np.ndarray) -> tuple[Placement, float]:
    """Snap an arbitrary fp onto the nearest +120-degrees ripple placement.

    For a map candidate at some raw fp, find the peak whose ``peak + period/3``
    target is closest and shift the pump there.

    Args:
        fp_ghz: The candidate (raw map) pump frequency (GHz).
        peaks_ghz: Passive |S42| peak frequencies (GHz).

    Returns:
        ``(placement, map_offset_deg)`` where ``placement`` is the snapped +120
        target and ``map_offset_deg`` is the *original* candidate offset (deg)
        relative to the same reference peak, so the shift can be reported.
    """
    peaks = np.sort(np.asarray(peaks_ghz, dtype=float))
    targets = [place_120(float(p), peaks) for p in peaks]
    j = int(np.argmin([abs(fp_ghz - t.fp_ghz) for t in targets]))
    t = targets[j]
    map_offset_deg = (fp_ghz - t.ref_peak_ghz) / t.period_ghz * 360.0
    return t, map_offset_deg


def auto_targets(
    freq_ghz: np.ndarray,
    s42_db: np.ndarray,
    band_ghz: tuple[float, float],
    map_pump_band: tuple[float, float],
    n_points: int,
) -> list[Placement]:
    """Auto-select the strongest peaks whose +120 fp lands in the map band.

    Args:
        freq_ghz: Fine passive frequency grid (GHz).
        s42_db: Passive |S42| (dB) on that grid.
        band_ghz: Band searched for ripple peaks.
        map_pump_band: Allowed range for the placed pump ``fp`` (GHz).
        n_points: Number of placements to return.

    Returns:
        Up to ``n_points`` placements, strongest S42 peak first.
    """
    peaks = find_s42_peaks(freq_ghz, s42_db, band_ghz)
    if peaks.size < 2:
        raise ValueError("not enough S42 peaks found to place pumps")

    lo, hi = map_pump_band
    cands: list[tuple[float, Placement]] = []
    for p in peaks:
        pl = place_120(float(p), peaks)
        if lo <= pl.fp_ghz <= hi:
            peak_db = float(np.interp(p, freq_ghz, s42_db))
            cands.append((peak_db, pl))

    cands.sort(key=lambda x: x[0], reverse=True)
    return [pl for _, pl in cands[:n_points]]


# =============================================================================
# Subprocess helpers: pump ladder + gain sweep
# =============================================================================

def _run(cmd: list[str], log: Path, timeout_s: float) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout_s}s\n")
            return 124


@dataclass
class PumpOutcome:
    """Result of one pump solve at a fixed fp / current.

    Attributes:
        accepted: The physical-acceptance flag used by the ladder: the
            continuation reached the full requested current (``source_scale ==
            1``) *and* the node flux is bounded (``flux_over_phi0 < FLUX_MAX``).
            This admits fold-edge solves whose Newton residual plateaus slightly
            above tolerance -- physical, but not ``VALID_CONVERGED`` -- while
            still rejecting runs that stalled below full scale (past-fold) or
            blew the flux up.
        converged: The strict flag: exp08 ``final_status == VALID_CONVERGED``.
        reached_full_scale: Whether continuation reached ``source_scale == 1``.
        coeff_rel: Final-step relative Fourier-coefficient residual.
    """

    accepted: bool
    converged: bool
    reached_full_scale: bool
    ratio_ic: float
    current_a: float
    pump_dir: Path
    final_status: str
    coeff_rel: float
    flux_over_phi0: float
    rc: int


def _read_pump_report(pump_dir: Path) -> dict[str, Any]:
    import json

    path = pump_dir / "pump_report.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def solve_pump(
    ipm_dir: str | Path,
    pump_dir: Path,
    *,
    fp_ghz: float,
    ratio_ic: float,
    ic_a: float,
    pump_port: int = 4,
    pump_mode_count: int = 10,
    nt: int = 40,
    continuation_steps: int = 20,
    continuation_mode: str = "adaptive",
    continuation_predictor: str = "secant",
    phi0_reduced: float = 2.067833848e-15 / (2.0 * math.pi),
    timeout_s: float = 300.0,
) -> PumpOutcome:
    """Solve the pump HB at ``fp_ghz`` with pump current ``ratio_ic * Ic``.

    Uses the JC positive-odd phasor basis (the converging IPM-JTWPA settings:
    ``positive_odd_jc``, K=10 modes, Nt=40, linear-phasor seed). The default
    **adaptive continuation + secant predictor** matches the trusted pump-map
    reference pass and is what reaches the current fold robustly; ``fixed`` with
    ``continuation_steps`` is available for a cheaper cold solve away from the
    fold.
    """
    current_a = float(ratio_ic) * float(ic_a)
    cmd = [
        sys.executable,
        EXP08,
        "--ipm-dir",
        str(ipm_dir),
        "--outdir",
        str(pump_dir),
        "--pump-port",
        str(pump_port),
        "--pump-freq-ghz",
        f"{fp_ghz:.12g}",
        "--pump-current-a",
        f"{current_a:.17g}",
        "--pump-mode-policy",
        "positive_odd_jc",
        "--pump-mode-count",
        str(pump_mode_count),
        "--nt",
        str(nt),
        "--initial-guess",
        "linear_phasor",
        "--continuation-mode",
        continuation_mode,
        "--continuation-steps",
        str(continuation_steps),
        "--continuation-predictor",
        continuation_predictor,
        "--quiet",
    ]
    rc = _run(cmd, pump_dir.parent / f"{pump_dir.name}.log", timeout_s)
    report = _read_pump_report(pump_dir)
    final_status = str(report.get("final_status", "MISSING"))
    psi_max = float(report.get("solution_summary", {}).get("branch_psi_max_abs", 0.0))
    flux = psi_max / phi0_reduced if phi0_reduced else math.inf

    last = (report.get("reports") or [{}])[-1]
    source_scale = float(last.get("source_scale", 0.0))
    coeff_rel = float(last.get("coeff_rel", math.inf))
    reached_full_scale = abs(source_scale - 1.0) < 1e-9

    solution_ok = rc == 0 and (pump_dir / "pump_solution.npz").exists()
    converged = solution_ok and final_status == "VALID_CONVERGED"
    accepted = solution_ok and reached_full_scale and flux < FLUX_MAX
    return PumpOutcome(
        accepted=accepted,
        converged=converged,
        reached_full_scale=reached_full_scale,
        ratio_ic=float(ratio_ic),
        current_a=current_a,
        pump_dir=pump_dir,
        final_status=final_status,
        coeff_rel=coeff_rel,
        flux_over_phi0=flux,
        rc=rc,
    )


def ladder_pump(
    ipm_dir: str | Path,
    point_dir: Path,
    *,
    fp_ghz: float,
    ic_a: float,
    ic_ladder: list[float],
    **pump_kwargs: Any,
) -> PumpOutcome | None:
    """Run an Ic ladder at ``fp_ghz``; return the strongest accepted solve.

    "Accepted" is the fold-edge physical criterion (:attr:`PumpOutcome.accepted`):
    full source scale reached and bounded node flux. Walking the ladder upward
    and keeping the last accepted rung lands on the strongest physical pump just
    below the fold.

    Args:
        ipm_dir: Design matrices directory.
        point_dir: Base directory; each rung writes ``pump_r<ratio>/``.
        fp_ghz: Pump frequency.
        ic_a: Reference median ``Ic`` (A).
        ic_ladder: Pump-current ratios (x Ic) to try, ascending.
        **pump_kwargs: Forwarded to :func:`solve_pump`.

    Returns:
        The :class:`PumpOutcome` for the largest accepted ratio, or ``None`` if
        none were accepted.
    """
    best: PumpOutcome | None = None
    for ratio in ic_ladder:
        tag = f"pump_r{ratio:g}".replace(".", "p")
        outcome = solve_pump(
            ipm_dir,
            point_dir / tag,
            fp_ghz=fp_ghz,
            ratio_ic=ratio,
            ic_a=ic_a,
            **pump_kwargs,
        )
        if outcome.accepted:
            best = outcome
    return best


def gain_sweep(
    ipm_dir: str | Path,
    pump_dir: Path,
    gain_dir: Path,
    *,
    fp_ghz: float,
    source_port: int,
    out_port: int,
    signal_start_ghz: float,
    signal_stop_ghz: float,
    points: int,
    sidebands: int = 10,
    gamma_nt: int = 96,
    timeout_s: float = 600.0,
) -> Path:
    """Run an exp09 signal sweep for one (source, out) port pair.

    Returns:
        Path to the written ``gain_sweep.csv``.
    """
    cmd = [
        sys.executable,
        EXP09,
        "--ipm-dir",
        str(ipm_dir),
        "--pump-dir",
        str(pump_dir),
        "--outdir",
        str(gain_dir),
        "--source-port",
        str(source_port),
        "--out-port",
        str(out_port),
        "--sweep",
        "--signal-start-ghz",
        f"{signal_start_ghz:.12g}",
        "--signal-stop-ghz",
        f"{signal_stop_ghz:.12g}",
        "--points",
        str(points),
        "--sidebands",
        str(sidebands),
        "--gamma-nt",
        str(gamma_nt),
        "--fallback-pump-freq-ghz",
        f"{fp_ghz:.12g}",
    ]
    _run(cmd, gain_dir.parent / f"{gain_dir.name}.log", timeout_s)
    return gain_dir / "gain_sweep.csv"


def read_gain_sweep(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``gain_sweep.csv`` into sorted ``(signal_ghz, gain_db)`` arrays."""
    import csv as _csv

    fx: list[float] = []
    gy: list[float] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                try:
                    fx.append(float(row["signal_ghz"]))
                    gy.append(float(row["gain_db"]))
                except (KeyError, ValueError):
                    continue
    order = np.argsort(fx)
    return np.asarray(fx)[order], np.asarray(gy)[order]
