"""Experiment 10: warm-started IPM pump/gain map with a cold-vs-warm gate.

This orchestrates exp08 (pump harmonic-balance solve) and exp09 (linearized
gain) over a pump-power x pump-frequency grid, comparing two traversal
strategies:

``cold``
    Every point is solved from scratch with the legacy path (zero initial guess
    + fixed 20-step continuation). This is the trusted reference.

``warmstart``
    Each frequency column is traversed in increasing pump power. The first point
    of a column is seeded with the ``linear_phasor`` guess and solved with
    adaptive continuation; every subsequent (higher-power) point warm-starts from
    the previous converged pump solution via ``--promote-from-pump-dir`` (a single
    full-scale Newton solve, no continuation).

``both``
    Runs the cold pass then the warm pass and emits a PASS/FAIL **gate**: warm
    start is accepted only if every point converged, the per-point gain agrees
    with the cold reference within ``--gate-gain-db``, and the warm pass is
    faster in total pump runtime. This is the validation experiment.

For a large warm-only map, ``--gate-spotcheck N`` recomputes ``N`` points cold
after the warm pass and folds their gain drift into the gate, so the big run is
still guarded without paying for a full cold map.

Pump current is derived from delivered power after subtracting the line
loss. Every drive port is an ideal current source in parallel with Z0, a
matched wave port (``I`` is the incident wave's own current amplitude), so
the default ``--power-convention legacy_traveling_wave`` inverts
``P_avail = I_peak^2 * Z0 / 2``, i.e. ``I_peak = sqrt(2 * P_W / Z0)`` (see
``twpa_solver.ports``). ``--power-convention norton`` selects the alternate
Norton-generator reading (``I_peak = sqrt(8 * P_W / Z0)``) for comparison --
for a fixed dBm the Norton current is half the legacy one, so a map
regenerated with the same dBm bounds under a different convention is a
different physical sweep. The loss defaults to the measured ``loss_A10``
model ``c + a*sqrt(f) + b*f`` (dB, f in GHz); pass a flat ``--attenuation-db``
to override it.
"""

from __future__ import annotations

import argparse
import os
import csv
import dataclasses
import gc
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

_THEMIS_FREQ_RE = re.compile(r"105C5_([0-9.]+)GHz\.npy$")

# Legacy subprocess paths are kept only for compatibility. The production
# default is the in-process package path below.
EXP08 = "experiments/exp08_full_ipm_pump_solve.py"
EXP09 = "experiments/exp09_full_ipm_gain_from_pump.py"

from twpa_solver import default_loss_model  # noqa: E402
from twpa_solver.loss import signal_loss_model  # noqa: E402
from twpa_solver.map import peak_rss_bytes, run_isolated_jobs  # noqa: E402
from twpa_solver.core import (  # noqa: E402
    default_loss_model_for,
    kinetic_dc_branch_flux,
    load_circuit,
)
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.signal.gamma import load_dc_branch_flux  # noqa: E402
import twpa_solver.pump.hb as exp08  # noqa: E402
import twpa_solver.signal as exp09  # noqa: E402
import twpa_solver.pump.basis as pump_basis  # noqa: E402
from twpa_solver.pump.backends.schur_operators import (  # noqa: E402
    SchurReducedProblem,
    build_schur_problem,
)
from twpa_solver.pump.backends.schur_partition import restrict  # noqa: E402
from twpa_solver.pump.diagnostics import residual_spectrum_summary  # noqa: E402
from twpa_solver.ports import (  # noqa: E402
    port_available_power_w,
    port_current_from_power_a,
)
from twpa_solver.port_roles import (
    resolve_mixing_order,
    resolve_port_roles,
)


# =============================================================================
# Units / helpers
# =============================================================================

def dbm_to_peak_current_a(
    power_dbm: float, *, attenuation_db: float, z0_ohm: float,
    convention: str = "legacy_traveling_wave",
) -> float:
    logger.debug(
        "dbm_to_peak_current_a_start power_dbm=%s attenuation_db=%s z0_ohm=%s convention=%s",
        power_dbm, attenuation_db, z0_ohm, convention,
    )
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    source_dbm = float(power_dbm) - float(attenuation_db)
    power_w = 1.0e-3 * 10.0 ** (source_dbm / 10.0)
    current_a = port_current_from_power_a(power_w, z0_ohm, convention=convention)
    logger.debug(
        "dbm_to_peak_current_a_result source_dbm=%s power_w=%s current_a=%s",
        source_dbm, power_w, current_a,
    )
    return current_a


def rf_squid_dc_branch_flux_from_external_fraction(
    circuit_dir: Path, circuit: Any, fraction: float,
) -> np.ndarray:
    """Return the self-consistent RF-SQUID DC phase offset.

    For a loop with finite ``Lm``, the external reduced flux is not the JJ
    phase.  The nonlinear branch offset must satisfy

        phi_dc = phi_ext - beta_L sin(phi_dc).

    Circuits without RF-SQUID parameters retain the historical direct-flux
    behavior.
    """
    external_phase = float(fraction) * 2.0 * math.pi
    resolved = Path(circuit_dir) / "design_resolved.json"
    if resolved.exists():
        try:
            parameters = json.loads(resolved.read_text(encoding="utf-8")).get(
                "parameters", {}
            )
        except (OSError, json.JSONDecodeError):
            parameters = {}
        if "Lm" in parameters and "Ic" in parameters:
            beta_l = float(parameters["Lm"]) * float(parameters["Ic"]) / float(circuit.phi0)
            phi_dc = brentq(
                lambda phase: phase - external_phase + beta_l * math.sin(phase),
                external_phase - beta_l - 0.5,
                external_phase + beta_l + 0.5,
            )
            return np.full(circuit.branch_count, phi_dc * circuit.phi0, dtype=float)
    return np.full(
        circuit.branch_count, external_phase * circuit.phi0, dtype=float
    )


def peak_current_to_power_dbm(current_a: float, freq_ghz: float, args: argparse.Namespace) -> float:
    """Inverse of ``dbm_to_peak_current_a``: on-chip peak current -> pump dBm.

    Available power follows ``args.power_convention`` (legacy_traveling_wave
    default: ``P_avail = I^2 Z0 / 2``); see ``twpa_solver.ports``.
    """
    if current_a <= 0.0:
        return float("-inf")
    power_w = port_available_power_w(
        current_a, float(args.z0_ohm), convention=args.power_convention
    )
    source_dbm = 10.0 * math.log10(power_w / 1.0e-3)
    result = source_dbm + attenuation_db_for(freq_ghz, args)
    logger.debug(
        "peak_current_to_power_dbm current_a=%s freq_ghz=%s -> dbm=%s",
        current_a, freq_ghz, result,
    )
    return result


def attenuation_db_for(freq_ghz: float, args: argparse.Namespace) -> float:
    """Line attenuation (dB) at ``freq_ghz``.

    Default: the measured loss_A10 model ``c + a*sqrt(f) + b*f`` (frequency
    dependent, f in GHz). A numeric ``--attenuation-db`` overrides it with a flat
    value.
    """
    attenuation_override = getattr(args, "attenuation_db", None)
    if attenuation_override is not None:
        logger.debug(
            "attenuation_db_for freq_ghz=%s -> flat_db=%s", freq_ghz, attenuation_override,
        )
        return float(attenuation_override)
    att = float(default_loss_model().attenuation_db(float(freq_ghz)))
    logger.debug(
        "attenuation_db_for freq_ghz=%s -> model_db=%s", freq_ghz, att,
    )
    return att


def pump_solution_is_valid(
    *,
    converged: bool,
    three_wm: bool,
    configured_full_residual_gate: float | None,
    full_residual_gate_passed: bool,
) -> bool:
    """Apply the reconstructed-residual gate when the caller requests it.

    Three-wave-mixing production solves retain their existing mandatory gate.
    Four-wave-mixing solves remain unchanged unless an explicit gate is
    configured by the caller.
    """
    gate_required = three_wm or configured_full_residual_gate is not None
    return bool(converged and (not gate_required or full_residual_gate_passed))


def signal_attenuation_db_for(freq_ghz: float, args: argparse.Namespace) -> float:
    """Signal-line attenuation used when referring measured signal powers."""
    override = getattr(args, "signal_attenuation_db", None)
    if override is not None:
        return float(override)
    return float(signal_loss_model().attenuation_db(float(freq_ghz)))


def signal_ghz_for(pump_freq_ghz: float, args: argparse.Namespace) -> float:
    """Readout signal frequency for a map cell.

    Physically the map sweeps the pump frequency, so the signal must track it at a
    fixed detuning ws = wp - detuning (default 100 MHz). An explicit --signal-ghz
    overrides this with a fixed absolute signal.
    """
    if getattr(args, "signal_ghz", None) is not None:
        logger.debug(
            "signal_ghz_for pump_freq_ghz=%s -> fixed_signal_ghz=%s",
            pump_freq_ghz, args.signal_ghz,
        )
        return float(args.signal_ghz)
    result = float(pump_freq_ghz) - float(getattr(args, "signal_detuning_mhz", 100.0)) / 1000.0
    logger.debug(
        "signal_ghz_for pump_freq_ghz=%s detuning_mhz=%s -> signal_ghz=%s",
        pump_freq_ghz, getattr(args, "signal_detuning_mhz", 100.0), result,
    )
    return result


def spectrum_offsets_mhz(args: argparse.Namespace) -> list[float]:
    """Signal offsets (MHz, relative to fp) for the per-cell spectrum mode.

    Symmetric ladder around the pump: +/- start, +/- (start+step), ... e.g.
    start=100, step=250, count=5 -> +/-100, +/-350, +/-600, +/-850, +/-1100.
    The -detuning trailing point (default -100) is a member, so the spectrum
    contains the map's headline signal.
    """
    pos = [args.signal_offset_start_mhz + i * args.signal_offset_step_mhz
           for i in range(args.signal_offset_count_per_side)]
    offsets = [float(-x) for x in reversed(pos)] + [float(x) for x in pos]
    logger.debug(
        "spectrum_offsets_mhz start=%s step=%s count_per_side=%s -> n_offsets=%s",
        args.signal_offset_start_mhz, args.signal_offset_step_mhz,
        args.signal_offset_count_per_side, len(offsets),
    )
    return offsets


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def slug_float(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def point_name(index: int, power_dbm: float, pump_freq_ghz: float) -> str:
    return (
        f"point_{index:04d}_p_{slug_float(power_dbm)}dbm_"
        f"fp_{slug_float(pump_freq_ghz)}ghz"
    )


def run_command(
    cmd: list[str], *, stdout_path: Path, stderr_path: Path, timeout_s: float
) -> tuple[int, float]:
    logger.debug(
        "run_command_start argv0=%s nargs=%s timeout_s=%s stdout_path=%s",
        cmd[0] if cmd else None, len(cmd), timeout_s, stdout_path,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True,
                timeout=timeout_s, check=False,
            )
        elapsed = time.perf_counter() - start
        logger.debug(
            "run_command_result returncode=%s elapsed_s=%s", proc.returncode, elapsed,
        )
        return int(proc.returncode), elapsed
    except subprocess.TimeoutExpired:
        with stderr_path.open("a", encoding="utf-8") as err:
            err.write(f"\nTIMEOUT after {timeout_s:.3f} s\n")
        elapsed = time.perf_counter() - start
        logger.debug("run_command_timeout timeout_s=%s elapsed_s=%s", timeout_s, elapsed)
        return 124, elapsed


# =============================================================================
# Metric extraction
# =============================================================================

def _final_failure_reason(report: dict[str, Any] | None) -> str | None:
    if report is None:
        return None
    reports = report.get("reports", [])
    final = reports[-1] if reports else {}
    reason = final.get("failure_reason")
    return str(reason) if reason else None


def boundary_predictor_status(
    current_over_ic: float | None,
    min_cos_phase: float | None,
) -> str:
    """Classify the current/tangent diagnostic without declaring a failure."""
    if current_over_ic is None and min_cos_phase is None:
        return "NOT_AVAILABLE"
    ratio = float(current_over_ic) if current_over_ic is not None else float("nan")
    tangent = float(min_cos_phase) if min_cos_phase is not None else float("nan")
    if (np.isfinite(ratio) and ratio >= 1.0) or (
        np.isfinite(tangent) and tangent < 0.0
    ):
        return "BOUNDARY_PREDICTED"
    if (
        np.isfinite(ratio) and ratio >= 0.9
        and np.isfinite(tangent) and tangent <= 0.2
    ):
        return "APPROACHING_BOUNDARY"
    return "SUBCRITICAL"


def pump_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {k: None for k in (
            "pump_runtime_s", "pump_factor_runtime_s",
            "pump_preconditioner_assembly_runtime_s",
            "pump_preconditioner_numeric_factor_runtime_s", "pump_coeff_rel",
            "pump_time_rel", "pump_newton_total", "pump_branch_current_max",
            "pump_branch_current_max_over_ic", "pump_strongest_branch_index",
            "pump_branch_min_cos_phase", "pump_boundary_predictor_status",
            "pump_residual_max_omitted_mode_rel", "sidebands",
            "single_tone_forced", "pump_failure_reason",
        )}
    reports = report.get("reports", [])
    final = reports[-1] if reports else {}
    summ = report.get("solution_summary", {})
    ratio = finite_or_none(summ.get("branch_current_max_over_ic"))
    min_cos = finite_or_none(summ.get("branch_min_cos_phase"))
    return {
        "pump_runtime_s": sum(finite_or_none(r.get("runtime_s")) or 0.0 for r in reports) if reports else None,
        "pump_factor_runtime_s": sum(finite_or_none(r.get("factor_runtime_s")) or 0.0 for r in reports) if reports else None,
        "pump_preconditioner_assembly_runtime_s": sum(finite_or_none(r.get("preconditioner_assembly_runtime_s")) or 0.0 for r in reports) if reports else None,
        "pump_preconditioner_numeric_factor_runtime_s": sum(finite_or_none(r.get("preconditioner_numeric_factor_runtime_s")) or 0.0 for r in reports) if reports else None,
        "pump_coeff_rel": finite_or_none(final.get("coeff_rel")),
        "pump_time_rel": finite_or_none(final.get("time_rel")),
        "pump_newton_total": int(sum(int(r.get("newton_iterations", 0)) for r in reports)),
        "pump_branch_current_max": finite_or_none(summ.get("branch_i_max_abs")),
        "pump_branch_current_max_over_ic": ratio,
        "pump_strongest_branch_index": summ.get("strongest_branch_index"),
        "pump_branch_min_cos_phase": min_cos,
        "pump_boundary_predictor_status": boundary_predictor_status(ratio, min_cos),
        "pump_residual_max_omitted_mode_rel": finite_or_none(
            report.get("metadata", {})
            .get("pump_residual_spectrum", {})
            .get("max_omitted_mode_rel")
        ),
        "pump_failure_reason": _final_failure_reason(report),
        "sidebands": report.get("metadata", {}).get("sidebands"),
    }


def gain_metrics(report: dict[str, Any] | None) -> dict[str, float | None]:
    if report is None:
        return {k: None for k in (
            "gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db",
            "signal_ghz", "linear_rel_residual", "gain_total_runtime_s",
            "gain_gamma_hat_runtime_s", "gain_khat_build_runtime_s",
            "gain_khat_off_runtime_s", "gain_matrix_assemble_runtime_s", "gain_factor_solve_runtime_s",
            "gain_baseline_off_runtime_s", "gain_baseline_pumpdiag_runtime_s",
        )}
    results = report.get("results", [])
    valid = [r for r in results if finite_or_none(r.get("gain_db")) is not None]
    best = max(valid, key=lambda r: float(r["gain_db"])) if valid else {}
    return {
        "gain_db": finite_or_none(best.get("gain_db")),
        "gain_vs_off_db": finite_or_none(best.get("gain_vs_off_db")),
        "gain_vs_pumpdiag_db": finite_or_none(best.get("gain_vs_pumpdiag_db")),
        "signal_ghz": finite_or_none(best.get("signal_ghz")),
        "linear_rel_residual": finite_or_none(best.get("linear_rel_residual")),
        "gain_total_runtime_s": finite_or_none(report.get("metadata", {}).get("total_runtime_s")),
        "gain_gamma_hat_runtime_s": finite_or_none(report.get("metadata", {}).get("gamma_hat_runtime_s")),
        "gain_khat_build_runtime_s": finite_or_none(report.get("metadata", {}).get("khat_build_runtime_s")),
        "gain_khat_off_runtime_s": finite_or_none(report.get("metadata", {}).get("khat_off_build_runtime_s")),
        "gain_matrix_assemble_runtime_s": finite_or_none(best.get("assemble_runtime_s")),
        "gain_factor_solve_runtime_s": finite_or_none(best.get("factor_solve_runtime_s")),
        "gain_baseline_off_runtime_s": finite_or_none(best.get("baseline_off_runtime_s")),
        "gain_baseline_pumpdiag_runtime_s": finite_or_none(best.get("baseline_pumpdiag_runtime_s")),
    }


def pump_status(report: dict[str, Any] | None, returncode: int) -> str:
    if returncode != 0:
        logger.debug("pump_status returncode=%s -> ERROR", returncode)
        return "ERROR"
    if report is None:
        logger.debug("pump_status report=None -> MISSING")
        return "MISSING"
    status = str(report.get("final_status", "UNKNOWN"))
    logger.debug("pump_status final_status=%s", status)
    return status


def gain_status(report: dict[str, Any] | None, returncode: int) -> str:
    if returncode != 0:
        logger.debug("gain_status returncode=%s -> ERROR", returncode)
        return "ERROR"
    if report is None:
        logger.debug("gain_status report=None -> MISSING")
        return "MISSING"
    results = report.get("results", [])
    if results and all(r.get("status") == "VALID_SOLVED" for r in results):
        logger.debug("gain_status n_results=%s -> VALID_SOLVED", len(results))
        return "VALID_SOLVED"
    logger.debug("gain_status n_results=%s -> UNKNOWN", len(results))
    return "UNKNOWN"


# =============================================================================
# Single point execution
# =============================================================================

@dataclass
class GridPoint:
    index: int
    i_power: int
    j_freq: int
    power_dbm: float
    pump_freq_ghz: float
    current_a: float


def pump_flags_cold(args: argparse.Namespace) -> list[str]:
    return [
        "--initial-guess", "zero",
        "--continuation-mode", "fixed",
        "--continuation-steps", str(args.continuation_steps),
    ]


def pump_flags_warm_seed(args: argparse.Namespace) -> list[str]:
    return [
        "--initial-guess", "linear_phasor",
        "--linear-seed-maxiter", str(args.linear_seed_maxiter),
        "--continuation-mode", "adaptive",
        "--adaptive-initial-step", str(args.adaptive_initial_step),
        "--adaptive-min-step", str(args.adaptive_min_step),
    ]


def run_point(
    point: GridPoint,
    pass_dir: Path,
    args: argparse.Namespace,
    *,
    pump_flags: list[str],
    promote_from: Path | None,
) -> dict[str, Any]:
    logger.debug(
        "run_point_start index=%s power_dbm=%s freq_ghz=%s promote_from=%s",
        point.index, point.power_dbm, point.pump_freq_ghz, promote_from,
    )
    pdir = pass_dir / "points" / point_name(point.index, point.power_dbm, point.pump_freq_ghz)
    pump_dir = pdir / "pump"
    gain_dir = pdir / "gain"
    pdir.mkdir(parents=True, exist_ok=True)

    point_start = time.perf_counter()

    # JC-source convention: JosephsonCircuits' frequency-domain port current maps
    # to a physical drive of 2*I*cos(wt) under the positive-phasor (x = 2 Re sum X)
    # reconstruction, so exp08's pump current must be 2x the physical port current
    # to match JC. This is the documented "pump scale 2" used by all exp14 parity
    # runs; without it the JTWPA is under-pumped ~2x and shows almost no gain.
    injected_current = point.current_a * args.pump_current_jc_scale
    pump_cmd = [
        args.python_executable, EXP08,
        "--ipm-dir", str(args.circuit_dir),
        "--outdir", str(pump_dir),
        "--pump-port", str(args.pump_port),
        "--pump-freq-ghz", f"{point.pump_freq_ghz:.12g}",
        "--pump-current-a", f"{injected_current:.17g}",
        "--pump-mode-policy", str(args.pump_mode_policy),
        "--nt", str(args.nt),
        "--newton-tol", str(args.newton_tol),
        "--quiet",
        *pump_flags,
    ]
    if args.pump_mode_count is not None:
        pump_cmd.extend(["--pump-mode-count", str(args.pump_mode_count)])
    else:
        pump_cmd.extend(["--harmonics", str(args.harmonics)])
    if promote_from is not None:
        pump_cmd.extend(["--promote-from-pump-dir", str(promote_from)])

    logger.debug("run_point_pump_subprocess_dispatch index=%s", point.index)
    pump_rc, pump_wall_runtime_s = run_command(
        pump_cmd,
        stdout_path=pdir / "pump_stdout.txt",
        stderr_path=pdir / "pump_stderr.txt",
        timeout_s=args.pump_timeout_s,
    )
    pump_report = read_json(pump_dir / "pump_report.json")
    p_status = pump_status(pump_report, pump_rc)
    logger.debug(
        "run_point_pump_result index=%s rc=%s pump_status=%s runtime_s=%s",
        point.index, pump_rc, p_status, pump_wall_runtime_s,
    )

    gain_rc = -1
    gain_report = None
    force_single_tone = bool(getattr(args, "force_single_tone", False))
    if p_status == "VALID_CONVERGED" and not force_single_tone:
        logger.debug("run_point_gain_subprocess_dispatch index=%s", point.index)
        gain_cmd = [
            args.python_executable, EXP09,
            "--ipm-dir", str(args.circuit_dir),
            "--pump-dir", str(pump_dir),
            "--outdir", str(gain_dir),
            "--z0-ohm", str(args.z0_ohm),
            "--source-port", str(args.source_port),
            "--out-port", str(args.out_port),
            "--signal-ghz", f"{signal_ghz_for(point.pump_freq_ghz, args):.12g}",
            "--sidebands", str(args.sidebands),
            "--gamma-nt", str(args.gamma_nt),
            "--fallback-pump-freq-ghz", f"{point.pump_freq_ghz:.12g}",
        ]
        gain_rc, gain_wall_runtime_s = run_command(
            gain_cmd,
            stdout_path=pdir / "gain_stdout.txt",
            stderr_path=pdir / "gain_stderr.txt",
            timeout_s=args.gain_timeout_s,
        )
        gain_report = read_json(gain_dir / "gain_report.json")
        logger.debug(
            "run_point_gain_result index=%s rc=%s runtime_s=%s",
            point.index, gain_rc, gain_wall_runtime_s,
        )
    else:
        gain_wall_runtime_s = None
        logger.debug(
            "run_point_gain_skipped index=%s reason=pump_not_converged pump_status=%s",
            point.index, p_status,
        )
    g_status = (
        "SKIPPED_SINGLE_TONE"
        if force_single_tone and p_status == "VALID_CONVERGED"
        else gain_status(gain_report, gain_rc)
    )

    status = "PASS" if (
        p_status == "VALID_CONVERGED"
        and g_status in {"VALID_SOLVED", "SKIPPED_SINGLE_TONE"}
    ) else "ERROR"
    logger.debug(
        "run_point_result index=%s status=%s pump_status=%s gain_status=%s",
        point.index, status, p_status, g_status,
    )

    row: dict[str, Any] = {
        "point_index": point.index,
        "i_power": point.i_power,
        "j_freq": point.j_freq,
        "pump_power_dbm": point.power_dbm,
        "pump_freq_ghz": point.pump_freq_ghz,
        "pump_current_peak_a": point.current_a,
        "status": status,
        "pump_status": p_status,
        "gain_status": g_status,
        "sidebands": int(args.sidebands),
        "single_tone_forced": force_single_tone,
        "warm_started": promote_from is not None,
        "elapsed_s": time.perf_counter() - point_start,
        "pump_wall_runtime_s": pump_wall_runtime_s,
        "gain_wall_runtime_s": gain_wall_runtime_s,
        "pump_dir": str(pump_dir),
    }
    row.update(pump_metrics(pump_report))
    row.update(gain_metrics(gain_report))
    signal_frequency = row.get("signal_ghz")
    row["signal_attenuation_db"] = signal_attenuation_db_for(
        float(signal_frequency) if signal_frequency is not None
        else signal_ghz_for(point.pump_freq_ghz, args),
        args,
    )
    return row


# =============================================================================
# Passes
# =============================================================================

def run_cold_pass(
    points: list[GridPoint], pass_dir: Path, args: argparse.Namespace
) -> list[dict[str, Any]]:
    logger.debug("run_cold_pass_start n_points=%s pass_dir=%s", len(points), pass_dir)
    rows: list[dict[str, Any]] = []
    total = len(points)
    for point in points:
        logger.debug(
            "run_cold_pass_loop_enter index=%s of=%s power_dbm=%s freq_ghz=%s",
            point.index, total, point.power_dbm, point.pump_freq_ghz,
        )
        row = run_point(point, pass_dir, args, pump_flags=pump_flags_cold(args), promote_from=None)
        rows.append(row)
        logger.debug(
            "run_cold_pass_loop_exit index=%s status=%s", point.index, row["status"],
        )
        print(
            f"[cold {point.index + 1}/{total}] P={point.power_dbm:.4g} dBm "
            f"fp={point.pump_freq_ghz:.4g} GHz status={row['status']} "
            f"gain={row.get('gain_db')} pump_s={row.get('pump_runtime_s')}",
            flush=True,
        )
    return rows


def run_warm_pass(
    points: list[GridPoint],
    pass_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Traverse each frequency column in increasing power, warm-starting."""
    logger.debug("run_warm_pass_start n_points=%s pass_dir=%s", len(points), pass_dir)
    by_col: dict[int, list[GridPoint]] = {}
    for point in points:
        by_col.setdefault(point.j_freq, []).append(point)

    rows: list[dict[str, Any]] = []
    total = len(points)
    done = 0
    for j in sorted(by_col):
        column = sorted(by_col[j], key=lambda p: p.power_dbm)
        logger.debug("run_warm_pass_column_start j_freq=%s n_points=%s", j, len(column))
        previous_pump_dir: Path | None = None
        for point in column:
            promote = previous_pump_dir
            # First point of the column (or after a failure) is seeded with
            # linear_phasor + adaptive; the rest warm-start from the neighbour
            # via a single full-scale Newton solve. Pass zero-init flags on
            # warm-start points so exp08 skips the unused linear_phasor seed.
            flags = ["--initial-guess", "zero"] if promote is not None else pump_flags_warm_seed(args)
            logger.debug(
                "run_warm_pass_point_enter index=%s power_dbm=%s mode=%s",
                point.index, point.power_dbm, "warm" if promote is not None else "seed",
            )
            row = run_point(
                point, pass_dir, args,
                pump_flags=flags,
                promote_from=promote,
            )
            # If a warm-start (promote) point diverged, retry once from a fresh
            # linear_phasor + adaptive seed (which has a fixed-continuation
            # fallback). This recovers stiff points where a single Newton solve
            # off the neighbour fails but a graded solve still converges.
            retried = False
            if row["status"] != "PASS" and promote is not None:
                logger.debug(
                    "run_warm_pass_retry_reseed index=%s prior_status=%s",
                    point.index, row["status"],
                )
                retry = run_point(
                    point, pass_dir, args,
                    pump_flags=pump_flags_warm_seed(args),
                    promote_from=None,
                )
                if retry["status"] == "PASS":
                    row = retry
                    retried = True
                logger.debug(
                    "run_warm_pass_retry_result index=%s retry_status=%s accepted=%s",
                    point.index, retry["status"], retried,
                )
            row["warm_retry_reseed"] = retried
            rows.append(row)
            done += 1
            logger.debug(
                "run_warm_pass_point_exit index=%s status=%s retried=%s",
                point.index, row["status"], retried,
            )
            print(
                f"[warm {done}/{total}] P={point.power_dbm:.4g} dBm "
                f"fp={point.pump_freq_ghz:.4g} GHz "
                f"{'WARM' if promote is not None else 'seed'}"
                f"{'+reseed' if retried else ''} "
                f"status={row['status']} gain={row.get('gain_db')} "
                f"pump_s={row.get('pump_runtime_s')}",
                flush=True,
            )
            # Only chain off a converged neighbour.
            if row["status"] == "PASS":
                previous_pump_dir = point_pump_dir(point, pass_dir)
            else:
                previous_pump_dir = None
    rows.sort(key=lambda r: r["point_index"])
    logger.debug("run_warm_pass_end n_rows=%s", len(rows))
    return rows


def point_pump_dir(point: GridPoint, pass_dir: Path) -> Path:
    return pass_dir / "points" / point_name(point.index, point.power_dbm, point.pump_freq_ghz) / "pump"


# =============================================================================
# In-process executor (no per-point subprocess imports; real_coupled precond)
# =============================================================================

class InProcessEngine:
    """Run the exp08 pump solve + exp09 gain in this process.

    The IPM matrices and the heavy numpy/scipy imports are paid once instead of
    per point. Numerics are identical to the subprocess path: the same exp08 and
    exp09 functions are called, and the pump solution is still round-tripped
    through ``pump_solution.npz`` so exp09's gamma/khat pipeline is byte-for-byte
    the same. ``real_coupled`` preconditioning gives a bit-identical pump
    solution while cutting GMRES iterations ~50x.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        logger.debug(
            "InProcessEngine_init_start circuit_dir=%s pump_port=%s source_port=%s "
            "out_port=%s pump_backend=%s pump_mode_policy=%s",
            args.circuit_dir, args.pump_port, args.source_port, args.out_port,
            getattr(args, "inproc_pump_backend", None), args.pump_mode_policy,
        )
        self.args = args
        self.ipm08 = load_circuit(args.circuit_dir)
        self.ipm09 = load_circuit(args.circuit_dir)
        if args.loss_model == "auto":
            args.loss_model = default_loss_model_for(self.ipm09)
        self.branch = make_branch_law(self.ipm08)
        if getattr(args, "dc_solution", None):
            self.dc_branch_flux = load_dc_branch_flux(args.dc_solution, self.ipm08)
        else:
            self.dc_branch_flux = kinetic_dc_branch_flux(
                self.ipm08, getattr(args, "dc_current_a", 0.0)
            )
        flux_fraction = getattr(args, "dc_branch_flux_over_phi0", None)
        if flux_fraction is not None:
            self.dc_branch_flux = rf_squid_dc_branch_flux_from_external_fraction(
                args.circuit_dir, self.ipm08, float(flux_fraction)
            )
        if self.dc_branch_flux is None:
            self.dc_branch_flux = np.zeros(self.ipm08.branch_count, dtype=float)
        self.pump_idx = self.ipm08.port_to_index[args.pump_port]
        self.source_idx = self.ipm09.port_to_index[args.source_port]
        self.out_idx = self.ipm09.port_to_index[args.out_port]
        self.ic_median = float(np.median(self.ipm08.Ic))
        self.ports = list(self.ipm08.port_to_index.values())
        # Schur partition is constant in power -> cache per pump frequency, but
        # LRU-bounded: each partition holds a large factorized `_fast_coupled`
        # block, and the warm pass finishes one frequency column before moving
        # on, so only the current column's partition is live. Caching every
        # frequency unbounded is what OOMs large maps (50 partitions ~ 16 GB).
        # Keep the last few (insertion-ordered dict acts as the LRU).
        self._schur_part_cache: dict[tuple[float, tuple[int, ...]], Any] = {}
        self._schur_cache_max = max(
            1, int(getattr(args, "inproc_schur_cache_size", 2))
        )
        self._signal_schur_part_cache: dict[tuple[Any, ...], Any] = {}
        self._signal_schur_cache_max = max(1, int(getattr(args, "inproc_schur_cache_size", 2)))
        logger.debug(
            "InProcessEngine_init_done pump_idx=%s source_idx=%s out_idx=%s "
            "n_ports=%s schur_cache_max=%s",
            self.pump_idx, self.source_idx, self.out_idx, len(self.ports),
            self._schur_cache_max,
        )

    @staticmethod
    def _release_cached_partition(part: Any) -> None:
        """Release one cached Schur partition before dropping its reference."""
        release = getattr(part, "release", None)
        if callable(release):
            release()
            return
        fast = getattr(part, "_fast_coupled", None)
        if fast is not None:
            fast_release = getattr(fast, "release", None)
            if callable(fast_release):
                fast_release()
            part._fast_coupled = None

    def clear_schur_cache(self) -> None:
        """Release all cached pump Schur partitions and native factors."""
        cached = list(self._schur_part_cache.values())
        self._schur_part_cache.clear()
        for part in cached:
            self._release_cached_partition(part)
        gc.collect()
        logger.debug(
            "engine_schur_cache_cleared released_partitions=%d", len(cached)
        )

    def _settings(self) -> exp08.NewtonKrylovSettings:
        # Intra-cell continuation predictor: tangent uses the exact lambda-tangent
        # Euler step; every other mode keeps the legacy (copy/secant-at-inter-cell)
        # behaviour, i.e. no intra-cell predictor on the seed path.
        continuation = getattr(
            self.args,
            "inproc_continuation",
            "adaptive_secant",
        )
        cont_pred = {
            "adaptive_secant": "secant",
            "adaptive_tangent": "tangent",
        }.get(continuation, "none")
        logger.debug(
            "engine_settings continuation=%s continuation_predictor=%s "
            "preconditioner=%s max_newton=%s gmres_maxiter=%s deadline_s=%s",
            continuation, cont_pred, self.args.inproc_preconditioner,
            self.args.inproc_max_newton, self.args.inproc_gmres_maxiter,
            self.args.inproc_solve_deadline_s,
        )
        high_power = bool(getattr(self.args, "high_power_recovery", False))
        max_newton = int(self.args.inproc_max_newton)
        if high_power:
            max_newton = max(max_newton, int(self.args.high_power_max_newton))
        stall_patience = (
            int(self.args.high_power_stall_patience)
            if high_power
            else int(self.args.inproc_stall_patience)
        )
        min_alpha = (
            float(self.args.high_power_min_alpha)
            if high_power
            else float(self.args.inproc_min_alpha)
        )
        return exp08.NewtonKrylovSettings(
            newton_tol=self.args.newton_tol, max_newton=max_newton, gmres_rtol=1e-7,
            gmres_atol=0.0, gmres_restart=60, gmres_maxiter=self.args.inproc_gmres_maxiter,
            min_alpha=min_alpha,
            preconditioner=self.args.inproc_preconditioner, compute_time_residual=True, verbose=False,
            continuation_predictor=cont_pred, jvp_mode="aft",
            stall_ratio=(
                float(self.args.high_power_stall_ratio)
                if high_power else float(self.args.inproc_stall_ratio)
            ),
            stall_patience=stall_patience,
            solve_deadline_s=self.args.inproc_solve_deadline_s,
            precond_reuse=self.args.inproc_precond_reuse,
            precond_reuse_refresh_gmres=self.args.inproc_precond_refresh_gmres,
        )

    def _build_problem(
        self,
        freq_ghz: float,
        current_a: float,
        *,
        harmonics: int | None = None,
        nt: int | None = None,
        mode_count: int | None = None,
    ):
        requested_harmonics = self.args.harmonics if harmonics is None else int(harmonics)
        requested_nt = self.args.nt if nt is None else int(nt)
        logger.debug(
            "engine_build_problem_start freq_ghz=%s current_a=%s policy=%s "
            "mode_count=%s harmonics=%s nt=%s",
            freq_ghz, current_a, self.args.pump_mode_policy,
            self.args.pump_mode_count if mode_count is None else mode_count,
            requested_harmonics, requested_nt,
        )
        omega = 2.0 * math.pi * freq_ghz * 1e9
        basis = pump_basis.resolve_pump_basis(
            policy=("dense_real" if self.args.mixing_order == 3
                     and self.args.pump_mode_policy == "positive_odd_jc"
                     else self.args.pump_mode_policy), omega_p=omega,
            harmonics=requested_harmonics,
            mode_count=(
                self.args.pump_mode_count if mode_count is None else int(mode_count)
            ),
            explicit_modes=None, design_meta=self.ipm08.summary,
        )
        if self.args.mixing_order == 3 and 0 not in basis.modes:
            # A flux-biased Josephson law generates a pump-induced DC component.
            # Keep it in the production HB unknown set; omitting it can make the
            # retained-mode residual appear converged while the full DAE is not.
            basis = pump_basis.with_dynamic_dc(basis)
        grid = exp08.HarmonicGrid(modes=basis.k, nt=requested_nt, omega=omega)
        problem = exp08.FullIPMPumpProblem(
            C=self.ipm08.C, G=self.ipm08.G, K=self.ipm08.K, Bphi=self.ipm08.Bphi,
            branch=self.branch, grid=grid, pump_node_index=self.pump_idx,
            pump_current_a=current_a, source_mode=basis.source_mode,
            loss_model=default_loss_model_for(self.ipm08),
            dc_branch_flux=self.dc_branch_flux,
        )
        logger.debug(
            "engine_build_problem_complete freq_ghz=%s omega=%s modes=%r "
            "problem_shape=%s",
            freq_ghz, omega, basis.modes, problem.zeros().shape,
        )
        return problem, basis, omega

    def project_to_base_pump_basis(
        self, freq_ghz: float, X: np.ndarray | None,
    ) -> np.ndarray | None:
        """Return a chained state in the next point's production basis.

        High-power harmonic enrichment may validate a point on a richer basis
        than the map's base ``positive_odd_jc`` basis.  The enriched state is
        useful for diagnostics and output, but it must be projected back before
        it becomes ``last_good_X``: recovery routines for the next map point
        are built on the base Schur problem.  Keeping this invariant prevents
        a richer failed/accepted iterate from entering a lower-dimensional
        problem and producing a misleading broadcast error.
        """
        if X is None:
            return None
        source = getattr(self, "_last_pump_basis", None)
        if source is None or X.ndim != 2 or X.shape[0] != source.n_modes:
            return X
        omega = 2.0 * math.pi * float(freq_ghz) * 1e9
        policy = (
            "dense_real"
            if self.args.mixing_order == 3
            and self.args.pump_mode_policy == "positive_odd_jc"
            else self.args.pump_mode_policy
        )
        destination = pump_basis.resolve_pump_basis(
            policy=policy,
            omega_p=omega,
            harmonics=self.args.harmonics,
            mode_count=self.args.pump_mode_count,
            explicit_modes=None,
            design_meta=self.ipm08.summary,
        )
        if self.args.mixing_order == 3 and 0 not in destination.modes:
            destination = pump_basis.with_dynamic_dc(destination)
        if source.modes == destination.modes:
            return X
        projected = pump_basis.promote_solution_to_basis(
            X, source, destination
        )
        logger.debug(
            "engine_project_chained_state_to_base_basis freq_ghz=%s "
            "src_modes=%s dst_modes=%s shape=%s",
            freq_ghz, source.modes, destination.modes, projected.shape,
        )
        return projected

    def _make_solve_problem(self, full_problem, freq_ghz: float):
        """The problem actually solved for a cell: Schur-reduced or full.

        For the Schur backend the per-frequency partition is cached (LRU); the
        retained solution vector has a constant (port-node) shape across all
        frequencies, so chained warm starts stay shape-compatible. The full
        backend returns the full problem unchanged.
        """
        if self.args.inproc_pump_backend != "schur_cpu_mt":
            logger.debug("engine_make_solve_problem backend=full freq_ghz=%s", freq_ghz)
            return full_problem
        cache = self._schur_part_cache
        cache_key = (float(freq_ghz), tuple(int(round(k)) for k in full_problem.grid.k))
        part = cache.pop(cache_key, None)  # pop-then-reinsert -> most-recent (LRU)
        logger.debug(
            "engine_make_solve_problem backend=schur freq_ghz=%s modes=%s "
            "cache_hit=%s cache_size_before=%d",
            freq_ghz, cache_key[1], part is not None, len(cache),
        )
        sprob = (SchurReducedProblem(full=full_problem, partition=part)
                 if part is not None
                 else build_schur_problem(full_problem, self.ports))
        cache[cache_key] = sprob.part
        while len(cache) > self._schur_cache_max:
            evicted = next(iter(cache))
            evicted_part = cache.pop(evicted)
            self._release_cached_partition(evicted_part)
            logger.debug("engine_schur_cache_evict key=%s", evicted)
            gc.collect()
        logger.debug(
            "engine_make_solve_problem_complete freq_ghz=%s retained=%d eliminated=%d "
            "cache_size_after=%d",
            freq_ghz, sprob.n, sprob.part.p, len(cache),
        )
        return sprob

    def _adaptive_harmonic_enrichment(
        self,
        point: GridPoint,
        injected: float,
        full_problem: Any,
        basis: pump_basis.PumpBasis,
        solve_problem: Any,
        X: np.ndarray,
        reports: list[Any],
        *,
        retry_failed: bool = False,
    ) -> tuple[Any, pump_basis.PumpBasis, Any, np.ndarray, list[Any], dict[str, Any]]:
        """Warm-promote a pump when omitted harmonics dominate.

        In high-power mode this method is also allowed to start from the last
        failed Newton iterate.  A failed low-basis solve can still contain the
        correct waveform envelope; enriching that iterate avoids throwing away
        the information that the residual spectrum just exposed.
        """
        info: dict[str, Any] = {
            "enabled": bool(
                getattr(self.args, "adaptive_harmonics", False)
                or getattr(self.args, "high_power_recovery", False)
            ),
            "initial_modes": list(basis.modes),
            "final_modes": list(basis.modes),
            "promotions": [],
            "stop_reason": "disabled",
        }
        if not info["enabled"]:
            return full_problem, basis, solve_problem, X, reports, info

        target_time_rel = float(
            getattr(self.args, "harmonic_enrichment_time_rel", 1e-4)
        )
        max_harmonic = int(
            getattr(
                self.args,
                "high_power_harmonic_max_mode",
                getattr(self.args, "harmonic_enrichment_max", 9),
            )
            if getattr(self.args, "high_power_recovery", False)
            else getattr(self.args, "harmonic_enrichment_max", 9)
        )
        current_full = (
            solve_problem.reconstruct_full(X)
            if solve_problem is not full_problem and hasattr(solve_problem, "reconstruct_full")
            else X
        )
        norms = full_problem.norms(current_full, 1.0, True)
        info["initial_time_rel"] = norms["time_rel"]
        info["stop_reason"] = "full_residual_gate"
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())
        last_converged = bool(reports and reports[-1].converged)

        while (
            (retry_failed and not last_converged)
            or float(norms["time_rel"] or 0.0) > target_time_rel
        ):
            next_max = max(int(max(basis.modes)), 1) + 2
            if next_max > max_harmonic:
                info["stop_reason"] = "harmonic_limit"
                break
            previous_time_rel = float(norms["time_rel"] or 0.0)
            if basis.policy == "positive_odd_jc":
                next_count = len(basis.modes) + 1
                next_full, next_basis, _next_omega = self._build_problem(
                    point.pump_freq_ghz,
                    injected,
                    harmonics=next_max,
                    nt=max(int(self.args.nt), 2 * next_max + 4),
                    mode_count=next_count,
                )
            else:
                next_full, next_basis, _next_omega = self._build_problem(
                    point.pump_freq_ghz,
                    injected,
                    harmonics=next_max,
                    nt=max(int(self.args.nt), 2 * next_max + 4),
                )
            promoted = pump_basis.promote_solution_to_basis(
                current_full, basis, next_basis
            )
            next_solve = self._make_solve_problem(
                next_full, point.pump_freq_ghz
            )
            next_seed = (
                restrict(promoted, next_solve.part)
                if next_solve is not next_full
                else promoted
            )
            next_X, next_report = solver.solve_one(next_solve, next_seed, 1.0)
            next_full_X = (
                next_solve.reconstruct_full(next_X)
                if next_solve is not next_full
                else next_X
            )
            next_norms = next_full.norms(next_full_X, 1.0, True)
            info["promotions"].append({
                "from_modes": list(basis.modes),
                "to_modes": list(next_basis.modes),
                "time_rel": next_norms["time_rel"],
                "coeff_rel": next_norms["coeff_rel"],
                "converged": bool(next_report.converged),
            })
            if (
                next_report.converged
                and last_converged
                and float(next_norms["time_rel"] or 0.0) >= previous_time_rel
            ):
                info["stop_reason"] = "no_residual_improvement"
                break
            full_problem, basis, solve_problem, X = (
                next_full, next_basis, next_solve, next_X
            )
            reports = [*reports, next_report]
            current_full = next_full_X
            norms = next_norms
            last_converged = bool(next_report.converged)
            info["final_modes"] = list(basis.modes)

        info["final_time_rel"] = norms["time_rel"]
        info["final_coeff_rel"] = norms["coeff_rel"]
        return full_problem, basis, solve_problem, X, reports, info

    def build_problem_for(self, point: GridPoint):
        """Full pump problem bundle for a grid cell (no Schur reduction).

        Used by the traversal orchestrator to rank predictor candidates by
        residual before solving, and reused by ``solve_point`` via ``prebuilt``
        so the (cheap) problem build is not paid twice.
        """
        injected = point.current_a * self.args.pump_current_jc_scale
        logger.debug(
            "engine_build_problem_for point=%s power_dbm=%s freq_ghz=%s "
            "physical_current_a=%s injected_current_a=%s",
            point.index, point.power_dbm, point.pump_freq_ghz,
            point.current_a, injected,
        )
        full_problem, basis, omega = self._build_problem(point.pump_freq_ghz, injected)
        return full_problem, basis, omega, injected

    def residual_norm(self, full_problem, X: np.ndarray | None) -> float:
        """Relative coefficient residual of guess ``X`` at full drive (lambda=1).

        The ranking key for the residual-ranked predictor portfolio. Returns
        ``inf`` for a missing or shape-mismatched guess so it sorts last.
        """
        if X is None or X.shape != full_problem.zeros().shape:
            logger.debug("engine_residual_norm_invalid_guess shape=%s", None if X is None else X.shape)
            return float("inf")
        try:
            value = float(full_problem.norms(X, 1.0, False)["coeff_rel"])
            logger.debug("engine_residual_norm value=%s", value)
            return value
        except (ValueError, FloatingPointError):
            logger.debug("engine_residual_norm_failed", exc_info=True)
            return float("inf")

    def solve_point(
        self, point: GridPoint, pass_dir: Path, *, mode: str, warm_X: np.ndarray | None,
        prebuilt: tuple | None = None, force_gain: bool = False,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        a = self.args
        pdir = pass_dir / "points" / point_name(point.index, point.power_dbm, point.pump_freq_ghz)
        pump_dir = pdir / "pump"
        gain_dir = pdir / "gain"
        pdir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "engine_solve_point_start point=%s power_dbm=%s freq_ghz=%s mode=%s "
            "warm_guess=%s force_gain=%s",
            point.index, point.power_dbm, point.pump_freq_ghz, mode,
            warm_X is not None, force_gain,
        )
        t0 = time.perf_counter()
        pump_wall_start = time.perf_counter()

        t_setup = time.perf_counter()
        if prebuilt is not None:
            full_problem, basis, omega, injected = prebuilt
        else:
            full_problem, basis, omega, injected = self.build_problem_for(point)
        pump_setup_runtime_s = time.perf_counter() - t_setup
        logger.debug(
            "engine_solve_point_problem_ready point=%s setup_s=%.6f injected_current_a=%s",
            point.index, pump_setup_runtime_s, injected,
        )
        # Optional Schur-reduced backend: solve on retained nodes, reconstruct
        # the full solution for write_results/exp09 (which need full-node X).
        use_schur = a.inproc_pump_backend == "schur_cpu_mt"
        t_schur = time.perf_counter()
        solve_problem = self._make_solve_problem(full_problem, point.pump_freq_ghz)
        pump_schur_setup_runtime_s = time.perf_counter() - t_schur
        logger.debug(
            "engine_solve_point_backend_ready point=%s use_schur=%s schur_setup_s=%.6f",
            point.index, use_schur, pump_schur_setup_runtime_s,
        )
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())

        # Harmonic enrichment changes the number of rows in the chained pump
        # state.  The next map point is intentionally rebuilt at the cheap base
        # basis before it is enriched again, so project the previous state onto
        # that basis explicitly.  Passing an enriched state directly to the
        # base Schur problem is a shape error, not a physical branch failure.
        if (
            mode == "warm"
            and warm_X is not None
            and hasattr(solve_problem, "zeros")
        ):
            expected_shape = solve_problem.zeros().shape
            if warm_X.shape != expected_shape:
                source_basis = getattr(self, "_last_pump_basis", None)
                if (
                    source_basis is None
                    and int(getattr(a, "mixing_order", 0)) == 3
                    and warm_X.ndim == 2
                    and warm_X.shape[0] >= 2
                ):
                    source_basis = pump_basis.PumpBasis(
                        modes=list(range(warm_X.shape[0])),
                        policy="dense_real", omega_p=omega,
                        source_mode=1,
                    )
                if (
                    source_basis is not None
                    and warm_X.ndim == 2
                    and warm_X.shape[1] == expected_shape[1]
                    and len(source_basis.modes) == warm_X.shape[0]
                ):
                    warm_X = pump_basis.promote_solution_to_basis(
                        warm_X, source_basis, basis
                    )
                    logger.debug(
                        "engine_warm_state_projected point=%s src_modes=%s "
                        "dst_modes=%s shape=%s",
                        point.index, source_basis.modes, basis.modes, warm_X.shape,
                    )
                else:
                    logger.warning(
                        "engine_warm_state_dropped point=%s warm_shape=%s "
                        "expected_shape=%s",
                        point.index, warm_X.shape, expected_shape,
                    )
                    warm_X = None

        t_solve = time.perf_counter()
        continuation_info: dict[str, Any] = {
            "method": "direct" if mode == "warm" and warm_X is not None else None,
            "steps": None,
            "reached_target": None,
            "fold_lambda": None,
            "runtime_s": None,
        }
        if mode == "warm" and warm_X is not None:
            logger.debug("engine_solve_point_continuation_dispatch point=%s method=direct", point.index)
            X, reports = solver.solve_direct(solve_problem, warm_X)
            continuation_info["steps"] = len(reports)
            continuation_info["reached_target"] = bool(
                reports
                and reports[-1].converged
                and abs(reports[-1].source_scale - 1.0) < 1e-12
            )
            continuation_info["runtime_s"] = time.perf_counter() - t_solve
        else:
            cont = getattr(a, "inproc_continuation", "adaptive_secant")
            logger.debug("engine_solve_point_continuation_dispatch point=%s method=%s", point.index, cont)
            continuation_info["method"] = cont
            continuation_start = time.perf_counter()
            X_seed = solve_problem.zeros()
            if cont == "fixed":
                X, reports = solver.solve_continuation(
                    solve_problem,
                    continuation_steps=a.continuation_steps,
                )
                continuation_info["steps"] = len(reports)
                continuation_info["reached_target"] = bool(
                    reports
                    and reports[-1].converged
                    and abs(reports[-1].source_scale - 1.0) < 1e-12
                )
            elif cont == "ptc":
                X, reports = solver.solve_pseudo_transient(solve_problem, X_seed)
                continuation_info["steps"] = (
                    reports[-1].newton_iterations if reports else 0
                )
                continuation_info["reached_target"] = bool(
                    reports and reports[-1].converged
                )
            elif cont == "arclength":
                X_arc, _lam, arc_info = solver.solve_arclength(
                    solve_problem,
                    X_seed,
                    0.0,
                    ds=a.inproc_arclength_ds,
                    max_steps=a.inproc_arclength_max_steps,
                    target_lam=1.0,
                    max_wall_s=a.inproc_solve_deadline_s,
                )
                X, reports = solver.solve_direct(solve_problem, X_arc)
                continuation_info["steps"] = arc_info.get("steps")
                continuation_info["reached_target"] = arc_info.get("reached_target")
                continuation_info["fold_lambda"] = arc_info.get("fold_lambda")
            else:
                predictor = "none" if cont == "adaptive_copy" else cont
                # A zero continuation deadline historically meant unlimited
                # adaptive work. For map fail-fast runs, inherit the per-solve
                # deadline so a failed cold seed cannot spend the whole map on
                # repeated adaptive/fallback attempts.
                continuation_deadline = float(
                    getattr(a, "inproc_continuation_deadline_s", 0.0)
                )
                if continuation_deadline <= 0.0:
                    continuation_deadline = float(a.inproc_solve_deadline_s)
                if cont == "affine":
                    X, reports, trace = solver.solve_affine_continuation(
                        solve_problem,
                        X_seed,
                        initial_step=a.adaptive_initial_step,
                        min_step=a.adaptive_min_step,
                        fallback_fixed_steps=a.inproc_fallback_fixed_steps,
                        max_wall_s=continuation_deadline,
                    )
                else:
                    X, reports, trace = solver.solve_adaptive_continuation(
                        solve_problem,
                        X_seed,
                        initial_step=a.adaptive_initial_step,
                        min_step=a.adaptive_min_step,
                        growth=1.5,
                        shrink=0.5,
                        fallback_fixed_steps=a.inproc_fallback_fixed_steps,
                        max_wall_s=continuation_deadline,
                    )
                continuation_info["method"] = cont
                continuation_info["steps"] = len(trace.attempted_lambdas)
                continuation_info["reached_target"] = bool(
                    reports
                    and reports[-1].converged
                    and abs(reports[-1].source_scale - 1.0) < 1e-12
                )
                if predictor == "none" and cont != "affine":
                    continuation_info["method"] = "adaptive_copy"
            continuation_info["runtime_s"] = time.perf_counter() - continuation_start
            logger.debug(
                "engine_solve_point_continuation_complete point=%s method=%s "
                "reports=%d reached_target=%s runtime_s=%.6f",
                point.index, continuation_info["method"], len(reports),
                continuation_info["reached_target"], continuation_info["runtime_s"],
            )

        pump_solve_wall_runtime_s = time.perf_counter() - t_solve

        converged = bool(reports and reports[-1].converged
                         and abs(reports[-1].source_scale - 1.0) < 1e-12)
        harmonic_info: dict[str, Any] = {
            "enabled": False,
            "initial_modes": list(getattr(basis, "modes", [])),
            "final_modes": list(getattr(basis, "modes", [])),
            "promotions": [],
            "stop_reason": "not_requested",
        }
        high_power_harmonic_retry = bool(
            getattr(a, "high_power_recovery", False)
            and X is not None
            and np.all(np.isfinite(X))
        )
        if (converged and (self.args.mixing_order == 3 or high_power_harmonic_retry)) or high_power_harmonic_retry:
            (
                full_problem,
                basis,
                solve_problem,
                X,
                reports,
                harmonic_info,
            ) = self._adaptive_harmonic_enrichment(
                point,
                injected,
                full_problem,
                basis,
                solve_problem,
                X,
                reports,
                retry_failed=high_power_harmonic_retry and not converged,
            )
            converged = bool(
                reports and reports[-1].converged
                and abs(reports[-1].source_scale - 1.0) < 1e-12
            )
        # A failed low-basis Newton solve can still carry a useful waveform
        # envelope.  In high-power mode the enrichment above promotes that
        # iterate to the basis required by the full residual.  Give PTC one
        # chance on this *same enriched problem* before the outer column
        # recovery ladder falls back to the previous accepted state.  The
        # earlier ladder-level PTC cannot do this because it deliberately
        # operates on the last accepted (base-basis) state.
        enriched_ptc_info: dict[str, Any] = {
            "attempted": False,
            "converged": False,
            "iterations": 0,
            "terminal_reason": None,
            "runtime_s": 0.0,
            "delta0_attempts": [],
        }
        if (
            getattr(a, "high_power_recovery", False)
            and not converged
            and np.all(np.isfinite(X))
            and bool(harmonic_info.get("promotions"))
        ):
            enriched_ptc_info["attempted"] = True
            ptc_t0 = time.perf_counter()
            ptc_settings = dataclasses.replace(
                self._settings(),
                stall_patience=0,
                solve_deadline_s=float(
                    getattr(a, "column_recovery_ptc_deadline_s", 90.0)
                ),
            )
            try:
                # A unit pseudo-timestep is not uniformly safe near a fold.
                # Try progressively less aggressive shifts from the same best
                # iterate, under one shared wall-clock budget.  This is a
                # bounded portfolio, not an unbounded iteration increase.
                delta0s = (1e-3, 1e-2, 1e-1, 1.0)
                best_X = np.array(X, copy=True)
                best_norm = float(
                    solve_problem.norms(best_X, 1.0, False)["coeff_rel"]
                )
                ptc_reports_all: list[Any] = []
                ptc_deadline = float(
                    getattr(a, "column_recovery_ptc_deadline_s", 90.0)
                )
                ptc_max_iter = int(
                    getattr(a, "column_recovery_ptc_max_iter", 128)
                )
                for delta0 in delta0s:
                    elapsed = time.perf_counter() - ptc_t0
                    remaining = ptc_deadline - elapsed
                    if remaining <= 0.0:
                        break
                    ptc_solver = exp08.HarmonicNewtonKrylovSolver(
                        dataclasses.replace(
                            ptc_settings, solve_deadline_s=remaining
                        )
                    )
                    X_ptc, ptc_reports = ptc_solver.solve_pseudo_transient(
                        solve_problem,
                        best_X,
                        delta0=delta0,
                        max_iter=ptc_max_iter,
                    )
                    ptc_reports_all.extend(ptc_reports)
                    trial_norm = float(
                        solve_problem.norms(X_ptc, 1.0, False)["coeff_rel"]
                    )
                    enriched_ptc_info["delta0_attempts"].append({
                        "delta0": delta0,
                        "coeff_rel": trial_norm,
                        "converged": bool(
                            ptc_reports
                            and ptc_reports[-1].converged
                            and abs(ptc_reports[-1].source_scale - 1.0) < 1e-12
                        ),
                    })
                    if trial_norm < best_norm:
                        best_X = np.array(X_ptc, copy=True)
                        best_norm = trial_norm
                    if (
                        ptc_reports
                        and ptc_reports[-1].converged
                        and abs(ptc_reports[-1].source_scale - 1.0) < 1e-12
                    ):
                        X = X_ptc
                        converged = True
                        enriched_ptc_info["terminal_reason"] = None
                        enriched_ptc_info["iterations"] = int(
                            ptc_reports[-1].newton_iterations
                        )
                        break
                else:
                    X = best_X
                if not converged:
                    X = best_X
                    reports = [*reports, *ptc_reports_all]
                    enriched_ptc_info["iterations"] = int(
                        ptc_reports_all[-1].newton_iterations
                        if ptc_reports_all else 0
                    )
                    enriched_ptc_info["terminal_reason"] = (
                        ptc_reports_all[-1].failure_reason
                        if ptc_reports_all else "ptc portfolio exhausted"
                    )
                else:
                    reports = [*reports, *ptc_reports]
                enriched_ptc_info["converged"] = converged
            except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
                enriched_ptc_info["terminal_reason"] = repr(exc)
                logger.debug(
                    "engine_enriched_ptc_failed point=%s", point.index,
                    exc_info=True,
                )
            enriched_ptc_info["runtime_s"] = time.perf_counter() - ptc_t0
            harmonic_info["enriched_ptc"] = enriched_ptc_info
        logger.debug(
            "engine_solve_point_pump_complete point=%s converged=%s reports=%d "
            "last_coeff_rel=%s solve_wall_s=%.6f",
            point.index, converged, len(reports),
            reports[-1].coeff_rel if reports else None, pump_solve_wall_runtime_s,
        )

        # X is retained-sized for the Schur backend; reconstruct full nodes.
        chain_X = X
        X_full = solve_problem.reconstruct_full(X) if use_schur else X
        full_time_rel = None
        full_gate = None
        full_gate_passed = True
        three_wm = int(getattr(a, "mixing_order", 0)) == 3
        configured_gate = getattr(a, "pump_full_residual_gate", None)
        if hasattr(full_problem, "norms"):
            full_time_rel = full_problem.norms(X_full, 1.0, True)["time_rel"]
            high_power_gate = (
                1e-7 if getattr(a, "high_power_recovery", False) else None
            )
            if configured_gate is not None:
                full_gate = float(configured_gate)
            elif high_power_gate is not None:
                full_gate = high_power_gate
            elif three_wm:
                full_gate = float(
                    getattr(a, "harmonic_enrichment_time_rel", 1e-4)
                )
            if full_gate is not None:
                full_gate_passed = bool(
                    full_time_rel is not None
                    and np.isfinite(float(full_time_rel))
                    and float(full_time_rel) <= full_gate
                )
        pump_valid = pump_solution_is_valid(
            converged=bool(converged),
            three_wm=three_wm,
            configured_full_residual_gate=configured_gate,
            full_residual_gate_passed=full_gate_passed,
        )
        validation_failure = None
        if converged and not pump_valid:
            validation_failure = "full harmonic residual gate failed"
        elif not converged:
            validation_failure = reports[-1].failure_reason if reports else "no Newton report"
        dc_flux = np.asarray(
            getattr(self, "dc_branch_flux", np.zeros(0, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        circuit_loss_model = (
            default_loss_model_for(self.ipm08)
            if hasattr(self, "ipm08") else None
        )
        actual_grid = getattr(full_problem, "grid", None)
        actual_nt = int(getattr(actual_grid, "nt", getattr(a, "nt", 0)))
        spectrum_info: dict[str, Any] = {}
        if (
            getattr(a, "pump_record_residual_spectrum", False)
            or getattr(a, "high_power_recovery", False)
        ) and np.all(np.isfinite(X_full)):
            try:
                spectrum_info = residual_spectrum_summary(
                    full_problem, X_full, 1.0
                )
            except (FloatingPointError, ValueError, OverflowError) as exc:
                spectrum_info = {"error": repr(exc)}

        metadata = {
            **basis.to_metadata(),
            "pump_freq_ghz": point.pump_freq_ghz,
            "nt": actual_nt, "omega_p": omega,
            "pump_current_a": injected,
            "pump_power_dbm_requested": point.power_dbm,
            "pump_power_convention": getattr(a, "power_convention", None),
            "attenuation_db": attenuation_db_for(point.pump_freq_ghz, a),
            "dc_branch_flux": dc_flux.tolist(),
            "dc_branch_flux_wb": float(dc_flux[0]) if dc_flux.size else 0.0,
            "pump_port": int(getattr(a, "pump_port", 0)),
            "source_port": int(getattr(a, "source_port", 0)),
            "out_port": int(getattr(a, "out_port", 0)),
            "circuit_dir": str(getattr(a, "circuit_dir", "")),
            "loss_model": circuit_loss_model,
            "harmonic_enrichment": harmonic_info,
            "production_hb_full_residual_rel": (
                None if full_time_rel is None else float(full_time_rel)
            ),
            "production_hb_full_residual_gate": full_gate,
            "production_hb_full_residual_passed": bool(full_gate_passed),
            "pump_full_residual_gate_source": (
                "explicit" if getattr(a, "pump_full_residual_gate", None) is not None
                else "high_power_default"
                if getattr(a, "high_power_recovery", False)
                else "3wm_harmonic_gate"
                if three_wm
                else "disabled"
            ),
            "pump_residual_spectrum": spectrum_info,
            "pump_validation_status": (
                "VALID_CONVERGED" if pump_valid
                else "FAIL_FULL_HARMONIC_RESIDUAL"
                if validation_failure == "full harmonic residual gate failed"
                else "FAIL"
            ),
            "pump_current_ratio_ic_median": injected / self.ic_median,
            "pump_model_switching": "not_modelled_ideal_sine_junction",
            "high_power_recovery": bool(
                getattr(a, "high_power_recovery", False)
            ),
            "pump_backend": a.inproc_pump_backend,
            "sidebands": int(getattr(a, "sidebands", 6)),
            # Production checkpoints are validation inputs.  Preserve enough
            # precision for the HB/TD handoff and independent residual checks.
            "pump_solution_dtype": getattr(a, "pump_solution_dtype", "float64")
            or "float64",
        }
        self._last_pump_basis = basis
        t_write = time.perf_counter()
        summary = exp08.summarize_solution(full_problem, X_full)
        exp08.write_results(pump_dir, X_full, reports, summary, metadata)
        pump_write_runtime_s = time.perf_counter() - t_write
        pump_wall_runtime_s = time.perf_counter() - pump_wall_start
        logger.debug(
            "engine_solve_point_pump_written point=%s path=%s write_s=%.6f wall_s=%.6f",
            point.index, pump_dir, pump_write_runtime_s, pump_wall_runtime_s,
        )

        row: dict[str, Any] = {
            "point_index": point.index, "i_power": point.i_power, "j_freq": point.j_freq,
            "pump_power_dbm": point.power_dbm, "pump_freq_ghz": point.pump_freq_ghz,
            "pump_current_peak_a": point.current_a, "warm_started": mode == "warm",
            "sidebands": int(getattr(a, "sidebands", 6)),
            "single_tone_forced": bool(getattr(a, "force_single_tone", False)),
            "pump_status": "VALID_CONVERGED" if pump_valid else "FAIL",
            "pump_runtime_s": float(sum(r.runtime_s for r in reports)),
            "pump_wall_runtime_s": pump_wall_runtime_s,
            "pump_setup_runtime_s": pump_setup_runtime_s,
            "pump_schur_setup_runtime_s": pump_schur_setup_runtime_s,
            "pump_solve_wall_runtime_s": pump_solve_wall_runtime_s,
            "pump_write_runtime_s": pump_write_runtime_s,
            "pump_factor_runtime_s": float(sum(r.factor_runtime_s for r in reports)),
            "pump_preconditioner_assembly_runtime_s": float(sum(getattr(r, "preconditioner_assembly_runtime_s", 0.0) for r in reports)),
            "pump_preconditioner_numeric_factor_runtime_s": float(sum(getattr(r, "preconditioner_numeric_factor_runtime_s", 0.0) for r in reports)),
            "pump_coeff_rel": float(reports[-1].coeff_rel) if reports else None,
            "pump_time_rel": (
                None if full_time_rel is None else float(full_time_rel)
            ),
            "pump_newton_total": int(sum(r.newton_iterations for r in reports)),
            "pump_gmres_total": int(sum(r.gmres_iterations_total for r in reports)),
            "pump_branch_current_max": finite_or_none(summary.get("branch_i_max_abs")),
            "pump_branch_current_max_over_ic": finite_or_none(
                summary.get("branch_current_max_over_ic")
            ),
            "pump_strongest_branch_index": summary.get("strongest_branch_index"),
            "pump_branch_min_cos_phase": finite_or_none(
                summary.get("branch_min_cos_phase")
            ),
            "pump_boundary_predictor_status": boundary_predictor_status(
                finite_or_none(summary.get("branch_current_max_over_ic")),
                finite_or_none(summary.get("branch_min_cos_phase")),
            ),
            "pump_residual_max_omitted_mode_rel": finite_or_none(
                spectrum_info.get("max_omitted_mode_rel")
            ),
            "pump_dominant_omitted_modes": spectrum_info.get(
                "dominant_omitted_modes"
            ),
            "pump_failure_reason": validation_failure,
            "pump_continuation_method": continuation_info["method"],
            "pump_continuation_steps": continuation_info["steps"],
            "pump_continuation_reached_target": continuation_info["reached_target"],
            "pump_continuation_fold_lambda": continuation_info["fold_lambda"],
            "pump_continuation_runtime_s": continuation_info["runtime_s"],
        }
        row.update({k: None for k in (
            "gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db", "signal_ghz",
            "linear_rel_residual", "gain_total_runtime_s", "gain_wall_runtime_s",
            "gain_gamma_hat_runtime_s", "gain_khat_build_runtime_s",
            "gain_khat_off_runtime_s", "gain_matrix_assemble_runtime_s",
            "gain_factor_solve_runtime_s", "gain_baseline_off_runtime_s",
            "gain_baseline_pumpdiag_runtime_s",
            "spectrum_peak_gain_db", "spectrum_peak_signal_ghz")})
        row["gain_status"] = "ERROR"

        # ``force_gain`` runs the gain solve on the last-iterate pump waveform
        # even when Newton did not converge (above-threshold / fold region), so
        # the diagnostic column resume can see what the gain does past the wall.
        if getattr(a, "force_single_tone", False):
            row["gain_status"] = "SKIPPED_SINGLE_TONE"
            row["single_tone_forced"] = True
            logger.debug(
                "engine_solve_point_gain_skipped point=%s reason=force_single_tone",
                point.index,
            )
        elif pump_valid or force_gain:
            logger.debug(
                "engine_solve_point_gain_dispatch point=%s reason=%s",
                point.index, "converged" if pump_valid else "force_gain",
            )
            g, gain_timing, spectrum = self._gain(pump_dir, gain_dir, point.pump_freq_ghz)
            row.update(gain_timing)
            if spectrum is not None:
                row["_spectrum"] = spectrum  # dropped from CSV; -> map_spectrum.npz
                gains = [gd for gd, st in zip(spectrum["gain_db"], spectrum["status"])
                         if st == "VALID_SOLVED"]
                if gains:
                    k = int(np.nanargmax(spectrum["gain_db"]))
                    row["spectrum_peak_gain_db"] = float(spectrum["gain_db"][k])
                    row["spectrum_peak_signal_ghz"] = float(spectrum["signal_ghz"][k])
            if g is not None and g.status == "VALID_SOLVED":
                row["gain_status"] = "VALID_SOLVED"
                row["gain_db"] = float(g.gain_db)
                row["gain_vs_off_db"] = float(g.gain_vs_off_db)
                row["gain_vs_pumpdiag_db"] = float(g.gain_vs_pumpdiag_db)
                row["signal_ghz"] = float(g.signal_ghz)
                row["linear_rel_residual"] = float(g.linear_rel_residual)
            logger.debug(
                "engine_solve_point_gain_complete point=%s gain_status=%s gain_db=%s",
                point.index, row["gain_status"], row["gain_db"],
            )
        else:
            logger.debug("engine_solve_point_gain_skipped point=%s pump_not_converged", point.index)

        row["signal_attenuation_db"] = signal_attenuation_db_for(
            float(row["signal_ghz"])
            if row.get("signal_ghz") is not None
            else signal_ghz_for(point.pump_freq_ghz, a),
            a,
        )
        row["status"] = "PASS" if (
            row["pump_status"] == "VALID_CONVERGED"
            and (
                row["gain_status"] == "VALID_SOLVED"
                or row["gain_status"] == "SKIPPED_SINGLE_TONE"
            )
        ) else "ERROR"
        row["elapsed_s"] = time.perf_counter() - t0
        row["pump_dir"] = str(pump_dir)
        if getattr(a, "compact_output", False):
            # The continuation state is retained in memory by the column
            # runner.  Once gain has been evaluated, ordinary map points only
            # need scalar telemetry; retaining every full X dominates disk
            # usage and is not required for a fresh restart.
            (pump_dir / "pump_solution.npz").unlink(missing_ok=True)
            row["compact_state_discarded"] = True
        logger.debug(
            "engine_solve_point_end point=%s status=%s pump_status=%s gain_status=%s elapsed_s=%.6f",
            point.index, row["status"], row["pump_status"], row["gain_status"], row["elapsed_s"],
        )
        # In force_gain mode return the last-iterate X regardless of convergence
        # so the caller can keep warm-starting up the column past the wall.
        return row, (X if (pump_valid or force_gain) else None)

    def solve_bridge(
        self, parent_X: np.ndarray, parent_current: float, parent_freq: float,
        target_current: float, target_freq: float, *, steps: int, mode: str,
    ) -> np.ndarray | None:
        """March a warm guess from a solved parent to the target along (P, f).

        Continuation in the physical parameters (not lambda): at each sub-step
        build the pump problem at the interpolated (current, frequency) and take
        one full-scale Newton solve warm-started from the previous sub-state.
        Returns the marched state near the target (a strong warm guess for the
        real target solve) or ``None`` if any sub-step fails.

        ``mode``: ``diagonal`` straight line; ``freq_first`` ramps frequency at
        parent power then power; ``power_first`` the reverse; ``adaptive`` walks
        the diagonal with a halving step on failure.
        """
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())

        def step_to(cur: float, frq: float, guess: np.ndarray) -> np.ndarray | None:
            prob, _basis, _omega = self._build_problem(frq, cur)
            solve_prob = self._make_solve_problem(prob, frq)
            X, report = solver.solve_one(solve_prob, guess, 1.0)
            return X if report.converged else None

        n = max(1, int(steps))
        if mode == "adaptive":
            guess = parent_X
            t, h = 0.0, 1.0 / n
            while t < 1.0 - 1e-9:
                nt = min(1.0, t + h)
                cur = parent_current + nt * (target_current - parent_current)
                frq = parent_freq + nt * (target_freq - parent_freq)
                nxt = step_to(cur, frq, guess)
                if nxt is None:
                    h *= 0.5
                    if h < 1.0 / 64.0:
                        return None
                    continue
                guess, t = nxt, nt
                h = min(1.0 / n, h * 1.5)
            return guess

        # Fixed paths. Build the (fraction-of-current, fraction-of-freq) schedule.
        fracs = [(k + 1) / n for k in range(n)]
        if mode == "freq_first":
            path = [(parent_current, parent_freq + fr * (target_freq - parent_freq)) for fr in fracs]
            path += [(parent_current + fr * (target_current - parent_current), target_freq) for fr in fracs]
        elif mode == "power_first":
            path = [(parent_current + fr * (target_current - parent_current), parent_freq) for fr in fracs]
            path += [(target_current, parent_freq + fr * (target_freq - parent_freq)) for fr in fracs]
        else:  # diagonal
            path = [(parent_current + fr * (target_current - parent_current),
                     parent_freq + fr * (target_freq - parent_freq)) for fr in fracs]

        guess = parent_X
        for cur, frq in path:
            nxt = step_to(cur, frq, guess)
            if nxt is None:
                return None
            guess = nxt
        return guess

    def solve_power_substep(
        self,
        freq_ghz: float,
        from_X: np.ndarray,
        from_current: float,
        to_current: float,
        *,
        init_db: float = 0.1,
        min_db: float = 0.005,
        deadline_s: float = 120.0,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Adaptive natural-parameter continuation along the map power axis.

        Walk the pump current from ``from_current`` (a converged state) to
        ``to_current`` (the failed target cell), warm-starting one full-scale
        Newton solve per micro-step. The step is measured in dBm (geometric in
        current: ``I *= 10**(step_db/20)``) so the schedule matches the physical
        gain-lobe spacing; it grows x1.5 on success and halves on failure. When
        the step must shrink below ``min_db`` the branch has a step-independent
        stall at that power (a numerical/fold boundary), and this returns
        ``None`` -- distinct from a coarse-grid miss, which recovers here.

        The returned ``X`` (retained-shape, like every chained warm state) is a
        strong guess for the real target solve, not the written solution: the
        caller re-runs ``solve_point`` from it so gain + files are produced by
        the normal path. Bounded by ``deadline_s`` wall time.
        """
        info: dict[str, Any] = {
            "reached_target": False, "substeps": 0, "min_step_db": init_db,
            "terminal_reason": "", "last_current": from_current,
        }
        if to_current <= from_current or from_X is None:
            info["terminal_reason"] = "noop"
            return None, info
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())
        # dBm distance is +20*log10(I2/I1); step geometrically in current.
        total_db = 20.0 * math.log10(to_current / from_current)
        t0 = time.perf_counter()
        guess = from_X
        cur = from_current
        done_db = 0.0            # dBm advanced from from_current
        step_db = min(init_db, total_db)
        while done_db < total_db - 1e-9:
            if time.perf_counter() - t0 > deadline_s:
                info["terminal_reason"] = "deadline"
                break
            trial_db = min(done_db + step_db, total_db)
            trial_cur = from_current * (10.0 ** (trial_db / 20.0))
            prob, _basis, _omega = self._build_problem(freq_ghz, trial_cur)
            solve_prob = self._make_solve_problem(prob, freq_ghz)
            X, report = solver.solve_one(solve_prob, guess, 1.0)
            info["substeps"] += 1
            if report.converged:
                guess, cur, done_db = X, trial_cur, trial_db
                info["last_current"] = cur
                step_db = min(init_db, step_db * 1.5)
            else:
                step_db *= 0.5
                info["min_step_db"] = min(info["min_step_db"], step_db)
                if step_db < min_db:
                    info["terminal_reason"] = "step_floor"
                    break
        if done_db >= total_db - 1e-9:
            info["reached_target"] = True
            info["terminal_reason"] = "reached"
            return guess, info
        return None, info

    def solve_arclength_forward(
        self,
        freq_ghz: float,
        from_X: np.ndarray,
        from_current: float,
        to_current: float,
        *,
        ds: float = 0.01,
        max_steps: int = 150,
        max_steps_after_fold: int | None = 30,
        deadline_s: float = 60.0,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Local pseudo-arclength continuation from one converged state.

        Milestone G0 (fold_plan.md) recovery tier 3: unlike ``solve_bridge``
        (physical-parameter Newton march, no fold awareness) and unlike the
        cold ``solve_arclength`` continuation option in ``solve_point``
        (starts from ``X=0``, ``mu=0`` every time), this starts the bordered
        corrector directly at the last converged point (``from_X``,
        ``mu0=from_current/to_current``) and marches to ``target_mu=1.0``
        (``i_ref=to_current``, so ``k=1`` -- ``solve_arclength_mu`` reduces
        to ``solve_arclength`` exactly). ``solve_arclength_mu`` already stops
        as soon as ``target_mu`` is reached, so this needs no separate
        bracket+Newton step -- "terminate immediately once the target is
        reached" is the underlying corrector's own behavior. Bounded
        ``max_steps``/``max_steps_after_fold``/``deadline_s`` keep this a
        LOCAL recovery attempt, not a full branch trace.
        """
        full_problem, _basis, _omega = self._build_problem(freq_ghz, to_current)
        solve_problem = self._make_solve_problem(full_problem, freq_ghz)
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())
        mu0 = from_current / to_current
        X_final, mu_final, info = solver.solve_arclength_mu(
            solve_problem, from_X, mu0, i_ref=to_current, target_mu=1.0,
            ds=ds, max_steps=max_steps, max_wall_s=deadline_s,
            rescale_every=5, max_steps_after_fold=max_steps_after_fold,
            step_control="adaptive", refine_fold=True,
        )
        info["mu_final"] = mu_final
        if info.get("reached_target"):
            return X_final, info
        return None, info

    def solve_pseudo_transient_recovery(
        self,
        freq_ghz: float,
        current_a: float,
        from_X: np.ndarray,
        *,
        deadline_s: float = 90.0,
        max_iter: int = 128,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Use pseudo-transient continuation as a bounded branch bridge.

        This is a numerical recovery method for a fixed-frequency target.  It
        integrates the algebraically regularised HB residual and then validates
        the returned state through the ordinary target solve.  It is intentionally
        not used as evidence of a physical boundary or as the default map path.
        """
        info: dict[str, Any] = {
            "reached_target": False,
            "terminal_reason": "",
            "iterations": 0,
            "runtime_s": 0.0,
        }
        if from_X is None:
            info["terminal_reason"] = "no_seed"
            return None, info
        started = time.perf_counter()
        full_problem, _basis, _omega = self._build_problem(freq_ghz, current_a)
        solve_problem = self._make_solve_problem(full_problem, freq_ghz)
        settings = dataclasses.replace(
            self._settings(),
            stall_patience=0,
            solve_deadline_s=float(deadline_s),
        )
        solver = exp08.HarmonicNewtonKrylovSolver(settings)
        try:
            X, reports = solver.solve_pseudo_transient(
                solve_problem,
                from_X,
                delta0=1.0,
                max_iter=int(max_iter),
            )
        except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
            info["terminal_reason"] = "exception"
            info["error"] = repr(exc)
            info["runtime_s"] = time.perf_counter() - started
            return None, info
        info["iterations"] = int(sum(r.newton_iterations for r in reports))
        info["last_coeff_rel"] = (
            float(reports[-1].coeff_rel) if reports else None
        )
        info["terminal_reason"] = (
            "reached" if reports and reports[-1].converged else
            reports[-1].failure_reason if reports else "no_report"
        )
        info["runtime_s"] = time.perf_counter() - started
        if reports and reports[-1].converged:
            info["reached_target"] = True
            return X, info
        return None, info

    def solve_frequency_substep(
        self,
        from_freq_ghz: float,
        to_freq_ghz: float,
        current_a: float,
        from_X: np.ndarray,
        *,
        init_step_ghz: float = 0.01,
        min_step_ghz: float = 0.0005,
        deadline_s: float = 60.0,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Adaptive natural-parameter continuation along frequency, fixed power.

        Milestone G0 recovery tier 4: mirrors ``solve_power_substep`` exactly
        (warm-started full-scale Newton per micro-step, grows x1.5 on
        success, halves on failure, ``step_floor`` when it must shrink below
        ``min_step_ghz``) but steps ``freq_ghz`` linearly at fixed
        ``current_a`` instead of stepping current geometrically at fixed
        frequency -- the two axes of the map are physically different
        (dispersion vs. drive amplitude), so frequency steps additively
        rather than in dB.
        """
        info: dict[str, Any] = {
            "reached_target": False, "substeps": 0, "min_step_ghz": init_step_ghz,
            "terminal_reason": "", "last_freq_ghz": from_freq_ghz,
        }
        if from_X is None or from_freq_ghz == to_freq_ghz:
            info["terminal_reason"] = "noop"
            return None, info
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())
        direction = 1.0 if to_freq_ghz > from_freq_ghz else -1.0
        total = abs(to_freq_ghz - from_freq_ghz)
        t0 = time.perf_counter()
        guess = from_X
        done = 0.0
        step = min(init_step_ghz, total)
        while done < total - 1e-12:
            if time.perf_counter() - t0 > deadline_s:
                info["terminal_reason"] = "deadline"
                break
            trial_done = min(done + step, total)
            trial_freq = from_freq_ghz + direction * trial_done
            prob, _basis, _omega = self._build_problem(trial_freq, current_a)
            solve_prob = self._make_solve_problem(prob, trial_freq)
            X, report = solver.solve_one(solve_prob, guess, 1.0)
            info["substeps"] += 1
            if report.converged:
                guess, done = X, trial_done
                info["last_freq_ghz"] = from_freq_ghz + direction * done
                step = min(init_step_ghz, step * 1.5)
            else:
                step *= 0.5
                info["min_step_ghz"] = min(info["min_step_ghz"], step)
                if step < min_step_ghz:
                    info["terminal_reason"] = "step_floor"
                    break
        if done >= total - 1e-12:
            info["reached_target"] = True
            info["terminal_reason"] = "reached"
            return guess, info
        return None, info

    def trace_column_arclength(
        self,
        freq_ghz: float,
        reference_current: float,
        X0: np.ndarray,
        current0: float,
        X1: np.ndarray,
        current1: float,
        targets: list[tuple[int, float]],
    ) -> tuple[dict[int, list[np.ndarray]], dict]:
        """Trace once from two map states and interpolate target-current crossings."""
        full_problem, _basis, _omega = self._build_problem(freq_ghz, reference_current)
        problem = self._make_solve_problem(full_problem, freq_ghz)
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())
        points, info = solver.trace_arclength_from_two_points(
            problem,
            X0,
            current0 / reference_current,
            X1,
            current1 / reference_current,
            ds=self.args.column_arclength_ds,
            max_steps=self.args.column_arclength_max_steps,
            max_wall_s=self.args.column_arclength_deadline_s,
        )
        guesses: dict[int, list[np.ndarray]] = {}
        for point_index, target_current in targets:
            target = target_current / reference_current
            for (Xa, la), (Xb, lb) in zip(points, points[1:]):
                if lb == la or (la - target) * (lb - target) > 0.0:
                    continue
                theta = (target - la) / (lb - la)
                if -1e-12 <= theta <= 1.0 + 1e-12:
                    guesses.setdefault(point_index, []).append(Xa + theta * (Xb - Xa))
        info["trace_points"] = len(points)
        info["target_crossings"] = sum(len(v) for v in guesses.values())
        return guesses, info

    def _gain(self, pump_dir: Path, gain_dir: Path, freq_ghz: float):
        a = self.args
        logger.debug(
            "gain_start pump_dir=%s gain_dir=%s pump_freq_ghz=%s sidebands=%s "
            "spectrum=%s signal_backend=%s signal_solver=%s",
            pump_dir, gain_dir, freq_ghz, a.sidebands, a.signal_spectrum,
            a.signal_backend, a.signal_solver,
        )
        gain_dir.mkdir(parents=True, exist_ok=True)
        t_all = time.perf_counter()
        pump = exp09.load_pump(pump_dir, fallback_pump_freq_ghz=freq_ghz)
        ms = exp09.sideband_list(a.sidebands)
        max_ell = max(abs(m - q) for m in ms for q in ms)
        logger.debug("gain_pump_loaded omega_p=%s modes=%r max_ell=%d", pump.omega_p, pump.modes, max_ell)
        t0 = time.perf_counter()
        gamma_hat = exp09.compute_gamma_hat(
            circuit=self.ipm09, pump=pump, max_ell=max_ell, gamma_nt=a.gamma_nt,
            dc_branch_flux=self.dc_branch_flux,
        )
        gamma_runtime_s = time.perf_counter() - t0
        logger.debug("gain_gamma_hat_complete n_coeffs=%d runtime_s=%.6f", len(gamma_hat), gamma_runtime_s)
        t0 = time.perf_counter()
        khat = exp09.build_khat(Bphi=self.ipm09.Bphi, gamma_hat=gamma_hat, drop_tol=0.0)
        khat_runtime_s = time.perf_counter() - t0
        logger.debug("gain_khat_complete n_blocks=%d runtime_s=%.6f", len(khat), khat_runtime_s)
        t0 = time.perf_counter()
        gamma_off = self.branch.tangent(self.dc_branch_flux[None, :])[0]
        khat_off_0 = (
            self.ipm09.Bphi @ sp.diags(gamma_off, offsets=0, format="csr") @ self.ipm09.Bphi.T
        ).astype(np.complex128).tocsr()
        khat_off_runtime_s = time.perf_counter() - t0
        logger.debug("gain_khat_off_complete nnz=%d runtime_s=%.6f", khat_off_0.nnz, khat_off_runtime_s)

        # Signal-frequency-independent Floquet conversion base: built once here,
        # reused by the trailing solve and every spectrum point (the dominant
        # speedup for multi-signal cells).
        khat_big_base = None
        khat_base_runtime_s = 0.0
        if a.signal_spectrum:
            t0 = time.perf_counter()
            khat_big_base = exp09.assemble_khat_conversion_base(self.ipm09, khat, ms)
            khat_base_runtime_s = time.perf_counter() - t0
            logger.debug(
                "gain_conversion_base_complete shape=%s nnz=%d runtime_s=%.6f",
                khat_big_base.shape, khat_big_base.nnz, khat_base_runtime_s,
            )

        target_signal_ghz = signal_ghz_for(freq_ghz, a)
        logger.debug("gain_target_signal_selected signal_ghz=%s", target_signal_ghz)
        g = None
        spectrum = None
        if a.signal_spectrum:
            offs = spectrum_offsets_mhz(a)
            logger.debug("gain_spectrum_dispatch n_offsets=%d workers=%d", len(offs), a.signal_workers)

            def one(off: float) -> tuple[float, float, Any]:
                fs = float(freq_ghz) + off / 1000.0
                gg = self._solve_signal(khat, khat_off_0, khat_big_base,
                                        pump.omega_p, fs)
                return off, fs, gg

            if a.signal_workers > 1:
                with ThreadPoolExecutor(max_workers=a.signal_workers) as pool:
                    items = list(pool.map(one, offs))
            else:
                items = [one(off) for off in offs]
            spectrum = {
                "offsets_mhz": [it[0] for it in items],
                "signal_ghz": [it[1] for it in items],
                "gain_db": [float(it[2].gain_db) for it in items],
                "status": [it[2].status for it in items],
            }
            logger.debug(
                "gain_spectrum_complete n_results=%d valid=%d",
                len(items), sum(item[2].status == "VALID_SOLVED" for item in items),
            )
            for _, fs, gg in items:
                if abs(float(fs) - target_signal_ghz) <= 1e-9:
                    g = gg
                    break

        if g is None:
            logger.debug("gain_trailing_dispatch signal_ghz=%s", target_signal_ghz)
            g = self._solve_signal(
                khat, khat_off_0, khat_big_base, pump.omega_p, target_signal_ghz
            )

        timing = {
            "gain_wall_runtime_s": time.perf_counter() - t_all,
            "gain_total_runtime_s": time.perf_counter() - t_all,
            "gain_gamma_hat_runtime_s": gamma_runtime_s,
            "gain_khat_build_runtime_s": khat_runtime_s + khat_base_runtime_s,
            "gain_khat_off_runtime_s": khat_off_runtime_s,
            "gain_matrix_assemble_runtime_s": float(g.assemble_runtime_s),
            "gain_factor_solve_runtime_s": float(g.factor_solve_runtime_s),
            "gain_baseline_off_runtime_s": float(g.baseline_off_runtime_s),
            "gain_baseline_pumpdiag_runtime_s": float(g.baseline_pumpdiag_runtime_s),
        }
        logger.debug(
            "gain_complete status=%s gain_db=%s assemble_s=%.6f solve_s=%.6f wall_s=%.6f",
            g.status, g.gain_db, g.assemble_runtime_s, g.factor_solve_runtime_s,
            timing["gain_wall_runtime_s"],
        )
        return g, timing, spectrum

    def _solve_signal(self, khat, khat_off_0, khat_big_base, omega_p, signal_ghz):
        a = self.args
        logger.debug(
            "signal_solve_start backend=%s solver=%s omega_p=%s signal_ghz=%s "
            "sidebands=%s",
            a.signal_backend, a.signal_solver, omega_p, signal_ghz, a.sidebands,
        )
        schur_part = None
        if a.signal_backend == "schur":
            key = (
                round(float(omega_p), 3),
                round(float(signal_ghz), 12),
                int(a.sidebands),
                int(self.source_idx),
                int(self.out_idx),
                a.loss_model,
            )
            schur_part = self._signal_schur_part_cache.get(key)
            logger.debug("signal_schur_cache_lookup hit=%s key=%r", schur_part is not None, key)
            if schur_part is None:
                schur_part = exp09.build_signal_schur_partition(
                    self.ipm09, omega_p, signal_ghz, a.sidebands,
                    self.source_idx, self.out_idx,
                    loss_model=a.loss_model,
                )
                self._signal_schur_part_cache[key] = schur_part
                if len(self._signal_schur_part_cache) > self._signal_schur_cache_max:
                    evicted = next(iter(self._signal_schur_part_cache))
                    self._signal_schur_part_cache.pop(evicted)
                    logger.debug("signal_schur_cache_evict key=%r", evicted)
        common = dict(
            circuit=self.ipm09, khat=khat, khat_off_0=khat_off_0,
            khat_big_base=khat_big_base, omega_p=omega_p, signal_ghz=signal_ghz,
            sidebands=a.sidebands, signal_m=0,
            idler_m=-(a.mixing_order - 1),
            source_index=self.source_idx, out_index=self.out_idx,
            source_current_a=1.0, source_port=a.source_port, out_port=a.out_port,
            z0_ohm=a.z0_ohm, loss_model=a.loss_model,
            linear_solver=a.signal_solver,
        )
        if a.signal_backend == "schur":
            result = exp09.solve_gain_one_schur(
                **common, include_baselines=not a.skip_baselines,
                schur_part=schur_part)
        else:
            result = exp09.solve_gain_one(**common)
        logger.debug(
            "signal_solve_complete backend=%s status=%s gain_db=%s residual=%s "
            "assemble_s=%.6f solve_s=%.6f",
            a.signal_backend, result.status, result.gain_db,
            result.linear_rel_residual, result.assemble_runtime_s,
            result.factor_solve_runtime_s,
        )
        return result


def run_cold_pass_inprocess(
    points: list[GridPoint], pass_dir: Path, engine: InProcessEngine
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(points)
    logger.debug("run_cold_pass_inprocess_start n_points=%d pass_dir=%s", total, pass_dir)
    for point in points:
        logger.debug("run_cold_pass_inprocess_point_enter index=%d", point.index)
        row, _ = engine.solve_point(point, pass_dir, mode="cold", warm_X=None)
        rows.append(row)
        logger.debug("run_cold_pass_inprocess_point_exit index=%d status=%s", point.index, row["status"])
        print(f"[cold {point.index + 1}/{total}] P={point.power_dbm:.4g} dBm "
              f"fp={point.pump_freq_ghz:.4g} GHz status={row['status']} "
              f"gain={row.get('gain_db')} pump_s={row.get('pump_runtime_s'):.3f}", flush=True)
    clear_cache = getattr(engine, "clear_schur_cache", None)
    if callable(clear_cache):
        clear_cache()
    logger.debug("run_cold_pass_inprocess_complete n_rows=%d", len(rows))
    return rows


def secant_guess(
    x_prevprev: np.ndarray, x_prev: np.ndarray,
    cur_prevprev: float, cur_prev: float, cur: float,
) -> np.ndarray:
    """Linear extrapolation of the pump state along the pump-current axis.

    Given the last two converged solutions ``x_prevprev`` (at ``cur_prevprev``)
    and ``x_prev`` (at ``cur_prev``), predict the solution at ``cur``:

        X_guess = x_prev + beta * (x_prev - x_prevprev),
        beta    = (cur - cur_prev) / (cur_prev - cur_prevprev).

    The current amplitude is the natural continuation parameter (the source term
    is linear in it). Only the initial guess changes -- physics is untouched.
    """
    denom = cur_prev - cur_prevprev
    if abs(denom) < 1e-30:
        logger.debug("secant_guess_degenerate denominator=%s", denom)
        return x_prev
    beta = (cur - cur_prev) / denom
    guess = x_prev + beta * (x_prev - x_prevprev)
    logger.debug(
        "secant_guess_complete cur_prevprev=%s cur_prev=%s cur=%s beta=%s shape=%s",
        cur_prevprev, cur_prev, cur, beta, guess.shape,
    )
    return guess


_COLUMN_TIER4_ANCHOR_OFFSETS_GHZ = (0.005, -0.005, 0.01, -0.01, 0.02, -0.02)


def _merge_column_recovery_result(
    failed_row: dict[str, Any],
    recovered_row: dict[str, Any],
    telemetry: dict[str, Any],
    route: str,
) -> dict[str, Any]:
    """Keep failed-attempt telemetry while replacing it with a valid result."""
    merged = {
        **recovered_row,
        **{key: value for key, value in failed_row.items() if key not in recovered_row},
    }
    merged.update(telemetry)
    merged["column_recovery_route"] = route
    return merged


def _column_frequency_detour_guess(
    engine: InProcessEngine,
    point: GridPoint,
    last_good_X: np.ndarray,
    target_current: float,
    *,
    anchor_deadline_s: float,
    substep_deadline_s: float,
    min_step_ghz: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Find a same-power nearby-frequency anchor and return a target guess."""
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    started = time.perf_counter()
    anchor: tuple[float, np.ndarray] | None = None
    anchor_attempts: list[dict[str, Any]] = []
    for offset in _COLUMN_TIER4_ANCHOR_OFFSETS_GHZ:
        if time.perf_counter() - started > anchor_deadline_s:
            break
        anchor_freq = point.pump_freq_ghz + offset
        problem, _basis, _omega = engine._build_problem(anchor_freq, target_current)
        solve_problem = engine._make_solve_problem(problem, anchor_freq)
        try:
            candidate, report = solver.solve_one(solve_problem, last_good_X, 1.0)
        except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
            anchor_attempts.append({"offset_ghz": offset, "error": str(exc)})
            continue
        anchor_attempts.append(
            {"offset_ghz": offset, "converged": bool(report.converged)}
        )
        if report.converged:
            anchor = (anchor_freq, candidate)
            break

    telemetry: dict[str, Any] = {
        "tier4_anchor_found": anchor is not None,
        "tier4_anchor_attempts": anchor_attempts,
        "tier4_anchor_runtime_s": time.perf_counter() - started,
    }
    if anchor is None:
        return None, telemetry

    anchor_freq, anchor_X = anchor
    guess, info = engine.solve_frequency_substep(
        anchor_freq,
        point.pump_freq_ghz,
        target_current,
        anchor_X,
        min_step_ghz=min_step_ghz,
        deadline_s=substep_deadline_s,
    )
    telemetry.update(
        {
            "tier4_anchor_freq_ghz": anchor_freq,
            "tier4_substeps": info.get("substeps"),
            "tier4_terminal_reason": info.get("terminal_reason"),
            "tier4_substep_runtime_s": info.get("runtime_s"),
        }
    )
    return guess, telemetry


SKIP_PAST_FOLD = "SKIP_PAST_FOLD"

# Row fields that a solved point fills but a skipped one leaves empty.
_SKIP_NONE_FIELDS = (
    "gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db", "signal_ghz",
    "linear_rel_residual", "gain_total_runtime_s", "gain_wall_runtime_s",
    "gain_gamma_hat_runtime_s", "gain_khat_build_runtime_s",
    "gain_khat_off_runtime_s", "gain_matrix_assemble_runtime_s",
    "gain_factor_solve_runtime_s", "gain_baseline_off_runtime_s",
    "gain_baseline_pumpdiag_runtime_s", "pump_runtime_s", "pump_wall_runtime_s",
    "pump_setup_runtime_s", "pump_schur_setup_runtime_s",
    "pump_solve_wall_runtime_s", "pump_write_runtime_s", "pump_factor_runtime_s",
    "pump_preconditioner_assembly_runtime_s",
    "pump_preconditioner_numeric_factor_runtime_s", "pump_coeff_rel",
    "pump_time_rel", "pump_newton_total", "pump_gmres_total",
    "pump_branch_current_max",
)


def past_fold_skip_row(point: GridPoint) -> dict[str, Any]:
    """Synthetic row for a cell skipped by the per-column fold short-circuit.

    Once a frequency column fails to reach full drive at some pump power, every
    higher-power cell is past the harmonic-balance fold -- a turning point with
    no re-convergence above it -- so it is marked past-fold without solving.
    Gain is NaN (a map hole), matching a genuine over-fold failure, and the row
    costs no solver time.
    """
    row: dict[str, Any] = {
        "point_index": point.index, "i_power": point.i_power,
        "j_freq": point.j_freq, "pump_power_dbm": point.power_dbm,
        "pump_freq_ghz": point.pump_freq_ghz,
        "pump_current_peak_a": point.current_a, "warm_started": False,
        "pump_status": SKIP_PAST_FOLD, "gain_status": SKIP_PAST_FOLD,
        "status": SKIP_PAST_FOLD, "warm_retry_reseed": False,
        "pump_predictor": "skip", "elapsed_s": 0.0, "pump_dir": "",
        "pump_failure_reason": "skipped after consecutive pump failures in column",
    }
    row.update({k: None for k in _SKIP_NONE_FIELDS})
    return row


def continuation_failure_is_fold_evidence(row: dict[str, Any]) -> bool:
    """Recognize a failed seed that explored continuation far enough to count.

    A first-step Newton failure is ambiguous and may just be a poor seed. Once
    adaptive continuation has attempted multiple lambda values but still did
    not reach full drive, repeated fail-fast failures are stronger local
    evidence that the column crossed the accessible harmonic-balance branch.
    """
    if row.get("status") == "PASS":
        return False
    method = row.get("pump_continuation_method")
    steps = row.get("pump_continuation_steps")
    return (
        method in {"adaptive_secant", "adaptive_copy", "adaptive_tangent", "affine"}
        and row.get("pump_continuation_reached_target") is False
        and isinstance(steps, (int, float))
        and int(steps) >= 2
    )


def run_warm_pass_inprocess(
    points: list[GridPoint], pass_dir: Path, engine: InProcessEngine,
    *, fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Warm-start each frequency column in increasing power.

    fail_fast: do not pay the reseed/adaptive-fallback recovery on a failed
    point, and keep warm-starting subsequent points from the last *converged*
    neighbour. Within a column the harmonic-balance fold is a turning point (no
    re-convergence above it), so this leaves the convergent points unchanged
    while letting over-fold points fail in ~one stalled solve instead of
    thrashing through the full recovery chain.
    """
    by_col: dict[int, list[GridPoint]] = {}
    for point in points:
        by_col.setdefault(point.j_freq, []).append(point)
    rows: list[dict[str, Any]] = []
    total = len(points)
    done = 0
    predictor = getattr(engine.args, "inproc_fold_predictor", "none")
    recovery_ladder = bool(getattr(engine.args, "column_recovery_ladder", False))
    exhaustive_high_power = bool(
        getattr(engine.args, "high_power_recovery", False)
    )
    logger.debug(
        "run_warm_pass_inprocess_start n_points=%d columns=%d predictor=%s "
        "fail_fast=%s patience=%d recovery_ladder=%s",
        len(points), len(by_col), predictor, fail_fast,
        int(getattr(engine.args, "fold_skip_patience", 0)),
        recovery_ladder,
    )
    scale = engine.args.pump_current_jc_scale
    patience = int(getattr(engine.args, "fold_skip_patience", 0))
    initial_seed = getattr(engine.args, "initial_pump_dir", None)
    if initial_seed and len(by_col) != 1:
        raise ValueError(
            "--initial-pump-dir currently supports exactly one frequency column"
        )
    for j in sorted(by_col):
        column = sorted(by_col[j], key=lambda p: p.power_dbm)
        logger.debug("warm_column_start j_freq=%d n_points=%d", j, len(column))
        prev_X: np.ndarray | None = None
        # Last two converged (injected_current, X) for the secant predictor.
        last_good_X: np.ndarray | None = None
        last_good_cur: float | None = None
        prevprev_X: np.ndarray | None = None
        prevprev_cur: float | None = None
        arclength_guesses: dict[int, list[np.ndarray]] = {}
        candidate_boundary_dbm: float | None = None
        verified_fold = False
        consec_fail = 0  # consecutive non-converged points at increasing power
        if initial_seed and column:
            seed_path = Path(initial_seed)
            try:
                loaded_X, loaded_basis = pump_basis.load_pump_basis_from_solution(
                    seed_path
                )
                seed_power = getattr(engine.args, "initial_pump_power_dbm", None)
                if seed_power is None:
                    raise ValueError(
                        "--initial-pump-power-dbm is required with "
                        "--initial-pump-dir"
                    )
                last_good_cur = dbm_to_peak_current_a(
                    float(seed_power),
                    attenuation_db=attenuation_db_for(
                        column[0].pump_freq_ghz, engine.args
                    ),
                    z0_ohm=engine.args.z0_ohm,
                    convention=engine.args.power_convention,
                ) * scale
                seed_full, seed_basis, _seed_omega = engine._build_problem(
                    column[0].pump_freq_ghz, last_good_cur
                )
                if loaded_basis.modes != seed_basis.modes:
                    raise ValueError(
                        "pump mode metadata does not match the current production basis: "
                        f"checkpoint={loaded_basis.modes} current={seed_basis.modes}"
                    )
                seed_problem = engine._make_solve_problem(
                    seed_full, column[0].pump_freq_ghz
                )
                expected_shape = seed_problem.zeros().shape
                if loaded_X.shape == expected_shape:
                    seed_state = loaded_X
                elif (
                    hasattr(seed_problem, "part")
                    and loaded_X.shape == (seed_full.H, seed_full.n)
                ):
                    seed_state = restrict(loaded_X, seed_problem.part)
                else:
                    raise ValueError(
                        "checkpoint state shape does not match the current full or "
                        f"Schur pump problem: checkpoint={loaded_X.shape} "
                        f"full={(seed_full.H, seed_full.n)} reduced={expected_shape}"
                    )
                seed_full_state = (
                    seed_problem.reconstruct_full(seed_state)
                    if hasattr(seed_problem, "reconstruct_full")
                    else seed_state
                )
                seed_norms = seed_full.norms(seed_full_state, 1.0, True)
                if (
                    not np.isfinite(float(seed_norms["coeff_rel"]))
                    or float(seed_norms["coeff_rel"]) > 1e-8
                ):
                    raise ValueError(
                        "checkpoint fails the current production residual gate: "
                        f"coeff_rel={seed_norms['coeff_rel']!r}"
                    )
                prev_X = seed_state
                last_good_X = seed_state
                print(
                    f"[warm] fp={column[0].pump_freq_ghz:.6g} GHz "
                    f"initial seed={seed_path} validated "
                    f"coeff_rel={float(seed_norms['coeff_rel']):.3e}",
                    flush=True,
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"invalid --initial-pump-dir {seed_path}: {exc}"
                ) from exc
        for idx, point in enumerate(column):
            cur = point.current_a * scale
            base_X = prev_X if not fail_fast else last_good_X
            if idx == 0 and last_good_cur is None:
                # The supplied seed is at the first target power. It is a
                # direct Newton initial guess, not a secant history point.
                base_X = last_good_X
            mode = "warm" if base_X is not None else "seed"

            # Predict the guess from the last two converged states. base_X is the
            # most recent converged solution (at last_good_cur) whenever it is set.
            use_secant = (
                predictor == "secant" and base_X is not None
                and prevprev_X is not None and prevprev_cur is not None
                and last_good_cur is not None and prevprev_X.shape == base_X.shape
            )
            guess = (secant_guess(prevprev_X, base_X, prevprev_cur, last_good_cur, cur)
                     if use_secant else base_X)
            pred_tag = "secant" if use_secant else "none"
            logger.debug(
                "warm_point_guess index=%d column_index=%d mode=%s predictor=%s "
                "guess_shape=%s current_a=%s",
                point.index, idx, mode, pred_tag, None if guess is None else guess.shape, cur,
            )

            row, X = engine.solve_point(point, pass_dir, mode=mode, warm_X=guess)
            logger.debug("warm_point_primary_result index=%d status=%s", point.index, row["status"])

            # Traced fresh on every failing cell that has a valid seed pair --
            # no per-column "once ever" lock, so one cell's crossing-less
            # trace never permanently strands later cells. Each attempt is
            # bounded by --inproc-solve-deadline-s, so retrying stays cheap.
            if (
                row["status"] != "PASS"
                and getattr(engine.args, "column_arclength_recovery", False)
                and prevprev_X is not None
                and prevprev_cur is not None
                and last_good_X is not None
                and last_good_cur is not None
            ):
                reference_current = column[-1].current_a * scale
                targets = [
                    (p.index, p.current_a * scale)
                    for p in column[idx:]
                ]
                arclength_guesses, arc_info = engine.trace_column_arclength(
                    point.pump_freq_ghz,
                    reference_current,
                    prevprev_X,
                    prevprev_cur,
                    last_good_X,
                    last_good_cur,
                    targets,
                )
                logger.debug("warm_point_arclength_trace index=%d info=%r", point.index, arc_info)
                verified_fold = verified_fold or bool(arc_info.get("fold_lambdas"))
                if arc_info.get("fold_lambdas"):
                    row["pump_column_arclength_fold_lambda"] = float(
                        arc_info["fold_lambdas"][0]
                    )
                    row["pump_column_arclength_terminal_reason"] = arc_info.get(
                        "terminal_reason"
                    )
                print(
                    f"[arclength] fp={point.pump_freq_ghz:.6g} GHz "
                    f"steps={arc_info.get('steps')} points={arc_info.get('trace_points')} "
                    f"folds={arc_info.get('fold_lambdas')} "
                    f"crossings={arc_info.get('target_crossings')} "
                    f"reason={arc_info.get('terminal_reason')}",
                    flush=True,
                )

            if row["status"] != "PASS" and point.index in arclength_guesses:
                # Cap to the first 2 target-crossing guesses: each is a full
                # Newton solve, and a stiff branch can have several crossings.
                for arc_guess in arclength_guesses[point.index][:2]:
                    logger.debug("warm_point_arclength_retry index=%d", point.index)
                    arc_row, arc_X = engine.solve_point(
                        point, pass_dir, mode="warm", warm_X=arc_guess,
                    )
                    if arc_row["status"] == "PASS":
                        row, X = arc_row, arc_X
                        pred_tag = f"{pred_tag}->arclength"
                        break

            # Overshoot guard: a bad extrapolation past the fold -> retry once
            # from the plain warm start before paying the reseed. Fail-fast
            # mode intentionally pays only one solve per cell.
            if row["status"] != "PASS" and use_secant and not fail_fast:
                logger.debug("warm_point_secant_fallback index=%d", point.index)
                row, X = engine.solve_point(point, pass_dir, mode="warm", warm_X=base_X)
                pred_tag = "secant_fallback"

            retried = False
            if (
                row["status"] != "PASS"
                and mode == "warm"
                and not fail_fast
                and not recovery_ladder
            ):
                logger.debug("warm_point_reseed_retry index=%d", point.index)
                row, X = engine.solve_point(point, pass_dir, mode="seed", warm_X=None)
                retried = row["status"] == "PASS"
                # A failing reseed can pay for 60-120+ Newton/PARDISO refactors
                # (full adaptive-then-fixed continuation ladder from scratch).
                # Collecting promptly here keeps that churn from fragmenting
                # the process heap and slowing down every later point in this
                # long-lived worker -- measured 4.00x slower PARDISO factor
                # time on later, otherwise-identical converged points without
                # this (same mechanism as the arclength recovery leak).
                gc.collect()

            if recovery_ladder and row["status"] != "PASS":
                recovery_started = time.perf_counter()
                # A failed Newton solve is never a physical boundary.  The
                # ordinary recovery ladder keeps its historical bounded
                # connected-branch diagnostic, while the explicit high-power
                # mode exhausts every configured recovery tier at every point.
                past_boundary = (
                    not exhaustive_high_power
                    and candidate_boundary_dbm is not None
                    and point.power_dbm >= candidate_boundary_dbm
                )
                recovered = False
                logger.info(
                    "column_recovery_start index=%d power_dbm=%s last_good=%s "
                    "past_boundary=%s",
                    point.index, point.power_dbm, last_good_cur, past_boundary,
                )

                # Tier 2: adaptive fixed-frequency power continuation. This is
                # deliberately attempted even above a previous failed target:
                # it is the cheap diagnostic that distinguishes a missed branch
                # from a branch that remains inaccessible at this frequency.
                if last_good_X is not None and last_good_cur is not None and cur > last_good_cur:
                    X_sub, sub_info = engine.solve_power_substep(
                        point.pump_freq_ghz, last_good_X, last_good_cur, cur,
                        init_db=engine.args.column_power_substep_init_db,
                        min_db=engine.args.column_power_substep_min_db,
                        deadline_s=engine.args.column_power_substep_deadline_s,
                    )
                    row["tier2_substeps"] = sub_info.get("substeps")
                    row["tier2_terminal_reason"] = sub_info.get("terminal_reason")
                    row["tier2_last_current"] = sub_info.get("last_current")
                    logger.info(
                        "column_recovery_tier2 index=%d substeps=%s reason=%s "
                        "reached=%s",
                        point.index, sub_info.get("substeps"),
                        sub_info.get("terminal_reason"),
                        sub_info.get("reached_target"),
                    )
                    if X_sub is not None:
                        sub_row, sub_X = engine.solve_point(
                            point, pass_dir, mode="warm", warm_X=X_sub
                        )
                        if sub_row["status"] == "PASS":
                            row = _merge_column_recovery_result(
                                row, sub_row,
                                {"tier2_recovered": True},
                                "POWER_SUBSTEP",
                            )
                            X = sub_X
                            pred_tag = f"{pred_tag}->power_substep"
                            recovered = True

                # Tier 3: local pseudo-arclength continuation on the same
                # frequency. Do not spend this budget after a previously
                # confirmed connected-branch boundary.
                if (
                    not recovered
                    and not past_boundary
                    and last_good_X is not None
                    and last_good_cur is not None
                    and cur > last_good_cur
                ):
                    arc_X, arc_info = engine.solve_arclength_forward(
                        point.pump_freq_ghz,
                        last_good_X,
                        last_good_cur,
                        cur,
                        ds=engine.args.column_recovery_tier3_ds,
                        max_steps=engine.args.column_recovery_tier3_max_steps,
                        deadline_s=engine.args.column_recovery_tier3_deadline_s,
                    )
                    row["tier3_steps"] = arc_info.get("steps")
                    row["tier3_trace_points"] = arc_info.get("trace_points")
                    row["tier3_fold_lambdas"] = arc_info.get("fold_lambdas")
                    row["tier3_terminal_reason"] = arc_info.get("terminal_reason")
                    logger.info(
                        "column_recovery_tier3 index=%d steps=%s reason=%s "
                        "reached=%s",
                        point.index, arc_info.get("steps"),
                        arc_info.get("terminal_reason"),
                        arc_info.get("reached_target"),
                    )
                    if arc_X is not None:
                        arc_row, arc_state = engine.solve_point(
                            point, pass_dir, mode="warm", warm_X=arc_X
                        )
                        if arc_row["status"] == "PASS":
                            row = _merge_column_recovery_result(
                                row, arc_row,
                                {"tier3_recovered": True},
                                "ARCLENGTH_RECOVERY",
                            )
                            X = arc_state
                            pred_tag = f"{pred_tag}->arclength"
                            recovered = True

                # Tier 3b: pseudo-transient continuation regularises the local
                # HB root solve without adding physical damping.  It is useful
                # when a valid target root exists but Newton's basin is too
                # narrow for either natural continuation or PALC.
                if (
                    not recovered
                    and not past_boundary
                    and last_good_X is not None
                    and last_good_cur is not None
                    and cur > last_good_cur
                    and exhaustive_high_power
                ):
                    ptc_X, ptc_info = engine.solve_pseudo_transient_recovery(
                        point.pump_freq_ghz,
                        cur,
                        last_good_X,
                        deadline_s=engine.args.column_recovery_ptc_deadline_s,
                        max_iter=engine.args.column_recovery_ptc_max_iter,
                    )
                    row["tier3b_recovered"] = False
                    row["tier3b_iterations"] = ptc_info.get("iterations")
                    row["tier3b_terminal_reason"] = ptc_info.get("terminal_reason")
                    row["tier3b_runtime_s"] = ptc_info.get("runtime_s")
                    logger.info(
                        "column_recovery_tier3b index=%d iterations=%s reason=%s "
                        "reached=%s",
                        point.index, ptc_info.get("iterations"),
                        ptc_info.get("terminal_reason"),
                        ptc_info.get("reached_target"),
                    )
                    if ptc_X is not None:
                        ptc_row, ptc_state = engine.solve_point(
                            point, pass_dir, mode="warm", warm_X=ptc_X
                        )
                        if ptc_row["status"] == "PASS":
                            row = _merge_column_recovery_result(
                                row,
                                ptc_row,
                                {"tier3b_recovered": True},
                                "PSEUDO_TRANSIENT_RECOVERY",
                            )
                            X = ptc_state
                            pred_tag = f"{pred_tag}->ptc"
                            recovered = True

                # Tier 4: solve at a nearby frequency at the same target power,
                # then walk back in frequency. This is useful when the target
                # column intersects a narrow continuation obstruction.
                if (
                    not recovered
                    and not past_boundary
                    and last_good_X is not None
                    and last_good_cur is not None
                    and cur > last_good_cur
                ):
                    detour_X, detour_info = _column_frequency_detour_guess(
                        engine,
                        point,
                        last_good_X,
                        cur,
                        anchor_deadline_s=engine.args.column_recovery_tier4_anchor_deadline_s,
                        substep_deadline_s=engine.args.column_recovery_tier4_substep_deadline_s,
                        min_step_ghz=engine.args.column_recovery_tier4_min_step_ghz,
                    )
                    row.update(detour_info)
                    logger.info(
                        "column_recovery_tier4 index=%d anchor=%s reason=%s",
                        point.index, detour_info.get("tier4_anchor_found"),
                        detour_info.get("tier4_terminal_reason"),
                    )
                    if detour_X is not None:
                        detour_row, detour_state = engine.solve_point(
                            point, pass_dir, mode="warm", warm_X=detour_X
                        )
                        if detour_row["status"] == "PASS":
                            row = _merge_column_recovery_result(
                                row, detour_row,
                                {"tier4_recovered": True},
                                "FREQUENCY_RECOVERY",
                            )
                            X = detour_state
                            pred_tag = f"{pred_tag}->frequency_detour"
                            recovered = True

                row["column_recovery_wall_s"] = time.perf_counter() - recovery_started
                if not recovered:
                    if past_boundary and not exhaustive_high_power:
                        row["column_recovery_route"] = "PAST_CONNECTED_BRANCH_BOUNDARY"
                    else:
                        row["column_recovery_route"] = "FAILED_NUMERICAL"
                        if not exhaustive_high_power:
                            candidate_boundary_dbm = point.power_dbm
                logger.info(
                    "column_recovery_end index=%d route=%s recovered=%s wall_s=%.3f",
                    point.index, row.get("column_recovery_route"), recovered,
                    row["column_recovery_wall_s"],
                )

            # Adaptive power-substep recovery: the coarse power step can miss a
            # gain-lobe crest that a finer natural continuation crosses (see
            # diagnostics/2c_measurement_comparison). Walk from the last
            # converged state up to this target in adaptive dBm micro-steps; a
            # step-independent stall (min_db floor) is a real numerical/fold
            # boundary and leaves the cell FAILED so the fold short-circuit can
            # act on it.
            if (
                row["status"] != "PASS"
                and getattr(engine.args, "column_power_substep", False)
                and not recovery_ladder
                and not fail_fast
                and last_good_X is not None
                and last_good_cur is not None
                and cur > last_good_cur
            ):
                X_sub, sub_info = engine.solve_power_substep(
                    point.pump_freq_ghz, last_good_X, last_good_cur, cur,
                    init_db=engine.args.column_power_substep_init_db,
                    min_db=engine.args.column_power_substep_min_db,
                    deadline_s=engine.args.column_power_substep_deadline_s,
                )
                logger.debug("warm_point_power_substep index=%d info=%r", point.index, sub_info)
                row["pump_power_substep_substeps"] = sub_info["substeps"]
                row["pump_power_substep_terminal_reason"] = sub_info["terminal_reason"]
                if X_sub is not None:
                    sub_row, sub_X = engine.solve_point(
                        point, pass_dir, mode="warm", warm_X=X_sub)
                    if sub_row["status"] == "PASS":
                        row, X = sub_row, sub_X
                        pred_tag = f"{pred_tag}->substep"
                elif sub_info["terminal_reason"] == "step_floor":
                    # Step-independent stall -> treat as fold evidence so the
                    # per-column short-circuit can stop retrying above it.
                    row["pump_power_substep_stall_dbm"] = point.power_dbm
                    verified_fold = True

            if recovery_ladder and row["status"] == "PASS":
                row.setdefault(
                    "column_recovery_route",
                    "ARCLENGTH_RECOVERY" if "arclength" in pred_tag else "DIRECT",
                )
            row["warm_retry_reseed"] = retried
            row["pump_predictor"] = pred_tag
            verified_fold = verified_fold or continuation_failure_is_fold_evidence(row)
            logger.debug(
                "warm_point_final index=%d status=%s predictor=%s verified_fold=%s "
                "consecutive_failures=%d",
                point.index, row["status"], pred_tag, verified_fold, consec_fail,
            )
            rows.append(row)
            done += 1
            print(f"[warm {done}/{total}] P={point.power_dbm:.4g} dBm "
                  f"fp={point.pump_freq_ghz:.4g} GHz {mode}"
                  f"{'+' + pred_tag if pred_tag != 'none' else ''}"
                  f"{'+reseed' if retried else ''} "
                  f"status={row['status']} gain={row.get('gain_db')} "
                  f"newton={row.get('pump_newton_total')} "
                  f"pump_s={row.get('pump_runtime_s'):.3f}", flush=True)
            if row["status"] == "PASS":
                # A validated high-power point may have been solved after
                # harmonic enrichment.  The next map point is assembled on
                # the base basis, so project the chained state explicitly
                # before storing it as a warm start.
                chained_X = (
                    engine.project_to_base_pump_basis(
                        point.pump_freq_ghz, X
                    )
                    if hasattr(engine, "project_to_base_pump_basis")
                    else X
                )
                prevprev_X, prevprev_cur = last_good_X, last_good_cur
                last_good_X, last_good_cur = chained_X, cur
                prev_X = chained_X
                consec_fail = 0
            else:
                prev_X = None  # non-fail-fast path re-seeds next point
                consec_fail += 1

            # Per-column fold short-circuit: after `patience` consecutive
            # non-converged points at increasing power, the column is past the
            # HB fold (a turning point -- no re-convergence above it), so mark
            # every remaining higher-power cell past-fold without solving.
            if (
                patience > 0
                and not recovery_ladder
                and verified_fold
                and consec_fail >= patience
                and idx + 1 < len(column)
            ):
                skipped = column[idx + 1:]
                for rest in skipped:
                    rows.append(past_fold_skip_row(rest))
                logger.debug("warm_fold_short_circuit column=%d skipped=%d", j, len(skipped))
                done += len(skipped)
                print(f"[warm {done}/{total}] fp={point.pump_freq_ghz:.4g} GHz "
                      f"fold short-circuit: skipped {len(skipped)} past-fold "
                      f"cells above P={point.power_dbm:.4g} dBm", flush=True)
                break
        # Each frequency column has an independent pump basis.  The chained
        # state is stored as a small retained coefficient array, so cached
        # partitions and their native factors are no longer needed here.
        clear_cache = getattr(engine, "clear_schur_cache", None)
        if callable(clear_cache):
            clear_cache()
    rows.sort(key=lambda r: r["point_index"])
    logger.debug("run_warm_pass_inprocess_complete n_rows=%d", len(rows))
    return rows


# =============================================================================
# Traversal orchestrator (inter-cell method suite: Phases 1-3)
# =============================================================================

from twpa_solver.pump import predictors as _predictors  # noqa: E402


def _grid_dims(points: list[GridPoint]) -> tuple[int, int]:
    return max(p.i_power for p in points) + 1, max(p.j_freq for p in points) + 1


def _traversal_order(points: list[GridPoint], strategy: str, direction: str
                     ) -> list[GridPoint]:
    """Solve order for a traversal strategy (list of GridPoints).

    ``column``/``nearest`` sort column-major (low->high power within a column);
    ``backbone`` solves the lowest-power frequency row first then each column
    upward; ``serpentine`` alternates power direction per column; ``floodfill``
    is a Prim (cheapest-neighbour) order from a central low-power seed.
    """
    n_power, n_freq = _grid_dims(points)
    logger.debug("traversal_order_start strategy=%s direction=%s grid=(%d,%d)", strategy, direction, n_power, n_freq)
    by_ij = {(p.i_power, p.j_freq): p for p in points}

    def col_order(js: list[int]) -> list[int]:
        if direction == "rtl":
            return sorted(js, reverse=True)
        if direction == "center_out":
            mid = (n_freq - 1) / 2.0
            return sorted(js, key=lambda j: abs(j - mid))
        if direction == "two_ended":
            lo, hi = sorted(js), sorted(js, reverse=True)
            out: list[int] = []
            for a, b in zip(lo, hi):
                out.append(a)
                if b != a and b not in out:
                    out.append(b)
            return [j for j in out if j in set(js)]
        return sorted(js)  # ltr

    all_js = sorted({p.j_freq for p in points})

    if strategy == "backbone":
        order: list[GridPoint] = []
        js = col_order(all_js)
        for j in js:  # backbone row (lowest power present in the column)
            col = sorted((p for p in points if p.j_freq == j), key=lambda p: p.i_power)
            if col:
                order.append(col[0])
        for j in js:  # each column upward from its backbone cell
            col = sorted((p for p in points if p.j_freq == j), key=lambda p: p.i_power)
            order.extend(col[1:])
        logger.debug("traversal_order_complete strategy=backbone n_points=%d", len(order))
        return order

    if strategy == "serpentine":
        order = []
        for k, j in enumerate(sorted(all_js)):
            col = sorted((p for p in points if p.j_freq == j), key=lambda p: p.i_power)
            order.extend(col if k % 2 == 0 else list(reversed(col)))
        logger.debug("traversal_order_complete strategy=serpentine n_points=%d", len(order))
        return order

    if strategy == "floodfill":
        import heapq
        powers = sorted({p.i_power for p in points})
        rangeP = max(1, n_power - 1)
        rangeF = max(1, n_freq - 1)
        start = by_ij.get((powers[0], n_freq // 2)) or points[0]
        visited: set[tuple[int, int]] = set()
        order = []
        heap: list[tuple[float, int, int]] = [(0.0, start.i_power, start.j_freq)]
        while heap:
            _cost, i, j = heapq.heappop(heap)
            if (i, j) in visited or (i, j) not in by_ij:
                continue
            visited.add((i, j))
            order.append(by_ij[(i, j)])
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if (ni, nj) in by_ij and (ni, nj) not in visited:
                    cost = abs(di) / rangeP + abs(dj) / rangeF
                    heapq.heappush(heap, (cost, ni, nj))
        logger.debug("traversal_order_complete strategy=floodfill n_points=%d", len(order))
        return order

    # column / nearest: column-major, ascending power.
    order = sorted(points, key=lambda p: (p.j_freq, p.i_power))
    logger.debug("traversal_order_complete strategy=%s n_points=%d", strategy, len(order))
    return order


def _nearest_solved(i: int, j: int, solved: dict, n_power: int, n_freq: int):
    """Nearest already-solved cell to (i,j) by normalised grid distance."""
    best = None
    best_d = float("inf")
    for (si, sj) in solved:
        d = abs(si - i) / max(1, n_power - 1) + abs(sj - j) / max(1, n_freq - 1)
        if d < best_d:
            best_d, best = d, (si, sj)
    return best


def _build_candidates(
    point: GridPoint, cur_t: float, solved: dict, n_power: int, n_freq: int,
) -> dict[str, np.ndarray | None]:
    """Predictor candidate guesses for a target cell from solved neighbours."""
    i, j = point.i_power, point.j_freq
    P, f = point.power_dbm, point.pump_freq_ghz

    def X(ii, jj):
        c = solved.get((ii, jj))
        return c["X"] if c else None

    def cur(ii, jj):
        c = solved.get((ii, jj))
        return c["current"] if c else None

    def frq(ii, jj):
        c = solved.get((ii, jj))
        return c["freq"] if c else None

    cands: dict[str, np.ndarray | None] = {}
    # copy: power parent, else nearest solved.
    parent = X(i - 1, j)
    if parent is None:
        nb = _nearest_solved(i, j, solved, n_power, n_freq)
        parent = solved[nb]["X"] if nb else None
    cands["copy"] = None if parent is None else _predictors.copy_predictor(parent)
    cands["power_secant"] = _predictors.axis_secant(
        X(i - 2, j), X(i - 1, j), cur(i - 2, j), cur(i - 1, j), cur_t)
    cands["freq_secant"] = _predictors.axis_secant(
        X(i, j - 2), X(i, j - 1), frq(i, j - 2), frq(i, j - 1), f)
    cands["corner"] = _predictors.corner_predictor(X(i, j - 1), X(i - 1, j), X(i - 1, j - 1))
    cands["diagonal"] = X(i - 1, j - 1)
    window = [(c["power"], c["freq"], c["X"]) for (si, sj), c in solved.items()
              if abs(si - i) <= 2 and abs(sj - j) <= 2]
    cands["plane"] = _predictors.plane_predictor(window, P, f)
    return cands


def _select_guess(
    point: GridPoint, cur_t: float, solved: dict, solve_problem, engine,
    n_power: int, n_freq: int, args: argparse.Namespace,
) -> tuple[np.ndarray | None, str, list[tuple[str, np.ndarray, float]]]:
    """Pick the initial guess for a cell per --predictor / --portfolio-policy.

    Returns (guess, tag, ranked) where ``ranked`` is the residual-sorted
    candidate list (non-empty only for the portfolio predictor; reused by the
    ranked recovery ladder). ``solve_problem`` is the Schur-reduced (or full)
    problem the cell is actually solved on, so candidate residuals match the
    chained warm-start state shape.
    """
    cands = _build_candidates(point, cur_t, solved, n_power, n_freq)
    predictor = args.predictor
    logger.debug("predictor_selection_start point=%d predictor=%s candidates=%r", point.index, predictor, {k: v is not None for k, v in cands.items()})
    if predictor == "portfolio":
        ranked = _predictors.rank_candidates(
            cands, lambda X: engine.residual_norm(solve_problem, X))
        if not ranked:
            logger.debug("predictor_selection_result point=%d tag=seed", point.index)
            return None, "seed", []
        logger.debug("predictor_selection_result point=%d tag=portfolio:%s", point.index, ranked[0][0])
        return ranked[0][1], f"portfolio:{ranked[0][0]}", ranked
    guess = cands.get(predictor)
    if guess is None:  # fall back to copy of best available parent
        guess = cands.get("copy")
        tag = "copy" if guess is not None else "seed"
        logger.debug("predictor_selection_result point=%d tag=%s", point.index, tag)
        return guess, tag, []
    logger.debug("predictor_selection_result point=%d tag=%s", point.index, predictor)
    return guess, predictor, []


def _attempt(engine, point, pass_dir, prebuilt, *, mode, warm_X):
    logger.debug("recovery_attempt_start point=%d mode=%s guess_shape=%s", point.index, mode, None if warm_X is None else warm_X.shape)
    row, X = engine.solve_point(point, pass_dir, mode=mode, warm_X=warm_X, prebuilt=prebuilt)
    logger.debug("recovery_attempt_complete point=%d status=%s", point.index, row["status"])
    return row, X, row["status"] == "PASS"


def _recover(
    engine, point, pass_dir, prebuilt, solve_problem, cur_t, solved,
    n_power, n_freq, ranked, args, failed_row, failed_X,
) -> tuple[dict, np.ndarray | None, bool, str]:
    """Recovery + fold-policy rescue ladder for a failed cell.

    Runs the --recovery ladder, then any extra --fold-policy attempt, and
    returns (row, X, converged, tag). ``converged`` False here means the cell is
    a genuine fold/skip candidate.
    """
    i, j = point.i_power, point.j_freq
    parent_i = solved.get((i - 1, j)) or solved.get((i, j - 1)) or solved.get((i - 1, j - 1))
    last_row, last_X = failed_row, failed_X
    arclength_fold_current: float | None = None

    def bridge_from(cell) -> tuple[dict, np.ndarray | None, bool] | None:
        if cell is None:
            return None
        guess = engine.solve_bridge(
            cell["X"], cell["current"], cell["freq"], cur_t, point.pump_freq_ghz,
            steps=args.bridge_steps, mode=args.bridge_mode)
        if guess is None:
            return None
        row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=guess)
        return (row, X, ok) if ok else None

    recovery = args.recovery
    logger.debug("recovery_start point=%d policy=%s fold_policy=%s ranked=%d", point.index, recovery, args.fold_policy, len(ranked))
    if recovery == "alt_parent":
        for cell in (solved.get((i - 1, j)), solved.get((i, j - 1)), solved.get((i - 1, j - 1))):
            if cell is None:
                continue
            row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=cell["X"])
            if ok:
                return row, X, True, "alt_parent"
            last_row, last_X = row, X
    elif recovery == "bridge":
        res = bridge_from(parent_i)
        if res:
            return res[0], res[1], True, "bridge"
    elif recovery == "ladder":
        # ranked[0] was already attempted as the initial portfolio guess.
        for _name, guess, _rho in (ranked[1:] if ranked else []):
            row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=guess)
            if ok:
                return row, X, True, "ladder_predictor"
            last_row, last_X = row, X
        res = bridge_from(parent_i)
        if res:
            return res[0], res[1], True, "ladder_bridge"

    # Fold-policy extra rescue before counting toward the skip.
    fp = args.fold_policy
    if fp in ("cross_axis", "combined"):
        cell = solved.get((i, j - 1))
        if cell is not None:
            row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=cell["X"])
            if ok:
                return row, X, True, "cross_axis"
            last_row, last_X = row, X
    if fp in ("bridge_gate", "combined"):
        res = bridge_from(parent_i)
        if res:
            return res[0], res[1], True, "fold_bridge"
    if fp == "arclength":
        # Round the fold: pseudo-arclength continuation to full drive, then a
        # warm target solve from the arclength state. Warm-started from the
        # best available converged neighbour (same lookup as the other
        # recovery ladders) instead of a cold (X=0, lambda=0) start -- the
        # trace then only has to cover the remaining distance to lambda=1,
        # not the whole 0->1 range. This matters beyond speed: each arclength
        # step can pay for several PARDISO refactors and many GMRES
        # iterations, each temporarily allocating a ~10-25 MB array (the
        # coupled Jacobian's packed index/Krylov buffers); a cold trace on a
        # stiff cell can rack up thousands of these within one recovery
        # attempt, which fragments the process heap over a long chunk run --
        # measured crashing with ArrayMemoryError ~170-470 points into a
        # chunk, at a different array size each time (not a fixed-size bug).
        # Warm-starting cuts the per-cell call volume by roughly the same
        # factor as the step-count reduction.
        if parent_i is not None and parent_i.get("current", 0.0) > 0.0:
            X0 = parent_i["X"]
            lam0 = min(parent_i["current"] / cur_t, 0.98)
        else:
            X0 = solve_problem.zeros()
            lam0 = 0.0
        # Cap GMRES iterations tighter than the main solve: the exact
        # preconditioner converges in ~1 iteration on a healthy point, and a
        # corrector call that needs far more than that deep in a stiff
        # region is already grinding, not converging -- let it fail fast
        # (halves ds and retries) rather than burn hundreds of PARDISO/GMRES
        # calls per corrector attempt.
        recovery_settings = dataclasses.replace(engine._settings(), gmres_maxiter=20)
        solver = exp08.HarmonicNewtonKrylovSolver(recovery_settings)
        # ds=0.1 measured too coarse near the map's high-power fold band: a
        # verified-converged seed (coeff_rel 1.4e-13) at fp=7.0 GHz, lambda
        # 0.551 had a plain-Newton convergence radius of only ~0.01-0.05 in
        # lambda (step +0.009 converged, +0.049 did not) -- both this
        # corrector and the unmodified library solve_arclength() failed to
        # accept even one ds=0.1 step from that exact point, confirmed by
        # calling solve_arclength directly. 0.02 sits inside the measured
        # radius with margin; the corrector still halves further on failure,
        # so this only removes wasted oversized first attempts, it does not
        # change behavior where ds=0.1 already worked fine.
        try:
            X_arc, _lam, info = solver.solve_arclength(
                solve_problem, X0, lam0, ds=0.02, max_steps=60, target_lam=1.0,
                max_wall_s=engine.args.inproc_solve_deadline_s,
                rescale_every=args.recovery_arclength_rescale_every,
                max_steps_after_fold=args.recovery_arclength_max_steps_after_fold,
            )
        except (RuntimeError, FloatingPointError, ValueError, OverflowError) as exc:
            # solve_arclength itself now catches a singular-factor RuntimeError
            # internally and returns a terminal_reason instead of raising; this
            # is a second line of defense so an unanticipated failure here
            # downgrades this one cell instead of escaping into the map loop.
            logger.debug(
                "recovery_arclength_exception point=%d error=%r", point.index, exc,
            )
            info = {
                "reached_target": False,
                "terminal_reason": "exception",
                "error": repr(exc),
            }
        finally:
            # Collect promptly so one cell's allocation churn doesn't carry
            # fragmentation pressure into the next cell's solve.
            gc.collect()
        if info.get("reached_target"):
            row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=X_arc)
            if ok:
                return row, X, True, "arclength"
            last_row, last_X = row, X
        elif info.get("fold_lambda") is not None:
            # A real fold was found (see docs/development/
            # arclength_fold_resolution_plan.md Phase 2) but even the
            # extended budget could not round it back to target_lam=1 --
            # record the fold's physical boundary current so whichever row
            # this function ultimately returns reports it instead of a bare
            # failure (injected just before each return below, since neither
            # fail_fast's last_row nor the final reseed attempt's fresh row
            # exists yet at this point).
            arclength_fold_current = float(info["fold_lambda"]) * cur_t

    # Fail-fast still permits the explicitly selected cheap recovery policy,
    # but does not pay for a fresh continuation after those attempts fail.
    if args.inproc_fail_fast:
        if arclength_fold_current is not None and last_row is not None:
            last_row["pump_arclength_fold_current_a"] = arclength_fold_current
        logger.debug("recovery_end point=%d tag=fail_fast", point.index)
        return last_row, last_X, False, "fail_fast"

    # Final fallback: fresh linear_phasor + adaptive reseed. Same PARDISO/GMRES
    # churn risk as run_warm_pass_inprocess's reseed retry -- collect promptly
    # so a failing point here doesn't degrade every later point in this
    # long-lived worker (see the reseed-retry fix in run_warm_pass_inprocess).
    row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="seed", warm_X=None)
    if not ok and arclength_fold_current is not None:
        row["pump_arclength_fold_current_a"] = arclength_fold_current
    gc.collect()
    logger.debug("recovery_end point=%d tag=reseed ok=%s", point.index, ok)
    return row, X, ok, "reseed"


def run_fold_follow(engine: InProcessEngine, freqs: np.ndarray, outdir: Path,
                    args: argparse.Namespace) -> None:
    """Trace the fold power vs frequency with pseudo-arclength -> fold_curve.csv.

    At each frequency, build the pump problem at the maximum-power reference
    current and run the arclength fold locator; the fold ``lambda`` scales the
    reference current, giving a fold current and thus a fold power (dBm).
    """
    from twpa_solver.pump.solver import fold_power
    scale = args.pump_current_jc_scale
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    rows: list[dict[str, Any]] = []
    for f in freqs:
        f = float(f)
        ref_phys = dbm_to_peak_current_a(
            args.pump_power_max_dbm, attenuation_db=attenuation_db_for(f, args),
            z0_ohm=args.z0_ohm, convention=args.power_convention)
        ref_injected = ref_phys * scale
        full_problem, _basis, _omega = engine._build_problem(f, ref_injected)
        # Solve on the Schur-reduced problem for speed (constant retained shape).
        problem = engine._make_solve_problem(full_problem, f)
        lam_fold = fold_power(
            solver, problem, max_steps=120,
            rescale_every=args.recovery_arclength_rescale_every,
        )
        fold_dbm = (peak_current_to_power_dbm(lam_fold * ref_injected / scale, f, args)
                    if lam_fold is not None else None)
        rows.append({"pump_freq_ghz": f, "fold_lambda": lam_fold,
                     "fold_power_dbm": fold_dbm})
        print(f"[fold-follow] fp={f:.4f} GHz fold_lambda={lam_fold} "
              f"fold_power_dbm={fold_dbm}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "fold_curve.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pump_freq_ghz", "fold_lambda", "fold_power_dbm"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


def run_map_traversal(
    points: list[GridPoint], pass_dir: Path, engine: InProcessEngine,
) -> list[dict[str, Any]]:
    """Warm pass with a pluggable traversal / predictor / recovery / fold policy.

    Keeps one in-process ``solved[(i,j)]`` state store shared across BOTH axes,
    so frequency-crossing methods (backbone, nearest, corner, plane, ...) can
    warm-start from converged neighbours in either direction. Requires a single
    process (enforced by the caller via --frequency-chunk-size 0).
    """
    a = engine.args
    n_power, n_freq = _grid_dims(points)
    scale = a.pump_current_jc_scale
    order = _traversal_order(points, a.traversal, a.backbone_direction)
    solved: dict[tuple[int, int], dict] = {}
    skip: set[tuple[int, int]] = set()
    col_fail: dict[int, int] = {}
    patience = int(getattr(a, "fold_skip_patience", 0))
    rows: list[dict[str, Any]] = []
    total = len(points)
    done = 0

    for point in order:
        i, j = point.i_power, point.j_freq
        if (i, j) in skip:
            rows.append(past_fold_skip_row(point))
            done += 1
            continue
        cur_t = point.current_a * scale
        prebuilt = engine.build_problem_for(point)
        solve_problem = engine._make_solve_problem(prebuilt[0], point.pump_freq_ghz)
        guess, tag, ranked = _select_guess(
            point, cur_t, solved, solve_problem, engine, n_power, n_freq, a)
        mode = "warm" if guess is not None else "seed"
        row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode=mode, warm_X=guess)

        if not ok and a.predictor == "portfolio" and a.portfolio_policy == "ranked":
            for name, candidate, _rho in ranked[1:]:
                row, X, ok = _attempt(
                    engine, point, pass_dir, prebuilt, mode="warm", warm_X=candidate,
                )
                tag = f"{tag}->ranked:{name}"
                if ok:
                    break

        if not ok:
            row, X, ok, rtag = _recover(
                engine, point, pass_dir, prebuilt, solve_problem, cur_t, solved,
                n_power, n_freq, ranked, a, row, X)
            tag = f"{tag}->{rtag}"

        row["warm_retry_reseed"] = "reseed" in tag
        row["pump_predictor"] = tag
        rows.append(row)
        done += 1
        print(f"[trav {done}/{total}] P={point.power_dbm:.4g} dBm "
              f"fp={point.pump_freq_ghz:.4g} GHz {a.traversal}:{tag} "
              f"status={row['status']} gain={row.get('gain_db')} "
              f"newton={row.get('pump_newton_total')}", flush=True)

        if ok and X is not None:
            solved[(i, j)] = {"X": X, "current": cur_t, "freq": point.pump_freq_ghz,
                              "power": point.power_dbm}
            col_fail[j] = 0
        else:
            col_fail[j] = col_fail.get(j, 0) + 1
            # Per-column fold short-circuit: skip higher-power cells in this
            # column once patience consecutive fails accrue at increasing power.
            if patience > 0 and col_fail[j] >= patience:
                for ii in range(i + 1, n_power):
                    skip.add((ii, j))

    rows.sort(key=lambda r: r["point_index"])
    clear_cache = getattr(engine, "clear_schur_cache", None)
    if callable(clear_cache):
        clear_cache()
    return rows


def uses_traversal_orchestrator(args: argparse.Namespace) -> bool:
    """Return whether generic traversal/recovery semantics are requested."""
    return (
        args.traversal != "column"
        or args.recovery != "reseed"
        or args.fold_policy != "patience"
    )


# =============================================================================
# Gate
# =============================================================================

@dataclass
class GateResult:
    evaluated: bool
    passed: bool
    reasons: list[str] = field(default_factory=list)
    max_gain_drift_db: float | None = None
    n_compared: int = 0
    warm_converged_frac: float | None = None
    n_warm_failed: int = 0
    cold_pump_runtime_s: float | None = None
    warm_pump_runtime_s: float | None = None
    cold_pump_mean_s: float | None = None
    warm_pump_mean_s: float | None = None
    pump_speedup: float | None = None


def total_pump_runtime(rows: list[dict[str, Any]]) -> float:
    return float(sum(finite_or_none(r.get("pump_runtime_s")) or 0.0 for r in rows))


def mean_pump_runtime(rows: list[dict[str, Any]]) -> float | None:
    vals = [
        finite_or_none(r.get("pump_runtime_s"))
        for r in rows
        if r.get("status") == "PASS" and finite_or_none(r.get("pump_runtime_s")) is not None
    ]
    return float(sum(vals) / len(vals)) if vals else None


def evaluate_gate(
    cold_rows: list[dict[str, Any]],
    warm_rows: list[dict[str, Any]],
    *,
    gate_gain_db: float,
    min_converged_frac: float,
) -> GateResult:
    reasons: list[str] = []

    n_warm = len(warm_rows)
    warm_failures = [r for r in warm_rows if r["status"] != "PASS"]
    converged_frac = (n_warm - len(warm_failures)) / n_warm if n_warm else None
    if converged_frac is not None and converged_frac < min_converged_frac:
        reasons.append(
            f"warm convergence {converged_frac:.4f} < {min_converged_frac:.4f} "
            f"({len(warm_failures)}/{n_warm} failed)"
        )

    cold_by_key = {(r["i_power"], r["j_freq"]): r for r in cold_rows}
    drifts: list[float] = []
    n_compared = 0
    for w in warm_rows:
        c = cold_by_key.get((w["i_power"], w["j_freq"]))
        if c is None or c["status"] != "PASS" or w["status"] != "PASS":
            continue
        gw = finite_or_none(w.get("gain_db"))
        gc = finite_or_none(c.get("gain_db"))
        if gw is None or gc is None:
            continue
        drifts.append(abs(gw - gc))
        n_compared += 1

    max_drift = max(drifts) if drifts else None
    if max_drift is None:
        reasons.append("no comparable cold/warm point pairs")
    elif max_drift > gate_gain_db:
        reasons.append(
            f"max gain drift {max_drift:.3e} dB > gate {gate_gain_db:.3e} dB"
        )

    # Per-point speedup. Comparing totals is invalid when the cold pass is a
    # sparse spot-check (5 points) against a full warm pass (e.g. 1225); use the
    # mean converged pump time per point instead.
    cold_mean = mean_pump_runtime(cold_rows)
    warm_mean = mean_pump_runtime(warm_rows)
    speedup = cold_mean / warm_mean if (cold_mean and warm_mean) else None
    if speedup is None or speedup <= 1.0:
        reasons.append("warm pass not faster than cold (per point)")

    return GateResult(
        evaluated=True,
        passed=not reasons,
        reasons=reasons,
        max_gain_drift_db=max_drift,
        n_compared=n_compared,
        warm_converged_frac=converged_frac,
        n_warm_failed=len(warm_failures),
        cold_pump_runtime_s=total_pump_runtime(cold_rows),
        warm_pump_runtime_s=total_pump_runtime(warm_rows),
        cold_pump_mean_s=cold_mean,
        warm_pump_mean_s=warm_mean,
        pump_speedup=speedup,
    )


# =============================================================================
# Output
# =============================================================================

def write_points_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "pass", "point_index", "i_power", "j_freq", "pump_power_dbm",
        "pump_freq_ghz", "pump_current_peak_a", "status", "pump_status",
        "gain_status", "warm_started", "warm_retry_reseed", "pump_predictor",
        "pump_failure_reason", "gain_failure_reason",
        "gain_db", "gain_vs_off_db",
        "gain_vs_pumpdiag_db", "signal_ghz", "signal_attenuation_db", "linear_rel_residual",
        "pump_runtime_s", "pump_wall_runtime_s", "pump_setup_runtime_s",
        "pump_schur_setup_runtime_s", "pump_solve_wall_runtime_s",
        "pump_write_runtime_s", "pump_factor_runtime_s",
        "pump_preconditioner_assembly_runtime_s",
        "pump_preconditioner_numeric_factor_runtime_s", "pump_newton_total",
        "pump_gmres_total", "pump_coeff_rel", "pump_time_rel", "pump_branch_current_max",
        "pump_branch_current_max_over_ic", "pump_strongest_branch_index",
        "pump_branch_min_cos_phase", "pump_boundary_predictor_status",
        "pump_residual_max_omitted_mode_rel", "sidebands", "single_tone_forced",
        "pump_dominant_omitted_modes",
        "pump_continuation_method", "pump_continuation_steps",
        "pump_continuation_reached_target", "pump_continuation_fold_lambda",
        "pump_continuation_runtime_s",
        "column_recovery_route", "column_recovery_wall_s",
        "tier2_recovered", "tier2_substeps", "tier2_terminal_reason",
        "tier2_last_current",
        "tier3_recovered", "tier3_steps", "tier3_trace_points",
        "tier3_fold_lambdas", "tier3_terminal_reason",
        "tier3b_recovered", "tier3b_iterations", "tier3b_terminal_reason",
        "tier3b_runtime_s",
        "tier4_recovered", "tier4_anchor_found", "tier4_anchor_freq_ghz",
        "tier4_anchor_attempts", "tier4_anchor_runtime_s", "tier4_substeps",
        "tier4_terminal_reason", "tier4_substep_runtime_s",
        "pump_column_arclength_fold_lambda",
        "pump_column_arclength_terminal_reason",
        "pump_arclength_fold_current_a",
        "gain_total_runtime_s", "gain_wall_runtime_s", "gain_gamma_hat_runtime_s",
        "gain_khat_build_runtime_s", "gain_khat_off_runtime_s",
        "gain_matrix_assemble_runtime_s", "gain_factor_solve_runtime_s",
        "gain_baseline_off_runtime_s", "gain_baseline_pumpdiag_runtime_s",
        "spectrum_peak_gain_db", "spectrum_peak_signal_ghz",
        "elapsed_s", "pump_dir",
        # Optional hybrid HB/TD telemetry.  Legacy maps leave these blank;
        # retaining them makes TD-assisted coverage auditable without storing
        # large transient states in the CSV.
        "hybrid_route", "hybrid_state", "hybrid_classification",
        "td_periods", "td_d1", "td_best_low_order_dn", "td_r_j",
        "td_phase_winding", "td_period1_projection", "td_projected_state_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def gain_grid(rows: list[dict[str, Any]], n_power: int, n_freq: int) -> np.ndarray:
    grid = np.full((n_power, n_freq), np.nan, dtype=float)
    for r in rows:
        value = finite_or_none(r.get("gain_db"))
        if value is not None and r["status"] == "PASS":
            grid[int(r["i_power"]), int(r["j_freq"])] = value
    return grid


def write_spectrum(
    path: Path, rows: list[dict[str, Any]], powers: np.ndarray,
    freqs: np.ndarray, offsets: list[float],
) -> None:
    """Write the per-cell signal spectrum as a (n_power, n_freq, n_offset) cube.

    Reads the ``_spectrum`` payload each PASS row carries (offsets aligned to
    ``offsets``); non-solved cells stay NaN. ``signal_ghz`` is fp+offset, so the
    absolute signal axis is per (offset, j_freq) -- stored as a 2D helper too.
    """
    n_off = len(offsets)
    cube = np.full((powers.size, freqs.size, n_off), np.nan, dtype=float)
    off_arr = np.asarray(offsets, dtype=float)
    for r in rows:
        spec = r.get("_spectrum")
        if not spec or r["status"] != "PASS":
            continue
        i, j = int(r["i_power"]), int(r["j_freq"])
        for k, (gd, st) in enumerate(zip(spec["gain_db"], spec["status"])):
            if st == "VALID_SOLVED" and k < n_off:
                cube[i, j, k] = float(gd)
    signal_ghz = freqs[None, :] + off_arr[:, None] / 1000.0  # (n_off, n_freq)
    np.savez(path, pump_power_dbm=powers, pump_frequency_ghz=freqs,
             signal_offset_mhz=off_arr, gain_spectrum_db=cube,
             signal_ghz=signal_ghz)


def write_arrays(
    path: Path,
    powers: np.ndarray,
    freqs: np.ndarray,
    grids: dict[str, np.ndarray],
) -> None:
    np.savez(path, pump_power_dbm=powers, pump_frequency_ghz=freqs, **grids)


def total_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [finite_or_none(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return float(sum(vals)) if vals else None


def timing_totals(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = [
        "pump_wall_runtime_s",
        "pump_runtime_s",
        "pump_setup_runtime_s",
        "pump_schur_setup_runtime_s",
        "pump_solve_wall_runtime_s",
        "pump_write_runtime_s",
        "pump_factor_runtime_s",
        "pump_preconditioner_assembly_runtime_s",
        "pump_preconditioner_numeric_factor_runtime_s",
        "gain_wall_runtime_s",
        "gain_total_runtime_s",
        "gain_gamma_hat_runtime_s",
        "gain_khat_build_runtime_s",
        "gain_khat_off_runtime_s",
        "gain_matrix_assemble_runtime_s",
        "gain_factor_solve_runtime_s",
        "gain_baseline_off_runtime_s",
        "gain_baseline_pumpdiag_runtime_s",
        "elapsed_s",
    ]
    return {key: total_metric(rows, key) for key in keys}


def write_summary(
    outdir: Path,
    args: argparse.Namespace,
    cold_rows: list[dict[str, Any]],
    warm_rows: list[dict[str, Any]],
    gate: GateResult,
    elapsed_s: float,
) -> None:
    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out

    from twpa_solver.map import coverage_summary

    coverage_rows = warm_rows if warm_rows else cold_rows
    summary: dict[str, Any] = {
        "mode": args.mode,
        "workflow_mode": getattr(args, "workflow_mode", None),
        "compact_output": bool(getattr(args, "compact_output", False)),
        "output_dir": str(outdir),
        "grid": {"n_power": args.n_power, "n_frequency": args.n_frequency},
        "grid_from_measurement_dir": (
            str(args.grid_from_measurement_dir)
            if args.grid_from_measurement_dir is not None else None
        ),
        "pump_power_dbm": [args.pump_power_min_dbm, args.pump_power_max_dbm],
        "pump_freq_ghz": [args.pump_freq_min_ghz, args.pump_freq_max_ghz],
        "attenuation_db": args.attenuation_db,
        "attenuation_model": ("flat" if args.attenuation_db is not None
                              else "loss_A10 c + a*sqrt(f) + b*f"),
        "z0_ohm": args.z0_ohm,
        "signal_ghz": args.signal_ghz,
        "signal_detuning_mhz": args.signal_detuning_mhz,
        "signal_attenuation_db": args.signal_attenuation_db,
        "signal_attenuation_model": (
            "flat" if args.signal_attenuation_db is not None else "loss_B1 c + a*sqrt(f) + b*f"
        ),
        "column_recovery_ladder": bool(getattr(args, "column_recovery_ladder", False)),
        "column_recovery_tier3_ds": getattr(args, "column_recovery_tier3_ds", None),
        "column_recovery_tier3_max_steps": getattr(args, "column_recovery_tier3_max_steps", None),
        "column_recovery_tier3_deadline_s": getattr(args, "column_recovery_tier3_deadline_s", None),
        "column_recovery_tier4_anchor_deadline_s": getattr(args, "column_recovery_tier4_anchor_deadline_s", None),
        "column_recovery_tier4_substep_deadline_s": getattr(args, "column_recovery_tier4_substep_deadline_s", None),
        "column_recovery_tier4_min_step_ghz": getattr(args, "column_recovery_tier4_min_step_ghz", None),
        "column_recovery_ptc_deadline_s": getattr(args, "column_recovery_ptc_deadline_s", None),
        "column_recovery_ptc_max_iter": getattr(args, "column_recovery_ptc_max_iter", None),
        "high_power_recovery": bool(getattr(args, "high_power_recovery", False)),
        "high_power_max_newton": getattr(args, "high_power_max_newton", None),
        "high_power_harmonic_max_mode": getattr(args, "high_power_harmonic_max_mode", None),
        "high_power_min_alpha": getattr(args, "high_power_min_alpha", None),
        "high_power_stall_patience": getattr(args, "high_power_stall_patience", None),
        "pump_full_residual_gate": getattr(args, "pump_full_residual_gate", None),
        "pump_record_residual_spectrum": bool(
            getattr(args, "pump_record_residual_spectrum", False)
        ),
        "dc_current_a": args.dc_current_a,
        "dc_solution": str(args.dc_solution) if args.dc_solution is not None else None,
        "signal_convention": ("fixed" if args.signal_ghz is not None
                              else f"ws = wp - {args.signal_detuning_mhz} MHz"),
        "power_convention": args.power_convention,
        "current_convention": (
            "I_peak = sqrt(8 * P_W / Z0), P = P_dbm - attenuation_db"
            if args.power_convention == "norton"
            else "I_peak = sqrt(2 * P_W / Z0), P = P_dbm - attenuation_db"
        ),
        "cold_status_counts": counts(cold_rows),
        "warm_status_counts": counts(warm_rows),
        "coverage": coverage_summary(coverage_rows),
        "cold_pump_runtime_s": total_pump_runtime(cold_rows) if cold_rows else None,
        "warm_pump_runtime_s": total_pump_runtime(warm_rows) if warm_rows else None,
        "cold_gain_runtime_s": total_metric(cold_rows, "gain_total_runtime_s") if cold_rows else None,
        "warm_gain_runtime_s": total_metric(warm_rows, "gain_total_runtime_s") if warm_rows else None,
        "cold_timing_totals": timing_totals(cold_rows) if cold_rows else {},
        "warm_timing_totals": timing_totals(warm_rows) if warm_rows else {},
        "elapsed_s": elapsed_s,
        "peak_rss_bytes": getattr(args, "peak_rss_bytes", None) or peak_rss_bytes(),
        "gate": {
            "evaluated": gate.evaluated,
            "passed": gate.passed,
            "reasons": gate.reasons,
            "gate_gain_db": args.gate_gain_db,
            "max_gain_drift_db": gate.max_gain_drift_db,
            "n_compared": gate.n_compared,
            "min_converged_frac": args.gate_min_converged_frac,
            "warm_converged_frac": gate.warm_converged_frac,
            "n_warm_failed": gate.n_warm_failed,
            "pump_speedup_per_point": gate.pump_speedup,
            "cold_pump_mean_s": gate.cold_pump_mean_s,
            "warm_pump_mean_s": gate.warm_pump_mean_s,
            "cold_pump_runtime_s": gate.cold_pump_runtime_s,
            "warm_pump_runtime_s": gate.warm_pump_runtime_s,
        },
    }
    (outdir / "map_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    verdict = "n/a"
    if gate.evaluated:
        verdict = "PASS" if gate.passed else "FAIL"
    lines = [
        "# IPM Warm-Started Pump/Gain Map (exp10)",
        "",
        f"- mode: `{args.mode}`",
        f"- grid: `{args.n_power} x {args.n_frequency}` "
        f"(power `{args.pump_power_min_dbm}`..`{args.pump_power_max_dbm}` dBm, "
        f"freq `{args.pump_freq_min_ghz}`..`{args.pump_freq_max_ghz}` GHz)",
        f"- cold status: `{counts(cold_rows)}`" if cold_rows else "- cold pass: not run",
        f"- warm status: `{counts(warm_rows)}`" if warm_rows else "- warm pass: not run",
        f"- elapsed: `{elapsed_s:.3f}` s",
        f"- warm pump/gain total: `{total_metric(warm_rows, 'pump_runtime_s')}` / `{total_metric(warm_rows, 'gain_total_runtime_s')}` s" if warm_rows else "- warm timing: not run",
        f"- cold pump/gain total: `{total_metric(cold_rows, 'pump_runtime_s')}` / `{total_metric(cold_rows, 'gain_total_runtime_s')}` s" if cold_rows else "- cold timing: not run",
        "",
        "## Gate",
        "",
        f"- verdict: **{verdict}**",
    ]
    if gate.evaluated:
        if gate.reasons:
            lines.append(f"- reasons: `{'; '.join(gate.reasons)}`")
        lines.extend([
            f"- warm converged: `{gate.warm_converged_frac}` "
            f"({gate.n_warm_failed} failed; min `{args.gate_min_converged_frac}`)",
            f"- compared point pairs: `{gate.n_compared}`",
            f"- max gain drift: `{gate.max_gain_drift_db}` dB (gate `{args.gate_gain_db}` dB)",
            f"- pump mean per point: cold `{gate.cold_pump_mean_s}` s, warm `{gate.warm_pump_mean_s}` s",
            f"- pump speedup (per point): `{gate.pump_speedup}`x",
        ])
    lines.extend(["", "## Artifacts", "",
                  "- `map_points.csv`", "- `map_arrays.npz`", "- `map_summary.json`"])
    (outdir / "map_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_measurement_grid(measurement_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load exact pump axes from a Themis ``105C5_*GHz.npy`` directory.

    The Themis frequency filenames are rounded to MHz and are not necessarily
    an exact ``linspace``.  Loading the axes from the files prevents a map
    requested with the same endpoints and point count from drifting by a few
    hundred kHz or more.  The power axis is taken from the first file and is
    required to agree across all files.
    """
    files = []
    for path in measurement_dir.glob("105C5_*GHz.npy"):
        match = _THEMIS_FREQ_RE.search(path.name)
        if match is not None:
            files.append((float(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(
            f"no Themis 105C5_*GHz.npy files found in {measurement_dir}"
        )

    frequencies = np.asarray([item[0] for item in files], dtype=float)
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError(f"measurement frequency grid is not strictly increasing: {measurement_dir}")

    first = np.load(files[0][1], allow_pickle=True).item()
    powers = np.asarray(first["PumpPower"], dtype=float).reshape(-1)
    if powers.size == 0 or np.any(~np.isfinite(powers)) or np.any(np.diff(powers) <= 0.0):
        raise ValueError(f"invalid PumpPower axis in {files[0][1]}")

    for _, path in files[1:]:
        data = np.load(path, allow_pickle=True).item()
        other = np.asarray(data["PumpPower"], dtype=float).reshape(-1)
        if other.shape != powers.shape or not np.allclose(other, powers, rtol=0.0, atol=1e-9):
            raise ValueError(f"PumpPower axis differs between {files[0][1]} and {path}")

    return powers, frequencies


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["cold", "warmstart", "both"], default="warmstart")
    p.add_argument("--loss-model", choices=["auto", "current_complex_c", "real_capacitance",
                                              "conjugate_complex_c", "complex_c_sign_omega",
                                              "conductance_signed_omega", "conductance_abs_omega",
                                              "conductance_abs_omega_opposite"], default="auto")
    p.add_argument("--executor", choices=["subprocess", "inprocess"], default="inprocess",
                   help="inprocess runs pump+gain in this process (no per-point import "
                   "tax); numerics are identical to the subprocess path.")
    p.add_argument("--inproc-gmres-maxiter", type=int, default=80,
                   help="GMRES outer restart cycles per Newton step (x gmres_restart=60 inner). "
                   "A low value (e.g. 4) bounds the per-Newton cost so over-fold solves don't "
                   "grind thousands of inner iterations; warm-start steps converge in <1 cycle.")
    p.add_argument("--inproc-solve-deadline-s", "--inproc-solve-deadline",
                   dest="inproc_solve_deadline_s", type=float, default=0.0,
                   help="Per-solve wall-time budget (s) for the in-process path; 0 disables. "
                   "Bounds stiff over-fold solves near the fold.")
    p.add_argument("--inproc-max-newton", type=int, default=16,
                   help="Max Newton iterations per in-process solve. A small cap (e.g. 10) "
                   "makes over-fold points fail fast; warm-start neighbours converge in few.")
    p.add_argument(
        "--inproc-stall-patience",
        type=int,
        default=4,
        help="Accepted Newton steps with weak contraction before a solve is marked stalled. "
        "Use 0 only for an explicitly bounded high-power diagnostic campaign.",
    )
    p.add_argument(
        "--inproc-stall-ratio",
        type=float,
        default=0.8,
        help="Residual reduction ratio above which a Newton step counts as stalled.",
    )
    p.add_argument(
        "--inproc-min-alpha",
        type=float,
        default=1.0 / 1024.0,
        help="Smallest ordinary Newton line-search factor before failure.",
    )
    p.add_argument(
        "--high-power-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the exhaustive high-power recovery path. This enables the bounded "
            "column ladder, disables its solver-stall shortcut, attempts pseudo-transient "
            "recovery, and records full residual/Ic diagnostics. It does not change "
            "the circuit, attenuation model, or declare a failed solve physical."
        ),
    )
    p.add_argument(
        "--high-power-max-newton",
        type=int,
        default=32,
        help="Newton cap used by --high-power-recovery after explicit recovery steps.",
    )
    p.add_argument(
        "--high-power-stall-patience",
        type=int,
        default=0,
        help="Stall patience for --high-power-recovery; 0 disables the early stall exit.",
    )
    p.add_argument(
        "--high-power-stall-ratio",
        type=float,
        default=0.95,
        help="Stall ratio used by --high-power-recovery when stall detection is enabled.",
    )
    p.add_argument(
        "--high-power-min-alpha",
        type=float,
        default=1.0 / 65536.0,
        help="Minimum Newton line-search factor used by --high-power-recovery.",
    )
    p.add_argument(
        "--high-power-harmonic-max-mode",
        type=int,
        default=35,
        help=(
            "Largest odd pump mode used by high-power harmonic enrichment. "
            "The basis is promoted one odd mode at a time from the failed or "
            "converged state; no mode is accepted without the full residual gate."
        ),
    )
    p.add_argument(
        "--inproc-fallback-fixed-steps",
        type=int,
        default=20,
        help="Fixed ladder length after adaptive continuation gives up. Lower this "
        "for bounded recovery campaigns; the fixed method itself still uses "
        "--continuation-steps.",
    )
    p.add_argument(
        "--inproc-continuation-deadline-s",
        type=float,
        default=0.0,
        help="Total wall-time budget for adaptive/affine continuation, excluding "
        "the final target solve. 0 inherits --inproc-solve-deadline-s.",
    )
    p.add_argument("--inproc-fail-fast", action="store_true",
                   help="In-process warm pass: skip reseed/fallback recovery on a failed "
                   "point and keep warm-starting from the last converged neighbour, so "
                   "over-fold points fail in ~one stalled solve. For high-power fold maps.")
    p.add_argument(
        "--initial-pump-dir",
        type=Path,
        default=None,
        help="Optional verified pump directory containing pump_solution.npz. "
        "Use it as the first warm-start state of each frequency column.",
    )
    p.add_argument(
        "--initial-pump-power-dbm",
        type=float,
        default=None,
        help="Power coordinate of --initial-pump-dir, required for "
        "secant/pseudo-arclength continuation.",
    )
    p.add_argument("--fold-skip-patience", type=int, default=0,
                   help="Per-column fold short-circuit (in-process warm pass): after "
                   "skip the remaining higher-power cells only after the optional "
                   "pseudo-arclength recovery has reported a turning point (marked "
                   "SKIP_PAST_FOLD, gain NaN). A failed target solve alone is never "
                   "treated as a fold. 0 disables skipping (default).")
    p.add_argument(
        "--column-arclength-recovery",
        action="store_true",
        help="On every failed cell in each legacy power column (not just the "
        "first), trace one scaled pseudo-arclength branch from the last two "
        "converged states and use its target-power crossings (capped to 2 "
        "guesses) as Newton recovery guesses. No per-column lock: a cell "
        "whose own trace found no crossing does not block later cells from "
        "trying again once bounded by --inproc-solve-deadline-s.",
    )
    p.add_argument("--column-arclength-ds", type=float, default=0.02)
    p.add_argument("--column-arclength-max-steps", type=int, default=80)
    p.add_argument(
        "--column-arclength-deadline-s",
        type=float,
        default=180.0,
        help="Total wall-time budget for each column pseudo-arclength trace. "
        "Separate from the per-target Newton deadline; 0 disables the trace "
        "deadline.",
    )
    p.add_argument(
        "--column-power-substep",
        action="store_true",
        help="On a failed warm cell, recover by adaptive natural-parameter "
        "continuation along the power axis from the last converged state: "
        "walk up in adaptive dBm micro-steps, warm-starting each. Crosses "
        "gain-lobe crests the coarse power grid misses; a step-independent "
        "stall (min-db floor) is recorded as a numerical/fold boundary "
        "rather than retried. See diagnostics/2c_measurement_comparison.",
    )
    p.add_argument(
        "--column-power-substep-init-db",
        type=float,
        default=0.1,
        help="Initial (and maximum) dBm micro-step for --column-power-substep; "
        "grows x1.5 on success, halves on failure.",
    )
    p.add_argument(
        "--column-power-substep-min-db",
        type=float,
        default=0.005,
        help="Minimum dBm micro-step for --column-power-substep. Below this the "
        "branch is declared to have a step-independent stall (fold/numerical "
        "boundary) and the cell is left failed.",
    )
    p.add_argument(
        "--column-power-substep-deadline-s",
        type=float,
        default=120.0,
        help="Per-target wall-time budget for the --column-power-substep walk.",
    )
    p.add_argument(
        "--column-recovery-ladder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the bounded high-power column recovery ladder: direct Newton, "
            "adaptive power substeps, local pseudo-arclength, then a nearby-frequency "
            "detour. The normal column map remains unchanged unless this is enabled."
        ),
    )
    p.add_argument(
        "--column-recovery-tier3-ds",
        type=float,
        default=0.01,
        help="Initial pseudo-arclength step for the high-power recovery ladder.",
    )
    p.add_argument(
        "--column-recovery-tier3-max-steps",
        type=int,
        default=150,
        help="Maximum local pseudo-arclength steps for the recovery ladder.",
    )
    p.add_argument(
        "--column-recovery-tier3-deadline-s",
        type=float,
        default=60.0,
        help="Wall-time budget for the local pseudo-arclength recovery tier.",
    )
    p.add_argument(
        "--column-recovery-tier4-anchor-deadline-s",
        type=float,
        default=20.0,
        help="Per-frequency-anchor solve budget for the frequency-detour tier.",
    )
    p.add_argument(
        "--column-recovery-tier4-substep-deadline-s",
        type=float,
        default=60.0,
        help="Wall-time budget for the frequency-detour substep tier.",
    )
    p.add_argument(
        "--column-recovery-tier4-min-step-ghz",
        type=float,
        default=0.0005,
        help="Minimum frequency substep for the recovery ladder.",
    )
    p.add_argument(
        "--column-recovery-ptc-deadline-s",
        type=float,
        default=90.0,
        help="Wall-time budget for pseudo-transient recovery after PALC fails.",
    )
    p.add_argument(
        "--column-recovery-ptc-max-iter",
        type=int,
        default=128,
        help="Maximum pseudo-transient iterations in the high-power recovery tier.",
    )
    p.add_argument(
        "--pump-full-residual-gate",
        type=float,
        default=None,
        help=(
            "Optional full reconstructed time-domain residual gate for every pump. "
            "If omitted, coefficient convergence remains the legacy acceptance gate; "
            "--high-power-recovery uses 1e-7 unless overridden."
        ),
    )
    p.add_argument(
        "--pump-record-residual-spectrum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist compact retained/omitted residual-spectrum telemetry per pump point.",
    )
    p.add_argument("--inproc-preconditioner",
                   choices=["mean_tangent", "real_coupled", "real_coupled_fast",
                            "spectral_coupled", "linear"],
                   default="real_coupled_fast",
                   help="Preconditioner for the in-process pump solve. mean_tangent "
                   "(default) is cheapest for small warm-start steps; real_coupled "
                   "cuts GMRES iters but its full-Jacobian LU is costlier per Newton.")
    p.add_argument("--inproc-pump-backend", choices=["full", "schur_cpu_mt"],
                   default="schur_cpu_mt",
                   help="In-process pump backend. 'full' (default, legacy) solves all "
                   "nodes. 'schur_cpu_mt' eliminates linear-internal nodes via an "
                   "assembled sparse Schur complement (constant per frequency) and "
                   "solves the retained system -- 2.5-4.5x faster at the high-power "
                   "fold, gain identical. Pair with --inproc-preconditioner mean_tangent.")
    p.add_argument("--inproc-schur-cache-size", type=int, default=None,
                   help="Max Schur partitions kept in memory (LRU). Default is 2 "
                   "for standard runs and 1 for --high-power-recovery. Each "
                   "partition owns large sparse factors; explicit cleanup is "
                   "performed on eviction.")
    p.add_argument("--inproc-precond-reuse", type=int, default=1,
                   help="Reuse the preconditioner factor for up to N consecutive Newton "
                   "steps (modified-Newton). 1 (default) refactors every step. N>1 "
                   "amortizes the LU across steps -- the big win for real_coupled near "
                   "the fold, where the exact LU barely changes between steps.")
    p.add_argument("--inproc-precond-refresh-gmres", type=int, default=0,
                   help="Force an early factor refresh when the previous Newton step's "
                   "GMRES iterations crossed this threshold (staleness guard for "
                   "--inproc-precond-reuse). 0 disables.")
    p.add_argument("--inproc-fold-predictor", choices=["none", "secant"],
                   default="secant",
                   help="In-process warm pass: build the next power point's initial "
                   "guess by extrapolating along the pump-current axis from the last "
                   "two converged solutions (secant), instead of copying the previous "
                   "solution. Cuts Newton steps near the fold where the state moves "
                   "fast with power. Physics unchanged (initial guess only); a failed "
                   "predicted solve falls back to the plain warm start.")
    # --- Inter-cell method suite (opt-in; default reproduces the column pass) ---
    # See docs/reports/pump_map_continuation_methods.tex + the campaign matrix.
    # Frequency-crossing traversals require a single process, so they force
    # --frequency-chunk-size 0 (they share one in-process solved-state store).
    p.add_argument("--traversal",
                   choices=["column", "backbone", "nearest", "serpentine", "floodfill"],
                   default="column",
                   help="Map traversal / warm-start order. 'column' (default) is "
                   "the legacy per-frequency-column low->high power pass with no "
                   "cross-column warm state. The others reuse converged cells "
                   "across BOTH axes (single process only).")
    p.add_argument("--backbone-direction",
                   choices=["ltr", "rtl", "center_out", "two_ended"],
                   default="center_out",
                   help="For --traversal backbone: order in which the lowest-power "
                   "frequency backbone row is solved before launching each upward "
                   "power column from it.")
    p.add_argument("--predictor",
                   choices=["copy", "power_secant", "freq_secant", "corner",
                            "plane", "portfolio"],
                   default="power_secant",
                   help="Inter-cell initial-guess predictor for the traversal "
                   "orchestrator (ignored by --traversal column, which uses "
                   "--inproc-fold-predictor). 'portfolio' ranks several by target "
                   "residual.")
    p.add_argument("--portfolio-policy", choices=["best", "ranked"], default="best",
                   help="--predictor portfolio: 'best' tries only the lowest-residual "
                   "candidate; 'ranked' tries candidates in ascending-residual order "
                   "until one converges.")
    p.add_argument("--recovery",
                   choices=["none", "reseed", "alt_parent", "bridge", "ladder"],
                   default="reseed",
                   help="Failed-cell recovery for the traversal orchestrator. "
                   "'none' keeps the initial failure; 'reseed' (legacy) does a "
                   "fresh linear_phasor+adaptive solve; "
                   "'alt_parent' first retries from power/freq/diagonal parents; "
                   "'bridge' continues from the best parent along (P,f); 'ladder' "
                   "residual-ranks parents then bridges from the best.")
    p.add_argument("--bridge-steps", type=int, default=4,
                   help="Sub-steps for bridge continuation (recovery=bridge/ladder "
                   "and --fold-policy bridge_gate/combined).")
    p.add_argument("--bridge-mode",
                   choices=["diagonal", "freq_first", "power_first", "adaptive"],
                   default="adaptive",
                   help="Path from parent (P0,f0) to target (P1,f1) for bridge "
                   "continuation.")
    p.add_argument("--fold-policy",
                   choices=["patience", "cross_axis", "bridge_gate", "combined", "arclength"],
                   default="patience",
                   help="When a failed cell counts toward the per-column fold "
                   "short-circuit. 'patience' (legacy) counts every fail; the "
                   "others require cross-axis / bridge / full recovery to also fail "
                   "first; 'arclength' rounds the fold with pseudo-arclength.")
    p.add_argument("--recovery-arclength-rescale-every", type=int, default=0,
                   help="With --fold-policy arclength: recompute the arclength "
                   "state_scale every N accepted steps (0 = disabled, matches "
                   "solve_arclength's own default -- no behavior change).")
    p.add_argument("--recovery-arclength-max-steps-after-fold", type=int, default=0,
                   help="With --fold-policy arclength: once a fold is detected, "
                   "extend the step budget to at least fold_step + this many more "
                   "steps (0 = disabled -- no behavior change; see "
                   "docs/development/arclength_fold_resolution_plan.md Phase 2/4).")
    p.add_argument("--inproc-continuation",
                   choices=["fixed", "adaptive_copy", "adaptive_secant",
                            "adaptive_tangent", "affine", "ptc", "arclength"],
                   default="adaptive_secant",
                   help="Intra-cell continuation for seed/cold cells (solver.py). "
                   "fixed is the 20-step reference; adaptive_copy is natural-parameter "
                   "continuation without prediction; adaptive_secant (default) uses "
                   "the previous two lambda states; adaptive_tangent uses the exact "
                   "lambda tangent; affine sizes steps from corrector contraction; "
                   "ptc is pseudo-transient; arclength augments state and lambda.")
    p.add_argument("--inproc-arclength-ds", type=float, default=0.1)
    p.add_argument("--inproc-arclength-max-steps", type=int, default=80)
    p.add_argument("--fold-follow", action="store_true",
                   help="Diagnostic: trace the fold power vs frequency with "
                   "pseudo-arclength and write fold_curve.csv; no gain map is run.")
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "exp10_pump_map_warmstart")
    p.add_argument("--circuit-dir", "--ipm-dir", dest="circuit_dir", type=Path, default=ROOT / "outputs" / "ipm_python_design")

    p.add_argument("--n-power", type=int, default=50)
    p.add_argument("--n-frequency", type=int, default=50)
    p.add_argument(
        "--grid-from-measurement-dir",
        type=Path,
        default=None,
        help=(
            "Use the exact PumpPower axis and numeric 105C5_*GHz filename "
            "frequency axis from a Themis measurement directory. Overrides "
            "--n-power/--n-frequency and the pump min/max values."
        ),
    )
    p.add_argument("--frequency-chunk-size", type=int, default=10,
                   help="Run frequency columns in separate worker processes of this "
                   "many columns each, then merge. 10 is the standard memory-safe "
                   "map behavior; 0 disables chunking.")
    p.add_argument(
        "--frequency-workers", type=int, default=1,
        help="Parallel frequency-chunk worker processes; 1 preserves serial behavior.",
    )
    p.add_argument(
        "--local-traversal-chunks",
        action="store_true",
        help="Allow non-column traversals to run as independent frequency-local "
        "chunks. This bounds native solver memory by restarting the process per "
        "chunk, at the cost of not sharing warm states across chunk boundaries.",
    )
    p.add_argument("--resume-chunks", action=argparse.BooleanOptionalAction, default=True,
                   help="With --frequency-chunk-size, skip chunk workers whose "
                   "map_points.csv/map_summary.json already look complete.")
    p.add_argument("--chunk-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--frequency-index-start", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--frequency-index-stop", type=int, default=None, help=argparse.SUPPRESS)
    # External power window. With the ~35 dB pump-band line loss + 50 ohm this
    # spans physical pump ~0.5..1.6 x median Ic; after the JC 2x scale the JTWPA
    # gain ridge runs from onset (~0 dB) up to ~12 dB near JC's 1.5 Ic point.
    p.add_argument("--pump-power-min-dbm", type=float, default=-30.0)
    p.add_argument("--pump-power-max-dbm", type=float, default=-20.0)
    p.add_argument("--pump-freq-min-ghz", type=float, default=7.0)
    p.add_argument("--pump-freq-max-ghz", type=float, default=8.0)
    p.add_argument("--attenuation-db", type=float, default=None,
                   help="Flat line attenuation (dB). If omitted, use the measured "
                   "loss_A10 frequency-dependent model c + a*sqrt(f) + b*f.")
    p.add_argument(
        "--dc-current-a", type=float, default=0.0,
        help="Uniform DC current through kinetic branches; zero preserves the legacy path.",
    )
    p.add_argument(
        "--dc-solution", type=Path, default=None,
        help="Optional dc_solution.npz (or directory) providing psi_dc/x_dc; overrides --dc-current-a.",
    )
    p.add_argument(
        "--dc-branch-flux-over-phi0", type=float, default=None,
        help="Uniform reduced external flux for every nonlinear branch, e.g. 0.33 "
             "for the RF-SQUID 3WM validation design.",
    )
    p.add_argument(
        "--signal-attenuation-db", type=float, default=None,
        help="Flat signal-line attenuation for signal-spectrum referral; defaults to loss_B1.",
    )
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument(
        "--power-convention",
        choices=("norton", "legacy_traveling_wave"),
        default="legacy_traveling_wave",
        help=(
            "Port drive current -> available power relation. Every drive "
            "port in the production netlists is an ideal current source in "
            "parallel with Z0, a matched wave port (I is the incident "
            "wave's own current amplitude), so available power is the "
            "traveling-wave I^2*Z0/2, not the Norton-generator I^2*Z0/8. "
            "'norton' is kept as a selectable alternate convention."
        ),
    )
    # Signal readout frequency. Default: track the pump at a fixed detuning
    # ws = wp - 100 MHz per cell (the physically correct choice for a map that
    # sweeps pump frequency). Pass --signal-ghz to force a fixed absolute signal.
    p.add_argument("--signal-ghz", type=float, default=None,
                   help="Fixed absolute signal frequency (GHz). If omitted, the "
                   "signal tracks each cell's pump at ws = wp - "
                   "--signal-detuning-mhz.")
    p.add_argument("--signal-detuning-mhz", type=float, default=100.0,
                   help="Signal detuning below the pump when --signal-ghz is not "
                   "set: ws = wp - detuning (default 100 MHz).")

    # Per-cell signal spectrum: solve a ladder of signal frequencies around each
    # pump cell (reusing one Floquet conversion base), not just the single
    # trailing point. The trailing gain_db / map_arrays are unchanged; the
    # spectrum is an additive (n_power, n_freq, n_offset) cube in map_spectrum.npz.
    p.add_argument("--signal-spectrum", action=argparse.BooleanOptionalAction, default=True,
                   help="Solve a spectrum of signal frequencies per cell (see below); "
                   "writes map_spectrum.npz. Reuses exp09's khat conversion base so "
                   "each extra signal point is cheap.")
    p.add_argument(
        "--force-single-tone", action="store_true",
        help="Solve and report the pump-only single-tone HB state; skip signal solves.",
    )
    p.add_argument("--signal-offset-start-mhz", type=float, default=100.0,
                   help="First |offset| from fp for the spectrum ladder (MHz).")
    p.add_argument("--signal-offset-step-mhz", type=float, default=500.0, #250
                   help="Spacing between spectrum offsets (MHz).")
    p.add_argument("--signal-offset-count-per-side", type=int, default=5,
                   help="Offsets per side; 5 -> 10 points (+/-100,+/-350,... MHz).")
    p.add_argument("--signal-workers", type=int, default=6,
                   help="Threads over spectrum signal points (1 = serial).")
    p.add_argument("--signal-backend", choices=["direct", "schur"], default="direct",
                   help="Signal linear backend for the gain solve.")
    p.add_argument("--signal-solver", choices=["superlu", "pardiso"], default="superlu",
                   help="Sparse solver for the signal system.")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Skip off/pumpdiag baseline solves (schur backend); gain_db stays valid.")

    p.add_argument("--pump-port", type=int, default=4)
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=2)
    # JTWPA (unbiased 4WM) pump basis: JC odd modes [1,3,...,2K-1], K=10 -> nt>=40.
    p.add_argument("--pump-mode-policy", default="positive_odd_jc")
    p.add_argument(
        "--mixing-order", type=int, choices=(3, 4), default=4,
        help="Parametric mixing order. 3 selects dense pump harmonics and "
             "idler_m=-1; 4 preserves the legacy idler_m=-2 path.",
    )
    p.add_argument("--pump-mode-count", type=int, default=10,
                   help="K for positive_odd_jc -> modes [1,3,...,2K-1]. Set with the basis policy.")
    p.add_argument("--harmonics", type=int, default=3,
                   help="Dense [1..H] harmonics; only used when --pump-mode-count is unset.")
    p.add_argument("--nt", type=int, default=40)
    p.add_argument(
        "--adaptive-harmonics", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "For biased 3WM (enabled by default), promote a converged DC-inclusive pump basis when "
            "the reconstructed residual remains above the harmonic gate. "
            "Each promotion is warm-started and rebuilds the Schur partition. "
            "Use --no-adaptive-harmonics only for diagnostic incomplete-basis runs."
        ),
    )
    p.add_argument(
        "--harmonic-enrichment-max", type=int, default=9,
        help="Maximum positive harmonic used by --adaptive-harmonics.",
    )
    p.add_argument(
        "--harmonic-enrichment-time-rel", type=float, default=1e-4,
        help="Full reconstructed residual target for harmonic enrichment.",
    )
    p.add_argument("--sidebands", type=int, default=6)
    p.add_argument("--gamma-nt", type=int, default=96)
    p.add_argument(
        "--pump-current-jc-scale",
        type=float,
        default=1.0,
        help="Multiply the physical port current by this before injecting (JC "
        "positive-phasor source convention; 2.0 matches JosephsonCircuits).",
    )

    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--newton-tol", type=float, default=1e-9)
    p.add_argument("--linear-seed-maxiter", type=int, default=5)
    p.add_argument("--adaptive-initial-step", type=float, default=1.0)
    p.add_argument("--adaptive-min-step", type=float, default=0.05)

    p.add_argument("--gate-gain-db", type=float, default=0.01)
    p.add_argument(
        "--gate-min-converged-frac",
        type=float,
        default=0.98,
        help="Gate passes only if at least this fraction of warm points converged "
        "(a few stiff points should not invalidate a large map).",
    )
    p.add_argument(
        "--gate-spotcheck",
        type=int,
        default=0,
        help="warmstart mode: recompute N points cold after the warm pass and "
        "fold their gain drift into the gate (corners+center first).",
    )

    p.add_argument("--pump-timeout-s", type=float, default=600.0)
    p.add_argument("--gain-timeout-s", type=float, default=300.0)
    p.add_argument("--python-executable", default=sys.executable)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--compact-output", action=argparse.BooleanOptionalAction, default=False,
        help="Discard ordinary per-point pump_solution.npz after gain evaluation; "
        "continuation keeps its state in memory.",
    )
    p.add_argument(
        "--allow-superlu-fallback",
        action="store_true",
        help="Debug only: allow real_coupled_fast to fall back to SuperLU if PARDISO fails.",
    )
    p.add_argument(
        "--log-factor-backend",
        action="store_true",
        help="Print whether real_coupled_fast actually factors with PARDISO or SuperLU.",
    )
    p.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="DEBUG",
        help="Log verbosity for the complete solver control flow (default: DEBUG).",
    )

    # Production defaults for the standard gain-map workflow.
    p.set_defaults(
    )

    args = p.parse_args(argv)
    if args.high_power_recovery:
        # The high-power profile is an explicit composite workflow.  Keep the
        # lower-level flags visible in the parsed namespace and enable only the
        # recovery ladder it is defined to extend.
        args.column_recovery_ladder = True
    if args.inproc_schur_cache_size is None:
        args.inproc_schur_cache_size = 1 if args.high_power_recovery else 2
    return args


def build_points(args: argparse.Namespace) -> tuple[list[GridPoint], np.ndarray, np.ndarray]:
    logger.debug(
        "build_points_start n_power=%d n_frequency=%d power_range=(%s,%s) "
        "frequency_range=(%s,%s)", args.n_power, args.n_frequency,
        args.pump_power_min_dbm, args.pump_power_max_dbm,
        args.pump_freq_min_ghz, args.pump_freq_max_ghz,
    )
    if args.grid_from_measurement_dir is not None:
        powers, freqs = load_measurement_grid(args.grid_from_measurement_dir)
    else:
        powers = np.linspace(args.pump_power_min_dbm, args.pump_power_max_dbm, args.n_power)
        freqs = np.linspace(args.pump_freq_min_ghz, args.pump_freq_max_ghz, args.n_frequency)
    points: list[GridPoint] = []
    index = 0
    for i, power_dbm in enumerate(powers):
        for j, freq in enumerate(freqs):
            current = dbm_to_peak_current_a(
                float(power_dbm),
                attenuation_db=attenuation_db_for(float(freq), args),
                z0_ohm=args.z0_ohm,
                convention=args.power_convention,
            )
            points.append(GridPoint(index, i, j, float(power_dbm), float(freq), current))
            index += 1
    logger.debug("build_points_complete n_points=%d", len(points))
    return points, powers, freqs


def select_spotcheck_points(points: list[GridPoint], n: int) -> list[GridPoint]:
    if n <= 0 or not points:
        return []
    n_power = max(p.i_power for p in points) + 1
    n_freq = max(p.j_freq for p in points) + 1
    by_ij = {(p.i_power, p.j_freq): p for p in points}
    priority = [
        (0, 0), (n_power - 1, 0), (0, n_freq - 1), (n_power - 1, n_freq - 1),
        (n_power // 2, n_freq // 2),
    ]
    chosen: list[GridPoint] = []
    seen: set[int] = set()
    for key in priority:
        pt = by_ij.get(key)
        if pt is not None and pt.index not in seen:
            chosen.append(pt)
            seen.add(pt.index)
    # Fill remaining slots with an even stride over the flattened grid.
    if len(chosen) < n:
        stride = max(1, len(points) // n)
        for pt in points[::stride]:
            if pt.index not in seen:
                chosen.append(pt)
                seen.add(pt.index)
            if len(chosen) >= n:
                break
    return chosen[:n]


_CHUNK_STRIP_VALUE_OPTS = {
    "--outdir",
    "--frequency-index-start",
    "--frequency-index-stop",
    "--frequency-chunk-size",
    "--n-frequency",
    "--pump-freq-min-ghz",
    "--pump-freq-max-ghz",
    "--gate-spotcheck",
}
_CHUNK_STRIP_FLAGS = {
    "--chunk-worker",
    "--overwrite",
    "--resume-chunks",
    "--no-resume-chunks",
}


def frequency_chunk_ranges(n_frequency: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return half-open frequency-column chunks."""
    n = int(n_frequency)
    size = int(chunk_size)
    if n <= 0:
        return []
    if size <= 0 or size >= n:
        return [(0, n)]
    return [(start, min(start + size, n)) for start in range(0, n, size)]


def _strip_chunk_driver_args(argv: list[str]) -> list[str]:
    """Remove parent/chunk-routing options before spawning a chunk worker."""
    cleaned: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in _CHUNK_STRIP_FLAGS:
            continue
        if token in _CHUNK_STRIP_VALUE_OPTS:
            skip_next = True
            continue
        if any(token.startswith(f"{opt}=") for opt in _CHUNK_STRIP_VALUE_OPTS):
            continue
        cleaned.append(token)
    return cleaned


def chunk_worker_command(
    base_argv: list[str],
    *,
    outdir: Path,
    n_frequency: int,
    pump_freq_min_ghz: float,
    pump_freq_max_ghz: float,
    overwrite: bool = False,
) -> list[str]:
    """Build the self-invocation used for one frequency-column chunk."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *_strip_chunk_driver_args(base_argv),
        "--chunk-worker",
        "--gate-spotcheck",
        "0",
        "--n-frequency",
        str(int(n_frequency)),
        "--pump-freq-min-ghz",
        f"{float(pump_freq_min_ghz):.12g}",
        "--pump-freq-max-ghz",
        f"{float(pump_freq_max_ghz):.12g}",
        "--outdir",
        str(outdir),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _expected_chunk_row_count(args: argparse.Namespace, start_col: int, stop_col: int) -> int:
    n_cols = max(0, int(stop_col) - int(start_col))
    pass_count = 2 if args.mode == "both" else 1
    return int(args.n_power) * n_cols * pass_count


def chunk_is_complete(
    chunk_dir: Path,
    args: argparse.Namespace,
    start_col: int,
    stop_col: int,
) -> bool:
    points_path = chunk_dir / "map_points.csv"
    summary_path = chunk_dir / "map_summary.json"
    if not points_path.exists() or not summary_path.exists():
        return False
    if args.signal_spectrum and args.mode in ("warmstart", "both") and not (chunk_dir / "map_spectrum.npz").exists():
        return False
    try:
        with points_path.open("r", encoding="utf-8", newline="") as f:
            n_rows = max(0, sum(1 for _ in f) - 1)
    except OSError:
        return False
    return n_rows == _expected_chunk_row_count(args, start_col, stop_col)


def read_chunk_rows(
    chunk_specs: list[tuple[Path, int, int]],
    *,
    global_n_frequency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cold_rows: list[dict[str, Any]] = []
    warm_rows: list[dict[str, Any]] = []
    for chunk_dir, start_col, _stop_col in chunk_specs:
        points_path = chunk_dir / "map_points.csv"
        with points_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cleaned = {k: _csv_value(v) for k, v in row.items()}
                pass_name = str(cleaned.pop("pass", ""))
                if "j_freq" in cleaned and "i_power" in cleaned:
                    global_j = int(start_col) + int(cleaned["j_freq"])
                    cleaned["j_freq"] = global_j
                    cleaned["point_index"] = int(cleaned["i_power"]) * int(global_n_frequency) + global_j
                if pass_name == "cold":
                    cold_rows.append(cleaned)
                elif pass_name == "warm":
                    warm_rows.append(cleaned)
    cold_rows.sort(key=lambda r: int(r["point_index"]))
    warm_rows.sort(key=lambda r: int(r["point_index"]))
    return cold_rows, warm_rows


def _csv_value(value: Any) -> Any:
    if value == "":
        return None
    if not isinstance(value, str):
        return value
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def merge_chunk_spectra(
    outpath: Path,
    chunk_specs: list[tuple[Path, int, int]],
    powers: np.ndarray,
    freqs: np.ndarray,
) -> bool:
    """Merge full-shape per-chunk spectrum cubes into one canonical NPZ."""
    merged: np.ndarray | None = None
    offsets: np.ndarray | None = None
    for chunk_dir, start_col, stop_col in chunk_specs:
        path = chunk_dir / "map_spectrum.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            cube = np.asarray(data["gain_spectrum_db"], dtype=float)
            if merged is None:
                merged = np.full((powers.size, freqs.size, cube.shape[2]), np.nan, dtype=float)
                offsets = np.asarray(data["signal_offset_mhz"], dtype=float)
            if cube.shape[1] == freqs.size:
                mask = np.isfinite(cube)
                merged[mask] = cube[mask]
            else:
                expected_cols = int(stop_col) - int(start_col)
                if cube.shape[1] != expected_cols:
                    raise ValueError(
                        f"chunk {chunk_dir} spectrum has {cube.shape[1]} frequency columns; "
                        f"expected {expected_cols}"
                    )
                merged[:, start_col:stop_col, :] = cube
    if merged is None or offsets is None:
        return False
    signal_ghz = freqs[None, :] + offsets[:, None] / 1000.0
    np.savez(
        outpath,
        pump_power_dbm=powers,
        pump_frequency_ghz=freqs,
        signal_offset_mhz=offsets,
        gain_spectrum_db=merged,
        signal_ghz=signal_ghz,
    )
    return True


def run_frequency_chunks(
    args: argparse.Namespace,
    raw_argv: list[str],
    outdir: Path,
    freqs: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[Path, int, int]]]:
    """Run the map in fresh 10-column worker processes and merge their rows."""
    ranges = frequency_chunk_ranges(args.n_frequency, args.frequency_chunk_size)
    chunk_root = outdir / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_specs: list[tuple[Path, int, int]] = []
    pending: list[tuple[int, Path, int, int, list[str], Path]] = []
    for chunk_index, (start_col, stop_col) in enumerate(ranges):
        chunk_dir = chunk_root / f"chunk_{chunk_index:03d}_cols_{start_col:03d}_{stop_col - 1:03d}"
        chunk_specs.append((chunk_dir, start_col, stop_col))
        fp0 = float(freqs[start_col])
        fp1 = float(freqs[stop_col - 1])
        if args.resume_chunks and chunk_is_complete(chunk_dir, args, start_col, stop_col):
            print(f"\n=== chunk {chunk_index} : already complete, skipping ===", flush=True)
            continue
        cmd = chunk_worker_command(
            raw_argv,
            outdir=chunk_dir,
            n_frequency=stop_col - start_col,
            pump_freq_min_ghz=fp0,
            pump_freq_max_ghz=fp1,
            overwrite="--overwrite" in raw_argv,
        )
        log_path = outdir / f"chunk_{chunk_index:03d}.log"
        pending.append((chunk_index, chunk_dir, start_col, stop_col, cmd, log_path))

    def run_chunk(item: tuple[int, Path, int, int, list[str], Path]) -> tuple[int, int, float, Path]:
        chunk_index, _chunk_dir, _start, _stop, cmd, log_path = item
        t0 = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        return chunk_index, int(proc.returncode), time.perf_counter() - t0, log_path

    results = run_isolated_jobs(pending, run_chunk, args.frequency_workers)
    for chunk_index, rc, elapsed, log_path in sorted(results):
        print(f"chunk {chunk_index} rc={rc} elapsed={elapsed:.1f}s log={log_path}", flush=True)
        if rc != 0:
            raise RuntimeError(f"frequency chunk {chunk_index} failed with return code {rc}")
    cold_rows, warm_rows = read_chunk_rows(chunk_specs, global_n_frequency=args.n_frequency)
    return cold_rows, warm_rows, chunk_specs


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(raw_argv)
    frequency_chunk_size_explicit = any(
        token == "--frequency-chunk-size"
        or token.startswith("--frequency-chunk-size=")
        for token in raw_argv
    )
    # Resolve roles and mixing order before building points or spawning chunk
    # workers.  InProcessEngine repeats this defensively for direct callers.
    runtime_circuit = load_circuit(args.circuit_dir)
    roles = resolve_port_roles(
        runtime_circuit,
        pump_port=args.pump_port,
        source_port=args.source_port,
        out_port=args.out_port,
    )
    for role, port in roles.items():
        setattr(args, role, port)
    args.mixing_order = resolve_mixing_order(
        args.mixing_order,
        dc_current_a=args.dc_current_a,
        dc_branch_flux_over_phi0=args.dc_branch_flux_over_phi0,
        dc_solution=args.dc_solution,
        design_meta=runtime_circuit.summary,
    )
    if args.frequency_workers < 1:
        raise ValueError("--frequency-workers must be >= 1")
    if (
        args.frequency_workers > 1
        and args.frequency_chunk_size > 0
        and not frequency_chunk_size_explicit
    ):
        args.frequency_chunk_size = max(
            1, int(np.ceil(args.n_frequency / args.frequency_workers))
        )
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    logger.debug("main_start argv=%r", raw_argv)

    if args.allow_superlu_fallback:
        os.environ.pop("TWPA_REQUIRE_PARDISO", None)
    else:
        os.environ["TWPA_REQUIRE_PARDISO"] = "1"

    if args.log_factor_backend:
        os.environ["TWPA_PARDISO_LOG"] = "1"
    outdir = args.outdir

    # Frequency-crossing traversals share one in-process solved-state store, so
    # they cannot be split across chunk worker processes. Force a single process
    # and widen the per-frequency Schur cache so a backbone row does not thrash.
    if (
        args.traversal != "column"
        and not args.chunk_worker
        and not args.local_traversal_chunks
    ):
        if int(args.frequency_chunk_size) > 0:
            print(f"traversal={args.traversal}: forcing --frequency-chunk-size 0 "
                  "(single process required for cross-column warm state)", flush=True)
            args.frequency_chunk_size = 0
        # NB: keep the Schur cache small (bounded RAM). A freq-crossing backbone
        # row rebuilds the per-frequency partition as it sweeps, but caching all
        # n_frequency partitions would OOM (~16 GB at 50 columns).

    points, powers, freqs = build_points(args)
    if args.grid_from_measurement_dir is not None:
        # Explicit grids are kept in one process.  The chunk subprocess CLI
        # represents frequency columns by a local linspace, which would lose
        # the nonuniform measurement axis we just loaded.
        args.n_power = int(powers.size)
        args.n_frequency = int(freqs.size)
        args.pump_power_min_dbm = float(powers[0])
        args.pump_power_max_dbm = float(powers[-1])
        args.pump_freq_min_ghz = float(freqs[0])
        args.pump_freq_max_ghz = float(freqs[-1])
        if args.frequency_chunk_size > 0:
            logger.info(
                "explicit measurement grid: disabling frequency chunk subprocesses"
            )
            args.frequency_chunk_size = 0

    use_chunk_driver = (
        not args.chunk_worker
        and args.executor == "inprocess"
        and int(args.frequency_chunk_size) > 0
        and int(args.n_frequency) > int(args.frequency_chunk_size)
    )
    logger.debug(
        "main_execution_plan executor=%s mode=%s traversal=%s chunk_driver=%s "
        "n_power=%s n_frequency=%s",
        args.executor, args.mode, args.traversal, use_chunk_driver,
        args.n_power, args.n_frequency,
    )
    if outdir.exists() and args.overwrite and not use_chunk_driver:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.debug(
        "main_grid_built n_points=%d power_range=(%s,%s) frequency_range=(%s,%s)",
        len(points), powers[0] if powers.size else None,
        powers[-1] if powers.size else None, freqs[0] if freqs.size else None,
        freqs[-1] if freqs.size else None,
    )

    if args.fold_follow:
        if args.executor != "inprocess":
            raise SystemExit("--fold-follow requires --executor inprocess")
        engine = InProcessEngine(args)
        logger.debug("main_fold_follow_dispatch n_frequencies=%d", len(freqs))
        run_fold_follow(engine, freqs, outdir, args)
        return 0

    if args.frequency_index_start is not None or args.frequency_index_stop is not None:
        start_col = 0 if args.frequency_index_start is None else int(args.frequency_index_start)
        stop_col = args.n_frequency if args.frequency_index_stop is None else int(args.frequency_index_stop)
        if start_col < 0 or stop_col > args.n_frequency or start_col >= stop_col:
            raise ValueError(
                f"invalid frequency chunk [{start_col}, {stop_col}) for "
                f"n_frequency={args.n_frequency}"
            )
        points = [p for p in points if start_col <= p.j_freq < stop_col]
        logger.debug(
            "main_frequency_slice start_col=%d stop_col=%d n_points=%d",
            start_col, stop_col, len(points),
        )
    start = time.perf_counter()

    cold_rows: list[dict[str, Any]] = []
    warm_rows: list[dict[str, Any]] = []
    chunk_specs: list[tuple[Path, int, int]] = []

    if use_chunk_driver:
        print(
            f"executor={args.executor} frequency_chunk_size={args.frequency_chunk_size}",
            flush=True,
        )
        cold_rows, warm_rows, chunk_specs = run_frequency_chunks(args, raw_argv, outdir, freqs)
        logger.debug(
            "main_chunk_driver_complete cold_rows=%d warm_rows=%d chunks=%d",
            len(cold_rows), len(warm_rows), len(chunk_specs),
        )
    else:
        engine = InProcessEngine(args) if args.executor == "inprocess" else None
        cold_pass = (lambda pts, d: run_cold_pass_inprocess(pts, d, engine)) if engine else \
            (lambda pts, d: run_cold_pass(pts, d, args))
        use_traversal_orchestrator = uses_traversal_orchestrator(args)
        if engine and use_traversal_orchestrator:
            warm_pass = lambda pts, d: run_map_traversal(pts, d, engine)
        elif engine:
            warm_pass = lambda pts, d: run_warm_pass_inprocess(pts, d, engine, fail_fast=args.inproc_fail_fast)
        else:
            warm_pass = lambda pts, d: run_warm_pass(pts, d, args)
        print(f"executor={args.executor} traversal={args.traversal}", flush=True)
        logger.debug(
            "main_pass_dispatch cold=%s warm=%s traversal_orchestrator=%s",
            args.mode in ("cold", "both"), args.mode in ("warmstart", "both"),
            use_traversal_orchestrator,
        )

        if args.mode in ("cold", "both"):
            cold_rows = cold_pass(points, outdir / "cold")
        if args.mode in ("warmstart", "both"):
            warm_rows = warm_pass(points, outdir / "warm")

    # Spot-check cold recompute for a warm-only run.
    if args.mode == "warmstart" and args.gate_spotcheck > 0:
        if "cold_pass" not in locals():
            engine = InProcessEngine(args) if args.executor == "inprocess" else None
            cold_pass = (lambda pts, d: run_cold_pass_inprocess(pts, d, engine)) if engine else \
                (lambda pts, d: run_cold_pass(pts, d, args))
        spot = select_spotcheck_points(points, args.gate_spotcheck)
        print(f"spot-checking {len(spot)} point(s) cold for the gate", flush=True)
        cold_rows = cold_pass(spot, outdir / "cold_spotcheck")
        logger.debug("main_spotcheck_complete n_points=%d", len(cold_rows))

    if args.mode == "both" or (args.mode == "warmstart" and cold_rows):
        gate = evaluate_gate(
            cold_rows,
            warm_rows,
            gate_gain_db=args.gate_gain_db,
            min_converged_frac=args.gate_min_converged_frac,
        )
    else:
        gate = GateResult(evaluated=False, passed=False, reasons=["gate not applicable for this mode"])
    logger.debug(
        "main_gate_result evaluated=%s passed=%s reasons=%r",
        gate.evaluated, gate.passed, gate.reasons,
    )

    # Persist tagged rows and grids.
    tagged: list[dict[str, Any]] = []
    for r in cold_rows:
        tagged.append({"pass": "cold", **r})
    for r in warm_rows:
        tagged.append({"pass": "warm", **r})
    write_points_csv(outdir / "map_points.csv", tagged)
    logger.debug("main_points_written path=%s n_rows=%d", outdir / "map_points.csv", len(tagged))

    grids: dict[str, np.ndarray] = {}
    if cold_rows and args.mode in ("cold", "both"):
        grids["gain_db_cold"] = gain_grid(cold_rows, args.n_power, args.n_frequency)
    if warm_rows:
        grids["gain_db_warm"] = gain_grid(warm_rows, args.n_power, args.n_frequency)
    if "gain_db_cold" in grids and "gain_db_warm" in grids:
        grids["gain_drift_db"] = np.abs(grids["gain_db_warm"] - grids["gain_db_cold"])
    write_arrays(outdir / "map_arrays.npz", powers, freqs, grids)
    logger.debug("main_arrays_written path=%s grid_names=%r", outdir / "map_arrays.npz", list(grids))

    if args.signal_spectrum and warm_rows:
        offsets = spectrum_offsets_mhz(args)
        merged = merge_chunk_spectra(outdir / "map_spectrum.npz", chunk_specs, powers, freqs) if chunk_specs else False
        if not merged:
            write_spectrum(outdir / "map_spectrum.npz", warm_rows, powers, freqs, offsets)
        print(f"wrote {outdir / 'map_spectrum.npz'} "
              f"({len(offsets)} signal offsets/cell)", flush=True)

    elapsed = time.perf_counter() - start
    write_summary(outdir, args, cold_rows, warm_rows, gate, elapsed)
    logger.debug("main_summary_written path=%s elapsed_s=%.6f", outdir / "map_summary.json", elapsed)

    print("", flush=True)
    if gate.evaluated:
        print(f"GATE={'PASS' if gate.passed else 'FAIL'}", flush=True)
        print(f"warm_converged_frac={gate.warm_converged_frac} (failed={gate.n_warm_failed})", flush=True)
        print(f"max_gain_drift_db={gate.max_gain_drift_db}", flush=True)
        print(f"pump_speedup_per_point={gate.pump_speedup}", flush=True)
        if gate.reasons:
            print(f"gate_reasons={'; '.join(gate.reasons)}", flush=True)
    print(f"wrote {outdir / 'map_summary.json'}", flush=True)
    print(f"elapsed_s={elapsed:.3f}", flush=True)

    if gate.evaluated and not gate.passed:
        logger.debug("main_end return_code=1")
        return 1
    logger.debug("main_end return_code=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
