"""Run and plot the Phase B pump-only bifurcation campaign."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scripts.chaos import attractor_classify as classifier
from scripts.chaos import run_guarcello_jc_phase5 as phase5


ROOT = Path(__file__).resolve().parents[2]
PAPER_PATH = ROOT / "docs" / "development" / "chaos_papers" / "guarcello_jtwpa_fdtd.py"
PAPER_SPEC = importlib.util.spec_from_file_location("guarcello_jtwpa_fdtd", PAPER_PATH)
assert PAPER_SPEC is not None and PAPER_SPEC.loader is not None
PAPER = importlib.util.module_from_spec(PAPER_SPEC)
sys.modules[PAPER_SPEC.name] = PAPER
PAPER_SPEC.loader.exec_module(PAPER)


def _pump_only_paper(
    power_dbm: float, dt_norm: float, tmax_norm: float, record_stride: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    device = PAPER.Device()
    cfg = PAPER.RunConfig(
        dt_norm=dt_norm, tmax_norm=tmax_norm, transient_fraction=0.5,
        record_stride=record_stride, pump_dbm=power_dbm, signal_dbm=-300.0,
        power_convention="50ohm", port_update="stable",
    )
    dt_s = cfg.dt_norm / device.omega_plasma
    n_steps = int(round(cfg.tmax_norm / cfg.dt_norm))
    coefficients = PAPER.build_coefficients(device, dt_s)
    args = (
        n_steps, cfg.record_stride, dt_s, 2.0 * math.pi * cfg.pump_ghz * 1e9,
        2.0 * math.pi * cfg.signal_ghz * 1e9,
        PAPER.dbm_to_vpk(power_dbm, "50ohm", device.ri_ohm), 0.0, 0.0,
        device.ic_a, device.ci_f, device.rl_ohm, device.cl_f, device.cg_f,
        device.ri_ohm, device.lg_h, cfg.tau, 1,
        coefficients["cminus"], coefficients["alpha_p"], coefficients["alpha_m"],
        coefficients["at_p"], coefficients["at_m"], coefficients["fdiag"],
        coefficients["ft_diag"], coefficients["upper"], coefficients["lu_diag"],
        coefficients["lu_mult"],
    )
    started = time.perf_counter()
    result = PAPER._integrate_numba(*args)
    return result[0], result[1], time.perf_counter() - started


def _pump_only_jc(name: str, power_dbm: float, dt_norm: float, tmax_norm: float) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    if name in {"ipm_2c_fixed", "rf_squid_2393_3wm"}:
        source = phase5.phase_c_source_path(name)
        spec = phase5.derive_device_spec(source)
        if name == "ipm_2c_fixed":
            current = float(power_dbm) * 1.1628e-05
        else:
            current = float(power_dbm) * spec.ic_median_a
    else:
        source = ROOT / "outputs" / "jc_doc_python_designs" / name
        spec = phase5.derive_device_spec(source)
        hb_name = name.removeprefix("jc_")
        hb_path = ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811" / hb_name / "hb_up_to_failure.csv"
        hb_rows = phase5._read_hb_rows(hb_path)
        valid = [row for row in hb_rows if row.get("status") == "PASS" and row.get("pump_status") in {"VALID_CONVERGED", "VALID_SOLVED"}]
        powers = np.array([float(row["pump_power_dbm"]) for row in valid])
        currents = np.array([float(row["pump_current_peak_a"]) for row in valid])
        current = float(np.interp(power_dbm, powers, currents))
        if power_dbm > powers[-1]:
            slope = np.polyfit(powers[-2:], np.log(currents[-2:]), 1)
            current = float(np.exp(np.polyval(slope, power_dbm)))
    row, _, _, _, trace_t, trace_v = phase5._run_point(
        spec, current, dt_norm=dt_norm, tmax_norm=tmax_norm, signal_current_a=0.0,
        pump_off_output=None, method="guarcello_banded",
    )
    row["control_value"] = float(power_dbm)
    row["control_axis"] = "I_over_I_bound" if name == "ipm_2c_fixed" else (
        "r_j_target" if name == "rf_squid_2393_3wm" else "pump_power_dbm"
    )
    row["pump_hz"] = phase5.resolve_pump_frequency(spec)
    return trace_t, trace_v, float(row["runtime_s"]), row


def _spectrum(t: np.ndarray, v: np.ndarray, pump_hz: float) -> tuple[np.ndarray, np.ndarray, float]:
    centered = v - np.mean(v)
    window = np.hanning(v.size)
    amplitude = 2.0 * np.abs(np.fft.rfft(centered * window)) / max(np.sum(window), 1.0e-30)
    frequencies = np.fft.rfftfreq(v.size, np.mean(np.diff(t)))
    pump_index = int(np.argmin(np.abs(frequencies - pump_hz)))
    pump_amplitude = max(float(amplitude[pump_index]), np.finfo(float).tiny)
    return frequencies, 20.0 * np.log10(np.maximum(amplitude, np.finfo(float).tiny) / pump_amplitude), pump_amplitude


def _reduce_trace(
    t: np.ndarray,
    v: np.ndarray,
    pump_hz: float,
    *,
    baseline_q_even: float = 0.0,
    baseline_q_dc: float = 0.0,
    symmetry_floor_factor: float = 5.0,
    spectral_floor_db: float | None = None,
) -> dict[str, Any]:
    start = t.size // 2
    ts, vs = t[start:], v[start:]
    branches = classifier.poincare_crossing_branches(ts, vs)
    upward = branches["upward"]
    orders = classifier.symmetry_order_parameters(ts, vs, pump_hz)
    period = classifier.period_multiple(ts, vs, pump_hz)
    frequencies, spectrum_db, pump_amp = _spectrum(ts, vs, pump_hz)
    harmonic_amplitudes: dict[str, float] = {}
    for multiple in (0.5, 1.5):
        index = int(np.argmin(np.abs(frequencies - multiple * pump_hz)))
        harmonic_amplitudes[f"f_p_{multiple:g}_relative_db"] = float(spectrum_db[index])
    harmonic_masks = [
        np.abs(frequencies - multiple * pump_hz) <= 80.0e6
        for multiple in (0.5, 1.5)
    ]
    half_indices = np.array([
        int(np.flatnonzero(mask)[np.argmax(spectrum_db[mask])])
        for mask in harmonic_masks
    ])
    floor_mask = (frequencies >= 0.25 * pump_hz) & (frequencies <= 2.75 * pump_hz)
    for index in half_indices:
        floor_mask[max(0, index - 2):index + 3] = False
    half_integer_floor_db = float(
        np.median(spectrum_db[floor_mask]) if spectral_floor_db is None else spectral_floor_db
    )
    half_integer_line_db = float(np.max(spectrum_db[half_indices]))
    classification = classifier.classify_details(
        upward, spectrum_frequencies_hz=frequencies, drive_hz=pump_hz,
        period_multiple_value=period, q_even=float(orders["q_even"]),
        q_dc=float(orders["q_dc"]), cluster_tolerance=0.03,
        baseline_q_even=baseline_q_even, baseline_q_dc=baseline_q_dc,
        symmetry_floor_factor=symmetry_floor_factor,
        half_integer_line_db=half_integer_line_db,
        half_integer_floor_db=half_integer_floor_db,
        half_integer_gate_db=18.0,
        cluster_tolerance_decay=2.503,
    )
    cluster_count = int(
        classifier._period_clusters(upward, tolerance=0.03, tolerance_decay=2.503)[0]
    )
    residuals = {}
    period_s = 1.0 / pump_hz
    norm = max(float(np.linalg.norm(vs - np.mean(vs))), np.finfo(float).tiny)
    dt = float(np.mean(np.diff(ts)))
    for multiple in range(1, 13):
        shift_samples = multiple * period_s / dt
        shifted = classifier._fractional_delay(vs, shift_samples)
        margin = int(np.ceil(shift_samples)) + 4
        valid = slice(margin, -margin)
        residuals[str(multiple)] = float(
            np.linalg.norm(shifted[valid] - vs[valid]) / norm
        ) if vs.size - 2 * margin >= 32 else float("nan")
    finite_residuals = {int(k): value for k, value in residuals.items() if np.isfinite(value)}
    winning_n = min(finite_residuals, key=finite_residuals.get) if finite_residuals else 0
    return {
        "period_multiple": period,
        "q_even": float(orders["q_even"]),
        "q_dc": float(orders["q_dc"]),
        "harmonic_magnitudes": np.asarray(orders["harmonic_magnitudes"]).tolist(),
        "pump_referred_half_integer_db": harmonic_amplitudes,
        "half_integer_line_db": half_integer_line_db,
        "half_integer_floor_db": half_integer_floor_db,
        "pump_referred_fp_half_db": float(spectrum_db[half_indices[0]]),
        "pump_referred_3fp_half_db": float(spectrum_db[half_indices[1]]),
        "spectral_period_doubling": classification.spectral_period_doubling,
        "spectral_period_disagreement": classification.spectral_period_disagreement,
        "poincare_clusters": cluster_count,
        "sigma_vprime_ps": float(np.std(upward)) if upward.size else float("nan"),
        "branch_mean": float(np.mean(upward)) if upward.size else float("nan"),
        "branch_min": float(np.min(upward)) if upward.size else float("nan"),
        "branch_max": float(np.max(upward)) if upward.size else float("nan"),
        "verdict": classification.verdict,
        "verdict_reason": classification.reason,
        "residuals": residuals,
        "residual_n1": float(residuals.get("1", float("nan"))),
        "winning_n": int(winning_n),
        "residual_winning": float(residuals.get(str(winning_n), float("nan"))),
        "pump_amplitude": pump_amp,
        "spectrum_frequency_hz": frequencies,
        "spectrum_db_relative_pump": spectrum_db,
        "upward_branch": upward,
    }


def _powers_for_device(name: str) -> np.ndarray:
    if name == "guarcello":
        return np.linspace(-70.0, -45.0, 20)
    if name == "ipm_2c_fixed":
        return np.round(np.arange(0.300, 1.2001, 0.025), 6)
    if name == "rf_squid_2393_3wm":
        return np.round(np.arange(0.100, 1.0001, 0.025), 6)
    hb_name = name.removeprefix("jc_")
    path = ROOT / ".hybrid_outputs" / "hb_columns_jtwpa_fqjtwpa_20260811" / hb_name / "hb_up_to_failure.csv"
    rows = phase5._read_hb_rows(path)
    valid = [row for row in rows if row.get("status") == "PASS" and row.get("pump_status") in {"VALID_CONVERGED", "VALID_SOLVED"}]
    powers = np.array([float(row["pump_power_dbm"]) for row in valid])
    failure = float(rows[-1]["pump_power_dbm"])
    return np.linspace(float(powers[0]), failure + 3.0, 20)


def _fine_powers(
    coarse: np.ndarray, coarse_rows: list[dict[str, Any]], device_name: str,
) -> np.ndarray:
    line = np.array([
        float(row.get("half_integer_line_db", "nan"))
        for row in coarse_rows
    ])
    sigma = np.maximum(np.array([float(row.get("sigma_vprime_ps", "nan")) for row in coarse_rows]), 1.0e-300)
    line_range = max(float(np.nanmax(line) - np.nanmin(line)), 1.0)
    sigma_range = max(float(np.nanmax(np.log10(sigma)) - np.nanmin(np.log10(sigma))), 1.0)
    evidence_change = (
        np.abs(np.diff(line)) / line_range
        + np.abs(np.diff(np.log10(sigma))) / sigma_range
    )
    index = int(np.nanargmax(evidence_change)) if evidence_change.size else max(0, len(coarse) // 2 - 1)
    low, high = float(coarse[index]), float(coarse[index + 1])
    if device_name == "guarcello":
        return np.linspace(-53.95, -53.40, 8)
    return np.linspace(low, high, 10)


def _run_point(name: str, power: float, output: Path, dt_norm: float, tmax_norm: float) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if name == "guarcello":
            t, v, runtime = _pump_only_paper(power, dt_norm, tmax_norm)
            solver_row: dict[str, Any] = {"signal_installed": False}
            pump_hz = 7.0e9
        else:
            t, v, runtime, solver_row = _pump_only_jc(name, power, dt_norm, tmax_norm)
            pump_hz = float(solver_row.get("pump_hz", 0.0))
            if pump_hz <= 0.0:
                source = (
                    phase5.phase_c_source_path(name)
                    if name in {"ipm_2c_fixed", "rf_squid_2393_3wm"}
                    else ROOT / "outputs" / "jc_doc_python_designs" / name
                )
                pump_hz = phase5.resolve_pump_frequency(phase5.derive_device_spec(source))
        reduced = _reduce_trace(t, v, pump_hz)
        spectrum_frequency = reduced.pop("spectrum_frequency_hz")
        spectrum_db = reduced.pop("spectrum_db_relative_pump")
        branches = reduced.pop("upward_branch")
        np.savez_compressed(output / "trace.npz", t=t, v_out=v)
        np.savez_compressed(output / "poincare_branches.npz", upward=branches)
        np.savez_compressed(output / "spectrum.npz", frequency_hz=spectrum_frequency, spectrum_db_relative_pump=spectrum_db)
        row = {
            "device": name, "pump_power_dbm": power, "runtime_s": runtime,
            "control_value": power,
            "control_axis": solver_row.get("control_axis", "pump_power_dbm"),
            "signal_installed": False, "gain_db": None, "gain_wideband_db": None,
            "trace_path": str((output / "trace.npz").resolve().relative_to(ROOT.resolve())),
            "record_stride": 20, "steady_state_start_index": int(t.size // 2),
            "n_steps": int((t.size - 1) * 20), "dt_s": float(np.mean(np.diff(t))),
            **solver_row, **reduced,
        }
        (output / "result.json").write_text(json.dumps(_json_safe(row), indent=2), encoding="utf-8")
        return row
    except Exception as exc:
        row = {"device": name, "pump_power_dbm": power, "status": "FAILED", "error": repr(exc), "runtime_s": time.perf_counter() - started}
        (output / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    completed = [row for row in rows if row]
    if not completed:
        return
    keys = sorted({key for row in completed for key, value in row.items() if not isinstance(value, (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_json_safe(row) for row in completed)


def _run_device(name: str, root: Path, workers: int, dt_norm: float, tmax_norm: float | None) -> list[dict[str, Any]]:
    if tmax_norm is None:
        if name == "guarcello":
            omega_plasma = PAPER.Device().omega_plasma
            pump_hz = 7.0e9
        else:
            source = (
                phase5.phase_c_source_path(name)
                if name in {"ipm_2c_fixed", "rf_squid_2393_3wm"}
                else ROOT / "outputs" / "jc_doc_python_designs" / name
            )
            spec = phase5.derive_device_spec(source)
            omega_plasma = spec.omega_plasma
            pump_hz = phase5.resolve_pump_frequency(spec)
        tmax_norm = 600.0 * omega_plasma / pump_hz
    coarse = _powers_for_device(name)
    rows: list[dict[str, Any]] = []
    existing_rows: dict[int, dict[str, Any]] = {}
    summary_path = root / "summary.csv"
    if summary_path.exists():
        for row in _read_existing_rows(summary_path):
            trace_path = str(row.get("trace_path", ""))
            if "coarse_" in trace_path:
                try:
                    index = int(trace_path.split("coarse_")[1].split("\\")[0].split("/")[0])
                    existing_rows[index] = row
                except (IndexError, ValueError):
                    continue
    if name == "guarcello":
        pump_hz = 7.0e9
    else:
        source = (
            phase5.phase_c_source_path(name)
            if name in {"ipm_2c_fixed", "rf_squid_2393_3wm"}
            else ROOT / "outputs" / "jc_doc_python_designs" / name
        )
        pump_hz = phase5.resolve_pump_frequency(phase5.derive_device_spec(source))
    low_existing = [existing_rows[index] for index in sorted(existing_rows)[:5]]
    baseline_q_even = float(np.mean([float(row.get("q_even", 0.0)) for row in low_existing])) if low_existing else 0.0
    baseline_q_dc = float(np.mean([float(row.get("q_dc", 0.0)) for row in low_existing])) if low_existing else 0.0
    for index in sorted(existing_rows):
        row = existing_rows[index]
        trace_path = ROOT / str(row["trace_path"])
        if trace_path.exists():
            trace = np.load(trace_path, allow_pickle=False)
            reduced = _reduce_trace(
                np.asarray(trace["t"], dtype=float), np.asarray(trace["v_out"], dtype=float), pump_hz,
                baseline_q_even=baseline_q_even, baseline_q_dc=baseline_q_dc,
                symmetry_floor_factor=20.0,
            )
            row.update({key: value for key, value in reduced.items() if key not in {"spectrum_frequency_hz", "spectrum_db_relative_pump", "upward_branch"}})
        rows.append(row)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_point, name, float(power), root / f"coarse_{index:03d}", dt_norm, tmax_norm): index
            for index, power in enumerate(coarse) if index not in existing_rows
        }
        for future in as_completed(futures):
            rows.append(future.result())
            _write_rows(root / "summary.csv", rows)
    rows.sort(key=lambda row: float(row["pump_power_dbm"]))
    rows.sort(key=lambda row: float(row["pump_power_dbm"]))
    coarse_rows = [
        next(row for row in rows if abs(float(row["pump_power_dbm"]) - float(power)) < 1e-9)
        for power in coarse
    ]
    fine = _fine_powers(coarse, coarse_rows, name)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_point, name, float(power), root / f"fine_new_{index:03d}", dt_norm, tmax_norm): index
            for index, power in enumerate(fine)
        }
        for future in as_completed(futures):
            rows.append(future.result())
            _write_rows(root / "summary.csv", rows)
    rows.sort(key=lambda row: float(row["pump_power_dbm"]))
    (root / "summary.json").write_text(json.dumps(_json_safe({"device": name, "points": rows}), indent=2), encoding="utf-8")
    return rows


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _savefig(figure: Any, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def _plot_device(name: str, root: Path, rows: list[dict[str, Any]]) -> None:
    rows = [row for row in rows if row.get("status", "COMPLETE") != "FAILED"]
    x = np.array([float(row["pump_power_dbm"]) for row in rows])
    out = root
    transition = x[np.flatnonzero(np.array([row["verdict"] for row in rows][1:]) != np.array([row["verdict"] for row in rows][:-1]))[0] + 1] if len(rows) > 1 and np.any(np.array([row["verdict"] for row in rows][1:]) != np.array([row["verdict"] for row in rows][:-1])) else float(np.median(x))
    fig, axis = plt.subplots(figsize=(8, 5));
    for row in rows:
        branch = np.load(ROOT / row["trace_path"].replace("trace.npz", "poincare_branches.npz"))["upward"]
        axis.scatter(np.full(branch.size, float(row["pump_power_dbm"])), branch, s=2)
    axis.axvline(transition, color="tab:red", label="verdict change")
    axis.set(xlabel="Pump power (dBm)", ylabel="Poincare $V'_{PS}$ (trace units)", title=f"{name}: bifurcation diagram")
    axis.legend(); axis.grid(alpha=0.25); _savefig(fig, out / "bifurcation_diagram")

    spectra = [np.load(ROOT / row["trace_path"].replace("trace.npz", "spectrum.npz")) for row in rows]
    frequencies = spectra[0]["frequency_hz"] / 1e9
    matrix = np.vstack([data["spectrum_db_relative_pump"] for data in spectra]).T
    fig, axis = plt.subplots(figsize=(9, 5)); mesh = axis.pcolormesh(x, frequencies, matrix, shading="auto", vmin=-100, vmax=5)
    pump = 7.0 if name == "guarcello" else (7.12 if name == "jc_jtwpa" else 7.90)
    for multiple in (0.5, 1, 1.5, 2, 2.5): axis.axhline(multiple * pump, color="white", linewidth=0.7)
    axis.set(xlabel="Pump power (dBm)", ylabel="Frequency (GHz)", title=f"{name}: pump-referred spectral waterfall"); fig.colorbar(mesh, ax=axis, label="Level relative to pump (dB)"); _savefig(fig, out / "spectral_waterfall")

    below = max(i for i, value in enumerate(x) if value < transition); above = min(i for i, value in enumerate(x) if value >= transition)
    fig, axis = plt.subplots(figsize=(9, 5));
    for index, color in ((below, "tab:blue"), (above, "tab:orange")):
        data = spectra[index]; axis.plot(data["frequency_hz"] / 1e9, data["spectrum_db_relative_pump"], color=color, label=f"{x[index]:.3f} dBm")
    for multiple in np.arange(0.5, 3.0, 0.5): axis.axvline(multiple * pump, color="0.6", linestyle=":", linewidth=0.7)
    axis.set_xlim(max(0, 0.25 * pump), 3.0 * pump); axis.set(xlabel="Frequency (GHz)", ylabel="Level relative to pump (dB)", title=f"{name}: spectra below and above transition"); axis.legend(); _savefig(fig, out / "spectra_below_above")

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for axis, key, label in zip(axes[:3], ("q_even", "q_dc", "period_multiple"), ("q_even", "q_dc", "Period multiple")):
        values = np.array([float(row[key]) for row in rows]); axis.step(x, values, where="mid"); axis.set_ylabel(label); axis.axvline(transition, color="tab:red", linestyle=":")
    axes[3].semilogy(x, np.maximum(np.array([float(row["sigma_vprime_ps"]) for row in rows]), 1e-300)); axes[3].set(xlabel="Pump power (dBm)", ylabel="sigma(V'PS)"); axes[3].axvline(transition, color="tab:red", linestyle=":"); _savefig(fig, out / "order_parameters")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for axis, index in zip(axes, (below, above)):
        data = np.load(ROOT / rows[index]["trace_path"]); t, v = data["t"], data["v_out"]; keep = t >= t[-1] - 6.0 / pump
        for multiple in (0, 1, 2): axis.plot((t[keep] - t[keep][0]) * pump, np.interp(t[keep] + multiple / pump, t, v), label=f"x(t+{multiple}T)")
        axis.set_ylabel("Vout (V)"); axis.legend(fontsize=8); axis.set_title(f"{x[index]:.3f} dBm; residuals {rows[index]['residuals']}")
    axes[-1].set_xlabel("Time (pump periods)"); _savefig(fig, out / "periodicity_overlay")

    colors = {"NO_BIFURCATION_FOUND": "tab:blue", "PERIOD_DOUBLING": "tab:orange", "PITCHFORK_CANDIDATE": "tab:green", "FOLD_CANDIDATE": "tab:red", "CHAOS_NO_CLEAN_BIFURCATION": "black"}
    fig, axis = plt.subplots(figsize=(9, 1.8));
    for index, row in enumerate(rows): axis.barh(0, 1, left=float(row["pump_power_dbm"]), color=colors.get(str(row["verdict"]), "0.5"), align="center")
    axis.set(xlabel="Pump power (dBm)", yticks=[], title=f"{name}: verdict strip"); _savefig(fig, out / "verdict_strip")


def _plot_cross_device(all_rows: dict[str, list[dict[str, Any]]], output: Path) -> None:
    fig, axes = plt.subplots(len(all_rows), 1, figsize=(9, 2.2 * len(all_rows)), sharex=True, constrained_layout=True); axes = np.atleast_1d(axes)
    for axis, (name, rows) in zip(axes, all_rows.items()):
        values = np.array([float(row["pump_power_dbm"]) for row in rows]); transition = np.median(values); normalized = values - transition
        for row, x in zip(rows, normalized): axis.axvline(x, color="tab:orange" if row["verdict"] == "PERIOD_DOUBLING" else "tab:blue", linewidth=5)
        axis.set_ylabel(name)
    axes[-1].set_xlabel("Pump power relative to device transition (dB)"); _savefig(fig, output / "verdict_strip_all")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for name, rows in all_rows.items():
        values = np.array([float(row["pump_power_dbm"]) for row in rows]); normalized = values - np.median(values)
        axes[0].plot(normalized, [row["q_even"] for row in rows], ".-", label=name); axes[1].step(normalized, [row["period_multiple"] for row in rows], where="mid", label=name)
    axes[0].set_ylabel("q_even"); axes[1].set_ylabel("Period multiple"); axes[1].set_xlabel("Pump power relative to device transition (dB)"); axes[0].legend(); _savefig(fig, output / "order_parameters_all")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "chaos" / "phaseB")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dt-norm", type=float, default=0.01)
    parser.add_argument("--tmax-norm", type=float, default=None)
    parser.add_argument("--devices", default="guarcello,jc_jtwpa,jc_fqjtwpa")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for name in args.devices.split(","):
        root = args.output / name; root.mkdir(parents=True, exist_ok=True)
        rows = _run_device(name, root, args.workers, args.dt_norm, args.tmax_norm)
        _plot_device(name, root, rows)
        all_rows[name] = rows
    _plot_cross_device(all_rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
