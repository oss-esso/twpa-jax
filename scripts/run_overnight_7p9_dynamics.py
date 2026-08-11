"""Run the specification-driven independent 7.9 GHz TD campaign.

The driver is deliberately an experiment orchestrator.  Every primary target
starts from the zero-pump equilibrium in a fresh child process.  Refinement,
deep diagnostics, initialization controls, ramp controls, and timestep checks
are separate experiments and never feed a state into another primary target.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .run_td_integrator_screen import process_rss_bytes
except ImportError:  # direct script execution
    from run_td_integrator_screen import process_rss_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / ".hybrid_outputs" / "hb_up_7p9_m35_to_m21" / "pass" / "points"
    / "point_0010_p_m24p4737dbm_fp_7p9ghz" / "pump"
)
DEFAULT_HB_POINTS = DEFAULT_CHECKPOINT.parent.parent
COARSE_POWERS = [
    -35.0, -33.0, -31.0, -29.0, -27.0, -26.0, -25.0,
    -24.4736842105, -24.0, -23.4210526316, -23.0, -22.5, -22.0,
    -21.0, -20.0, -19.0, -18.0, -17.0, -16.0, -15.0,
]
HOLD_CHECKPOINTS = (40, 90, 140, 250, 440)


def power_label(power_dbm: float) -> str:
    return (f"p_{power_dbm:+.6f}dbm".replace("+", "p").replace("-", "m").replace(".", "p"))


def load_reference(checkpoint: Path, freq_ghz: float, pump_port: int) -> tuple[float, float]:
    report = json.loads((checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    if report.get("final_status") != "VALID_CONVERGED":
        raise ValueError(f"reference checkpoint is not validated: {checkpoint}")
    metadata = report.get("metadata", {})
    report_freq = metadata.get("pump_frequency_ghz", metadata.get("pump_freq_ghz"))
    if report_freq is not None and abs(float(report_freq) - freq_ghz) > 1e-9:
        raise ValueError(f"reference frequency is {report_freq}, expected {freq_ghz}")
    report_port = metadata.get("pump_port")
    if report_port is not None and int(report_port) != pump_port:
        raise ValueError(f"reference pump port is {report_port}, expected {pump_port}")
    return float(metadata["pump_power_dbm_requested"]), float(metadata["pump_current_a"])


def regime(summary: dict[str, Any]) -> str:
    """Apply the decay-aware policy before interpreting the raw classifier."""
    integrator = summary.get("integrator") or {}
    if not integrator.get("success", False):
        return "TRANSIENT_NUMERICAL_FAILURE"
    decay = str((summary.get("decay_aware") or {}).get("class", ""))
    raw = str(summary.get("classification", ""))
    if decay in {"UNRESOLVED_SLOW_RELAXATION", "RELAXING_TO_PERIOD1"}:
        return "UNRESOLVED_LONG_TRANSIENT"
    if decay == "PERIOD_1" or raw == "PERIOD_1":
        return "PERIOD1"
    if raw == "RUNNING_PHASE":
        return "RUNNING_PHASE"
    if raw in {"PERIOD_2", "PERIOD_3"}:
        return raw
    if raw == "QUASIPERIODIC_OR_PERIOD_N":
        return "PERIOD_N_OR_QUASIPERIODIC"
    if raw == "BROADBAND_OR_CHAOTIC" or decay == "PERSISTENT_NONPERIODIC":
        return "BROADBAND_NONPERIODIC"
    return "UNRESOLVED_LONG_TRANSIENT"


def read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_child(
    args: Any,
    *,
    power_dbm: float,
    target_dir: Path,
    hold_periods: int,
    reference_power: float,
    reference_current: float,
    checkpoint: Path,
    initialization_mode: str = "zero_pump_equilibrium",
    ramp_periods: int | None = None,
    max_step: float | None = None,
    transient_restart: Path | None = None,
) -> dict[str, Any]:
    target_current = reference_current * 10.0 ** ((power_dbm - reference_power) / 20.0)
    target_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-B", str(ROOT / "scripts" / "h1_transient_branch_transfer.py"),
        "--circuit-dir", str(args.circuit_dir), "--checkpoint", str(checkpoint),
        "--outdir", str(target_dir), "--freq-ghz", str(args.freq_ghz),
        "--pump-port", str(args.pump_port), "--target-current-a", repr(target_current),
        "--ramp-periods", str(args.ramp_periods if ramp_periods is None else ramp_periods),
        "--hold-periods", str(hold_periods), "--method", "implicit_trapezoid",
        "--max-step", str(args.delta_theta if max_step is None else max_step),
        "--atol", str(args.atol), "--max-newton", str(args.max_newton),
        "--checkpoint-periods", "10", "--compact-output",
        "--compact-sample-count", "256", "--compact-history-states", "1024",
        "--skip-projection",
    ]
    if transient_restart is not None:
        command += ["--transient-restart", str(transient_restart)]
    else:
        command += ["--initialization-mode", initialization_mode]
    stdout_path = target_dir / "stdout.log"
    stderr_path = target_dir / "stderr.log"
    monitor_path = target_dir / "memory_monitor.jsonl"
    started = time.perf_counter()
    peak_rss = 0
    memory_exceeded = False
    timed_out = False
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
        monitor_path.open("w", encoding="utf-8") as monitor,
    ):
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
        deadline = time.perf_counter() + args.point_timeout_s
        while process.poll() is None:
            rss = process_rss_bytes(process.pid)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
                monitor.write(json.dumps({"elapsed_s": time.perf_counter() - started, "rss_bytes": rss}) + "\n")
                monitor.flush()
                if rss > args.memory_limit_gb * 1024**3:
                    memory_exceeded = True
                    process.terminate()
                    break
            if time.perf_counter() > deadline:
                timed_out = True
                process.terminate()
                break
            time.sleep(2.0)
        try:
            return_code = process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
    summary_path = target_dir / "summary.json"
    summary = read_summary(summary_path)
    record: dict[str, Any] = {
        "experiment_type": "independent_zero_pump_upward_turn_on",
        "target_power_dbm": power_dbm,
        "target_current_a": target_current,
        "actual_source_current_a": (summary.get("source_telemetry") or {}).get("target_source_current_a", target_current),
        "initialization_source": "zero_pump_equilibrium_q0_p0" if transient_restart is None and initialization_mode == "zero_pump_equilibrium" else initialization_mode,
        "previous_target_restart_used": False,
        "same_target_restart_used": False,
        "ramp_periods": args.ramp_periods if ramp_periods is None else ramp_periods,
        "hold_periods": hold_periods,
        "delta_theta": args.delta_theta if max_step is None else max_step,
        "checkpoint": str(checkpoint),
        "return_code": return_code,
        "runtime_s": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "memory_limit_exceeded": memory_exceeded,
        "timed_out": timed_out,
        "summary_path": str(summary_path),
        "artifact_dir": str(target_dir),
        "classification": summary.get("classification"),
        "decay_aware": summary.get("decay_aware"),
        "regime": regime(summary) if summary else "TRANSIENT_NUMERICAL_FAILURE",
        "integrator": summary.get("integrator"),
        "stroboscopic": summary.get("stroboscopic"),
        "checkpoint_diagnostics": summary.get("checkpoint_diagnostics", []),
        "source_telemetry": summary.get("source_telemetry"),
        "mean_phase_winding_cycles": summary.get("mean_phase_winding_cycles"),
        "spectral_artifact": str(target_dir / "late_time_spectrum.npz"),
    }
    if transient_restart is not None:
        record["initialization_source"] = "same_target_td_restart"
        record["same_target_restart_used"] = True
    (target_dir / "campaign_record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def run_target(
    args: Any, power_dbm: float, phase_dir: Path, reference_power: float,
    reference_current: float, *, hold_periods: int = 440,
    checkpoint: Path | None = None, initialization_mode: str = "zero_pump_equilibrium",
    ramp_periods: int | None = None, max_step: float | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    target_root = phase_dir / (name or power_label(power_dbm))
    record = run_child(
        args, power_dbm=power_dbm, target_dir=target_root,
        hold_periods=hold_periods, reference_power=reference_power,
        reference_current=reference_current, checkpoint=checkpoint or args.checkpoint,
        initialization_mode=initialization_mode, ramp_periods=ramp_periods,
        max_step=max_step,
    )
    return {"power_dbm": power_dbm, "final": record, "stages": [record]}


def run_primary_target(
    args: Any, power_dbm: float, phase_dir: Path, reference_power: float,
    reference_current: float,
) -> dict[str, Any]:
    """Run one independent target with same-target adaptive hold extensions."""
    target_root = phase_dir / power_label(power_dbm)
    stages: list[dict[str, Any]] = []
    previous_total = 0
    restart: Path | None = None
    for checkpoint in HOLD_CHECKPOINTS:
        if checkpoint > args.primary_hold_periods:
            break
        local_hold = checkpoint - previous_total
        stage_dir = target_root / f"hold_{checkpoint:04d}"
        stage = run_child(
            args, power_dbm=power_dbm, target_dir=stage_dir,
            hold_periods=local_hold, reference_power=reference_power,
            reference_current=reference_current, checkpoint=args.checkpoint,
            ramp_periods=args.ramp_periods if restart is None else 0,
            transient_restart=restart,
        )
        stage["stage_hold_periods"] = local_hold
        stage["period_offset"] = previous_total
        stage["total_hold_periods"] = checkpoint
        stage["hold_periods"] = checkpoint
        stage["ramp_end_periods"] = args.ramp_periods
        stages.append(stage)
        previous_total = checkpoint
        summary = read_summary(Path(stage["summary_path"]))
        decay_class = str((summary.get("decay_aware") or {}).get("class", ""))
        robust_period1 = stage.get("regime") == "PERIOD1" and decay_class == "PERIOD_1"
        robust_running = stage.get("regime") == "RUNNING_PHASE" and abs(float(stage.get("mean_phase_winding_cycles") or 0.0)) > 0.1
        if robust_period1 or robust_running:
            break
        candidate = stage_dir / "restart_checkpoints" / "transient_restart.npz"
        if not candidate.exists():
            break
        restart = candidate
    final = stages[-1]
    checkpoint_observations = [
        {
            "hold_periods": stage["total_hold_periods"],
            "classification": stage.get("classification"),
            "decay_aware": stage.get("decay_aware"),
            "artifact_dir": stage["artifact_dir"],
        }
        for stage in stages
    ]
    return {
        "power_dbm": power_dbm, "final": final, "stages": stages,
        "checkpoint_observations": checkpoint_observations,
    }


def append_summary(path: Path, campaign: dict[str, Any]) -> None:
    path.write_text(json.dumps(campaign, indent=2, default=str), encoding="utf-8")


def all_primary_records(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    records = list(campaign.get("coarse", [])) + list(campaign.get("boundary_refinement", []))
    return sorted(records, key=lambda item: float(item["power_dbm"]))


def unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[float, dict[str, Any]] = {}
    for item in records:
        result[round(float(item["power_dbm"]), 8)] = item
    return list(result.values())


def first_boundary(records: list[dict[str, Any]]) -> tuple[float, float] | None:
    ordered = unique_records(sorted(records, key=lambda item: float(item["power_dbm"])))
    candidates = []
    for left, right in zip(ordered, ordered[1:]):
        lreg, rreg = left["final"].get("regime"), right["final"].get("regime")
        if (lreg == "PERIOD1") != (rreg == "PERIOD1"):
            candidates.append((float(left["power_dbm"]), float(right["power_dbm"])))
    return candidates[0] if candidates else None


def representative_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = unique_records(sorted(records, key=lambda item: float(item["power_dbm"])))
    boundary = first_boundary(ordered)
    if boundary is None:
        return []
    lower, upper = boundary
    below = [item for item in ordered if float(item["power_dbm"]) <= lower and item["final"].get("regime") == "PERIOD1"]
    above = [item for item in ordered if float(item["power_dbm"]) >= upper and item["final"].get("regime") != "PERIOD1"]
    if not below or not above:
        return []
    deeper = [item for item in above if float(item["power_dbm"]) > float(above[0]["power_dbm"]) + 1e-8]
    return [below[-1], above[0], deeper[0] if deeper else above[0]]


def read_hb_points(root: Path) -> list[tuple[float, Path]]:
    points: list[tuple[float, Path]] = []
    if not root.exists():
        return points
    for report_path in root.glob("point_*/pump/pump_report.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("final_status") != "VALID_CONVERGED":
                continue
            metadata = report.get("metadata", {})
            points.append((float(metadata["pump_power_dbm_requested"]), report_path.parent))
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return sorted(points)


def control_checkpoint(points: list[tuple[float, Path]], power: float, exact: bool) -> Path | None:
    matches = [item for item in points if abs(item[0] - power) <= 1e-5]
    if matches:
        return matches[0][1]
    if exact:
        return None
    lower = [item for item in points if item[0] < power]
    return lower[-1][1] if lower else None


def run_plot_package(campaign_dir: Path) -> None:
    command = [sys.executable, "-B", str(ROOT / "scripts" / "plot_overnight_7p9_dynamics.py"), "--campaign-dir", str(campaign_dir)]
    subprocess.run(command, cwd=ROOT, check=False)


def choose_extension(campaign: dict[str, Any]) -> tuple[str, str]:
    critical = campaign.get("deep_classification", [])
    final_runs = [
        item["final"]
        for key in ("coarse", "boundary_refinement", "deep_classification", "initialization_controls", "ramp_rate", "timestep_validation")
        for item in campaign.get(key, [])
        if "final" in item
    ]
    unresolved = [item for item in final_runs if item.get("regime") in {"UNRESOLVED_LONG_TRANSIENT", "TRANSIENT_NUMERICAL_FAILURE"}]
    first = next((item.get("final", {}) for item in critical if item.get("final", {}).get("regime") != "PERIOD1"), {})
    raw = first.get("classification")
    if unresolved:
        return "INSUFFICIENT_EVIDENCE", "At least one critical or map run remained unresolved."
    if raw in {"PERIOD_2", "PERIOD_3"}:
        n = raw.split("_")[-1]
        return "PERIOD_N_HB", f"The first deep non-PERIOD1 point was classified as PERIOD_{n}; the d{n} closure and spectral evidence must be reviewed in the deep artifacts."
    if raw == "QUASIPERIODIC_OR_PERIOD_N":
        return "TWO_FREQUENCY_HB", "The first bounded non-periodic point was not assigned a persistent low-order period; inspect its discrete spectrum and Poincare curve."
    if raw == "BROADBAND_OR_CHAOTIC":
        return "FINITE_PERIOD_HB_NOT_APPROPRIATE", "The first deep transition was broadband/non-periodic rather than a confirmed finite period."
    if raw == "RUNNING_PHASE":
        return "FINITE_PERIOD_HB_NOT_APPROPRIATE", "Running phase was observed before a confirmed bounded periodic replacement."
    return "NO_EXTENSION_YET", "No specific nonlinear HB ansatz was demonstrated by the available diagnostics."


def representative_analysis(record: dict[str, Any]) -> dict[str, Any]:
    summary = read_summary(Path(record["summary_path"]))
    strobe = summary.get("stroboscopic") or {}
    tail = strobe.get("tail_median_by_n") or {}
    finite_tail = [
        (int(key[1:]), float(value))
        for key, value in tail.items()
        if np.isfinite(float(value))
    ]
    best_n, best_dn = min(finite_tail, key=lambda item: item[1]) if finite_tail else (None, None)
    peaks: list[tuple[float, float]] = []
    spectrum_path = Path(record.get("spectral_artifact", ""))
    if spectrum_path.exists():
        with np.load(spectrum_path) as spectrum:
            frequency = np.asarray(spectrum["frequency_ghz"], dtype=float)
            amplitude = np.asarray(spectrum["amplitude"], dtype=float)
        if amplitude.size > 1:
            candidates = np.argsort(amplitude[1:])[-5:] + 1
            peaks = sorted(
                [(float(frequency[index]), float(amplitude[index])) for index in candidates],
                key=lambda item: item[1], reverse=True,
            )
    poincare_count = min(len(strobe.get("pump_flux", [])), len(strobe.get("state_norm", [])))
    return {
        "power_dbm": float(record["target_power_dbm"]),
        "regime": record.get("regime"),
        "tail": {f"d{n}": value for n, value in finite_tail},
        "best_n": best_n,
        "best_dn": best_dn,
        "spectrum_peaks_ghz": [frequency for frequency, _ in peaks[:3]],
        "poincare_points": poincare_count,
    }


def write_report(campaign: dict[str, Any], path: Path) -> None:
    records = unique_records(all_primary_records(campaign))
    period1 = [item for item in records if item["final"].get("regime") == "PERIOD1"]
    nonperiodic = [item for item in records if item["final"].get("regime") != "PERIOD1"]
    boundary = first_boundary(records)
    extension, reason = choose_extension(campaign)
    unresolved = [item for item in campaign.get("simulations", []) if item.get("regime") in {"UNRESOLVED_LONG_TRANSIENT", "TRANSIENT_NUMERICAL_FAILURE"}]
    deep_analysis = [
        representative_analysis(item["final"])
        for item in campaign.get("deep_classification", [])
        if "final" in item
    ]
    running = [item for item in records if item["final"].get("regime") == "RUNNING_PHASE"]
    running_threshold = min((float(item["power_dbm"]) for item in running), default=float("nan"))
    direct_controls = [
        item["final"] for item in campaign.get("initialization_controls", [])
        if "direct_hb" in str(item.get("artifact_dir", ""))
    ]
    warm_controls = [
        item["final"] for item in campaign.get("initialization_controls", [])
        if "warm_lower_hb" in str(item.get("artifact_dir", ""))
    ]
    basin_differences = [
        item for item in campaign.get("initialization_controls", [])
        if "final" in item and item["final"].get("regime") != next(
            (base["final"].get("regime") for base in records
             if abs(float(base["power_dbm"]) - float(item["power_dbm"])) < 1e-7),
            item["final"].get("regime"),
        )
    ]
    ramp_groups: dict[float, set[str]] = {}
    for item in campaign.get("ramp_rate", []):
        if "final" in item:
            ramp_groups.setdefault(float(item["power_dbm"]), set()).add(str(item["final"].get("regime")))
    timestep_matches: list[tuple[float, bool]] = []
    for item in campaign.get("timestep_validation", []):
        if "final" not in item:
            continue
        base = next(
            (base["final"] for base in records
             if abs(float(base["power_dbm"]) - float(item["power_dbm"])) < 1e-7),
            None,
        )
        timestep_matches.append((float(item["power_dbm"]), base is not None and item["final"].get("regime") == base.get("regime")))
    lines = [
        "# Overnight 7.9 GHz 2c dynamical-regime campaign",
        "",
        f"Total wall-clock time: {campaign.get('finished_unix', campaign.get('started_unix', 0)) - campaign.get('started_unix', 0):.1f} s.",
        f"Simulations: {len(campaign.get('simulations', []))}.",
        "",
        "## Primary independent map",
        "",
        "| Power (dBm) | Regime | Raw | Decay-aware | Hold | Source current (A) |",
        "|---:|---|---|---|---:|---:|",
    ]
    for item in records:
        final = item["final"]
        lines.append(f"| {float(item['power_dbm']):+.6f} | {final.get('regime')} | {final.get('classification')} | {(final.get('decay_aware') or {}).get('class')} | {final.get('hold_periods')} | {float(final.get('actual_source_current_a', final.get('target_current_a', 0.0))):.6e} |")
    lines += [
        "", "## Boundary and interpretation", "",
        f"Highest independently ramp-selected PERIOD1 point: {max((float(item['power_dbm']) for item in period1), default=float('nan')):+.6f} dBm.",
        f"Lowest independently selected non-PERIOD1 point: {min((float(item['power_dbm']) for item in nonperiodic), default=float('nan')):+.6f} dBm.",
        f"First PERIOD1-loss bracket: {boundary if boundary else 'not established'}.",
        f"HB-extension decision: **{extension}**.",
        f"Rationale: {reason}",
        f"Running-phase threshold (separate from PERIOD1 loss): {running_threshold:+.6f} dBm." if np.isfinite(running_threshold) else "Running-phase threshold (separate from PERIOD1 loss): not observed.",
        "",
        "The primary map uses zero-pump equilibrium and a fresh upward ramp at every target. Descending runs, direct HB controls, and warm-HB controls are not used to define the primary boundary.",
        "",
        "## Representative dN, spectrum, and Poincare diagnostics",
        "",
        "| Power (dBm) | Regime | d1 | d2 | d3 | Best dN | Poincare points | Dominant frequencies (GHz) |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in deep_analysis:
        best = f"d{item['best_n']}={item['best_dn']:.3e}" if item["best_n"] is not None else "not available"
        lines.append(
            f"| {item['power_dbm']:+.6f} | {item['regime']} | "
            f"{item['tail'].get('d1', float('nan')):.3e} | {item['tail'].get('d2', float('nan')):.3e} | "
            f"{item['tail'].get('d3', float('nan')):.3e} | {best} | {item['poincare_points']} | "
            f"{', '.join(f'{freq:.4f}' for freq in item['spectrum_peaks_ghz']) or 'not available'} |"
        )
    lines += [
        "",
        "The dN values are late-window medians, not full-history maxima. A PERIOD_N interpretation requires persistent closure, repeatable Poincare clusters, and matching subharmonic spectral structure; the table is evidence for review, not an automatic PERIOD_N confirmation.",
        "",
        "## History and numerical controls",
        "",
        f"Direct-HB controls: {len(direct_controls)}; warm-lower-HB controls: {len(warm_controls)}; basin-dependent classification differences: {len(basin_differences)}.",
        f"Ramp-rate control regime sets: {', '.join(f'{power:+.4f} dBm={sorted(regimes)}' for power, regimes in sorted(ramp_groups.items())) or 'not completed'}.",
        f"Timestep Δθ=0.025 classification matches the primary run: {sum(match for _, match in timestep_matches)}/{len(timestep_matches)}.",
        "",
        "The Fourier artifacts contain the late-time spectrum and the deep plots mark fp, fp/2, fp/3, fp/4, and low-frequency components. The Poincare artifacts contain the pump-flux/state-norm stroboscopic projection.",
        "",
        "## Controls and unresolved runs",
        "",
        f"Initialization controls: {len(campaign.get('initialization_controls', []))} records.",
        f"Ramp-rate controls: {len(campaign.get('ramp_rate', []))} records.",
        f"Timestep controls: {len(campaign.get('timestep_validation', []))} records.",
        f"Unresolved runs: {len(unresolved)}.",
    ]
    for item in unresolved:
        lines.append(f"- {item.get('experiment_type')} at {item.get('target_power_dbm', item.get('power_dbm'))} dBm: {item.get('artifact_dir')}")
    lines += [
        "", "## Evidence categories", "",
        "- Facts directly demonstrated: independent target restarts, source-current provenance, recurrence histories, late spectra, Poincare projections, winding telemetry, and integrator/resource telemetry are persisted per run.",
        "- Plausible interpretation: the extension decision above, subject to visual inspection of the deep plots.",
        "- Unresolved questions: any unresolved runs listed above and any classification that changes in the timestep or initialization controls.",
        "", "Key plots are under `plots/`, with one summary figure per primary target and deep dN/spectrum/Poincare figures for representative states.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    campaign["hb_extension_decision"] = {"choice": extension, "reason": reason}
    campaign["unresolved_runs"] = unresolved
    campaign["report_path"] = str(path)


def parse_args() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hb-points-root", type=Path, default=DEFAULT_HB_POINTS)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--ramp-periods", type=int, default=40)
    parser.add_argument("--delta-theta", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max-newton", type=int, default=12)
    parser.add_argument("--memory-limit-gb", type=float, default=1.0)
    parser.add_argument("--point-timeout-s", type=float, default=2400.0)
    parser.add_argument("--deadline-hours", type=float, default=7.5)
    parser.add_argument("--refinement-spacing-dbm", type=float, default=0.2)
    parser.add_argument("--validation-hold-periods", type=int, default=250)
    parser.add_argument(
        "--powers-dbm", type=float, nargs="+", default=None,
        help="Override the coarse map powers for a controlled smoke run; the default is the full specification range.",
    )
    parser.add_argument(
        "--primary-hold-periods", type=int, default=440,
        help="Fixed-drive hold for primary map points; retain 440 for the full campaign.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in ("coarse", "boundary_refinement", "deep_classification", "initialization_controls", "ramp_rate", "timestep_validation", "plots"):
        (args.outdir / name).mkdir(exist_ok=True)
    reference_power, reference_current = load_reference(args.checkpoint, args.freq_ghz, args.pump_port)
    coarse_powers = list(args.powers_dbm) if args.powers_dbm is not None else COARSE_POWERS
    deadline = time.perf_counter() + args.deadline_hours * 3600.0
    campaign: dict[str, Any] = {
        "protocol": "independent_zero_pump_equilibrium_upward_turn_on",
        "specification": "docs/development/overnight_td_spec.md",
        "circuit_dir": str(args.circuit_dir), "freq_ghz": args.freq_ghz,
        "pump_port": args.pump_port, "reference_checkpoint": str(args.checkpoint),
        "reference_power_dbm": reference_power, "reference_current_a": reference_current,
        "attenuation_override": None, "ramp_periods": args.ramp_periods,
        "delta_theta": args.delta_theta, "hold_checkpoints": HOLD_CHECKPOINTS,
        "primary_hold_periods": args.primary_hold_periods,
        "coarse_powers_dbm": coarse_powers, "coarse": [], "boundary_refinement": [],
        "deep_classification": [], "initialization_controls": [], "ramp_rate": [],
        "timestep_validation": [], "simulations": [], "started_unix": time.time(),
    }
    summary_path = args.outdir / "campaign_summary.json"

    def save() -> None:
        simulations: list[dict[str, Any]] = []
        for key in ("coarse", "boundary_refinement", "deep_classification", "initialization_controls", "ramp_rate", "timestep_validation"):
            for item in campaign[key]:
                simulations.extend(item.get("stages", [item["final"]]) if "final" in item else [])
        campaign["simulations"] = simulations
        append_summary(summary_path, campaign)

    for power in coarse_powers:
        if time.perf_counter() >= deadline:
            break
        print(f"[coarse] starting {power:+.6f} dBm", flush=True)
        result = run_primary_target(args, power, args.outdir / "coarse", reference_power, reference_current)
        campaign["coarse"].append(result); save()

    coarse_records = campaign["coarse"]
    transitions = []
    ordered = unique_records(sorted(coarse_records, key=lambda item: float(item["power_dbm"])))
    for left, right in zip(ordered, ordered[1:]):
        if left["final"].get("regime") != right["final"].get("regime"):
            transitions.append((float(left["power_dbm"]), float(right["power_dbm"])))
    for lower, upper in transitions:
        spacing = max(float(args.refinement_spacing_dbm), 0.1)
        points = [lower + spacing * index for index in range(1, int((upper - lower) / spacing) + 1) if lower + spacing * index < upper - 1e-8]
        for power in points:
            if time.perf_counter() >= deadline:
                break
            if any(abs(float(item["power_dbm"]) - power) < 1e-7 for item in all_primary_records(campaign)):
                continue
            print(f"[refine] starting {power:+.6f} dBm", flush=True)
            result = run_primary_target(args, power, args.outdir / "boundary_refinement", reference_power, reference_current)
            campaign["boundary_refinement"].append(result); save()

    reps = representative_records(all_primary_records(campaign))
    hb_points = read_hb_points(args.hb_points_root)
    for index, item in enumerate(reps):
        if time.perf_counter() >= deadline:
            break
        power = float(item["power_dbm"])
        deep = run_target(args, power, args.outdir / "deep_classification", reference_power, reference_current, name=f"{index:02d}_{power_label(power)}")
        campaign["deep_classification"].append(deep); save()
        exact = control_checkpoint(hb_points, power, True)
        lower = control_checkpoint(hb_points, power, False)
        if exact is not None:
            direct = run_target(args, power, args.outdir / "initialization_controls", reference_power, reference_current, hold_periods=args.validation_hold_periods, checkpoint=exact, initialization_mode="hb_periodic", ramp_periods=0, name=f"{index:02d}_{power_label(power)}_direct_hb")
            campaign["initialization_controls"].append(direct); save()
        if lower is not None:
            warm = run_target(args, power, args.outdir / "initialization_controls", reference_power, reference_current, hold_periods=args.validation_hold_periods, checkpoint=lower, initialization_mode="hb_periodic", ramp_periods=args.ramp_periods, name=f"{index:02d}_{power_label(power)}_warm_lower_hb")
            campaign["initialization_controls"].append(warm); save()

    boundary = first_boundary(all_primary_records(campaign))
    if boundary is not None:
        for power in boundary:
            for ramp in (20, 40, 80):
                if time.perf_counter() >= deadline:
                    break
                if ramp == 40:
                    primary = next((item for item in campaign["coarse"] + campaign["boundary_refinement"] if abs(float(item["power_dbm"]) - power) < 1e-7), None)
                    if primary is not None:
                        campaign["ramp_rate"].append({"power_dbm": power, "ramp_periods": ramp, "reused_primary": True, "final": primary["final"]}); save(); continue
                control = run_target(args, power, args.outdir / "ramp_rate", reference_power, reference_current, hold_periods=args.validation_hold_periods, ramp_periods=ramp, name=f"{power_label(power)}_ramp_{ramp:03d}")
                campaign["ramp_rate"].append(control); save()

    for index, item in enumerate(reps):
        if time.perf_counter() >= deadline:
            break
        power = float(item["power_dbm"])
        control = run_target(args, power, args.outdir / "timestep_validation", reference_power, reference_current, hold_periods=args.validation_hold_periods, max_step=0.025, name=f"{index:02d}_{power_label(power)}_dt_0p025")
        campaign["timestep_validation"].append(control); save()

    campaign["finished_unix"] = time.time()
    campaign["deadline_reached"] = time.perf_counter() >= deadline
    save()
    run_plot_package(args.outdir)
    write_report(campaign, args.outdir / "morning_report.md")
    append_summary(summary_path, campaign)
    print(json.dumps({"campaign_summary": str(summary_path), "morning_report": str(args.outdir / 'morning_report.md'), "simulations": len(campaign["simulations"])}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
