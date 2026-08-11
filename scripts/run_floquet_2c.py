"""Run matrix-free Floquet analysis on validated 2c PERIOD1 HB checkpoints.

This CLI is intentionally a thin experiment wrapper.  The tangent map and
Arnoldi implementation live under ``twpa_solver.stability``.  It accepts only
validated HB checkpoints; failed ordinary-HB points are recorded as an
obstruction and are never silently interpreted as a physical bifurcation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.h1_transient_branch_transfer import (  # noqa: E402
    build_system,
    implicit_trapezoid_ramp,
)
from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.stability import (  # noqa: E402
    build_hb_periodic_orbit,
    build_monodromy_operator,
    classify_multiplier,
    compute_floquet_multipliers,
    floquet_exponents,
    track_multiplier_branches,
)
from twpa_solver.stability.monodromy import load_hb_periodic_orbit  # noqa: E402


def _checkpoint_power(path: Path, report: dict[str, Any]) -> float:
    metadata = report.get("metadata", {})
    value = metadata.get("pump_power_dbm_requested")
    if value is not None:
        return float(value)
    token = path.parent.name
    return float("nan") if "p_" not in token else float("nan")


def _discover_checkpoints(root: Path) -> list[Path]:
    result: list[Path] = []
    for pump in sorted(root.glob("point_*/pump")):
        report_path = pump / "pump_report.json"
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            continue
        if report.get("final_status") == "VALID_CONVERGED":
            result.append(pump)
    if not result:
        raise FileNotFoundError(f"no VALID_CONVERGED HB checkpoints under {root}")
    return result


def _discover_checkpoint_union(roots: list[Path]) -> list[Path]:
    """Discover validated checkpoints from baseline and recovered roots."""
    unique: dict[str, Path] = {}
    for root in roots:
        for checkpoint in _discover_checkpoints(root):
            unique[str(checkpoint.resolve())] = checkpoint
    return list(unique.values())


def _dc_flux(report: dict[str, Any], branch_count: int) -> np.ndarray:
    metadata = report.get("metadata", {})
    value = metadata.get("dc_branch_flux", metadata.get("dc_branch_flux_wb", 0.0))
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(branch_count, float(array[0]))
    if array.size != branch_count:
        raise ValueError("checkpoint DC flux has an incompatible branch count")
    return array


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.Process().memory_info().rss / (1024.0**2))


def _closure_error(system: Any, orbit_state: np.ndarray, current: float, step: float) -> dict[str, Any]:
    _, states, integrator = implicit_trapezoid_ramp(
        system,
        orbit_state,
        current,
        current,
        2.0 * math.pi,
        0.0,
        step,
        newton_tol=1.0e-8,
        max_newton=12,
        min_step_theta=step / 4.0,
    )
    if states.shape[1] == 0:
        return {
            "validated": False,
            "relative_error": float("inf"),
            "integrator": integrator,
        }
    final = states[:, -1]
    error = float(np.linalg.norm(final - orbit_state) / max(np.linalg.norm(orbit_state), 1.0e-300))
    return {
        "validated": bool(integrator.get("success", False)),
        "relative_error": error,
        "integrator": integrator,
    }


def _plot_results(results: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    powers = np.asarray([row["power_dbm"] for row in results])
    colors = np.linspace(0.0, 1.0, max(1, len(results)))
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    circle = np.exp(1j * np.linspace(0.0, 2.0 * math.pi, 512))
    ax.plot(circle.real, circle.imag, "k--", linewidth=0.8, label="unit circle")
    for color, row in zip(colors, results):
        values = np.asarray(row["multipliers_real"]) + 1j * np.asarray(row["multipliers_imag"])
        ax.scatter(values.real, values.imag, color=plt.cm.viridis(color), s=18, label=f"{row['power_dbm']:.3f} dBm")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("2c PERIOD1 Floquet multipliers")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "multipliers_complex.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(powers, [row["spectral_radius"] for row in results], "o-")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Pump power [dBm]")
    ax.set_ylabel("spectral radius")
    ax.set_title("Floquet spectral radius")
    fig.tight_layout()
    fig.savefig(output / "spectral_radius_vs_power.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.0))
    branch_count = max((len(row["tracked_real"]) for row in results), default=0)
    tracked = np.full((len(results), branch_count), np.nan + 1j * np.nan)
    for row_index, row in enumerate(results):
        values = np.asarray(row["tracked_real"]) + 1j * np.asarray(row["tracked_imag"])
        tracked[row_index, : values.size] = values
    for branch in range(tracked.shape[1]):
        axes[0].plot(powers, tracked[:, branch].real, ".-")
        axes[1].plot(powers, tracked[:, branch].imag, ".-")
    axes[0].axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    axes[0].axhline(-1.0, color="k", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Re(lambda)")
    axes[1].set_ylabel("Im(lambda)")
    axes[1].set_xlabel("Pump power [dBm]")
    fig.suptitle("Tracked Floquet branches")
    fig.tight_layout()
    fig.savefig(output / "tracked_multipliers_vs_power.png", dpi=160)
    plt.close(fig)

    distances_plus = []
    distances_minus = []
    for row in results:
        values = np.asarray(row["multipliers_real"]) + 1j * np.asarray(row["multipliers_imag"])
        distances_plus.append(float(np.min(np.abs(values - 1.0))) if values.size else np.nan)
        distances_minus.append(float(np.min(np.abs(values + 1.0))) if values.size else np.nan)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(powers, distances_plus, "o-", label="min |lambda - 1|")
    ax.semilogy(powers, distances_minus, "s-", label="min |lambda + 1|")
    ax.set_xlabel("Pump power [dBm]")
    ax.set_ylabel("distance")
    ax.set_title("Distance to special real crossings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "distance_to_special_crossings.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(powers, [row["e_folding_periods"] for row in results], "o-")
    ax.set_xlabel("Pump power [dBm]")
    ax.set_ylabel("periods per e-fold")
    ax.set_title("Floquet relaxation/growth time estimate")
    fig.tight_layout()
    fig.savefig(output / "floquet_efolding_periods.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.plot(powers, np.ones_like(powers), "k.", label="validated HB checkpoint")
    stable = [
        row["spectral_radius"] < 1.0
        if np.isfinite(row["spectral_radius"])
        else np.nan
        for row in results
    ]
    ax.plot(powers, np.asarray(stable, dtype=float), "bo", label="Floquet stable")
    td = [row.get("td_classification") for row in results]
    td_indices = [index for index, value in enumerate(td) if value]
    if td_indices:
        ax.plot(
            powers[td_indices],
            np.full(len(td_indices), 0.5),
            "rx",
            label="TD classification available",
        )
    ax.set_yticks([0.0, 0.5, 1.0], ["unstable/unresolved", "TD", "HB exists"])
    ax.set_xlabel("Pump power [dBm]")
    ax.set_title("HB + Floquet + optional TD overlay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "hb_floquet_td_overlay.png", dpi=160)
    plt.close(fig)


def _load_td_labels(path: Path | None) -> dict[float, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    candidates = payload.get("results", payload.get("runs", []))
    labels: dict[float, str] = {}
    for row in candidates:
        if not isinstance(row, dict):
            continue
        power = row.get("power_dbm", row.get("pump_power_dbm"))
        label = row.get("classification_decay_aware", row.get("classification"))
        if power is not None and label is not None:
            labels[float(power)] = str(label)
    return labels


def run(args: argparse.Namespace) -> dict[str, Any]:
    circuit = load_circuit(args.circuit_dir)
    roots = [args.checkpoint_root, *args.additional_checkpoint_root]
    checkpoints = [Path(value) for value in args.checkpoints] if args.checkpoints else _discover_checkpoint_union(roots)
    checkpoints.sort(
        key=lambda value: float(
            json.loads((value / "pump_report.json").read_text())["metadata"][
                "pump_power_dbm_requested"
            ]
        )
    )
    checkpoints = checkpoints[: args.max_points] if args.max_points else checkpoints
    n_steps = int(math.ceil(2.0 * math.pi / args.step_theta))
    if n_steps % 2:
        n_steps += 1
    results: list[dict[str, Any]] = []
    td_labels = _load_td_labels(args.td_summary)
    previous: np.ndarray | None = None
    for checkpoint in checkpoints:
        started = time.perf_counter()
        report = json.loads((checkpoint / "pump_report.json").read_text())
        current = float(report["metadata"]["pump_current_a"])
        dc_flux = _dc_flux(report, circuit.branch_count)
        system = build_system(args.circuit_dir, args.freq_ghz, args.pump_port, dc_flux)
        orbit, _ = load_hb_periodic_orbit(
            checkpoint,
            system.omega,
            system.phi0,
            steps_per_period=n_steps,
        )
        state = orbit.initial_state(system)
        closure = _closure_error(system, state, current, args.step_theta) if not args.skip_closure else {"validated": False, "skipped": True}
        factor_started = time.perf_counter()
        operator = build_monodromy_operator(system, orbit, max_step_theta=args.step_theta)
        factor_runtime_s = time.perf_counter() - factor_started
        floquet = compute_floquet_multipliers(
            operator,
            eigenvalues=args.eigenvalues,
            tol=args.eigensolver_tol,
            maxiter=args.eigensolver_maxiter,
            ncv=args.eigensolver_ncv,
            seed=args.seed,
        )
        values = floquet.multipliers
        if values.size == 0 or (previous is not None and values.size != previous.size):
            tracked = np.empty(0, dtype=np.complex128)
        else:
            tracked = values if previous is None else track_multiplier_branches(previous, values)
        previous = tracked.copy() if tracked.size else previous
        rho = float(np.max(np.abs(values))) if values.size else float("nan")
        period_s = 2.0 * math.pi / system.omega
        exponents = floquet_exponents(values, period_s) if values.size else np.empty(0, dtype=np.complex128)
        power = _checkpoint_power(checkpoint, report)
        row = {
            "checkpoint": str(checkpoint),
            "power_dbm": power,
            "pump_current_a": current,
            "state_dimension": operator.shape[0],
            "steps_per_period": n_steps,
            "step_theta": 2.0 * math.pi / n_steps,
            "closure": closure,
            "multipliers_real": values.real.tolist(),
            "multipliers_imag": values.imag.tolist(),
            "multipliers_abs": np.abs(values).tolist(),
            "multipliers_angle_rad": np.angle(values).tolist(),
            "floquet_exponents_real_per_s": exponents.real.tolist(),
            "floquet_exponents_imag_per_s": exponents.imag.tolist(),
            "tracked_real": tracked.real.tolist(),
            "tracked_imag": tracked.imag.tolist(),
            "special_classifications": [classify_multiplier(value) for value in values],
            "spectral_radius": rho,
            "e_folding_periods": float(1.0 / abs(math.log(rho))) if np.isfinite(rho) and rho != 1.0 else float("nan"),
            "arnoldi": {
                "runtime_s": floquet.runtime_s,
                "matvecs": floquet.matvecs,
                "requested_eigenvalues": floquet.requested_eigenvalues,
                "which": floquet.which,
                "converged": floquet.converged,
                "message": floquet.message,
                "average_matvec_s": (
                    floquet.runtime_s / floquet.matvecs
                    if floquet.matvecs
                    else float("nan")
                ),
            },
            "factorization_runtime_s": factor_runtime_s,
            "rss_mb": _rss_mb(),
            "runtime_s": time.perf_counter() - started,
        }
        if power in td_labels:
            row["td_classification"] = td_labels[power]
        results.append(row)
        status = "converged" if floquet.converged else "unresolved"
        print(f"{power: .6f} dBm: rho={rho:.9f} floquet={status} closure={closure.get('relative_error', float('nan')):.3e} runtime={row['runtime_s']:.1f}s")

    results.sort(key=lambda row: row["power_dbm"])
    payload = {
        "metadata": {
            "circuit_dir": str(args.circuit_dir),
            "frequency_ghz": args.freq_ghz,
            "pump_port": args.pump_port,
            "step_theta_requested": args.step_theta,
            "eigenvalues": args.eigenvalues,
            "state_space": "(delta_q_d, delta_p_d, delta_q_a), p_a reconstructed from the algebraic velocity constraint",
            "monodromy": "matrix-free tangent of implicit trapezoid; one sparse Schur factor per phase step reused by Arnoldi",
            "bifurcation_gate": "NO_NEW_HB_ANSATZ_UNTIL_CONTINUOUS_CROSSING_IS_CONFIRMED",
            "td_summary": str(args.td_summary) if args.td_summary else None,
        },
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "floquet_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, args.output / "floquet_report.md")
    if args.plots and results:
        _plot_results(results, args.output / "plots")
    return payload


def _write_report(payload: dict[str, Any], path: Path) -> None:
    rows = payload["results"]
    lines = [
        "# 2c PERIOD1 Floquet analysis",
        "",
        "The production state is `(delta_q_d, delta_p_d, delta_q_a)`. Algebraic `p_a` is reconstructed from the differentiated algebraic constraint. The monodromy is never assembled; Arnoldi applies the cached implicit-trapezoid tangent steps.",
        "",
        "| power [dBm] | spectral radius | closure error | e-fold periods |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        closure = row["closure"].get("relative_error", float("nan"))
        lines.append(
            f"| {row['power_dbm']:.6f} | {row['spectral_radius']:.9f} | {closure:.3e} | {row['e_folding_periods']:.3g} |"
        )
    lines.extend(
        [
            "",
            "Interpretation is classification-level only when the closure error is at the expected trapezoid/HB floor and the result is timestep-converged. No PERIOD2, torus, or other HB ansatz is implemented by this command.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / ".hybrid_outputs" / "hb_up_7p9_m35_to_m21" / "pass" / "points")
    parser.add_argument(
        "--additional-checkpoint-root",
        type=Path,
        action="append",
        default=[],
        help="Additional validated PERIOD1 checkpoint roots, e.g. recovered PALC points.",
    )
    parser.add_argument("--checkpoints", nargs="+", help="explicit pump checkpoint directories")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".hybrid_outputs" / "floquet_7p9_2c_v1",
        help="runtime output directory; kept separate from the overnight TD campaign",
    )
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--step-theta", type=float, default=0.05)
    parser.add_argument("--eigenvalues", type=int, default=12)
    parser.add_argument("--eigensolver-tol", type=float, default=1.0e-8)
    parser.add_argument("--eigensolver-maxiter", type=int, default=80)
    parser.add_argument("--eigensolver-ncv", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--skip-closure", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--td-summary",
        type=Path,
        default=None,
        help="optional machine-readable TD campaign summary for the overlay plot",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
