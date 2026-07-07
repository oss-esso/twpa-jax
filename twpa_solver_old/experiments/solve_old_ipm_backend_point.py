"""Single-point backend adapter for the canonical Julia old-IPM map runner."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr, spsolve
from scipy.optimize import least_squares

from twpa_solver_old.importers.julia_circuit_json import import_julia_circuit_json
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver_old.residuals.conversion import build_conversion_sparameters


SUPPORTED_BACKENDS = {
    "scipy-least-squares",
    "scipy-root",
    "scipy-newton-krylov",
    "jax-dense-newton",
    "jax-newton-krylov",
    "pseudo-transient",
}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, Any]
    try:
        if args.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported backend {args.backend!r}")
        imported = import_julia_circuit_json(args.circuit_json)
        result = _solve_independent_backend(imported, args, outdir, started)
    except Exception as exc:
        result = {
            "backend": args.backend,
            "status": "FAILED_EXCEPTION",
            "success": False,
            "gain_db_max": None,
            "raw_gain_trace": None,
            "gain_trace_json": None,
            "convergence_mask_value": 0,
            "finite_mask_value": 0,
            "solver_warning_mask_value": 1,
            "residual_norm": None,
            "infinity_norm": None,
            "initial_residual_norm": None,
            "initial_infinity_norm": None,
            "initial_residual_inf": None,
            "final_residual_inf": None,
            "residual_reduction_factor": None,
            "iterations": 0,
            "function_evals": 0,
            "jacobian_evals": 0,
            "linear_solves": 0,
            "point_runtime_s": time.perf_counter() - started,
            "runtime_s": time.perf_counter() - started,
            "solver_message": repr(exc),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "metadata": {
                "circuit_json": str(args.circuit_json),
                "surrogate_topology_used": False,
            },
        }
    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(outdir / "result.json")


def _solve_independent_backend(imported, args: argparse.Namespace, outdir: Path, started: float) -> dict[str, Any]:
    """Run a real old-IPM residual attempt for every independent backend name.

    The exact old-IPM residual assembly is shared.  Backends that do not yet have a
    distinct full-size implementation are explicitly tagged in metadata, but they
    still evaluate and attempt to reduce the exact imported residual instead of
    returning a placeholder row.
    """
    strategy_by_backend = {
        "scipy-least-squares": "scipy_least_squares_sparse_newton",
        "scipy-root": "scipy_root_sparse_newton_compat",
        "scipy-newton-krylov": "scipy_newton_krylov_sparse_jvp_compat",
        "jax-dense-newton": "jax_dense_newton_size_guard_sparse_compat",
        "jax-newton-krylov": "jax_newton_krylov_jvp_compat",
        "pseudo-transient": "pseudo_transient_sparse_newton_compat",
    }
    result = _solve_scipy_least_squares(imported, args, outdir, started)
    metadata = dict(result.get("metadata", {}))
    metadata["requested_backend"] = args.backend
    metadata["backend_strategy"] = strategy_by_backend.get(args.backend, "unknown")
    metadata["backend_core"] = "exact_old_ipm_aft_residual_analytic_sparse_jacobian"
    metadata["native_backend_complete"] = args.backend == "scipy-least-squares"
    if args.backend != "scipy-least-squares":
        result["solver_message"] = (
            f"{result.get('solver_message', '')}; {args.backend} used exact old-IPM "
            "residual with the shared analytic sparse Newton/least-squares core. "
            "This is a numeric backend attempt, not a placeholder, but not yet a "
            "native full-size implementation."
        ).strip("; ")
    result["metadata"] = metadata
    return _complete_backend_schema(result)


def _solve_scipy_least_squares(imported, args: argparse.Namespace, outdir: Path, started: float) -> dict[str, Any]:
    model = imported.model
    requested_harmonics = int(args.pump_harmonics)
    effective_harmonics = min(requested_harmonics, int(args.max_effective_pump_harmonics))
    config_base = {
        "pump_frequency_hz": args.pump_frequency_ghz * 1e9,
        "harmonics": effective_harmonics,
        "residual_scale_a": args.residual_scale_a,
        "time_samples": max(16, 4 * effective_harmonics + 2),
    }
    residual = PumpAFTResidual(
        model,
        PumpAFTConfig(source_current_peak_a=args.pump_current_a, **config_base),
    )
    x0 = residual.initial_guess()
    history: list[dict[str, float | int]] = []
    continuation_rows: list[dict[str, float | int | str]] = []
    jacobian_evals = 0

    def wrapped_for(res: PumpAFTResidual):
        def wrapped(x: np.ndarray) -> np.ndarray:
            r = res(x)
            eval_idx = len(history)
            if not np.all(np.isfinite(r)):
                history.append({"eval": eval_idx, "l2": float("inf"), "inf": float("inf")})
                return np.full_like(r, 1e300)
            history.append(
                {
                    "eval": eval_idx,
                    "l2": float(np.linalg.norm(r)),
                    "inf": float(np.max(np.abs(r))) if r.size else 0.0,
                }
            )
            return r

        return wrapped

    def jac_for(res: PumpAFTResidual):
        def jac(x: np.ndarray) -> sparse.csr_matrix:
            nonlocal jacobian_evals
            jacobian_evals += 1
            return res.jacobian_sparse(x)

        return jac

    initial_vec = residual(x0)
    initial_l2 = float(np.linalg.norm(initial_vec))
    initial_inf = float(np.max(np.abs(initial_vec))) if initial_vec.size else 0.0

    def solve_stage(res: PumpAFTResidual, start: np.ndarray, label: str):
        before = res(start)
        before_inf = float(np.max(np.abs(before))) if before.size else 0.0
        before_l2 = float(np.linalg.norm(before))
        if args.newton_first_enabled:
            newton_x, newton_iterations, newton_inf = _sparse_newton_refine(
                res,
                start,
                tolerance=args.convergence_residual_inf,
                max_iterations=args.newton_refine_maxiter,
            )
            if newton_inf <= before_inf:
                newton_after = res(newton_x)
                newton_l2 = float(np.linalg.norm(newton_after))
                continuation_rows.append(
                    {
                        "stage": label,
                        "source_current_peak_a": res.config.source_current_peak_a,
                        "initial_l2": before_l2,
                        "initial_inf": before_inf,
                        "linear_kick_l2": before_l2,
                        "linear_kick_inf": before_inf,
                        "linear_kick_iterations": 0,
                        "linear_kick_stop_code": 0,
                        "newton_refine_iterations": newton_iterations,
                        "newton_refine_inf": newton_inf,
                        "final_l2": newton_l2,
                        "final_inf": newton_inf,
                        "nfev": 0,
                        "njev": newton_iterations,
                        "success": str(newton_inf <= args.convergence_residual_inf),
                        "message": "sparse Newton first path",
                    }
                )
                if newton_inf <= args.convergence_residual_inf:
                    return (
                        SimpleNamespace(
                            x=newton_x,
                            nfev=0,
                            njev=newton_iterations,
                            success=True,
                            message="sparse Newton first path",
                        ),
                        newton_after,
                        newton_l2,
                        newton_inf,
                    )
                start = newton_x
                before = newton_after
                before_inf = newton_inf
                before_l2 = newton_l2
        kicked_start = start
        kick_inf = before_inf
        kick_l2 = before_l2
        kick_iterations = 0
        kick_stop_code = 0
        if args.linear_kick_enabled and before_inf > args.convergence_residual_inf:
            jac0 = res.jacobian_sparse(start)
            kick = lsmr(
                jac0,
                -before,
                atol=args.linear_kick_tolerance,
                btol=args.linear_kick_tolerance,
                maxiter=args.linear_kick_maxiter,
            )
            candidate = start + args.linear_kick_damping * kick[0]
            kicked = res(candidate)
            candidate_inf = float(np.max(np.abs(kicked))) if kicked.size else 0.0
            if np.all(np.isfinite(kicked)) and candidate_inf <= before_inf:
                kicked_start = candidate
                kick_inf = candidate_inf
                kick_l2 = float(np.linalg.norm(kicked))
            kick_stop_code = int(kick[1])
            kick_iterations = int(kick[2])
        out = least_squares(
            wrapped_for(res),
            kicked_start,
            jac=jac_for(res),
            method="trf",
            max_nfev=args.max_nfev,
            ftol=args.optimizer_tolerance,
            xtol=args.optimizer_tolerance,
            gtol=args.optimizer_tolerance,
            x_scale="jac",
        )
        refined_x, newton_iterations, newton_inf = _sparse_newton_refine(
            res,
            out.x,
            tolerance=args.convergence_residual_inf,
            max_iterations=args.newton_refine_maxiter if args.newton_refine_enabled else 0,
        )
        after = res(refined_x)
        after_inf = float(np.max(np.abs(after))) if after.size else 0.0
        after_l2 = float(np.linalg.norm(after))
        continuation_rows.append(
            {
                "stage": label,
                "source_current_peak_a": res.config.source_current_peak_a,
                "initial_l2": before_l2,
                "initial_inf": before_inf,
                "linear_kick_l2": kick_l2,
                "linear_kick_inf": kick_inf,
                "linear_kick_iterations": kick_iterations,
                "linear_kick_stop_code": kick_stop_code,
                "newton_refine_iterations": newton_iterations,
                "newton_refine_inf": newton_inf,
                "final_l2": after_l2,
                "final_inf": after_inf,
                "nfev": int(getattr(out, "nfev", 0) or 0),
                "njev": int(getattr(out, "njev", 0) or 0),
                "success": str(bool(out.success)),
                "message": str(out.message),
            }
        )
        out.x = refined_x
        return out, after, after_l2, after_inf

    def continuation_scales() -> list[float]:
        if not args.continuation_enabled:
            return [1.0]
        scales = [float(item) for item in args.continuation_scales.split(",") if item.strip()]
        if not scales or scales[-1] != 1.0:
            scales.append(1.0)
        return scales

    status = "FAILED_MAX_NFEV"
    message = ""
    solution = x0
    residual_vec: np.ndarray
    residual_l2 = initial_l2
    residual_inf = initial_inf
    total_nfev = 0
    total_njev = 0
    scipy_success = False
    try:
        for scale in continuation_scales():
            stage_residual = PumpAFTResidual(
                model,
                PumpAFTConfig(source_current_peak_a=args.pump_current_a * scale, **config_base),
            )
            scipy_result, residual_vec, residual_l2, residual_inf = solve_stage(
                stage_residual,
                solution,
                f"scale_{scale:g}",
            )
            solution = scipy_result.x
            total_nfev += int(getattr(scipy_result, "nfev", 0) or 0)
            total_njev += int(getattr(scipy_result, "njev", 0) or 0)
            message = str(scipy_result.message)
            scipy_success = bool(scipy_result.success)
        residual = PumpAFTResidual(
            model,
            PumpAFTConfig(source_current_peak_a=args.pump_current_a, **config_base),
        )
        residual_vec = residual(solution)
        residual_l2 = float(np.linalg.norm(residual_vec))
        residual_inf = float(np.max(np.abs(residual_vec))) if residual_vec.size else 0.0
        if not np.all(np.isfinite(residual_vec)):
            status = "FAILED_NONFINITE_RESIDUAL"
        elif residual_inf <= args.convergence_residual_inf:
            status = "VALID_CONVERGED"
        elif residual_inf < initial_inf:
            status = "RESIDUAL_REDUCED_NOT_CONVERGED"
        elif total_nfev >= args.max_nfev * len(continuation_scales()):
            status = "FAILED_MAX_NFEV"
        else:
            status = "FINITE_NONCONVERGED"
    except MemoryError as exc:
        residual_vec = residual(x0)
        residual_l2 = float(np.linalg.norm(residual_vec))
        residual_inf = float(np.max(np.abs(residual_vec))) if residual_vec.size else 0.0
        status = "FAILED_MEMORY"
        message = repr(exc)
    except Exception as exc:
        residual_vec = residual(x0)
        residual_l2 = float(np.linalg.norm(residual_vec))
        residual_inf = float(np.max(np.abs(residual_vec))) if residual_vec.size else 0.0
        status = "FAILED_NUMERICALLY"
        message = repr(exc)

    gain_db_max = None
    raw_gain_trace = None
    if status == "VALID_CONVERGED":
        try:
            gain_result = build_conversion_sparameters(
                model,
                residual,
                solution,
                args.signal_frequency_ghz * 1e9,
                args.sidebands,
                pump_success=True,
                pump_status=status,
            )
            gain_db_max = gain_result.signal_gain_db
            raw_gain_trace = [gain_db_max]
        except Exception as exc:
            status = "FAILED_NUMERICALLY"
            message = f"{message}; conversion failed: {exc!r}"

    _write_residual_history(outdir / "residual_history.csv", history)
    _write_continuation_history(outdir / "continuation_history.csv", continuation_rows)
    _write_block_norms(outdir / "residual_block_norms.csv", residual.diagnostic_by_harmonic(solution))
    np.savez_compressed(
        outdir / "pump_solution_coefficients.npz",
        solution=solution,
        requested_harmonics=requested_harmonics,
        effective_harmonics=effective_harmonics,
    )
    runtime = time.perf_counter() - started
    runtime_summary = {
        "runtime_s": runtime,
        "max_nfev": args.max_nfev,
        "residual_evaluations_recorded": len(history),
        "jacobian_evaluations_recorded": jacobian_evals,
        "total_nfev": total_nfev,
        "total_njev": total_njev,
        "linear_kick_enabled": bool(args.linear_kick_enabled),
        "newton_first_enabled": bool(args.newton_first_enabled),
        "newton_refine_enabled": bool(args.newton_refine_enabled),
        "requested_harmonics": requested_harmonics,
        "effective_harmonics": effective_harmonics,
    }
    (outdir / "runtime_summary.json").write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")

    metadata = _base_metadata(imported, args)
    metadata.update(
        {
            "requested_pump_harmonics": requested_harmonics,
            "effective_pump_harmonics": effective_harmonics,
            "harmonics_downgraded": effective_harmonics != requested_harmonics,
            "residual_size": int(residual.size),
            "num_unknowns": int(residual.size),
            "num_residuals": int(residual.size),
            "time_samples": int(residual.time_samples),
            "max_nfev": int(args.max_nfev),
            "num_function_evals": int(total_nfev),
            "num_jacobian_evals": int(total_njev or jacobian_evals),
            "jacobian_strategy": "analytic_sparse_aft",
            "linear_kick_enabled": bool(args.linear_kick_enabled),
            "linear_kick_tolerance": args.linear_kick_tolerance,
            "linear_kick_maxiter": args.linear_kick_maxiter,
            "linear_kick_damping": args.linear_kick_damping,
            "newton_first_enabled": bool(args.newton_first_enabled),
            "newton_refine_enabled": bool(args.newton_refine_enabled),
            "newton_refine_maxiter": args.newton_refine_maxiter,
            "continuation_enabled": bool(args.continuation_enabled),
            "continuation_scales": continuation_scales(),
            "initial_residual_l2": initial_l2,
            "initial_residual_inf": initial_inf,
            "final_residual_l2": residual_l2,
            "final_residual_inf": residual_inf,
            "residual_reduction_factor": (initial_inf / residual_inf) if residual_inf > 0 else float("inf"),
            "conversion_attempted": status in {"VALID_CONVERGED", "FAILED_NUMERICALLY"},
            "signal_frequency_ghz": args.signal_frequency_ghz,
            "sidebands": args.sidebands,
        }
    )
    return _complete_backend_schema({
        "backend": args.backend,
        "status": status,
        "success": status == "VALID_CONVERGED",
        "gain_db_max": gain_db_max,
        "raw_gain_trace": raw_gain_trace,
        "residual_norm": residual_l2,
        "infinity_norm": residual_inf,
        "initial_residual_inf": initial_inf,
        "final_residual_inf": residual_inf,
        "residual_reduction_factor": (initial_inf / residual_inf) if residual_inf > 0 else float("inf"),
        "jacobian_strategy": "analytic_sparse_aft",
        "linear_kick_enabled": bool(args.linear_kick_enabled),
        "newton_first_enabled": bool(args.newton_first_enabled),
        "newton_refine_enabled": bool(args.newton_refine_enabled),
        "continuation_enabled": bool(args.continuation_enabled),
        "effective_pump_harmonics": effective_harmonics,
        "requested_pump_harmonics": requested_harmonics,
        "num_unknowns": int(residual.size),
        "num_residuals": int(residual.size),
        "num_function_evals": int(total_nfev),
        "num_jacobian_evals": int(total_njev or jacobian_evals),
        "runtime_s": runtime,
        "solver_message": message,
        "metadata": metadata,
    })


def _complete_backend_schema(result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status", "FAILED_NUMERICALLY"))
    success = bool(result.get("success", False))
    gain = result.get("gain_db_max")
    finite_gain = isinstance(gain, (int, float)) and np.isfinite(float(gain))
    runtime = float(result.get("runtime_s") or result.get("point_runtime_s") or 0.0)
    initial_l2 = result.get("initial_residual_norm", result.get("initial_residual_l2"))
    initial_inf = result.get("initial_infinity_norm", result.get("initial_residual_inf"))
    final_inf = result.get("final_residual_inf", result.get("infinity_norm"))
    residual_norm = result.get("residual_norm", result.get("final_residual_l2"))
    metadata = result.setdefault("metadata", {})
    if initial_l2 is None:
        initial_l2 = metadata.get("initial_residual_l2", metadata.get("initial_residual_norm"))
    if initial_inf is None:
        initial_inf = metadata.get("initial_residual_inf", metadata.get("initial_infinity_norm"))
    result.setdefault("gain_trace_json", result.get("raw_gain_trace"))
    result["convergence_mask_value"] = 1 if status == "VALID_CONVERGED" else 0
    result["finite_mask_value"] = 1 if finite_gain else 0
    result["solver_warning_mask_value"] = 0 if success else 1
    result["initial_residual_norm"] = initial_l2
    result["initial_infinity_norm"] = initial_inf
    result.setdefault("initial_residual_inf", initial_inf)
    result.setdefault("final_residual_inf", final_inf)
    result.setdefault("residual_reduction_factor", None)
    result["iterations"] = int(result.get("iterations") or metadata.get("num_jacobian_evals") or 0)
    result["function_evals"] = int(result.get("function_evals") or result.get("num_function_evals") or 0)
    result["jacobian_evals"] = int(result.get("jacobian_evals") or result.get("num_jacobian_evals") or 0)
    result["linear_solves"] = int(result.get("linear_solves") or result["jacobian_evals"])
    result["point_runtime_s"] = runtime
    result["runtime_s"] = runtime
    result["residual_norm"] = residual_norm
    result["infinity_norm"] = final_inf
    result.setdefault("solver_message", "")
    result.setdefault("error_type", "" if success else status)
    result.setdefault("error_message", "" if success else result.get("solver_message", ""))
    return result


def _base_metadata(imported, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "circuit_json": str(args.circuit_json),
        "node_count": imported.model.num_nodes,
        "element_count": imported.model.metadata.get("element_count"),
        "josephson_junction_count": imported.model.metadata.get("josephson_junction_count"),
        "mutual_coupling_count": imported.model.metadata.get("mutual_coupling_count"),
        "pump_frequency_ghz": args.pump_frequency_ghz,
        "external_pump_power_dbm": args.external_pump_power_dbm,
        "source_power_dbm": args.source_power_dbm,
        "pump_current_a": args.pump_current_a,
        "pump_harmonics": args.pump_harmonics,
        "modulation_harmonics": args.modulation_harmonics,
        "pump_nodes": list(imported.model.pump_nodes),
        "surrogate_topology_used": False,
    }


def _write_residual_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["eval", "l2", "inf"])
        writer.writeheader()
        writer.writerows(rows)


def _sparse_newton_refine(
    residual: PumpAFTResidual,
    x0: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float]:
    x = np.asarray(x0, dtype=float).copy()
    r = residual(x)
    best_inf = float(np.max(np.abs(r))) if r.size else 0.0
    iterations = 0
    for _ in range(max_iterations):
        if best_inf <= tolerance:
            break
        jac = residual.jacobian_sparse(x)
        try:
            step = spsolve(jac, -r)
        except Exception:
            break
        if not np.all(np.isfinite(step)):
            break
        accepted = False
        damping = 1.0
        while damping >= 1e-8:
            candidate = x + damping * step
            candidate_r = residual(candidate)
            candidate_inf = float(np.max(np.abs(candidate_r))) if candidate_r.size else 0.0
            if np.all(np.isfinite(candidate_r)) and candidate_inf < best_inf:
                x = candidate
                r = candidate_r
                best_inf = candidate_inf
                iterations += 1
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            break
    return x, iterations, best_inf


def _write_block_norms(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["harmonic", "residual_l2_a", "residual_inf_a"])
        writer.writeheader()
        writer.writerows(rows)


def _write_continuation_history(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "stage",
        "source_current_peak_a",
        "initial_l2",
        "initial_inf",
        "final_l2",
        "final_inf",
        "nfev",
        "njev",
        "success",
        "message",
        "linear_kick_l2",
        "linear_kick_inf",
        "linear_kick_iterations",
        "linear_kick_stop_code",
        "newton_refine_iterations",
        "newton_refine_inf",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit-json", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--pump-frequency-ghz", type=float, required=True)
    parser.add_argument("--external-pump-power-dbm", type=float, required=True)
    parser.add_argument("--source-power-dbm", type=float, required=True)
    parser.add_argument("--pump-current-a", type=float, required=True)
    parser.add_argument("--pump-harmonics", type=int, required=True)
    parser.add_argument("--modulation-harmonics", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-effective-pump-harmonics", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--residual-scale-a", type=float, default=1e-6)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-8)
    parser.add_argument("--convergence-residual-inf", type=float, default=1e-6)
    parser.add_argument("--signal-frequency-ghz", type=float, default=6.0)
    parser.add_argument("--sidebands", type=int, default=0)
    parser.add_argument("--continuation-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continuation-scales", default="0,0.01,0.03,0.1,0.3,1")
    parser.add_argument("--linear-kick-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linear-kick-tolerance", type=float, default=1e-10)
    parser.add_argument("--linear-kick-maxiter", type=int, default=2000)
    parser.add_argument("--linear-kick-damping", type=float, default=1.0)
    parser.add_argument("--newton-first-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--newton-refine-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--newton-refine-maxiter", type=int, default=12)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
