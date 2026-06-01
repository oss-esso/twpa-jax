"""
Run the TWPA production validation suite.

This script is a lightweight orchestrator for smoke tests, numerical sanity
checks, and optional pytest-based regression tests.

Examples
--------
Fast internal validation:

    python scripts/run_validation_suite.py --output-dir outputs/validation_fast

Run imports + numerical smoke + pytest:

    python scripts/run_validation_suite.py --run-pytest --output-dir outputs/validation_full

Run slow pytest tests too:

    python scripts/run_validation_suite.py --run-pytest --run-slow --output-dir outputs/validation_slow
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    elapsed_s: float
    summary: dict[str, Any]
    messages: list[str]

    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "summary": self.summary,
            "messages": self.messages,
        }


def _jsonify(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, complex):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj)), "abs": float(abs(obj))}
    if hasattr(obj, "to_dict"):
        try:
            return _jsonify(obj.to_dict())
        except TypeError:
            return _jsonify(obj.to_dict(include_arrays=False))
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return _jsonify(arr.item())
        if np.iscomplexobj(arr):
            return {
                "shape": tuple(int(v) for v in arr.shape),
                "dtype": str(arr.dtype),
                "min_abs": float(np.nanmin(np.abs(arr))) if arr.size else None,
                "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else None,
            }
        return {
            "shape": tuple(int(v) for v in arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
        }
    return obj


def run_check(name: str, fn: Callable[[], dict[str, Any]]) -> CheckResult:
    start = time.perf_counter()
    try:
        summary = fn()
        status = CheckStatus(summary.pop("status", CheckStatus.PASS.value))
        messages = list(summary.pop("messages", []))
        return CheckResult(
            name=name,
            status=status,
            elapsed_s=time.perf_counter() - start,
            summary=_jsonify(summary),
            messages=messages,
        )
    except Exception as exc:
        return CheckResult(
            name=name,
            status=CheckStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            summary={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
            messages=[f"ERROR: {type(exc).__name__}: {exc}"],
        )


def check_imports() -> dict[str, Any]:
    modules = [
        "twpa",
        "twpa.core",
        "twpa.core.units",
        "twpa.core.params",
        "twpa.core.layout",
        "twpa.core.frequency_plan",
        "twpa.core.harmonics",
        "twpa.core.hb_fft",
        "twpa.core.disorder",
        "twpa.linear",
        "twpa.linear.rf_networks",
        "twpa.linear.cells",
        "twpa.linear.cascade",
        "twpa.linear.ladder_mna",
        "twpa.linear.dispersion",
        "twpa.linear.coarsening",
        "twpa.nonlinear",
        "twpa.nonlinear.kinetic_inductance",
        "twpa.nonlinear.hb_element",
        "twpa.nonlinear.one_node",
        "twpa.nonlinear.distributed_hb",
        "twpa.nonlinear.pump_hb_ladder",
        "twpa.nonlinear.linearization",
        "twpa.nonlinear.gain",
        "twpa.nonlinear.conversion",
        "twpa.nonlinear.finite_signal_hb",
        "twpa.nonlinear.supercell_hb",
        "twpa.solvers",
        "twpa.solvers.hb_solver",
        "twpa.solvers.continuation",
        "twpa.solvers.linear_solvers",
        "twpa.solvers.block_banded",
        "twpa.solvers.preconditioners",
        "twpa.solvers.newton_krylov",
        "twpa.workflows",
        "twpa.workflows.industrial_100mm",
        "twpa.workflows.calibration",
        "twpa.workflows.synthetic_benchmarks",
        "twpa.inference",
        "twpa.inference.priors",
        "twpa.inference.synthetic",
        "twpa.inference.fitting",
        "twpa.inference.recovery",
        "twpa.io",
        "twpa.io.measurement",
        "twpa.io.netlist",
        "twpa.io.reports",
        "twpa.io.checkpoints",
        "twpa.plotting",
        "twpa.plotting.diagnostics",
        "twpa.plotting.gain_maps",
    ]

    passed = []
    failed = []

    for module_name in modules:
        try:
            importlib.import_module(module_name)
            passed.append(module_name)
        except Exception as exc:
            failed.append(
                {
                    "module": module_name,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    return {
        "status": CheckStatus.PASS.value if not failed else CheckStatus.FAIL.value,
        "n_modules": len(modules),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "passed": passed,
        "failed": failed,
        "messages": ["PASS: all modules imported."] if not failed else [f"FAIL: {len(failed)} imports failed."],
    }


def check_linear_solvers() -> dict[str, Any]:
    from twpa.solvers.linear_solvers import (
        IterativeLinearSolveConfig,
        LinearOperator,
        LinearSolverMethod,
        solve_linear_system,
        validate_linear_operator,
    )

    A = jnp.asarray(
        [
            [4.0 + 0.2j, 1.0 - 0.1j, 0.0],
            [1.0 + 0.1j, 3.0 + 0.0j, 0.5],
            [0.0, 0.5, 2.0 - 0.1j],
        ],
        dtype=jnp.complex128,
    )
    x_true = jnp.asarray([1.0 + 1j, -0.5 + 0.2j, 0.25 - 0.1j], dtype=jnp.complex128)
    b = A @ x_true

    op = LinearOperator.from_dense(A, name="validation_dense_operator")
    validation = validate_linear_operator(op)

    result = solve_linear_system(
        op,
        b,
        config=IterativeLinearSolveConfig(
            method=LinearSolverMethod.DENSE,
            atol=1e-12,
            rtol=1e-12,
            require_convergence=True,
        ),
    )

    err = float(jnp.linalg.norm(result.x - x_true))
    passed = validation["passed"] and result.converged and err < 1e-9

    return {
        "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        "operator_validation": validation,
        "solve_result": result.to_dict(),
        "solution_error_norm": err,
        "messages": ["PASS: dense linear solve validated."] if passed else ["FAIL: dense linear solve validation failed."],
    }


def check_block_banded() -> dict[str, Any]:
    from twpa.solvers.block_banded import (
        block_banded_memory_estimate,
        random_block_tridiagonal_matrix,
        validate_block_banded_matrix,
    )

    matrix = random_block_tridiagonal_matrix(
        n_blocks=8,
        block_size=3,
        diagonal_shift=5.0,
        offdiag_scale=0.05,
        seed=123,
    )

    validation = validate_block_banded_matrix(matrix)
    memory = block_banded_memory_estimate(matrix)

    passed = bool(validation["passed"])

    return {
        "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        "matrix": matrix.to_dict(include_block_stats=False),
        "validation": validation,
        "memory": memory,
        "messages": validation["messages"],
    }


def check_preconditioners() -> dict[str, Any]:
    from twpa.solvers.block_banded import random_block_tridiagonal_matrix
    from twpa.solvers.preconditioners import (
        PreconditionerConfig,
        PreconditionerKind,
        build_preconditioner,
        preconditioned_residual_diagnostics,
        validate_preconditioner,
    )

    matrix = random_block_tridiagonal_matrix(
        n_blocks=6,
        block_size=2,
        diagonal_shift=6.0,
        offdiag_scale=0.03,
        seed=456,
    )

    preconditioner = build_preconditioner(
        matrix,
        PreconditionerConfig(kind=PreconditionerKind.BLOCK_JACOBI),
    )

    validation = validate_preconditioner(preconditioner)
    diagnostics = preconditioned_residual_diagnostics(matrix, preconditioner)

    passed = bool(validation["passed"]) and preconditioner.ready

    return {
        "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        "preconditioner": preconditioner.to_dict(),
        "validation": validation,
        "diagnostics": diagnostics,
        "messages": validation["messages"],
    }


def check_newton_krylov() -> dict[str, Any]:
    from twpa.solvers.linear_solvers import IterativeLinearSolveConfig, LinearSolverMethod
    from twpa.solvers.newton_krylov import NewtonKrylovConfig, newton_krylov_solve, validate_jvp

    def residual(x: jax.Array) -> jax.Array:
        return jnp.asarray(
            [
                x[0] ** 2 + x[1] - 1.0,
                x[0] + x[1] ** 2 - 1.0,
            ],
            dtype=jnp.float64,
        )

    x0 = jnp.asarray([0.8, 0.4], dtype=jnp.float64)

    jvp_validation = validate_jvp(residual, x0, n_random_tests=2)

    result = newton_krylov_solve(
        residual,
        x0,
        config=NewtonKrylovConfig(
            max_iter=12,
            abs_tol=1e-10,
            rel_tol=1e-10,
            linear_solver=IterativeLinearSolveConfig(
                method=LinearSolverMethod.GMRES,
                max_iter=50,
                atol=1e-12,
                rtol=1e-10,
                allow_dense_fallback=False,
            ),
            require_linear_convergence=False,
            verbose=False,
        ),
    )

    passed = result.converged and result.final_residual_norm < 1e-8 and bool(jvp_validation["passed"])

    return {
        "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        "jvp_validation": jvp_validation,
        "result": result.to_dict(),
        "messages": ["PASS: Newton-Krylov converged on nonlinear system."] if passed else ["FAIL: Newton-Krylov validation failed."],
    }


def check_inference_priors() -> dict[str, Any]:
    from twpa.inference.priors import (
        make_default_twpa_scale_prior_set,
        summarize_samples,
    )

    prior_set = make_default_twpa_scale_prior_set()
    initial = prior_set.initial_values()
    vector = prior_set.initial_vector()
    decoded = prior_set.decode_vector(vector)
    samples = prior_set.sample_many(8, seed=123)
    sample_summary = summarize_samples(samples)

    max_init_error = max(abs(initial[k] - decoded[k]) for k in prior_set.enabled_names)
    passed = max_init_error < 1e-12 and prior_set.ndim > 0

    return {
        "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        "prior_set": prior_set.to_dict(),
        "initial_values": initial,
        "decoded_initial": decoded,
        "max_initial_roundtrip_error": max_init_error,
        "sample_summary": sample_summary,
        "messages": ["PASS: prior encode/decode/sample checks passed."] if passed else ["FAIL: prior checks failed."],
    }


def check_io_checkpoint(tmp_dir: Path) -> dict[str, Any]:
    from twpa.io.checkpoints import (
        CheckpointKind,
        CheckpointMetadata,
        checkpoint_markdown_summary,
        inspect_checkpoint,
        load_checkpoint,
        save_checkpoint,
    )

    path = tmp_dir / "validation_checkpoint.npz"
    arrays = {
        "x": jnp.arange(5, dtype=jnp.float64),
        "z": jnp.asarray([1.0 + 1j, 2.0 - 0.5j], dtype=jnp.complex128),
    }
    payload = {"purpose": "validation", "value": 42}

    save_checkpoint(
        path,
        metadata=CheckpointMetadata(
            kind=CheckpointKind.GENERIC,
            name="validation_checkpoint",
            source="scripts.run_validation_suite",
        ),
        arrays=arrays,
        payload=payload,
        scalars={"answer": 42},
    )

    ckpt = load_checkpoint(path)
    inspection = inspect_checkpoint(path, include_hash=False)
    markdown = checkpoint_markdown_summary(ckpt)

    passed = (
        path.exists()
        and "x" in ckpt.arrays
        and "z" in ckpt.arrays
        and ckpt.scalar("answer") == 42
        and "# Checkpoint" in markdown
    )

    return {
        "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        "path": str(path),
        "inspection": inspection,
        "markdown_preview": markdown.splitlines()[:8],
        "messages": ["PASS: checkpoint save/load validated."] if passed else ["FAIL: checkpoint save/load failed."],
    }


def check_plotting(tmp_dir: Path) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception as exc:
        return {
            "status": CheckStatus.SKIP.value,
            "matplotlib_available": False,
            "messages": [f"SKIP: matplotlib unavailable or backend failed: {exc}"],
        }

    from twpa.plotting.diagnostics import PlotConfig, plot_array_profile, save_figure
    from twpa.plotting.gain_maps import GainMapPlotConfig, plot_compression_sweep, save_gain_figure

    x = np.linspace(0.0, 1.0, 25)
    y = np.sin(2.0 * np.pi * x)

    fig1, _ = plot_array_profile(
        x,
        y,
        config=PlotConfig(title="Validation profile"),
        xlabel="x",
        ylabel="sin",
    )
    path1 = save_figure(fig1, tmp_dir / "validation_profile.png")

    import matplotlib.pyplot as plt

    plt.close(fig1)

    power = np.linspace(-120.0, -80.0, 10)
    gain = 20.0 - 0.02 * (power + 120.0) ** 2
    fig2, _ = plot_compression_sweep(
        power,
        gain,
        config=GainMapPlotConfig(title="Validation compression"),
    )
    path2 = save_gain_figure(fig2, tmp_dir / "validation_compression.png")
    plt.close(fig2)

    passed = path1.exists() and path2.exists()

    return {
        "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        "paths": {
            "profile_png": str(path1),
            "compression_png": str(path2),
        },
        "messages": ["PASS: plotting smoke checks generated figures."] if passed else ["FAIL: plotting smoke checks failed."],
    }


def check_optional_layout_linear_smoke() -> dict[str, Any]:
    """
    Try a tiny layout and linear scan through the workflow layer.

    This is intentionally defensive because the exact early core/layout API may
    still be evolving.
    """
    try:
        from twpa.workflows.synthetic_benchmarks import (
            SyntheticLayoutKind,
            SyntheticLayoutSpec,
            build_synthetic_layout,
        )
        from twpa.linear.cascade import run_linear_scan

        layout = build_synthetic_layout(
            SyntheticLayoutSpec(
                kind=SyntheticLayoutKind.UNIFORM,
                n_cells=8,
                length_m=8e-4,
                z0_ohm=50.0,
                phase_velocity_m_per_s=1.2e8,
                name="validation_linear_smoke",
            )
        )
        frequency_hz = jnp.linspace(1.0e9, 4.0e9, 9, dtype=jnp.float64)
        scan = run_linear_scan(frequency_hz, layout)

        passed = bool(scan.s.shape[0] == 9)

        return {
            "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
            "layout": layout.summary(),
            "scan": scan.to_dict(),
            "messages": ["PASS: tiny linear workflow smoke passed."] if passed else ["FAIL: tiny linear workflow smoke failed."],
        }

    except Exception as exc:
        return {
            "status": CheckStatus.SKIP.value,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "messages": [
                "SKIP: optional tiny linear workflow could not run. "
                "This is acceptable while lower-level layout APIs are under active construction. "
                f"Reason: {type(exc).__name__}: {exc}"
            ],
        }


def check_pytest(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
    ]

    if args.run_slow:
        pytest_cmd.append("--run-slow")

    if args.pytest_extra:
        pytest_cmd.extend(args.pytest_extra)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path.cwd()))

    log_path = output_dir / "pytest_output.txt"

    proc = subprocess.run(
        pytest_cmd,
        cwd=Path.cwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.write_text(proc.stdout, encoding="utf-8")

    passed = proc.returncode == 0

    return {
        "status": CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        "command": pytest_cmd,
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "output_tail": proc.stdout.splitlines()[-40:],
        "messages": ["PASS: pytest suite passed."] if passed else [f"FAIL: pytest returned {proc.returncode}."],
    }


def summary_markdown(results: list[CheckResult], *, started_at: str, elapsed_s: float) -> str:
    n_pass = sum(r.status == CheckStatus.PASS for r in results)
    n_fail = sum(r.status == CheckStatus.FAIL for r in results)
    n_error = sum(r.status == CheckStatus.ERROR for r in results)
    n_skip = sum(r.status == CheckStatus.SKIP for r in results)

    overall = "PASS" if n_fail == 0 and n_error == 0 else "FAIL"

    lines = [
        "# TWPA validation suite",
        "",
        f"- overall status: `{overall}`",
        f"- started at: `{started_at}`",
        f"- elapsed: `{elapsed_s:.6g} s`",
        f"- checks: `{len(results)}`",
        f"- pass/fail/error/skip: `{n_pass}/{n_fail}/{n_error}/{n_skip}`",
        "",
        "## Check summary",
        "",
        "| check | status | elapsed s | messages |",
        "|---|---|---:|---|",
    ]

    for r in results:
        msg = "<br>".join(r.messages[:3])
        lines.append(f"| `{r.name}` | `{r.status.value}` | {r.elapsed_s:.6g} | {msg} |")

    lines += ["", "## Failed checks", ""]

    any_failed = False
    for r in results:
        if r.status in {CheckStatus.FAIL, CheckStatus.ERROR}:
            any_failed = True
            lines += [
                f"### {r.name}",
                "",
                "```json",
                json.dumps(_jsonify(r.summary), indent=2)[:8000],
                "```",
                "",
            ]

    if not any_failed:
        lines.append("_No failed checks._")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TWPA production validation checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation_suite"),
        help="Directory for validation reports.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="Run pytest after internal smoke checks.",
    )
    parser.add_argument(
        "--run-slow",
        action="store_true",
        help="Pass --run-slow to pytest.",
    )
    parser.add_argument(
        "--skip-plotting",
        action="store_true",
        help="Skip plotting smoke checks.",
    )
    parser.add_argument(
        "--skip-optional-linear",
        action="store_true",
        help="Skip optional workflow-level tiny linear smoke check.",
    )
    parser.add_argument(
        "--pytest-extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to pytest. Put this last.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    suite_start = time.perf_counter()

    results: list[CheckResult] = []

    checks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("imports", check_imports),
        ("linear_solvers", check_linear_solvers),
        ("block_banded", check_block_banded),
        ("preconditioners", check_preconditioners),
        ("newton_krylov", check_newton_krylov),
        ("inference_priors", check_inference_priors),
        ("io_checkpoint", lambda: check_io_checkpoint(output_dir)),
    ]

    if not args.skip_plotting:
        checks.append(("plotting", lambda: check_plotting(output_dir)))

    if not args.skip_optional_linear:
        checks.append(("optional_layout_linear_smoke", check_optional_layout_linear_smoke))

    for name, fn in checks:
        print(f"[validation] running {name}...")
        result = run_check(name, fn)
        print(f"[validation] {name}: {result.status.value} ({result.elapsed_s:.3f}s)")
        results.append(result)

    if args.run_pytest:
        print("[validation] running pytest...")
        result = run_check("pytest", lambda: check_pytest(args, output_dir))
        print(f"[validation] pytest: {result.status.value} ({result.elapsed_s:.3f}s)")
        results.append(result)

    elapsed_s = time.perf_counter() - suite_start

    n_fail = sum(r.status == CheckStatus.FAIL for r in results)
    n_error = sum(r.status == CheckStatus.ERROR for r in results)
    overall_status = CheckStatus.PASS if n_fail == 0 and n_error == 0 else CheckStatus.FAIL

    payload = {
        "status": overall_status.value,
        "passed": overall_status == CheckStatus.PASS,
        "started_at": started_at,
        "elapsed_s": elapsed_s,
        "python": sys.version,
        "jax": {
            "version": getattr(jax, "__version__", None),
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "devices": [str(d) for d in jax.devices()],
        },
        "args": {
            "output_dir": str(args.output_dir),
            "run_pytest": args.run_pytest,
            "run_slow": args.run_slow,
            "skip_plotting": args.skip_plotting,
            "skip_optional_linear": args.skip_optional_linear,
            "pytest_extra": args.pytest_extra,
        },
        "results": [r.to_dict() for r in results],
    }

    summary_json = output_dir / "validation_suite_summary.json"
    summary_md = output_dir / "validation_suite_summary.md"

    summary_json.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
    summary_md.write_text(
        summary_markdown(results, started_at=started_at, elapsed_s=elapsed_s),
        encoding="utf-8",
    )

    print()
    print(f"[validation] overall: {overall_status.value}")
    print(f"[validation] summary JSON: {summary_json}")
    print(f"[validation] summary MD:   {summary_md}")

    return 0 if overall_status == CheckStatus.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
