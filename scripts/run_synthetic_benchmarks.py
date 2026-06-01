#!/usr/bin/env python3
"""
Run synthetic regression benchmarks for the JAX-backed TWPA simulator.

This script exercises the simulator stack on controlled layouts:

    linear cascade
    MNA comparison
    dispersion extraction
    coarsening convergence
    one-node nonlinear HB
    distributed nonlinear HB
    pump-HB smoke solve
    optional gain smoke solve

It is intended as the first thing to run after adding or modifying solver code.

Examples
--------
Fast linear-only smoke suite:

    python scripts/run_synthetic_benchmarks.py ^
      --preset fast-linear ^
      --output-dir runs/synth_fast_linear ^
      --save-artifacts

Small nonlinear suite:

    python scripts/run_synthetic_benchmarks.py ^
      --preset small-nonlinear ^
      --I-star-A 1e-3 ^
      --pump-frequency-ghz 6 ^
      --pump-current-rms-A 1e-8 ^
      --output-dir runs/synth_small_nonlinear ^
      --save-artifacts

Custom suite:

    python scripts/run_synthetic_benchmarks.py ^
      --layout-kinds uniform stub disorder lossy ^
      --n-cells 32 64 ^
      --length-mm 1.0 ^
      --f-min-ghz 1 ^
      --f-max-ghz 12 ^
      --I-star-A 1e-3 ^
      --run-nonlinear ^
      --output-dir runs/synth_custom
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import numpy as np

import jax
import jax.numpy as jnp

from twpa.core import frequency_plan as frequency_plan_module
from twpa.core.params import NonlinearParams
from twpa.linear.cells import CellModelConfig, CellModelKind
from twpa.linear.cascade import CascadeConfig, CascadeStrategy
from twpa.linear.coarsening import CoarseningHierarchyConfig, CoarseningMethod
from twpa.linear.dispersion import DispersionConfig, DispersionExtractionMethod
from twpa.nonlinear.gain import GainSolveConfig, GainSweepConfig
from twpa.solvers.hb_solver import DenseNewtonConfig, LinearSolveMethod
from twpa.workflows.synthetic_benchmarks import (
    SyntheticBenchmarkConfig,
    SyntheticBenchmarkSuiteResult,
    SyntheticCoarseningBenchmarkConfig,
    SyntheticLayoutKind,
    SyntheticLayoutSpec,
    SyntheticLinearBenchmarkConfig,
    SyntheticNonlinearBenchmarkConfig,
    build_synthetic_layout,
    make_fast_linear_synthetic_config,
    make_small_nonlinear_synthetic_config,
    run_synthetic_benchmarks,
    summarize_synthetic_benchmarks_markdown,
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, complex):
        return {
            "real": float(np.real(obj)),
            "imag": float(np.imag(obj)),
            "abs": float(abs(obj)),
        }
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return jsonify(obj.to_dict())
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            if np.iscomplexobj(arr):
                return jsonify(complex(arr))
            return arr.item()
        return {
            "array_shape": tuple(int(s) for s in arr.shape),
            "array_dtype": str(arr.dtype),
            "min_abs": float(np.nanmin(np.abs(arr))) if arr.size else None,
            "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else None,
        }
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonify(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run synthetic TWPA simulator regression benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--preset",
        choices=["fast-linear", "small-nonlinear", "custom"],
        default="custom",
        help="Benchmark preset.",
    )

    parser.add_argument(
        "--layout-kinds",
        nargs="+",
        choices=["uniform", "stub", "disorder", "lossy"],
        default=["uniform", "stub", "disorder", "lossy"],
        help="Synthetic layout families for custom preset.",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        nargs="+",
        default=[32],
        help="Cell counts for custom layouts.",
    )
    parser.add_argument("--length-mm", type=float, default=1.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity", type=float, default=1.20e8)
    parser.add_argument("--L-per-m-H", type=float, default=None)
    parser.add_argument("--C-per-m-F", type=float, default=None)
    parser.add_argument("--R-per-m-ohm", type=float, default=0.0)
    parser.add_argument("--G-per-m-S", type=float, default=0.0)
    parser.add_argument("--stub-period-cells", type=int, default=8)
    parser.add_argument("--stub-fraction", type=float, default=0.20)
    parser.add_argument("--disorder-std-fraction", type=float, default=0.01)
    parser.add_argument("--disorder-seed", type=int, default=123)

    parser.add_argument("--f-min-ghz", type=float, default=1.0)
    parser.add_argument("--f-max-ghz", type=float, default=12.0)
    parser.add_argument("--n-frequency-points", type=int, default=151)

    parser.add_argument(
        "--cell-model",
        choices=[k.value for k in CellModelKind],
        default=CellModelKind.PI.value,
    )
    parser.add_argument(
        "--cascade-strategy",
        choices=[s.value for s in CascadeStrategy],
        default=CascadeStrategy.AUTO.value,
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--cells-per-supercell", type=int, default=8)
    parser.add_argument(
        "--dispersion-method",
        choices=[m.value for m in DispersionExtractionMethod],
        default=DispersionExtractionMethod.BOTH.value,
    )
    parser.add_argument(
        "--run-mna-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-uniform-baseline-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cutoff-safety-factor", type=float, default=2.0)

    parser.add_argument(
        "--coarsening-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--coarsening-target-cells",
        type=int,
        nargs="*",
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--coarsening-method",
        choices=[m.value for m in CoarseningMethod],
        default=CoarseningMethod.EXACT_GROUP_SUM.value,
    )

    parser.add_argument(
        "--run-nonlinear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable nonlinear stages for custom preset.",
    )
    parser.add_argument("--I-star-A", type=float, default=None)
    parser.add_argument("--beta-nl", type=float, default=1.0)
    parser.add_argument("--quartic-coefficient", type=float, default=0.0)
    parser.add_argument("--dc-bias-A", type=float, default=0.0)
    parser.add_argument("--pump-frequency-ghz", type=float, default=6.0)
    parser.add_argument("--pump-current-rms-A", type=float, default=1e-8)
    parser.add_argument("--n-pump-harmonics", type=int, default=3)
    parser.add_argument("--max-cells-for-dense-hb", type=int, default=64)
    parser.add_argument("--run-one-node", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-distributed-hb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-pump-hb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-gain-smoke", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--abs-tol", type=float, default=1e-9)
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    parser.add_argument("--step-tol", type=float, default=1e-12)
    parser.add_argument("--damping-initial", type=float, default=1.0)
    parser.add_argument("--damping-min", type=float, default=1e-6)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument(
        "--linear-solve-method",
        choices=[m.value for m in LinearSolveMethod],
        default=LinearSolveMethod.AUTO.value,
    )

    # Optional gain smoke settings.
    parser.add_argument("--gain-signal-frequency-ghz", type=float, default=None)
    parser.add_argument("--gain-idler-frequency-ghz", type=float, default=None)
    parser.add_argument("--gain-signal-label", default="signal")
    parser.add_argument("--gain-idler-label", default="idler")
    parser.add_argument("--gain-signal-current-rms-A", type=complex, default=1e-12 + 0j)

    parser.add_argument("--output-dir", default="runs/synthetic_benchmarks")
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-on-validation-error", action="store_true")
    parser.add_argument("--jax-enable-x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    return parser


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def build_nonlinear_params(args: argparse.Namespace) -> NonlinearParams | None:
    if args.I_star_A is None:
        return None
    return NonlinearParams(
        I_star_A=args.I_star_A,
        beta_nl=args.beta_nl,
        quartic_coefficient=args.quartic_coefficient,
        dc_bias_A=args.dc_bias_A,
    )


def synthetic_kind_from_cli(value: str) -> SyntheticLayoutKind:
    return {
        "uniform": SyntheticLayoutKind.UNIFORM,
        "stub": SyntheticLayoutKind.STUB_PERIODIC,
        "disorder": SyntheticLayoutKind.WEAK_DISORDER,
        "lossy": SyntheticLayoutKind.LOSSY_UNIFORM,
    }[value]


def build_custom_layout_specs(args: argparse.Namespace) -> tuple[SyntheticLayoutSpec, ...]:
    specs: list[SyntheticLayoutSpec] = []

    for kind_label in args.layout_kinds:
        kind = synthetic_kind_from_cli(kind_label)

        for n_cells in args.n_cells:
            name = f"custom_{kind_label}_{args.length_mm:g}mm_{n_cells}cell"

            # For lossy layouts, use explicit loss arguments if given; otherwise
            # assign a small nonzero loss so the case is meaningfully different.
            R_per_m = args.R_per_m_ohm
            G_per_m = args.G_per_m_S
            if kind == SyntheticLayoutKind.LOSSY_UNIFORM:
                if R_per_m == 0.0:
                    R_per_m = 20.0
                if G_per_m == 0.0:
                    G_per_m = 1e-6

            specs.append(
                SyntheticLayoutSpec(
                    kind=kind,
                    n_cells=int(n_cells),
                    length_m=args.length_mm * 1e-3,
                    z0_ohm=args.z0_ohm,
                    phase_velocity_m_per_s=args.phase_velocity,
                    L_per_m_H=args.L_per_m_H,
                    C_per_m_F=args.C_per_m_F,
                    R_per_m_ohm=R_per_m,
                    G_per_m_S=G_per_m,
                    stub_period_cells=max(1, args.stub_period_cells),
                    stub_fraction=args.stub_fraction,
                    disorder_std_fraction=args.disorder_std_fraction,
                    disorder_seed=args.disorder_seed,
                    name=name,
                )
            )

    return tuple(specs)


def build_linear_config(args: argparse.Namespace) -> SyntheticLinearBenchmarkConfig:
    return SyntheticLinearBenchmarkConfig(
        frequency_min_hz=args.f_min_ghz * 1e9,
        frequency_max_hz=args.f_max_ghz * 1e9,
        n_frequency_points=args.n_frequency_points,
        cell_model=CellModelConfig(
            kind=CellModelKind(args.cell_model),
            include_stub_capacitance=True,
            include_resonator_loading=True,
        ),
        cascade=CascadeConfig(
            strategy=CascadeStrategy(args.cascade_strategy),
            chunk_size=args.chunk_size,
            cells_per_supercell=args.cells_per_supercell,
            allow_remainder=True,
        ),
        dispersion=DispersionConfig(
            method=DispersionExtractionMethod(args.dispersion_method),
            cells_per_supercell=max(1, args.cells_per_supercell),
        ),
        run_mna_comparison=args.run_mna_comparison,
        run_uniform_baseline_comparison=args.run_uniform_baseline_comparison,
        cutoff_safety_factor=args.cutoff_safety_factor,
    )


def build_coarsening_config(args: argparse.Namespace) -> SyntheticCoarseningBenchmarkConfig:
    targets = tuple(int(x) for x in args.coarsening_target_cells)
    if not targets:
        targets = (16, 32, 64)

    return SyntheticCoarseningBenchmarkConfig(
        enabled=args.coarsening_enabled,
        hierarchy=CoarseningHierarchyConfig(
            target_cell_counts=targets,
            method=CoarseningMethod(args.coarsening_method),
            preserve_supercells=True,
            cells_per_supercell=1,
            include_original=True,
        ),
    )


def build_solver_config(args: argparse.Namespace) -> DenseNewtonConfig:
    return DenseNewtonConfig(
        max_iter=args.max_iter,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        step_tol=args.step_tol,
        damping_initial=args.damping_initial,
        damping_min=args.damping_min,
        regularization=args.regularization,
        linear_solve_method=LinearSolveMethod(args.linear_solve_method),
        verbose=args.verbose,
    )


def build_nonlinear_config(args: argparse.Namespace) -> SyntheticNonlinearBenchmarkConfig:
    nonlinear = build_nonlinear_params(args)

    if not args.run_nonlinear:
        nonlinear = None

    return SyntheticNonlinearBenchmarkConfig(
        nonlinear_params=nonlinear,
        pump_frequency_hz=args.pump_frequency_ghz * 1e9,
        pump_current_rms_A=args.pump_current_rms_A,
        n_pump_harmonics=args.n_pump_harmonics,
        max_cells_for_dense_hb=args.max_cells_for_dense_hb,
        run_one_node=args.run_one_node,
        run_distributed_hb=args.run_distributed_hb,
        run_pump_hb=args.run_pump_hb,
        run_gain_smoke=args.run_gain_smoke,
        solver=build_solver_config(args),
    )


def build_custom_config(args: argparse.Namespace) -> SyntheticBenchmarkConfig:
    return SyntheticBenchmarkConfig(
        layout_specs=build_custom_layout_specs(args),
        linear=build_linear_config(args),
        coarsening=build_coarsening_config(args),
        nonlinear=build_nonlinear_config(args),
        output_dir=args.output_dir,
        save_artifacts=args.save_artifacts,
        stop_on_error=args.stop_on_error,
        name="custom_synthetic_benchmarks",
    )


def build_config(args: argparse.Namespace) -> SyntheticBenchmarkConfig:
    if args.preset == "fast-linear":
        cfg = make_fast_linear_synthetic_config(
            output_dir=args.output_dir,
            save_artifacts=args.save_artifacts,
        )
        # Preserve top-level runtime controls.
        return cfg.with_updates(
            stop_on_error=args.stop_on_error,
        )

    if args.preset == "small-nonlinear":
        nonlinear = build_nonlinear_params(args)
        if nonlinear is None:
            raise ValueError("--preset small-nonlinear requires --I-star-A")

        cfg = make_small_nonlinear_synthetic_config(
            nonlinear_params=nonlinear,
            output_dir=args.output_dir,
            save_artifacts=args.save_artifacts,
        )

        return cfg.with_updates(
            nonlinear=replace_nonlinear_solver_and_runtime(cfg.nonlinear, args),
            stop_on_error=args.stop_on_error,
        )

    return build_custom_config(args)


def replace_nonlinear_solver_and_runtime(
    base: SyntheticNonlinearBenchmarkConfig,
    args: argparse.Namespace,
) -> SyntheticNonlinearBenchmarkConfig:
    return base.with_updates(
        pump_frequency_hz=args.pump_frequency_ghz * 1e9,
        pump_current_rms_A=args.pump_current_rms_A,
        n_pump_harmonics=args.n_pump_harmonics,
        max_cells_for_dense_hb=args.max_cells_for_dense_hb,
        run_one_node=args.run_one_node,
        run_distributed_hb=args.run_distributed_hb,
        run_pump_hb=args.run_pump_hb,
        run_gain_smoke=args.run_gain_smoke,
        solver=build_solver_config(args),
    )


# ---------------------------------------------------------------------------
# Optional gain smoke frequency-plan factories
# ---------------------------------------------------------------------------

def _try_call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**filtered)
    except TypeError:
        return fn(**kwargs)


def make_target_gain_plan(
    *,
    pump_frequency_hz: float,
    signal_frequency_hz: float,
    idler_frequency_hz: float,
    pump_label: str = "pump",
    signal_label: str = "signal",
    idler_label: str = "idler",
    n_pump_harmonics: int = 3,
    include_negative: bool = True,
    include_dc: bool = False,
) -> Any:
    """
    Compatibility helper for creating a pump/signal/idler FrequencyPlan.
    """
    fpmod = frequency_plan_module

    constructor_names = [
        "make_pump_signal_idler_plan",
        "make_signal_idler_plan",
        "make_dp4wm_plan",
        "make_dp4wm_frequency_plan",
        "make_gain_plan",
        "make_small_signal_plan",
    ]

    kwargs = {
        "pump_frequency_hz": pump_frequency_hz,
        "signal_frequency_hz": signal_frequency_hz,
        "idler_frequency_hz": idler_frequency_hz,
        "pump_label": pump_label,
        "signal_label": signal_label,
        "idler_label": idler_label,
        "n_pump_harmonics": n_pump_harmonics,
        "n_harmonics": n_pump_harmonics,
        "include_negative": include_negative,
        "include_negative_frequencies": include_negative,
        "include_dc": include_dc,
        "sort": "frequency",
    }

    errors: list[str] = []
    for name in constructor_names:
        fn = getattr(fpmod, name, None)
        if fn is None:
            continue
        try:
            return _try_call_with_supported_kwargs(fn, kwargs)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    frequencies: list[float] = []
    labels: list[str] = []

    def add(label: str, freq: float) -> None:
        labels.append(label)
        frequencies.append(float(freq))

    if include_dc:
        add("dc", 0.0)

    for h in range(1, n_pump_harmonics + 1):
        pos_label = pump_label if h == 1 else f"{h}{pump_label}"
        add(pos_label, h * pump_frequency_hz)
        if include_negative:
            add(f"-{pos_label}", -h * pump_frequency_hz)

    add(signal_label, signal_frequency_hz)
    add(idler_label, idler_frequency_hz)

    if include_negative:
        add(f"-{signal_label}", -signal_frequency_hz)
        add(f"-{idler_label}", -idler_frequency_hz)

    order = np.argsort(np.asarray(frequencies))
    frequencies_sorted = [frequencies[i] for i in order]
    labels_sorted = [labels[i] for i in order]

    generic_constructors = [
        getattr(fpmod, "make_frequency_plan", None),
        getattr(fpmod, "make_plan_from_frequencies", None),
        getattr(fpmod, "FrequencyPlan", None),
    ]

    candidate_kwargs = [
        {
            "frequencies_hz": jnp.asarray(frequencies_sorted, dtype=jnp.float64),
            "labels": tuple(labels_sorted),
            "reference_pump_hz": pump_frequency_hz,
            "kind": "custom",
        },
        {
            "frequencies_hz": jnp.asarray(frequencies_sorted, dtype=jnp.float64),
            "tone_labels": tuple(labels_sorted),
            "reference_pump_hz": pump_frequency_hz,
            "kind": "custom",
        },
        {
            "frequency_hz": jnp.asarray(frequencies_sorted, dtype=jnp.float64),
            "labels": tuple(labels_sorted),
            "reference_pump_hz": pump_frequency_hz,
        },
    ]

    for ctor in generic_constructors:
        if ctor is None:
            continue
        for kw in candidate_kwargs:
            try:
                return _try_call_with_supported_kwargs(ctor, kw)
            except Exception as exc:
                errors.append(f"{getattr(ctor, '__name__', ctor)}: {exc}")

    raise RuntimeError(
        "Could not construct pump/signal/idler FrequencyPlan for gain smoke. "
        "Add make_pump_signal_idler_plan(...) to twpa.core.frequency_plan. "
        f"Errors: {errors}"
    )


def build_gain_factories(args: argparse.Namespace) -> tuple[Callable[[Any], Any] | None, Callable[[Any], GainSweepConfig] | None]:
    if not args.run_gain_smoke:
        return None, None

    def target_plan_factory(pump_result: Any) -> Any:
        fp = pump_result.drive.pump_frequency_hz
        fs = (
            args.gain_signal_frequency_ghz * 1e9
            if args.gain_signal_frequency_ghz is not None
            else fp - 1.0e9
        )
        fi = (
            args.gain_idler_frequency_ghz * 1e9
            if args.gain_idler_frequency_ghz is not None
            else 2.0 * fp - fs
        )

        return make_target_gain_plan(
            pump_frequency_hz=fp,
            signal_frequency_hz=float(fs),
            idler_frequency_hz=float(fi),
            pump_label="pump",
            signal_label=args.gain_signal_label,
            idler_label=args.gain_idler_label,
            n_pump_harmonics=args.n_pump_harmonics,
            include_negative=True,
            include_dc=False,
        )

    def sweep_config_factory(target_plan: Any) -> GainSweepConfig:
        point = GainSolveConfig(
            signal_label=args.gain_signal_label,
            idler_label=args.gain_idler_label,
            input_node=0,
            output_node=None,
            signal_current_rms_A=args.gain_signal_current_rms_A,
            set_conjugate=True,
            input_impedance_ohm=50.0,
            output_impedance_ohm=50.0,
        )
        return GainSweepConfig(
            points=(point,),
            require_all_converged=True,
            name="synthetic_gain_smoke",
        )

    return target_plan_factory, sweep_config_factory


# ---------------------------------------------------------------------------
# Run / reporting
# ---------------------------------------------------------------------------

def print_suite_summary(result: SyntheticBenchmarkSuiteResult) -> None:
    print("[synthetic] final status:", result.status.value)
    print("[synthetic] passed:", result.passed)

    for layout_result in result.layout_results:
        print(
            f"[synthetic] {layout_result.layout.name}: "
            f"{layout_result.status.value}"
        )
        for stage in layout_result.stage_results:
            print(
                f"  - {stage.stage.value}: {stage.status.value} "
                f"({stage.elapsed_s:.3g} s)"
                + (f" ERROR={stage.error}" if stage.error else "")
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", bool(args.jax_enable_x64))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    target_plan_factory, sweep_config_factory = build_gain_factories(args)

    print("[synthetic] starting benchmark suite")
    print(f"[synthetic] preset: {args.preset}")
    print(f"[synthetic] layouts: {len(cfg.layout_specs)}")
    print(f"[synthetic] JAX backend: {jax.default_backend()}")
    print(f"[synthetic] x64: {bool(jax.config.jax_enable_x64)}")

    if cfg.nonlinear.nonlinear_params is None:
        print("[synthetic] nonlinear stages: disabled")
    else:
        print("[synthetic] nonlinear stages: enabled")
        print(f"[synthetic] I*: {cfg.nonlinear.nonlinear_params.I_star_A:.9e} A")
        print(f"[synthetic] pump frequency: {cfg.nonlinear.pump_frequency_hz / 1e9:.9g} GHz")
        print(f"[synthetic] pump current RMS: {cfg.nonlinear.pump_current_rms_A:.9e} A")
        print(f"[synthetic] dense-HB max cells: {cfg.nonlinear.max_cells_for_dense_hb}")

    result = run_synthetic_benchmarks(
        cfg,
        target_plan_factory=target_plan_factory,
        sweep_config_factory=sweep_config_factory,
    )

    print_suite_summary(result)

    # The workflow writes artifacts when save_artifacts=True. We also write a
    # CLI-specific summary so command-line args are preserved exactly.
    summary = result.to_dict()
    summary["cli_args"] = vars(args)

    cli_summary_path = write_json(output_dir / "synthetic_benchmarks_cli_summary.json", summary)
    cli_md_path = output_dir / "synthetic_benchmarks_cli_summary.md"
    cli_md_path.write_text(summarize_synthetic_benchmarks_markdown(result), encoding="utf-8")

    summary["cli_artifacts"] = {
        "summary_json": str(cli_summary_path),
        "summary_md": str(cli_md_path),
    }
    write_json(output_dir / "synthetic_benchmarks_cli_summary.json", summary)

    print("[synthetic] artifacts:")
    for key, path in result.artifact_paths.items():
        print(f"  - {key}: {path}")
    for key, path in summary["cli_artifacts"].items():
        print(f"  - {key}: {path}")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run(args)
    except Exception as exc:
        print(f"[synthetic] ERROR: {exc}", file=sys.stderr)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "synthetic_benchmarks_error.json",
            {
                "passed": False,
                "error": str(exc),
                "cli_args": vars(args),
            },
        )
        return 2

    if args.fail_on_validation_error and not summary.get("passed", False):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
