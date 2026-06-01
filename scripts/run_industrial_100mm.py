#!/usr/bin/env python3
"""
Run the industrial 100 mm / 20,000-cell TWPA workflow.

This is the top-level command-line orchestrator for the production-facing stack:

    full 100 mm / 20,000-cell layout
      -> linear validation and dispersion extraction
      -> optional coarsening hierarchy
      -> reduced-layout pump HB
      -> optional small-signal gain map
      -> JSON / NPZ / Markdown artifacts

Important
---------
The current nonlinear HB backend is dense/reference. Therefore:

    - full 20,000-cell linear validation is supported,
    - full 20,000-cell nonlinear HB is not the intended path yet,
    - nonlinear pump/gain stages should use --pump-layout-target-cells,
    - convergence should be checked by repeating N_eff = 100, 200, 500, ...

Examples
--------
Full 20,000-cell linear-only run:

    python scripts/run_industrial_100mm.py ^
      --mode linear-only ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --f-min-ghz 1 ^
      --f-max-ghz 16 ^
      --n-frequency-points 401 ^
      --output-dir runs/industrial_linear ^
      --save-artifacts

Reduced pump-HB run:

    python scripts/run_industrial_100mm.py ^
      --mode pump-only ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --pump-layout-target-cells 200 ^
      --pump-frequency-ghz 8 ^
      --pump-power-dbm -80 ^
      --I-star-A 1e-3 ^
      --output-dir runs/industrial_pump_neff200 ^
      --save-artifacts

Reduced pump + gain map:

    python scripts/run_industrial_100mm.py ^
      --mode full ^
      --enable-gain ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --pump-layout-target-cells 200 ^
      --pump-frequency-ghz 8 ^
      --pump-power-dbm -80 ^
      --signal-min-ghz 4 ^
      --signal-max-ghz 7 ^
      --n-signal-points 21 ^
      --I-star-A 1e-3 ^
      --output-dir runs/industrial_gain_neff200 ^
      --save-artifacts
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
from twpa.core.hb_fft import HBProjectionConfig
from twpa.linear.cells import CellModelConfig, CellModelKind
from twpa.linear.cascade import CascadeConfig, CascadeStrategy
from twpa.linear.coarsening import CoarseningHierarchyConfig, CoarseningMethod
from twpa.linear.dispersion import DispersionConfig, DispersionExtractionMethod, StopbandMetric
from twpa.nonlinear.distributed_hb import DistributedHBConfig, DistributedHBTerminationKind
from twpa.nonlinear.gain import GainSolveConfig, GainSweepConfig
from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig, PumpHBLadderConfig
from twpa.solvers.hb_solver import DenseNewtonConfig, LinearSolveMethod
from twpa.workflows.industrial_100mm import (
    Industrial100mmWorkflowConfig,
    IndustrialCoarseningStageConfig,
    IndustrialGainStageConfig,
    IndustrialLayoutSpec,
    IndustrialLinearStageConfig,
    IndustrialPumpStageConfig,
    IndustrialRunMode,
    run_industrial_100mm_workflow,
    summarize_workflow_markdown,
)


# ---------------------------------------------------------------------------
# Serialization
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
        description="Run industrial 100 mm / 20,000-cell TWPA workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["linear-only", "pump-only", "gain", "full"],
        default="linear-only",
        help="Workflow mode.",
    )

    parser.add_argument("--name", default="industrial_100mm_cli")
    parser.add_argument("--n-cells", type=int, default=20_000)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity", type=float, default=1.20e8)
    parser.add_argument("--L-per-m-H", type=float, default=None)
    parser.add_argument("--C-per-m-F", type=float, default=None)
    parser.add_argument("--R-per-m-ohm", type=float, default=0.0)
    parser.add_argument("--G-per-m-S", type=float, default=0.0)

    parser.add_argument("--stub-period-cells", type=int, default=0)
    parser.add_argument("--stub-offset", type=int, default=0)
    parser.add_argument("--stub-fraction", type=float, default=0.0)
    parser.add_argument("--stub-cap-F", type=float, default=0.0)

    parser.add_argument("--f-min-ghz", type=float, default=1.0)
    parser.add_argument("--f-max-ghz", type=float, default=16.0)
    parser.add_argument("--n-frequency-points", type=int, default=401)
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
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--cells-per-supercell", type=int, default=1)
    parser.add_argument(
        "--dispersion-method",
        choices=[m.value for m in DispersionExtractionMethod],
        default=DispersionExtractionMethod.BOTH.value,
    )
    parser.add_argument(
        "--stopband-metric",
        choices=[m.value for m in StopbandMetric],
        default=StopbandMetric.BOTH.value,
    )
    parser.add_argument("--stopband-s21-threshold-db", type=float, default=-10.0)
    parser.add_argument("--stopband-alpha-threshold", type=float, default=1.0)
    parser.add_argument(
        "--expect-stopband",
        choices=["yes", "no", "unknown"],
        default="unknown",
    )

    parser.add_argument(
        "--coarsening-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pump-layout-target-cells", type=int, default=200)
    parser.add_argument(
        "--coarsening-target-cells",
        type=int,
        nargs="*",
        default=[50, 100, 200, 500, 1000, 2000, 5000],
        help="Hierarchy target cell counts.",
    )
    parser.add_argument(
        "--coarsening-method",
        choices=[m.value for m in CoarseningMethod],
        default=CoarseningMethod.EXACT_GROUP_SUM.value,
    )
    parser.add_argument(
        "--compare-coarsening-dispersion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--I-star-A", type=float, default=None)
    parser.add_argument("--beta-nl", type=float, default=1.0)
    parser.add_argument("--quartic-coefficient", type=float, default=0.0)
    parser.add_argument("--dc-bias-A", type=float, default=0.0)

    parser.add_argument("--pump-frequency-ghz", type=float, default=8.0)
    drive_group = parser.add_mutually_exclusive_group(required=False)
    drive_group.add_argument("--pump-current-rms-A", type=float, default=None)
    drive_group.add_argument("--pump-power-dbm", type=float, default=None)
    drive_group.add_argument("--pump-power-W", type=float, default=None)
    parser.add_argument("--source-impedance-ohm", type=float, default=50.0)
    parser.add_argument("--pump-phase-rad", type=float, default=0.0)
    parser.add_argument("--pump-label", default="pump")

    parser.add_argument("--input-node", type=int, default=0)
    parser.add_argument("--output-node", type=int, default=-1)
    parser.add_argument(
        "--termination-kind",
        choices=[k.value for k in DistributedHBTerminationKind],
        default=DistributedHBTerminationKind.SHUNT_CONDUCTANCE.value,
    )
    parser.add_argument("--source-conductance-S", type=float, default=1.0 / 50.0)
    parser.add_argument("--load-conductance-S", type=float, default=1.0 / 50.0)
    parser.add_argument(
        "--include-stub-capacitance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-series-resistance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-layout-shunt-conductance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--n-pump-harmonics", type=int, default=3)
    parser.add_argument(
        "--include-negative-frequencies",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--include-dc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-samples", type=int, default=None)
    parser.add_argument("--oversampling", type=int, default=8)

    parser.add_argument("--max-iter", type=int, default=50)
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
    parser.add_argument("--fail-on-nonconvergence", action="store_true")

    parser.add_argument(
        "--pump-sweep-ghz",
        type=float,
        nargs="*",
        default=[],
        help="Optional pump frequency sweep in GHz.",
    )

    parser.add_argument(
        "--enable-gain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gain stage even if mode is full/gain.",
    )
    parser.add_argument("--signal-min-ghz", type=float, default=4.0)
    parser.add_argument("--signal-max-ghz", type=float, default=7.0)
    parser.add_argument("--n-signal-points", type=int, default=21)
    parser.add_argument("--signal-label-prefix", default="signal")
    parser.add_argument("--idler-label-prefix", default="idler")
    parser.add_argument("--signal-current-rms-A", type=complex, default=1e-12 + 0j)
    parser.add_argument(
        "--set-signal-conjugate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-nonpositive-idlers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--output-dir", default="runs/industrial_100mm")
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-validation-error", action="store_true")
    parser.add_argument("--jax-enable-x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    return parser


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def mode_from_arg(value: str) -> IndustrialRunMode:
    return {
        "linear-only": IndustrialRunMode.LINEAR_ONLY,
        "pump-only": IndustrialRunMode.PUMP_ONLY,
        "gain": IndustrialRunMode.GAIN,
        "full": IndustrialRunMode.FULL,
    }[value]


def expected_stopband_from_arg(value: str) -> bool | None:
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def build_layout_spec(args: argparse.Namespace) -> IndustrialLayoutSpec:
    return IndustrialLayoutSpec(
        length_m=args.length_mm * 1e-3,
        n_cells=args.n_cells,
        z0_ohm=args.z0_ohm,
        phase_velocity_m_per_s=args.phase_velocity,
        L_per_m_H=args.L_per_m_H,
        C_per_m_F=args.C_per_m_F,
        R_per_m_ohm=args.R_per_m_ohm,
        G_per_m_S=args.G_per_m_S,
        stub_period_cells=args.stub_period_cells,
        stub_offset=args.stub_offset,
        C_stub_loaded_F=args.stub_cap_F,
        C_stub_loaded_fraction_of_base=args.stub_fraction,
        name=f"{args.name}_layout",
    )


def build_linear_config(args: argparse.Namespace) -> IndustrialLinearStageConfig:
    return IndustrialLinearStageConfig(
        frequency_min_hz=args.f_min_ghz * 1e9,
        frequency_max_hz=args.f_max_ghz * 1e9,
        n_frequency_points=args.n_frequency_points,
        cell_model=CellModelConfig(
            kind=CellModelKind(args.cell_model),
            include_stub_capacitance=args.include_stub_capacitance,
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
            cells_per_supercell=args.cells_per_supercell,
            stopband_s21_threshold_db=args.stopband_s21_threshold_db,
            stopband_alpha_threshold_np_per_m=args.stopband_alpha_threshold,
        ),
        expected_stopband=expected_stopband_from_arg(args.expect_stopband),
        stopband_metric=StopbandMetric(args.stopband_metric),
        stopband_s21_threshold_db=args.stopband_s21_threshold_db,
        stopband_alpha_threshold_np_per_m=args.stopband_alpha_threshold,
    )


def build_coarsening_config(args: argparse.Namespace) -> IndustrialCoarseningStageConfig:
    targets = tuple(int(x) for x in args.coarsening_target_cells)
    if args.pump_layout_target_cells not in targets:
        targets = tuple(sorted(set(targets + (args.pump_layout_target_cells,))))

    return IndustrialCoarseningStageConfig(
        enabled=args.coarsening_enabled,
        hierarchy=CoarseningHierarchyConfig(
            target_cell_counts=targets,
            method=CoarseningMethod(args.coarsening_method),
            preserve_supercells=True,
            cells_per_supercell=max(1, args.cells_per_supercell),
            include_original=True,
        ),
        compare_dispersion=args.compare_coarsening_dispersion,
    )


def build_nonlinear_params(args: argparse.Namespace) -> NonlinearParams | None:
    if args.I_star_A is None:
        return None
    return NonlinearParams(
        I_star_A=args.I_star_A,
        beta_nl=args.beta_nl,
        quartic_coefficient=args.quartic_coefficient,
        dc_bias_A=args.dc_bias_A,
    )


def build_pump_drive(args: argparse.Namespace) -> PumpDriveConfig:
    fp = args.pump_frequency_ghz * 1e9

    if args.pump_power_dbm is not None:
        return PumpDriveConfig.from_available_power_dbm(
            pump_frequency_hz=fp,
            power_dbm=args.pump_power_dbm,
            source_impedance_ohm=args.source_impedance_ohm,
            pump_label=args.pump_label,
            phase_rad=args.pump_phase_rad,
            input_node=args.input_node,
        )

    if args.pump_power_W is not None:
        return PumpDriveConfig.from_available_power_watt(
            pump_frequency_hz=fp,
            power_W=args.pump_power_W,
            source_impedance_ohm=args.source_impedance_ohm,
            pump_label=args.pump_label,
            phase_rad=args.pump_phase_rad,
            input_node=args.input_node,
        )

    current = 1e-8 if args.pump_current_rms_A is None else args.pump_current_rms_A
    return PumpDriveConfig.from_current_rms(
        pump_frequency_hz=fp,
        current_rms_A=current,
        source_impedance_ohm=args.source_impedance_ohm,
        pump_label=args.pump_label,
        phase_rad=args.pump_phase_rad,
        input_node=args.input_node,
    )


def build_pump_config(args: argparse.Namespace) -> PumpHBLadderConfig:
    distributed = DistributedHBConfig(
        input_node=args.input_node,
        output_node=args.output_node,
        termination_kind=DistributedHBTerminationKind(args.termination_kind),
        source_conductance_S=args.source_conductance_S,
        load_conductance_S=args.load_conductance_S,
        include_stub_capacitance=args.include_stub_capacitance,
        include_series_resistance=args.include_series_resistance,
        use_layout_shunt_conductance=args.use_layout_shunt_conductance,
        name="industrial_distributed_hb",
    )

    projection = HBProjectionConfig(
        n_time_samples=args.time_samples,
        oversampling=args.oversampling,
        force_real_time_signal=True,
        enforce_conjugate_symmetry=True,
    )

    solver = DenseNewtonConfig(
        max_iter=args.max_iter,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        step_tol=args.step_tol,
        damping_initial=args.damping_initial,
        damping_min=args.damping_min,
        regularization=args.regularization,
        linear_solve_method=LinearSolveMethod(args.linear_solve_method),
        fail_on_nonconvergence=args.fail_on_nonconvergence,
        verbose=args.verbose,
    )

    return PumpHBLadderConfig(
        n_pump_harmonics=args.n_pump_harmonics,
        include_negative_frequencies=args.include_negative_frequencies,
        include_dc=args.include_dc,
        distributed=distributed,
        projection=projection,
        solver=solver,
        name="industrial_pump_hb",
    )


def build_pump_stage_config(args: argparse.Namespace) -> IndustrialPumpStageConfig:
    return IndustrialPumpStageConfig(
        enabled=mode_from_arg(args.mode) in {
            IndustrialRunMode.PUMP_ONLY,
            IndustrialRunMode.GAIN,
            IndustrialRunMode.FULL,
        },
        pump_layout_target_n_cells=args.pump_layout_target_cells,
        pump_drive=build_pump_drive(args),
        pump_config=build_pump_config(args),
        nonlinear_params=build_nonlinear_params(args),
        sweep_frequencies_hz=tuple(float(x) * 1e9 for x in args.pump_sweep_ghz),
        reuse_previous_solution_in_sweep=True,
    )


def build_gain_stage_config(args: argparse.Namespace) -> IndustrialGainStageConfig:
    mode = mode_from_arg(args.mode)
    enabled = args.enable_gain or mode in {IndustrialRunMode.GAIN}
    if mode == IndustrialRunMode.FULL:
        enabled = args.enable_gain
    return IndustrialGainStageConfig(enabled=enabled)


def build_workflow_config(args: argparse.Namespace) -> Industrial100mmWorkflowConfig:
    return Industrial100mmWorkflowConfig(
        mode=mode_from_arg(args.mode),
        layout=build_layout_spec(args),
        linear=build_linear_config(args),
        coarsening=build_coarsening_config(args),
        pump=build_pump_stage_config(args),
        gain=build_gain_stage_config(args),
        output_dir=args.output_dir,
        save_artifacts=args.save_artifacts,
        name=args.name,
    )


# ---------------------------------------------------------------------------
# Gain-plan compatibility
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
    pump_label: str,
    signal_label: str,
    idler_label: str,
    n_pump_harmonics: int,
    include_negative: bool,
    include_dc: bool,
) -> Any:
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
        "Could not construct pump/signal/idler FrequencyPlan. "
        "Add make_pump_signal_idler_plan(...) to twpa.core.frequency_plan "
        f"or adapt this script. Errors: {errors}"
    )


def make_gain_factories(args: argparse.Namespace) -> tuple[Callable[[Any], Any], Callable[[Any], GainSweepConfig]]:
    signal_freqs = jnp.linspace(
        args.signal_min_ghz * 1e9,
        args.signal_max_ghz * 1e9,
        args.n_signal_points,
        dtype=jnp.float64,
    )

    def target_plan_factory(pump_result: Any) -> Any:
        drive = pump_result.drive
        fp = drive.pump_frequency_hz

        # The workflow gain stage expects one target plan for one sweep. To keep
        # labels unique, put every signal/idler pair in the same custom plan if
        # the project supports it. If not, this fallback constructs the first
        # pair only, and users should use scripts/run_gain_map.py for per-point
        # plans.
        fs0 = float(signal_freqs[0])
        fi0 = 2.0 * fp - fs0
        return make_target_gain_plan(
            pump_frequency_hz=fp,
            signal_frequency_hz=fs0,
            idler_frequency_hz=fi0,
            pump_label=args.pump_label,
            signal_label=f"{args.signal_label_prefix}_0",
            idler_label=f"{args.idler_label_prefix}_0",
            n_pump_harmonics=args.n_pump_harmonics,
            include_negative=args.include_negative_frequencies,
            include_dc=args.include_dc,
        )

    def sweep_config_factory(target_plan: Any) -> GainSweepConfig:
        # Conservative default: one gain point. Full multi-frequency map is
        # handled by scripts/run_gain_map.py because each point may require a
        # different FrequencyPlan depending on available constructors.
        point = GainSolveConfig(
            signal_label=f"{args.signal_label_prefix}_0",
            idler_label=f"{args.idler_label_prefix}_0",
            input_node=args.input_node,
            output_node=None if args.output_node < 0 else args.output_node,
            signal_current_rms_A=args.signal_current_rms_A,
            set_conjugate=args.set_signal_conjugate,
            input_impedance_ohm=args.source_impedance_ohm,
            output_impedance_ohm=1.0 / args.load_conductance_S if args.load_conductance_S > 0 else 50.0,
        )
        return GainSweepConfig(
            points=(point,),
            require_all_converged=True,
            name="industrial_gain_smoke",
        )

    return target_plan_factory, sweep_config_factory


# ---------------------------------------------------------------------------
# Run / reporting
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", bool(args.jax_enable_x64))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_workflow_config(args)

    gain_factories = (None, None)
    if cfg.gain.enabled:
        gain_factories = make_gain_factories(args)

    print("[industrial] starting workflow")
    print(f"[industrial] mode: {cfg.mode.value}")
    print(f"[industrial] fine cells: {cfg.layout.n_cells}")
    print(f"[industrial] length: {cfg.layout.length_m:.6g} m")
    print(f"[industrial] JAX backend: {jax.default_backend()}")

    if cfg.mode in {IndustrialRunMode.PUMP_ONLY, IndustrialRunMode.GAIN, IndustrialRunMode.FULL}:
        if cfg.pump.nonlinear_params is None:
            raise ValueError(
                "Pump/gain modes require --I-star-A. "
                "For linear-only validation, use --mode linear-only."
            )
        print(f"[industrial] reduced pump target cells: {cfg.pump.pump_layout_target_n_cells}")
        print(f"[industrial] pump frequency: {cfg.pump.pump_drive.pump_frequency_hz / 1e9:.9g} GHz")
        print(f"[industrial] pump current RMS: {cfg.pump.pump_drive.current_rms_A:.9e} A")
        print(f"[industrial] pump available power: {cfg.pump.pump_drive.available_power_dbm:.9g} dBm")

    result = run_industrial_100mm_workflow(
        cfg,
        target_plan_factory=gain_factories[0],
        sweep_config_factory=gain_factories[1],
    )

    summary = result.to_dict()
    summary["cli_args"] = vars(args)

    summary_path = write_json(output_dir / "industrial_100mm_cli_summary.json", summary)
    md_path = output_dir / "industrial_100mm_cli_summary.md"
    md_path.write_text(summarize_workflow_markdown(result), encoding="utf-8")

    summary["cli_artifacts"] = {
        "summary_json": str(summary_path),
        "summary_md": str(md_path),
    }
    write_json(output_dir / "industrial_100mm_cli_summary.json", summary)

    print(f"[industrial] final status: {result.status.value}")
    print(f"[industrial] passed: {result.passed}")
    print("[industrial] artifacts:")
    for key, path in summary.get("artifact_paths", {}).items():
        print(f"  - {key}: {path}")
    for key, path in summary["cli_artifacts"].items():
        print(f"  - {key}: {path}")

    if cfg.gain.enabled:
        print(
            "[industrial] note: this script's gain stage is a smoke path. "
            "Use scripts/run_gain_map.py for a full multi-frequency gain map."
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run(args)
    except Exception as exc:
        print(f"[industrial] ERROR: {exc}", file=sys.stderr)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "industrial_100mm_error.json",
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
