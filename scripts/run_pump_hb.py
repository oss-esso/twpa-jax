#!/usr/bin/env python3
"""
Run pump-only harmonic balance for a TWPA ladder.

This script is the command-line entry point for the nonlinear pump-HB layer:

    layout
      -> optional coarsening / reduced effective layout
      -> nonlinear KI branch model
      -> pump-only FrequencyPlan
      -> distributed HB residual
      -> dense Newton solve
      -> pump profile summary
      -> JSON / NPZ / Markdown artifacts

Important
---------
The current backend is the dense/reference HB solver. It is meant for reduced
layouts and correctness validation, not direct 20,000-cell nonlinear HB.

Recommended usage for industrial 100 mm lines:

    1. validate the full 20,000-cell layout linearly with run_linear_validation.py
    2. use --coarsen-target-cells 100, 200, 500, ...
    3. sweep N_eff until pump profiles and gain converge
    4. only later replace the dense backend with block-banded/Newton-Krylov

Examples
--------
Small synthetic pump solve:

    python scripts/run_pump_hb.py ^
      --layout-kind synthetic-uniform ^
      --n-cells 32 ^
      --length-mm 1.0 ^
      --pump-frequency-ghz 6.0 ^
      --pump-current-rms-A 1e-8 ^
      --I-star-A 1e-3 ^
      --output-dir runs/pump_small

Industrial 100 mm layout, reduced to 200 cells:

    python scripts/run_pump_hb.py ^
      --layout-kind industrial ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --coarsen-target-cells 200 ^
      --pump-frequency-ghz 8.0 ^
      --pump-power-dbm -80 ^
      --I-star-A 1e-3 ^
      --output-dir runs/pump_100mm_neff200

Pump-power continuation:

    python scripts/run_pump_hb.py ^
      --layout-kind synthetic-uniform ^
      --n-cells 64 ^
      --length-mm 2.0 ^
      --pump-frequency-ghz 6.0 ^
      --continuation ^
      --continuation-kind current ^
      --start-current-rms-A 1e-10 ^
      --stop-current-rms-A 1e-7 ^
      --continuation-steps 9 ^
      --I-star-A 1e-3 ^
      --output-dir runs/pump_continuation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import numpy as np

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout
from twpa.core.params import NonlinearParams
from twpa.core.hb_fft import HBProjectionConfig
from twpa.linear.coarsening import (
    CoarseningConfig,
    CoarseningMethod,
    coarsen_layout,
    make_uniform_surrogate_layout,
)
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    PumpHBLadderResult,
    PumpContinuationKind,
    PumpContinuationResult,
    pump_solution_table,
    solve_pump_current_continuation,
    solve_pump_hb_ladder,
    solve_pump_power_dbm_continuation,
)
from twpa.nonlinear.distributed_hb import (
    DistributedHBConfig,
    DistributedHBTerminationKind,
)
from twpa.solvers.hb_solver import DenseNewtonConfig, LinearSolveMethod
from twpa.solvers.continuation import ContinuationSolverConfig
from twpa.workflows.industrial_100mm import (
    IndustrialLayoutSpec,
    build_industrial_layout,
)
from twpa.workflows.synthetic_benchmarks import (
    SyntheticLayoutKind,
    SyntheticLayoutSpec,
    build_synthetic_layout,
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


def write_npz(path: Path, **arrays: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pump-only harmonic balance for a TWPA ladder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Layout
    parser.add_argument(
        "--layout-kind",
        choices=[
            "industrial",
            "synthetic-uniform",
            "synthetic-stub",
            "synthetic-disorder",
            "synthetic-lossy",
        ],
        default="synthetic-uniform",
        help="Layout family to build.",
    )
    parser.add_argument("--name", default=None, help="Optional layout/run name.")
    parser.add_argument("--n-cells", type=int, default=64, help="Number of fine cells.")
    parser.add_argument("--length-mm", type=float, default=1.0, help="Line length in mm.")
    parser.add_argument("--z0-ohm", type=float, default=50.0, help="Reference/target impedance.")
    parser.add_argument(
        "--phase-velocity",
        type=float,
        default=1.20e8,
        help="Target phase velocity in m/s.",
    )
    parser.add_argument("--L-per-m-H", type=float, default=None, help="Explicit L per metre.")
    parser.add_argument("--C-per-m-F", type=float, default=None, help="Explicit C per metre.")
    parser.add_argument("--R-per-m-ohm", type=float, default=0.0, help="Series loss per metre.")
    parser.add_argument("--G-per-m-S", type=float, default=0.0, help="Shunt loss per metre.")

    parser.add_argument(
        "--stub-period-cells",
        type=int,
        default=0,
        help="Periodic stub-loading period in cells. Zero disables for industrial layout.",
    )
    parser.add_argument(
        "--stub-offset",
        type=int,
        default=0,
        help="Loaded-cell offset within the stub period.",
    )
    parser.add_argument(
        "--stub-fraction",
        type=float,
        default=0.0,
        help="Stub capacitance as fraction of base cell capacitance.",
    )
    parser.add_argument(
        "--stub-cap-F",
        type=float,
        default=0.0,
        help="Absolute loaded-cell stub capacitance.",
    )
    parser.add_argument(
        "--disorder-std-fraction",
        type=float,
        default=0.01,
        help="Multiplicative disorder std fraction for synthetic-disorder.",
    )
    parser.add_argument("--disorder-seed", type=int, default=123, help="Disorder seed.")

    # Coarsening
    parser.add_argument(
        "--coarsen-target-cells",
        type=int,
        default=None,
        help="If provided, reduce the layout before nonlinear HB.",
    )
    parser.add_argument(
        "--coarsen-method",
        choices=[m.value for m in CoarseningMethod],
        default=CoarseningMethod.EXACT_GROUP_SUM.value,
        help="Coarsening method.",
    )
    parser.add_argument(
        "--coarsen-factor",
        type=int,
        default=None,
        help="Fine cells per coarse cell. If omitted, inferred from target cells.",
    )
    parser.add_argument(
        "--allow-coarsen-remainder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow a final smaller coarsening group.",
    )

    # Nonlinear parameters
    parser.add_argument("--I-star-A", type=float, required=True, help="Kinetic current scale I*.")
    parser.add_argument(
        "--beta-nl",
        type=float,
        default=1.0,
        help="Cubic kinetic-inductance coefficient.",
    )
    parser.add_argument(
        "--quartic-coefficient",
        type=float,
        default=0.0,
        help="Optional quartic nonlinear correction.",
    )
    parser.add_argument(
        "--dc-bias-A",
        type=float,
        default=0.0,
        help="Optional DC bias current.",
    )

    # Pump drive
    parser.add_argument("--pump-frequency-ghz", type=float, required=True, help="Pump frequency.")
    drive_group = parser.add_mutually_exclusive_group(required=False)
    drive_group.add_argument(
        "--pump-current-rms-A",
        type=float,
        default=None,
        help="Equivalent Norton pump RMS current.",
    )
    drive_group.add_argument(
        "--pump-power-dbm",
        type=float,
        default=None,
        help="Available pump power in dBm.",
    )
    drive_group.add_argument(
        "--pump-power-W",
        type=float,
        default=None,
        help="Available pump power in watts.",
    )
    parser.add_argument(
        "--source-impedance-ohm",
        type=float,
        default=50.0,
        help="Source impedance for power/current conversion.",
    )
    parser.add_argument("--pump-phase-rad", type=float, default=0.0, help="Pump phase.")
    parser.add_argument("--pump-label", default="pump", help="Pump label in FrequencyPlan.")

    # Distributed boundary model
    parser.add_argument("--input-node", type=int, default=0, help="Input/source node.")
    parser.add_argument(
        "--output-node",
        type=int,
        default=-1,
        help="Output/load node. -1 means final node.",
    )
    parser.add_argument(
        "--termination-kind",
        choices=[k.value for k in DistributedHBTerminationKind],
        default=DistributedHBTerminationKind.SHUNT_CONDUCTANCE.value,
        help="Termination model.",
    )
    parser.add_argument(
        "--source-conductance-S",
        type=float,
        default=1.0 / 50.0,
        help="Input shunt source conductance.",
    )
    parser.add_argument(
        "--load-conductance-S",
        type=float,
        default=1.0 / 50.0,
        help="Output shunt load conductance.",
    )
    parser.add_argument(
        "--include-stub-capacitance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include layout C_stub in HB shunt capacitance.",
    )
    parser.add_argument(
        "--include-series-resistance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include layout R_series in branch KVL.",
    )
    parser.add_argument(
        "--use-layout-shunt-conductance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include layout G_shunt in node admittance.",
    )

    # HB frequency/projection
    parser.add_argument(
        "--n-pump-harmonics",
        type=int,
        default=3,
        help="Number of pump harmonics in the pump-only plan.",
    )
    parser.add_argument(
        "--include-negative-frequencies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include negative-frequency conjugate tones.",
    )
    parser.add_argument(
        "--include-dc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include DC tone in pump-only plan.",
    )
    parser.add_argument(
        "--time-samples",
        type=int,
        default=None,
        help="Number of time samples for HB projection. Omit for automatic.",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        default=8,
        help="HB projection oversampling factor.",
    )

    # Dense Newton solver
    parser.add_argument("--max-iter", type=int, default=50, help="Newton max iterations.")
    parser.add_argument("--abs-tol", type=float, default=1e-9, help="Absolute residual tolerance.")
    parser.add_argument("--rel-tol", type=float, default=1e-9, help="Relative residual tolerance.")
    parser.add_argument("--step-tol", type=float, default=1e-12, help="Step tolerance.")
    parser.add_argument(
        "--damping-initial",
        type=float,
        default=1.0,
        help="Initial Newton damping.",
    )
    parser.add_argument(
        "--damping-min",
        type=float,
        default=1e-6,
        help="Minimum line-search damping.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.0,
        help="Dense linear-solve regularization.",
    )
    parser.add_argument(
        "--linear-solve-method",
        choices=[m.value for m in LinearSolveMethod],
        default=LinearSolveMethod.AUTO.value,
        help="Dense Newton linear solve method.",
    )
    parser.add_argument(
        "--fail-on-nonconvergence",
        action="store_true",
        help="Raise/exit nonzero if Newton does not converge.",
    )

    # Continuation
    parser.add_argument(
        "--continuation",
        action="store_true",
        help="Run pump continuation instead of one direct solve.",
    )
    parser.add_argument(
        "--continuation-kind",
        choices=["current", "power-dbm"],
        default="current",
        help="Continuation variable.",
    )
    parser.add_argument("--start-current-rms-A", type=float, default=1e-10)
    parser.add_argument("--stop-current-rms-A", type=float, default=1e-8)
    parser.add_argument("--start-power-dbm", type=float, default=-120.0)
    parser.add_argument("--stop-power-dbm", type=float, default=-80.0)
    parser.add_argument("--continuation-steps", type=int, default=9)
    parser.add_argument(
        "--continuation-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adaptive continuation step shrinking/growth.",
    )
    parser.add_argument("--continuation-max-retries", type=int, default=8)
    parser.add_argument("--continuation-shrink-factor", type=float, default=0.5)
    parser.add_argument("--continuation-growth-factor", type=float, default=1.25)

    # Output/runtime
    parser.add_argument(
        "--output-dir",
        default="runs/pump_hb",
        help="Directory for JSON/NPZ/Markdown outputs.",
    )
    parser.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save numerical arrays as NPZ.",
    )
    parser.add_argument(
        "--fail-on-validation-error",
        action="store_true",
        help="Return nonzero exit code if solve/continuation fails.",
    )
    parser.add_argument(
        "--jax-enable-x64",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable JAX x64.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose solver/report output.",
    )

    return parser


# ---------------------------------------------------------------------------
# Object construction
# ---------------------------------------------------------------------------

def build_layout_from_args(args: argparse.Namespace) -> LineLayout:
    length_m = args.length_mm * 1e-3
    name = args.name

    if args.layout_kind == "industrial":
        spec = IndustrialLayoutSpec(
            length_m=length_m,
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
            name=name or f"industrial_{args.length_mm:g}mm_{args.n_cells}cell",
        )
        return build_industrial_layout(spec)

    kind_map = {
        "synthetic-uniform": SyntheticLayoutKind.UNIFORM,
        "synthetic-stub": SyntheticLayoutKind.STUB_PERIODIC,
        "synthetic-disorder": SyntheticLayoutKind.WEAK_DISORDER,
        "synthetic-lossy": SyntheticLayoutKind.LOSSY_UNIFORM,
    }

    spec = SyntheticLayoutSpec(
        kind=kind_map[args.layout_kind],
        n_cells=args.n_cells,
        length_m=length_m,
        z0_ohm=args.z0_ohm,
        phase_velocity_m_per_s=args.phase_velocity,
        L_per_m_H=args.L_per_m_H,
        C_per_m_F=args.C_per_m_F,
        R_per_m_ohm=args.R_per_m_ohm,
        G_per_m_S=args.G_per_m_S,
        stub_period_cells=max(1, args.stub_period_cells or 1),
        stub_fraction=args.stub_fraction,
        disorder_std_fraction=args.disorder_std_fraction,
        disorder_seed=args.disorder_seed,
        name=name or f"{args.layout_kind}_{args.length_mm:g}mm_{args.n_cells}cell",
    )
    return build_synthetic_layout(spec)


def maybe_coarsen_layout(layout: LineLayout, args: argparse.Namespace) -> tuple[LineLayout, dict[str, Any]]:
    if args.coarsen_target_cells is None:
        return layout, {
            "enabled": False,
            "message": "coarsening disabled",
            "original_n_cells": layout.n_cells,
            "reduced_n_cells": layout.n_cells,
        }

    target = int(args.coarsen_target_cells)
    if target <= 0:
        raise ValueError("--coarsen-target-cells must be positive")
    if target >= layout.n_cells:
        return layout, {
            "enabled": True,
            "message": "target cell count >= original; using original layout",
            "original_n_cells": layout.n_cells,
            "reduced_n_cells": layout.n_cells,
        }

    method = CoarseningMethod(args.coarsen_method)

    if method == CoarseningMethod.UNIFORM_SURROGATE:
        reduced = make_uniform_surrogate_layout(
            layout,
            target_n_cells=target,
            name=f"{layout.name}_Neff{target}",
        )
        return reduced, {
            "enabled": True,
            "method": method.value,
            "original_n_cells": layout.n_cells,
            "target_n_cells": target,
            "reduced_n_cells": reduced.n_cells,
            "warning": "uniform surrogate can destroy periodic loading physics",
        }

    factor = args.coarsen_factor
    if factor is None:
        factor = max(1, int(round(layout.n_cells / target)))

    result = coarsen_layout(
        layout,
        CoarseningConfig(
            method=method,
            factor=factor,
            target_n_cells=target if method == CoarseningMethod.REPEAT_SUPERCELL else None,
            allow_remainder=args.allow_coarsen_remainder,
        ),
        name=f"{layout.name}_Neff{target}",
    )

    return result.reduced, {
        "enabled": True,
        "method": method.value,
        "factor": factor,
        "original_n_cells": layout.n_cells,
        "target_n_cells": target,
        "reduced_n_cells": result.reduced.n_cells,
        "report": result.report,
    }


def build_nonlinear_params_from_args(args: argparse.Namespace) -> NonlinearParams:
    return NonlinearParams(
        I_star_A=args.I_star_A,
        beta_nl=args.beta_nl,
        quartic_coefficient=args.quartic_coefficient,
        dc_bias_A=args.dc_bias_A,
    )


def build_pump_drive_from_args(args: argparse.Namespace) -> PumpDriveConfig:
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


def build_distributed_config_from_args(args: argparse.Namespace) -> DistributedHBConfig:
    return DistributedHBConfig(
        input_node=args.input_node,
        output_node=args.output_node,
        termination_kind=DistributedHBTerminationKind(args.termination_kind),
        source_conductance_S=args.source_conductance_S,
        load_conductance_S=args.load_conductance_S,
        include_stub_capacitance=args.include_stub_capacitance,
        include_series_resistance=args.include_series_resistance,
        use_layout_shunt_conductance=args.use_layout_shunt_conductance,
        name="pump_hb_distributed",
    )


def build_projection_config_from_args(args: argparse.Namespace) -> HBProjectionConfig:
    return HBProjectionConfig(
        n_time_samples=args.time_samples,
        oversampling=args.oversampling,
        force_real_time_signal=True,
        enforce_conjugate_symmetry=True,
    )


def build_solver_config_from_args(args: argparse.Namespace) -> DenseNewtonConfig:
    return DenseNewtonConfig(
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


def build_pump_config_from_args(args: argparse.Namespace) -> PumpHBLadderConfig:
    return PumpHBLadderConfig(
        n_pump_harmonics=args.n_pump_harmonics,
        include_negative_frequencies=args.include_negative_frequencies,
        include_dc=args.include_dc,
        distributed=build_distributed_config_from_args(args),
        projection=build_projection_config_from_args(args),
        solver=build_solver_config_from_args(args),
        name="pump_hb_cli",
    )


def build_continuation_config_from_args(args: argparse.Namespace) -> ContinuationSolverConfig:
    return ContinuationSolverConfig(
        adaptive=args.continuation_adaptive,
        max_step_retries=args.continuation_max_retries,
        shrink_factor=args.continuation_shrink_factor,
        growth_factor=args.continuation_growth_factor,
        stop_on_failure=True,
        reuse_previous_solution=True,
        use_secant_predictor=True,
    )


# ---------------------------------------------------------------------------
# Main workflows
# ---------------------------------------------------------------------------

def run_direct_pump_solve(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig,
    args: argparse.Namespace,
) -> PumpHBLadderResult:
    print("[pump-hb] running direct pump solve")
    print(f"[pump-hb] pump frequency: {drive.pump_frequency_hz / 1e9:.9g} GHz")
    print(f"[pump-hb] pump Norton RMS current: {drive.current_rms_A:.9e} A")
    print(f"[pump-hb] available pump power: {drive.available_power_dbm:.9g} dBm")

    return solve_pump_hb_ladder(
        layout,
        nonlinear_params,
        drive=drive,
        pump_config=pump_config,
        metadata={
            "cli": "scripts/run_pump_hb.py",
            "mode": "direct",
        },
    )


def run_pump_continuation(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig,
    args: argparse.Namespace,
) -> PumpContinuationResult:
    print("[pump-hb] running pump continuation")

    cont_cfg = build_continuation_config_from_args(args)

    if args.continuation_kind == "power-dbm":
        print(
            "[pump-hb] continuation over available power: "
            f"{args.start_power_dbm:.6g} -> {args.stop_power_dbm:.6g} dBm "
            f"({args.continuation_steps} steps)"
        )
        return solve_pump_power_dbm_continuation(
            layout,
            nonlinear_params,
            pump_frequency_hz=drive.pump_frequency_hz,
            start_power_dbm=args.start_power_dbm,
            stop_power_dbm=args.stop_power_dbm,
            n_steps=args.continuation_steps,
            source_impedance_ohm=drive.source_impedance_ohm,
            pump_config=pump_config,
            continuation_config=cont_cfg,
        )

    print(
        "[pump-hb] continuation over Norton RMS current: "
        f"{args.start_current_rms_A:.6e} -> {args.stop_current_rms_A:.6e} A "
        f"({args.continuation_steps} steps)"
    )
    return solve_pump_current_continuation(
        layout,
        nonlinear_params,
        pump_frequency_hz=drive.pump_frequency_hz,
        start_current_rms_A=args.start_current_rms_A,
        stop_current_rms_A=args.stop_current_rms_A,
        n_steps=args.continuation_steps,
        source_impedance_ohm=drive.source_impedance_ohm,
        pump_config=pump_config,
        continuation_config=cont_cfg,
    )


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def make_direct_summary(
    *,
    fine_layout: LineLayout,
    hb_layout: LineLayout,
    coarsening_report: dict[str, Any],
    nonlinear_params: NonlinearParams,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig,
    result: PumpHBLadderResult,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "mode": "direct",
        "passed": bool(result.converged),
        "fine_layout": fine_layout.summary(),
        "hb_layout": hb_layout.summary(),
        "coarsening": coarsening_report,
        "nonlinear_params": nonlinear_params.to_dict(),
        "drive": drive.to_dict(),
        "pump_config": pump_config.to_dict(),
        "result": result.to_dict(),
        "solver_summary": result.solver_result.report.summary_line(),
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "cli_args": vars(args),
    }


def make_continuation_summary(
    *,
    fine_layout: LineLayout,
    hb_layout: LineLayout,
    coarsening_report: dict[str, Any],
    nonlinear_params: NonlinearParams,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig,
    result: PumpContinuationResult,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "mode": "continuation",
        "passed": bool(result.converged),
        "fine_layout": fine_layout.summary(),
        "hb_layout": hb_layout.summary(),
        "coarsening": coarsening_report,
        "nonlinear_params": nonlinear_params.to_dict(),
        "base_drive": drive.to_dict(),
        "pump_config": pump_config.to_dict(),
        "continuation": result.to_dict(),
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "cli_args": vars(args),
    }


def export_direct_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    result: PumpHBLadderResult,
    *,
    save_npz: bool,
) -> dict[str, str]:
    paths: dict[str, str] = {}

    paths["summary_json"] = str(write_json(output_dir / "pump_hb_summary.json", summary))

    md = make_direct_markdown_summary(summary, result)
    md_path = output_dir / "pump_hb_summary.md"
    md_path.write_text(md, encoding="utf-8")
    paths["summary_md"] = str(md_path)

    if save_npz:
        paths["solution_npz"] = str(
            write_npz(
                output_dir / "pump_hb_solution.npz",
                frequencies_hz=result.frequency_plan.frequencies_hz,
                node_voltage_coeffs_V=result.state.node_voltage_coeffs_V,
                branch_current_coeffs_A=result.state.branch_current_coeffs_A,
                injected_current_coeffs_A=result.distributed_result.injected_current_coeffs_A,
                residual_kcl_A=result.residual.kcl_A,
                residual_branch_kvl_V=result.residual.branch_kvl_V,
            )
        )

    return paths


def export_continuation_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    result: PumpContinuationResult,
    *,
    save_npz: bool,
) -> dict[str, str]:
    paths: dict[str, str] = {}

    paths["summary_json"] = str(write_json(output_dir / "pump_hb_continuation_summary.json", summary))

    md = make_continuation_markdown_summary(summary, result)
    md_path = output_dir / "pump_hb_continuation_summary.md"
    md_path.write_text(md, encoding="utf-8")
    paths["summary_md"] = str(md_path)

    if save_npz:
        values = jnp.asarray(result.values, dtype=jnp.float64)

        arrays: dict[str, Any] = {
            "continuation_values": values,
        }

        if result.final_state is not None:
            arrays["final_node_voltage_coeffs_V"] = result.final_state.node_voltage_coeffs_V
            arrays["final_branch_current_coeffs_A"] = result.final_state.branch_current_coeffs_A

        paths["continuation_npz"] = str(
            write_npz(
                output_dir / "pump_hb_continuation_arrays.npz",
                **arrays,
            )
        )

    return paths


# ---------------------------------------------------------------------------
# Markdown reports
# ---------------------------------------------------------------------------

def make_direct_markdown_summary(
    summary: dict[str, Any],
    result: PumpHBLadderResult,
) -> str:
    profile = result.profile
    solver = result.solver_result.report

    lines = [
        "# Pump HB summary",
        "",
        f"- status: `{'PASS' if summary['passed'] else 'FAIL'}`",
        f"- layout: `{summary['hb_layout']['name']}`",
        f"- cells used in HB: `{summary['hb_layout']['n_cells']}`",
        f"- fine layout cells: `{summary['fine_layout']['n_cells']}`",
        f"- pump frequency: `{result.drive.pump_frequency_hz / 1e9:.9g} GHz`",
        f"- available pump power: `{result.drive.available_power_dbm:.9g} dBm`",
        f"- Norton RMS current: `{result.drive.current_rms_A:.9e} A`",
        "",
        "## Pump solution",
        "",
        pump_solution_table(result),
        "",
        "## Solver",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| status | `{solver.status.value}` |",
        f"| converged | `{solver.converged}` |",
        f"| iterations | `{solver.iterations}` |",
        f"| initial residual | `{solver.initial_residual_norm:.6e}` |",
        f"| final residual | `{solver.final_residual_norm:.6e}` |",
        f"| relative residual | `{solver.final_relative_residual_norm:.6e}` |",
        f"| unknown size | `{solver.unknown_size}` |",
        f"| residual size | `{solver.residual_size}` |",
        "",
        "## Profile",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| max node voltage abs V | `{profile.max_node_voltage_abs_V:.9e}` |",
        f"| max branch current coeff abs A | `{profile.max_branch_current_abs_A:.9e}` |",
        f"| max branch current peak time A | `{profile.max_branch_current_peak_time_A:.9e}` |",
        f"| max I/Istar | `{profile.max_pump_current_ratio:.9e}` |",
        f"| input pump voltage abs V | `{profile.input_pump_voltage_abs_V:.9e}` |",
        f"| output pump voltage abs V | `{profile.output_pump_voltage_abs_V:.9e}` |",
        f"| output/input voltage gain dB | `{profile.output_to_input_voltage_gain_db:.9g}` |",
    ]

    if summary["coarsening"].get("enabled"):
        lines += [
            "",
            "## Coarsening",
            "",
            f"- method: `{summary['coarsening'].get('method')}`",
            f"- original cells: `{summary['coarsening'].get('original_n_cells')}`",
            f"- reduced cells: `{summary['coarsening'].get('reduced_n_cells')}`",
        ]

    return "\n".join(lines)


def make_continuation_markdown_summary(
    summary: dict[str, Any],
    result: PumpContinuationResult,
) -> str:
    cont = result.continuation

    lines = [
        "# Pump HB continuation summary",
        "",
        f"- status: `{'PASS' if summary['passed'] else 'FAIL'}`",
        f"- continuation kind: `{result.continuation_kind.value}`",
        f"- accepted steps: `{cont.n_accepted}`",
        f"- failed steps: `{cont.n_failed_steps}`",
        f"- last value: `{cont.last_value}`",
        "",
        "## Steps",
        "",
        "| step | target | status | accepted | retries | final residual | message |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]

    for report in cont.step_reports:
        final_res = (
            float("nan")
            if report.solver_report is None
            else report.solver_report.final_residual_norm
        )
        lines.append(
            f"| {report.step_index} | {report.target_value:.9g} | "
            f"`{report.status.value}` | {int(report.accepted)} | "
            f"{report.retries} | {final_res:.6e} | {report.message} |"
        )

    if summary["coarsening"].get("enabled"):
        lines += [
            "",
            "## Coarsening",
            "",
            f"- method: `{summary['coarsening'].get('method')}`",
            f"- original cells: `{summary['coarsening'].get('original_n_cells')}`",
            f"- reduced cells: `{summary['coarsening'].get('reduced_n_cells')}`",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", bool(args.jax_enable_x64))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fine_layout = build_layout_from_args(args)
    hb_layout, coarsening_report = maybe_coarsen_layout(fine_layout, args)

    nonlinear_params = build_nonlinear_params_from_args(args)
    drive = build_pump_drive_from_args(args)
    pump_config = build_pump_config_from_args(args)

    print(f"[pump-hb] JAX backend: {jax.default_backend()}")
    print(f"[pump-hb] fine layout: {fine_layout.name}")
    print(f"[pump-hb] fine cells: {fine_layout.n_cells}")
    print(f"[pump-hb] HB layout: {hb_layout.name}")
    print(f"[pump-hb] HB cells: {hb_layout.n_cells}")
    print(f"[pump-hb] HB length: {hb_layout.total_length_m:.6g} m")
    print(f"[pump-hb] I*: {nonlinear_params.I_star_A:.9e} A")

    if args.continuation:
        cont_result = run_pump_continuation(
            hb_layout,
            nonlinear_params,
            drive,
            pump_config,
            args,
        )
        summary = make_continuation_summary(
            fine_layout=fine_layout,
            hb_layout=hb_layout,
            coarsening_report=coarsening_report,
            nonlinear_params=nonlinear_params,
            drive=drive,
            pump_config=pump_config,
            result=cont_result,
            args=args,
        )
        paths = export_continuation_artifacts(
            output_dir,
            summary,
            cont_result,
            save_npz=args.save_npz,
        )
        summary["artifact_paths"] = paths
        write_json(output_dir / "pump_hb_continuation_summary.json", summary)

        print(f"[pump-hb] continuation status: {'PASS' if summary['passed'] else 'FAIL'}")
        print(f"[pump-hb] accepted steps: {cont_result.continuation.n_accepted}")
        print(f"[pump-hb] failed steps: {cont_result.continuation.n_failed_steps}")

    else:
        direct_result = run_direct_pump_solve(
            hb_layout,
            nonlinear_params,
            drive,
            pump_config,
            args,
        )
        summary = make_direct_summary(
            fine_layout=fine_layout,
            hb_layout=hb_layout,
            coarsening_report=coarsening_report,
            nonlinear_params=nonlinear_params,
            drive=drive,
            pump_config=pump_config,
            result=direct_result,
            args=args,
        )
        paths = export_direct_artifacts(
            output_dir,
            summary,
            direct_result,
            save_npz=args.save_npz,
        )
        summary["artifact_paths"] = paths
        write_json(output_dir / "pump_hb_summary.json", summary)

        print(f"[pump-hb] direct solve status: {'PASS' if summary['passed'] else 'FAIL'}")
        print(f"[pump-hb] solver: {direct_result.solver_result.report.summary_line()}")
        print(
            "[pump-hb] max I/I*: "
            f"{direct_result.profile.max_pump_current_ratio:.6e}"
        )

    print("[pump-hb] artifacts:")
    for key, path in summary["artifact_paths"].items():
        print(f"  - {key}: {path}")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run(args)
    except Exception as exc:
        print(f"[pump-hb] ERROR: {exc}", file=sys.stderr)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "pump_hb_error.json",
            {
                "passed": False,
                "error": str(exc),
                "cli_args": vars(args),
            },
        )
        return 2

    if args.fail_on_validation_error and not summary["passed"]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
