#!/usr/bin/env python3
"""
Run simulator calibration / parameter extraction.

This script is the command-line entry point for fitting simulator parameters to
measurement-like data.

Supported workflows
-------------------
1. Pump-off linear S-parameter calibration

       layout + S-parameter NPZ
           -> fit L/C/stub/loss scale factors
           -> calibrated layout
           -> residual/report artifacts

2. Pump-on gain calibration

       layout + nonlinear params + pump drive + gain NPZ
           -> pump HB
           -> small-signal gain sweep
           -> fit nonlinear/pump scale factors

3. Combined calibration

       S-parameter data + gain data
           -> joint residual

Expected S-parameter NPZ arrays
-------------------------------
Required:
    frequency_hz

Optional, at least one:
    s          complex array, shape (F, 2, 2)
    s21_db     real array, shape (F,)

Expected gain NPZ arrays
------------------------
Required:
    signal_frequency_hz
    signal_gain_db

Optional:
    idler_frequency_hz
    idler_conversion_db
    signal_labels
    idler_labels

Examples
--------
Linear pump-off calibration:

    python scripts/run_calibration.py ^
      --mode linear ^
      --layout-kind industrial ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --sparam-npz data/pump_off_sparams.npz ^
      --fit-L-scale ^
      --fit-C-scale ^
      --fit-C-stub-scale ^
      --output-dir runs/cal_linear

Gain calibration on reduced layout:

    python scripts/run_calibration.py ^
      --mode gain ^
      --layout-kind industrial ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --coarsen-target-cells 200 ^
      --gain-npz data/pump_on_gain.npz ^
      --pump-frequency-ghz 8 ^
      --pump-power-dbm -80 ^
      --I-star-A 1e-3 ^
      --fit-I-star-scale ^
      --fit-pump-current-scale ^
      --output-dir runs/cal_gain

Combined calibration:

    python scripts/run_calibration.py ^
      --mode combined ^
      --sparam-npz data/pump_off_sparams.npz ^
      --gain-npz data/pump_on_gain.npz ^
      --I-star-A 1e-3 ^
      --pump-frequency-ghz 8 ^
      --pump-power-dbm -80 ^
      --output-dir runs/cal_combined
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
from twpa.core.layout import LineLayout
from twpa.core.params import NonlinearParams
from twpa.linear.cascade import CascadeConfig, CascadeStrategy
from twpa.linear.cells import CellModelConfig, CellModelKind
from twpa.linear.coarsening import (
    CoarseningConfig,
    CoarseningMethod,
    coarsen_layout,
    make_uniform_surrogate_layout,
)
from twpa.nonlinear.gain import GainSolveConfig, GainSweepConfig
from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig, PumpHBLadderConfig
from twpa.nonlinear.distributed_hb import DistributedHBConfig, DistributedHBTerminationKind
from twpa.core.hb_fft import HBProjectionConfig
from twpa.solvers.hb_solver import DenseNewtonConfig, LinearSolveMethod
from twpa.workflows.industrial_100mm import IndustrialLayoutSpec, build_industrial_layout
from twpa.workflows.synthetic_benchmarks import (
    SyntheticLayoutKind,
    SyntheticLayoutSpec,
    build_synthetic_layout,
)
from twpa.workflows.calibration import (
    CalibrationOptimizerConfig,
    CalibrationOptimizerMethod,
    CalibrationParameterSpec,
    CalibrationTarget,
    CalibrationVectorSpec,
    GainCalibrationData,
    ParameterTransform,
    SParameterCalibrationData,
    calibrate,
    calibration_summary_markdown,
    export_calibration_artifacts,
    finite_difference_residual_jacobian,
    load_sparameter_calibration_npz,
    make_default_linear_parameter_spec,
    make_default_nonlinear_parameter_spec,
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
        description="Run TWPA simulator calibration / parameter extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["linear", "gain", "combined"],
        default="linear",
        help="Calibration mode.",
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
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-cells", type=int, default=256)
    parser.add_argument("--length-mm", type=float, default=2.0)
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
    parser.add_argument("--disorder-std-fraction", type=float, default=0.01)
    parser.add_argument("--disorder-seed", type=int, default=123)

    # Optional reduced calibration layout
    parser.add_argument("--coarsen-target-cells", type=int, default=None)
    parser.add_argument(
        "--coarsen-method",
        choices=[m.value for m in CoarseningMethod],
        default=CoarseningMethod.EXACT_GROUP_SUM.value,
    )
    parser.add_argument("--coarsen-factor", type=int, default=None)
    parser.add_argument(
        "--allow-coarsen-remainder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Linear simulation settings
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
        "--include-stub-capacitance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-resonator-loading",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Data
    parser.add_argument("--sparam-npz", default=None, help="Pump-off S-parameter NPZ.")
    parser.add_argument("--gain-npz", default=None, help="Pump-on gain NPZ.")
    parser.add_argument("--frequency-key", default="frequency_hz")
    parser.add_argument("--s-key", default="s")
    parser.add_argument("--s21-db-key", default="s21_db")
    parser.add_argument("--signal-frequency-key", default="signal_frequency_hz")
    parser.add_argument("--signal-gain-db-key", default="signal_gain_db")
    parser.add_argument("--idler-frequency-key", default="idler_frequency_hz")
    parser.add_argument("--idler-conversion-db-key", default="idler_conversion_db")
    parser.add_argument("--signal-labels-key", default="signal_labels")
    parser.add_argument("--idler-labels-key", default="idler_labels")

    parser.add_argument("--weight-complex", type=float, default=1.0)
    parser.add_argument("--weight-s21-db", type=float, default=1.0)
    parser.add_argument("--weight-signal-gain-db", type=float, default=1.0)
    parser.add_argument("--weight-idler-conversion-db", type=float, default=1.0)

    parser.add_argument("--use-s11", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-s21", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-s12", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-s22", action=argparse.BooleanOptionalAction, default=True)

    # Nonlinear/pump setup
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

    # HB solver
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

    # Parameter selection
    parser.add_argument("--fit-L-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-C-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-C-stub-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit-R-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit-G-scale", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--fit-I-star-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit-beta-nl-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit-pump-current-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit-pump-power-offset-db", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--L-scale-bounds", type=float, nargs=2, default=(0.5, 2.0))
    parser.add_argument("--C-scale-bounds", type=float, nargs=2, default=(0.5, 2.0))
    parser.add_argument("--C-stub-scale-bounds", type=float, nargs=2, default=(0.0, 5.0))
    parser.add_argument("--R-scale-bounds", type=float, nargs=2, default=(0.0, 20.0))
    parser.add_argument("--G-scale-bounds", type=float, nargs=2, default=(0.0, 20.0))
    parser.add_argument("--I-star-scale-bounds", type=float, nargs=2, default=(0.1, 10.0))
    parser.add_argument("--beta-nl-scale-bounds", type=float, nargs=2, default=(0.1, 10.0))
    parser.add_argument("--pump-current-scale-bounds", type=float, nargs=2, default=(0.1, 10.0))
    parser.add_argument("--pump-power-offset-db-bounds", type=float, nargs=2, default=(-6.0, 6.0))

    # Optimizer
    parser.add_argument(
        "--optimizer",
        choices=[m.value for m in CalibrationOptimizerMethod],
        default=CalibrationOptimizerMethod.AUTO.value,
    )
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--xtol", type=float, default=1e-8)
    parser.add_argument("--ftol", type=float, default=1e-8)
    parser.add_argument("--gtol", type=float, default=1e-8)
    parser.add_argument("--coordinate-initial-step-fraction", type=float, default=0.10)
    parser.add_argument("--coordinate-step-decay", type=float, default=0.5)
    parser.add_argument("--coordinate-min-step-fraction", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=1234)

    # Diagnostics/output
    parser.add_argument("--finite-difference-jacobian", action="store_true")
    parser.add_argument("--output-dir", default="runs/calibration")
    parser.add_argument("--fail-on-validation-error", action="store_true")
    parser.add_argument("--jax-enable-x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    return parser


# ---------------------------------------------------------------------------
# Layout and model construction
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
            name=name or f"cal_{args.layout_kind}_{args.length_mm:g}mm_{args.n_cells}cell",
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
        name=name or f"cal_{args.layout_kind}_{args.length_mm:g}mm_{args.n_cells}cell",
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
            name=f"{layout.name}_cal_Neff{target}",
        )
        return reduced, {
            "enabled": True,
            "method": method.value,
            "original_n_cells": layout.n_cells,
            "target_n_cells": target,
            "reduced_n_cells": reduced.n_cells,
            "warning": "uniform surrogate can destroy periodic/loading physics",
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
        name=f"{layout.name}_cal_Neff{target}",
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


def build_nonlinear_params(args: argparse.Namespace) -> NonlinearParams | None:
    if args.I_star_A is None:
        return None
    return NonlinearParams(
        I_star_A=args.I_star_A,
        beta_nl=args.beta_nl,
        quartic_coefficient=args.quartic_coefficient,
        dc_bias_A=args.dc_bias_A,
    )


def build_pump_drive(args: argparse.Namespace) -> PumpDriveConfig | None:
    if args.mode == "linear" and args.gain_npz is None:
        return None

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


def build_cell_model(args: argparse.Namespace) -> CellModelConfig:
    return CellModelConfig(
        kind=CellModelKind(args.cell_model),
        include_stub_capacitance=args.include_stub_capacitance,
        include_resonator_loading=args.include_resonator_loading,
    )


def build_cascade_config(args: argparse.Namespace) -> CascadeConfig:
    return CascadeConfig(
        strategy=CascadeStrategy(args.cascade_strategy),
        chunk_size=args.chunk_size,
        cells_per_supercell=args.cells_per_supercell,
        allow_remainder=True,
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
        name="calibration_distributed_hb",
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
        verbose=args.verbose,
    )

    return PumpHBLadderConfig(
        n_pump_harmonics=args.n_pump_harmonics,
        include_negative_frequencies=args.include_negative_frequencies,
        include_dc=args.include_dc,
        distributed=distributed,
        projection=projection,
        solver=solver,
        name="calibration_pump_hb",
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sparameter_data(args: argparse.Namespace) -> SParameterCalibrationData | None:
    if args.sparam_npz is None:
        return None

    data = load_sparameter_calibration_npz(
        args.sparam_npz,
        frequency_key=args.frequency_key,
        s_key=args.s_key,
        s21_db_key=args.s21_db_key,
        weight_complex=args.weight_complex,
        weight_s21_db=args.weight_s21_db,
    )

    return SParameterCalibrationData(
        frequency_hz=data.frequency_hz,
        s=data.s,
        s21_db=data.s21_db,
        weight_complex=args.weight_complex,
        weight_s21_db=args.weight_s21_db,
        use_s11=args.use_s11,
        use_s21=args.use_s21,
        use_s12=args.use_s12,
        use_s22=args.use_s22,
        metadata={
            **dict(data.metadata or {}),
            "source_path": args.sparam_npz,
        },
    )


def _load_optional_string_array(npz: Any, key: str, n: int, prefix: str) -> tuple[str, ...]:
    if key not in npz:
        return tuple(f"{prefix}_{i}" for i in range(n))

    arr = np.asarray(npz[key])
    if arr.shape[0] != n:
        raise ValueError(f"{key} length {arr.shape[0]} does not match expected {n}")

    labels = []
    for x in arr.tolist():
        if isinstance(x, bytes):
            labels.append(x.decode("utf-8"))
        else:
            labels.append(str(x))
    return tuple(labels)


def load_gain_data(args: argparse.Namespace) -> tuple[GainCalibrationData | None, dict[str, Any]]:
    if args.gain_npz is None:
        return None, {}

    npz = np.load(args.gain_npz, allow_pickle=True)

    if args.signal_frequency_key not in npz:
        raise ValueError(f"gain NPZ missing {args.signal_frequency_key!r}")
    if args.signal_gain_db_key not in npz:
        raise ValueError(f"gain NPZ missing {args.signal_gain_db_key!r}")

    signal_frequency_hz = jnp.asarray(npz[args.signal_frequency_key], dtype=jnp.float64)
    signal_gain_db = jnp.asarray(npz[args.signal_gain_db_key], dtype=jnp.float64)

    if signal_frequency_hz.ndim != 1:
        raise ValueError("signal_frequency_hz must be 1D")
    if signal_gain_db.shape != signal_frequency_hz.shape:
        raise ValueError("signal_gain_db must have same shape as signal_frequency_hz")

    n = int(signal_frequency_hz.shape[0])

    signal_labels = _load_optional_string_array(
        npz,
        args.signal_labels_key,
        n,
        "signal",
    )

    idler_frequency_hz = None
    if args.idler_frequency_key in npz:
        idler_frequency_hz = jnp.asarray(npz[args.idler_frequency_key], dtype=jnp.float64)
        if idler_frequency_hz.shape != signal_frequency_hz.shape:
            raise ValueError("idler_frequency_hz must have same shape as signal_frequency_hz")
    else:
        pump_frequency_hz = args.pump_frequency_ghz * 1e9
        idler_frequency_hz = 2.0 * pump_frequency_hz - signal_frequency_hz

    idler_conversion_db = None
    if args.idler_conversion_db_key in npz:
        idler_conversion_db = jnp.asarray(npz[args.idler_conversion_db_key], dtype=jnp.float64)
        if idler_conversion_db.shape != signal_frequency_hz.shape:
            raise ValueError("idler_conversion_db must have same shape as signal_frequency_hz")

    idler_labels = _load_optional_string_array(
        npz,
        args.idler_labels_key,
        n,
        "idler",
    )

    data = GainCalibrationData(
        signal_labels=signal_labels,
        signal_gain_db=signal_gain_db,
        idler_labels=idler_labels,
        idler_conversion_db=idler_conversion_db,
        weight_signal_gain_db=args.weight_signal_gain_db,
        weight_idler_conversion_db=args.weight_idler_conversion_db,
        metadata={
            "source_path": args.gain_npz,
            "signal_frequency_hz": signal_frequency_hz,
            "idler_frequency_hz": idler_frequency_hz,
        },
    )

    extra = {
        "signal_frequency_hz": signal_frequency_hz,
        "idler_frequency_hz": idler_frequency_hz,
        "signal_labels": signal_labels,
        "idler_labels": idler_labels,
    }

    return data, extra


# ---------------------------------------------------------------------------
# Frequency-plan/gain factories
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


def make_multi_gain_plan(
    *,
    pump_frequency_hz: float,
    signal_frequency_hz: Any,
    idler_frequency_hz: Any,
    pump_label: str,
    signal_labels: tuple[str, ...],
    idler_labels: tuple[str, ...],
    n_pump_harmonics: int,
    include_negative: bool,
    include_dc: bool,
) -> Any:
    """
    Build a target FrequencyPlan containing pump harmonics plus all signal/idler tones.
    """
    fpmod = frequency_plan_module

    # Prefer native project constructors if one supports vector signals.
    constructor_names = [
        "make_multi_signal_idler_plan",
        "make_gain_sweep_plan",
        "make_pump_signal_idler_plan",
        "make_signal_idler_plan",
        "make_dp4wm_plan",
        "make_gain_plan",
        "make_small_signal_plan",
    ]

    native_kwargs = {
        "pump_frequency_hz": pump_frequency_hz,
        "signal_frequency_hz": signal_frequency_hz,
        "signal_frequencies_hz": signal_frequency_hz,
        "idler_frequency_hz": idler_frequency_hz,
        "idler_frequencies_hz": idler_frequency_hz,
        "pump_label": pump_label,
        "signal_labels": signal_labels,
        "idler_labels": idler_labels,
        "signal_label": signal_labels[0],
        "idler_label": idler_labels[0],
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
            plan = _try_call_with_supported_kwargs(fn, native_kwargs)
            # Require all labels if possible.
            for label in signal_labels:
                plan.position_of_label(label)
            for label in idler_labels:
                plan.position_of_label(label)
            return plan
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

    fs = np.asarray(signal_frequency_hz, dtype=float)
    fi = np.asarray(idler_frequency_hz, dtype=float)

    for label, freq in zip(signal_labels, fs.tolist()):
        add(label, freq)
        if include_negative:
            add(f"-{label}", -freq)

    for label, freq in zip(idler_labels, fi.tolist()):
        add(label, freq)
        if include_negative:
            add(f"-{label}", -freq)

    # Remove exact duplicate labels by keeping first occurrence.
    seen: set[str] = set()
    unique_pairs: list[tuple[str, float]] = []
    for label, freq in zip(labels, frequencies):
        if label not in seen:
            unique_pairs.append((label, freq))
            seen.add(label)

    order = np.argsort(np.asarray([freq for _, freq in unique_pairs]))
    labels_sorted = [unique_pairs[i][0] for i in order]
    frequencies_sorted = [unique_pairs[i][1] for i in order]

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
        "Could not construct gain target FrequencyPlan. "
        "Add make_multi_signal_idler_plan(...) or make_frequency_plan(...) to "
        f"twpa.core.frequency_plan. Errors: {errors}"
    )


def build_gain_factories(
    args: argparse.Namespace,
    gain_extra: dict[str, Any],
) -> tuple[Callable[[Any], Any], Callable[[Any], GainSweepConfig]]:
    signal_frequency_hz = gain_extra["signal_frequency_hz"]
    idler_frequency_hz = gain_extra["idler_frequency_hz"]
    signal_labels = tuple(gain_extra["signal_labels"])
    idler_labels = tuple(gain_extra["idler_labels"])

    def target_plan_factory(pump_result: Any) -> Any:
        return make_multi_gain_plan(
            pump_frequency_hz=pump_result.drive.pump_frequency_hz,
            signal_frequency_hz=signal_frequency_hz,
            idler_frequency_hz=idler_frequency_hz,
            pump_label=args.pump_label,
            signal_labels=signal_labels,
            idler_labels=idler_labels,
            n_pump_harmonics=args.n_pump_harmonics,
            include_negative=args.include_negative_frequencies,
            include_dc=args.include_dc,
        )

    def sweep_config_factory(target_plan: Any) -> GainSweepConfig:
        output_node = None if args.output_node < 0 else args.output_node
        output_impedance = 1.0 / args.load_conductance_S if args.load_conductance_S > 0 else 50.0

        points = tuple(
            GainSolveConfig(
                signal_label=sig,
                idler_label=idl,
                input_node=args.input_node,
                output_node=output_node,
                signal_current_rms_A=1e-12 + 0j,
                set_conjugate=True,
                input_impedance_ohm=args.source_impedance_ohm,
                output_impedance_ohm=output_impedance,
            )
            for sig, idl in zip(signal_labels, idler_labels)
        )
        return GainSweepConfig(
            points=points,
            require_all_converged=True,
            name="calibration_gain_sweep",
        )

    return target_plan_factory, sweep_config_factory


# ---------------------------------------------------------------------------
# Parameter specification
# ---------------------------------------------------------------------------

def add_param(
    params: list[CalibrationParameterSpec],
    name: str,
    initial: float,
    bounds: tuple[float, float] | list[float],
    transform: ParameterTransform,
    enabled: bool,
    description: str,
) -> None:
    if not enabled:
        return
    lower, upper = float(bounds[0]), float(bounds[1])
    init = float(np.clip(initial, lower, upper))
    params.append(
        CalibrationParameterSpec(
            name=name,
            initial=init,
            lower=lower,
            upper=upper,
            transform=transform,
            enabled=True,
            description=description,
        )
    )


def build_parameter_spec(args: argparse.Namespace) -> CalibrationVectorSpec:
    params: list[CalibrationParameterSpec] = []

    if args.mode in {"linear", "combined"}:
        add_param(params, "L_scale", 1.0, args.L_scale_bounds, ParameterTransform.LOG, args.fit_L_scale, "Global L scale.")
        add_param(params, "C_scale", 1.0, args.C_scale_bounds, ParameterTransform.LOG, args.fit_C_scale, "Global C scale.")
        add_param(
            params,
            "C_stub_scale",
            1.0,
            args.C_stub_scale_bounds,
            ParameterTransform.LINEAR,
            args.fit_C_stub_scale,
            "Global stub capacitance scale.",
        )
        add_param(
            params,
            "R_scale",
            1.0,
            args.R_scale_bounds,
            ParameterTransform.LINEAR,
            args.fit_R_scale,
            "Global series loss scale.",
        )
        add_param(
            params,
            "G_scale",
            1.0,
            args.G_scale_bounds,
            ParameterTransform.LINEAR,
            args.fit_G_scale,
            "Global shunt loss scale.",
        )

    if args.mode in {"gain", "combined"}:
        add_param(
            params,
            "I_star_scale",
            1.0,
            args.I_star_scale_bounds,
            ParameterTransform.LOG,
            args.fit_I_star_scale,
            "Global I* scale.",
        )
        add_param(
            params,
            "beta_nl_scale",
            1.0,
            args.beta_nl_scale_bounds,
            ParameterTransform.LOG,
            args.fit_beta_nl_scale,
            "Global nonlinear beta scale.",
        )
        add_param(
            params,
            "pump_current_scale",
            1.0,
            args.pump_current_scale_bounds,
            ParameterTransform.LOG,
            args.fit_pump_current_scale,
            "Effective pump current scale.",
        )
        add_param(
            params,
            "pump_power_offset_db",
            0.0,
            args.pump_power_offset_db_bounds,
            ParameterTransform.LINEAR,
            args.fit_pump_power_offset_db,
            "Effective pump power offset in dB.",
        )

    if not params:
        raise ValueError(
            "No enabled calibration parameters. Enable at least one --fit-* option."
        )

    return CalibrationVectorSpec(tuple(params))


# ---------------------------------------------------------------------------
# Target / optimizer
# ---------------------------------------------------------------------------

def build_target(
    layout: LineLayout,
    args: argparse.Namespace,
    gain_extra: dict[str, Any],
) -> CalibrationTarget:
    nonlinear = build_nonlinear_params(args)
    drive = build_pump_drive(args)

    if args.mode in {"gain", "combined"}:
        if nonlinear is None:
            raise ValueError("Gain/combined calibration requires --I-star-A")
        if drive is None:
            raise ValueError("Gain/combined calibration requires pump drive")

    target_plan_factory = None
    sweep_config_factory = None
    if args.mode in {"gain", "combined"}:
        target_plan_factory, sweep_config_factory = build_gain_factories(args, gain_extra)

    return CalibrationTarget(
        base_layout=layout,
        base_nonlinear_params=nonlinear,
        cell_model=build_cell_model(args),
        cascade=build_cascade_config(args),
        pump_drive=drive,
        pump_config=build_pump_config(args) if args.mode in {"gain", "combined"} else None,
        target_plan_factory=target_plan_factory,
        sweep_config_factory=sweep_config_factory,
        metadata={
            "source": "scripts/run_calibration.py",
            "mode": args.mode,
        },
    )


def build_optimizer_config(args: argparse.Namespace) -> CalibrationOptimizerConfig:
    return CalibrationOptimizerConfig(
        method=CalibrationOptimizerMethod(args.optimizer),
        max_evaluations=args.max_evaluations,
        xtol=args.xtol,
        ftol=args.ftol,
        gtol=args.gtol,
        verbose=args.verbose,
        coordinate_initial_step_fraction=args.coordinate_initial_step_fraction,
        coordinate_step_decay=args.coordinate_step_decay,
        coordinate_min_step_fraction=args.coordinate_min_step_fraction,
        random_seed=args.random_seed,
    )


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", bool(args.jax_enable_x64))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fine_layout = build_layout_from_args(args)
    cal_layout, coarsening_report = maybe_coarsen_layout(fine_layout, args)

    sparam_data = load_sparameter_data(args)
    gain_data, gain_extra = load_gain_data(args)

    if args.mode == "linear" and sparam_data is None:
        raise ValueError("--mode linear requires --sparam-npz")
    if args.mode == "gain" and gain_data is None:
        raise ValueError("--mode gain requires --gain-npz")
    if args.mode == "combined" and sparam_data is None and gain_data is None:
        raise ValueError("--mode combined requires --sparam-npz and/or --gain-npz")

    target = build_target(cal_layout, args, gain_extra)
    parameter_spec = build_parameter_spec(args)
    optimizer = build_optimizer_config(args)

    print("[calibration] starting")
    print(f"[calibration] mode: {args.mode}")
    print(f"[calibration] fine layout: {fine_layout.name} ({fine_layout.n_cells} cells)")
    print(f"[calibration] calibration layout: {cal_layout.name} ({cal_layout.n_cells} cells)")
    print(f"[calibration] parameters: {', '.join(parameter_spec.enabled_names)}")
    print(f"[calibration] optimizer: {optimizer.selected_method().value}")
    print(f"[calibration] JAX backend: {jax.default_backend()}")

    result = calibrate(
        target,
        parameter_spec,
        sparameter_data=sparam_data,
        gain_data=gain_data,
        optimizer_config=optimizer,
    )

    print(f"[calibration] success: {result.success}")
    print(f"[calibration] loss: {result.loss:.6e}")
    print(f"[calibration] residual norm: {result.residual_norm:.6e}")

    paths = export_calibration_artifacts(
        result,
        output_dir,
        prefix=f"calibration_{args.mode}",
    )

    summary = result.to_dict()
    summary["fine_layout"] = fine_layout.summary()
    summary["calibration_layout"] = cal_layout.summary()
    summary["coarsening"] = coarsening_report
    summary["target"] = target.to_dict()
    summary["sparameter_data"] = None if sparam_data is None else sparam_data.to_dict()
    summary["gain_data"] = None if gain_data is None else gain_data.to_dict()
    summary["runtime"] = {
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }
    summary["cli_args"] = vars(args)
    summary["artifact_paths"] = paths

    if args.finite_difference_jacobian:
        print("[calibration] computing finite-difference residual Jacobian diagnostic")
        summary["finite_difference_jacobian"] = finite_difference_residual_jacobian(
            target,
            parameter_spec,
            result.best_encoded_vector,
            sparameter_data=sparam_data,
            gain_data=gain_data,
        )

    write_json(output_dir / f"calibration_{args.mode}_cli_summary.json", summary)

    md_path = output_dir / f"calibration_{args.mode}_cli_summary.md"
    md_path.write_text(calibration_summary_markdown(result), encoding="utf-8")
    summary["artifact_paths"]["cli_summary_md"] = str(md_path)

    print("[calibration] best parameters:")
    for name, value in result.best_parameters.items():
        print(f"  - {name}: {value:.12g}")

    print("[calibration] artifacts:")
    for key, path in summary["artifact_paths"].items():
        print(f"  - {key}: {path}")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run(args)
    except Exception as exc:
        print(f"[calibration] ERROR: {exc}", file=sys.stderr)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "calibration_error.json",
            {
                "success": False,
                "error": str(exc),
                "cli_args": vars(args),
            },
        )
        return 2

    if args.fail_on_validation_error and not summary.get("success", False):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
