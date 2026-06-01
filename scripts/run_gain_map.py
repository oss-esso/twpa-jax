#!/usr/bin/env python3
"""
Run pump-HB + small-signal gain map for a TWPA ladder.

This script is the command-line entry point for the reference nonlinear gain
pipeline:

    layout
      -> optional coarsening
      -> pump-only HB solve
      -> for each signal frequency:
           target pump/signal/idler frequency plan
           linearization around pumped state
           small-signal solve
           signal gain + idler conversion
      -> JSON / NPZ / Markdown artifacts

Important
---------
The current backend is dense/reference. It is intended for reduced layouts and
validation/convergence studies, not direct 20,000-cell nonlinear gain maps.

Recommended industrial flow:

    python scripts/run_linear_validation.py --layout-kind industrial --n-cells 20000 ...

    python scripts/run_gain_map.py ^
      --layout-kind industrial ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --coarsen-target-cells 100 ^
      --pump-frequency-ghz 8 ^
      --pump-power-dbm -80 ^
      --signal-min-ghz 4 ^
      --signal-max-ghz 7 ^
      --n-signal-points 31 ^
      --I-star-A 1e-3 ^
      --output-dir runs/gain_neff100

Then repeat for N_eff = 200, 500, 1000 until the gain map converges.

Frequency-plan compatibility
----------------------------
This script tries to use project-native frequency-plan constructors if present.
If the current twpa.core.frequency_plan module does not yet expose a dedicated
signal/idler plan builder, it falls back to trying common FrequencyPlan
constructor signatures.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import numpy as np

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout
from twpa.core.params import NonlinearParams
from twpa.core.hb_fft import HBProjectionConfig
from twpa.core import frequency_plan as frequency_plan_module
from twpa.linear.coarsening import (
    CoarseningConfig,
    CoarseningMethod,
    coarsen_layout,
    make_uniform_surrogate_layout,
)
from twpa.nonlinear.distributed_hb import (
    DistributedHBConfig,
    DistributedHBTerminationKind,
)
from twpa.nonlinear.gain import (
    GainPointResult,
    GainSolveConfig,
    GainSweepConfig,
    solve_gain_sweep_from_pump,
)
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    PumpHBLadderResult,
    pump_solution_table,
    solve_pump_hb_ladder,
)
from twpa.solvers.hb_solver import DenseNewtonConfig, LinearSolveMethod
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
        description="Run pump-HB + small-signal gain map for a TWPA ladder.",
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
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument("--length-mm", type=float, default=1.0)
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

    # Coarsening
    parser.add_argument(
        "--coarsen-target-cells",
        type=int,
        default=None,
        help="Reduce fine layout before nonlinear pump/gain solve.",
    )
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

    # Nonlinear parameters
    parser.add_argument("--I-star-A", type=float, required=True)
    parser.add_argument("--beta-nl", type=float, default=1.0)
    parser.add_argument("--quartic-coefficient", type=float, default=0.0)
    parser.add_argument("--dc-bias-A", type=float, default=0.0)

    # Pump drive
    parser.add_argument("--pump-frequency-ghz", type=float, required=True)
    drive_group = parser.add_mutually_exclusive_group(required=False)
    drive_group.add_argument("--pump-current-rms-A", type=float, default=None)
    drive_group.add_argument("--pump-power-dbm", type=float, default=None)
    drive_group.add_argument("--pump-power-W", type=float, default=None)
    parser.add_argument("--source-impedance-ohm", type=float, default=50.0)
    parser.add_argument("--pump-phase-rad", type=float, default=0.0)
    parser.add_argument("--pump-label", default="pump")

    # Boundary model
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

    # Pump HB plan/projection
    parser.add_argument("--n-pump-harmonics", type=int, default=3)
    parser.add_argument(
        "--include-negative-frequencies",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--include-dc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-samples", type=int, default=None)
    parser.add_argument("--oversampling", type=int, default=8)

    # Gain map signal grid
    signal_group = parser.add_mutually_exclusive_group(required=False)
    signal_group.add_argument(
        "--signal-ghz",
        type=float,
        nargs="+",
        default=None,
        help="Explicit signal frequencies in GHz.",
    )
    signal_group.add_argument(
        "--signal-detuning-ghz",
        type=float,
        nargs="+",
        default=None,
        help="Explicit |fp - fs| detunings in GHz.",
    )

    parser.add_argument("--signal-min-ghz", type=float, default=None)
    parser.add_argument("--signal-max-ghz", type=float, default=None)
    parser.add_argument("--n-signal-points", type=int, default=21)
    parser.add_argument(
        "--signal-side",
        choices=["lower", "upper"],
        default="lower",
        help="Used with detuning or auto range. lower means fs = fp - detuning.",
    )
    parser.add_argument("--signal-label", default="signal")
    parser.add_argument("--idler-label", default="idler")
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
    parser.add_argument("--stop-on-first-gain-failure", action="store_true")

    # Dense solver
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

    # Output/runtime
    parser.add_argument("--output-dir", default="runs/gain_map")
    parser.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-validation-error", action="store_true")
    parser.add_argument("--jax-enable-x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

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
        name="gain_map_distributed",
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
        name="gain_map_pump_hb",
    )


# ---------------------------------------------------------------------------
# Frequency-plan construction
# ---------------------------------------------------------------------------

def signal_frequencies_from_args(args: argparse.Namespace) -> jax.Array:
    fp = args.pump_frequency_ghz * 1e9

    if args.signal_ghz is not None:
        values = jnp.asarray([x * 1e9 for x in args.signal_ghz], dtype=jnp.float64)
        return values

    if args.signal_detuning_ghz is not None:
        detuning = jnp.asarray([x * 1e9 for x in args.signal_detuning_ghz], dtype=jnp.float64)
        if args.signal_side == "lower":
            return fp - detuning
        return fp + detuning

    if args.signal_min_ghz is not None or args.signal_max_ghz is not None:
        if args.signal_min_ghz is None or args.signal_max_ghz is None:
            raise ValueError("Provide both --signal-min-ghz and --signal-max-ghz")
        return jnp.linspace(
            args.signal_min_ghz * 1e9,
            args.signal_max_ghz * 1e9,
            args.n_signal_points,
            dtype=jnp.float64,
        )

    # Default: lower sideband from 0.5 to 3 GHz below pump.
    detuning = jnp.linspace(0.5e9, 3.0e9, args.n_signal_points, dtype=jnp.float64)
    return fp - detuning if args.signal_side == "lower" else fp + detuning


def idler_frequency_dp4wm(pump_frequency_hz: float, signal_frequency_hz: float) -> float:
    return 2.0 * float(pump_frequency_hz) - float(signal_frequency_hz)


def _try_call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """
    Call fn with the subset of kwargs accepted by its signature.

    If the signature is not inspectable, try all kwargs.
    """
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
    """
    Create a FrequencyPlan containing pump, signal, and idler tones.

    This compatibility helper first tries project-native constructors and then
    falls back to common FrequencyPlan constructor signatures.
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
            neg_label = f"-{pos_label}"
            add(neg_label, -h * pump_frequency_hz)

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

    generic_kwargs_candidates = [
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
        for candidate_kwargs in generic_kwargs_candidates:
            try:
                return _try_call_with_supported_kwargs(ctor, candidate_kwargs)
            except Exception as exc:
                errors.append(f"{getattr(ctor, '__name__', ctor)}: {exc}")

    raise RuntimeError(
        "Could not construct a pump/signal/idler FrequencyPlan. "
        "Add a constructor such as make_pump_signal_idler_plan(...) to "
        "twpa.core.frequency_plan, or adapt make_target_gain_plan(). "
        f"Tried constructors/errors: {errors}"
    )


# ---------------------------------------------------------------------------
# Gain map workflow
# ---------------------------------------------------------------------------

def solve_pump(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig,
) -> PumpHBLadderResult:
    print("[gain-map] solving pump HB")
    print(f"[gain-map] pump frequency: {drive.pump_frequency_hz / 1e9:.9g} GHz")
    print(f"[gain-map] pump current RMS: {drive.current_rms_A:.9e} A")
    print(f"[gain-map] available pump power: {drive.available_power_dbm:.9g} dBm")

    result = solve_pump_hb_ladder(
        layout,
        nonlinear_params,
        drive=drive,
        pump_config=pump_config,
        metadata={"cli": "scripts/run_gain_map.py", "stage": "pump"},
    )

    print(f"[gain-map] pump converged: {result.converged}")
    print(f"[gain-map] pump residual: {result.residual.norm:.6e}")
    print(f"[gain-map] max I/I*: {result.profile.max_pump_current_ratio:.6e}")

    return result


def solve_one_gain_point(
    pump_result: PumpHBLadderResult,
    *,
    signal_frequency_hz: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], GainPointResult | None]:
    pump_frequency_hz = pump_result.drive.pump_frequency_hz
    idler_frequency_hz = idler_frequency_dp4wm(pump_frequency_hz, signal_frequency_hz)

    if idler_frequency_hz <= 0.0 and args.skip_nonpositive_idlers:
        return (
            {
                "status": "skip",
                "reason": "nonpositive idler frequency",
                "signal_frequency_hz": float(signal_frequency_hz),
                "idler_frequency_hz": float(idler_frequency_hz),
                "signal_gain_db": np.nan,
                "matched_power_gain_db": np.nan,
                "idler_conversion_db": np.nan,
                "converged": False,
            },
            None,
        )

    target_plan = make_target_gain_plan(
        pump_frequency_hz=pump_frequency_hz,
        signal_frequency_hz=signal_frequency_hz,
        idler_frequency_hz=idler_frequency_hz,
        pump_label=args.pump_label,
        signal_label=args.signal_label,
        idler_label=args.idler_label,
        n_pump_harmonics=args.n_pump_harmonics,
        include_negative=args.include_negative_frequencies,
        include_dc=args.include_dc,
    )

    out_node = None if args.output_node < 0 else args.output_node

    point_cfg = GainSolveConfig(
        signal_label=args.signal_label,
        idler_label=args.idler_label,
        input_node=args.input_node,
        output_node=out_node,
        signal_current_rms_A=args.signal_current_rms_A,
        set_conjugate=args.set_signal_conjugate,
        input_impedance_ohm=args.source_impedance_ohm,
        output_impedance_ohm=1.0 / args.load_conductance_S if args.load_conductance_S > 0.0 else 50.0,
    )

    sweep_cfg = GainSweepConfig(
        points=(point_cfg,),
        require_all_converged=True,
        name=f"gain_point_{signal_frequency_hz / 1e9:.6g}GHz",
    )

    sweep = solve_gain_sweep_from_pump(
        pump_result,
        target_plan=target_plan,
        sweep_config=sweep_cfg,
    )

    point = sweep.points[0]

    row = {
        "status": point.status.value,
        "signal_frequency_hz": float(signal_frequency_hz),
        "signal_frequency_GHz": float(signal_frequency_hz / 1e9),
        "idler_frequency_hz": float(idler_frequency_hz),
        "idler_frequency_GHz": float(idler_frequency_hz / 1e9),
        "converged": bool(point.converged),
        "signal_gain_db": float(point.signal_gain_db),
        "matched_power_gain_db": float(point.matched_power_gain_db),
        "idler_conversion_db": (
            None if point.idler_conversion_db is None else float(point.idler_conversion_db)
        ),
        "signal_gain_complex": point.signal_gain_complex,
        "idler_conversion_complex": point.idler_conversion_complex,
        "linear_solve": point.solve.linear_solve.to_dict(),
        "target_plan": target_plan.to_dict(),
        "gain_point": point.to_dict(),
    }

    return row, point


def run_gain_map(
    *,
    fine_layout: LineLayout,
    hb_layout: LineLayout,
    coarsening_report: dict[str, Any],
    nonlinear_params: NonlinearParams,
    drive: PumpDriveConfig,
    pump_config: PumpHBLadderConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pump_result = solve_pump(
        hb_layout,
        nonlinear_params,
        drive,
        pump_config,
    )

    signal_freqs = signal_frequencies_from_args(args)

    rows: list[dict[str, Any]] = []
    full_points: list[GainPointResult] = []

    print(f"[gain-map] signal points: {len(signal_freqs)}")

    for idx, fs in enumerate(signal_freqs.tolist()):
        print(f"[gain-map] point {idx + 1}/{len(signal_freqs)}: fs={fs / 1e9:.9g} GHz")
        try:
            row, point = solve_one_gain_point(
                pump_result,
                signal_frequency_hz=float(fs),
                args=args,
            )
            rows.append(row)
            if point is not None:
                full_points.append(point)

            status = row.get("status")
            gain = row.get("signal_gain_db")
            print(f"[gain-map]   status={status}, gain={gain}")

            if args.stop_on_first_gain_failure and not row.get("converged", False):
                print("[gain-map] stopping after first gain failure")
                break

        except Exception as exc:
            row = {
                "status": "error",
                "signal_frequency_hz": float(fs),
                "signal_frequency_GHz": float(fs / 1e9),
                "idler_frequency_hz": float(idler_frequency_dp4wm(drive.pump_frequency_hz, fs)),
                "idler_frequency_GHz": float(idler_frequency_dp4wm(drive.pump_frequency_hz, fs) / 1e9),
                "converged": False,
                "signal_gain_db": np.nan,
                "matched_power_gain_db": np.nan,
                "idler_conversion_db": np.nan,
                "error": str(exc),
            }
            rows.append(row)
            print(f"[gain-map]   ERROR: {exc}")

            if args.stop_on_first_gain_failure:
                break

    signal_gain_db = np.asarray(
        [np.nan if row.get("signal_gain_db") is None else row.get("signal_gain_db") for row in rows],
        dtype=float,
    )
    matched_power_gain_db = np.asarray(
        [np.nan if row.get("matched_power_gain_db") is None else row.get("matched_power_gain_db") for row in rows],
        dtype=float,
    )
    idler_conversion_db = np.asarray(
        [
            np.nan if row.get("idler_conversion_db") is None else row.get("idler_conversion_db")
            for row in rows
        ],
        dtype=float,
    )
    converged = np.asarray([bool(row.get("converged", False)) for row in rows], dtype=bool)

    passed = bool(pump_result.converged and np.all(converged)) if rows else False

    summary = {
        "mode": "gain_map",
        "passed": passed,
        "pump_passed": bool(pump_result.converged),
        "gain_all_converged": bool(np.all(converged)) if rows else False,
        "n_gain_points": len(rows),
        "n_gain_converged": int(np.sum(converged)),
        "fine_layout": fine_layout.summary(),
        "hb_layout": hb_layout.summary(),
        "coarsening": coarsening_report,
        "nonlinear_params": nonlinear_params.to_dict(),
        "drive": drive.to_dict(),
        "pump_config": pump_config.to_dict(),
        "pump_result": pump_result.to_dict(),
        "gain_rows": rows,
        "gain_summary": {
            "signal_gain_db_min": float(np.nanmin(signal_gain_db)) if signal_gain_db.size else None,
            "signal_gain_db_max": float(np.nanmax(signal_gain_db)) if signal_gain_db.size else None,
            "matched_power_gain_db_min": (
                float(np.nanmin(matched_power_gain_db)) if matched_power_gain_db.size else None
            ),
            "matched_power_gain_db_max": (
                float(np.nanmax(matched_power_gain_db)) if matched_power_gain_db.size else None
            ),
            "idler_conversion_db_min": (
                float(np.nanmin(idler_conversion_db))
                if idler_conversion_db.size and np.any(np.isfinite(idler_conversion_db))
                else None
            ),
            "idler_conversion_db_max": (
                float(np.nanmax(idler_conversion_db))
                if idler_conversion_db.size and np.any(np.isfinite(idler_conversion_db))
                else None
            ),
        },
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "cli_args": vars(args),
    }

    return summary


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def export_artifacts(output_dir: Path, summary: dict[str, Any], *, save_npz: bool) -> dict[str, str]:
    paths: dict[str, str] = {}

    paths["summary_json"] = str(write_json(output_dir / "gain_map_summary.json", summary))

    md_path = output_dir / "gain_map_summary.md"
    md_path.write_text(make_markdown_summary(summary), encoding="utf-8")
    paths["summary_md"] = str(md_path)

    csv_path = output_dir / "gain_map_points.csv"
    write_gain_rows_csv(csv_path, summary["gain_rows"])
    paths["points_csv"] = str(csv_path)

    if save_npz:
        rows = summary["gain_rows"]
        paths["arrays_npz"] = str(
            write_npz(
                output_dir / "gain_map_arrays.npz",
                signal_frequency_hz=np.asarray([r["signal_frequency_hz"] for r in rows], dtype=float),
                idler_frequency_hz=np.asarray([r["idler_frequency_hz"] for r in rows], dtype=float),
                converged=np.asarray([r.get("converged", False) for r in rows], dtype=bool),
                signal_gain_db=np.asarray(
                    [np.nan if r.get("signal_gain_db") is None else r.get("signal_gain_db") for r in rows],
                    dtype=float,
                ),
                matched_power_gain_db=np.asarray(
                    [
                        np.nan if r.get("matched_power_gain_db") is None else r.get("matched_power_gain_db")
                        for r in rows
                    ],
                    dtype=float,
                ),
                idler_conversion_db=np.asarray(
                    [
                        np.nan if r.get("idler_conversion_db") is None else r.get("idler_conversion_db")
                        for r in rows
                    ],
                    dtype=float,
                ),
            )
        )

    summary["artifact_paths"] = paths
    write_json(output_dir / "gain_map_summary.json", summary)
    return paths


def write_gain_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "status",
        "converged",
        "signal_frequency_hz",
        "signal_frequency_GHz",
        "idler_frequency_hz",
        "idler_frequency_GHz",
        "signal_gain_db",
        "matched_power_gain_db",
        "idler_conversion_db",
        "error",
    ]

    lines = [",".join(columns)]

    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if value is None:
                value = ""
            if isinstance(value, str):
                value = '"' + value.replace('"', '""') + '"'
            values.append(str(value))
        lines.append(",".join(values))

    path.write_text("\n".join(lines), encoding="utf-8")


def make_markdown_summary(summary: dict[str, Any]) -> str:
    pump = summary["pump_result"]
    drive = summary["drive"]
    gain_summary = summary["gain_summary"]

    lines = [
        "# Gain map summary",
        "",
        f"- status: `{'PASS' if summary['passed'] else 'FAIL'}`",
        f"- pump status: `{'PASS' if summary['pump_passed'] else 'FAIL'}`",
        f"- gain points: `{summary['n_gain_converged']}/{summary['n_gain_points']}` converged",
        f"- layout: `{summary['hb_layout']['name']}`",
        f"- HB cells: `{summary['hb_layout']['n_cells']}`",
        f"- fine cells: `{summary['fine_layout']['n_cells']}`",
        f"- pump frequency: `{drive['pump_frequency_GHz']:.9g} GHz`",
        f"- pump power: `{drive['available_power_dbm']:.9g} dBm`",
        f"- pump Norton current RMS: `{drive['current_rms_A']:.9e} A`",
        "",
        "## Pump solution",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| converged | `{pump['converged']}` |",
        f"| pump residual norm | `{pump['residual']['combined_norm']:.6e}` |",
        f"| max I/Istar | `{pump['profile']['max_pump_current_ratio']:.6e}` |",
        f"| pump output/input voltage gain dB | `{pump['profile']['output_to_input_voltage_gain_db']:.6g}` |",
        "",
        "## Gain summary",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| signal gain min dB | `{gain_summary['signal_gain_db_min']}` |",
        f"| signal gain max dB | `{gain_summary['signal_gain_db_max']}` |",
        f"| matched power gain min dB | `{gain_summary['matched_power_gain_db_min']}` |",
        f"| matched power gain max dB | `{gain_summary['matched_power_gain_db_max']}` |",
        f"| idler conversion min dB | `{gain_summary['idler_conversion_db_min']}` |",
        f"| idler conversion max dB | `{gain_summary['idler_conversion_db_max']}` |",
        "",
        "## Gain points",
        "",
        "| idx | status | fs GHz | fi GHz | gain dB | matched power dB | idler conv dB |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for idx, row in enumerate(summary["gain_rows"]):
        gain = row.get("signal_gain_db")
        pgain = row.get("matched_power_gain_db")
        idler = row.get("idler_conversion_db")
        lines.append(
            f"| {idx} | `{row.get('status')}` | "
            f"{row.get('signal_frequency_GHz', np.nan):.6g} | "
            f"{row.get('idler_frequency_GHz', np.nan):.6g} | "
            f"{'' if gain is None else gain} | "
            f"{'' if pgain is None else pgain} | "
            f"{'' if idler is None else idler} |"
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

    print(f"[gain-map] JAX backend: {jax.default_backend()}")
    print(f"[gain-map] fine layout: {fine_layout.name}")
    print(f"[gain-map] fine cells: {fine_layout.n_cells}")
    print(f"[gain-map] HB layout: {hb_layout.name}")
    print(f"[gain-map] HB cells: {hb_layout.n_cells}")
    print(f"[gain-map] HB length: {hb_layout.total_length_m:.6g} m")

    summary = run_gain_map(
        fine_layout=fine_layout,
        hb_layout=hb_layout,
        coarsening_report=coarsening_report,
        nonlinear_params=nonlinear_params,
        drive=drive,
        pump_config=pump_config,
        args=args,
    )

    paths = export_artifacts(output_dir, summary, save_npz=args.save_npz)

    print("[gain-map] artifacts:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")

    print(f"[gain-map] final status: {'PASS' if summary['passed'] else 'FAIL'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run(args)
    except Exception as exc:
        print(f"[gain-map] ERROR: {exc}", file=sys.stderr)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "gain_map_error.json",
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
