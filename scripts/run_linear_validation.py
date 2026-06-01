#!/usr/bin/env python3
"""
Run pump-off linear validation for a TWPA layout.

This script is the command-line entry point for the linear foundation of the
simulator:

    layout
      -> cell sanity checks
      -> ABCD cascade validation
      -> S-parameters
      -> dispersion extraction
      -> stopband detection
      -> optional NPZ/JSON/Markdown artifacts

Examples
--------
Fast synthetic uniform validation:

    python scripts/run_linear_validation.py \
      --layout-kind synthetic-uniform \
      --n-cells 128 \
      --length-mm 2.0 \
      --f-min-ghz 1 \
      --f-max-ghz 12 \
      --n-frequency-points 301 \
      --output-dir runs/linear_uniform

Industrial 100 mm / 20,000-cell linear validation:

    python scripts/run_linear_validation.py \
      --layout-kind industrial \
      --n-cells 20000 \
      --length-mm 100 \
      --f-min-ghz 1 \
      --f-max-ghz 16 \
      --n-frequency-points 401 \
      --cascade-strategy auto \
      --chunk-size 512 \
      --output-dir runs/linear_100mm

Stub-loaded periodic layout:

    python scripts/run_linear_validation.py \
      --layout-kind synthetic-stub \
      --n-cells 2000 \
      --length-mm 10 \
      --stub-period-cells 40 \
      --stub-fraction 0.25 \
      --cells-per-supercell 40 \
      --output-dir runs/linear_stub
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import numpy as np

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout
from twpa.linear.cells import (
    CellModelConfig,
    CellModelKind,
    layout_cell_parameter_summary,
    validate_layout_cells,
)
from twpa.linear.cascade import (
    CascadeConfig,
    CascadeStrategy,
    run_linear_scan,
    validate_cascade,
)
from twpa.linear.dispersion import (
    DispersionConfig,
    DispersionExtractionMethod,
    StopbandMetric,
    detect_stopbands,
    extract_layout_dispersion,
    validate_dispersion_result,
)
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
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
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
# CLI construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pump-off linear validation for a TWPA layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    parser.add_argument("--n-cells", type=int, default=256, help="Number of cells.")
    parser.add_argument("--length-mm", type=float, default=2.0, help="Line length in mm.")
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

    parser.add_argument("--f-min-ghz", type=float, default=1.0, help="Minimum frequency in GHz.")
    parser.add_argument("--f-max-ghz", type=float, default=12.0, help="Maximum frequency in GHz.")
    parser.add_argument(
        "--n-frequency-points",
        type=int,
        default=301,
        help="Number of frequency samples.",
    )

    parser.add_argument(
        "--cell-model",
        choices=[kind.value for kind in CellModelKind],
        default=CellModelKind.PI.value,
        help="Cell ABCD model.",
    )
    parser.add_argument(
        "--cascade-strategy",
        choices=[strategy.value for strategy in CascadeStrategy],
        default=CascadeStrategy.AUTO.value,
        help="ABCD cascade strategy.",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for chunked cascade.")
    parser.add_argument(
        "--cells-per-supercell",
        type=int,
        default=1,
        help="Supercell size for periodic/Bloch analysis.",
    )
    parser.add_argument(
        "--include-stub-capacitance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include C_stub in the linear cell model.",
    )
    parser.add_argument(
        "--include-resonator-loading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include resonator loading if present in layout.",
    )

    parser.add_argument(
        "--dispersion-method",
        choices=[m.value for m in DispersionExtractionMethod],
        default=DispersionExtractionMethod.BOTH.value,
        help="Dispersion extraction method.",
    )
    parser.add_argument(
        "--stopband-metric",
        choices=[m.value for m in StopbandMetric],
        default=StopbandMetric.BOTH.value,
        help="Stopband detection metric.",
    )
    parser.add_argument(
        "--stopband-s21-threshold-db",
        type=float,
        default=-10.0,
        help="S21 threshold for stopband detection.",
    )
    parser.add_argument(
        "--stopband-alpha-threshold",
        type=float,
        default=1.0,
        help="Bloch alpha threshold in Np/m for stopband detection.",
    )
    parser.add_argument(
        "--expect-stopband",
        choices=["yes", "no", "unknown"],
        default="unknown",
        help="Whether validation should require a stopband.",
    )

    parser.add_argument(
        "--cutoff-safety-factor",
        type=float,
        default=2.0,
        help="Cell cutoff guard safety factor.",
    )
    parser.add_argument(
        "--det-tolerance",
        type=float,
        default=1e-6,
        help="ABCD determinant validation tolerance.",
    )
    parser.add_argument(
        "--passivity-tolerance",
        type=float,
        default=1e-7,
        help="S-parameter passivity validation tolerance.",
    )

    parser.add_argument(
        "--output-dir",
        default="runs/linear_validation",
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
        help="Return nonzero exit code if validation fails.",
    )
    parser.add_argument(
        "--jax-enable-x64",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable JAX x64.",
    )

    return parser


# ---------------------------------------------------------------------------
# Layout construction
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

    kind = kind_map[args.layout_kind]
    spec = SyntheticLayoutSpec(
        kind=kind,
        n_cells=args.n_cells,
        length_m=length_m,
        z0_ohm=args.z0_ohm,
        phase_velocity_m_per_s=args.phase_velocity,
        L_per_m_H=args.L_per_m_H,
        C_per_m_F=args.C_per_m_F,
        R_per_m_ohm=args.R_per_m_ohm,
        G_per_m_S=args.G_per_m_S,
        stub_period_cells=max(1, args.stub_period_cells or args.cells_per_supercell or 1),
        stub_fraction=args.stub_fraction,
        disorder_std_fraction=args.disorder_std_fraction,
        disorder_seed=args.disorder_seed,
        name=name or f"{args.layout_kind}_{args.length_mm:g}mm_{args.n_cells}cell",
    )
    return build_synthetic_layout(spec)


def expected_stopband_from_arg(value: str) -> bool | None:
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


# ---------------------------------------------------------------------------
# Validation workflow
# ---------------------------------------------------------------------------

def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", bool(args.jax_enable_x64))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = build_layout_from_args(args)
    frequency_hz = jnp.linspace(
        args.f_min_ghz * 1e9,
        args.f_max_ghz * 1e9,
        args.n_frequency_points,
        dtype=jnp.float64,
    )

    cell_model = CellModelConfig(
        kind=CellModelKind(args.cell_model),
        include_stub_capacitance=args.include_stub_capacitance,
        include_resonator_loading=args.include_resonator_loading,
    )

    cascade_config = CascadeConfig(
        strategy=CascadeStrategy(args.cascade_strategy),
        chunk_size=args.chunk_size,
        cells_per_supercell=args.cells_per_supercell,
        allow_remainder=True,
    )

    dispersion_config = DispersionConfig(
        method=DispersionExtractionMethod(args.dispersion_method),
        cells_per_supercell=args.cells_per_supercell,
        stopband_s21_threshold_db=args.stopband_s21_threshold_db,
        stopband_alpha_threshold_np_per_m=args.stopband_alpha_threshold,
    )

    print(f"[linear-validation] layout: {layout.name}")
    print(f"[linear-validation] cells: {layout.n_cells}")
    print(f"[linear-validation] length: {layout.total_length_m:.6g} m")
    print(
        "[linear-validation] frequency: "
        f"{args.f_min_ghz:.6g}â€“{args.f_max_ghz:.6g} GHz "
        f"({args.n_frequency_points} points)"
    )
    print(f"[linear-validation] JAX backend: {jax.default_backend()}")

    cell_report = validate_layout_cells(
        frequency_hz,
        layout,
        config=cell_model,
        cutoff_safety_factor=args.cutoff_safety_factor,
    )

    print(f"[linear-validation] cell cutoff guard: {cell_report.cutoff_guard_passed}")

    cascade_report = validate_cascade(
        frequency_hz,
        layout,
        cell_model=cell_model,
        cascade_config=cascade_config,
        det_tolerance=args.det_tolerance,
        passivity_tolerance=args.passivity_tolerance,
    )

    print(f"[linear-validation] cascade validation: {cascade_report.passed}")

    scan = run_linear_scan(
        frequency_hz,
        layout,
        cell_model=cell_model,
        cascade_config=cascade_config,
    )

    dispersion = extract_layout_dispersion(
        frequency_hz,
        layout,
        cell_model=cell_model,
        cascade_config=cascade_config,
        dispersion_config=dispersion_config,
    )

    dispersion_report = validate_dispersion_result(
        dispersion,
        layout_name=layout.name,
        expected_stopband=expected_stopband_from_arg(args.expect_stopband),
        stopband_metric=StopbandMetric(args.stopband_metric),
        s21_threshold_db=args.stopband_s21_threshold_db,
        alpha_threshold_np_per_m=args.stopband_alpha_threshold,
    )

    stopbands = detect_stopbands(
        dispersion,
        metric=StopbandMetric(args.stopband_metric),
        s21_threshold_db=args.stopband_s21_threshold_db,
        alpha_threshold_np_per_m=args.stopband_alpha_threshold,
    )

    print(f"[linear-validation] dispersion validation: {dispersion_report.passed}")
    print(f"[linear-validation] stopbands detected: {len(stopbands)}")
    print(
        "[linear-validation] S21 dB min/max: "
        f"{float(jnp.min(scan.s21_db)):.4g} / {float(jnp.max(scan.s21_db)):.4g}"
    )

    passed = bool(
        cell_report.cutoff_guard_passed
        and cascade_report.passed
        and dispersion_report.passed
    )

    summary = {
        "passed": passed,
        "layout": layout.summary(),
        "layout_cell_parameter_summary": layout_cell_parameter_summary(layout),
        "frequency": {
            "min_hz": float(frequency_hz[0]),
            "max_hz": float(frequency_hz[-1]),
            "n_points": int(frequency_hz.shape[0]),
        },
        "configs": {
            "cell_model": cell_model.to_dict(),
            "cascade": cascade_config.to_dict(),
            "dispersion": dispersion_config.to_dict(),
            "cli_args": vars(args),
        },
        "reports": {
            "cell": cell_report.to_dict(),
            "cascade": cascade_report.to_dict(),
            "linear_scan": scan.to_dict(),
            "dispersion": dispersion.to_dict(),
            "dispersion_validation": dispersion_report.to_dict(),
            "stopbands": [sb.to_dict() for sb in stopbands],
        },
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
    }

    summary_path = write_json(output_dir / "linear_validation_summary.json", summary)
    markdown_path = output_dir / "linear_validation_summary.md"
    markdown_path.write_text(make_markdown_summary(summary), encoding="utf-8")

    artifact_paths = {
        "summary_json": str(summary_path),
        "summary_md": str(markdown_path),
    }

    if args.save_npz:
        npz_path = write_npz(
            output_dir / "linear_validation_arrays.npz",
            frequency_hz=frequency_hz,
            s=scan.s,
            s21=scan.s21,
            s21_db=scan.s21_db,
            abcd=scan.abcd,
            beta_eff_rad_per_m=scan.beta_eff_rad_per_m,
            group_delay_s=scan.group_delay_s,
            beta_preferred_rad_per_m=dispersion.beta_preferred_rad_per_m,
            alpha_preferred_np_per_m=dispersion.alpha_preferred_np_per_m,
            beta_s21_rad_per_m=(
                jnp.asarray([])
                if dispersion.beta_s21_rad_per_m is None
                else dispersion.beta_s21_rad_per_m
            ),
            beta_bloch_rad_per_m=(
                jnp.asarray([])
                if dispersion.beta_bloch_rad_per_m is None
                else dispersion.beta_bloch_rad_per_m
            ),
            alpha_bloch_np_per_m=(
                jnp.asarray([])
                if dispersion.alpha_bloch_np_per_m is None
                else dispersion.alpha_bloch_np_per_m
            ),
        )
        artifact_paths["arrays_npz"] = str(npz_path)

    summary["artifact_paths"] = artifact_paths
    write_json(output_dir / "linear_validation_summary.json", summary)

    print("[linear-validation] artifacts:")
    for key, path in artifact_paths.items():
        print(f"  - {key}: {path}")

    print(f"[linear-validation] final status: {'PASS' if passed else 'FAIL'}")

    return summary


def make_markdown_summary(summary: dict[str, Any]) -> str:
    reports = summary["reports"]
    layout = summary["layout"]
    freq = summary["frequency"]

    scan = reports["linear_scan"]
    cell = reports["cell"]
    cascade = reports["cascade"]
    dispersion_validation = reports["dispersion_validation"]
    stopbands = reports["stopbands"]

    lines = [
        "# Linear validation summary",
        "",
        f"- status: `{'PASS' if summary['passed'] else 'FAIL'}`",
        f"- layout: `{layout['name']}`",
        f"- cells: `{layout['n_cells']}`",
        f"- length: `{layout['total_length_m']:.6g} m`",
        f"- frequency range: `{freq['min_hz'] / 1e9:.6g}`â€“`{freq['max_hz'] / 1e9:.6g} GHz`",
        f"- frequency points: `{freq['n_points']}`",
        "",
        "## Main checks",
        "",
        "| check | status |",
        "|---|---:|",
        f"| cell cutoff guard | `{cell.get('cutoff_guard_passed')}` |",
        f"| cascade validation | `{cascade.get('passed')}` |",
        f"| dispersion validation | `{dispersion_validation.get('passed')}` |",
        "",
        "## Linear response",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| S21 min dB | `{scan['s21_db_min']:.6g}` |",
        f"| S21 max dB | `{scan['s21_db_max']:.6g}` |",
        f"| beta min rad/m | `{scan['beta_eff_min_rad_per_m']:.6g}` |",
        f"| beta max rad/m | `{scan['beta_eff_max_rad_per_m']:.6g}` |",
        f"| group delay min s | `{scan['group_delay_min_s']:.6e}` |",
        f"| group delay max s | `{scan['group_delay_max_s']:.6e}` |",
        "",
        "## Stopbands",
        "",
    ]

    if not stopbands:
        lines.append("No stopband intervals detected.")
    else:
        lines += [
            "| idx | start GHz | stop GHz | center GHz | width GHz | min S21 dB | max alpha Np/m |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for idx, sb in enumerate(stopbands):
            min_s21 = "" if sb.get("min_s21_db") is None else f"{sb['min_s21_db']:.6g}"
            max_alpha = "" if sb.get("max_alpha_np_per_m") is None else f"{sb['max_alpha_np_per_m']:.6g}"
            lines.append(
                f"| {idx} | {sb['start_GHz']:.6g} | {sb['stop_GHz']:.6g} | "
                f"{sb['center_GHz']:.6g} | {sb['width_GHz']:.6g} | "
                f"{min_s21} | {max_alpha} |"
            )

    lines += [
        "",
        "## Messages",
        "",
    ]

    for msg in cell.get("messages", []):
        lines.append(f"- cell: {msg}")
    for msg in cascade.get("messages", []):
        lines.append(f"- cascade: {msg}")
    for msg in dispersion_validation.get("messages", []):
        lines.append(f"- dispersion: {msg}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_validation(args)
    except Exception as exc:
        print(f"[linear-validation] ERROR: {exc}", file=sys.stderr)
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                output_dir / "linear_validation_error.json",
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
