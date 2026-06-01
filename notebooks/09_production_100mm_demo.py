# %% [markdown]
# # 09 — Production 100 mm / 20,000-cell KI-TWPA demo
#
# This notebook-style script demonstrates the production simulation workflow:
#
# 1. Build a synthetic 100 mm / 20,000-cell line.
# 2. Run full-layout pump-off linear validation.
# 3. Extract dispersion and stopband diagnostics.
# 4. Build reduced nonlinear layouts.
# 5. Solve pump-only harmonic balance on reduced layouts.
# 6. Optionally run a small-signal gain smoke calculation.
# 7. Export JSON/NPZ/Markdown artifacts.
#
# The nonlinear backend used here is the dense/reference backend, so the
# nonlinear section intentionally runs on reduced layouts. Full 20,000-cell
# nonlinear HB requires the later block-banded / matrix-free backend.

# %%
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math
import os
import sys
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

print("JAX backend:", jax.default_backend())
print("JAX x64:", bool(jax.config.jax_enable_x64))


# %% [markdown]
# ## Imports from the simulator package

# %%
from twpa.core.params import NonlinearParams
from twpa.linear.cells import CellModelConfig, CellModelKind
from twpa.linear.cascade import CascadeConfig, CascadeStrategy
from twpa.linear.dispersion import (
    DispersionConfig,
    DispersionExtractionMethod,
    StopbandMetric,
    compute_dp4wm_phase_matching,
    detect_stopbands,
    extract_layout_dispersion,
    validate_dispersion_result,
)
from twpa.linear.coarsening import (
    CoarseningConfig,
    CoarseningMethod,
    coarsen_layout,
    make_uniform_surrogate_layout,
)
from twpa.nonlinear.distributed_hb import DistributedHBConfig, DistributedHBTerminationKind
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    pump_solution_table,
    solve_pump_hb_ladder,
)
from twpa.solvers.hb_solver import DenseNewtonConfig, LinearSolveMethod
from twpa.core.hb_fft import HBProjectionConfig
from twpa.workflows.industrial_100mm import (
    Industrial100mmWorkflowConfig,
    IndustrialLayoutSpec,
    IndustrialLinearStageConfig,
    IndustrialCoarseningStageConfig,
    IndustrialPumpStageConfig,
    IndustrialRunMode,
    build_industrial_layout,
    run_linear_stage,
    run_coarsening_stage,
    run_pump_stage,
    run_industrial_100mm_workflow,
    summarize_workflow_markdown,
)


# %% [markdown]
# ## Output directory

# %%
RUN_DIR = Path("runs/notebook_09_production_100mm_demo")
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("Run directory:", RUN_DIR.resolve())


# %% [markdown]
# ## Helper functions

# %%
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
    if isinstance(obj, (tuple, list)):
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


def print_section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def safe_float(x: Any) -> float:
    return float(np.asarray(x))


# %% [markdown]
# ## Device / layout definition
#
# This is a synthetic 100 mm line with 20,000 cells. The default parameters are
# intentionally conservative and are meant to exercise the simulator mechanics,
# not to represent a final fitted device.

# %%
layout_spec = IndustrialLayoutSpec(
    length_m=0.100,
    n_cells=20_000,
    z0_ohm=50.0,
    phase_velocity_m_per_s=1.20e8,
    L_per_m_H=None,
    C_per_m_F=None,
    R_per_m_ohm=0.0,
    G_per_m_S=0.0,
    stub_period_cells=0,
    stub_offset=0,
    C_stub_loaded_F=0.0,
    C_stub_loaded_fraction_of_base=0.0,
    name="demo_100mm_20000cell",
)

layout = build_industrial_layout(layout_spec)

print_section("Full layout")
print(layout.summary())
print("Cell dx:", layout_spec.dx_m, "m")
print("L per m:", layout_spec.effective_L_per_m_H, "H/m")
print("C per m:", layout_spec.effective_C_per_m_F, "F/m")


# %% [markdown]
# ## Linear validation configuration

# %%
frequency_hz = jnp.linspace(1.0e9, 16.0e9, 401, dtype=jnp.float64)

cell_model = CellModelConfig(
    kind=CellModelKind.PI,
    include_stub_capacitance=True,
    include_resonator_loading=True,
)

cascade_config = CascadeConfig(
    strategy=CascadeStrategy.AUTO,
    chunk_size=512,
    cells_per_supercell=1,
    allow_remainder=True,
)

dispersion_config = DispersionConfig(
    method=DispersionExtractionMethod.BOTH,
    cells_per_supercell=1,
    stopband_s21_threshold_db=-10.0,
    stopband_alpha_threshold_np_per_m=1.0,
)

linear_stage_config = IndustrialLinearStageConfig(
    frequency_min_hz=float(frequency_hz[0]),
    frequency_max_hz=float(frequency_hz[-1]),
    n_frequency_points=int(frequency_hz.shape[0]),
    cell_model=cell_model,
    cascade=cascade_config,
    dispersion=dispersion_config,
    expected_stopband=None,
    stopband_metric=StopbandMetric.BOTH,
    stopband_s21_threshold_db=-10.0,
    stopband_alpha_threshold_np_per_m=1.0,
)


# %% [markdown]
# ## Run full 20,000-cell linear validation

# %%
print_section("Running full-layout linear validation")

linear_result = run_linear_stage(layout, linear_stage_config)

print("Linear status:", linear_result.status.value)
print("S21 dB min/max:", linear_result.scan.to_dict()["s21_db_min"], linear_result.scan.to_dict()["s21_db_max"])
print("Stopbands:", len(linear_result.stopbands))
print("Dispersion validation passed:", linear_result.dispersion_report.get("passed"))


# %%
linear_summary_path = write_json(
    RUN_DIR / "linear_stage_summary.json",
    linear_result.to_dict(),
)

linear_arrays_path = write_npz(
    RUN_DIR / "linear_stage_arrays.npz",
    frequency_hz=frequency_hz,
    s=linear_result.scan.s,
    s21=linear_result.scan.s21,
    s21_db=linear_result.scan.s21_db,
    abcd=linear_result.scan.abcd,
    beta_eff_rad_per_m=linear_result.scan.beta_eff_rad_per_m,
    group_delay_s=linear_result.scan.group_delay_s,
    beta_preferred_rad_per_m=linear_result.dispersion.beta_preferred_rad_per_m,
    alpha_preferred_np_per_m=linear_result.dispersion.alpha_preferred_np_per_m,
)

print("Wrote:", linear_summary_path)
print("Wrote:", linear_arrays_path)


# %% [markdown]
# ## Phase-matching diagnostic
#
# This gives a first pump-off DP4WM phase-matching scan. It does not replace the
# pump-HB + small-signal gain calculation, but it is useful for selecting pump
# and signal regions.

# %%
print_section("DP4WM phase-matching diagnostic")

pump_frequency_hz = 8.0e9
signal_frequency_hz = jnp.linspace(4.0e9, 7.5e9, 151, dtype=jnp.float64)

phase_matching = compute_dp4wm_phase_matching(
    linear_result.dispersion,
    pump_frequency_hz=pump_frequency_hz,
    signal_frequency_hz=signal_frequency_hz,
    nonlinear_delta_beta_rad_per_m=0.0,
)

phase_matching_summary = phase_matching.to_dict()

print("Pump GHz:", pump_frequency_hz / 1e9)
print("Signal GHz range:", float(signal_frequency_hz[0] / 1e9), float(signal_frequency_hz[-1] / 1e9))
print("Phase matching keys:", list(phase_matching_summary.keys()))

phase_matching_path = write_json(
    RUN_DIR / "phase_matching_summary.json",
    phase_matching_summary,
)

phase_matching_arrays_path = write_npz(
    RUN_DIR / "phase_matching_arrays.npz",
    signal_frequency_hz=signal_frequency_hz,
    **{
        key: value
        for key, value in phase_matching_summary.items()
        if hasattr(value, "shape")
    },
)

print("Wrote:", phase_matching_path)


# %% [markdown]
# ## Build reduced nonlinear layouts
#
# The dense nonlinear HB backend should not be used directly on 20,000 cells.
# We create reduced effective layouts and run pump-HB on those.

# %%
print_section("Building reduced layouts")

target_cell_counts = [50, 100, 200]

reduced_layouts = {}

for target_n in target_cell_counts:
    if target_n >= layout.n_cells:
        reduced_layouts[target_n] = layout
        continue

    factor = max(1, int(round(layout.n_cells / target_n)))

    result = coarsen_layout(
        layout,
        CoarseningConfig(
            method=CoarseningMethod.EXACT_GROUP_SUM,
            factor=factor,
            target_n_cells=None,
            allow_remainder=True,
        ),
        name=f"{layout.name}_Neff{target_n}",
    )

    reduced_layouts[target_n] = result.reduced
    print(f"Target {target_n}: factor={factor}, actual={result.reduced.n_cells}")

for target_n, red in reduced_layouts.items():
    print(target_n, red.summary())


# %% [markdown]
# ## Nonlinear and pump-HB configuration

# %%
nonlinear_params = NonlinearParams(
    I_star_A=1.0e-3,
    beta_nl=1.0,
    quartic_coefficient=0.0,
    dc_bias_A=0.0,
)

drive = PumpDriveConfig.from_current_rms(
    pump_frequency_hz=pump_frequency_hz,
    current_rms_A=1.0e-8,
    source_impedance_ohm=50.0,
    pump_label="pump",
    phase_rad=0.0,
    input_node=0,
)

distributed_config = DistributedHBConfig(
    input_node=0,
    output_node=-1,
    termination_kind=DistributedHBTerminationKind.SHUNT_CONDUCTANCE,
    source_conductance_S=1.0 / 50.0,
    load_conductance_S=1.0 / 50.0,
    include_stub_capacitance=True,
    include_series_resistance=True,
    use_layout_shunt_conductance=True,
    name="demo_distributed_hb",
)

projection_config = HBProjectionConfig(
    n_time_samples=None,
    oversampling=8,
    force_real_time_signal=True,
    enforce_conjugate_symmetry=True,
)

solver_config = DenseNewtonConfig(
    max_iter=40,
    abs_tol=1e-9,
    rel_tol=1e-9,
    step_tol=1e-12,
    damping_initial=1.0,
    damping_min=1e-6,
    regularization=0.0,
    linear_solve_method=LinearSolveMethod.AUTO,
    fail_on_nonconvergence=False,
    verbose=True,
)

pump_config = PumpHBLadderConfig(
    n_pump_harmonics=3,
    include_negative_frequencies=True,
    include_dc=False,
    distributed=distributed_config,
    projection=projection_config,
    solver=solver_config,
    name="demo_pump_hb",
)

print("Nonlinear params:", nonlinear_params.to_dict())
print("Drive:", drive.to_dict())
print("Pump config:", pump_config.to_dict())


# %% [markdown]
# ## Run pump-HB convergence over reduced layouts

# %%
print_section("Running pump-HB on reduced layouts")

pump_results = {}

for target_n, red_layout in reduced_layouts.items():
    print_section(f"Pump HB: N_eff={target_n}, actual cells={red_layout.n_cells}")

    result = solve_pump_hb_ladder(
        red_layout,
        nonlinear_params,
        drive=drive,
        pump_config=pump_config,
        metadata={
            "notebook": "09_production_100mm_demo",
            "target_n_cells": target_n,
        },
    )

    pump_results[target_n] = result

    print("Converged:", result.converged)
    print("Residual:", result.residual.norm)
    print("Max I/I*:", result.profile.max_pump_current_ratio)
    print("Pump output/input dB:", result.profile.output_to_input_voltage_gain_db)
    print()
    print(pump_solution_table(result))

    out_prefix = RUN_DIR / f"pump_Neff{target_n}"
    write_json(out_prefix.with_suffix(".json"), result.to_dict())
    write_npz(
        out_prefix.with_suffix(".npz"),
        frequencies_hz=result.frequency_plan.frequencies_hz,
        node_voltage_coeffs_V=result.state.node_voltage_coeffs_V,
        branch_current_coeffs_A=result.state.branch_current_coeffs_A,
        injected_current_coeffs_A=result.distributed_result.injected_current_coeffs_A,
        residual_kcl_A=result.residual.kcl_A,
        residual_branch_kvl_V=result.residual.branch_kvl_V,
    )


# %% [markdown]
# ## Pump convergence summary

# %%
pump_convergence_rows = []

for target_n, result in pump_results.items():
    pump_convergence_rows.append(
        {
            "target_n_cells": target_n,
            "actual_n_cells": result.layout.n_cells,
            "converged": result.converged,
            "residual_norm": result.residual.norm,
            "max_I_over_Istar": result.profile.max_pump_current_ratio,
            "max_node_voltage_abs_V": result.profile.max_node_voltage_abs_V,
            "max_branch_current_peak_A": result.profile.max_branch_current_peak_time_A,
            "pump_output_input_gain_db": result.profile.output_to_input_voltage_gain_db,
        }
    )

print_section("Pump convergence rows")
for row in pump_convergence_rows:
    print(row)

pump_convergence_path = write_json(
    RUN_DIR / "pump_convergence_summary.json",
    {
        "rows": pump_convergence_rows,
        "drive": drive.to_dict(),
        "nonlinear_params": nonlinear_params.to_dict(),
    },
)

print("Wrote:", pump_convergence_path)


# %% [markdown]
# ## Optional one-point gain smoke test
#
# This section requires the project to expose a compatible frequency-plan
# constructor for pump/signal/idler plans. If the constructor does not exist yet,
# this block will report the issue and can be skipped.

# %%
from twpa.core import frequency_plan as frequency_plan_module
from twpa.nonlinear.gain import GainSolveConfig, GainSweepConfig, solve_gain_sweep_from_pump

def try_make_gain_plan(
    *,
    pump_frequency_hz: float,
    signal_frequency_hz: float,
    idler_frequency_hz: float,
    n_pump_harmonics: int = 3,
):
    constructors = [
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
        "pump_label": "pump",
        "signal_label": "signal",
        "idler_label": "idler",
        "n_pump_harmonics": n_pump_harmonics,
        "n_harmonics": n_pump_harmonics,
        "include_negative": True,
        "include_negative_frequencies": True,
        "include_dc": False,
        "sort": "frequency",
    }

    errors = []

    for name in constructors:
        fn = getattr(frequency_plan_module, name, None)
        if fn is None:
            continue

        try:
            import inspect

            sig = inspect.signature(fn)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                plan = fn(**kwargs)
            else:
                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                plan = fn(**filtered)

            plan.position_of_label("pump")
            plan.position_of_label("signal")
            plan.position_of_label("idler")
            return plan

        except Exception as exc:
            errors.append(f"{name}: {exc}")

    # Fallback to a generic constructor if available.
    frequencies = [
        -3 * pump_frequency_hz,
        -pump_frequency_hz,
        -signal_frequency_hz,
        -idler_frequency_hz,
        pump_frequency_hz,
        signal_frequency_hz,
        idler_frequency_hz,
        3 * pump_frequency_hz,
    ]
    labels = [
        "-3pump",
        "-pump",
        "-signal",
        "-idler",
        "pump",
        "signal",
        "idler",
        "3pump",
    ]

    order = np.argsort(np.asarray(frequencies))
    frequencies = [frequencies[i] for i in order]
    labels = [labels[i] for i in order]

    generic_constructors = [
        getattr(frequency_plan_module, "make_frequency_plan", None),
        getattr(frequency_plan_module, "make_plan_from_frequencies", None),
        getattr(frequency_plan_module, "FrequencyPlan", None),
    ]

    candidate_kwargs = [
        {
            "frequencies_hz": jnp.asarray(frequencies, dtype=jnp.float64),
            "labels": tuple(labels),
            "reference_pump_hz": pump_frequency_hz,
            "kind": "custom",
        },
        {
            "frequencies_hz": jnp.asarray(frequencies, dtype=jnp.float64),
            "tone_labels": tuple(labels),
            "reference_pump_hz": pump_frequency_hz,
            "kind": "custom",
        },
    ]

    for ctor in generic_constructors:
        if ctor is None:
            continue
        for kw in candidate_kwargs:
            try:
                import inspect

                sig = inspect.signature(ctor)
                filtered = {k: v for k, v in kw.items() if k in sig.parameters}
                return ctor(**filtered)
            except Exception as exc:
                errors.append(f"{getattr(ctor, '__name__', ctor)}: {exc}")

    raise RuntimeError(
        "No compatible gain-plan constructor found. "
        f"Errors: {errors}"
    )


# %%
RUN_GAIN_SMOKE = False

if RUN_GAIN_SMOKE and pump_results:
    print_section("Running one-point gain smoke")

    best_target_n = max(pump_results.keys())
    pump_result = pump_results[best_target_n]

    signal_frequency_hz_smoke = 6.0e9
    idler_frequency_hz_smoke = 2.0 * pump_frequency_hz - signal_frequency_hz_smoke

    target_plan = try_make_gain_plan(
        pump_frequency_hz=pump_frequency_hz,
        signal_frequency_hz=signal_frequency_hz_smoke,
        idler_frequency_hz=idler_frequency_hz_smoke,
        n_pump_harmonics=3,
    )

    gain_config = GainSweepConfig(
        points=(
            GainSolveConfig(
                signal_label="signal",
                idler_label="idler",
                input_node=0,
                output_node=None,
                signal_current_rms_A=1.0e-12 + 0j,
                set_conjugate=True,
                input_impedance_ohm=50.0,
                output_impedance_ohm=50.0,
            ),
        ),
        require_all_converged=True,
        name="notebook_gain_smoke",
    )

    gain_sweep = solve_gain_sweep_from_pump(
        pump_result,
        target_plan=target_plan,
        sweep_config=gain_config,
    )

    print("Gain sweep passed:", gain_sweep.passed)
    print("Signal gain dB:", gain_sweep.points[0].signal_gain_db)
    print("Idler conversion dB:", gain_sweep.points[0].idler_conversion_db)

    write_json(RUN_DIR / "gain_smoke_summary.json", gain_sweep.to_dict())

else:
    print_section("Gain smoke skipped")
    print("Set RUN_GAIN_SMOKE = True after confirming the frequency-plan constructor is available.")


# %% [markdown]
# ## Full workflow wrapper demo
#
# This calls the higher-level workflow API. It is useful for checking the
# production wrapper itself. For notebook speed, it uses the same reduced pump
# target.

# %%
print_section("Running workflow wrapper demo")

workflow_config = Industrial100mmWorkflowConfig(
    mode=IndustrialRunMode.PUMP_ONLY,
    layout=layout_spec,
    linear=linear_stage_config,
    coarsening=IndustrialCoarseningStageConfig(
        enabled=True,
    ),
    pump=IndustrialPumpStageConfig(
        enabled=True,
        pump_layout_target_n_cells=100,
        pump_drive=drive,
        pump_config=pump_config,
        nonlinear_params=nonlinear_params,
    ),
    output_dir=str(RUN_DIR / "workflow_wrapper"),
    save_artifacts=True,
    name="notebook_09_workflow_wrapper",
)

workflow_result = run_industrial_100mm_workflow(workflow_config)

print("Workflow status:", workflow_result.status.value)
print("Workflow passed:", workflow_result.passed)
print(summarize_workflow_markdown(workflow_result))

write_json(
    RUN_DIR / "workflow_wrapper_summary.json",
    workflow_result.to_dict(),
)


# %% [markdown]
# ## Final artifact index

# %%
artifact_index = {
    "run_dir": str(RUN_DIR.resolve()),
    "linear_stage_summary": str(linear_summary_path),
    "linear_stage_arrays": str(linear_arrays_path),
    "phase_matching_summary": str(phase_matching_path),
    "pump_convergence_summary": str(pump_convergence_path),
    "workflow_wrapper_summary": str(RUN_DIR / "workflow_wrapper_summary.json"),
    "pump_results": {
        str(target_n): {
            "json": str(RUN_DIR / f"pump_Neff{target_n}.json"),
            "npz": str(RUN_DIR / f"pump_Neff{target_n}.npz"),
        }
        for target_n in pump_results.keys()
    },
}

write_json(RUN_DIR / "artifact_index.json", artifact_index)

print_section("Artifact index")
print(json.dumps(artifact_index, indent=2))


# %% [markdown]
# ## Notes for scaling to the full industrial backend
#
# At this point, the workflow should have:
#
# - a full 20,000-cell linear response,
# - dispersion and phase-matching diagnostics,
# - reduced-layout pump-HB convergence points,
# - exported pump solutions for downstream linearization/gain studies.
#
# The next engineering step is replacing the dense distributed HB internals with
# a structured backend:
#
# - block-banded residual assembly,
# - matrix-free JVPs,
# - Newton-Krylov or sparse direct solves,
# - continuation over pump current/power,
# - validation against the dense reduced-layout results produced here.