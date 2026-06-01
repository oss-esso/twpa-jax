"""
twpa.plotting.diagnostics
=========================

Matplotlib diagnostic plots for TWPA simulation and inference workflows.

The functions in this module are intentionally lightweight and object-tolerant:
they accept simulator result objects when available, but also accept raw arrays.

All plotting functions return ``(fig, ax)`` and do not call ``plt.show()``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import jax.numpy as jnp


ArrayLike = Any


@dataclass(frozen=True)
class PlotConfig:
    """
    Common plotting configuration.
    """

    figsize: tuple[float, float] = (7.0, 4.5)
    dpi: int = 140
    title: str | None = None
    grid: bool = True
    tight_layout: bool = True
    frequency_unit: str = "GHz"
    length_unit: str = "mm"
    linewidth: float = 1.8
    marker: str | None = None
    label: str | None = None
    name: str = "plot"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency_unit", self.frequency_unit.strip())
        object.__setattr__(self, "length_unit", self.length_unit.strip())
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if self.linewidth <= 0.0:
            raise ValueError("linewidth must be positive")

    def with_updates(self, **kwargs: Any) -> "PlotConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "figsize": tuple(float(v) for v in self.figsize),
            "dpi": int(self.dpi),
            "title": self.title,
            "grid": self.grid,
            "tight_layout": self.tight_layout,
            "frequency_unit": self.frequency_unit,
            "length_unit": self.length_unit,
            "linewidth": self.linewidth,
            "marker": self.marker,
            "label": self.label,
            "name": self.name,
        }


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _asarray(x: Any, *, dtype: Any | None = None) -> np.ndarray:
    if x is None:
        raise ValueError("Cannot convert None to array")
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _maybe_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            continue
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
        if isinstance(obj, Mapping) and name in obj:
            value = obj[name]
            if value is not None:
                return value
    return default


def _frequency_scale(unit: str) -> tuple[float, str]:
    unit_l = unit.lower()
    if unit_l == "hz":
        return 1.0, "Hz"
    if unit_l == "khz":
        return 1e-3, "kHz"
    if unit_l == "mhz":
        return 1e-6, "MHz"
    if unit_l == "ghz":
        return 1e-9, "GHz"
    raise ValueError(f"Unsupported frequency unit {unit!r}")


def _length_scale(unit: str) -> tuple[float, str]:
    unit_l = unit.lower()
    if unit_l == "m":
        return 1.0, "m"
    if unit_l == "cm":
        return 1e2, "cm"
    if unit_l == "mm":
        return 1e3, "mm"
    if unit_l in {"um", "µm"}:
        return 1e6, "µm"
    raise ValueError(f"Unsupported length unit {unit!r}")


def _new_fig_ax(config: PlotConfig):
    plt = _plt()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    return fig, ax


def _finish(fig: Any, ax: Any, config: PlotConfig, *, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if config.title:
        ax.set_title(config.title)
    if config.grid:
        ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if handles and any(labels):
        ax.legend()
    if config.tight_layout:
        fig.tight_layout()
    return fig, ax


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    dpi: int | None = None,
    bbox_inches: str = "tight",
) -> Path:
    """
    Save a matplotlib figure and return the path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches=bbox_inches)
    return p


def plot_s21(
    result_or_frequency: Any,
    s21_db: ArrayLike | None = None,
    *,
    config: PlotConfig | None = None,
    label: str | None = None,
):
    """
    Plot S21 magnitude in dB versus frequency.

    Accepts either:
        - a LinearScanResult-like object with ``frequency_hz`` and ``s21_db``;
        - ``frequency_hz`` plus explicit ``s21_db``.
    """
    cfg = config or PlotConfig(title="S21 response")
    fscale, flabel = _frequency_scale(cfg.frequency_unit)

    if s21_db is None:
        frequency_hz = _maybe_attr(result_or_frequency, "frequency_hz")
        s21_db_value = _maybe_attr(result_or_frequency, "s21_db")
        if s21_db_value is None:
            s = _maybe_attr(result_or_frequency, "s")
            if s is None:
                raise ValueError("Could not find s21_db or s array on result object")
            s_arr = _asarray(s)
            s21_db_value = 20.0 * np.log10(np.maximum(np.abs(s_arr[:, 1, 0]), 1e-300))
    else:
        frequency_hz = result_or_frequency
        s21_db_value = s21_db

    f = _asarray(frequency_hz, dtype=float) * fscale
    y = _asarray(s21_db_value, dtype=float)

    fig, ax = _new_fig_ax(cfg)
    ax.plot(
        f,
        y,
        linewidth=cfg.linewidth,
        marker=cfg.marker,
        label=label or cfg.label,
    )

    return _finish(fig, ax, cfg, xlabel=f"Frequency ({flabel})", ylabel="S21 (dB)")


def plot_dispersion(
    dispersion_or_frequency: Any,
    beta_rad_per_m: ArrayLike | None = None,
    *,
    alpha_np_per_m: ArrayLike | None = None,
    config: PlotConfig | None = None,
    quantity: str = "beta",
):
    """
    Plot extracted dispersion.

    Parameters
    ----------
    quantity:
        ``"beta"``, ``"alpha"``, or ``"both"``.
    """
    cfg = config or PlotConfig(title="Dispersion")
    fscale, flabel = _frequency_scale(cfg.frequency_unit)

    if beta_rad_per_m is None:
        frequency_hz = _maybe_attr(dispersion_or_frequency, "frequency_hz")
        beta = _maybe_attr(
            dispersion_or_frequency,
            "beta_preferred_rad_per_m",
            "beta_eff_rad_per_m",
            "beta_rad_per_m",
        )
        alpha = _maybe_attr(
            dispersion_or_frequency,
            "alpha_preferred_np_per_m",
            "alpha_np_per_m",
            default=alpha_np_per_m,
        )
    else:
        frequency_hz = dispersion_or_frequency
        beta = beta_rad_per_m
        alpha = alpha_np_per_m

    if frequency_hz is None:
        raise ValueError("Could not find frequency_hz")
    if beta is None and quantity in {"beta", "both"}:
        raise ValueError("Could not find beta array")

    f = _asarray(frequency_hz, dtype=float) * fscale

    fig, ax = _new_fig_ax(cfg)

    if quantity in {"beta", "both"}:
        ax.plot(
            f,
            _asarray(beta, dtype=float),
            linewidth=cfg.linewidth,
            marker=cfg.marker,
            label="beta",
        )

    if quantity in {"alpha", "both"}:
        if alpha is None:
            raise ValueError("quantity requires alpha but alpha array is missing")
        ax.plot(
            f,
            _asarray(alpha, dtype=float),
            linewidth=cfg.linewidth,
            marker=cfg.marker,
            label="alpha",
        )

    ylabel = {
        "beta": "β (rad/m)",
        "alpha": "α (Np/m)",
        "both": "β (rad/m), α (Np/m)",
    }.get(quantity, "Dispersion")

    return _finish(fig, ax, cfg, xlabel=f"Frequency ({flabel})", ylabel=ylabel)


def plot_stopbands(
    result_or_frequency: Any,
    s21_db: ArrayLike | None = None,
    stopbands: Sequence[Any] | None = None,
    *,
    config: PlotConfig | None = None,
):
    """
    Plot S21 and shade detected stopbands.

    ``stopbands`` may contain objects or dictionaries with start/end frequency
    fields such as ``start_frequency_hz`` / ``end_frequency_hz``.
    """
    cfg = config or PlotConfig(title="Stopband diagnostics")
    fig, ax = plot_s21(result_or_frequency, s21_db, config=cfg)

    if stopbands is None:
        stopbands = _maybe_attr(result_or_frequency, "stopbands", default=())

    fscale, _ = _frequency_scale(cfg.frequency_unit)

    for band in stopbands or ():
        start = _maybe_attr(
            band,
            "start_frequency_hz",
            "frequency_start_hz",
            "f_start_hz",
            "start_hz",
        )
        end = _maybe_attr(
            band,
            "end_frequency_hz",
            "frequency_end_hz",
            "f_end_hz",
            "end_hz",
        )

        if start is None or end is None:
            continue

        ax.axvspan(float(start) * fscale, float(end) * fscale, alpha=0.2)

    if cfg.tight_layout:
        fig.tight_layout()

    return fig, ax


def _layout_positions_m(layout: Any, n_cells: int) -> np.ndarray:
    length_m = _maybe_attr(layout, "length_m")
    if length_m is not None:
        lengths = _asarray(length_m, dtype=float)
        if lengths.shape == (n_cells,):
            return np.concatenate([[0.0], np.cumsum(lengths)])
    total = _maybe_attr(layout, "total_length_m", default=None)
    if total is None:
        total = 1.0
    return np.linspace(0.0, float(total), n_cells + 1)


def plot_pump_profile(
    pump_result: Any,
    *,
    config: PlotConfig | None = None,
    quantity: str = "current_ratio",
):
    """
    Plot a pump-HB spatial profile.

    Supported quantities:
        - ``"current_ratio"``: branch pump-current magnitude divided by I*
          when available, otherwise normalized by max current.
        - ``"current_abs"``: branch pump-current magnitude.
        - ``"voltage_abs"``: node pump-voltage magnitude.
    """
    cfg = config or PlotConfig(title="Pump profile")
    lscale, llabel = _length_scale(cfg.length_unit)

    state = _maybe_attr(pump_result, "state")
    layout = _maybe_attr(pump_result, "layout")
    plan = _maybe_attr(pump_result, "frequency_plan", "plan")

    if state is None:
        distributed = _maybe_attr(pump_result, "distributed_result")
        state = _maybe_attr(distributed, "state")
        layout = layout or _maybe_attr(distributed, "layout")
        plan = plan or _maybe_attr(distributed, "frequency_plan", "plan")

    if state is None:
        raise ValueError("Could not find pump state")

    pump_label = "pump"
    drive = _maybe_attr(pump_result, "drive", "pump_drive")
    if drive is not None:
        pump_label = _maybe_attr(drive, "pump_label", default="pump")

    if plan is not None and hasattr(plan, "position_of_label"):
        pump_idx = int(plan.position_of_label(pump_label))
    else:
        pump_idx = 0

    fig, ax = _new_fig_ax(cfg)

    if quantity in {"current_ratio", "current_abs"}:
        branch = _maybe_attr(state, "branch_current_coeffs_A")
        if branch is None:
            raise ValueError("branch_current_coeffs_A is required for current profile")
        branch_arr = _asarray(branch)
        y = np.abs(branch_arr[pump_idx])
        n_cells = y.shape[0]
        x = 0.5 * (_layout_positions_m(layout, n_cells)[:-1] + _layout_positions_m(layout, n_cells)[1:])

        ylabel = "|I_pump| (A)"
        if quantity == "current_ratio":
            nonlinear = _maybe_attr(pump_result, "nonlinear_params")
            I_star = _maybe_attr(nonlinear, "I_star_A")
            if I_star is None:
                profile = _maybe_attr(pump_result, "profile")
                I_star = _maybe_attr(profile, "I_star_A")
            if I_star is not None and float(I_star) > 0.0:
                y = y / float(I_star)
                ylabel = "|I_pump| / I*"
            else:
                ymax = max(float(np.nanmax(y)), 1e-300)
                y = y / ymax
                ylabel = "Normalized |I_pump|"

    elif quantity == "voltage_abs":
        voltage = _maybe_attr(state, "node_voltage_coeffs_V")
        if voltage is None:
            raise ValueError("node_voltage_coeffs_V is required for voltage profile")
        voltage_arr = _asarray(voltage)
        y = np.abs(voltage_arr[pump_idx])
        n_cells = y.shape[0] - 1
        x = _layout_positions_m(layout, n_cells)
        ylabel = "|V_pump| (V)"

    else:
        raise ValueError(f"Unsupported pump profile quantity {quantity!r}")

    ax.plot(
        x * lscale,
        y,
        linewidth=cfg.linewidth,
        marker=cfg.marker,
        label=cfg.label,
    )

    return _finish(fig, ax, cfg, xlabel=f"Position ({llabel})", ylabel=ylabel)


def plot_newton_history(
    result_or_records: Any,
    *,
    config: PlotConfig | None = None,
    y: str = "residual_norm",
):
    """
    Plot Newton or Newton-Krylov iteration history.

    Accepts a result object with ``records``/``history`` or a sequence of record
    objects/dictionaries.
    """
    cfg = config or PlotConfig(title="Newton history")

    records = _maybe_attr(result_or_records, "records", "history", default=result_or_records)
    records = list(records or [])

    if not records:
        raise ValueError("No iteration records found")

    iterations = []
    values = []

    for idx, rec in enumerate(records):
        it = _maybe_attr(rec, "iteration", "iter", default=idx)
        val = _maybe_attr(rec, y)
        if val is None and y == "residual_norm":
            val = _maybe_attr(rec, "norm", "residual")
        if val is None:
            continue
        iterations.append(int(it))
        values.append(float(val))

    if not values:
        raise ValueError(f"No values found for {y!r}")

    fig, ax = _new_fig_ax(cfg)
    ax.semilogy(
        iterations,
        values,
        linewidth=cfg.linewidth,
        marker=cfg.marker or "o",
        label=cfg.label,
    )

    return _finish(fig, ax, cfg, xlabel="Iteration", ylabel=y)


def plot_fit_history(
    fit_result_or_records: Any,
    *,
    config: PlotConfig | None = None,
    y: str = "loss",
):
    """
    Plot fitting objective history.

    Accepts a FitResult-like object with ``records`` or an explicit record
    sequence.
    """
    cfg = config or PlotConfig(title="Fit history")

    records = _maybe_attr(fit_result_or_records, "records", default=fit_result_or_records)
    records = list(records or [])

    if not records:
        raise ValueError("No fit records found")

    xs = []
    ys = []

    for idx, rec in enumerate(records):
        ev = _maybe_attr(rec, "evaluation_index", default=idx + 1)
        val = _maybe_attr(rec, y)
        if val is None:
            continue
        xs.append(int(ev))
        ys.append(float(val))

    if not ys:
        raise ValueError(f"No values found for {y!r}")

    fig, ax = _new_fig_ax(cfg)
    ax.semilogy(
        xs,
        ys,
        linewidth=cfg.linewidth,
        marker=cfg.marker,
        label=cfg.label,
    )

    return _finish(fig, ax, cfg, xlabel="Evaluation", ylabel=y)


def plot_recovery_truth_vs_fit(
    recovery_or_comparison: Any,
    *,
    config: PlotConfig | None = None,
    parameter_names: Sequence[str] | None = None,
):
    """
    Plot fitted parameter values versus true values.

    Accepts:
        - output of ``compare_fit_to_truth``;
        - a RecoveryTrialResult-like object with ``truth_comparison``;
        - a RecoveryExperimentResult-like object, in which case all trial rows
          are concatenated.
    """
    cfg = config or PlotConfig(title="Recovery: truth vs fit")

    rows: list[Mapping[str, Any]] = []

    if hasattr(recovery_or_comparison, "trials"):
        for trial in recovery_or_comparison.trials:
            comp = _maybe_attr(trial, "truth_comparison", default={})
            rows.extend(comp.get("rows", []))
    else:
        comp = _maybe_attr(recovery_or_comparison, "truth_comparison", default=recovery_or_comparison)
        rows.extend(comp.get("rows", []) if isinstance(comp, Mapping) else [])

    if parameter_names is not None:
        allowed = set(str(p) for p in parameter_names)
        rows = [row for row in rows if str(row.get("name")) in allowed]

    rows = [row for row in rows if row.get("fit") is not None and row.get("true") is not None]

    if not rows:
        raise ValueError("No truth/fit rows available to plot")

    truth = np.asarray([float(row["true"]) for row in rows], dtype=float)
    fit = np.asarray([float(row["fit"]) for row in rows], dtype=float)
    labels = [str(row.get("name", i)) for i, row in enumerate(rows)]

    fig, ax = _new_fig_ax(cfg)
    ax.scatter(truth, fit)

    lo = float(np.nanmin([np.nanmin(truth), np.nanmin(fit)]))
    hi = float(np.nanmax([np.nanmax(truth), np.nanmax(fit)]))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)

    for x, yv, label in zip(truth, fit, labels):
        ax.annotate(label, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)

    return _finish(fig, ax, cfg, xlabel="True parameter", ylabel="Fitted parameter")


def plot_array_profile(
    x: ArrayLike,
    y: ArrayLike,
    *,
    config: PlotConfig | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
):
    """
    Generic single-profile plot helper used by scripts.
    """
    cfg = config or PlotConfig()

    fig, ax = _new_fig_ax(cfg)
    ax.plot(
        _asarray(x, dtype=float),
        _asarray(y, dtype=float),
        linewidth=cfg.linewidth,
        marker=cfg.marker,
        label=cfg.label,
    )
    return _finish(fig, ax, cfg, xlabel=xlabel, ylabel=ylabel)


def save_diagnostic_bundle(
    output_dir: str | Path,
    *,
    linear_result: Any | None = None,
    dispersion_result: Any | None = None,
    pump_result: Any | None = None,
    fit_result: Any | None = None,
    recovery_result: Any | None = None,
    prefix: str = "diagnostics",
    config: PlotConfig | None = None,
) -> dict[str, str]:
    """
    Save a standard set of diagnostic figures when the corresponding result
    objects are provided.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = config or PlotConfig()
    paths: dict[str, str] = {}

    if linear_result is not None:
        fig, _ = plot_s21(
            linear_result,
            config=cfg.with_updates(title="S21 response"),
        )
        p = save_figure(fig, out / f"{prefix}_s21.png", dpi=cfg.dpi)
        paths["s21_png"] = str(p)
        _plt().close(fig)

    if dispersion_result is not None:
        fig, _ = plot_dispersion(
            dispersion_result,
            config=cfg.with_updates(title="Dispersion"),
            quantity="beta",
        )
        p = save_figure(fig, out / f"{prefix}_dispersion_beta.png", dpi=cfg.dpi)
        paths["dispersion_beta_png"] = str(p)
        _plt().close(fig)

    if pump_result is not None:
        fig, _ = plot_pump_profile(
            pump_result,
            config=cfg.with_updates(title="Pump current profile"),
            quantity="current_ratio",
        )
        p = save_figure(fig, out / f"{prefix}_pump_profile.png", dpi=cfg.dpi)
        paths["pump_profile_png"] = str(p)
        _plt().close(fig)

    if fit_result is not None:
        fig, _ = plot_fit_history(
            fit_result,
            config=cfg.with_updates(title="Fit objective history"),
            y="loss",
        )
        p = save_figure(fig, out / f"{prefix}_fit_history.png", dpi=cfg.dpi)
        paths["fit_history_png"] = str(p)
        _plt().close(fig)

    if recovery_result is not None:
        fig, _ = plot_recovery_truth_vs_fit(
            recovery_result,
            config=cfg.with_updates(title="Recovery truth vs fit"),
        )
        p = save_figure(fig, out / f"{prefix}_recovery_truth_vs_fit.png", dpi=cfg.dpi)
        paths["recovery_truth_vs_fit_png"] = str(p)
        _plt().close(fig)

    return paths


__all__ = [
    "ArrayLike",
    "PlotConfig",
    "save_figure",
    "plot_s21",
    "plot_dispersion",
    "plot_stopbands",
    "plot_pump_profile",
    "plot_newton_history",
    "plot_fit_history",
    "plot_recovery_truth_vs_fit",
    "plot_array_profile",
    "save_diagnostic_bundle",
]