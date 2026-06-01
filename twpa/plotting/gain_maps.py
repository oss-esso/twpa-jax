"""
twpa.plotting.gain_maps
=======================

Matplotlib plotting helpers for TWPA gain sweeps, gain maps, operating maps,
and compression sweeps.

All functions return ``(fig, ax)`` and do not call ``plt.show()``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ArrayLike = Any


@dataclass(frozen=True)
class GainMapPlotConfig:
    """
    Common gain-map plotting configuration.
    """

    figsize: tuple[float, float] = (7.0, 4.8)
    dpi: int = 140
    title: str | None = None
    grid: bool = True
    tight_layout: bool = True
    frequency_unit: str = "GHz"
    pump_power_unit: str = "dBm"
    gain_label: str = "Gain (dB)"
    linewidth: float = 1.8
    marker: str | None = None
    cmap: str = "viridis"
    show_colorbar: bool = True
    contour_levels: Sequence[float] | None = None
    vmin: float | None = None
    vmax: float | None = None
    label: str | None = None
    name: str = "gain_map_plot"

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if self.linewidth <= 0.0:
            raise ValueError("linewidth must be positive")
        if self.contour_levels is not None:
            object.__setattr__(self, "contour_levels", tuple(float(x) for x in self.contour_levels))

    def with_updates(self, **kwargs: Any) -> "GainMapPlotConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "figsize": tuple(float(v) for v in self.figsize),
            "dpi": int(self.dpi),
            "title": self.title,
            "grid": self.grid,
            "tight_layout": self.tight_layout,
            "frequency_unit": self.frequency_unit,
            "pump_power_unit": self.pump_power_unit,
            "gain_label": self.gain_label,
            "linewidth": self.linewidth,
            "marker": self.marker,
            "cmap": self.cmap,
            "show_colorbar": self.show_colorbar,
            "contour_levels": None if self.contour_levels is None else list(self.contour_levels),
            "vmin": self.vmin,
            "vmax": self.vmax,
            "label": self.label,
            "name": self.name,
        }


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _asarray(x: Any, *, dtype: Any | None = None) -> np.ndarray:
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
    u = unit.strip().lower()
    if u == "hz":
        return 1.0, "Hz"
    if u == "khz":
        return 1e-3, "kHz"
    if u == "mhz":
        return 1e-6, "MHz"
    if u == "ghz":
        return 1e-9, "GHz"
    raise ValueError(f"Unsupported frequency unit {unit!r}")


def _new_fig_ax(config: GainMapPlotConfig):
    plt = _plt()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    return fig, ax


def _finish(fig: Any, ax: Any, config: GainMapPlotConfig, *, xlabel: str, ylabel: str):
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


def save_gain_figure(
    fig: Any,
    path: str | Path,
    *,
    dpi: int | None = None,
    bbox_inches: str = "tight",
) -> Path:
    """
    Save a gain-map figure.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches=bbox_inches)
    return p


def _gain_points_to_arrays(gain_sweep: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Extract signal frequency, signal gain dB, and idler conversion dB arrays from
    a GainSweepResult-like object.
    """
    points = list(_maybe_attr(gain_sweep, "points", default=()) or ())
    if not points:
        raise ValueError("gain_sweep has no points")

    signal_frequency = []
    signal_gain = []
    idler_conversion = []

    for idx, point in enumerate(points):
        freq = _maybe_attr(point, "signal_frequency_hz", "frequency_hz", default=None)

        if freq is None:
            cfg = _maybe_attr(point, "config", "solve_config", default=None)
            freq = _maybe_attr(cfg, "signal_frequency_hz", default=None)

        if freq is None:
            metadata = _maybe_attr(point, "metadata", default={}) or {}
            freq = _maybe_attr(metadata, "signal_frequency_hz", default=np.nan)

        gain = _maybe_attr(point, "signal_gain_db", "gain_db", default=None)
        if gain is None:
            raise ValueError(f"Could not extract signal_gain_db from point {idx}")

        idler = _maybe_attr(point, "idler_conversion_db", "conversion_db", default=np.nan)

        signal_frequency.append(float(freq))
        signal_gain.append(float(gain))
        idler_conversion.append(np.nan if idler is None else float(idler))

    idler_array = np.asarray(idler_conversion, dtype=float)
    if np.all(np.isnan(idler_array)):
        idler_array = None

    return (
        np.asarray(signal_frequency, dtype=float),
        np.asarray(signal_gain, dtype=float),
        idler_array,
    )


def plot_gain_sweep(
    gain_sweep_or_frequency: Any,
    gain_db: ArrayLike | None = None,
    *,
    idler_conversion_db: ArrayLike | None = None,
    config: GainMapPlotConfig | None = None,
    show_idler: bool = True,
):
    """
    Plot one signal-gain sweep versus signal frequency.

    Accepts either:
        - a GainSweepResult-like object with ``points``;
        - explicit ``signal_frequency_hz`` and ``gain_db`` arrays.
    """
    cfg = config or GainMapPlotConfig(title="Gain sweep")
    fscale, flabel = _frequency_scale(cfg.frequency_unit)

    if gain_db is None:
        frequency_hz, signal_gain_db, idler_db = _gain_points_to_arrays(gain_sweep_or_frequency)
    else:
        frequency_hz = _asarray(gain_sweep_or_frequency, dtype=float)
        signal_gain_db = _asarray(gain_db, dtype=float)
        idler_db = None if idler_conversion_db is None else _asarray(idler_conversion_db, dtype=float)

    fig, ax = _new_fig_ax(cfg)

    ax.plot(
        frequency_hz * fscale,
        signal_gain_db,
        linewidth=cfg.linewidth,
        marker=cfg.marker,
        label=cfg.label or "signal gain",
    )

    if show_idler and idler_db is not None:
        ax.plot(
            frequency_hz * fscale,
            idler_db,
            linewidth=cfg.linewidth,
            marker=cfg.marker,
            label="idler conversion",
        )

    return _finish(fig, ax, cfg, xlabel=f"Signal frequency ({flabel})", ylabel="Gain / conversion (dB)")


def _extract_gain_map_arrays(
    gain_map_or_x: Any,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract x/y/z arrays from a GainOperatingMap-like object or explicit arrays.
    """
    if y is not None and z is not None:
        x_arr = _asarray(gain_map_or_x, dtype=float)
        y_arr = _asarray(y, dtype=float)
        z_arr = _asarray(z, dtype=float)
        return x_arr, y_arr, z_arr

    obj = gain_map_or_x

    x_arr = _maybe_attr(
        obj,
        "signal_frequency_hz",
        "frequency_hz",
        "x",
        "x_values",
        default=None,
    )
    y_arr = _maybe_attr(
        obj,
        "pump_power_dbm",
        "pump_current_rms_A",
        "y",
        "y_values",
        default=None,
    )
    z_arr = _maybe_attr(
        obj,
        "signal_gain_db",
        "gain_db",
        "gain_db_grid",
        "z",
        "z_values",
        default=None,
    )

    if x_arr is None or y_arr is None or z_arr is None:
        if isinstance(obj, Mapping):
            for x_key in ["signal_frequency_hz", "frequency_hz", "x"]:
                for y_key in ["pump_power_dbm", "pump_current_rms_A", "y"]:
                    for z_key in ["signal_gain_db", "gain_db", "gain_db_grid", "z"]:
                        if x_key in obj and y_key in obj and z_key in obj:
                            return (
                                _asarray(obj[x_key], dtype=float),
                                _asarray(obj[y_key], dtype=float),
                                _asarray(obj[z_key], dtype=float),
                            )
        raise ValueError("Could not extract gain-map x/y/z arrays")

    return _asarray(x_arr, dtype=float), _asarray(y_arr, dtype=float), _asarray(z_arr, dtype=float)


def _reshape_map_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize gain-map arrays to X, Y, Z 2D grids.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if z.ndim == 2:
        if x.ndim == 1 and y.ndim == 1:
            X, Y = np.meshgrid(x, y)
            if z.shape == X.shape:
                return X, Y, z
            if z.T.shape == X.shape:
                return X, Y, z.T
        if x.shape == z.shape and y.shape == z.shape:
            return x, y, z
        raise ValueError(f"Could not align 2D z shape {z.shape} with x/y shapes {x.shape}/{y.shape}")

    if z.ndim == 1:
        if x.ndim == 1 and y.ndim == 1 and x.shape == y.shape == z.shape:
            x_unique = np.unique(x)
            y_unique = np.unique(y)
            X, Y = np.meshgrid(x_unique, y_unique)
            Z = np.full_like(X, np.nan, dtype=float)
            for xv, yv, zv in zip(x, y, z):
                xi = int(np.where(x_unique == xv)[0][0])
                yi = int(np.where(y_unique == yv)[0][0])
                Z[yi, xi] = zv
            return X, Y, Z

    raise ValueError(f"Unsupported gain-map array shapes: x={x.shape}, y={y.shape}, z={z.shape}")


def plot_gain_map(
    gain_map_or_x: Any,
    y: ArrayLike | None = None,
    gain_db: ArrayLike | None = None,
    *,
    config: GainMapPlotConfig | None = None,
    y_label: str | None = None,
):
    """
    Plot a 2D gain map.

    Accepts either:
        - a GainOperatingMap-like object/dictionary;
        - explicit ``x``, ``y``, and ``gain_db`` arrays.

    The x-axis is interpreted as signal frequency in Hz. The y-axis is usually
    pump power in dBm or pump current depending on the supplied arrays.
    """
    cfg = config or GainMapPlotConfig(title="Gain map")
    fscale, flabel = _frequency_scale(cfg.frequency_unit)

    x_arr, y_arr, z_arr = _extract_gain_map_arrays(gain_map_or_x, y, gain_db)
    X, Y, Z = _reshape_map_arrays(x_arr, y_arr, z_arr)

    fig, ax = _new_fig_ax(cfg)

    mesh = ax.pcolormesh(
        X * fscale,
        Y,
        Z,
        shading="auto",
        cmap=cfg.cmap,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
    )

    if cfg.show_colorbar:
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(cfg.gain_label)

    if cfg.contour_levels is not None:
        cs = ax.contour(
            X * fscale,
            Y,
            Z,
            levels=list(cfg.contour_levels),
            linewidths=0.8,
        )
        ax.clabel(cs, inline=True, fontsize=8)

    return _finish(
        fig,
        ax,
        cfg,
        xlabel=f"Signal frequency ({flabel})",
        ylabel=y_label or f"Pump power ({cfg.pump_power_unit})",
    )


def plot_operating_map(
    operating_map_or_x: Any,
    y: ArrayLike | None = None,
    metric: ArrayLike | None = None,
    *,
    config: GainMapPlotConfig | None = None,
    metric_label: str = "Metric",
    y_label: str | None = None,
):
    """
    Plot a generic operating map.

    This is the same as ``plot_gain_map`` but with a configurable colorbar
    label for non-gain metrics such as convergence, max current ratio, ripple,
    or bandwidth.
    """
    cfg = (config or GainMapPlotConfig(title="Operating map")).with_updates(
        gain_label=metric_label
    )

    return plot_gain_map(
        operating_map_or_x,
        y,
        metric,
        config=cfg,
        y_label=y_label,
    )


def _compression_points_to_arrays(sweep: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    points = list(_maybe_attr(sweep, "points", default=()) or ())
    if not points:
        raise ValueError("compression sweep has no points")

    power_dbm = []
    signal_gain_db = []
    matched_gain_db = []

    for idx, point in enumerate(points):
        p = _maybe_attr(point, "signal_power_dbm", "input_power_dbm", default=None)
        if p is None:
            raise ValueError(f"Could not extract signal_power_dbm from compression point {idx}")

        g = _maybe_attr(point, "signal_gain_db", "gain_db", default=None)
        if g is None:
            result = _maybe_attr(point, "result", default=None)
            obs = _maybe_attr(result, "observables", default=None)
            g = _maybe_attr(obs, "signal_gain_db", default=None)

        mg = _maybe_attr(point, "matched_power_gain_db", default=None)
        if mg is None:
            result = _maybe_attr(point, "result", default=None)
            obs = _maybe_attr(result, "observables", default=None)
            mg = _maybe_attr(obs, "matched_power_gain_db", default=np.nan)

        if g is None:
            raise ValueError(f"Could not extract gain from compression point {idx}")

        power_dbm.append(float(p))
        signal_gain_db.append(float(g))
        matched_gain_db.append(np.nan if mg is None else float(mg))

    mg_arr = np.asarray(matched_gain_db, dtype=float)
    if np.all(np.isnan(mg_arr)):
        mg_arr = None

    return (
        np.asarray(power_dbm, dtype=float),
        np.asarray(signal_gain_db, dtype=float),
        mg_arr,
    )


def plot_compression_sweep(
    sweep_or_power: Any,
    gain_db: ArrayLike | None = None,
    *,
    matched_power_gain_db: ArrayLike | None = None,
    reference_gain_db: float | None = None,
    compression_db: float = 1.0,
    config: GainMapPlotConfig | None = None,
):
    """
    Plot finite-signal gain compression versus input signal power.
    """
    cfg = config or GainMapPlotConfig(title="Gain compression")

    if gain_db is None:
        power_dbm, signal_gain_db, matched_gain = _compression_points_to_arrays(sweep_or_power)
        if reference_gain_db is None:
            ref = _maybe_attr(sweep_or_power, "small_signal_reference_gain_db", default=None)
            reference_gain_db = None if ref is None else float(ref)
    else:
        power_dbm = _asarray(sweep_or_power, dtype=float)
        signal_gain_db = _asarray(gain_db, dtype=float)
        matched_gain = None if matched_power_gain_db is None else _asarray(matched_power_gain_db, dtype=float)

    fig, ax = _new_fig_ax(cfg)

    ax.plot(
        power_dbm,
        signal_gain_db,
        linewidth=cfg.linewidth,
        marker=cfg.marker,
        label=cfg.label or "signal gain",
    )

    if matched_gain is not None:
        ax.plot(
            power_dbm,
            matched_gain,
            linewidth=cfg.linewidth,
            marker=cfg.marker,
            label="matched power gain",
        )

    ref_gain = float(reference_gain_db) if reference_gain_db is not None else float(signal_gain_db[0])
    target_gain = ref_gain - float(compression_db)

    ax.axhline(ref_gain, linestyle="--", linewidth=1.0, label="reference")
    ax.axhline(target_gain, linestyle=":", linewidth=1.0, label=f"{compression_db:g} dB compression")

    return _finish(fig, ax, cfg, xlabel="Input signal power (dBm)", ylabel="Gain (dB)")


def plot_bandwidth_summary(
    center_frequency_hz: ArrayLike,
    bandwidth_hz: ArrayLike,
    *,
    config: GainMapPlotConfig | None = None,
):
    """
    Plot bandwidth versus center frequency or operating point.
    """
    cfg = config or GainMapPlotConfig(title="Bandwidth summary")
    fscale, flabel = _frequency_scale(cfg.frequency_unit)

    x = _asarray(center_frequency_hz, dtype=float)
    y = _asarray(bandwidth_hz, dtype=float)

    fig, ax = _new_fig_ax(cfg)
    ax.plot(
        x * fscale,
        y * fscale,
        linewidth=cfg.linewidth,
        marker=cfg.marker,
        label=cfg.label,
    )

    return _finish(fig, ax, cfg, xlabel=f"Center frequency ({flabel})", ylabel=f"Bandwidth ({flabel})")


def gain_map_summary_table(
    gain_map_or_x: Any,
    y: ArrayLike | None = None,
    gain_db: ArrayLike | None = None,
) -> str:
    """
    Markdown summary table for a gain map.
    """
    x_arr, y_arr, z_arr = _extract_gain_map_arrays(gain_map_or_x, y, gain_db)
    X, Y, Z = _reshape_map_arrays(x_arr, y_arr, z_arr)

    return "\n".join(
        [
            "| quantity | value |",
            "|---|---:|",
            f"| signal frequency min GHz | {np.nanmin(X) / 1e9:.6g} |",
            f"| signal frequency max GHz | {np.nanmax(X) / 1e9:.6g} |",
            f"| pump/operating coordinate min | {np.nanmin(Y):.6g} |",
            f"| pump/operating coordinate max | {np.nanmax(Y):.6g} |",
            f"| gain min dB | {np.nanmin(Z):.6g} |",
            f"| gain max dB | {np.nanmax(Z):.6g} |",
            f"| grid shape | {Z.shape} |",
        ]
    )


def save_gain_diagnostic_bundle(
    output_dir: str | Path,
    *,
    gain_sweep: Any | None = None,
    gain_map: Any | None = None,
    compression_sweep: Any | None = None,
    prefix: str = "gain",
    config: GainMapPlotConfig | None = None,
) -> dict[str, str]:
    """
    Save standard gain diagnostic figures.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = config or GainMapPlotConfig()
    paths: dict[str, str] = {}

    if gain_sweep is not None:
        fig, _ = plot_gain_sweep(
            gain_sweep,
            config=cfg.with_updates(title="Gain sweep"),
        )
        p = save_gain_figure(fig, out / f"{prefix}_sweep.png", dpi=cfg.dpi)
        paths["gain_sweep_png"] = str(p)
        _plt().close(fig)

    if gain_map is not None:
        fig, _ = plot_gain_map(
            gain_map,
            config=cfg.with_updates(title="Gain map"),
        )
        p = save_gain_figure(fig, out / f"{prefix}_map.png", dpi=cfg.dpi)
        paths["gain_map_png"] = str(p)
        _plt().close(fig)

    if compression_sweep is not None:
        fig, _ = plot_compression_sweep(
            compression_sweep,
            config=cfg.with_updates(title="Gain compression"),
        )
        p = save_gain_figure(fig, out / f"{prefix}_compression.png", dpi=cfg.dpi)
        paths["compression_png"] = str(p)
        _plt().close(fig)

    return paths


__all__ = [
    "ArrayLike",
    "GainMapPlotConfig",
    "save_gain_figure",
    "plot_gain_sweep",
    "plot_gain_map",
    "plot_operating_map",
    "plot_compression_sweep",
    "plot_bandwidth_summary",
    "gain_map_summary_table",
    "save_gain_diagnostic_bundle",
]