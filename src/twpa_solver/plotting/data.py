"""Load and normalize saved gain-map outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MapData:
    """Saved map data loaded from one run directory."""

    run_dir: Path
    points: pd.DataFrame
    arrays: dict[str, np.ndarray]
    spectrum: dict[str, np.ndarray]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        return {name: data[name] for name in data.files}


def _rename_first(df: pd.DataFrame, names: tuple[str, ...], canonical: str) -> None:
    for name in names:
        if name in df.columns and canonical not in df.columns:
            df.rename(columns={name: canonical}, inplace=True)
            return


def _normalize_points(points: pd.DataFrame) -> pd.DataFrame:
    df = points.copy()
    _rename_first(df, ("pump_frequency_ghz", "pump_freq", "fp_ghz"), "pump_freq_ghz")
    _rename_first(df, ("pump_power", "power_dbm"), "pump_power_dbm")
    if "point_index" not in df.columns:
        df["point_index"] = np.arange(len(df), dtype=int)
    if "status" not in df.columns:
        df["status"] = "UNKNOWN"
    return df


def _first_array(data: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def _normalize_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(arrays)
    power = _first_array(out, ("pump_power_dbm_grid", "pump_power_dbm"))
    freq = _first_array(out, ("pump_freq_ghz_grid", "pump_frequency_ghz", "pump_freq_ghz"))
    gain = _first_array(out, ("peak_gain_db", "gain_db_warm", "gain_db"))
    if power is not None:
        out["pump_power_dbm_grid"] = power
    if freq is not None:
        out["pump_freq_ghz_grid"] = freq
    if gain is not None:
        out["peak_gain_db"] = gain
        out.setdefault("valid_mask", np.isfinite(gain))
    return out


def _flat_spectrum_from_grid(
    spectrum: dict[str, np.ndarray],
    points: pd.DataFrame,
) -> dict[str, np.ndarray]:
    gain_grid = _first_array(spectrum, ("gain_spectrum_db",))
    power_grid = _first_array(spectrum, ("pump_power_dbm",))
    freq_grid = _first_array(spectrum, ("pump_frequency_ghz", "pump_freq_ghz"))
    signal_grid = _first_array(spectrum, ("signal_ghz", "signal_freq_ghz"))
    if gain_grid is None or gain_grid.ndim != 3:
        return {}

    rows: list[np.ndarray] = []
    signal_rows: list[np.ndarray] = []
    row_power: list[float] = []
    row_freq: list[float] = []
    point_index: list[int] = []
    statuses: list[str] = []

    for pos, row in points.reset_index(drop=True).iterrows():
        i = int(row["i_power"]) if "i_power" in row else pos // gain_grid.shape[1]
        j = int(row["j_freq"]) if "j_freq" in row else pos % gain_grid.shape[1]
        if i < 0 or j < 0 or i >= gain_grid.shape[0] or j >= gain_grid.shape[1]:
            continue
        rows.append(np.asarray(gain_grid[i, j, :], dtype=float))
        if signal_grid is not None:
            sig = np.asarray(signal_grid)
            if sig.ndim == 1:
                signal_rows.append(sig.astype(float))
            elif sig.ndim == 2 and sig.shape[1] == gain_grid.shape[1]:
                signal_rows.append(sig[:, j].astype(float))
            elif sig.ndim == 2 and sig.shape[0] == gain_grid.shape[1]:
                signal_rows.append(sig[j, :].astype(float))
        row_power.append(float(row.get("pump_power_dbm", power_grid[i] if power_grid is not None else np.nan)))
        row_freq.append(float(row.get("pump_freq_ghz", freq_grid[j] if freq_grid is not None else np.nan)))
        point_index.append(int(row.get("point_index", pos)))
        statuses.append(str(row.get("status", "UNKNOWN")))

    if not rows:
        return {}
    out: dict[str, np.ndarray] = {
        "gain_db": np.vstack(rows),
        "point_index": np.asarray(point_index, dtype=int),
        "pump_power_dbm": np.asarray(row_power, dtype=float),
        "pump_freq_ghz": np.asarray(row_freq, dtype=float),
        "status": np.asarray(statuses, dtype=object),
    }
    if signal_rows:
        sig_stack = np.vstack(signal_rows)
        if np.allclose(sig_stack, sig_stack[0], equal_nan=True):
            out["signal_freq_ghz"] = sig_stack[0]
        else:
            out["signal_freq_ghz"] = sig_stack
    return out


def _normalize_spectrum(
    spectrum: dict[str, np.ndarray],
    points: pd.DataFrame,
) -> dict[str, np.ndarray]:
    out = dict(spectrum)
    flat = _flat_spectrum_from_grid(out, points)
    out.update(flat)

    freq = _first_array(out, ("signal_freq_ghz", "signal_ghz"))
    gain = _first_array(out, ("gain_db", "gain_spectrum_db"))
    if freq is not None:
        out["signal_freq_ghz"] = np.asarray(freq, dtype=float)
    if gain is not None and gain.ndim <= 2:
        out["gain_db"] = np.asarray(gain, dtype=float)
    return out


def load_map_data(run_dir: Path | str) -> MapData:
    """Load map CSV/NPZ files and expose canonical names where possible."""
    root = Path(run_dir)
    points_path = root / "map_points.csv"
    if not points_path.exists():
        raise FileNotFoundError(f"missing map points file: {points_path}")

    points = _normalize_points(pd.read_csv(points_path))
    arrays = _normalize_arrays(_load_npz(root / "map_arrays.npz"))
    spectrum = _normalize_spectrum(_load_npz(root / "map_spectrum.npz"), points)
    return MapData(run_dir=root, points=points, arrays=arrays, spectrum=spectrum)


def spectrum_for_point(data: MapData, point_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return signal frequency and gain trace for one point index."""
    spectrum = data.spectrum
    if "gain_db" not in spectrum or "signal_freq_ghz" not in spectrum:
        raise KeyError("map_spectrum.npz must contain gain_db and signal_freq_ghz data")
    point_ids = np.asarray(
        spectrum.get("point_index", np.arange(np.asarray(spectrum["gain_db"]).shape[0])),
        dtype=int,
    )
    matches = np.flatnonzero(point_ids == int(point_index))
    if matches.size == 0:
        raise KeyError(f"point_index {point_index} not found in spectrum data")
    row = int(matches[0])
    gain = np.asarray(spectrum["gain_db"], dtype=float)
    freq = np.asarray(spectrum["signal_freq_ghz"], dtype=float)
    if gain.ndim != 2:
        raise ValueError("canonical gain_db spectrum must be 2D")
    if freq.ndim == 1:
        return freq, gain[row]
    if freq.ndim == 2:
        return freq[row], gain[row]
    raise ValueError("signal_freq_ghz must be 1D or 2D")
