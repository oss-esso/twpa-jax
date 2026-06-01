"""
twpa.io.measurement
===================

Measurement dataset loaders and normalizers for TWPA calibration workflows.

Supported inputs
----------------
NPZ S-parameter files
    Required:
        frequency_hz

    Optional:
        s                       complex, shape (F, 2, 2)
        s11, s21, s12, s22       complex, shape (F,)
        s11_real/s11_imag, ...   real columns/arrays
        s21_db                   real, shape (F,)

NPZ gain files
    Required:
        signal_frequency_hz
        signal_gain_db

    Optional:
        idler_frequency_hz
        idler_conversion_db
        signal_labels
        idler_labels

CSV S-parameter files
    Required:
        frequency_hz or frequency_GHz/frequency_MHz

    Optional:
        s21_db
        s11_real/s11_imag, s21_real/s21_imag, ...

CSV gain files
    Required:
        signal_frequency_hz or signal_frequency_GHz/signal_frequency_MHz
        signal_gain_db

    Optional:
        idler_frequency_hz or idler_frequency_GHz/idler_frequency_MHz
        idler_conversion_db

The normalized objects can be converted to the calibration data containers in
``twpa.workflows.calibration``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import csv
import json
import numpy as np

import jax
import jax.numpy as jnp


ArrayLike = Any


class MeasurementKind(str, Enum):
    """Supported measurement dataset families."""

    SPARAMETER = "sparameter"
    GAIN = "gain"


class MeasurementFileFormat(str, Enum):
    """Supported measurement file formats."""

    AUTO = "auto"
    NPZ = "npz"
    CSV = "csv"


@dataclass(frozen=True)
class MeasurementLoadConfig:
    """
    Measurement loading configuration.

    Parameters
    ----------
    file_format:
        ``auto``, ``npz``, or ``csv``.
    frequency_key:
        Frequency key for pump-off S-parameter data.
    s_key:
        Full complex S-matrix key.
    s21_db_key:
        S21 magnitude key in dB.
    signal_frequency_key:
        Signal-frequency key for gain data.
    idler_frequency_key:
        Idler-frequency key for gain data.
    signal_gain_db_key:
        Signal gain key in dB.
    idler_conversion_db_key:
        Idler conversion key in dB.
    frequency_unit:
        Unit used when CSV keys do not explicitly contain Hz/GHz/MHz.
    allow_missing_complex_s:
        Permit S-parameter data with only s21_db.
    allow_missing_idler:
        Permit gain data without idler conversion.
    """

    file_format: MeasurementFileFormat = MeasurementFileFormat.AUTO

    frequency_key: str = "frequency_hz"
    s_key: str = "s"
    s21_db_key: str = "s21_db"

    signal_frequency_key: str = "signal_frequency_hz"
    idler_frequency_key: str = "idler_frequency_hz"
    signal_gain_db_key: str = "signal_gain_db"
    idler_conversion_db_key: str = "idler_conversion_db"

    signal_labels_key: str = "signal_labels"
    idler_labels_key: str = "idler_labels"

    frequency_unit: str = "Hz"
    delimiter: str = ","
    allow_missing_complex_s: bool = True
    allow_missing_idler: bool = True
    metadata_json_key: str = "metadata_json"
    name: str = "measurement_load"

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_format", MeasurementFileFormat(self.file_format))
        unit = self.frequency_unit.strip().lower()
        if unit not in {"hz", "khz", "mhz", "ghz"}:
            raise ValueError("frequency_unit must be one of Hz, kHz, MHz, GHz")
        object.__setattr__(self, "frequency_unit", unit)

    def with_updates(self, **kwargs: Any) -> "MeasurementLoadConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_format": self.file_format.value,
            "frequency_key": self.frequency_key,
            "s_key": self.s_key,
            "s21_db_key": self.s21_db_key,
            "signal_frequency_key": self.signal_frequency_key,
            "idler_frequency_key": self.idler_frequency_key,
            "signal_gain_db_key": self.signal_gain_db_key,
            "idler_conversion_db_key": self.idler_conversion_db_key,
            "signal_labels_key": self.signal_labels_key,
            "idler_labels_key": self.idler_labels_key,
            "frequency_unit": self.frequency_unit,
            "delimiter": self.delimiter,
            "allow_missing_complex_s": self.allow_missing_complex_s,
            "allow_missing_idler": self.allow_missing_idler,
            "metadata_json_key": self.metadata_json_key,
            "name": self.name,
        }


@dataclass(frozen=True)
class SParameterMeasurement:
    """
    Normalized pump-off S-parameter measurement.

    Attributes
    ----------
    frequency_hz:
        Frequency grid, shape ``(F,)``.
    s:
        Complex S matrix, shape ``(F, 2, 2)``, optional.
    s21_db:
        S21 magnitude in dB, optional.
    """

    frequency_hz: jax.Array
    s: jax.Array | None = None
    s21_db: jax.Array | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        f = jnp.asarray(self.frequency_hz, dtype=jnp.float64)
        if f.ndim != 1:
            raise ValueError("frequency_hz must be 1D")
        if f.size == 0:
            raise ValueError("frequency_hz may not be empty")

        s = None
        if self.s is not None:
            s = jnp.asarray(self.s, dtype=jnp.complex128)
            if s.shape != (f.shape[0], 2, 2):
                raise ValueError(f"s must have shape {(f.shape[0], 2, 2)}, got {s.shape}")

        s21_db = None
        if self.s21_db is not None:
            s21_db = jnp.asarray(self.s21_db, dtype=jnp.float64)
            if s21_db.shape != f.shape:
                raise ValueError(f"s21_db must have shape {f.shape}, got {s21_db.shape}")

        if s is None and s21_db is None:
            raise ValueError("At least one of s or s21_db must be provided")

        if s21_db is None and s is not None:
            s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(s[:, 1, 0]), 1e-300))

        object.__setattr__(self, "frequency_hz", f)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "s21_db", s21_db)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_frequency(self) -> int:
        return int(self.frequency_hz.shape[0])

    @property
    def s21(self) -> jax.Array | None:
        if self.s is None:
            return None
        return self.s[:, 1, 0]

    def to_calibration_data(self) -> Any:
        """
        Convert to ``twpa.workflows.calibration.SParameterCalibrationData``.
        """
        from twpa.workflows.calibration import SParameterCalibrationData

        return SParameterCalibrationData(
            frequency_hz=self.frequency_hz,
            s=self.s,
            s21_db=self.s21_db,
            metadata={
                **dict(self.metadata or {}),
                "source": "SParameterMeasurement.to_calibration_data",
            },
        )

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "frequency_hz": np.asarray(self.frequency_hz),
            "s21_db": np.asarray(self.s21_db) if self.s21_db is not None else None,
            "metadata_json": json.dumps(self.to_dict(include_arrays=False)),
        }
        if self.s is not None:
            payload["s"] = np.asarray(self.s)

        payload = {k: v for k, v in payload.items() if v is not None}
        np.savez_compressed(path, **payload)
        return path

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        out = {
            "kind": MeasurementKind.SPARAMETER.value,
            "n_frequency": self.n_frequency,
            "frequency_min_hz": float(self.frequency_hz[0]),
            "frequency_max_hz": float(self.frequency_hz[-1]),
            "has_s": self.s is not None,
            "has_s21_db": self.s21_db is not None,
            "metadata": dict(self.metadata or {}),
        }

        if self.s21_db is not None:
            out.update(
                {
                    "s21_db_min": float(jnp.nanmin(self.s21_db)),
                    "s21_db_max": float(jnp.nanmax(self.s21_db)),
                }
            )

        if include_arrays:
            out["frequency_hz"] = np.asarray(self.frequency_hz).tolist()
            if self.s21_db is not None:
                out["s21_db"] = np.asarray(self.s21_db).tolist()
            if self.s is not None:
                out["s_shape"] = tuple(int(v) for v in self.s.shape)

        return out


@dataclass(frozen=True)
class GainMeasurement:
    """
    Normalized pump-on gain measurement.
    """

    signal_frequency_hz: jax.Array
    signal_gain_db: jax.Array
    idler_frequency_hz: jax.Array | None = None
    idler_conversion_db: jax.Array | None = None
    signal_labels: tuple[str, ...] | None = None
    idler_labels: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        fs = jnp.asarray(self.signal_frequency_hz, dtype=jnp.float64)
        gain = jnp.asarray(self.signal_gain_db, dtype=jnp.float64)

        if fs.ndim != 1:
            raise ValueError("signal_frequency_hz must be 1D")
        if fs.size == 0:
            raise ValueError("signal_frequency_hz may not be empty")
        if gain.shape != fs.shape:
            raise ValueError("signal_gain_db must have same shape as signal_frequency_hz")

        fi = None
        if self.idler_frequency_hz is not None:
            fi = jnp.asarray(self.idler_frequency_hz, dtype=jnp.float64)
            if fi.shape != fs.shape:
                raise ValueError("idler_frequency_hz must have same shape as signal_frequency_hz")

        idler = None
        if self.idler_conversion_db is not None:
            idler = jnp.asarray(self.idler_conversion_db, dtype=jnp.float64)
            if idler.shape != fs.shape:
                raise ValueError("idler_conversion_db must have same shape as signal_frequency_hz")

        signal_labels = self.signal_labels
        if signal_labels is None:
            signal_labels = tuple(f"signal_{i}" for i in range(int(fs.shape[0])))
        else:
            signal_labels = tuple(str(x) for x in signal_labels)
            if len(signal_labels) != fs.shape[0]:
                raise ValueError("signal_labels length mismatch")

        idler_labels = self.idler_labels
        if idler_labels is None:
            idler_labels = tuple(f"idler_{i}" for i in range(int(fs.shape[0])))
        else:
            idler_labels = tuple(str(x) for x in idler_labels)
            if len(idler_labels) != fs.shape[0]:
                raise ValueError("idler_labels length mismatch")

        object.__setattr__(self, "signal_frequency_hz", fs)
        object.__setattr__(self, "signal_gain_db", gain)
        object.__setattr__(self, "idler_frequency_hz", fi)
        object.__setattr__(self, "idler_conversion_db", idler)
        object.__setattr__(self, "signal_labels", signal_labels)
        object.__setattr__(self, "idler_labels", idler_labels)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_points(self) -> int:
        return int(self.signal_frequency_hz.shape[0])

    def to_calibration_data(self) -> Any:
        """
        Convert to ``twpa.workflows.calibration.GainCalibrationData``.
        """
        from twpa.workflows.calibration import GainCalibrationData

        return GainCalibrationData(
            signal_labels=self.signal_labels,
            signal_gain_db=self.signal_gain_db,
            idler_labels=self.idler_labels,
            idler_conversion_db=self.idler_conversion_db,
            metadata={
                **dict(self.metadata or {}),
                "signal_frequency_hz": self.signal_frequency_hz,
                "idler_frequency_hz": self.idler_frequency_hz,
                "source": "GainMeasurement.to_calibration_data",
            },
        )

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "signal_frequency_hz": np.asarray(self.signal_frequency_hz),
            "signal_gain_db": np.asarray(self.signal_gain_db),
            "signal_labels": np.asarray(self.signal_labels),
            "idler_labels": np.asarray(self.idler_labels),
            "metadata_json": json.dumps(self.to_dict(include_arrays=False)),
        }
        if self.idler_frequency_hz is not None:
            payload["idler_frequency_hz"] = np.asarray(self.idler_frequency_hz)
        if self.idler_conversion_db is not None:
            payload["idler_conversion_db"] = np.asarray(self.idler_conversion_db)

        np.savez_compressed(path, **payload)
        return path

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        out = {
            "kind": MeasurementKind.GAIN.value,
            "n_points": self.n_points,
            "signal_frequency_min_hz": float(self.signal_frequency_hz[0]),
            "signal_frequency_max_hz": float(self.signal_frequency_hz[-1]),
            "signal_gain_db_min": float(jnp.nanmin(self.signal_gain_db)),
            "signal_gain_db_max": float(jnp.nanmax(self.signal_gain_db)),
            "has_idler_frequency": self.idler_frequency_hz is not None,
            "has_idler_conversion": self.idler_conversion_db is not None,
            "metadata": dict(self.metadata or {}),
        }

        if self.idler_conversion_db is not None:
            out["idler_conversion_db_min"] = float(jnp.nanmin(self.idler_conversion_db))
            out["idler_conversion_db_max"] = float(jnp.nanmax(self.idler_conversion_db))

        if include_arrays:
            out["signal_frequency_hz"] = np.asarray(self.signal_frequency_hz).tolist()
            out["signal_gain_db"] = np.asarray(self.signal_gain_db).tolist()
            if self.idler_frequency_hz is not None:
                out["idler_frequency_hz"] = np.asarray(self.idler_frequency_hz).tolist()
            if self.idler_conversion_db is not None:
                out["idler_conversion_db"] = np.asarray(self.idler_conversion_db).tolist()
            out["signal_labels"] = list(self.signal_labels)
            out["idler_labels"] = list(self.idler_labels)

        return out


def _resolve_format(path: Path, config: MeasurementLoadConfig) -> MeasurementFileFormat:
    if config.file_format != MeasurementFileFormat.AUTO:
        return config.file_format

    suffix = path.suffix.lower()
    if suffix == ".npz":
        return MeasurementFileFormat.NPZ
    if suffix in {".csv", ".txt"}:
        return MeasurementFileFormat.CSV

    raise ValueError(f"Could not infer measurement file format from suffix {suffix!r}")


def _frequency_scale_from_key(key: str, default_unit: str) -> float:
    lower = key.lower()
    if lower.endswith("_ghz") or lower == "frequency_ghz":
        return 1e9
    if lower.endswith("_mhz") or lower == "frequency_mhz":
        return 1e6
    if lower.endswith("_khz") or lower == "frequency_khz":
        return 1e3
    if lower.endswith("_hz") or lower == "frequency_hz":
        return 1.0

    unit = default_unit.lower()
    return {
        "hz": 1.0,
        "khz": 1e3,
        "mhz": 1e6,
        "ghz": 1e9,
    }[unit]


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in mapping:
            return key
    return None


def _load_metadata_from_npz(npz: Any, config: MeasurementLoadConfig) -> dict[str, Any]:
    if config.metadata_json_key not in npz:
        return {}
    try:
        raw = npz[config.metadata_json_key]
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))
    except Exception as exc:
        return {"metadata_parse_error": str(exc)}


def _labels_from_npz(npz: Any, key: str, n: int, prefix: str) -> tuple[str, ...]:
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


def _complex_array_from_real_imag(mapping: Mapping[str, Any], base: str) -> jax.Array | None:
    real_key = f"{base}_real"
    imag_key = f"{base}_imag"
    if real_key in mapping and imag_key in mapping:
        return jnp.asarray(mapping[real_key], dtype=jnp.float64) + 1j * jnp.asarray(mapping[imag_key], dtype=jnp.float64)
    return None


def _s_matrix_from_components(mapping: Mapping[str, Any]) -> jax.Array | None:
    components = {}
    for name in ["s11", "s21", "s12", "s22"]:
        if name in mapping:
            components[name] = jnp.asarray(mapping[name], dtype=jnp.complex128)
        else:
            comp = _complex_array_from_real_imag(mapping, name)
            if comp is not None:
                components[name] = comp

    if not components:
        return None

    if "s21" not in components:
        raise ValueError("Partial S-parameter components require at least s21")

    n = int(components["s21"].shape[0])
    s = jnp.zeros((n, 2, 2), dtype=jnp.complex128)

    if "s11" in components:
        s = s.at[:, 0, 0].set(components["s11"])
    if "s21" in components:
        s = s.at[:, 1, 0].set(components["s21"])
    if "s12" in components:
        s = s.at[:, 0, 1].set(components["s12"])
    if "s22" in components:
        s = s.at[:, 1, 1].set(components["s22"])

    return s


def load_sparameter_measurement_npz(
    path: str | Path,
    config: MeasurementLoadConfig | None = None,
) -> SParameterMeasurement:
    """
    Load S-parameter measurement from NPZ.
    """
    cfg = config or MeasurementLoadConfig(file_format=MeasurementFileFormat.NPZ)
    p = Path(path)
    npz = np.load(p, allow_pickle=True)

    freq_key = _first_present(
        npz,
        [
            cfg.frequency_key,
            "frequency_hz",
            "frequency_GHz",
            "frequency_MHz",
            "f_hz",
            "f_GHz",
            "freq_hz",
            "freq_GHz",
        ],
    )
    if freq_key is None:
        raise ValueError(f"{p}: no frequency key found")

    frequency_hz = jnp.asarray(npz[freq_key], dtype=jnp.float64) * _frequency_scale_from_key(
        freq_key,
        cfg.frequency_unit,
    )

    s = None
    if cfg.s_key in npz:
        s = jnp.asarray(npz[cfg.s_key], dtype=jnp.complex128)
    else:
        s = _s_matrix_from_components(npz)

    s21_db = None
    if cfg.s21_db_key in npz:
        s21_db = jnp.asarray(npz[cfg.s21_db_key], dtype=jnp.float64)
    elif "S21_dB" in npz:
        s21_db = jnp.asarray(npz["S21_dB"], dtype=jnp.float64)

    if s is None and s21_db is None and not cfg.allow_missing_complex_s:
        raise ValueError(f"{p}: missing complex S matrix/components and s21_db")

    metadata = {
        "source_path": str(p),
        "format": "npz",
        "frequency_key": freq_key,
        "load_config": cfg.to_dict(),
        **_load_metadata_from_npz(npz, cfg),
    }

    return SParameterMeasurement(
        frequency_hz=frequency_hz,
        s=s,
        s21_db=s21_db,
        metadata=metadata,
    )


def _read_csv_columns(path: Path, *, delimiter: str = ",") -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: CSV is empty")

    columns: dict[str, list[Any]] = {name: [] for name in reader.fieldnames}

    for row in rows:
        for name in reader.fieldnames:
            value = row.get(name, "")
            if value is None or str(value).strip() == "":
                columns[name].append(np.nan)
            else:
                try:
                    columns[name].append(float(value))
                except ValueError:
                    columns[name].append(str(value))

    out: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        try:
            out[name] = np.asarray(values, dtype=float)
        except Exception:
            out[name] = np.asarray(values, dtype=object)
    return out


def load_sparameter_measurement_csv(
    path: str | Path,
    config: MeasurementLoadConfig | None = None,
) -> SParameterMeasurement:
    """
    Load S-parameter measurement from CSV.
    """
    cfg = config or MeasurementLoadConfig(file_format=MeasurementFileFormat.CSV)
    p = Path(path)
    cols = _read_csv_columns(p, delimiter=cfg.delimiter)

    freq_key = _first_present(
        cols,
        [
            cfg.frequency_key,
            "frequency_hz",
            "frequency_GHz",
            "frequency_MHz",
            "f_hz",
            "f_GHz",
            "freq_hz",
            "freq_GHz",
            "frequency",
            "freq",
        ],
    )
    if freq_key is None:
        raise ValueError(f"{p}: no frequency column found")

    frequency_hz = jnp.asarray(cols[freq_key], dtype=jnp.float64) * _frequency_scale_from_key(
        freq_key,
        cfg.frequency_unit,
    )

    s = _s_matrix_from_components(cols)

    s21_db = None
    s21_db_key = _first_present(cols, [cfg.s21_db_key, "S21_dB", "s21_dB", "s21_db"])
    if s21_db_key is not None:
        s21_db = jnp.asarray(cols[s21_db_key], dtype=jnp.float64)

    if s is None and s21_db is None and not cfg.allow_missing_complex_s:
        raise ValueError(f"{p}: missing S-parameter columns")

    return SParameterMeasurement(
        frequency_hz=frequency_hz,
        s=s,
        s21_db=s21_db,
        metadata={
            "source_path": str(p),
            "format": "csv",
            "frequency_key": freq_key,
            "columns": list(cols.keys()),
            "load_config": cfg.to_dict(),
        },
    )


def load_sparameter_measurement(
    path: str | Path,
    config: MeasurementLoadConfig | None = None,
) -> SParameterMeasurement:
    """
    Load S-parameter measurement from NPZ or CSV.
    """
    cfg = config or MeasurementLoadConfig()
    p = Path(path)
    fmt = _resolve_format(p, cfg)

    if fmt == MeasurementFileFormat.NPZ:
        return load_sparameter_measurement_npz(p, cfg.with_updates(file_format=MeasurementFileFormat.NPZ))
    if fmt == MeasurementFileFormat.CSV:
        return load_sparameter_measurement_csv(p, cfg.with_updates(file_format=MeasurementFileFormat.CSV))

    raise ValueError(f"Unsupported measurement format {fmt}")


def load_gain_measurement_npz(
    path: str | Path,
    config: MeasurementLoadConfig | None = None,
) -> GainMeasurement:
    """
    Load gain measurement from NPZ.
    """
    cfg = config or MeasurementLoadConfig(file_format=MeasurementFileFormat.NPZ)
    p = Path(path)
    npz = np.load(p, allow_pickle=True)

    sig_key = _first_present(
        npz,
        [
            cfg.signal_frequency_key,
            "signal_frequency_hz",
            "signal_frequency_GHz",
            "signal_frequency_MHz",
            "frequency_hz",
            "frequency_GHz",
        ],
    )
    if sig_key is None:
        raise ValueError(f"{p}: no signal frequency key found")

    gain_key = _first_present(npz, [cfg.signal_gain_db_key, "gain_db", "signal_gain_dB", "gain_dB"])
    if gain_key is None:
        raise ValueError(f"{p}: no signal gain dB key found")

    signal_frequency_hz = jnp.asarray(npz[sig_key], dtype=jnp.float64) * _frequency_scale_from_key(
        sig_key,
        cfg.frequency_unit,
    )
    signal_gain_db = jnp.asarray(npz[gain_key], dtype=jnp.float64)

    idler_frequency_hz = None
    idler_key = _first_present(
        npz,
        [
            cfg.idler_frequency_key,
            "idler_frequency_hz",
            "idler_frequency_GHz",
            "idler_frequency_MHz",
        ],
    )
    if idler_key is not None:
        idler_frequency_hz = jnp.asarray(npz[idler_key], dtype=jnp.float64) * _frequency_scale_from_key(
            idler_key,
            cfg.frequency_unit,
        )

    idler_conversion_db = None
    idler_conv_key = _first_present(
        npz,
        [cfg.idler_conversion_db_key, "idler_conversion_dB", "conversion_db", "conversion_dB"],
    )
    if idler_conv_key is not None:
        idler_conversion_db = jnp.asarray(npz[idler_conv_key], dtype=jnp.float64)
    elif not cfg.allow_missing_idler:
        raise ValueError(f"{p}: missing idler conversion key")

    n = int(signal_frequency_hz.shape[0])
    signal_labels = _labels_from_npz(npz, cfg.signal_labels_key, n, "signal")
    idler_labels = _labels_from_npz(npz, cfg.idler_labels_key, n, "idler")

    return GainMeasurement(
        signal_frequency_hz=signal_frequency_hz,
        signal_gain_db=signal_gain_db,
        idler_frequency_hz=idler_frequency_hz,
        idler_conversion_db=idler_conversion_db,
        signal_labels=signal_labels,
        idler_labels=idler_labels,
        metadata={
            "source_path": str(p),
            "format": "npz",
            "signal_frequency_key": sig_key,
            "signal_gain_db_key": gain_key,
            "load_config": cfg.to_dict(),
            **_load_metadata_from_npz(npz, cfg),
        },
    )


def load_gain_measurement_csv(
    path: str | Path,
    config: MeasurementLoadConfig | None = None,
) -> GainMeasurement:
    """
    Load gain measurement from CSV.
    """
    cfg = config or MeasurementLoadConfig(file_format=MeasurementFileFormat.CSV)
    p = Path(path)
    cols = _read_csv_columns(p, delimiter=cfg.delimiter)

    sig_key = _first_present(
        cols,
        [
            cfg.signal_frequency_key,
            "signal_frequency_hz",
            "signal_frequency_GHz",
            "signal_frequency_MHz",
            "frequency_hz",
            "frequency_GHz",
            "frequency",
            "freq",
        ],
    )
    if sig_key is None:
        raise ValueError(f"{p}: no signal frequency column found")

    gain_key = _first_present(cols, [cfg.signal_gain_db_key, "gain_db", "signal_gain_dB", "gain_dB"])
    if gain_key is None:
        raise ValueError(f"{p}: no signal gain dB column found")

    signal_frequency_hz = jnp.asarray(cols[sig_key], dtype=jnp.float64) * _frequency_scale_from_key(
        sig_key,
        cfg.frequency_unit,
    )
    signal_gain_db = jnp.asarray(cols[gain_key], dtype=jnp.float64)

    idler_frequency_hz = None
    idler_key = _first_present(
        cols,
        [
            cfg.idler_frequency_key,
            "idler_frequency_hz",
            "idler_frequency_GHz",
            "idler_frequency_MHz",
        ],
    )
    if idler_key is not None:
        idler_frequency_hz = jnp.asarray(cols[idler_key], dtype=jnp.float64) * _frequency_scale_from_key(
            idler_key,
            cfg.frequency_unit,
        )

    idler_conversion_db = None
    idler_conv_key = _first_present(
        cols,
        [cfg.idler_conversion_db_key, "idler_conversion_dB", "conversion_db", "conversion_dB"],
    )
    if idler_conv_key is not None:
        idler_conversion_db = jnp.asarray(cols[idler_conv_key], dtype=jnp.float64)
    elif not cfg.allow_missing_idler:
        raise ValueError(f"{p}: missing idler conversion column")

    n = int(signal_frequency_hz.shape[0])
    signal_labels = tuple(f"signal_{i}" for i in range(n))
    idler_labels = tuple(f"idler_{i}" for i in range(n))

    return GainMeasurement(
        signal_frequency_hz=signal_frequency_hz,
        signal_gain_db=signal_gain_db,
        idler_frequency_hz=idler_frequency_hz,
        idler_conversion_db=idler_conversion_db,
        signal_labels=signal_labels,
        idler_labels=idler_labels,
        metadata={
            "source_path": str(p),
            "format": "csv",
            "signal_frequency_key": sig_key,
            "signal_gain_db_key": gain_key,
            "columns": list(cols.keys()),
            "load_config": cfg.to_dict(),
        },
    )


def load_gain_measurement(
    path: str | Path,
    config: MeasurementLoadConfig | None = None,
) -> GainMeasurement:
    """
    Load gain measurement from NPZ or CSV.
    """
    cfg = config or MeasurementLoadConfig()
    p = Path(path)
    fmt = _resolve_format(p, cfg)

    if fmt == MeasurementFileFormat.NPZ:
        return load_gain_measurement_npz(p, cfg.with_updates(file_format=MeasurementFileFormat.NPZ))
    if fmt == MeasurementFileFormat.CSV:
        return load_gain_measurement_csv(p, cfg.with_updates(file_format=MeasurementFileFormat.CSV))

    raise ValueError(f"Unsupported measurement format {fmt}")


def measurement_summary_markdown(
    measurement: SParameterMeasurement | GainMeasurement,
) -> str:
    """
    Markdown summary for a measurement object.
    """
    if isinstance(measurement, SParameterMeasurement):
        d = measurement.to_dict(include_arrays=False)
        lines = [
            "# S-parameter measurement",
            "",
            f"- points: `{d['n_frequency']}`",
            f"- frequency range: `{d['frequency_min_hz'] / 1e9:.6g}`–`{d['frequency_max_hz'] / 1e9:.6g} GHz`",
            f"- has complex S: `{d['has_s']}`",
            f"- has S21 dB: `{d['has_s21_db']}`",
        ]
        if "s21_db_min" in d:
            lines.append(f"- S21 dB range: `{d['s21_db_min']:.6g}` to `{d['s21_db_max']:.6g}`")
        return "\n".join(lines)

    d = measurement.to_dict(include_arrays=False)
    lines = [
        "# Gain measurement",
        "",
        f"- points: `{d['n_points']}`",
        f"- signal frequency range: `{d['signal_frequency_min_hz'] / 1e9:.6g}`–`{d['signal_frequency_max_hz'] / 1e9:.6g} GHz`",
        f"- signal gain dB range: `{d['signal_gain_db_min']:.6g}` to `{d['signal_gain_db_max']:.6g}`",
        f"- has idler frequency: `{d['has_idler_frequency']}`",
        f"- has idler conversion: `{d['has_idler_conversion']}`",
    ]
    if "idler_conversion_db_min" in d:
        lines.append(
            f"- idler conversion dB range: `{d['idler_conversion_db_min']:.6g}` "
            f"to `{d['idler_conversion_db_max']:.6g}`"
        )
    return "\n".join(lines)


__all__ = [
    "ArrayLike",
    "MeasurementKind",
    "MeasurementFileFormat",
    "MeasurementLoadConfig",
    "SParameterMeasurement",
    "GainMeasurement",
    "load_sparameter_measurement_npz",
    "load_sparameter_measurement_csv",
    "load_sparameter_measurement",
    "load_gain_measurement_npz",
    "load_gain_measurement_csv",
    "load_gain_measurement",
    "measurement_summary_markdown",
]