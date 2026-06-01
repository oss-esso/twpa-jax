"""
twpa.inference.synthetic
========================

Synthetic measurement generation for TWPA inference and recovery studies.

This module creates measurement-like datasets from the simulator:

    - pump-off S-parameters,
    - pump-on gain curves,
    - combined synthetic calibration datasets.

The output objects are intentionally close to the calibration data expected by
``twpa.workflows.calibration`` while also preserving clean/noisy arrays,
metadata, true parameters, and reproducible noise settings.

Typical use
-----------
Generate pump-off synthetic S-parameters:

    dataset = generate_synthetic_sparameters(
        layout,
        frequency_hz=jnp.linspace(1e9, 12e9, 301),
        noise=SyntheticNoiseConfig(s_db_std=0.05, seed=123),
    )
    dataset.save_npz("synthetic_sparams.npz")

Generate pump-on synthetic gain:

    dataset = generate_synthetic_gain_data(
        layout,
        nonlinear_params,
        pump_drive=drive,
        signal_frequency_hz=jnp.linspace(4e9, 7e9, 31),
        pump_config=pump_config,
        noise=SyntheticNoiseConfig(gain_db_std=0.1, seed=456),
    )
    dataset.save_npz("synthetic_gain.npz")
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import inspect
import json
import numpy as np

import jax
import jax.numpy as jnp

from twpa.core import frequency_plan as frequency_plan_module
from twpa.core.layout import LineLayout, make_layout_from_arrays
from twpa.core.params import NonlinearParams
from twpa.linear.cascade import CascadeConfig, LinearScanResult, run_linear_scan
from twpa.linear.cells import CellModelConfig
from twpa.nonlinear.gain import (
    GainSolveConfig,
    GainSweepConfig,
    GainSweepResult,
    solve_gain_sweep_from_pump,
)
from twpa.nonlinear.pump_hb_ladder import (
    PumpDriveConfig,
    PumpHBLadderConfig,
    PumpHBLadderResult,
    solve_pump_hb_ladder,
)


ArrayLike = Any
TargetPlanFactory = Callable[[PumpHBLadderResult], Any]
SweepConfigFactory = Callable[[Any], GainSweepConfig]


class SyntheticMeasurementKind(str, Enum):
    """Supported synthetic measurement families."""

    SPARAMETER = "sparameter"
    GAIN = "gain"
    COMBINED = "combined"


@dataclass(frozen=True)
class SyntheticNoiseConfig:
    """
    Noise model for synthetic datasets.

    Parameters
    ----------
    s_complex_std_abs:
        Additive complex standard deviation for raw complex S-parameters.
        The noise is circular complex with real/imag std
        ``s_complex_std_abs / sqrt(2)``.
    s_db_std:
        Additive Gaussian standard deviation for S-parameter magnitudes in dB.
    gain_db_std:
        Additive Gaussian standard deviation for signal gain in dB.
    idler_db_std:
        Additive Gaussian standard deviation for idler conversion in dB.
    relative_complex_std:
        Optional multiplicative complex noise level applied as
        ``S * (1 + eta)``.
    seed:
        RNG seed.
    """

    s_complex_std_abs: float = 0.0
    s_db_std: float = 0.0
    gain_db_std: float = 0.0
    idler_db_std: float = 0.0
    relative_complex_std: float = 0.0
    seed: int | None = None
    name: str = "synthetic_noise"

    def __post_init__(self) -> None:
        for field_name in [
            "s_complex_std_abs",
            "s_db_std",
            "gain_db_std",
            "idler_db_std",
            "relative_complex_std",
        ]:
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)

        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))

    @property
    def is_noiseless(self) -> bool:
        return (
            self.s_complex_std_abs == 0.0
            and self.s_db_std == 0.0
            and self.gain_db_std == 0.0
            and self.idler_db_std == 0.0
            and self.relative_complex_std == 0.0
        )

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def with_updates(self, **kwargs: Any) -> "SyntheticNoiseConfig":
        return replace(self, **kwargs)

    def add_complex_noise(self, array: ArrayLike, *, rng: np.random.Generator | None = None) -> jax.Array:
        arr = jnp.asarray(array, dtype=jnp.complex128)
        if rng is None:
            rng = self.rng()

        out = np.asarray(arr).astype(np.complex128, copy=True)

        if self.relative_complex_std > 0.0:
            scale = self.relative_complex_std / np.sqrt(2.0)
            eta = scale * (
                rng.standard_normal(out.shape)
                + 1j * rng.standard_normal(out.shape)
            )
            out = out * (1.0 + eta)

        if self.s_complex_std_abs > 0.0:
            scale = self.s_complex_std_abs / np.sqrt(2.0)
            eta = scale * (
                rng.standard_normal(out.shape)
                + 1j * rng.standard_normal(out.shape)
            )
            out = out + eta

        return jnp.asarray(out, dtype=jnp.complex128)

    def add_s_db_noise(self, array_db: ArrayLike, *, rng: np.random.Generator | None = None) -> jax.Array:
        arr = np.asarray(array_db, dtype=float)
        if rng is None:
            rng = self.rng()
        if self.s_db_std <= 0.0:
            return jnp.asarray(arr, dtype=jnp.float64)
        return jnp.asarray(arr + rng.normal(0.0, self.s_db_std, size=arr.shape), dtype=jnp.float64)

    def add_gain_db_noise(self, array_db: ArrayLike, *, rng: np.random.Generator | None = None) -> jax.Array:
        arr = np.asarray(array_db, dtype=float)
        if rng is None:
            rng = self.rng()
        if self.gain_db_std <= 0.0:
            return jnp.asarray(arr, dtype=jnp.float64)
        return jnp.asarray(arr + rng.normal(0.0, self.gain_db_std, size=arr.shape), dtype=jnp.float64)

    def add_idler_db_noise(self, array_db: ArrayLike, *, rng: np.random.Generator | None = None) -> jax.Array | None:
        if array_db is None:
            return None
        arr = np.asarray(array_db, dtype=float)
        if rng is None:
            rng = self.rng()
        if self.idler_db_std <= 0.0:
            return jnp.asarray(arr, dtype=jnp.float64)
        return jnp.asarray(arr + rng.normal(0.0, self.idler_db_std, size=arr.shape), dtype=jnp.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "s_complex_std_abs": self.s_complex_std_abs,
            "s_db_std": self.s_db_std,
            "gain_db_std": self.gain_db_std,
            "idler_db_std": self.idler_db_std,
            "relative_complex_std": self.relative_complex_std,
            "seed": self.seed,
            "is_noiseless": self.is_noiseless,
            "name": self.name,
        }


@dataclass(frozen=True)
class SyntheticSParameterDataset:
    """
    Synthetic pump-off S-parameter dataset.

    Arrays
    ------
    frequency_hz:
        Frequency grid, shape ``(F,)``.
    s_clean:
        Clean complex S matrix, shape ``(F, 2, 2)``.
    s_noisy:
        Noisy complex S matrix, shape ``(F, 2, 2)``.
    s21_db_clean:
        Clean S21 magnitude in dB, shape ``(F,)``.
    s21_db_noisy:
        Noisy S21 magnitude in dB, shape ``(F,)``.
    """

    frequency_hz: jax.Array
    s_clean: jax.Array
    s_noisy: jax.Array
    s21_db_clean: jax.Array
    s21_db_noisy: jax.Array
    noise: SyntheticNoiseConfig
    true_parameters: Mapping[str, float] | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        frequency_hz = jnp.asarray(self.frequency_hz, dtype=jnp.float64)
        s_clean = jnp.asarray(self.s_clean, dtype=jnp.complex128)
        s_noisy = jnp.asarray(self.s_noisy, dtype=jnp.complex128)
        s21_db_clean = jnp.asarray(self.s21_db_clean, dtype=jnp.float64)
        s21_db_noisy = jnp.asarray(self.s21_db_noisy, dtype=jnp.float64)

        if frequency_hz.ndim != 1:
            raise ValueError("frequency_hz must be 1D")
        if s_clean.shape != (frequency_hz.shape[0], 2, 2):
            raise ValueError("s_clean must have shape (F, 2, 2)")
        if s_noisy.shape != s_clean.shape:
            raise ValueError("s_noisy must have same shape as s_clean")
        if s21_db_clean.shape != frequency_hz.shape:
            raise ValueError("s21_db_clean must have shape (F,)")
        if s21_db_noisy.shape != frequency_hz.shape:
            raise ValueError("s21_db_noisy must have shape (F,)")

        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "s_clean", s_clean)
        object.__setattr__(self, "s_noisy", s_noisy)
        object.__setattr__(self, "s21_db_clean", s21_db_clean)
        object.__setattr__(self, "s21_db_noisy", s21_db_noisy)
        object.__setattr__(
            self,
            "true_parameters",
            {str(k): float(v) for k, v in dict(self.true_parameters or {}).items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_frequency(self) -> int:
        return int(self.frequency_hz.shape[0])

    def to_calibration_data(self) -> Any:
        """
        Convert to ``twpa.workflows.calibration.SParameterCalibrationData``.
        """
        from twpa.workflows.calibration import SParameterCalibrationData

        return SParameterCalibrationData(
            frequency_hz=self.frequency_hz,
            s=self.s_noisy,
            s21_db=self.s21_db_noisy,
            metadata={
                **dict(self.metadata or {}),
                "source": "SyntheticSParameterDataset.to_calibration_data",
                "true_parameters": dict(self.true_parameters or {}),
            },
        )

    def save_npz(self, path: str | Path, *, include_clean: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "frequency_hz": np.asarray(self.frequency_hz),
            "s": np.asarray(self.s_noisy),
            "s21_db": np.asarray(self.s21_db_noisy),
            "metadata_json": json.dumps(self.to_dict(include_arrays=False)),
        }

        if include_clean:
            payload.update(
                {
                    "s_clean": np.asarray(self.s_clean),
                    "s21_db_clean": np.asarray(self.s21_db_clean),
                }
            )

        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "SyntheticSParameterDataset":
        npz = np.load(path, allow_pickle=True)
        metadata = {}
        noise = SyntheticNoiseConfig()

        if "metadata_json" in npz:
            metadata_payload = json.loads(str(npz["metadata_json"].item()))
            metadata = metadata_payload.get("metadata", {})
            noise = SyntheticNoiseConfig(**{
                k: v
                for k, v in metadata_payload.get("noise", {}).items()
                if k in SyntheticNoiseConfig.__dataclass_fields__
            })

        s_noisy = jnp.asarray(npz["s"], dtype=jnp.complex128)
        s_clean = jnp.asarray(npz["s_clean"], dtype=jnp.complex128) if "s_clean" in npz else s_noisy

        s21_noisy = jnp.asarray(npz["s21_db"], dtype=jnp.float64)
        s21_clean = jnp.asarray(npz["s21_db_clean"], dtype=jnp.float64) if "s21_db_clean" in npz else s21_noisy

        return cls(
            frequency_hz=jnp.asarray(npz["frequency_hz"], dtype=jnp.float64),
            s_clean=s_clean,
            s_noisy=s_noisy,
            s21_db_clean=s21_clean,
            s21_db_noisy=s21_noisy,
            noise=noise,
            true_parameters={},
            metadata={
                **metadata,
                "loaded_from": str(path),
            },
        )

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        out = {
            "kind": SyntheticMeasurementKind.SPARAMETER.value,
            "n_frequency": self.n_frequency,
            "frequency_min_hz": float(self.frequency_hz[0]),
            "frequency_max_hz": float(self.frequency_hz[-1]),
            "s21_db_clean_min": float(jnp.nanmin(self.s21_db_clean)),
            "s21_db_clean_max": float(jnp.nanmax(self.s21_db_clean)),
            "s21_db_noisy_min": float(jnp.nanmin(self.s21_db_noisy)),
            "s21_db_noisy_max": float(jnp.nanmax(self.s21_db_noisy)),
            "noise": self.noise.to_dict(),
            "true_parameters": dict(self.true_parameters or {}),
            "metadata": dict(self.metadata or {}),
        }

        if include_arrays:
            out["frequency_hz"] = np.asarray(self.frequency_hz).tolist()
            out["s21_db_clean"] = np.asarray(self.s21_db_clean).tolist()
            out["s21_db_noisy"] = np.asarray(self.s21_db_noisy).tolist()

        return out


@dataclass(frozen=True)
class SyntheticGainDataset:
    """
    Synthetic pump-on gain dataset.

    Arrays
    ------
    signal_frequency_hz:
        Signal frequency grid.
    idler_frequency_hz:
        Idler frequency grid.
    signal_gain_db_clean:
        Clean signal gain.
    signal_gain_db_noisy:
        Noisy signal gain.
    idler_conversion_db_clean:
        Clean idler conversion, optional.
    idler_conversion_db_noisy:
        Noisy idler conversion, optional.
    """

    signal_frequency_hz: jax.Array
    idler_frequency_hz: jax.Array
    signal_gain_db_clean: jax.Array
    signal_gain_db_noisy: jax.Array
    idler_conversion_db_clean: jax.Array | None
    idler_conversion_db_noisy: jax.Array | None
    signal_labels: tuple[str, ...]
    idler_labels: tuple[str, ...]
    noise: SyntheticNoiseConfig
    true_parameters: Mapping[str, float] | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        fs = jnp.asarray(self.signal_frequency_hz, dtype=jnp.float64)
        fi = jnp.asarray(self.idler_frequency_hz, dtype=jnp.float64)
        gain_clean = jnp.asarray(self.signal_gain_db_clean, dtype=jnp.float64)
        gain_noisy = jnp.asarray(self.signal_gain_db_noisy, dtype=jnp.float64)

        if fs.ndim != 1:
            raise ValueError("signal_frequency_hz must be 1D")
        if fi.shape != fs.shape:
            raise ValueError("idler_frequency_hz must have same shape as signal_frequency_hz")
        if gain_clean.shape != fs.shape:
            raise ValueError("signal_gain_db_clean must have same shape as signal_frequency_hz")
        if gain_noisy.shape != fs.shape:
            raise ValueError("signal_gain_db_noisy must have same shape as signal_frequency_hz")

        idler_clean = None
        idler_noisy = None
        if self.idler_conversion_db_clean is not None:
            idler_clean = jnp.asarray(self.idler_conversion_db_clean, dtype=jnp.float64)
            if idler_clean.shape != fs.shape:
                raise ValueError("idler_conversion_db_clean must have same shape as signal_frequency_hz")
        if self.idler_conversion_db_noisy is not None:
            idler_noisy = jnp.asarray(self.idler_conversion_db_noisy, dtype=jnp.float64)
            if idler_noisy.shape != fs.shape:
                raise ValueError("idler_conversion_db_noisy must have same shape as signal_frequency_hz")

        signal_labels = tuple(str(x) for x in self.signal_labels)
        idler_labels = tuple(str(x) for x in self.idler_labels)
        if len(signal_labels) != fs.shape[0]:
            raise ValueError("signal_labels length must match signal_frequency_hz")
        if len(idler_labels) != fs.shape[0]:
            raise ValueError("idler_labels length must match signal_frequency_hz")

        object.__setattr__(self, "signal_frequency_hz", fs)
        object.__setattr__(self, "idler_frequency_hz", fi)
        object.__setattr__(self, "signal_gain_db_clean", gain_clean)
        object.__setattr__(self, "signal_gain_db_noisy", gain_noisy)
        object.__setattr__(self, "idler_conversion_db_clean", idler_clean)
        object.__setattr__(self, "idler_conversion_db_noisy", idler_noisy)
        object.__setattr__(self, "signal_labels", signal_labels)
        object.__setattr__(self, "idler_labels", idler_labels)
        object.__setattr__(
            self,
            "true_parameters",
            {str(k): float(v) for k, v in dict(self.true_parameters or {}).items()},
        )
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
            signal_gain_db=self.signal_gain_db_noisy,
            idler_labels=self.idler_labels,
            idler_conversion_db=self.idler_conversion_db_noisy,
            metadata={
                **dict(self.metadata or {}),
                "signal_frequency_hz": self.signal_frequency_hz,
                "idler_frequency_hz": self.idler_frequency_hz,
                "source": "SyntheticGainDataset.to_calibration_data",
                "true_parameters": dict(self.true_parameters or {}),
            },
        )

    def save_npz(self, path: str | Path, *, include_clean: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "signal_frequency_hz": np.asarray(self.signal_frequency_hz),
            "idler_frequency_hz": np.asarray(self.idler_frequency_hz),
            "signal_gain_db": np.asarray(self.signal_gain_db_noisy),
            "signal_labels": np.asarray(self.signal_labels),
            "idler_labels": np.asarray(self.idler_labels),
            "metadata_json": json.dumps(self.to_dict(include_arrays=False)),
        }

        if self.idler_conversion_db_noisy is not None:
            payload["idler_conversion_db"] = np.asarray(self.idler_conversion_db_noisy)

        if include_clean:
            payload["signal_gain_db_clean"] = np.asarray(self.signal_gain_db_clean)
            if self.idler_conversion_db_clean is not None:
                payload["idler_conversion_db_clean"] = np.asarray(self.idler_conversion_db_clean)

        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "SyntheticGainDataset":
        npz = np.load(path, allow_pickle=True)
        metadata = {}
        noise = SyntheticNoiseConfig()

        if "metadata_json" in npz:
            metadata_payload = json.loads(str(npz["metadata_json"].item()))
            metadata = metadata_payload.get("metadata", {})
            noise = SyntheticNoiseConfig(**{
                k: v
                for k, v in metadata_payload.get("noise", {}).items()
                if k in SyntheticNoiseConfig.__dataclass_fields__
            })

        fs = jnp.asarray(npz["signal_frequency_hz"], dtype=jnp.float64)
        fi = jnp.asarray(npz["idler_frequency_hz"], dtype=jnp.float64)

        gain_noisy = jnp.asarray(npz["signal_gain_db"], dtype=jnp.float64)
        gain_clean = (
            jnp.asarray(npz["signal_gain_db_clean"], dtype=jnp.float64)
            if "signal_gain_db_clean" in npz
            else gain_noisy
        )

        idler_noisy = (
            jnp.asarray(npz["idler_conversion_db"], dtype=jnp.float64)
            if "idler_conversion_db" in npz
            else None
        )
        idler_clean = (
            jnp.asarray(npz["idler_conversion_db_clean"], dtype=jnp.float64)
            if "idler_conversion_db_clean" in npz
            else idler_noisy
        )

        signal_labels = tuple(str(x) for x in np.asarray(npz["signal_labels"]).tolist())
        idler_labels = tuple(str(x) for x in np.asarray(npz["idler_labels"]).tolist())

        return cls(
            signal_frequency_hz=fs,
            idler_frequency_hz=fi,
            signal_gain_db_clean=gain_clean,
            signal_gain_db_noisy=gain_noisy,
            idler_conversion_db_clean=idler_clean,
            idler_conversion_db_noisy=idler_noisy,
            signal_labels=signal_labels,
            idler_labels=idler_labels,
            noise=noise,
            true_parameters={},
            metadata={
                **metadata,
                "loaded_from": str(path),
            },
        )

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        out = {
            "kind": SyntheticMeasurementKind.GAIN.value,
            "n_points": self.n_points,
            "signal_frequency_min_hz": float(self.signal_frequency_hz[0]),
            "signal_frequency_max_hz": float(self.signal_frequency_hz[-1]),
            "signal_gain_db_clean_min": float(jnp.nanmin(self.signal_gain_db_clean)),
            "signal_gain_db_clean_max": float(jnp.nanmax(self.signal_gain_db_clean)),
            "signal_gain_db_noisy_min": float(jnp.nanmin(self.signal_gain_db_noisy)),
            "signal_gain_db_noisy_max": float(jnp.nanmax(self.signal_gain_db_noisy)),
            "has_idler_conversion": self.idler_conversion_db_noisy is not None,
            "noise": self.noise.to_dict(),
            "true_parameters": dict(self.true_parameters or {}),
            "metadata": dict(self.metadata or {}),
        }

        if self.idler_conversion_db_noisy is not None:
            out["idler_conversion_db_noisy_min"] = float(jnp.nanmin(self.idler_conversion_db_noisy))
            out["idler_conversion_db_noisy_max"] = float(jnp.nanmax(self.idler_conversion_db_noisy))

        if include_arrays:
            out["signal_frequency_hz"] = np.asarray(self.signal_frequency_hz).tolist()
            out["idler_frequency_hz"] = np.asarray(self.idler_frequency_hz).tolist()
            out["signal_gain_db_clean"] = np.asarray(self.signal_gain_db_clean).tolist()
            out["signal_gain_db_noisy"] = np.asarray(self.signal_gain_db_noisy).tolist()
            if self.idler_conversion_db_clean is not None:
                out["idler_conversion_db_clean"] = np.asarray(self.idler_conversion_db_clean).tolist()
            if self.idler_conversion_db_noisy is not None:
                out["idler_conversion_db_noisy"] = np.asarray(self.idler_conversion_db_noisy).tolist()

        return out


@dataclass(frozen=True)
class SyntheticCombinedDataset:
    """
    Combined S-parameter and gain synthetic dataset.
    """

    sparameters: SyntheticSParameterDataset | None = None
    gain: SyntheticGainDataset | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.sparameters is None and self.gain is None:
            raise ValueError("At least one of sparameters or gain must be provided")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def save_npz_bundle(self, directory: str | Path, *, prefix: str = "synthetic") -> dict[str, str]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}

        if self.sparameters is not None:
            paths["sparameters_npz"] = str(
                self.sparameters.save_npz(directory / f"{prefix}_sparameters.npz")
            )

        if self.gain is not None:
            paths["gain_npz"] = str(
                self.gain.save_npz(directory / f"{prefix}_gain.npz")
            )

        summary_path = directory / f"{prefix}_combined_summary.json"
        summary_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        paths["summary_json"] = str(summary_path)

        return paths

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": SyntheticMeasurementKind.COMBINED.value,
            "sparameters": None if self.sparameters is None else self.sparameters.to_dict(include_arrays=False),
            "gain": None if self.gain is None else self.gain.to_dict(include_arrays=False),
            "metadata": dict(self.metadata or {}),
        }


def apply_parameter_scales_to_layout(
    layout: LineLayout,
    parameters: Mapping[str, float] | None = None,
    *,
    name: str | None = None,
) -> LineLayout:
    """
    Apply common calibration/recovery scale parameters to a layout.

    Supported keys:
        L_scale, C_scale, C_stub_scale, R_scale, G_scale
    """
    p = dict(parameters or {})

    L_scale = float(p.get("L_scale", 1.0))
    C_scale = float(p.get("C_scale", 1.0))
    C_stub_scale = float(p.get("C_stub_scale", 1.0))
    R_scale = float(p.get("R_scale", 1.0))
    G_scale = float(p.get("G_scale", 1.0))

    return make_layout_from_arrays(
        length_m=layout.length_m,
        L_series_H=layout.L_series_H * L_scale,
        C_shunt_F=layout.C_shunt_F * C_scale,
        R_series_ohm=layout.R_series_ohm * R_scale,
        G_shunt_S=layout.G_shunt_S * G_scale,
        C_stub_F=layout.C_stub_F * C_stub_scale,
        L_res_H=layout.L_res_H,
        C_res_F=layout.C_res_F,
        C_couple_F=layout.C_couple_F,
        z0_ohm=layout.z0_ohm,
        name=name or f"{layout.name}_scaled",
        metadata={
            **dict(layout.metadata or {}),
            "source": "apply_parameter_scales_to_layout",
            "base_layout": layout.name,
            "parameter_scales": {
                "L_scale": L_scale,
                "C_scale": C_scale,
                "C_stub_scale": C_stub_scale,
                "R_scale": R_scale,
                "G_scale": G_scale,
            },
        },
    )


def apply_parameter_scales_to_nonlinear_params(
    nonlinear_params: NonlinearParams,
    parameters: Mapping[str, float] | None = None,
) -> NonlinearParams:
    """
    Apply common nonlinear scale parameters.

    Supported keys:
        I_star_scale, beta_nl_scale
    """
    p = dict(parameters or {})
    I_star_scale = float(p.get("I_star_scale", 1.0))
    beta_nl_scale = float(p.get("beta_nl_scale", 1.0))

    return NonlinearParams(
        I_star_A=nonlinear_params.I_star_A * I_star_scale,
        beta_nl=nonlinear_params.beta_nl * beta_nl_scale,
        quartic_coefficient=nonlinear_params.quartic_coefficient,
        dc_bias_A=nonlinear_params.dc_bias_A,
    )


def apply_parameter_scales_to_pump_drive(
    pump_drive: PumpDriveConfig,
    parameters: Mapping[str, float] | None = None,
) -> PumpDriveConfig:
    """
    Apply common pump scale parameters.

    Supported keys:
        pump_current_scale, pump_power_offset_db
    """
    p = dict(parameters or {})
    current_scale = float(p.get("pump_current_scale", 1.0))
    power_offset_db = float(p.get("pump_power_offset_db", 0.0))

    current = pump_drive.current_rms_A * current_scale
    if power_offset_db != 0.0:
        current = current * float(10.0 ** (power_offset_db / 20.0))

    return PumpDriveConfig.from_current_rms(
        pump_frequency_hz=pump_drive.pump_frequency_hz,
        current_rms_A=current,
        source_impedance_ohm=pump_drive.source_impedance_ohm,
        pump_label=pump_drive.pump_label,
        phase_rad=pump_drive.phase_rad,
        input_node=pump_drive.input_node,
    )


def generate_synthetic_sparameters(
    layout: LineLayout,
    *,
    frequency_hz: ArrayLike,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    noise: SyntheticNoiseConfig | None = None,
    true_parameters: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SyntheticSParameterDataset:
    """
    Generate a synthetic pump-off S-parameter dataset.
    """
    noise_cfg = noise or SyntheticNoiseConfig()
    rng = noise_cfg.rng()

    true_layout = apply_parameter_scales_to_layout(
        layout,
        true_parameters,
        name=f"{layout.name}_synthetic_truth",
    )

    f = jnp.asarray(frequency_hz, dtype=jnp.float64)

    scan = run_linear_scan(
        f,
        true_layout,
        cell_model=cell_model or CellModelConfig(),
        cascade_config=cascade_config or CascadeConfig(),
    )

    s_clean = scan.s
    s_noisy = noise_cfg.add_complex_noise(s_clean, rng=rng)

    s21_db_clean = scan.s21_db
    s21_db_noisy = noise_cfg.add_s_db_noise(s21_db_clean, rng=rng)

    return SyntheticSParameterDataset(
        frequency_hz=f,
        s_clean=s_clean,
        s_noisy=s_noisy,
        s21_db_clean=s21_db_clean,
        s21_db_noisy=s21_db_noisy,
        noise=noise_cfg,
        true_parameters=dict(true_parameters or {}),
        metadata={
            "source": "generate_synthetic_sparameters",
            "base_layout": layout.summary(),
            "true_layout": true_layout.summary(),
            "cell_model": (cell_model or CellModelConfig()).to_dict(),
            "cascade_config": (cascade_config or CascadeConfig()).to_dict(),
            "linear_scan": scan.to_dict(),
            **dict(metadata or {}),
        },
    )


def _try_call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**filtered)
    except TypeError:
        return fn(**kwargs)


def make_gain_frequency_plan(
    *,
    pump_frequency_hz: float,
    signal_frequency_hz: Sequence[float] | ArrayLike,
    idler_frequency_hz: Sequence[float] | ArrayLike,
    pump_label: str = "pump",
    signal_labels: Sequence[str] | None = None,
    idler_labels: Sequence[str] | None = None,
    n_pump_harmonics: int = 3,
    include_negative: bool = True,
    include_dc: bool = False,
) -> Any:
    """
    Build a target plan containing pump harmonics plus all signal/idler tones.
    """
    fs = np.asarray(signal_frequency_hz, dtype=float)
    fi = np.asarray(idler_frequency_hz, dtype=float)

    if fs.ndim != 1 or fi.ndim != 1:
        raise ValueError("signal_frequency_hz and idler_frequency_hz must be 1D")
    if fs.shape != fi.shape:
        raise ValueError("signal and idler frequency arrays must have same shape")

    n = int(fs.shape[0])
    sig_labels = tuple(signal_labels or tuple(f"signal_{i}" for i in range(n)))
    idl_labels = tuple(idler_labels or tuple(f"idler_{i}" for i in range(n)))

    if len(sig_labels) != n:
        raise ValueError("signal_labels length mismatch")
    if len(idl_labels) != n:
        raise ValueError("idler_labels length mismatch")

    fpmod = frequency_plan_module

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
        "signal_frequency_hz": jnp.asarray(fs, dtype=jnp.float64),
        "signal_frequencies_hz": jnp.asarray(fs, dtype=jnp.float64),
        "idler_frequency_hz": jnp.asarray(fi, dtype=jnp.float64),
        "idler_frequencies_hz": jnp.asarray(fi, dtype=jnp.float64),
        "pump_label": pump_label,
        "signal_labels": sig_labels,
        "idler_labels": idl_labels,
        "signal_label": sig_labels[0],
        "idler_label": idl_labels[0],
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
            for label in sig_labels:
                plan.position_of_label(label)
            for label in idl_labels:
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
        label = pump_label if h == 1 else f"{h}{pump_label}"
        add(label, h * pump_frequency_hz)
        if include_negative:
            add(f"-{label}", -h * pump_frequency_hz)

    for label, freq in zip(sig_labels, fs.tolist()):
        add(label, freq)
        if include_negative:
            add(f"-{label}", -freq)

    for label, freq in zip(idl_labels, fi.tolist()):
        add(label, freq)
        if include_negative:
            add(f"-{label}", -freq)

    unique: list[tuple[str, float]] = []
    seen: set[str] = set()
    for label, freq in zip(labels, frequencies):
        if label in seen:
            continue
        seen.add(label)
        unique.append((label, freq))

    order = np.argsort(np.asarray([freq for _, freq in unique]))
    labels_sorted = tuple(unique[i][0] for i in order)
    freqs_sorted = jnp.asarray([unique[i][1] for i in order], dtype=jnp.float64)

    generic_constructors = [
        getattr(fpmod, "make_frequency_plan", None),
        getattr(fpmod, "make_plan_from_frequencies", None),
        getattr(fpmod, "FrequencyPlan", None),
    ]

    candidate_kwargs = [
        {
            "frequencies_hz": freqs_sorted,
            "labels": labels_sorted,
            "reference_pump_hz": pump_frequency_hz,
            "kind": "synthetic_gain",
        },
        {
            "frequencies_hz": freqs_sorted,
            "tone_labels": labels_sorted,
            "reference_pump_hz": pump_frequency_hz,
            "kind": "synthetic_gain",
        },
        {
            "frequency_hz": freqs_sorted,
            "labels": labels_sorted,
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
        "Could not construct synthetic gain FrequencyPlan. "
        "Add make_multi_signal_idler_plan(...) or make_frequency_plan(...) to "
        f"twpa.core.frequency_plan. Errors: {errors}"
    )


def make_gain_sweep_config_for_frequencies(
    *,
    signal_labels: Sequence[str],
    idler_labels: Sequence[str],
    input_node: int = 0,
    output_node: int | None = None,
    signal_current_rms_A: complex = 1e-12 + 0j,
    input_impedance_ohm: float = 50.0,
    output_impedance_ohm: float = 50.0,
    name: str = "synthetic_gain_sweep",
) -> GainSweepConfig:
    """
    Build GainSweepConfig for one point per signal/idler label pair.
    """
    sig = tuple(signal_labels)
    idl = tuple(idler_labels)
    if len(sig) != len(idl):
        raise ValueError("signal_labels and idler_labels must have same length")

    points = tuple(
        GainSolveConfig(
            signal_label=s,
            idler_label=i,
            input_node=input_node,
            output_node=output_node,
            signal_current_rms_A=signal_current_rms_A,
            set_conjugate=True,
            input_impedance_ohm=input_impedance_ohm,
            output_impedance_ohm=output_impedance_ohm,
        )
        for s, i in zip(sig, idl)
    )

    return GainSweepConfig(
        points=points,
        require_all_converged=True,
        name=name,
    )


def generate_synthetic_gain_data(
    layout: LineLayout,
    nonlinear_params: NonlinearParams,
    *,
    pump_drive: PumpDriveConfig,
    signal_frequency_hz: ArrayLike,
    idler_frequency_hz: ArrayLike | None = None,
    pump_config: PumpHBLadderConfig | None = None,
    noise: SyntheticNoiseConfig | None = None,
    true_parameters: Mapping[str, float] | None = None,
    signal_labels: Sequence[str] | None = None,
    idler_labels: Sequence[str] | None = None,
    target_plan_factory: TargetPlanFactory | None = None,
    sweep_config_factory: SweepConfigFactory | None = None,
    input_node: int = 0,
    output_node: int | None = None,
    signal_current_rms_A: complex = 1e-12 + 0j,
    metadata: Mapping[str, Any] | None = None,
) -> SyntheticGainDataset:
    """
    Generate synthetic pump-on gain data.
    """
    noise_cfg = noise or SyntheticNoiseConfig()
    rng = noise_cfg.rng()

    parameters = dict(true_parameters or {})
    true_layout = apply_parameter_scales_to_layout(
        layout,
        parameters,
        name=f"{layout.name}_synthetic_gain_truth",
    )
    true_nonlinear = apply_parameter_scales_to_nonlinear_params(nonlinear_params, parameters)
    true_drive = apply_parameter_scales_to_pump_drive(pump_drive, parameters)

    fs = jnp.asarray(signal_frequency_hz, dtype=jnp.float64)
    if fs.ndim != 1:
        raise ValueError("signal_frequency_hz must be 1D")

    if idler_frequency_hz is None:
        fi = 2.0 * true_drive.pump_frequency_hz - fs
    else:
        fi = jnp.asarray(idler_frequency_hz, dtype=jnp.float64)
        if fi.shape != fs.shape:
            raise ValueError("idler_frequency_hz must match signal_frequency_hz")

    n = int(fs.shape[0])
    sig_labels = tuple(signal_labels or tuple(f"signal_{i}" for i in range(n)))
    idl_labels = tuple(idler_labels or tuple(f"idler_{i}" for i in range(n)))

    if len(sig_labels) != n:
        raise ValueError("signal_labels length mismatch")
    if len(idl_labels) != n:
        raise ValueError("idler_labels length mismatch")

    pump_cfg = pump_config or PumpHBLadderConfig()

    pump_result = solve_pump_hb_ladder(
        true_layout,
        true_nonlinear,
        drive=true_drive,
        pump_config=pump_cfg,
        metadata={
            "source": "generate_synthetic_gain_data",
            "stage": "pump",
        },
    )

    if target_plan_factory is None:
        target_plan = make_gain_frequency_plan(
            pump_frequency_hz=true_drive.pump_frequency_hz,
            signal_frequency_hz=fs,
            idler_frequency_hz=fi,
            pump_label=true_drive.pump_label,
            signal_labels=sig_labels,
            idler_labels=idl_labels,
            n_pump_harmonics=pump_cfg.n_pump_harmonics,
            include_negative=pump_cfg.include_negative_frequencies,
            include_dc=pump_cfg.include_dc,
        )
    else:
        target_plan = target_plan_factory(pump_result)

    if sweep_config_factory is None:
        output_impedance = (
            1.0 / pump_cfg.distributed.load_conductance_S
            if pump_cfg.distributed.load_conductance_S > 0.0
            else true_drive.source_impedance_ohm
        )
        sweep_config = make_gain_sweep_config_for_frequencies(
            signal_labels=sig_labels,
            idler_labels=idl_labels,
            input_node=input_node,
            output_node=output_node,
            signal_current_rms_A=signal_current_rms_A,
            input_impedance_ohm=true_drive.source_impedance_ohm,
            output_impedance_ohm=output_impedance,
        )
    else:
        sweep_config = sweep_config_factory(target_plan)

    gain_sweep = solve_gain_sweep_from_pump(
        pump_result,
        target_plan=target_plan,
        sweep_config=sweep_config,
    )

    signal_gain_clean = jnp.asarray(
        [point.signal_gain_db for point in gain_sweep.points],
        dtype=jnp.float64,
    )
    idler_clean_values = [
        point.idler_conversion_db
        for point in gain_sweep.points
    ]

    if any(v is not None for v in idler_clean_values):
        idler_clean = jnp.asarray(
            [jnp.nan if v is None else float(v) for v in idler_clean_values],
            dtype=jnp.float64,
        )
    else:
        idler_clean = None

    signal_gain_noisy = noise_cfg.add_gain_db_noise(signal_gain_clean, rng=rng)
    idler_noisy = noise_cfg.add_idler_db_noise(idler_clean, rng=rng)

    return SyntheticGainDataset(
        signal_frequency_hz=fs,
        idler_frequency_hz=fi,
        signal_gain_db_clean=signal_gain_clean,
        signal_gain_db_noisy=signal_gain_noisy,
        idler_conversion_db_clean=idler_clean,
        idler_conversion_db_noisy=idler_noisy,
        signal_labels=sig_labels,
        idler_labels=idl_labels,
        noise=noise_cfg,
        true_parameters=parameters,
        metadata={
            "source": "generate_synthetic_gain_data",
            "base_layout": layout.summary(),
            "true_layout": true_layout.summary(),
            "pump_drive": true_drive.to_dict(),
            "pump_result": pump_result.to_dict(),
            "target_plan": target_plan.to_dict(),
            "sweep_config": sweep_config.to_dict(),
            "gain_sweep": gain_sweep.to_dict(),
            **dict(metadata or {}),
        },
    )


def generate_combined_synthetic_dataset(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    *,
    sparameter_frequency_hz: ArrayLike | None = None,
    signal_frequency_hz: ArrayLike | None = None,
    pump_drive: PumpDriveConfig | None = None,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    pump_config: PumpHBLadderConfig | None = None,
    noise: SyntheticNoiseConfig | None = None,
    true_parameters: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SyntheticCombinedDataset:
    """
    Generate a combined synthetic S-parameter and gain dataset.

    Either or both frequency grids may be supplied.
    """
    sparams = None
    gain = None

    if sparameter_frequency_hz is not None:
        sparams = generate_synthetic_sparameters(
            layout,
            frequency_hz=sparameter_frequency_hz,
            cell_model=cell_model,
            cascade_config=cascade_config,
            noise=noise,
            true_parameters=true_parameters,
            metadata={
                "combined_dataset_stage": "sparameters",
                **dict(metadata or {}),
            },
        )

    if signal_frequency_hz is not None:
        if nonlinear_params is None:
            raise ValueError("nonlinear_params is required when generating gain data")
        if pump_drive is None:
            raise ValueError("pump_drive is required when generating gain data")

        gain = generate_synthetic_gain_data(
            layout,
            nonlinear_params,
            pump_drive=pump_drive,
            signal_frequency_hz=signal_frequency_hz,
            pump_config=pump_config,
            noise=noise,
            true_parameters=true_parameters,
            metadata={
                "combined_dataset_stage": "gain",
                **dict(metadata or {}),
            },
        )

    return SyntheticCombinedDataset(
        sparameters=sparams,
        gain=gain,
        metadata={
            "source": "generate_combined_synthetic_dataset",
            "true_parameters": dict(true_parameters or {}),
            **dict(metadata or {}),
        },
    )


def dataset_summary_markdown(
    dataset: SyntheticSParameterDataset | SyntheticGainDataset | SyntheticCombinedDataset,
) -> str:
    """
    Human-readable Markdown summary for synthetic datasets.
    """
    if isinstance(dataset, SyntheticSParameterDataset):
        d = dataset.to_dict(include_arrays=False)
        return "\n".join(
            [
                "# Synthetic S-parameter dataset",
                "",
                f"- points: `{d['n_frequency']}`",
                f"- frequency range: `{d['frequency_min_hz'] / 1e9:.6g}`–`{d['frequency_max_hz'] / 1e9:.6g} GHz`",
                f"- noisy S21 dB range: `{d['s21_db_noisy_min']:.6g}` to `{d['s21_db_noisy_max']:.6g}`",
                f"- noise: `{d['noise']}`",
                "",
                "## True parameters",
                "",
                *[f"- `{k}`: `{v}`" for k, v in d["true_parameters"].items()],
            ]
        )

    if isinstance(dataset, SyntheticGainDataset):
        d = dataset.to_dict(include_arrays=False)
        return "\n".join(
            [
                "# Synthetic gain dataset",
                "",
                f"- points: `{d['n_points']}`",
                f"- signal frequency range: `{d['signal_frequency_min_hz'] / 1e9:.6g}`–`{d['signal_frequency_max_hz'] / 1e9:.6g} GHz`",
                f"- noisy gain dB range: `{d['signal_gain_db_noisy_min']:.6g}` to `{d['signal_gain_db_noisy_max']:.6g}`",
                f"- has idler conversion: `{d['has_idler_conversion']}`",
                f"- noise: `{d['noise']}`",
                "",
                "## True parameters",
                "",
                *[f"- `{k}`: `{v}`" for k, v in d["true_parameters"].items()],
            ]
        )

    d = dataset.to_dict()
    lines = ["# Combined synthetic dataset", ""]
    lines.append(f"- has S-parameters: `{dataset.sparameters is not None}`")
    lines.append(f"- has gain: `{dataset.gain is not None}`")
    lines.append("")
    if dataset.sparameters is not None:
        lines.append("## S-parameters")
        lines.append("")
        lines.append(dataset_summary_markdown(dataset.sparameters))
        lines.append("")
    if dataset.gain is not None:
        lines.append("## Gain")
        lines.append("")
        lines.append(dataset_summary_markdown(dataset.gain))
    return "\n".join(lines)


__all__ = [
    "ArrayLike",
    "TargetPlanFactory",
    "SweepConfigFactory",
    "SyntheticMeasurementKind",
    "SyntheticNoiseConfig",
    "SyntheticSParameterDataset",
    "SyntheticGainDataset",
    "SyntheticCombinedDataset",
    "apply_parameter_scales_to_layout",
    "apply_parameter_scales_to_nonlinear_params",
    "apply_parameter_scales_to_pump_drive",
    "generate_synthetic_sparameters",
    "make_gain_frequency_plan",
    "make_gain_sweep_config_for_frequencies",
    "generate_synthetic_gain_data",
    "generate_combined_synthetic_dataset",
    "dataset_summary_markdown",
]