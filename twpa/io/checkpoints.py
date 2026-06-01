"""
twpa.io.checkpoints
===================

Checkpoint save/load utilities for TWPA simulations and calibration workflows.

This module provides a simple, explicit checkpoint format based on:

    - compressed NPZ arrays,
    - JSON metadata,
    - optional JSON payload summaries.

The checkpoint format is intentionally transparent. A checkpoint file is a
single ``.npz`` archive containing:

    metadata_json
        JSON-serialized CheckpointMetadata.

    payload_json
        Optional JSON-serializable payload.

    array:<name>
        NumPy/JAX arrays.

    scalar:<name>
        Scalar values encoded as zero-dimensional arrays.

Large simulator objects should store their arrays here and use metadata/payload
for reconstruction information.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import hashlib
import json
import time
import uuid

import numpy as np

import jax
import jax.numpy as jnp

from twpa.io.reports import jsonify, now_iso, runtime_environment


ArrayLike = Any


class CheckpointKind(str, Enum):
    """Supported checkpoint families."""

    GENERIC = "generic"
    LINEAR_SCAN = "linear_scan"
    PUMP_HB = "pump_hb"
    GAIN_SWEEP = "gain_sweep"
    CALIBRATION = "calibration"
    SYNTHETIC_BENCHMARK = "synthetic_benchmark"
    INDUSTRIAL_WORKFLOW = "industrial_workflow"


class CheckpointCompression(str, Enum):
    """Checkpoint compression mode."""

    COMPRESSED = "compressed"
    UNCOMPRESSED = "uncompressed"


@dataclass(frozen=True)
class CheckpointMetadata:
    """
    Metadata stored with every checkpoint.

    Parameters
    ----------
    kind:
        Checkpoint family.
    name:
        Human-readable name.
    version:
        Checkpoint schema version.
    run_id:
        Unique run identifier. Generated if omitted.
    created_at:
        UTC timestamp. Generated if omitted.
    description:
        Human-readable description.
    tags:
        Free-form tags.
    source:
        Source script/module/notebook.
    extra:
        Additional metadata.
    """

    kind: CheckpointKind = CheckpointKind.GENERIC
    name: str = "checkpoint"
    version: str = "1.0"
    run_id: str | None = None
    created_at: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    source: str = ""
    extra: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CheckpointKind(self.kind))
        object.__setattr__(self, "tags", tuple(str(t) for t in self.tags))
        object.__setattr__(self, "extra", dict(self.extra or {}))

        if self.run_id is None:
            object.__setattr__(self, "run_id", str(uuid.uuid4()))
        if self.created_at is None:
            object.__setattr__(self, "created_at", now_iso())

    def with_updates(self, **kwargs: Any) -> "CheckpointMetadata":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "description": self.description,
            "tags": list(self.tags),
            "source": self.source,
            "extra": jsonify(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointMetadata":
        return cls(
            kind=CheckpointKind(data.get("kind", CheckpointKind.GENERIC)),
            name=str(data.get("name", "checkpoint")),
            version=str(data.get("version", "1.0")),
            run_id=data.get("run_id"),
            created_at=data.get("created_at"),
            description=str(data.get("description", "")),
            tags=tuple(data.get("tags", ())),
            source=str(data.get("source", "")),
            extra=dict(data.get("extra", {})),
        )


@dataclass(frozen=True)
class Checkpoint:
    """
    Loaded checkpoint object.
    """

    metadata: CheckpointMetadata
    arrays: Mapping[str, jax.Array]
    payload: Mapping[str, Any] | None = None
    scalars: Mapping[str, Any] | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arrays",
            {str(k): jnp.asarray(v) for k, v in dict(self.arrays or {}).items()},
        )
        object.__setattr__(self, "payload", dict(self.payload or {}))
        object.__setattr__(self, "scalars", dict(self.scalars or {}))

    def array(self, name: str) -> jax.Array:
        if name not in self.arrays:
            raise KeyError(name)
        return self.arrays[name]

    def scalar(self, name: str, default: Any = None) -> Any:
        return dict(self.scalars or {}).get(name, default)

    def to_dict(self, *, include_array_values: bool = False) -> dict[str, Any]:
        arrays_dict: dict[str, Any] = {}
        for key, value in self.arrays.items():
            arr = np.asarray(value)
            if include_array_values:
                if np.iscomplexobj(arr):
                    arrays_dict[key] = {
                        "real": np.real(arr).tolist(),
                        "imag": np.imag(arr).tolist(),
                    }
                else:
                    arrays_dict[key] = arr.tolist()
            else:
                arrays_dict[key] = {
                    "shape": tuple(int(v) for v in arr.shape),
                    "dtype": str(arr.dtype),
                    "min_abs": float(np.nanmin(np.abs(arr))) if arr.size else None,
                    "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else None,
                }

        return {
            "metadata": self.metadata.to_dict(),
            "arrays": arrays_dict,
            "payload": jsonify(self.payload),
            "scalars": jsonify(self.scalars),
            "path": self.path,
        }


def _json_dumps(payload: Any) -> str:
    return json.dumps(jsonify(payload), sort_keys=True, indent=2)


def _hash_file(path: Path, *, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _split_checkpoint_npz(npz: Any) -> tuple[CheckpointMetadata, dict[str, jax.Array], dict[str, Any], dict[str, Any]]:
    metadata_payload = {}
    payload = {}
    arrays: dict[str, jax.Array] = {}
    scalars: dict[str, Any] = {}

    if "metadata_json" in npz:
        raw = npz["metadata_json"]
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata_payload = json.loads(str(raw))

    if "payload_json" in npz:
        raw = npz["payload_json"]
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(str(raw))

    for key in npz.files:
        if key in {"metadata_json", "payload_json"}:
            continue

        value = npz[key]

        if key.startswith("array:"):
            arrays[key.split("array:", 1)[1]] = jnp.asarray(value)
        elif key.startswith("scalar:"):
            name = key.split("scalar:", 1)[1]
            if value.shape == ():
                scalars[name] = value.item()
            else:
                scalars[name] = value.tolist()
        else:
            arrays[key] = jnp.asarray(value)

    metadata = CheckpointMetadata.from_dict(metadata_payload)
    return metadata, arrays, payload, scalars


def save_checkpoint(
    path: str | Path,
    *,
    metadata: CheckpointMetadata | None = None,
    arrays: Mapping[str, ArrayLike] | None = None,
    payload: Mapping[str, Any] | None = None,
    scalars: Mapping[str, Any] | None = None,
    compression: CheckpointCompression = CheckpointCompression.COMPRESSED,
    include_runtime: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Save a checkpoint to a single NPZ file.

    Parameters
    ----------
    path:
        Output path.
    metadata:
        Checkpoint metadata. If omitted, generic metadata is generated.
    arrays:
        Mapping of array names to arrays.
    payload:
        JSON-serializable payload.
    scalars:
        Scalar values.
    compression:
        Compressed or uncompressed NPZ.
    include_runtime:
        Add runtime environment information to metadata.extra.
    overwrite:
        If False, raise when path already exists.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and not overwrite:
        raise FileExistsError(p)

    compression = CheckpointCompression(compression)

    meta = metadata or CheckpointMetadata()
    extra = dict(meta.extra or {})
    if include_runtime:
        extra["runtime_environment"] = runtime_environment()
    meta = meta.with_updates(extra=extra)

    payload_dict = dict(payload or {})
    scalar_dict = dict(scalars or {})
    array_dict = dict(arrays or {})

    npz_payload: dict[str, Any] = {
        "metadata_json": np.asarray(_json_dumps(meta.to_dict())),
        "payload_json": np.asarray(_json_dumps(payload_dict)),
    }

    for key, value in array_dict.items():
        npz_payload[f"array:{key}"] = np.asarray(value)

    for key, value in scalar_dict.items():
        npz_payload[f"scalar:{key}"] = np.asarray(value)

    if compression == CheckpointCompression.COMPRESSED:
        np.savez_compressed(p, **npz_payload)
    else:
        np.savez(p, **npz_payload)

    return p


def load_checkpoint(path: str | Path) -> Checkpoint:
    """
    Load a checkpoint from an NPZ file.
    """
    p = Path(path)
    npz = np.load(p, allow_pickle=True)
    metadata, arrays, payload, scalars = _split_checkpoint_npz(npz)

    return Checkpoint(
        metadata=metadata,
        arrays=arrays,
        payload=payload,
        scalars=scalars,
        path=str(p),
    )


def inspect_checkpoint(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    """
    Inspect checkpoint metadata and array summaries without returning the full
    object to user code.
    """
    p = Path(path)
    ckpt = load_checkpoint(p)
    out = ckpt.to_dict(include_array_values=False)

    stat = p.stat()
    out["file"] = {
        "path": str(p),
        "size_bytes": int(stat.st_size),
        "modified_at": stat.st_mtime,
    }

    if include_hash:
        out["file"]["sha256"] = _hash_file(p)

    return out


def save_object_checkpoint(
    path: str | Path,
    obj: Any,
    *,
    kind: CheckpointKind = CheckpointKind.GENERIC,
    name: str = "object_checkpoint",
    arrays: Mapping[str, ArrayLike] | None = None,
    extra_payload: Mapping[str, Any] | None = None,
    source: str = "",
) -> Path:
    """
    Save an arbitrary object through its ``to_dict`` method plus optional arrays.

    This is useful for result objects whose arrays are already summarized in
    ``to_dict`` but where selected raw arrays should be preserved.
    """
    if hasattr(obj, "to_dict"):
        payload = obj.to_dict()
    else:
        payload = jsonify(obj)

    if extra_payload:
        if isinstance(payload, dict):
            payload = {**payload, **dict(extra_payload)}
        else:
            payload = {"object": payload, **dict(extra_payload)}

    metadata = CheckpointMetadata(
        kind=kind,
        name=name,
        source=source,
        extra={
            "object_type": type(obj).__name__,
            "object_module": type(obj).__module__,
        },
    )

    return save_checkpoint(
        path,
        metadata=metadata,
        arrays=arrays,
        payload=payload if isinstance(payload, Mapping) else {"payload": payload},
    )


def checkpoint_from_linear_scan(
    path: str | Path,
    scan: Any,
    *,
    name: str = "linear_scan_checkpoint",
    source: str = "",
) -> Path:
    """
    Save a linear scan result checkpoint.
    """
    arrays = {}
    for attr in [
        "frequency_hz",
        "s",
        "s11",
        "s21",
        "s12",
        "s22",
        "s21_db",
        "abcd",
        "beta_eff_rad_per_m",
        "group_delay_s",
    ]:
        if hasattr(scan, attr):
            value = getattr(scan, attr)
            if value is not None:
                arrays[attr] = value

    return save_object_checkpoint(
        path,
        scan,
        kind=CheckpointKind.LINEAR_SCAN,
        name=name,
        arrays=arrays,
        source=source,
    )


def checkpoint_from_pump_result(
    path: str | Path,
    pump_result: Any,
    *,
    name: str = "pump_hb_checkpoint",
    source: str = "",
) -> Path:
    """
    Save a pump-HB result checkpoint.
    """
    arrays = {}

    if hasattr(pump_result, "frequency_plan") and hasattr(pump_result.frequency_plan, "frequencies_hz"):
        arrays["frequencies_hz"] = pump_result.frequency_plan.frequencies_hz

    if hasattr(pump_result, "state"):
        state = pump_result.state
        for attr in [
            "node_voltage_coeffs_V",
            "branch_current_coeffs_A",
        ]:
            if hasattr(state, attr):
                arrays[attr] = getattr(state, attr)

    if hasattr(pump_result, "residual"):
        residual = pump_result.residual
        for attr in [
            "kcl_A",
            "branch_kvl_V",
            "norm",
        ]:
            if hasattr(residual, attr):
                value = getattr(residual, attr)
                if hasattr(value, "shape"):
                    arrays[f"residual_{attr}"] = value

    return save_object_checkpoint(
        path,
        pump_result,
        kind=CheckpointKind.PUMP_HB,
        name=name,
        arrays=arrays,
        source=source,
    )


def checkpoint_from_gain_sweep(
    path: str | Path,
    gain_sweep: Any,
    *,
    name: str = "gain_sweep_checkpoint",
    source: str = "",
) -> Path:
    """
    Save a gain sweep result checkpoint.
    """
    arrays = {}

    if hasattr(gain_sweep, "points"):
        try:
            arrays["signal_gain_db"] = np.asarray(
                [p.signal_gain_db for p in gain_sweep.points],
                dtype=float,
            )
            arrays["idler_conversion_db"] = np.asarray(
                [
                    np.nan if getattr(p, "idler_conversion_db", None) is None else p.idler_conversion_db
                    for p in gain_sweep.points
                ],
                dtype=float,
            )
        except Exception:
            pass

    return save_object_checkpoint(
        path,
        gain_sweep,
        kind=CheckpointKind.GAIN_SWEEP,
        name=name,
        arrays=arrays,
        source=source,
    )


def write_checkpoint_index(
    checkpoint_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    include_hash: bool = True,
) -> Path:
    """
    Write a JSON index for a collection of checkpoints.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for path in checkpoint_paths:
        try:
            entries.append(inspect_checkpoint(path, include_hash=include_hash))
        except Exception as exc:
            entries.append(
                {
                    "file": {"path": str(path)},
                    "error": str(exc),
                }
            )

    payload = {
        "created_at": now_iso(),
        "n_checkpoints": len(entries),
        "checkpoints": entries,
    }

    output_path.write_text(json.dumps(jsonify(payload), indent=2), encoding="utf-8")
    return output_path


def checkpoint_markdown_summary(checkpoint: Checkpoint | str | Path) -> str:
    """
    Markdown summary for a checkpoint.
    """
    ckpt = load_checkpoint(checkpoint) if not isinstance(checkpoint, Checkpoint) else checkpoint
    d = ckpt.to_dict(include_array_values=False)

    lines = [
        "# Checkpoint",
        "",
        f"- name: `{ckpt.metadata.name}`",
        f"- kind: `{ckpt.metadata.kind.value}`",
        f"- run id: `{ckpt.metadata.run_id}`",
        f"- created at: `{ckpt.metadata.created_at}`",
        f"- path: `{ckpt.path}`",
        "",
        "## Arrays",
        "",
        "| name | shape | dtype | max abs |",
        "|---|---:|---|---:|",
    ]

    for name, info in d["arrays"].items():
        lines.append(
            f"| `{name}` | `{info.get('shape')}` | `{info.get('dtype')}` | "
            f"{info.get('max_abs')} |"
        )

    if ckpt.scalars:
        lines += ["", "## Scalars", ""]
        for key, value in ckpt.scalars.items():
            lines.append(f"- `{key}`: `{value}`")

    if ckpt.metadata.description:
        lines += ["", "## Description", "", ckpt.metadata.description]

    return "\n".join(lines)


__all__ = [
    "ArrayLike",
    "CheckpointKind",
    "CheckpointCompression",
    "CheckpointMetadata",
    "Checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "inspect_checkpoint",
    "save_object_checkpoint",
    "checkpoint_from_linear_scan",
    "checkpoint_from_pump_result",
    "checkpoint_from_gain_sweep",
    "write_checkpoint_index",
    "checkpoint_markdown_summary",
]