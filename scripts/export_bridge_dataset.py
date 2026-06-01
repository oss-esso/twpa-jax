"""
Export a unified TWPA bridge dataset.

This script gathers outputs from the linear, pump-HB, gain, gain-map,
compression, calibration, and synthetic-recovery workflows into one portable
dataset bundle.

The goal is to make it easy to move data between:

    - notebooks,
    - thesis figures,
    - external fitting tools,
    - MATLAB / Julia / LabVIEW-style analysis,
    - archived simulation runs.

It accepts a mix of NPZ, CSV, JSON, and markdown artifacts and exports:

    - bridge_dataset.npz
    - bridge_manifest.json
    - bridge_manifest.md
    - optional normalized CSV tables
    - optional copied source artifacts
    - optional HDF5 if h5py is installed

Examples
--------
Export a bundle from explicit workflow artifacts:

    python scripts/export_bridge_dataset.py ^
      --layout-csv outputs/full_pump_hb_100mm/full_pump_hb_100mm_layout_components.csv ^
      --pump-npz outputs/full_pump_hb_100mm/full_pump_hb_100mm_arrays.npz ^
      --gain-npz outputs/gain_from_pumped_solution/gain_from_pumped_solution_arrays.npz ^
      --output-dir outputs/bridge_dataset

Export from a full gain-map run:

    python scripts/export_bridge_dataset.py ^
      --gain-map-npz outputs/full_gain_map_100mm/full_gain_map_100mm_cube.npz ^
      --gain-map-summary-json outputs/full_gain_map_100mm/full_gain_map_100mm_summary.json ^
      --output-dir outputs/bridge_gain_map

Auto-discover common artifacts below an outputs directory:

    python scripts/export_bridge_dataset.py ^
      --scan-root outputs ^
      --output-dir outputs/bridge_auto

Create CSV-heavy bundle for external plotting:

    python scripts/export_bridge_dataset.py ^
      --pump-npz outputs/full_pump_hb_100mm/full_pump_hb_100mm_arrays.npz ^
      --gain-npz outputs/gain_from_pumped_solution/gain_from_pumped_solution_arrays.npz ^
      --export-csv ^
      --copy-sources ^
      --output-dir outputs/bridge_external
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from twpa.io.reports import jsonify as report_jsonify


class RunStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class SourceKind(str, Enum):
    LAYOUT_CSV = "layout_csv"
    LINEAR_NPZ = "linear_npz"
    LINEAR_SUMMARY_JSON = "linear_summary_json"
    PUMP_NPZ = "pump_npz"
    PUMP_SUMMARY_JSON = "pump_summary_json"
    GAIN_NPZ = "gain_npz"
    GAIN_SUMMARY_JSON = "gain_summary_json"
    GAIN_MAP_NPZ = "gain_map_npz"
    GAIN_MAP_SUMMARY_JSON = "gain_map_summary_json"
    COMPRESSION_NPZ = "compression_npz"
    COMPRESSION_SUMMARY_JSON = "compression_summary_json"
    FIT_NPZ = "fit_npz"
    FIT_SUMMARY_JSON = "fit_summary_json"
    SYNTHETIC_NPZ = "synthetic_npz"
    SYNTHETIC_SUMMARY_JSON = "synthetic_summary_json"
    MEASUREMENT_CSV = "measurement_csv"
    MEASUREMENT_NPZ = "measurement_npz"
    EXTRA_JSON = "extra_json"
    EXTRA_CSV = "extra_csv"
    EXTRA_MD = "extra_md"
    EXTRA_FILE = "extra_file"


@dataclass(frozen=True)
class SourceArtifact:
    kind: SourceKind
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "metadata": jsonify(self.metadata),
        }


@dataclass(frozen=True)
class BridgeDatasetConfig:
    output_dir: str
    name: str
    scan_root: str | None

    layout_csv: str | None
    linear_npz: str | None
    linear_summary_json: str | None
    pump_npz: str | None
    pump_summary_json: str | None
    gain_npz: str | None
    gain_summary_json: str | None
    gain_map_npz: str | None
    gain_map_summary_json: str | None
    compression_npz: str | None
    compression_summary_json: str | None
    fit_npz: str | None
    fit_summary_json: str | None
    synthetic_npz: str | None
    synthetic_summary_json: str | None
    measurement_csv: str | None
    measurement_npz: str | None
    extra_json: tuple[str, ...]
    extra_csv: tuple[str, ...]
    extra_md: tuple[str, ...]
    extra_file: tuple[str, ...]

    include_arrays: bool
    include_metadata: bool
    export_csv: bool
    export_hdf5: bool
    copy_sources: bool
    overwrite: bool

    max_csv_rows: int
    max_manifest_chars_per_source: int
    embed_full_json: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageResult:
    name: str
    status: RunStatus
    elapsed_s: float
    summary: Mapping[str, Any]
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "summary": jsonify(self.summary),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class BridgeDatasetResult:
    config: BridgeDatasetConfig
    status: RunStatus
    elapsed_s: float
    sources: tuple[SourceArtifact, ...]
    stages: tuple[StageResult, ...]
    artifact_paths: Mapping[str, str]
    metadata: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "config": self.config.to_dict(),
            "sources": [s.to_dict() for s in self.sources],
            "stages": [s.to_dict() for s in self.stages],
            "artifact_paths": dict(self.artifact_paths),
            "metadata": jsonify(self.metadata),
        }


def jsonify(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, complex):
        return {
            "real": float(np.real(obj)),
            "imag": float(np.imag(obj)),
            "abs": float(abs(obj)),
        }
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return jsonify(obj.item())
        if obj.dtype.kind in {"U", "S", "O"}:
            if obj.size <= 64:
                return [jsonify(v) for v in obj.tolist()]
            return {
                "array_shape": tuple(int(v) for v in obj.shape),
                "array_dtype": str(obj.dtype),
                "preview": [jsonify(v) for v in obj.reshape(-1)[:10].tolist()],
            }
        if np.iscomplexobj(obj):
            return {
                "array_shape": tuple(int(v) for v in obj.shape),
                "array_dtype": str(obj.dtype),
                "min_abs": float(np.nanmin(np.abs(obj))) if obj.size else None,
                "max_abs": float(np.nanmax(np.abs(obj))) if obj.size else None,
                "mean_abs": float(np.nanmean(np.abs(obj))) if obj.size else None,
            }
        return {
            "array_shape": tuple(int(v) for v in obj.shape),
            "array_dtype": str(obj.dtype),
            "min": float(np.nanmin(obj)) if obj.size else None,
            "max": float(np.nanmax(obj)) if obj.size else None,
            "mean": float(np.nanmean(obj)) if obj.size else None,
        }
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    return report_jsonify(obj)


def safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "item"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def maybe_parse_metadata_json(value: Any) -> dict[str, Any]:
    try:
        raw = value
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))
    except Exception as exc:
        return {"metadata_parse_error": f"{type(exc).__name__}: {exc}"}


def np_array_summary(arr: Any) -> dict[str, Any]:
    x = np.asarray(arr)
    out: dict[str, Any] = {
        "shape": tuple(int(v) for v in x.shape),
        "dtype": str(x.dtype),
        "size": int(x.size),
    }
    if x.size == 0:
        return out
    if x.dtype.kind in {"U", "S", "O"}:
        flat = x.reshape(-1)
        out["preview"] = [jsonify(v) for v in flat[: min(8, flat.size)].tolist()]
        return out
    if np.iscomplexobj(x):
        out.update(
            {
                "min_abs": float(np.nanmin(np.abs(x))),
                "max_abs": float(np.nanmax(np.abs(x))),
                "mean_abs": float(np.nanmean(np.abs(x))),
            }
        )
    else:
        out.update(
            {
                "min": float(np.nanmin(x)),
                "max": float(np.nanmax(x)),
                "mean": float(np.nanmean(x)),
            }
        )
    return out


def read_csv_preview(path: Path, *, max_rows: int = 10) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(dict(row))

    return {
        "columns": list(reader.fieldnames or []),
        "preview_rows": rows,
    }


def summarize_csv_numeric(path: Path) -> dict[str, Any]:
    preview = read_csv_preview(path, max_rows=5)
    columns = preview["columns"]

    numeric: dict[str, list[float]] = {c: [] for c in columns}

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        n_rows = 0
        for row in reader:
            n_rows += 1
            for c in columns:
                try:
                    numeric[c].append(float(row.get(c, "")))
                except Exception:
                    pass

    numeric_summary = {}
    for key, values in numeric.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            numeric_summary[key] = {
                "n_finite": int(arr.size),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }

    return {
        "n_rows": int(n_rows),
        "columns": columns,
        "numeric_summary": numeric_summary,
        "preview_rows": preview["preview_rows"],
    }


def make_source(kind: SourceKind, path: str | None, metadata: Mapping[str, Any] | None = None) -> SourceArtifact | None:
    if path is None:
        return None

    p = Path(path)
    if p.exists():
        size = p.stat().st_size
        try:
            digest = sha256_file(p)
        except Exception:
            digest = None
        exists = True
    else:
        size = None
        digest = None
        exists = False

    return SourceArtifact(
        kind=kind,
        path=str(p),
        exists=exists,
        size_bytes=size,
        sha256=digest,
        metadata=dict(metadata or {}),
    )


def discover_file(scan_root: Path, patterns: Sequence[str]) -> str | None:
    if not scan_root.exists():
        return None

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(scan_root.rglob(pattern))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def auto_discover(config: BridgeDatasetConfig) -> dict[str, str | None]:
    if config.scan_root is None:
        return {}

    root = Path(config.scan_root)

    return {
        "layout_csv": config.layout_csv
        or discover_file(root, ["full_pump_hb_100mm_layout_components.csv", "*layout_components.csv"]),
        "linear_npz": config.linear_npz
        or discover_file(root, ["linear_100mm_baseline_arrays.npz", "*linear*arrays.npz", "*linear*results.npz"]),
        "linear_summary_json": config.linear_summary_json
        or discover_file(root, ["linear_100mm_baseline_summary.json", "*linear*summary.json"]),
        "pump_npz": config.pump_npz
        or discover_file(root, ["full_pump_hb_100mm_arrays.npz", "pump_hb_small_ladder_arrays.npz", "*pump*arrays.npz"]),
        "pump_summary_json": config.pump_summary_json
        or discover_file(root, ["full_pump_hb_100mm_summary.json", "pump_hb_small_ladder_summary.json", "*pump*summary.json"]),
        "gain_npz": config.gain_npz
        or discover_file(root, ["gain_from_pumped_solution_arrays.npz", "*gain_from*pump*arrays.npz"]),
        "gain_summary_json": config.gain_summary_json
        or discover_file(root, ["gain_from_pumped_solution_summary.json", "*gain_from*pump*summary.json"]),
        "gain_map_npz": config.gain_map_npz
        or discover_file(root, ["full_gain_map_100mm_cube.npz", "*gain_map*cube.npz"]),
        "gain_map_summary_json": config.gain_map_summary_json
        or discover_file(root, ["full_gain_map_100mm_summary.json", "*gain_map*summary.json"]),
        "compression_npz": config.compression_npz
        or discover_file(root, ["compression_sweep_arrays.npz", "*compression*arrays.npz"]),
        "compression_summary_json": config.compression_summary_json
        or discover_file(root, ["compression_sweep_summary.json", "*compression*summary.json"]),
        "fit_npz": config.fit_npz
        or discover_file(root, ["fit_measurements_arrays.npz", "*fit*measurements*arrays.npz"]),
        "fit_summary_json": config.fit_summary_json
        or discover_file(root, ["fit_measurements_summary.json", "*fit*measurements*summary.json"]),
        "synthetic_npz": config.synthetic_npz
        or discover_file(root, ["synthetic_recovery_datasets.npz", "*synthetic*recovery*.npz"]),
        "synthetic_summary_json": config.synthetic_summary_json
        or discover_file(root, ["synthetic_recovery_summary.json", "*synthetic*recovery*summary.json"]),
        "measurement_csv": config.measurement_csv,
        "measurement_npz": config.measurement_npz,
    }


def materialize_sources(config: BridgeDatasetConfig) -> tuple[SourceArtifact, ...]:
    discovered = auto_discover(config)

    def value(name: str, explicit: str | None) -> str | None:
        return discovered.get(name) or explicit

    sources: list[SourceArtifact] = []

    items = [
        (SourceKind.LAYOUT_CSV, value("layout_csv", config.layout_csv)),
        (SourceKind.LINEAR_NPZ, value("linear_npz", config.linear_npz)),
        (SourceKind.LINEAR_SUMMARY_JSON, value("linear_summary_json", config.linear_summary_json)),
        (SourceKind.PUMP_NPZ, value("pump_npz", config.pump_npz)),
        (SourceKind.PUMP_SUMMARY_JSON, value("pump_summary_json", config.pump_summary_json)),
        (SourceKind.GAIN_NPZ, value("gain_npz", config.gain_npz)),
        (SourceKind.GAIN_SUMMARY_JSON, value("gain_summary_json", config.gain_summary_json)),
        (SourceKind.GAIN_MAP_NPZ, value("gain_map_npz", config.gain_map_npz)),
        (SourceKind.GAIN_MAP_SUMMARY_JSON, value("gain_map_summary_json", config.gain_map_summary_json)),
        (SourceKind.COMPRESSION_NPZ, value("compression_npz", config.compression_npz)),
        (SourceKind.COMPRESSION_SUMMARY_JSON, value("compression_summary_json", config.compression_summary_json)),
        (SourceKind.FIT_NPZ, value("fit_npz", config.fit_npz)),
        (SourceKind.FIT_SUMMARY_JSON, value("fit_summary_json", config.fit_summary_json)),
        (SourceKind.SYNTHETIC_NPZ, value("synthetic_npz", config.synthetic_npz)),
        (SourceKind.SYNTHETIC_SUMMARY_JSON, value("synthetic_summary_json", config.synthetic_summary_json)),
        (SourceKind.MEASUREMENT_CSV, value("measurement_csv", config.measurement_csv)),
        (SourceKind.MEASUREMENT_NPZ, value("measurement_npz", config.measurement_npz)),
    ]

    for kind, path in items:
        source = make_source(kind, path)
        if source is not None:
            sources.append(source)

    for path in config.extra_json:
        source = make_source(SourceKind.EXTRA_JSON, path)
        if source is not None:
            sources.append(source)

    for path in config.extra_csv:
        source = make_source(SourceKind.EXTRA_CSV, path)
        if source is not None:
            sources.append(source)

    for path in config.extra_md:
        source = make_source(SourceKind.EXTRA_MD, path)
        if source is not None:
            sources.append(source)

    for path in config.extra_file:
        source = make_source(SourceKind.EXTRA_FILE, path)
        if source is not None:
            sources.append(source)

    unique: dict[tuple[str, str], SourceArtifact] = {}
    for source in sources:
        key = (source.kind.value, str(Path(source.path).resolve()))
        unique.setdefault(key, source)
    return tuple(unique.values())


def namespace_for_kind(kind: SourceKind) -> str:
    return {
        SourceKind.LAYOUT_CSV: "layout",
        SourceKind.LINEAR_NPZ: "linear",
        SourceKind.LINEAR_SUMMARY_JSON: "linear_summary",
        SourceKind.PUMP_NPZ: "pump",
        SourceKind.PUMP_SUMMARY_JSON: "pump_summary",
        SourceKind.GAIN_NPZ: "gain",
        SourceKind.GAIN_SUMMARY_JSON: "gain_summary",
        SourceKind.GAIN_MAP_NPZ: "gain_map",
        SourceKind.GAIN_MAP_SUMMARY_JSON: "gain_map_summary",
        SourceKind.COMPRESSION_NPZ: "compression",
        SourceKind.COMPRESSION_SUMMARY_JSON: "compression_summary",
        SourceKind.FIT_NPZ: "fit",
        SourceKind.FIT_SUMMARY_JSON: "fit_summary",
        SourceKind.SYNTHETIC_NPZ: "synthetic",
        SourceKind.SYNTHETIC_SUMMARY_JSON: "synthetic_summary",
        SourceKind.MEASUREMENT_CSV: "measurement_csv",
        SourceKind.MEASUREMENT_NPZ: "measurement",
        SourceKind.EXTRA_JSON: "extra_json",
        SourceKind.EXTRA_CSV: "extra_csv",
        SourceKind.EXTRA_MD: "extra_md",
        SourceKind.EXTRA_FILE: "extra_file",
    }[kind]


def load_npz_payload(path: Path, *, prefix: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz = np.load(path, allow_pickle=True)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "npz_keys": list(npz.files),
        "arrays": {},
    }

    for key in npz.files:
        value = npz[key]
        if key == "metadata_json":
            metadata["metadata_json"] = maybe_parse_metadata_json(value)
            continue

        namespaced = f"{prefix}__{safe_name(key)}"
        arrays[namespaced] = np.asarray(value)
        metadata["arrays"][namespaced] = np_array_summary(value)

    return arrays, metadata


def load_json_payload(
    path: Path,
    *,
    prefix: str,
    max_chars: int,
    embed_full_json: bool = False,
) -> dict[str, Any]:
    data = load_json(path)
    text = json.dumps(jsonify(data), indent=2)
    payload = {
        "prefix": prefix,
        "path": str(path),
        "json_preview": text[:max_chars],
        "truncated": len(text) > max_chars,
        "original_char_count": len(text),
    }
    if embed_full_json:
        payload["json"] = data
    return payload


def csv_to_structured_arrays(
    path: Path,
    *,
    prefix: str,
    max_rows: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, list[Any]] = {}
    n_rows = 0

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for col in columns:
            arrays[col] = []

        for row in reader:
            if max_rows is not None and n_rows >= max_rows:
                break
            n_rows += 1
            for col in columns:
                value = row.get(col, "")
                try:
                    arrays[col].append(float(value))
                except Exception:
                    arrays[col].append(str(value))

    out: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}

    for col, values in arrays.items():
        key = f"{prefix}__{safe_name(col)}"
        try:
            arr = np.asarray(values, dtype=float)
        except Exception:
            arr = np.asarray(values, dtype=object)

        out[key] = arr
        summaries[key] = np_array_summary(arr)

    metadata = {
        "columns": columns,
        "n_rows_loaded": n_rows,
        "arrays": summaries,
    }
    return out, metadata


def load_sources_into_bridge(
    sources: Sequence[SourceArtifact],
    config: BridgeDatasetConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    manifest_sources: dict[str, Any] = {}

    seen_prefix_counts: dict[str, int] = {}

    for source in sources:
        p = Path(source.path)
        prefix_base = namespace_for_kind(source.kind)
        count = seen_prefix_counts.get(prefix_base, 0)
        seen_prefix_counts[prefix_base] = count + 1
        prefix = prefix_base if count == 0 else f"{prefix_base}_{count}"

        source_manifest: dict[str, Any] = {
            "kind": source.kind.value,
            "path": source.path,
            "exists": source.exists,
            "size_bytes": source.size_bytes,
            "sha256": source.sha256,
            "prefix": prefix,
        }

        if not source.exists:
            source_manifest["status"] = "missing"
            manifest_sources[prefix] = source_manifest
            continue

        try:
            suffix = p.suffix.lower()

            if suffix == ".npz":
                if config.include_arrays:
                    payload, meta = load_npz_payload(p, prefix=prefix)
                    arrays.update(payload)
                    source_manifest.update(meta)
                else:
                    with np.load(p, allow_pickle=True) as npz:
                        source_manifest["npz_keys"] = list(npz.files)

            elif suffix == ".json":
                if config.include_metadata:
                    source_manifest.update(
                        load_json_payload(
                            p,
                            prefix=prefix,
                            max_chars=config.max_manifest_chars_per_source,
                            embed_full_json=config.embed_full_json,
                        )
                    )

            elif suffix == ".csv":
                csv_summary = summarize_csv_numeric(p)
                source_manifest["csv_summary"] = csv_summary

                if config.include_arrays:
                    payload, meta = csv_to_structured_arrays(
                        p,
                        prefix=prefix,
                        max_rows=None if config.max_csv_rows <= 0 else config.max_csv_rows,
                    )
                    arrays.update(payload)
                    source_manifest["csv_arrays"] = meta

            elif suffix in {".md", ".txt"}:
                text = p.read_text(encoding="utf-8", errors="replace")
                source_manifest["text_preview"] = text[: config.max_manifest_chars_per_source]
                source_manifest["truncated"] = len(text) > config.max_manifest_chars_per_source

            else:
                source_manifest["status"] = "tracked_binary_or_unknown"

            source_manifest["status"] = source_manifest.get("status", "loaded")

        except Exception as exc:
            source_manifest["status"] = "error"
            source_manifest["exception_type"] = type(exc).__name__
            source_manifest["exception_message"] = str(exc)
            source_manifest["traceback"] = traceback.format_exc()

        manifest_sources[prefix] = source_manifest

    return arrays, manifest_sources


def infer_bridge_roles(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    keys = set(arrays.keys())

    def first_contains(*parts: str) -> str | None:
        for key in sorted(keys):
            low = key.lower()
            if all(part.lower() in low for part in parts):
                return key
        return None

    roles = {
        "pump_branch_current": first_contains("pump", "branch_current"),
        "pump_node_voltage": first_contains("pump", "node_voltage"),
        "pump_frequencies": first_contains("pump", "frequencies_hz"),
        "gain_signal_frequency": first_contains("gain", "signal_frequency_hz"),
        "gain_signal_gain_db": first_contains("gain", "signal_gain_db"),
        "gain_idler_conversion_db": first_contains("gain", "idler_conversion_db"),
        "gain_map_cube": first_contains("gain_map", "gain_db"),
        "gain_map_pump_frequency": first_contains("gain_map", "pump_frequency"),
        "gain_map_pump_current_ratio": first_contains("gain_map", "pump_current"),
        "compression_power_dbm": first_contains("compression", "signal_power_dbm"),
        "compression_gain_drop_db": first_contains("compression", "gain_drop_db"),
        "fit_measured_gain_db": first_contains("fit", "measured_signal_gain_db"),
        "fit_fitted_gain_db": first_contains("fit", "fitted_signal_gain_db"),
        "layout_length_m": first_contains("layout", "length_m"),
        "layout_l_series_h": first_contains("layout", "L_series_H"),
        "layout_c_shunt_f": first_contains("layout", "C_shunt_F"),
    }

    present = {k: v for k, v in roles.items() if v is not None}

    return {
        "roles": roles,
        "present_roles": present,
        "n_roles_present": len(present),
    }


def make_bridge_arrays(
    arrays: Mapping[str, np.ndarray],
    manifest_sources: Mapping[str, Any],
    config: BridgeDatasetConfig,
) -> dict[str, np.ndarray]:
    bridge = dict(arrays)

    bridge["bridge__created_unix_time_s"] = np.asarray(time.time(), dtype=np.float64)
    bridge["bridge__source_count"] = np.asarray(len(manifest_sources), dtype=np.int64)
    bridge["bridge__array_count"] = np.asarray(len(arrays), dtype=np.int64)

    role_info = infer_bridge_roles(arrays)
    role_json = json.dumps(jsonify(role_info))
    bridge["bridge__role_manifest_json"] = np.asarray(role_json)

    return bridge


def write_npz(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["bridge__metadata_json"] = np.asarray(json.dumps(jsonify(metadata)))
    np.savez_compressed(path, **payload)
    return path


def write_hdf5(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> Path | None:
    try:
        import h5py
    except Exception:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps(jsonify(metadata))

        for key, value in arrays.items():
            arr = np.asarray(value)
            group_name = safe_name(key)

            if arr.dtype.kind in {"U", "O"}:
                str_arr = arr.astype(str)
                h5.create_dataset(group_name, data=str_arr.astype("S"))
            else:
                h5.create_dataset(group_name, data=arr, compression="gzip")

    return path


def write_array_catalog_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "array_key",
        "shape",
        "dtype",
        "size",
        "min",
        "max",
        "mean",
        "min_abs",
        "max_abs",
        "mean_abs",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for key, arr in sorted(arrays.items()):
            summary = np_array_summary(arr)
            row = {
                "array_key": key,
                "shape": json.dumps(summary.get("shape")),
                "dtype": summary.get("dtype"),
                "size": summary.get("size"),
                "min": summary.get("min"),
                "max": summary.get("max"),
                "mean": summary.get("mean"),
                "min_abs": summary.get("min_abs"),
                "max_abs": summary.get("max_abs"),
                "mean_abs": summary.get("mean_abs"),
            }
            writer.writerow(row)

    return path


def write_gain_curve_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> Path | None:
    role = infer_bridge_roles(arrays)["roles"]
    f_key = role.get("gain_signal_frequency")
    g_key = role.get("gain_signal_gain_db")
    i_key = role.get("gain_idler_conversion_db")

    if f_key is None or g_key is None:
        return None

    f = np.asarray(arrays[f_key], dtype=float).reshape(-1)
    g = np.asarray(arrays[g_key], dtype=float).reshape(-1)
    idler = None if i_key is None else np.asarray(arrays[i_key], dtype=float).reshape(-1)

    n = min(f.size, g.size)
    if n == 0:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frequency_hz", "frequency_ghz", "signal_gain_db"]
    if idler is not None:
        fields.append("idler_conversion_db")

    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for idx in range(n):
            row = {
                "frequency_hz": float(f[idx]),
                "frequency_ghz": float(f[idx] / 1e9),
                "signal_gain_db": float(g[idx]),
            }
            if idler is not None and idx < idler.size:
                row["idler_conversion_db"] = float(idler[idx])
            writer.writerow(row)

    return path


def write_pump_profile_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> Path | None:
    role = infer_bridge_roles(arrays)["roles"]
    branch_key = role.get("pump_branch_current")
    freq_key = role.get("pump_frequencies")

    if branch_key is None:
        return None

    branch = np.asarray(arrays[branch_key])
    if branch.ndim != 2:
        return None

    if freq_key is not None:
        freqs = np.asarray(arrays[freq_key], dtype=float).reshape(-1)
        if freqs.size == branch.shape[0]:
            pump_idx = int(np.argmax(np.abs(freqs)))
        else:
            pump_idx = 0
    else:
        pump_idx = min(branch.shape[0] - 1, branch.shape[0] // 2)

    profile = branch[pump_idx, :]
    n = profile.size

    z = None
    for key in arrays:
        if "z_branches_m" in key.lower():
            candidate = np.asarray(arrays[key], dtype=float).reshape(-1)
            if candidate.size == n:
                z = candidate
                break

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as out:
        fields = [
            "cell_index",
            "z_m",
            "z_mm",
            "pump_current_abs_a",
            "pump_current_phase_rad",
            "pump_current_real_a",
            "pump_current_imag_a",
        ]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for idx in range(n):
            zi = None if z is None else float(z[idx])
            writer.writerow(
                {
                    "cell_index": idx,
                    "z_m": zi,
                    "z_mm": None if zi is None else zi * 1e3,
                    "pump_current_abs_a": float(np.abs(profile[idx])),
                    "pump_current_phase_rad": float(np.angle(profile[idx])),
                    "pump_current_real_a": float(np.real(profile[idx])),
                    "pump_current_imag_a": float(np.imag(profile[idx])),
                }
            )

    return path


def write_gain_map_max_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> Path | None:
    role = infer_bridge_roles(arrays)["roles"]
    cube_key = role.get("gain_map_cube")
    fp_key = role.get("gain_map_pump_frequency")
    ir_key = role.get("gain_map_pump_current_ratio")

    if cube_key is None or fp_key is None or ir_key is None:
        return None

    cube = np.asarray(arrays[cube_key], dtype=float)
    pump_f = np.asarray(arrays[fp_key], dtype=float).reshape(-1)
    pump_i = np.asarray(arrays[ir_key], dtype=float).reshape(-1)

    if cube.ndim != 3:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as out:
        fields = [
            "pump_frequency_ghz",
            "pump_current_ratio",
            "max_gain_db",
            "best_signal_index",
        ]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for i in range(min(pump_f.size, cube.shape[0])):
            for j in range(min(pump_i.size, cube.shape[1])):
                y = cube[i, j, :]
                if np.any(np.isfinite(y)):
                    k = int(np.nanargmax(y))
                    max_gain = float(y[k])
                else:
                    k = -1
                    max_gain = np.nan

                writer.writerow(
                    {
                        "pump_frequency_ghz": float(pump_f[i]),
                        "pump_current_ratio": float(pump_i[j]),
                        "max_gain_db": max_gain,
                        "best_signal_index": k,
                    }
                )

    return path


def export_normalized_csvs(output_dir: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    csv_dir = output_dir / "csv"
    paths: dict[str, str] = {}

    catalog = write_array_catalog_csv(csv_dir / "bridge_array_catalog.csv", arrays)
    paths["array_catalog_csv"] = str(catalog)

    gain_curve = write_gain_curve_csv(csv_dir / "bridge_gain_curve.csv", arrays)
    if gain_curve is not None:
        paths["gain_curve_csv"] = str(gain_curve)

    pump_profile = write_pump_profile_csv(csv_dir / "bridge_pump_profile.csv", arrays)
    if pump_profile is not None:
        paths["pump_profile_csv"] = str(pump_profile)

    gain_map = write_gain_map_max_csv(csv_dir / "bridge_gain_map_max.csv", arrays)
    if gain_map is not None:
        paths["gain_map_max_csv"] = str(gain_map)

    return paths


def copy_source_files(output_dir: Path, sources: Sequence[SourceArtifact]) -> dict[str, str]:
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    for source in sources:
        if not source.exists:
            continue

        p = Path(source.path)
        dst_name = f"{source.kind.value}__{safe_name(p.name)}"
        dst = source_dir / dst_name

        counter = 1
        while dst.exists():
            dst = source_dir / f"{source.kind.value}__{counter}__{safe_name(p.name)}"
            counter += 1

        shutil.copy2(p, dst)
        paths[f"source_copy_{source.kind.value}_{len(paths)}"] = str(dst)

    return paths


def manifest_markdown(
    *,
    result_status: RunStatus,
    config: BridgeDatasetConfig,
    sources: Sequence[SourceArtifact],
    manifest_sources: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    artifact_paths: Mapping[str, str],
    role_info: Mapping[str, Any],
) -> str:
    lines = [
        "# TWPA bridge dataset",
        "",
        f"- status: `{result_status.value}`",
        f"- name: `{config.name}`",
        f"- source count: `{len(sources)}`",
        f"- array count: `{len(arrays)}`",
        f"- roles present: `{role_info.get('n_roles_present')}`",
        "",
        "## Recognized roles",
        "",
        "| role | array key |",
        "|---|---|",
    ]

    for role, key in role_info.get("roles", {}).items():
        lines.append(f"| `{role}` | `{key}` |")

    lines += [
        "",
        "## Sources",
        "",
        "| prefix | kind | exists | size bytes | path |",
        "|---|---|---|---:|---|",
    ]

    for prefix, item in manifest_sources.items():
        lines.append(
            f"| `{prefix}` | `{item.get('kind')}` | `{item.get('exists')}` | "
            f"{item.get('size_bytes')} | `{item.get('path')}` |"
        )

    lines += [
        "",
        "## Arrays",
        "",
        "| key | shape | dtype |",
        "|---|---:|---|",
    ]

    for key, arr in sorted(arrays.items()):
        summary = np_array_summary(arr)
        lines.append(
            f"| `{key}` | `{summary.get('shape')}` | `{summary.get('dtype')}` |"
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "| key | path |",
        "|---|---|",
    ]

    for key, path in artifact_paths.items():
        lines.append(f"| `{key}` | `{path}` |")

    return "\n".join(lines)


def export_bridge_dataset(
    *,
    config: BridgeDatasetConfig,
    sources: Sequence[SourceArtifact],
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, Any], dict[str, np.ndarray]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays_raw, manifest_sources = load_sources_into_bridge(sources, config)
    arrays = make_bridge_arrays(arrays_raw, manifest_sources, config)
    role_info = infer_bridge_roles(arrays)

    metadata = {
        "config": config.to_dict(),
        "created_unix_time_s": time.time(),
        "python": sys.version,
        "sources": {k: jsonify(v) for k, v in manifest_sources.items()},
        "role_info": role_info,
        "array_summaries": {
            key: np_array_summary(value)
            for key, value in sorted(arrays.items())
        },
    }

    artifact_paths: dict[str, str] = {}

    npz_path = write_npz(output_dir / "bridge_dataset.npz", arrays, metadata)
    artifact_paths["bridge_npz"] = str(npz_path)

    manifest_json_path = output_dir / "bridge_manifest.json"
    manifest_json_path.write_text(json.dumps(jsonify(metadata), indent=2), encoding="utf-8")
    artifact_paths["manifest_json"] = str(manifest_json_path)

    if config.export_hdf5:
        hdf5_path = write_hdf5(output_dir / "bridge_dataset.h5", arrays, metadata)
        if hdf5_path is not None:
            artifact_paths["bridge_hdf5"] = str(hdf5_path)
        else:
            hdf5_error = output_dir / "bridge_hdf5_unavailable.txt"
            hdf5_error.write_text("h5py is not installed; HDF5 export skipped.\n", encoding="utf-8")
            artifact_paths["hdf5_unavailable_txt"] = str(hdf5_error)

    if config.export_csv:
        artifact_paths.update(export_normalized_csvs(output_dir, arrays))

    if config.copy_sources:
        artifact_paths.update(copy_source_files(output_dir, sources))

    manifest_md_path = output_dir / "bridge_manifest.md"
    manifest_md_path.write_text(
        manifest_markdown(
            result_status=RunStatus.PASS,
            config=config,
            sources=sources,
            manifest_sources=manifest_sources,
            arrays=arrays,
            artifact_paths=artifact_paths,
            role_info=role_info,
        ),
        encoding="utf-8",
    )
    artifact_paths["manifest_md"] = str(manifest_md_path)

    return artifact_paths, metadata, arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a unified TWPA bridge dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bridge_dataset"))
    parser.add_argument("--name", type=str, default="twpa_bridge_dataset")
    parser.add_argument(
        "--scan-root",
        type=str,
        default=None,
        help="Optional root directory to auto-discover common workflow artifacts.",
    )

    parser.add_argument("--layout-csv", type=str, default=None)

    parser.add_argument("--linear-npz", type=str, default=None)
    parser.add_argument("--linear-summary-json", type=str, default=None)

    parser.add_argument("--pump-npz", type=str, default=None)
    parser.add_argument("--pump-summary-json", type=str, default=None)

    parser.add_argument("--gain-npz", type=str, default=None)
    parser.add_argument("--gain-summary-json", type=str, default=None)

    parser.add_argument("--gain-map-npz", type=str, default=None)
    parser.add_argument("--gain-map-summary-json", type=str, default=None)

    parser.add_argument("--compression-npz", type=str, default=None)
    parser.add_argument("--compression-summary-json", type=str, default=None)

    parser.add_argument("--fit-npz", type=str, default=None)
    parser.add_argument("--fit-summary-json", type=str, default=None)

    parser.add_argument("--synthetic-npz", type=str, default=None)
    parser.add_argument("--synthetic-summary-json", type=str, default=None)

    parser.add_argument("--measurement-csv", type=str, default=None)
    parser.add_argument("--measurement-npz", type=str, default=None)

    parser.add_argument("--extra-json", type=str, nargs="*", default=[])
    parser.add_argument("--extra-csv", type=str, nargs="*", default=[])
    parser.add_argument("--extra-md", type=str, nargs="*", default=[])
    parser.add_argument("--extra-file", type=str, nargs="*", default=[])

    parser.add_argument("--no-arrays", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--export-hdf5", action="store_true")
    parser.add_argument("--copy-sources", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=0,
        help="Maximum rows to load from each source CSV into NPZ arrays. 0 means all rows.",
    )
    parser.add_argument("--max-manifest-chars-per-source", type=int, default=20000)
    parser.add_argument(
        "--embed-full-json",
        action="store_true",
        help="Embed full source JSON payloads in bridge manifests for debugging.",
    )

    return parser


def resolve_config(args: argparse.Namespace) -> BridgeDatasetConfig:
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. "
            "Use --overwrite or choose a new --output-dir."
        )

    if args.scan_root is not None and not Path(args.scan_root).exists():
        raise FileNotFoundError(args.scan_root)

    explicit_paths = [
        args.layout_csv,
        args.linear_npz,
        args.linear_summary_json,
        args.pump_npz,
        args.pump_summary_json,
        args.gain_npz,
        args.gain_summary_json,
        args.gain_map_npz,
        args.gain_map_summary_json,
        args.compression_npz,
        args.compression_summary_json,
        args.fit_npz,
        args.fit_summary_json,
        args.synthetic_npz,
        args.synthetic_summary_json,
        args.measurement_csv,
        args.measurement_npz,
        *args.extra_json,
        *args.extra_csv,
        *args.extra_md,
        *args.extra_file,
    ]

    for path in explicit_paths:
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(path)

    if args.max_csv_rows < 0:
        raise ValueError("--max-csv-rows must be non-negative")

    if args.max_manifest_chars_per_source <= 0:
        raise ValueError("--max-manifest-chars-per-source must be positive")

    return BridgeDatasetConfig(
        output_dir=str(output_dir),
        name=str(args.name),
        scan_root=args.scan_root,
        layout_csv=args.layout_csv,
        linear_npz=args.linear_npz,
        linear_summary_json=args.linear_summary_json,
        pump_npz=args.pump_npz,
        pump_summary_json=args.pump_summary_json,
        gain_npz=args.gain_npz,
        gain_summary_json=args.gain_summary_json,
        gain_map_npz=args.gain_map_npz,
        gain_map_summary_json=args.gain_map_summary_json,
        compression_npz=args.compression_npz,
        compression_summary_json=args.compression_summary_json,
        fit_npz=args.fit_npz,
        fit_summary_json=args.fit_summary_json,
        synthetic_npz=args.synthetic_npz,
        synthetic_summary_json=args.synthetic_summary_json,
        measurement_csv=args.measurement_csv,
        measurement_npz=args.measurement_npz,
        extra_json=tuple(str(x) for x in args.extra_json),
        extra_csv=tuple(str(x) for x in args.extra_csv),
        extra_md=tuple(str(x) for x in args.extra_md),
        extra_file=tuple(str(x) for x in args.extra_file),
        include_arrays=not bool(args.no_arrays),
        include_metadata=not bool(args.no_metadata),
        export_csv=bool(args.export_csv),
        export_hdf5=bool(args.export_hdf5),
        copy_sources=bool(args.copy_sources),
        overwrite=bool(args.overwrite),
        max_csv_rows=int(args.max_csv_rows),
        max_manifest_chars_per_source=int(args.max_manifest_chars_per_source),
        embed_full_json=bool(args.embed_full_json),
    )


def finalize_result(result: BridgeDatasetResult, output_dir: Path) -> int:
    summary_json = output_dir / "bridge_export_summary.json"
    summary_md = output_dir / "bridge_export_summary.md"

    artifact_paths = {
        **dict(result.artifact_paths),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }

    result = BridgeDatasetResult(
        config=result.config,
        status=result.status,
        elapsed_s=result.elapsed_s,
        sources=result.sources,
        stages=result.stages,
        artifact_paths=artifact_paths,
        metadata=result.metadata,
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Bridge export summary",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- sources: `{len(result.sources)}`",
        f"- artifacts: `{len(result.artifact_paths)}`",
        "",
        "## Stages",
        "",
        "| stage | status | elapsed s | messages |",
        "|---|---|---:|---|",
    ]

    for stage in result.stages:
        msg = "<br>".join(stage.messages[:3])
        lines.append(
            f"| `{stage.name}` | `{stage.status.value}` | "
            f"{stage.elapsed_s:.6g} | {msg} |"
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "| key | path |",
        "|---|---|",
    ]

    for key, path in result.artifact_paths.items():
        lines.append(f"| `{key}` | `{path}` |")

    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"[bridge-export] status: {result.status.value}")
    print(f"[bridge-export] bridge NPZ: {result.artifact_paths.get('bridge_npz')}")
    print(f"[bridge-export] manifest:   {result.artifact_paths.get('manifest_json')}")
    print(f"[bridge-export] summary:    {summary_json}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[bridge-export] invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    if config.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages: list[StageResult] = []
    artifact_paths: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "python": sys.version,
        "script": "scripts/export_bridge_dataset.py",
    }

    source_start = time.perf_counter()
    try:
        sources = materialize_sources(config)
        n_existing = sum(1 for s in sources if s.exists)

        if n_existing == 0:
            stages.append(
                StageResult(
                    name="collect_sources",
                    status=RunStatus.FAIL,
                    elapsed_s=time.perf_counter() - source_start,
                    summary={"n_sources": len(sources), "n_existing": n_existing},
                    messages=("FAIL: no existing source artifacts were found.",),
                )
            )
            result = BridgeDatasetResult(
                config=config,
                status=RunStatus.ERROR,
                elapsed_s=time.perf_counter() - start,
                sources=sources,
                stages=tuple(stages),
                artifact_paths=artifact_paths,
                metadata=metadata,
            )
            return finalize_result(result, output_dir)

        stages.append(
            StageResult(
                name="collect_sources",
                status=RunStatus.PASS,
                elapsed_s=time.perf_counter() - source_start,
                summary={
                    "n_sources": len(sources),
                    "n_existing": n_existing,
                    "sources": [s.to_dict() for s in sources],
                },
                messages=(f"PASS: collected {n_existing} existing source artifacts.",),
            )
        )

    except Exception as exc:
        stages.append(
            StageResult(
                name="collect_sources",
                status=RunStatus.ERROR,
                elapsed_s=time.perf_counter() - source_start,
                summary={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                messages=(f"ERROR: source collection failed: {exc}",),
            )
        )
        sources = tuple()

    if not sources:
        result = BridgeDatasetResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            sources=sources,
            stages=tuple(stages),
            artifact_paths=artifact_paths,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    export_start = time.perf_counter()
    try:
        artifact_paths, bridge_metadata, arrays = export_bridge_dataset(
            config=config,
            sources=sources,
            output_dir=output_dir,
        )
        metadata.update(
            {
                "bridge_metadata": bridge_metadata,
                "n_arrays": len(arrays),
                "role_info": infer_bridge_roles(arrays),
            }
        )
        stages.append(
            StageResult(
                name="export_bridge",
                status=RunStatus.PASS if arrays else RunStatus.PARTIAL,
                elapsed_s=time.perf_counter() - export_start,
                summary={
                    "n_arrays": len(arrays),
                    "artifact_paths": artifact_paths,
                    "role_info": infer_bridge_roles(arrays),
                },
                messages=(
                    "PASS: bridge dataset exported."
                    if arrays
                    else "PARTIAL: bridge exported with metadata only; no arrays were loaded.",
                ),
            )
        )
    except Exception as exc:
        stages.append(
            StageResult(
                name="export_bridge",
                status=RunStatus.ERROR,
                elapsed_s=time.perf_counter() - export_start,
                summary={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                messages=(f"ERROR: bridge export failed: {exc}",),
            )
        )

    hard_fail = any(s.status in {RunStatus.FAIL, RunStatus.ERROR} for s in stages)
    partial = any(s.status == RunStatus.PARTIAL for s in stages)

    if hard_fail:
        status = RunStatus.ERROR
    elif partial:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.PASS

    result = BridgeDatasetResult(
        config=config,
        status=status,
        elapsed_s=time.perf_counter() - start,
        sources=sources,
        stages=tuple(stages),
        artifact_paths=artifact_paths,
        metadata=metadata,
    )

    return finalize_result(result, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
