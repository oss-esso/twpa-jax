"""
twpa.io.reports
===============

Report-generation helpers for TWPA simulation, calibration, and inference runs.

This module centralizes practical artifact writing:

    - JSON reports,
    - Markdown summaries,
    - NPZ array bundles,
    - run-directory indexes,
    - lightweight serialization of dataclasses/JAX/NumPy objects.

The goal is to make every long simulation run reproducible and inspectable
without each script reimplementing the same serialization code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, is_dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import datetime as _dt
import json
import platform
import sys
import time
import traceback

import numpy as np

import jax
import jax.numpy as jnp


ArrayLike = Any


class ReportFormat(str, Enum):
    """Supported report formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    NPZ = "npz"
    TEXT = "text"


class ReportStatus(str, Enum):
    """Generic run/report status."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReportArtifact:
    """
    One artifact produced by a run.
    """

    key: str
    path: str
    format: ReportFormat
    description: str = ""
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", ReportFormat(self.format))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "format": self.format.value,
            "description": self.description,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RunReport:
    """
    Generic report object for one simulator run.

    Parameters
    ----------
    name:
        Run/report name.
    status:
        Pass/fail/partial/error status.
    summary:
        Compact human-readable summary dictionary.
    payload:
        Full serializable payload.
    artifacts:
        Artifact entries.
    started_at:
        ISO timestamp.
    elapsed_s:
        Runtime in seconds.
    metadata:
        Additional metadata.
    """

    name: str
    status: ReportStatus = ReportStatus.UNKNOWN
    summary: Mapping[str, Any] | None = None
    payload: Mapping[str, Any] | None = None
    artifacts: tuple[ReportArtifact, ...] = ()
    started_at: str | None = None
    elapsed_s: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RunReport.name may not be empty")
        object.__setattr__(self, "status", ReportStatus(self.status))
        object.__setattr__(self, "summary", dict(self.summary or {}))
        object.__setattr__(self, "payload", dict(self.payload or {}))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if self.started_at is None:
            object.__setattr__(self, "started_at", now_iso())

    @property
    def passed(self) -> bool:
        return self.status == ReportStatus.PASS

    def with_artifact(self, artifact: ReportArtifact) -> "RunReport":
        return replace(self, artifacts=tuple(self.artifacts) + (artifact,))

    def with_artifacts(self, artifacts: Sequence[ReportArtifact]) -> "RunReport":
        return replace(self, artifacts=tuple(self.artifacts) + tuple(artifacts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "passed": self.passed,
            "summary": jsonify(self.summary),
            "payload": jsonify(self.payload),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_s,
            "metadata": jsonify(self.metadata),
        }


@dataclass
class RunTimer:
    """
    Small context-manager timer for report metadata.

    Example
    -------
    with RunTimer("linear_scan") as timer:
        ...
    print(timer.elapsed_s)
    """

    name: str = "run"
    start_s: float | None = None
    end_s: float | None = None
    started_at: str | None = None

    def __enter__(self) -> "RunTimer":
        self.started_at = now_iso()
        self.start_s = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.end_s = time.perf_counter()

    @property
    def elapsed_s(self) -> float | None:
        if self.start_s is None:
            return None
        end = time.perf_counter() if self.end_s is None else self.end_s
        return end - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_s,
        }


def now_iso() -> str:
    """
    Current UTC timestamp in ISO-8601 format.
    """
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def runtime_environment() -> dict[str, Any]:
    """
    Capture lightweight runtime metadata.
    """
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "jax": {
            "version": getattr(jax, "__version__", None),
            "default_backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "devices": [str(d) for d in jax.devices()],
        },
        "numpy": {
            "version": np.__version__,
        },
        "timestamp_utc": now_iso(),
    }


def jsonify(obj: Any) -> Any:
    """
    Convert common simulator objects into JSON-compatible structures.

    Handles:
        - Path
        - Enum
        - dataclasses
        - objects with to_dict()
        - NumPy/JAX arrays
        - complex numbers
        - NumPy scalars
        - mappings/sequences
    """
    if obj is None:
        return None

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, complex):
        return {
            "real": float(np.real(obj)),
            "imag": float(np.imag(obj)),
            "abs": float(abs(obj)),
        }

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return jsonify(obj.to_dict())
        except TypeError:
            return jsonify(obj.to_dict(include_arrays=False))

    if is_dataclass(obj):
        return jsonify(asdict(obj))

    if isinstance(obj, Mapping):
        return {str(k): jsonify(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return [jsonify(v) for v in obj]

    if isinstance(obj, list):
        return [jsonify(v) for v in obj]

    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)

        if arr.ndim == 0:
            if np.iscomplexobj(arr):
                return jsonify(complex(arr))
            value = arr.item()
            return jsonify(value)

        if np.iscomplexobj(arr):
            return {
                "array_shape": tuple(int(v) for v in arr.shape),
                "array_dtype": str(arr.dtype),
                "min_abs": float(np.nanmin(np.abs(arr))) if arr.size else None,
                "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else None,
                "mean_abs": float(np.nanmean(np.abs(arr))) if arr.size else None,
            }

        return {
            "array_shape": tuple(int(v) for v in arr.shape),
            "array_dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "mean": float(np.nanmean(arr)) if arr.size else None,
        }

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def full_array_jsonify(obj: Any) -> Any:
    """
    JSON converter that expands arrays into lists.

    Use only for small arrays.
    """
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {
                "real": np.real(arr).tolist(),
                "imag": np.imag(arr).tolist(),
            }
        return arr.tolist()

    if isinstance(obj, Mapping):
        return {str(k): full_array_jsonify(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [full_array_jsonify(v) for v in obj]

    return jsonify(obj)


def write_json_report(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    include_runtime: bool = False,
) -> Path:
    """
    Write a JSON report.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = jsonify(payload)
    if include_runtime:
        if isinstance(data, dict):
            data = {
                **data,
                "runtime_environment": runtime_environment(),
            }
        else:
            data = {
                "payload": data,
                "runtime_environment": runtime_environment(),
            }

    p.write_text(json.dumps(data, indent=indent, sort_keys=True), encoding="utf-8")
    return p


def write_markdown_report(
    path: str | Path,
    markdown: str,
) -> Path:
    """
    Write a Markdown report.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown, encoding="utf-8")
    return p


def write_text_report(
    path: str | Path,
    text: str,
) -> Path:
    """
    Write a plain-text report.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def write_npz_report(
    path: str | Path,
    **arrays: Any,
) -> Path:
    """
    Write a compressed NPZ array bundle.

    Non-array scalar metadata should be encoded separately as JSON. If an
    ``metadata`` keyword is provided, it is written as ``metadata_json``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    for key, value in arrays.items():
        if value is None:
            continue
        if key == "metadata":
            payload["metadata_json"] = json.dumps(jsonify(value))
        elif isinstance(value, (str, bytes)):
            payload[key] = np.asarray(value)
        else:
            payload[key] = np.asarray(value)

    np.savez_compressed(p, **payload)
    return p


def markdown_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    float_format: str = ".6g",
    empty: str = "",
) -> str:
    """
    Render a list of dictionaries as a Markdown table.
    """
    if not rows:
        return "_No rows._"

    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(str(key))
        columns = tuple(seen)
    else:
        columns = tuple(str(c) for c in columns)

    def fmt(value: Any) -> str:
        if value is None:
            return empty
        if isinstance(value, Enum):
            return f"`{value.value}`"
        if isinstance(value, bool):
            return f"`{value}`"
        if isinstance(value, (float, np.floating)):
            return format(float(value), float_format)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, complex):
            return f"{abs(value):{float_format}}∠{np.angle(value):{float_format}}"
        if isinstance(value, str):
            return value.replace("\n", " ")
        return str(jsonify(value)).replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(fmt(row.get(col)) for col in columns)
            + " |"
        )

    return "\n".join(lines)


def key_value_markdown(
    mapping: Mapping[str, Any],
    *,
    title: str | None = None,
) -> str:
    """
    Render a compact key-value dictionary as Markdown.
    """
    lines: list[str] = []
    if title:
        lines.extend([f"## {title}", ""])

    for key, value in mapping.items():
        v = jsonify(value)
        if isinstance(v, dict):
            lines.append(f"- `{key}`: `{json.dumps(v)}`")
        else:
            lines.append(f"- `{key}`: `{v}`")

    return "\n".join(lines)


def run_report_markdown(report: RunReport) -> str:
    """
    Convert a RunReport to a Markdown summary.
    """
    lines = [
        f"# {report.name}",
        "",
        f"- status: `{report.status.value}`",
        f"- passed: `{report.passed}`",
        f"- started: `{report.started_at}`",
        f"- elapsed: `{report.elapsed_s}` s",
        "",
    ]

    if report.summary:
        lines += [
            "## Summary",
            "",
            key_value_markdown(report.summary),
            "",
        ]

    if report.artifacts:
        lines += [
            "## Artifacts",
            "",
            "| key | format | path | description |",
            "|---|---|---|---|",
        ]
        for a in report.artifacts:
            lines.append(
                f"| `{a.key}` | `{a.format.value}` | `{a.path}` | {a.description} |"
            )
        lines.append("")

    if report.metadata:
        lines += [
            "## Metadata",
            "",
            "```json",
            json.dumps(jsonify(report.metadata), indent=2),
            "```",
            "",
        ]

    return "\n".join(lines)


def write_run_report_bundle(
    report: RunReport,
    output_dir: str | Path,
    *,
    prefix: str = "run",
    include_payload_json: bool = True,
    include_runtime: bool = True,
) -> dict[str, str]:
    """
    Write a standard report bundle.

    Produces:
        - <prefix>_summary.json
        - <prefix>_summary.md
        - optionally <prefix>_payload.json
        - <prefix>_artifact_index.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    summary_payload = report.to_dict()
    if not include_payload_json:
        summary_payload = {
            **summary_payload,
            "payload": {
                "omitted": True,
                "reason": "include_payload_json=False",
            },
        }

    summary_json = write_json_report(
        out / f"{prefix}_summary.json",
        summary_payload,
        include_runtime=include_runtime,
    )
    paths["summary_json"] = str(summary_json)

    summary_md = write_markdown_report(
        out / f"{prefix}_summary.md",
        run_report_markdown(report),
    )
    paths["summary_md"] = str(summary_md)

    if include_payload_json:
        payload_json = write_json_report(
            out / f"{prefix}_payload.json",
            report.payload,
            include_runtime=False,
        )
        paths["payload_json"] = str(payload_json)

    artifact_index = {
        "report_artifacts": [a.to_dict() for a in report.artifacts],
        "bundle_paths": paths,
        "created_at": now_iso(),
    }
    index_json = write_json_report(
        out / f"{prefix}_artifact_index.json",
        artifact_index,
        include_runtime=False,
    )
    paths["artifact_index_json"] = str(index_json)

    return paths


def exception_report(
    exc: BaseException,
    *,
    name: str = "error_report",
    metadata: Mapping[str, Any] | None = None,
) -> RunReport:
    """
    Build a RunReport from an exception.
    """
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return RunReport(
        name=name,
        status=ReportStatus.ERROR,
        summary={
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
        payload={
            "traceback": tb,
        },
        metadata={
            **dict(metadata or {}),
            "runtime_environment": runtime_environment(),
        },
    )


def write_exception_report(
    exc: BaseException,
    output_dir: str | Path,
    *,
    prefix: str = "error",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """
    Write standard artifacts for an exception.
    """
    report = exception_report(
        exc,
        name=f"{prefix}_report",
        metadata=metadata,
    )
    return write_run_report_bundle(
        report,
        output_dir,
        prefix=prefix,
        include_payload_json=True,
        include_runtime=True,
    )


def artifact_index_from_paths(
    paths: Mapping[str, str | Path],
    *,
    descriptions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build a compact artifact index from a mapping of path keys.
    """
    descriptions = dict(descriptions or {})

    artifacts = []
    for key, path in paths.items():
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".json":
            fmt = ReportFormat.JSON
        elif suffix in {".md", ".markdown"}:
            fmt = ReportFormat.MARKDOWN
        elif suffix == ".npz":
            fmt = ReportFormat.NPZ
        else:
            fmt = ReportFormat.TEXT

        artifacts.append(
            ReportArtifact(
                key=key,
                path=str(p),
                format=fmt,
                description=descriptions.get(key, ""),
            ).to_dict()
        )

    return {
        "artifacts": artifacts,
        "created_at": now_iso(),
    }


def write_artifact_index(
    paths: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    descriptions: Mapping[str, str] | None = None,
) -> Path:
    """
    Write an artifact index JSON.
    """
    return write_json_report(
        output_path,
        artifact_index_from_paths(paths, descriptions=descriptions),
    )


__all__ = [
    "ArrayLike",
    "ReportFormat",
    "ReportStatus",
    "ReportArtifact",
    "RunReport",
    "RunTimer",
    "now_iso",
    "runtime_environment",
    "jsonify",
    "full_array_jsonify",
    "write_json_report",
    "write_markdown_report",
    "write_text_report",
    "write_npz_report",
    "markdown_table",
    "key_value_markdown",
    "run_report_markdown",
    "write_run_report_bundle",
    "exception_report",
    "write_exception_report",
    "artifact_index_from_paths",
    "write_artifact_index",
]