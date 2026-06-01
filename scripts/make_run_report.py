"""
Create a consolidated TWPA run report.

This script scans workflow output directories, reads summary JSON/markdown files,
collects status, stages, artifacts, metrics, and configuration snippets, then
exports a human-readable run report.

It is meant to sit at the end of the production workflow chain:

    scripts/linear_100mm_baseline.py
    scripts/full_pump_hb_100mm.py
    scripts/gain_from_pumped_solution.py
    scripts/full_gain_map_100mm.py
    scripts/compression_sweep.py
    scripts/synthetic_recovery.py
    scripts/fit_measurements.py
    scripts/export_bridge_dataset.py

Examples
--------
Report everything under outputs:

    python scripts/make_run_report.py --scan-root outputs --output-dir outputs/run_report

Report selected summaries:

    python scripts/make_run_report.py ^
      --summary-json outputs/full_pump_hb_100mm/full_pump_hb_100mm_summary.json ^
      --summary-json outputs/full_gain_map_100mm/full_gain_map_100mm_summary.json ^
      --output-dir outputs/run_report_selected

Fail CI if any run is ERROR:

    python scripts/make_run_report.py ^
      --scan-root outputs ^
      --fail-on-error ^
      --output-dir outputs/run_report_ci
"""

from __future__ import annotations

import argparse
import csv
import json
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
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunReportConfig:
    scan_root: str | None
    summary_json: tuple[str, ...]
    summary_md: tuple[str, ...]
    output_dir: str
    name: str
    recursive: bool
    fail_on_error: bool
    fail_on_partial: bool
    include_markdown_bodies: bool
    max_markdown_chars: int
    max_json_chars: int
    make_plots: bool
    overwrite: bool
    include_report_summaries: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SummaryRecord:
    path: str
    kind: str
    workflow: str
    status: RunStatus
    passed: bool | None
    elapsed_s: float | None
    n_artifacts: int
    n_stages: int
    n_points: int | None
    n_runs: int | None
    n_trials: int | None
    config: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    stage_rows: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]
    messages: tuple[str, ...]
    raw_preview: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "workflow": self.workflow,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "n_artifacts": self.n_artifacts,
            "n_stages": self.n_stages,
            "n_points": self.n_points,
            "n_runs": self.n_runs,
            "n_trials": self.n_trials,
            "config": jsonify(self.config),
            "artifact_paths": dict(self.artifact_paths),
            "stage_rows": [jsonify(r) for r in self.stage_rows],
            "metrics": jsonify(self.metrics),
            "metadata": jsonify(self.metadata),
            "messages": list(self.messages),
            "raw_preview": jsonify(self.raw_preview),
        }


@dataclass(frozen=True)
class MarkdownRecord:
    path: str
    workflow: str
    size_bytes: int
    preview: str
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "workflow": self.workflow,
            "size_bytes": self.size_bytes,
            "preview": self.preview,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class RunReportResult:
    config: RunReportConfig
    status: RunStatus
    elapsed_s: float
    summaries: tuple[SummaryRecord, ...]
    markdowns: tuple[MarkdownRecord, ...]
    artifact_paths: Mapping[str, str]
    aggregate: Mapping[str, Any]
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
            "summaries": [s.to_dict() for s in self.summaries],
            "markdowns": [m.to_dict() for m in self.markdowns],
            "artifact_paths": dict(self.artifact_paths),
            "aggregate": jsonify(self.aggregate),
            "metadata": jsonify(self.metadata),
        }


def jsonify(obj: Any) -> Any:
    return report_jsonify(obj)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def parse_status(value: Any) -> RunStatus:
    if value is None:
        return RunStatus.UNKNOWN
    text = str(value).strip().lower()
    for status in RunStatus:
        if status.value == text:
            return status
    return RunStatus.UNKNOWN


def infer_workflow_from_path(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name

    known = [
        "linear_100mm_baseline",
        "extract_dispersion",
        "pump_hb_small_ladder",
        "pump_hb_scaling",
        "gain_from_pumped_solution",
        "effective_cell_convergence",
        "full_pump_hb_100mm",
        "full_gain_map_100mm",
        "compression_sweep",
        "synthetic_recovery",
        "fit_measurements",
        "bridge_export",
        "bridge_dataset",
    ]

    for key in known:
        if key in name:
            return key
        if key in parent.lower():
            return key

    if name.endswith("_summary.json"):
        return name.removesuffix("_summary.json")
    if name.endswith("_summary.md"):
        return name.removesuffix("_summary.md")
    return parent or path.stem


def nested_get(mapping: Mapping[str, Any], path: Sequence[Any], default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if isinstance(cur, Mapping):
            cur = cur.get(key, default)
        elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)) and isinstance(key, int):
            if 0 <= key < len(cur):
                cur = cur[key]
            else:
                return default
        else:
            return default
    return cur


def compact_mapping(mapping: Mapping[str, Any], *, max_chars: int) -> Mapping[str, Any]:
    text = json.dumps(jsonify(mapping), indent=2)
    if len(text) <= max_chars:
        return mapping
    return {
        "preview_json": text[:max_chars],
        "truncated": True,
        "original_char_count": len(text),
    }


def _message_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    items = [str(item) for item in value]
    if items and all(len(item) == 1 for item in items):
        return ["".join(items)]
    return items


def extract_stage_rows(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for stage in data.get("stages", []) or []:
        if not isinstance(stage, Mapping):
            continue
        rows.append(
            {
                "name": stage.get("name"),
                "status": stage.get("status"),
                "passed": stage.get("passed"),
                "elapsed_s": stage.get("elapsed_s"),
                "messages": _message_list(stage.get("messages")),
            }
        )
    return tuple(rows)


def extract_messages(data: Mapping[str, Any]) -> tuple[str, ...]:
    messages: list[str] = []

    for stage in data.get("stages", []) or []:
        if isinstance(stage, Mapping):
            for msg in _message_list(stage.get("messages")):
                messages.append(str(msg))

    for point in data.get("points", []) or []:
        if isinstance(point, Mapping) and point.get("status") in {"fail", "error"}:
            for msg in _message_list(point.get("messages")):
                messages.append(str(msg))

    for run in data.get("runs", []) or []:
        if isinstance(run, Mapping) and run.get("status") in {"fail", "error"}:
            for msg in _message_list(run.get("messages")):
                messages.append(str(msg))

    for trial in data.get("trials", []) or []:
        if isinstance(trial, Mapping) and trial.get("status") in {"fail", "error"}:
            for msg in _message_list(trial.get("messages")):
                messages.append(str(msg))

    return tuple(messages[:20])


def extract_metrics(data: Mapping[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), Mapping) else {}

    metrics: dict[str, Any] = {}

    for key in [
        "n_runs",
        "n_pass",
        "n_partial",
        "n_fail",
        "n_error",
        "n_points",
        "n_trials",
        "n_passed",
        "elapsed_s",
    ]:
        if key in data:
            metrics[key] = data[key]

    aggregate = metadata.get("aggregate") if isinstance(metadata, Mapping) else None
    if isinstance(aggregate, Mapping):
        for key in [
            "best",
            "n_points",
            "n_pass",
            "n_partial",
            "n_error",
            "finite_gain_fraction",
            "comparison_pass_count",
            "max_abs_rel_error_mean",
            "max_abs_rel_error_max",
        ]:
            if key in aggregate:
                metrics[f"aggregate_{key}"] = aggregate[key]

    fit_metrics = metadata.get("fit_metrics") if isinstance(metadata, Mapping) else None
    if isinstance(fit_metrics, Mapping):
        for key, value in fit_metrics.items():
            metrics[f"fit_{key}"] = value

    compression = metadata.get("compression") if isinstance(metadata, Mapping) else None
    if isinstance(compression, Mapping):
        for key, value in compression.items():
            metrics[f"compression_{key}"] = value

    # Pick up common stage-level metrics.
    for stage in data.get("stages", []) or []:
        if not isinstance(stage, Mapping):
            continue
        name = str(stage.get("name", "stage"))
        summary = stage.get("summary", {})
        if not isinstance(summary, Mapping):
            continue
        for key in [
            "max_gain_db",
            "max_gain_frequency_hz",
            "gain_rms_error_db",
            "gain_max_abs_error_db",
            "finite",
            "solver_function",
            "max_current_ratio",
            "pump_output_input_current_ratio",
            "rms_s21_error_db",
            "max_abs_s21_error_db",
        ]:
            if key in summary:
                metrics[f"{name}_{key}"] = summary[key]

    return metrics


def load_summary_record(path: Path, config: RunReportConfig) -> SummaryRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: summary JSON root is not an object")

    workflow = infer_workflow_from_path(path)
    status = parse_status(data.get("status"))

    artifact_paths = data.get("artifact_paths", {})
    if not isinstance(artifact_paths, Mapping):
        artifact_paths = {}

    cfg = data.get("config", {})
    if not isinstance(cfg, Mapping):
        cfg = {}

    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}

    stage_rows = extract_stage_rows(data)
    metrics = extract_metrics(data)

    return SummaryRecord(
        path=str(path),
        kind="summary_json",
        workflow=workflow,
        status=status,
        passed=data.get("passed") if isinstance(data.get("passed"), bool) else None,
        elapsed_s=safe_float(data.get("elapsed_s")),
        n_artifacts=len(artifact_paths),
        n_stages=len(stage_rows),
        n_points=int(data["n_points"]) if isinstance(data.get("n_points"), int) else None,
        n_runs=int(data["n_runs"]) if isinstance(data.get("n_runs"), int) else None,
        n_trials=int(data["n_trials"]) if isinstance(data.get("n_trials"), int) else None,
        config=compact_mapping(cfg, max_chars=config.max_json_chars),
        artifact_paths={str(k): str(v) for k, v in artifact_paths.items()},
        stage_rows=stage_rows,
        metrics=compact_mapping(metrics, max_chars=config.max_json_chars),
        metadata=compact_mapping(metadata, max_chars=config.max_json_chars),
        messages=extract_messages(data),
        raw_preview=compact_mapping(data, max_chars=config.max_json_chars),
    )


def load_markdown_record(path: Path, config: RunReportConfig) -> MarkdownRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    preview = text if config.include_markdown_bodies else text[: config.max_markdown_chars]
    truncated = len(text) > len(preview)

    return MarkdownRecord(
        path=str(path),
        workflow=infer_workflow_from_path(path),
        size_bytes=path.stat().st_size,
        preview=preview,
        truncated=truncated,
    )


def discover_summary_jsons(config: RunReportConfig) -> list[Path]:
    paths = [Path(p) for p in config.summary_json]

    if config.scan_root is not None:
        root = Path(config.scan_root)
        if config.recursive:
            paths.extend(root.rglob("*summary.json"))
        else:
            paths.extend(root.glob("*summary.json"))

    unique: dict[str, Path] = {}
    output_dir = Path(config.output_dir).resolve()
    for p in paths:
        resolved = p.resolve()
        if output_dir == resolved or output_dir in resolved.parents:
            continue
        if not config.include_report_summaries and p.name == "make_run_report_summary.json":
            continue
        if p.exists() and p.is_file():
            unique[str(resolved)] = p

    workflow_summaries: list[Path] = []
    for path in unique.values():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            workflow_summaries.append(path)
            continue
        if isinstance(data, Mapping) and "status" in data:
            workflow_summaries.append(path)

    return sorted(workflow_summaries, key=lambda p: str(p))


def discover_summary_markdowns(config: RunReportConfig) -> list[Path]:
    paths = [Path(p) for p in config.summary_md]

    if config.scan_root is not None:
        root = Path(config.scan_root)
        if config.recursive:
            paths.extend(root.rglob("*summary.md"))
        else:
            paths.extend(root.glob("*summary.md"))

    unique: dict[str, Path] = {}
    output_dir = Path(config.output_dir).resolve()
    for p in paths:
        resolved = p.resolve()
        if output_dir == resolved or output_dir in resolved.parents:
            continue
        if not config.include_report_summaries and p.name.startswith("make_run_report_"):
            continue
        if p.exists() and p.is_file():
            unique[str(resolved)] = p

    return sorted(unique.values(), key=lambda p: str(p))


def aggregate_records(records: Sequence[SummaryRecord]) -> dict[str, Any]:
    by_status = {status.value: 0 for status in RunStatus}
    by_workflow: dict[str, list[SummaryRecord]] = {}

    for record in records:
        by_status[record.status.value] = by_status.get(record.status.value, 0) + 1
        by_workflow.setdefault(record.workflow, []).append(record)

    elapsed = np.asarray(
        [r.elapsed_s for r in records if r.elapsed_s is not None],
        dtype=float,
    )

    workflow_summary = {}
    for workflow, rows in sorted(by_workflow.items()):
        workflow_summary[workflow] = {
            "n": len(rows),
            "statuses": {
                status.value: sum(1 for r in rows if r.status == status)
                for status in RunStatus
            },
            "latest_path": rows[-1].path if rows else None,
            "total_elapsed_s": float(np.nansum([r.elapsed_s or 0.0 for r in rows])),
        }

    failing = [r for r in records if r.status in {RunStatus.FAIL, RunStatus.ERROR}]
    unknown = [r for r in records if r.status == RunStatus.UNKNOWN]
    partial = [r for r in records if r.status == RunStatus.PARTIAL]

    return {
        "n_summaries": len(records),
        "by_status": by_status,
        "workflow_summary": workflow_summary,
        "n_fail_or_error": len(failing),
        "n_unknown": len(unknown),
        "n_fail_or_error_or_unknown": len(failing) + len(unknown),
        "n_partial": len(partial),
        "total_elapsed_s": float(np.nansum(elapsed)) if elapsed.size else None,
        "mean_elapsed_s": float(np.nanmean(elapsed)) if elapsed.size else None,
        "max_elapsed_s": float(np.nanmax(elapsed)) if elapsed.size else None,
        "failing_paths": [r.path for r in failing],
        "unknown_paths": [r.path for r in unknown],
        "partial_paths": [r.path for r in partial],
    }


def write_summary_csv(path: Path, records: Sequence[SummaryRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "workflow",
        "status",
        "passed",
        "elapsed_s",
        "n_artifacts",
        "n_stages",
        "n_points",
        "n_runs",
        "n_trials",
        "path",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "workflow": record.workflow,
                    "status": record.status.value,
                    "passed": record.passed,
                    "elapsed_s": record.elapsed_s,
                    "n_artifacts": record.n_artifacts,
                    "n_stages": record.n_stages,
                    "n_points": record.n_points,
                    "n_runs": record.n_runs,
                    "n_trials": record.n_trials,
                    "path": record.path,
                }
            )

    return path


def write_stage_csv(path: Path, records: Sequence[SummaryRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "workflow",
        "summary_status",
        "stage_name",
        "stage_status",
        "stage_passed",
        "stage_elapsed_s",
        "messages",
        "summary_path",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for record in records:
            for stage in record.stage_rows:
                writer.writerow(
                    {
                        "workflow": record.workflow,
                        "summary_status": record.status.value,
                        "stage_name": stage.get("name"),
                        "stage_status": stage.get("status"),
                        "stage_passed": stage.get("passed"),
                        "stage_elapsed_s": stage.get("elapsed_s"),
                        "messages": " | ".join(str(m) for m in stage.get("messages", [])[:5]),
                        "summary_path": record.path,
                    }
                )

    return path


def write_artifact_csv(path: Path, records: Sequence[SummaryRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "workflow",
        "status",
        "artifact_key",
        "artifact_path",
        "exists",
        "size_bytes",
        "summary_path",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for record in records:
            for key, artifact_path in record.artifact_paths.items():
                p = Path(artifact_path)
                writer.writerow(
                    {
                        "workflow": record.workflow,
                        "status": record.status.value,
                        "artifact_key": key,
                        "artifact_path": artifact_path,
                        "exists": p.exists(),
                        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
                        "summary_path": record.path,
                    }
                )

    return path


def markdown_status_emoji(status: RunStatus) -> str:
    return {
        RunStatus.PASS: "✅",
        RunStatus.PARTIAL: "⚠️",
        RunStatus.FAIL: "❌",
        RunStatus.ERROR: "❌",
        RunStatus.UNKNOWN: "❓",
    }[status]


def write_report_markdown(
    path: Path,
    *,
    config: RunReportConfig,
    records: Sequence[SummaryRecord],
    markdowns: Sequence[MarkdownRecord],
    aggregate: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {config.name}",
        "",
        f"- generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- scan root: `{config.scan_root}`",
        f"- summaries: `{len(records)}`",
        f"- markdown summaries: `{len(markdowns)}`",
        f"- status counts: `{aggregate.get('by_status')}`",
        f"- fail/error/unknown: `{aggregate.get('n_fail_or_error_or_unknown')}`",
        f"- partial: `{aggregate.get('n_partial')}`",
        "",
        "## Workflow status",
        "",
        "| workflow | status | elapsed s | stages | artifacts | points/runs/trials | summary |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for record in records:
        count = record.n_points
        if count is None:
            count = record.n_runs
        if count is None:
            count = record.n_trials

        lines.append(
            f"| {markdown_status_emoji(record.status)} `{record.workflow}` | "
            f"`{record.status.value}` | "
            f"{record.elapsed_s if record.elapsed_s is not None else ''} | "
            f"{record.n_stages} | "
            f"{record.n_artifacts} | "
            f"{count if count is not None else ''} | "
            f"`{record.path}` |"
        )

    failing = [r for r in records if r.status in {RunStatus.FAIL, RunStatus.ERROR}]
    unknown = [r for r in records if r.status == RunStatus.UNKNOWN]
    partial = [r for r in records if r.status == RunStatus.PARTIAL]

    if failing:
        lines += [
            "",
            "## Failures / errors",
            "",
        ]

        for record in failing:
            lines += [
                f"### {markdown_status_emoji(record.status)} {record.workflow}",
                "",
                f"- status: `{record.status.value}`",
                f"- path: `{record.path}`",
                "",
            ]
            if record.messages:
                lines.append("Messages:")
                for msg in record.messages[:10]:
                    lines.append(f"- {msg}")
                lines.append("")

    if partial:
        lines += [
            "",
            "## Partial runs",
            "",
        ]

        for record in partial:
            lines += [
                f"### {markdown_status_emoji(record.status)} {record.workflow}",
                "",
                f"- path: `{record.path}`",
                f"- elapsed: `{record.elapsed_s}`",
                "",
            ]

            if record.messages:
                for msg in record.messages[:8]:
                    lines.append(f"- {msg}")
                lines.append("")

    if unknown:
        lines += [
            "",
            "## Unknown-status summaries",
            "",
            "These summaries lack enough evidence for PASS/FAIL classification.",
            "",
        ]
        for record in unknown:
            lines.append(f"- `{record.workflow}`: `{record.path}`")

    lines += [
        "",
        "## Stage matrix",
        "",
        "| workflow | stage | status | elapsed s | messages |",
        "|---|---|---|---:|---|",
    ]

    for record in records:
        if not record.stage_rows:
            lines.append(
                f"| `{record.workflow}` | — | `{record.status.value}` | {record.elapsed_s or ''} | — |"
            )
            continue

        for stage in record.stage_rows:
            messages = "<br>".join(str(m) for m in stage.get("messages", [])[:2])
            lines.append(
                f"| `{record.workflow}` | `{stage.get('name')}` | "
                f"`{stage.get('status')}` | {stage.get('elapsed_s') or ''} | {messages} |"
            )

    lines += [
        "",
        "## Artifact coverage",
        "",
        "| workflow | artifact | exists | path |",
        "|---|---|---|---|",
    ]

    for record in records:
        for key, artifact_path in record.artifact_paths.items():
            exists = Path(artifact_path).exists()
            lines.append(
                f"| `{record.workflow}` | `{key}` | `{exists}` | `{artifact_path}` |"
            )

    lines += [
        "",
        "## Metrics preview",
        "",
    ]

    for record in records:
        if not record.metrics:
            continue

        lines += [
            f"### {record.workflow}",
            "",
            "```json",
            json.dumps(jsonify(record.metrics), indent=2)[:5000],
            "```",
            "",
        ]

    if markdowns:
        lines += [
            "",
            "## Markdown summary previews",
            "",
        ]

        for md in markdowns:
            lines += [
                f"### {md.workflow}",
                "",
                f"- path: `{md.path}`",
                f"- truncated: `{md.truncated}`",
                "",
                "```markdown",
                md.preview[: config.max_markdown_chars],
                "```",
                "",
            ]

    lines += [
        "",
        "## Report artifacts",
        "",
        "| key | path |",
        "|---|---|",
    ]

    for key, artifact_path in artifact_paths.items():
        lines.append(f"| `{key}` | `{artifact_path}` |")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report_html(markdown_path: Path, html_path: Path) -> Path:
    text = markdown_path.read_text(encoding="utf-8")

    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{markdown_path.stem}</title>
<style>
body {{
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 2rem auto;
  max-width: 1200px;
  line-height: 1.5;
  padding: 0 1rem;
}}
pre {{
  background: #f6f8fa;
  padding: 1rem;
  overflow-x: auto;
  border-radius: 8px;
}}
code {{
  background: #f6f8fa;
  padding: 0.1rem 0.25rem;
  border-radius: 4px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
}}
th, td {{
  border: 1px solid #ddd;
  padding: 0.4rem;
  vertical-align: top;
}}
th {{
  background: #f6f8fa;
}}
</style>
</head>
<body>
<pre>{escaped}</pre>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def write_plots(output_dir: Path, records: Sequence[SummaryRecord]) -> dict[str, str]:
    paths: dict[str, str] = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        err = output_dir / "plotting_unavailable.txt"
        err.write_text(str(exc), encoding="utf-8")
        return {"plotting_unavailable_txt": str(err)}

    if not records:
        return paths

    status_names = [s.value for s in RunStatus]
    counts = [sum(1 for r in records if r.status == s) for s in RunStatus]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    ax.bar(status_names, counts)
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")
    ax.set_title("Run status counts")
    ax.grid(True, axis="y")
    fig.tight_layout()
    p = output_dir / "run_status_counts.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths["status_counts_png"] = str(p)

    elapsed_records = [r for r in records if r.elapsed_s is not None]
    if elapsed_records:
        labels = [r.workflow for r in elapsed_records]
        values = [float(r.elapsed_s or 0.0) for r in elapsed_records]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
        y = np.arange(len(labels))
        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_title("Workflow elapsed time")
        ax.grid(True, axis="x")
        fig.tight_layout()
        p = output_dir / "workflow_elapsed_times.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths["elapsed_times_png"] = str(p)

    return paths


def export_report(
    *,
    config: RunReportConfig,
    records: Sequence[SummaryRecord],
    markdowns: Sequence[MarkdownRecord],
    output_dir: Path,
    elapsed_s: float,
    metadata: Mapping[str, Any],
) -> RunReportResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_records(records)
    artifacts: dict[str, str] = {}

    summary_csv = write_summary_csv(output_dir / "run_report_summaries.csv", records)
    stage_csv = write_stage_csv(output_dir / "run_report_stages.csv", records)
    artifact_csv = write_artifact_csv(output_dir / "run_report_artifacts.csv", records)

    artifacts["summaries_csv"] = str(summary_csv)
    artifacts["stages_csv"] = str(stage_csv)
    artifacts["artifacts_csv"] = str(artifact_csv)

    if config.make_plots:
        artifacts.update(write_plots(output_dir, records))

    report_md = output_dir / "run_report.md"
    write_report_markdown(
        report_md,
        config=config,
        records=records,
        markdowns=markdowns,
        aggregate=aggregate,
        artifact_paths=artifacts,
    )
    artifacts["report_md"] = str(report_md)

    report_html = write_report_html(report_md, output_dir / "run_report.html")
    artifacts["report_html"] = str(report_html)

    if not records:
        status = RunStatus.ERROR
    elif aggregate["n_fail_or_error"] > 0:
        status = RunStatus.ERROR
    elif aggregate["n_partial"] > 0 or aggregate["n_unknown"] > 0:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.PASS

    summary_json = output_dir / "make_run_report_summary.json"
    artifacts["summary_json"] = str(summary_json)

    result = RunReportResult(
        config=config,
        status=status,
        elapsed_s=elapsed_s,
        summaries=tuple(records),
        markdowns=tuple(markdowns),
        artifact_paths=artifacts,
        aggregate=aggregate,
        metadata=metadata,
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a consolidated TWPA run report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--scan-root", type=str, default="outputs")
    parser.add_argument("--summary-json", type=str, action="append", default=[])
    parser.add_argument("--summary-md", type=str, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run_report"))
    parser.add_argument("--name", type=str, default="TWPA Run Report")

    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--fail-on-partial", action="store_true")
    parser.add_argument("--include-markdown-bodies", action="store_true")
    parser.add_argument("--max-markdown-chars", type=int, default=12000)
    parser.add_argument("--max-json-chars", type=int, default=20000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-report-summaries",
        action="store_true",
        help="Include prior generated run-report summaries during recursive discovery.",
    )

    return parser


def resolve_config(args: argparse.Namespace) -> RunReportConfig:
    if args.scan_root is not None and args.scan_root != "" and not Path(args.scan_root).exists():
        if not args.summary_json and not args.summary_md:
            raise FileNotFoundError(args.scan_root)

    for path in args.summary_json:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    for path in args.summary_md:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    if args.max_markdown_chars <= 0:
        raise ValueError("--max-markdown-chars must be positive")

    if args.max_json_chars <= 0:
        raise ValueError("--max-json-chars must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. "
            "Use --overwrite or choose a new --output-dir."
        )

    scan_root = args.scan_root
    if scan_root == "":
        scan_root = None

    return RunReportConfig(
        scan_root=scan_root,
        summary_json=tuple(str(p) for p in args.summary_json),
        summary_md=tuple(str(p) for p in args.summary_md),
        output_dir=str(output_dir),
        name=str(args.name),
        recursive=not bool(args.no_recursive),
        fail_on_error=bool(args.fail_on_error),
        fail_on_partial=bool(args.fail_on_partial),
        include_markdown_bodies=bool(args.include_markdown_bodies),
        max_markdown_chars=int(args.max_markdown_chars),
        max_json_chars=int(args.max_json_chars),
        make_plots=not bool(args.no_plots),
        overwrite=bool(args.overwrite),
        include_report_summaries=bool(args.include_report_summaries),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[run-report] invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    if config.overwrite and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "python": sys.version,
        "script": "scripts/make_run_report.py",
    }

    records: list[SummaryRecord] = []
    markdowns: list[MarkdownRecord] = []

    summary_paths = discover_summary_jsons(config)
    markdown_paths = discover_summary_markdowns(config)

    for path in summary_paths:
        try:
            records.append(load_summary_record(path, config))
        except Exception as exc:
            error_record = SummaryRecord(
                path=str(path),
                kind="summary_json",
                workflow=infer_workflow_from_path(path),
                status=RunStatus.ERROR,
                passed=False,
                elapsed_s=None,
                n_artifacts=0,
                n_stages=0,
                n_points=None,
                n_runs=None,
                n_trials=None,
                config={},
                artifact_paths={},
                stage_rows=(),
                metrics={},
                metadata={},
                messages=(f"ERROR: failed to parse summary JSON: {type(exc).__name__}: {exc}",),
                raw_preview={"traceback": traceback.format_exc()},
            )
            records.append(error_record)

    for path in markdown_paths:
        try:
            markdowns.append(load_markdown_record(path, config))
        except Exception:
            markdowns.append(
                MarkdownRecord(
                    path=str(path),
                    workflow=infer_workflow_from_path(path),
                    size_bytes=0,
                    preview=traceback.format_exc(),
                    truncated=False,
                )
            )

    elapsed_s = time.perf_counter() - start

    result = export_report(
        config=config,
        records=records,
        markdowns=markdowns,
        output_dir=output_dir,
        elapsed_s=elapsed_s,
        metadata=metadata,
    )

    print()
    print(f"[run-report] status: {result.status.value}")
    print(f"[run-report] summaries: {len(result.summaries)}")
    print(f"[run-report] report MD: {result.artifact_paths.get('report_md')}")
    print(f"[run-report] report HTML: {result.artifact_paths.get('report_html')}")
    print(f"[run-report] summary JSON: {result.artifact_paths.get('summary_json')}")

    if config.fail_on_error and result.status == RunStatus.ERROR:
        return 1
    if config.fail_on_partial and result.status in {RunStatus.ERROR, RunStatus.PARTIAL}:
        return 1

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())
