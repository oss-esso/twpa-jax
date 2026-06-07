from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOTS = {
    "batch_hbsolve": Path(r"D:\Projects\Thesis\outputs\benchmarks\jc3m_k_benchmark_batch"),
    "batch_jtl_direct": Path(r"D:\Projects\Thesis\outputs\benchmarks\jc3m_m4_benchmark_batch_jtl_hblinsolve"),
    "batch_jtl_rf_direct": Path(r"D:\Projects\Thesis\outputs\benchmarks\jc3m_m5_benchmark_batch_jtl_rf_hblinsolve"),
    "batch_jtl_rf_ethz_direct": Path(r"D:\Projects\Thesis\outputs\benchmarks\jc3m_m6_benchmark_batch_jtl_rf_ethz_hblinsolve"),
}

OUT = Path(r"D:\Projects\Thesis\outputs\benchmarks\jc3m_m7_direct_linear_backend_report")
OUT.mkdir(parents=True, exist_ok=True)

def load_rows(root: Path) -> list[dict[str, str]]:
    path = root / "benchmark_runs.csv"
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def case_key(row: dict[str, str]) -> str:
    for key in ("case", "case_name", "name", "benchmark_case"):
        if key in row and row[key]:
            return row[key]
    run_dir = row.get("run_dir", "") or row.get("output_dir", "")
    if run_dir:
        return Path(run_dir).name.split("__rep")[0]
    return "UNKNOWN"

def float_field(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] not in ("", None):
            try:
                return float(row[name])
            except ValueError:
                pass
    return None

summary_rows = []
for label, root in ROOTS.items():
    rows = load_rows(root)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(case_key(row), []).append(row)

    for case, case_rows in sorted(groups.items()):
        wall = [float_field(r, "python_wall_time_s", "python_runtime_s", "runtime_s") for r in case_rows]
        wall = [x for x in wall if x is not None]
        status_counts: dict[str, int] = {}
        for r in case_rows:
            status_counts[r.get("status", "UNKNOWN")] = status_counts.get(r.get("status", "UNKNOWN"), 0) + 1

        summary_rows.append({
            "mode": label,
            "case": case,
            "runs": len(case_rows),
            "python_mean_s": mean(wall) if wall else None,
            "statuses": status_counts,
        })

# direct backend status details
status_rows = []
for label, root in ROOTS.items():
    runs_root = root / "runs"
    if not runs_root.exists():
        continue
    for status_path in sorted(runs_root.glob("*/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status_rows.append({
            "mode": label,
            "run": status_path.parent.name,
            "status": status.get("status"),
            "backend": status.get("jc_backend"),
            "setup_cache_integration": (status.get("cache_telemetry") or {}).get("setup_cache_integration"),
            "runtime_s": status.get("runtime_s"),
        })

report = {
    "phase": "JC-3M-M7 direct linear backend final report",
    "decision": "Stop rollout after JTL, RF-JTL, and ETHZ-JTL direct hblinsolve backends. Do not patch lumped_jpa_linear without a separate equivalence probe.",
    "summary_rows": summary_rows,
    "status_rows": status_rows,
}

(OUT / "direct_linear_backend_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

with (OUT / "direct_linear_backend_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["mode", "case", "runs", "python_mean_s", "statuses"])
    writer.writeheader()
    writer.writerows(summary_rows)

with (OUT / "direct_backend_statuses.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["mode", "run", "status", "backend", "setup_cache_integration", "runtime_s"])
    writer.writeheader()
    writer.writerows(status_rows)

md = [
    "# JC-3M-M7 Direct Linear Backend Report",
    "",
    "## Decision",
    "",
    report["decision"],
    "",
    "## Summary",
    "",
    "| mode | case | runs | python_mean_s | statuses |",
    "|---|---|---:|---:|---|",
]
for r in summary_rows:
    md.append(f"| {r['mode']} | {r['case']} | {r['runs']} | {r['python_mean_s']} | {r['statuses']} |")

md += [
    "",
    "## Direct backend statuses",
    "",
    "| mode | run | status | backend | setup_cache_integration | runtime_s |",
    "|---|---|---|---|---|---:|",
]
for r in status_rows:
    if r["setup_cache_integration"] and "hblinsolve_direct" in str(r["setup_cache_integration"]):
        md.append(
            f"| {r['mode']} | {r['run']} | {r['status']} | {r['backend']} | "
            f"{r['setup_cache_integration']} | {r['runtime_s']} |"
        )

(OUT / "direct_linear_backend_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", OUT)
