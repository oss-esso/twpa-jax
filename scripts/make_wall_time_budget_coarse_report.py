from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

WORKSPACE = Path(r"D:\Projects\Thesis")
OUT = WORKSPACE / "outputs"
REPORT = OUT / "jc_profiles" / "jc3m_wall_time_budget_report"
REPORT.mkdir(parents=True, exist_ok=True)

BENCHMARKS = {
    "oneshot_hbsolve": OUT / "benchmarks" / "jc3m_k_benchmark_oneshot_baseline" / "benchmark_runs.csv",
    "batch_hbsolve": OUT / "benchmarks" / "jc3m_k_benchmark_batch" / "benchmark_runs.csv",
    "batch_jtl_direct": OUT / "benchmarks" / "jc3m_m4_benchmark_batch_jtl_hblinsolve" / "benchmark_runs.csv",
    "batch_jtl_rf_direct": OUT / "benchmarks" / "jc3m_m5_benchmark_batch_jtl_rf_hblinsolve" / "benchmark_runs.csv",
    "batch_jtl_rf_ethz_direct": OUT / "benchmarks" / "jc3m_m6_benchmark_batch_jtl_rf_ethz_hblinsolve" / "benchmark_runs.csv",
}

def parse_float(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return None

def case_name(row):
    for key in ("case", "case_name", "name", "benchmark_case"):
        if row.get(key):
            return row[key]
    run_dir = row.get("run_dir") or row.get("output_dir") or ""
    if run_dir:
        return Path(run_dir).name.split("__rep")[0]
    return "ALL_OR_UNKNOWN"

rows_out = []

for mode, path in BENCHMARKS.items():
    if not path.exists():
        print("MISSING", path)
        continue

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        py_wall = parse_float(row.get("python_wall_time_s"))
        julia_wall = parse_float(row.get("julia_status_runtime_s"))
        hbsolve = parse_float(row.get("hbsolve_s"))

        out = {
            "mode": mode,
            "case": case_name(row),
            "status": row.get("status", ""),
            "python_wall_time_s": py_wall,
            "julia_status_runtime_s": julia_wall,
            "hbsolve_s": hbsolve,
            "hbsolve_pct_python_wall": (100.0 * hbsolve / py_wall) if hbsolve is not None and py_wall and py_wall > 0 else None,
            "hbsolve_pct_julia_wall": (100.0 * hbsolve / julia_wall) if hbsolve is not None and julia_wall and julia_wall > 0 else None,
            "python_overhead_s": (py_wall - julia_wall) if py_wall is not None and julia_wall is not None else None,
            "python_overhead_pct_python_wall": (100.0 * (py_wall - julia_wall) / py_wall) if py_wall and julia_wall is not None and py_wall > 0 else None,
        }
        rows_out.append(out)

# Write raw rows.
raw_path = REPORT / "wall_time_budget_coarse_rows.csv"
with raw_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "mode",
            "case",
            "status",
            "python_wall_time_s",
            "julia_status_runtime_s",
            "hbsolve_s",
            "hbsolve_pct_python_wall",
            "hbsolve_pct_julia_wall",
            "python_overhead_s",
            "python_overhead_pct_python_wall",
        ],
    )
    writer.writeheader()
    writer.writerows(rows_out)

# Aggregate by mode/case.
groups = {}
for r in rows_out:
    groups.setdefault((r["mode"], r["case"]), []).append(r)

summary = []
for (mode, case), rs in sorted(groups.items()):
    def avg(key):
        vals = [r[key] for r in rs if r[key] is not None]
        return mean(vals) if vals else None

    summary.append({
        "mode": mode,
        "case": case,
        "n": len(rs),
        "python_wall_mean_s": avg("python_wall_time_s"),
        "julia_wall_mean_s": avg("julia_status_runtime_s"),
        "hbsolve_mean_s": avg("hbsolve_s"),
        "hbsolve_pct_python_wall_mean": avg("hbsolve_pct_python_wall"),
        "hbsolve_pct_julia_wall_mean": avg("hbsolve_pct_julia_wall"),
        "python_overhead_pct_mean": avg("python_overhead_pct_python_wall"),
    })

summary_path = REPORT / "wall_time_budget_coarse_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "mode",
            "case",
            "n",
            "python_wall_mean_s",
            "julia_wall_mean_s",
            "hbsolve_mean_s",
            "hbsolve_pct_python_wall_mean",
            "hbsolve_pct_julia_wall_mean",
            "python_overhead_pct_mean",
        ],
    )
    writer.writeheader()
    writer.writerows(summary)

md = [
    "# Coarse Wall-Time Budget",
    "",
    "This report uses existing benchmark artifacts only.",
    "",
    "It can show `hbsolve_s` as a share of Python wall time and Julia runtime, but it cannot yet split `hbsolve_s` into the five internal HB blocks.",
    "",
    "| mode | case | n | python wall mean s | julia wall mean s | hbsolve mean s | hbsolve % python wall | hbsolve % julia wall | python overhead % |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]

for r in summary:
    md.append(
        f"| {r['mode']} | {r['case']} | {r['n']} | "
        f"{r['python_wall_mean_s']} | {r['julia_wall_mean_s']} | {r['hbsolve_mean_s']} | "
        f"{r['hbsolve_pct_python_wall_mean']} | {r['hbsolve_pct_julia_wall_mean']} | "
        f"{r['python_overhead_pct_mean']} |"
    )

(REPORT / "wall_time_budget_coarse_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", REPORT)
print("summary:", summary_path)
print("raw:", raw_path)
