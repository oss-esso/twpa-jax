from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

WORKSPACE = Path(r"D:\Projects\Thesis")
HARMONIA = WORKSPACE / "Harmonia.jl"
OUT = WORKSPACE / "outputs" / "jc_profiles" / "jc3m_wall_time_budget_one_shot_vs_batch"
OUT.mkdir(parents=True, exist_ok=True)

PLAN = [
    ("jtl_linear", "hbsolve", 64),
    ("jtl_linear", "hbsolve", 256),
    ("jtl_linear", "hbsolve", 2048),
    ("jtl_linear", "hblinsolve_direct", 64),
    ("jtl_linear", "hblinsolve_direct", 256),
    ("jtl_linear", "hblinsolve_direct", 2048),

    ("rf_jtl_linear", "hbsolve", 64),
    ("rf_jtl_linear", "hbsolve", 256),
    ("rf_jtl_linear", "hbsolve", 2376),
    ("rf_jtl_linear", "hblinsolve_direct", 64),
    ("rf_jtl_linear", "hblinsolve_direct", 256),
    ("rf_jtl_linear", "hblinsolve_direct", 2376),

    ("ethz_jtl_linear", "hbsolve", 32),
    ("ethz_jtl_linear", "hbsolve", 128),
    ("ethz_jtl_linear", "hbsolve", 2048),
    ("ethz_jtl_linear", "hblinsolve_direct", 32),
    ("ethz_jtl_linear", "hblinsolve_direct", 128),
    ("ethz_jtl_linear", "hblinsolve_direct", 2048),

    ("lumped_jpa_linear", "hbsolve", 0),
]

def run(cmd: list[str], cwd: Path) -> tuple[int, float, str, str]:
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.returncode, time.perf_counter() - t0, p.stdout, p.stderr

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def f(x):
    if x in (None, ""):
        return None
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return None

def pct(num, den):
    if num is None or den in (None, 0):
        return None
    return 100.0 * num / den

rows = []

# A. batch run: all cases in one Julia process
batch_dir = OUT / "batch_all"
cmd = [
    "julia",
    "--project=.",
    "experiments/wall_time_budget/run_template_wall_time_budget.jl",
    "--outdir",
    str(batch_dir),
    "--repetitions",
    "1",
]
rc, py_wall, stdout, stderr = run(cmd, HARMONIA)
(batch_dir / "outer_stdout.log").write_text(stdout, encoding="utf-8")
(batch_dir / "outer_stderr.log").write_text(stderr, encoding="utf-8")

if rc != 0:
    raise SystemExit(f"Batch run failed rc={rc}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

batch_manifest = read_json(batch_dir / "wall_time_budget_manifest.json")
for r in read_csv(batch_dir / "wall_time_budget_rows.csv"):
    if r["rep"] != "1":
        continue

    runtime_s = f(r["runtime_s"])
    case_wall_s = f(r["case_total_s"])

    rows.append({
        "mode": "batch_one_process",
        "template": r["template"],
        "backend": r["backend"],
        "scale_value": r["scale_value"],
        "rep": r["rep"],
        "status": r["status"],
        "n_elements": r["n_elements"],
        "n_nodes": r["n_nodes"],
        "jc_tuple_count": r["jc_tuple_count"],

        "python_wall_time_s": py_wall,
        "julia_script_total_wall_s": batch_manifest.get("script_total_wall_s"),
        "startup_import_s": batch_manifest.get("startup_import_s"),
        "include_setup_s": batch_manifest.get("include_setup_s"),
        "batch_outer_unattributed_s": batch_manifest.get("batch_outer_unattributed_s"),

        "case_wall_s": case_wall_s,
        "runtime_s": runtime_s,
        "runtime_pct_case_wall": pct(runtime_s, case_wall_s),

        # Batch-level shared denominators.
        "case_wall_pct_python_wall": pct(case_wall_s, py_wall),
        "runtime_pct_python_wall": pct(runtime_s, py_wall),
        "startup_import_pct_python_wall": pct(batch_manifest.get("startup_import_s"), py_wall),
        "include_setup_pct_python_wall": pct(batch_manifest.get("include_setup_s"), py_wall),
        "batch_outer_unattributed_pct_python_wall": pct(batch_manifest.get("batch_outer_unattributed_s"), py_wall),
        "failure_reason": r.get("failure_reason", ""),
    })

# B. one-shot runs: one Julia process per case
for template, backend, scale in PLAN:
    run_dir = OUT / "one_shot" / f"{template}__{backend}__scale{scale}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "julia",
        "--project=.",
        "experiments/wall_time_budget/run_template_wall_time_budget.jl",
        "--outdir",
        str(run_dir),
        "--template",
        template,
        "--backend",
        backend,
        "--scale",
        str(scale),
        "--repetitions",
        "1",
    ]

    rc, py_wall, stdout, stderr = run(cmd, HARMONIA)
    (run_dir / "outer_stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "outer_stderr.log").write_text(stderr, encoding="utf-8")

    if rc != 0:
        rows.append({
            "mode": "one_shot_process_per_case",
            "template": template,
            "backend": backend,
            "scale_value": scale,
            "rep": "1",
            "status": "FAIL",
            "python_wall_time_s": py_wall,
            "failure_reason": stderr[-1000:],
        })
        continue

    manifest = read_json(run_dir / "wall_time_budget_manifest.json")
    case_rows = [r for r in read_csv(run_dir / "wall_time_budget_rows.csv") if r["rep"] == "1"]
    if not case_rows:
        rows.append({
            "mode": "one_shot_process_per_case",
            "template": template,
            "backend": backend,
            "scale_value": scale,
            "rep": "1",
            "status": "FAIL_NO_ROW",
            "python_wall_time_s": py_wall,
            "failure_reason": "No rep=1 row found",
        })
        continue

    r = case_rows[0]
    runtime_s = f(r["runtime_s"])
    case_wall_s = f(r["case_total_s"])

    rows.append({
        "mode": "one_shot_process_per_case",
        "template": r["template"],
        "backend": r["backend"],
        "scale_value": r["scale_value"],
        "rep": r["rep"],
        "status": r["status"],
        "n_elements": r["n_elements"],
        "n_nodes": r["n_nodes"],
        "jc_tuple_count": r["jc_tuple_count"],

        "python_wall_time_s": py_wall,
        "julia_script_total_wall_s": manifest.get("script_total_wall_s"),
        "startup_import_s": manifest.get("startup_import_s"),
        "include_setup_s": manifest.get("include_setup_s"),
        "batch_outer_unattributed_s": manifest.get("batch_outer_unattributed_s"),

        "case_wall_s": case_wall_s,
        "runtime_s": runtime_s,
        "runtime_pct_case_wall": pct(runtime_s, case_wall_s),

        "case_wall_pct_python_wall": pct(case_wall_s, py_wall),
        "runtime_pct_python_wall": pct(runtime_s, py_wall),
        "startup_import_pct_python_wall": pct(manifest.get("startup_import_s"), py_wall),
        "include_setup_pct_python_wall": pct(manifest.get("include_setup_s"), py_wall),
        "batch_outer_unattributed_pct_python_wall": pct(manifest.get("batch_outer_unattributed_s"), py_wall),
        "failure_reason": r.get("failure_reason", ""),
    })

csv_path = OUT / "one_shot_vs_batch_rows.csv"
fieldnames = sorted(set().union(*(row.keys() for row in rows)))
with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Markdown summary: only successful rows.
md = [
    "# One-shot vs batch wall-time budget",
    "",
    "Data collection only. Percentages use Python wall time as denominator.",
    "",
    "| mode | template | backend | scale | elements | Python wall s | runtime s | runtime % wall | case wall % wall | include/setup % wall |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    if row.get("status") != "PASS":
        continue
    md.append(
        f"| {row.get('mode')} | {row.get('template')} | {row.get('backend')} | {row.get('scale_value')} | "
        f"{row.get('n_elements', '')} | {row.get('python_wall_time_s', '')} | {row.get('runtime_s', '')} | "
        f"{row.get('runtime_pct_python_wall', '')} | {row.get('case_wall_pct_python_wall', '')} | "
        f"{row.get('include_setup_pct_python_wall', '')} |"
    )

report_path = OUT / "one_shot_vs_batch_report.md"
report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", csv_path)
print("WROTE", report_path)
