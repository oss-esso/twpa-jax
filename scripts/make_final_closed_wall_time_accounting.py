from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_wall_time_budget_one_shot_vs_batch")
CSV = ROOT / "one_shot_vs_batch_rows.csv"
OUT = ROOT / "final_closed_accounting"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

def f(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return None

def pct(num, den):
    if num is None or den in (None, 0):
        return None
    return 100.0 * num / den

with CSV.open("r", newline="", encoding="utf-8") as file:
    rows = [r for r in csv.DictReader(file) if r.get("status") == "PASS"]

batch = [r for r in rows if r["mode"] == "batch_one_process"]
oneshot = [r for r in rows if r["mode"] == "one_shot_process_per_case"]

def first_float(rs, key):
    vals = [f(r.get(key)) for r in rs]
    vals = [v for v in vals if v is not None]
    return vals[0] if vals else None

# Batch-level accounting.
batch_python_wall_s = first_float(batch, "python_wall_time_s")
batch_julia_script_wall_s = first_float(batch, "julia_script_total_wall_s")
batch_startup_s = first_float(batch, "startup_import_s")
batch_include_s = first_float(batch, "include_setup_s")
batch_outer_s = first_float(batch, "batch_outer_unattributed_s")
batch_runtime_sum_s = sum(f(r["runtime_s"]) or 0.0 for r in batch)
batch_case_wall_sum_s = sum(f(r["case_wall_s"]) or 0.0 for r in batch)

batch_python_outer_s = None
if batch_python_wall_s is not None and batch_julia_script_wall_s is not None:
    batch_python_outer_s = batch_python_wall_s - batch_julia_script_wall_s

batch_closed_residual_s = None
if batch_python_wall_s is not None:
    batch_closed_residual_s = (
        batch_python_wall_s
        - (batch_python_outer_s or 0.0)
        - (batch_startup_s or 0.0)
        - (batch_include_s or 0.0)
        - (batch_runtime_sum_s or 0.0)
    )

batch_closed_sum_s = (
    (batch_python_outer_s or 0.0)
    + (batch_startup_s or 0.0)
    + (batch_include_s or 0.0)
    + (batch_runtime_sum_s or 0.0)
    + (batch_closed_residual_s or 0.0)
)

batch_closed = {
    "mode": "batch_one_process",
    "n_cases": len(batch),
    "python_wall_s": batch_python_wall_s,
    "julia_script_wall_s": batch_julia_script_wall_s,
    "python_outer_overhead_s": batch_python_outer_s,
    "startup_import_s": batch_startup_s,
    "include_setup_s": batch_include_s,
    "runtime_sum_s": batch_runtime_sum_s,
    "case_wall_sum_s": batch_case_wall_sum_s,
    "closed_residual_s": batch_closed_residual_s,
    "closed_sum_s": batch_closed_sum_s,
    "closure_error_s": None if batch_python_wall_s is None else batch_python_wall_s - batch_closed_sum_s,
    "python_outer_overhead_pct_wall": pct(batch_python_outer_s, batch_python_wall_s),
    "startup_import_pct_wall": pct(batch_startup_s, batch_python_wall_s),
    "include_setup_pct_wall": pct(batch_include_s, batch_python_wall_s),
    "runtime_sum_pct_wall": pct(batch_runtime_sum_s, batch_python_wall_s),
    "case_wall_sum_pct_wall": pct(batch_case_wall_sum_s, batch_python_wall_s),
    "closed_residual_pct_wall": pct(batch_closed_residual_s, batch_python_wall_s),
}

# Per-case batch contributions: denominator is total batch Python wall.
batch_cases = []
for r in batch:
    runtime_s = f(r["runtime_s"])
    case_wall_s = f(r["case_wall_s"])
    batch_cases.append({
        "template": r["template"],
        "backend": r["backend"],
        "scale_value": r["scale_value"],
        "n_elements": r["n_elements"],
        "n_nodes": r["n_nodes"],
        "jc_tuple_count": r["jc_tuple_count"],
        "runtime_s": runtime_s,
        "case_wall_s": case_wall_s,
        "runtime_pct_batch_python_wall": pct(runtime_s, batch_python_wall_s),
        "case_wall_pct_batch_python_wall": pct(case_wall_s, batch_python_wall_s),
    })

# One-shot closed accounting.
oneshot_closed = []
for r in oneshot:
    py_wall = f(r["python_wall_time_s"])
    julia_wall = f(r["julia_script_total_wall_s"])
    python_outer = py_wall - julia_wall if py_wall is not None and julia_wall is not None else None

    startup = f(r["startup_import_s"])
    include = f(r["include_setup_s"])
    runtime = f(r["runtime_s"])
    case_wall = f(r["case_wall_s"])
    julia_outer = f(r["batch_outer_unattributed_s"])

    oneshot_closed.append({
        "template": r["template"],
        "backend": r["backend"],
        "scale_value": r["scale_value"],
        "n_elements": r["n_elements"],
        "n_nodes": r["n_nodes"],
        "jc_tuple_count": r["jc_tuple_count"],
        "python_wall_s": py_wall,
        "julia_script_wall_s": julia_wall,
        "python_outer_overhead_s": python_outer,
        "startup_import_s": startup,
        "include_setup_s": include,
        "runtime_s": runtime,
        "case_wall_s": case_wall,
        "julia_outer_unattributed_s": julia_outer,
        "python_outer_overhead_pct_wall": pct(python_outer, py_wall),
        "startup_import_pct_wall": pct(startup, py_wall),
        "include_setup_pct_wall": pct(include, py_wall),
        "runtime_pct_wall": pct(runtime, py_wall),
        "case_wall_pct_wall": pct(case_wall, py_wall),
        "julia_outer_unattributed_pct_wall": pct(julia_outer, py_wall),
    })

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT / "batch_case_contributions_closed.csv", batch_cases)
write_csv(OUT / "one_shot_closed_accounting.csv", oneshot_closed)

with (OUT / "batch_closed_accounting.json").open("w", encoding="utf-8") as file:
    json.dump(batch_closed, file, indent=2)

# Plot 1: batch closed wall composition.
labels = [
    "Python outer",
    "Startup/import",
    "Include/setup",
    "Runtime sum",
    "Closed residual",
]
values = [
    batch_python_outer_s or 0.0,
    batch_startup_s or 0.0,
    batch_include_s or 0.0,
    batch_runtime_sum_s or 0.0,
    batch_closed_residual_s or 0.0,
]

plt.figure()
plt.bar(labels, values)
plt.ylabel("Seconds")
plt.title("Closed batch wall-time composition")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(PLOTS / "closed_batch_wall_composition.png", dpi=180)
plt.close()

# Plot 2: batch runtime contributions.
top_batch = sorted(batch_cases, key=lambda r: r["runtime_s"] or 0.0, reverse=True)
plt.figure(figsize=(10, max(5, 0.35 * len(top_batch))))
plt.barh(
    [f"{r['template']}\n{r['backend']}\nscale={r['scale_value']}" for r in top_batch],
    [r["runtime_s"] or 0.0 for r in top_batch],
)
plt.xlabel("Runtime inside batch (s)")
plt.title("Batch case runtime contributions")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "batch_case_runtime_contributions.png", dpi=180)
plt.close()

# Plot 3: closed one-shot wall composition for largest selected cases.
selected = [
    r for r in oneshot_closed
    if (
        (r["template"] == "rf_jtl_linear" and r["scale_value"] == "2376") or
        (r["template"] == "ethz_jtl_linear" and r["scale_value"] == "2048") or
        (r["template"] == "jtl_linear" and r["scale_value"] == "2048")
    )
]
selected = sorted(selected, key=lambda r: (r["template"], r["backend"]))

row_labels = [f"{r['template']}\n{r['backend']}\nscale={r['scale_value']}" for r in selected]
python_outer = [r["python_outer_overhead_s"] or 0.0 for r in selected]
startup = [r["startup_import_s"] or 0.0 for r in selected]
include = [r["include_setup_s"] or 0.0 for r in selected]
runtime = [r["runtime_s"] or 0.0 for r in selected]
julia_outer = [r["julia_outer_unattributed_s"] or 0.0 for r in selected]

plt.figure(figsize=(11, max(5, 0.45 * len(row_labels))))
left = [0.0] * len(row_labels)
plt.barh(row_labels, python_outer, left=left, label="Python outer")
left = [a + b for a, b in zip(left, python_outer)]
plt.barh(row_labels, startup, left=left, label="Startup/import")
left = [a + b for a, b in zip(left, startup)]
plt.barh(row_labels, include, left=left, label="Include/setup")
left = [a + b for a, b in zip(left, include)]
plt.barh(row_labels, runtime, left=left, label="Runtime")
left = [a + b for a, b in zip(left, runtime)]
plt.barh(row_labels, julia_outer, left=left, label="Julia outer")
plt.xlabel("One-shot wall time (s)")
plt.title("Closed one-shot wall composition")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(PLOTS / "closed_one_shot_wall_composition_selected_large_cases.png", dpi=180)
plt.close()

# Plot 4: runtime scaling by elements.
plt.figure()
for key in sorted(set((r["template"], r["backend"]) for r in batch_cases)):
    data = [
        r for r in batch_cases
        if (r["template"], r["backend"]) == key
        and r["runtime_s"] is not None
        and f(r["n_elements"]) is not None
    ]
    if len(data) < 2:
        continue
    data = sorted(data, key=lambda r: f(r["n_elements"]))
    plt.plot(
        [f(r["n_elements"]) for r in data],
        [r["runtime_s"] for r in data],
        marker="o",
        label=f"{key[0]} / {key[1]}",
    )

plt.xlabel("Elements")
plt.ylabel("Runtime (s)")
plt.title("Runtime scaling by topology size")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(PLOTS / "runtime_vs_elements.png", dpi=180)
plt.close()

# Markdown report.
md = [
    "# Final closed wall-time accounting",
    "",
    "Data collection only. All percentages use the correct wall-time denominator.",
    "",
    "## Batch closed accounting",
    "",
    "| field | value |",
    "|---|---:|",
]

for k, v in batch_closed.items():
    md.append(f"| {k} | {v} |")

md += [
    "",
    "## Batch case runtime contributions",
    "",
    "| template | backend | scale | elements | runtime s | runtime % batch wall |",
    "|---|---|---:|---:|---:|---:|",
]

for r in sorted(batch_cases, key=lambda x: x["runtime_s"] or 0.0, reverse=True):
    md.append(
        f"| {r['template']} | {r['backend']} | {r['scale_value']} | {r['n_elements']} | "
        f"{r['runtime_s']} | {r['runtime_pct_batch_python_wall']} |"
    )

md += [
    "",
    "## One-shot closed accounting",
    "",
    "| template | backend | scale | elements | wall s | runtime s | runtime % wall | Python outer % | include/setup % | Julia outer % |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for r in sorted(oneshot_closed, key=lambda x: x["python_wall_s"] or 0.0, reverse=True):
    md.append(
        f"| {r['template']} | {r['backend']} | {r['scale_value']} | {r['n_elements']} | "
        f"{r['python_wall_s']} | {r['runtime_s']} | {r['runtime_pct_wall']} | "
        f"{r['python_outer_overhead_pct_wall']} | {r['include_setup_pct_wall']} | "
        f"{r['julia_outer_unattributed_pct_wall']} |"
    )

md += [
    "",
    "## Plots",
    "",
    "- `plots/closed_batch_wall_composition.png`",
    "- `plots/batch_case_runtime_contributions.png`",
    "- `plots/closed_one_shot_wall_composition_selected_large_cases.png`",
    "- `plots/runtime_vs_elements.png`",
]

report = OUT / "final_closed_wall_time_accounting.md"
report.write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", OUT)
print("REPORT", report)
print("PLOTS", PLOTS)
