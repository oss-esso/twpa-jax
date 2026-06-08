from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_wall_time_budget_one_shot_vs_batch")
CSV = ROOT / "one_shot_vs_batch_rows.csv"
OUT = ROOT / "corrected_accounting"
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
    rows = list(csv.DictReader(file))

rows = [r for r in rows if r.get("status") == "PASS"]

batch = [r for r in rows if r["mode"] == "batch_one_process"]
oneshot = [r for r in rows if r["mode"] == "one_shot_process_per_case"]

batch_wall = sorted(set(f(r["python_wall_time_s"]) for r in batch))
batch_wall = [x for x in batch_wall if x is not None]
batch_wall_s = batch_wall[0] if batch_wall else None

def first_val(rs, key):
    vals = [f(r.get(key)) for r in rs]
    vals = [v for v in vals if v is not None]
    return vals[0] if vals else None

batch_startup_s = first_val(batch, "startup_import_s")
batch_include_s = first_val(batch, "include_setup_s")
batch_outer_s = first_val(batch, "batch_outer_unattributed_s")
batch_runtime_sum_s = sum(f(r["runtime_s"]) or 0 for r in batch)
batch_case_sum_s = sum(f(r["case_wall_s"]) or 0 for r in batch)

batch_mode_accounting = {
    "mode": "batch_one_process",
    "n_cases": len(batch),
    "batch_wall_s": batch_wall_s,
    "startup_import_s": batch_startup_s,
    "include_setup_s": batch_include_s,
    "runtime_sum_s": batch_runtime_sum_s,
    "case_wall_sum_s": batch_case_sum_s,
    "outer_unattributed_s": batch_outer_s,
    "startup_pct_batch_wall": pct(batch_startup_s, batch_wall_s),
    "include_setup_pct_batch_wall": pct(batch_include_s, batch_wall_s),
    "runtime_sum_pct_batch_wall": pct(batch_runtime_sum_s, batch_wall_s),
    "case_wall_sum_pct_batch_wall": pct(batch_case_sum_s, batch_wall_s),
    "outer_unattributed_pct_batch_wall": pct(batch_outer_s, batch_wall_s),
}

# Batch case contribution rows: denominator is total batch wall.
batch_case_rows = []
for r in batch:
    runtime_s = f(r["runtime_s"])
    case_wall_s = f(r["case_wall_s"])
    batch_case_rows.append({
        "mode": "batch_one_process",
        "template": r["template"],
        "backend": r["backend"],
        "scale_value": r["scale_value"],
        "n_elements": r["n_elements"],
        "n_nodes": r["n_nodes"],
        "jc_tuple_count": r["jc_tuple_count"],
        "batch_wall_s": batch_wall_s,
        "runtime_s": runtime_s,
        "case_wall_s": case_wall_s,
        "runtime_pct_batch_wall": pct(runtime_s, batch_wall_s),
        "case_wall_pct_batch_wall": pct(case_wall_s, batch_wall_s),
    })

# One-shot rows: denominator is that row's true Python wall.
oneshot_rows = []
for r in oneshot:
    py_wall = f(r["python_wall_time_s"])
    runtime_s = f(r["runtime_s"])
    startup_s = f(r["startup_import_s"])
    include_s = f(r["include_setup_s"])
    outer_s = f(r["batch_outer_unattributed_s"])
    case_wall_s = f(r["case_wall_s"])

    oneshot_rows.append({
        "mode": "one_shot_process_per_case",
        "template": r["template"],
        "backend": r["backend"],
        "scale_value": r["scale_value"],
        "n_elements": r["n_elements"],
        "n_nodes": r["n_nodes"],
        "jc_tuple_count": r["jc_tuple_count"],
        "python_wall_s": py_wall,
        "startup_import_s": startup_s,
        "include_setup_s": include_s,
        "runtime_s": runtime_s,
        "case_wall_s": case_wall_s,
        "outer_unattributed_s": outer_s,
        "startup_pct_wall": pct(startup_s, py_wall),
        "include_setup_pct_wall": pct(include_s, py_wall),
        "runtime_pct_wall": pct(runtime_s, py_wall),
        "case_wall_pct_wall": pct(case_wall_s, py_wall),
        "outer_unattributed_pct_wall": pct(outer_s, py_wall),
    })

def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT / "batch_case_contributions.csv", batch_case_rows)
write_csv(OUT / "one_shot_case_wall_accounting.csv", oneshot_rows)

with (OUT / "batch_mode_accounting.json").open("w", encoding="utf-8") as file:
    json.dump(batch_mode_accounting, file, indent=2)

# Plot 1: batch total wall composition.
labels = ["Startup/import", "Include/setup", "Case/runtime sum", "Outer unattributed"]
values = [
    batch_startup_s or 0,
    batch_include_s or 0,
    batch_case_sum_s or 0,
    batch_outer_s or 0,
]

plt.figure()
plt.bar(labels, values)
plt.ylabel("Seconds")
plt.title("Batch total wall-time composition")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(PLOTS / "batch_total_wall_composition.png", dpi=180)
plt.close()

# Plot 2: batch case runtime contributions.
top_batch = sorted(batch_case_rows, key=lambda r: r["runtime_s"] or 0, reverse=True)
plt.figure(figsize=(10, max(5, 0.35 * len(top_batch))))
plt.barh(
    [f"{r['template']}\n{r['backend']}\nscale={r['scale_value']}" for r in top_batch],
    [r["runtime_s"] or 0 for r in top_batch],
)
plt.xlabel("Runtime inside batch (s)")
plt.title("Batch case runtime contributions")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "batch_case_runtime_contributions.png", dpi=180)
plt.close()

# Plot 3: one-shot wall composition for largest selected cases.
selected = [
    r for r in oneshot_rows
    if (
        (r["template"] == "rf_jtl_linear" and r["scale_value"] == "2376") or
        (r["template"] == "ethz_jtl_linear" and r["scale_value"] == "2048") or
        (r["template"] == "jtl_linear" and r["scale_value"] == "2048")
    )
]
selected = sorted(selected, key=lambda r: (r["template"], r["backend"]))

labels = [f"{r['template']}\n{r['backend']}\nscale={r['scale_value']}" for r in selected]
startup = [r["startup_import_s"] or 0 for r in selected]
include = [r["include_setup_s"] or 0 for r in selected]
runtime = [r["runtime_s"] or 0 for r in selected]
outer = [r["outer_unattributed_s"] or 0 for r in selected]

plt.figure(figsize=(11, max(5, 0.45 * len(labels))))
left = [0] * len(labels)
plt.barh(labels, startup, left=left, label="Startup/import")
left = [x + y for x, y in zip(left, startup)]
plt.barh(labels, include, left=left, label="Include/setup")
left = [x + y for x, y in zip(left, include)]
plt.barh(labels, runtime, left=left, label="Runtime")
left = [x + y for x, y in zip(left, runtime)]
plt.barh(labels, outer, left=left, label="Outer unattributed")
plt.xlabel("One-shot wall time (s)")
plt.title("One-shot wall composition for selected large cases")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(PLOTS / "one_shot_wall_composition_selected_large_cases.png", dpi=180)
plt.close()

# Plot 4: runtime vs elements, true runtime only.
plt.figure()
for key in sorted(set((r["template"], r["backend"]) for r in batch_case_rows)):
    data = [
        r for r in batch_case_rows
        if (r["template"], r["backend"]) == key
        and f(r["n_elements"]) is not None
        and r["runtime_s"] is not None
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
plt.title("Runtime scaling by template/backend")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(PLOTS / "runtime_vs_elements.png", dpi=180)
plt.close()

md = [
    "# Corrected wall-time accounting",
    "",
    "Data collection only.",
    "",
    "This version does not amortize the batch wall time as a denominator for individual cases. For batch mode, individual rows are reported as contributions to the total batch wall time.",
    "",
    "## Batch total accounting",
    "",
    "| field | value |",
    "|---|---:|",
]

for k, v in batch_mode_accounting.items():
    md.append(f"| {k} | {v} |")

md += [
    "",
    "## Batch case contributions",
    "",
    "| template | backend | scale | elements | runtime s | runtime % batch wall |",
    "|---|---|---:|---:|---:|---:|",
]

for r in sorted(batch_case_rows, key=lambda x: x["runtime_s"] or 0, reverse=True):
    md.append(
        f"| {r['template']} | {r['backend']} | {r['scale_value']} | {r['n_elements']} | "
        f"{r['runtime_s']} | {r['runtime_pct_batch_wall']} |"
    )

md += [
    "",
    "## One-shot case accounting",
    "",
    "| template | backend | scale | elements | wall s | runtime s | runtime % wall | include/setup % wall | outer unattributed % wall |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]

for r in sorted(oneshot_rows, key=lambda x: x["python_wall_s"] or 0, reverse=True):
    md.append(
        f"| {r['template']} | {r['backend']} | {r['scale_value']} | {r['n_elements']} | "
        f"{r['python_wall_s']} | {r['runtime_s']} | {r['runtime_pct_wall']} | "
        f"{r['include_setup_pct_wall']} | {r['outer_unattributed_pct_wall']} |"
    )

md += [
    "",
    "## Plots",
    "",
    "- `plots/batch_total_wall_composition.png`",
    "- `plots/batch_case_runtime_contributions.png`",
    "- `plots/one_shot_wall_composition_selected_large_cases.png`",
    "- `plots/runtime_vs_elements.png`",
]

report = OUT / "corrected_wall_time_accounting.md"
report.write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", OUT)
print("REPORT", report)
print("PLOTS", PLOTS)
