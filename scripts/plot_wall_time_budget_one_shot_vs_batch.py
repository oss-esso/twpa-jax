from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_wall_time_budget_one_shot_vs_batch")
CSV = ROOT / "one_shot_vs_batch_rows.csv"
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def parse_float(x):
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

with CSV.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

pass_rows = [r for r in rows if r.get("status") == "PASS"]

batch_rows = [r for r in pass_rows if r.get("mode") == "batch_one_process"]
oneshot_rows = [r for r in pass_rows if r.get("mode") == "one_shot_process_per_case"]

# Batch raw wall is repeated on every row. Use one value and amortize it.
batch_wall_values = sorted(set(parse_float(r["python_wall_time_s"]) for r in batch_rows))
batch_wall_values = [x for x in batch_wall_values if x is not None]
batch_total_wall_s = batch_wall_values[0] if batch_wall_values else None
batch_n = len(batch_rows)
batch_amortized_wall_s = batch_total_wall_s / batch_n if batch_total_wall_s and batch_n else None

# Shared batch components from any batch row.
def first_batch_value(key):
    vals = [parse_float(r.get(key)) for r in batch_rows]
    vals = [v for v in vals if v is not None]
    return vals[0] if vals else None

batch_startup_s = first_batch_value("startup_import_s")
batch_include_s = first_batch_value("include_setup_s")
batch_outer_s = first_batch_value("batch_outer_unattributed_s")

batch_startup_amortized_s = batch_startup_s / batch_n if batch_startup_s is not None and batch_n else None
batch_include_amortized_s = batch_include_s / batch_n if batch_include_s is not None and batch_n else None
batch_outer_amortized_s = batch_outer_s / batch_n if batch_outer_s is not None and batch_n else None

normalized = []

for r in pass_rows:
    runtime_s = parse_float(r.get("runtime_s"))
    case_wall_s = parse_float(r.get("case_wall_s"))
    py_wall_s = parse_float(r.get("python_wall_time_s"))

    if r["mode"] == "batch_one_process":
        effective_wall_s = batch_amortized_wall_s
        startup_s = batch_startup_amortized_s
        include_s = batch_include_amortized_s
        outer_s = batch_outer_amortized_s
        mode_label = "batch amortized"
    else:
        effective_wall_s = py_wall_s
        startup_s = parse_float(r.get("startup_import_s"))
        include_s = parse_float(r.get("include_setup_s"))
        outer_s = parse_float(r.get("batch_outer_unattributed_s"))
        mode_label = "one-shot"

    runtime_remainder_s = None
    if effective_wall_s is not None:
        parts = [x for x in [startup_s, include_s, runtime_s, outer_s] if x is not None]
        runtime_remainder_s = effective_wall_s - sum(parts)

    normalized.append({
        "mode": r["mode"],
        "mode_label": mode_label,
        "template": r["template"],
        "backend": r["backend"],
        "scale_value": r["scale_value"],
        "n_elements": parse_float(r.get("n_elements")),
        "n_nodes": parse_float(r.get("n_nodes")),
        "jc_tuple_count": parse_float(r.get("jc_tuple_count")),
        "raw_python_wall_s": py_wall_s,
        "effective_wall_s": effective_wall_s,
        "runtime_s": runtime_s,
        "startup_s": startup_s,
        "include_setup_s": include_s,
        "outer_unattributed_s": outer_s,
        "wall_remainder_s": runtime_remainder_s,
        "runtime_pct_effective_wall": pct(runtime_s, effective_wall_s),
        "startup_pct_effective_wall": pct(startup_s, effective_wall_s),
        "include_setup_pct_effective_wall": pct(include_s, effective_wall_s),
        "outer_unattributed_pct_effective_wall": pct(outer_s, effective_wall_s),
    })

norm_csv = ROOT / "one_shot_vs_batch_normalized_rows.csv"
fieldnames = [
    "mode", "mode_label", "template", "backend", "scale_value",
    "n_elements", "n_nodes", "jc_tuple_count",
    "raw_python_wall_s", "effective_wall_s",
    "runtime_s", "startup_s", "include_setup_s", "outer_unattributed_s", "wall_remainder_s",
    "runtime_pct_effective_wall", "startup_pct_effective_wall",
    "include_setup_pct_effective_wall", "outer_unattributed_pct_effective_wall",
]
with norm_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(normalized)

# Summary by mode/template/backend/scale.
summary = []
for key in sorted(set((r["mode_label"], r["template"], r["backend"], r["scale_value"]) for r in normalized)):
    rs = [r for r in normalized if (r["mode_label"], r["template"], r["backend"], r["scale_value"]) == key]
    def avg(k):
        vals = [r[k] for r in rs if r[k] is not None]
        return mean(vals) if vals else None

    summary.append({
        "mode_label": key[0],
        "template": key[1],
        "backend": key[2],
        "scale_value": key[3],
        "n": len(rs),
        "effective_wall_s": avg("effective_wall_s"),
        "runtime_s": avg("runtime_s"),
        "startup_s": avg("startup_s"),
        "include_setup_s": avg("include_setup_s"),
        "outer_unattributed_s": avg("outer_unattributed_s"),
        "runtime_pct_effective_wall": avg("runtime_pct_effective_wall"),
        "startup_pct_effective_wall": avg("startup_pct_effective_wall"),
        "include_setup_pct_effective_wall": avg("include_setup_pct_effective_wall"),
        "outer_unattributed_pct_effective_wall": avg("outer_unattributed_pct_effective_wall"),
    })

summary_csv = ROOT / "one_shot_vs_batch_normalized_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

# Plot 1: effective wall time by row.
top = sorted(
    [r for r in summary if r["effective_wall_s"] is not None],
    key=lambda r: r["effective_wall_s"],
    reverse=True,
)

labels = [
    f"{r['mode_label']}\n{r['template']}\n{r['backend']}\nscale={r['scale_value']}"
    for r in top
]
values = [r["effective_wall_s"] for r in top]

plt.figure(figsize=(10, max(5, 0.32 * len(labels))))
plt.barh(labels, values)
plt.xlabel("Effective wall time per case (s)")
plt.title("One-shot vs amortized batch wall time")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "effective_wall_time_per_case.png", dpi=180)
plt.close()

# Plot 2: runtime vs elements, one line per template/backend/mode.
plt.figure()
for key in sorted(set((r["mode_label"], r["template"], r["backend"]) for r in normalized)):
    data = [
        r for r in normalized
        if (r["mode_label"], r["template"], r["backend"]) == key
        and r["n_elements"] is not None
        and r["runtime_s"] is not None
        and r["n_elements"] > 0
    ]
    if len(data) < 2:
        continue
    data = sorted(data, key=lambda r: r["n_elements"])
    plt.plot(
        [r["n_elements"] for r in data],
        [r["runtime_s"] for r in data],
        marker="o",
        label=f"{key[0]} / {key[1]} / {key[2]}",
    )

plt.xlabel("Elements")
plt.ylabel("Runtime inside template call (s)")
plt.title("Runtime scaling by topology size")
plt.legend(fontsize="x-small")
plt.tight_layout()
plt.savefig(PLOTS / "runtime_vs_elements_by_mode.png", dpi=180)
plt.close()

# Plot 3: wall composition for selected largest rows.
selected = [
    r for r in summary
    if (
        (r["template"] == "rf_jtl_linear" and r["scale_value"] == "2376") or
        (r["template"] == "ethz_jtl_linear" and r["scale_value"] == "2048") or
        (r["template"] == "jtl_linear" and r["scale_value"] == "2048")
    )
]
selected = sorted(selected, key=lambda r: (r["template"], r["backend"], r["mode_label"]))

labels = [
    f"{r['mode_label']}\n{r['template']}\n{r['backend']}\nscale={r['scale_value']}"
    for r in selected
]
startup = [r["startup_s"] or 0 for r in selected]
include = [r["include_setup_s"] or 0 for r in selected]
runtime = [r["runtime_s"] or 0 for r in selected]
outer = [r["outer_unattributed_s"] or 0 for r in selected]

plt.figure(figsize=(11, max(5, 0.4 * len(labels))))
left = [0] * len(labels)
plt.barh(labels, startup, left=left, label="Startup/import")
left = [a + b for a, b in zip(left, startup)]
plt.barh(labels, include, left=left, label="Include/setup")
left = [a + b for a, b in zip(left, include)]
plt.barh(labels, runtime, left=left, label="Runtime")
left = [a + b for a, b in zip(left, runtime)]
plt.barh(labels, outer, left=left, label="Outer unattributed")
plt.xlabel("Effective wall time per case (s)")
plt.title("Wall-time composition for largest selected cases")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(PLOTS / "wall_composition_selected_large_cases.png", dpi=180)
plt.close()

md = [
    "# One-shot vs batch normalized summary",
    "",
    "Data collection only. Batch rows use amortized batch wall time per successful case.",
    "",
    f"- Batch total wall time: `{batch_total_wall_s}` s",
    f"- Batch successful rows: `{batch_n}`",
    f"- Batch amortized wall time per row: `{batch_amortized_wall_s}` s",
    "",
    "| mode | template | backend | scale | effective wall s | runtime s | runtime % effective wall | startup % | include/setup % | outer unattributed % |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
]

for r in summary:
    md.append(
        f"| {r['mode_label']} | {r['template']} | {r['backend']} | {r['scale_value']} | "
        f"{r['effective_wall_s']} | {r['runtime_s']} | "
        f"{r['runtime_pct_effective_wall']} | {r['startup_pct_effective_wall']} | "
        f"{r['include_setup_pct_effective_wall']} | {r['outer_unattributed_pct_effective_wall']} |"
    )

md += [
    "",
    "## Plots",
    "",
    "- `plots/effective_wall_time_per_case.png`",
    "- `plots/runtime_vs_elements_by_mode.png`",
    "- `plots/wall_composition_selected_large_cases.png`",
]

report = ROOT / "one_shot_vs_batch_normalized_summary.md"
report.write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", norm_csv)
print("WROTE", summary_csv)
print("WROTE", report)
print("WROTE", PLOTS)
