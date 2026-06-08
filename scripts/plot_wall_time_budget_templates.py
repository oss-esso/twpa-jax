from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_wall_time_budget_templates")
CSV = ROOT / "wall_time_budget_rows.csv"
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def parse_float(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return None

with CSV.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Use warm reps only for scaling plots.
warm = [r for r in rows if r.get("cold_or_warm") == "warm" and r.get("status") == "PASS"]

groups = {}
for r in warm:
    key = (r["template"], r["backend"], r["scale_value"])
    groups.setdefault(key, []).append(r)

summary = []
for (template, backend, scale), rs in sorted(groups.items()):
    runtimes = [parse_float(r["runtime_s"]) for r in rs]
    runtimes = [x for x in runtimes if x is not None]
    elems = [parse_float(r["n_elements"]) for r in rs]
    elems = [x for x in elems if x is not None]
    nodes = [parse_float(r["n_nodes"]) for r in rs]
    nodes = [x for x in nodes if x is not None]
    tuples = [parse_float(r["jc_tuple_count"]) for r in rs]
    tuples = [x for x in tuples if x is not None]

    summary.append({
        "template": template,
        "backend": backend,
        "scale_value": scale,
        "n": len(rs),
        "runtime_mean_s": mean(runtimes) if runtimes else None,
        "runtime_min_s": min(runtimes) if runtimes else None,
        "runtime_max_s": max(runtimes) if runtimes else None,
        "n_elements": mean(elems) if elems else None,
        "n_nodes": mean(nodes) if nodes else None,
        "jc_tuple_count": mean(tuples) if tuples else None,
    })

summary_csv = ROOT / "wall_time_budget_warm_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "template", "backend", "scale_value", "n",
            "runtime_mean_s", "runtime_min_s", "runtime_max_s",
            "n_elements", "n_nodes", "jc_tuple_count",
        ],
    )
    writer.writeheader()
    writer.writerows(summary)

# Plot 1: runtime vs elements, grouped by template/backend.
plt.figure()
for (template, backend), rs in sorted({(r["template"], r["backend"]): [] for r in summary}.items()):
    data = [
        r for r in summary
        if r["template"] == template and r["backend"] == backend
        and r["n_elements"] is not None
        and r["runtime_mean_s"] is not None
        and r["n_elements"] > 0
    ]
    if not data:
        continue
    data = sorted(data, key=lambda r: r["n_elements"])
    plt.plot(
        [r["n_elements"] for r in data],
        [r["runtime_mean_s"] for r in data],
        marker="o",
        label=f"{template} / {backend}",
    )

plt.xlabel("Elements")
plt.ylabel("Mean warm runtime (s)")
plt.title("Warm runtime vs topology size")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(PLOTS / "runtime_vs_elements.png", dpi=180)
plt.close()

# Plot 2: runtime vs scale value.
plt.figure()
for (template, backend), rs in sorted({(r["template"], r["backend"]): [] for r in summary}.items()):
    data = [
        r for r in summary
        if r["template"] == template and r["backend"] == backend
        and r["runtime_mean_s"] is not None
        and parse_float(r["scale_value"]) is not None
    ]
    if not data:
        continue
    data = sorted(data, key=lambda r: float(r["scale_value"]))
    plt.plot(
        [float(r["scale_value"]) for r in data],
        [r["runtime_mean_s"] for r in data],
        marker="o",
        label=f"{template} / {backend}",
    )

plt.xlabel("Scale value")
plt.ylabel("Mean warm runtime (s)")
plt.title("Warm runtime vs requested scale")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(PLOTS / "runtime_vs_scale.png", dpi=180)
plt.close()

# Plot 3: top warm runtime rows.
top = sorted(
    [r for r in summary if r["runtime_mean_s"] is not None],
    key=lambda r: r["runtime_mean_s"],
    reverse=True,
)[:12]

plt.figure()
labels = [
    f"{r['template']}\n{r['backend']}\nscale={r['scale_value']}"
    for r in top
]
plt.barh(labels, [r["runtime_mean_s"] for r in top])
plt.xlabel("Mean warm runtime (s)")
plt.title("Top warm runtimes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "top_warm_runtimes.png", dpi=180)
plt.close()

md = [
    "# WT-4A warm runtime summary",
    "",
    "This summarizes in-process template runtime only. It does not yet include full process startup/HDF5 teardown accounting.",
    "",
    "| template | backend | scale | elements | nodes | tuples | mean warm runtime s | min | max |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]

for r in summary:
    md.append(
        f"| {r['template']} | {r['backend']} | {r['scale_value']} | "
        f"{r['n_elements']} | {r['n_nodes']} | {r['jc_tuple_count']} | "
        f"{r['runtime_mean_s']} | {r['runtime_min_s']} | {r['runtime_max_s']} |"
    )

md += [
    "",
    "## Plots",
    "",
    "- `plots/runtime_vs_elements.png`",
    "- `plots/runtime_vs_scale.png`",
    "- `plots/top_warm_runtimes.png`",
]

(ROOT / "wall_time_budget_warm_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", summary_csv)
print("WROTE", ROOT / "wall_time_budget_warm_summary.md")
print("WROTE", PLOTS)
