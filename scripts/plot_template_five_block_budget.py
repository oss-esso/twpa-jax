from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_template_five_block_selected")
CSV = ROOT / "template_five_block_rows.csv"
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def f(x):
    if x is None or x == "":
        return None
    return float(str(x).replace(",", "."))

with CSV.open("r", newline="", encoding="utf-8") as file:
    rows = [r for r in csv.DictReader(file) if r.get("status") == "PASS"]

groups = {}
for r in rows:
    key = (r["template"], r["scale_value"])
    groups.setdefault(key, []).append(r)

summary = []
for (template, scale), rs in sorted(groups.items()):
    def avg(key):
        vals = [f(r[key]) for r in rs if r.get(key) not in ("", None)]
        return mean(vals) if vals else None

    summary.append({
        "template": template,
        "scale_value": scale,
        "n": len(rs),
        "runtime_s": avg("runtime_s"),
        "n_elements": avg("n_elements"),
        "n_nodes": avg("n_nodes"),
        "jc_tuple_count": avg("jc_tuple_count"),
        "max_abs_diff_vs_public": max(f(r["max_abs_diff_vs_public"]) for r in rs),
        "B1_frequency_pct": avg("block1_frequency_pct_runtime"),
        "B2_setup_matrix_pct": avg("block2_circuit_matrix_setup_pct_runtime"),
        "B3_nonlinear_pct": avg("block3_nonlinear_pump_pct_runtime"),
        "B4_linearized_pct": avg("block4_linearized_solve_pct_runtime"),
        "B5_output_pct": avg("block5_output_conversion_pct_runtime"),
    })

summary_csv = ROOT / "template_five_block_mean_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

labels = [f"{r['template']}\nscale={r['scale_value']}" for r in summary]
b1 = [r["B1_frequency_pct"] or 0 for r in summary]
b2 = [r["B2_setup_matrix_pct"] or 0 for r in summary]
b3 = [r["B3_nonlinear_pct"] or 0 for r in summary]
b4 = [r["B4_linearized_pct"] or 0 for r in summary]
b5 = [r["B5_output_pct"] or 0 for r in summary]

plt.figure(figsize=(9, 5))
bottom = [0] * len(labels)
for vals, name in [
    (b1, "B1 frequency"),
    (b2, "B2 setup/matrix"),
    (b3, "B3 nonlinear"),
    (b4, "B4 linearized"),
    (b5, "B5 output"),
]:
    plt.bar(labels, vals, bottom=bottom, label=name)
    bottom = [x + y for x, y in zip(bottom, vals)]

plt.ylabel("Share of staged runtime (%)")
plt.title("WT-4B template five-block runtime shares")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(PLOTS / "template_five_block_shares.png", dpi=180)
plt.close()

plt.figure()
plt.bar(labels, [r["runtime_s"] for r in summary])
plt.ylabel("Mean staged runtime (s)")
plt.title("WT-4B staged runtime by template")
plt.tight_layout()
plt.savefig(PLOTS / "template_five_block_runtime.png", dpi=180)
plt.close()

md = [
    "# WT-4B template five-block summary",
    "",
    "Data collection only. Rows are means over successful repetitions.",
    "",
    "| template | scale | elements | runtime s | max diff | B1 % | B2 % | B3 % | B4 % | B5 % |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for r in summary:
    md.append(
        f"| {r['template']} | {r['scale_value']} | {r['n_elements']} | "
        f"{r['runtime_s']} | {r['max_abs_diff_vs_public']} | "
        f"{r['B1_frequency_pct']} | {r['B2_setup_matrix_pct']} | "
        f"{r['B3_nonlinear_pct']} | {r['B4_linearized_pct']} | {r['B5_output_pct']} |"
    )

md += [
    "",
    "## Plots",
    "",
    "- `plots/template_five_block_shares.png`",
    "- `plots/template_five_block_runtime.png`",
]

report = ROOT / "template_five_block_mean_summary.md"
report.write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", summary_csv)
print("WROTE", report)
print("WROTE", PLOTS)
