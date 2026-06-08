from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_rf_jtl_colleague_regime_boundary")
CSV = ROOT / "rf_jtl_colleague_regime_rows.csv"
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def f(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return None

with CSV.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

# Plot 1: finite status map over N and pump frequency/current.
status_rank = {"PASS": 1, "NONFINITE": 0, "FAIL": -1}
xs = []
ys = []
cs = []

for r in rows:
    x = f(r["n_cell"])
    y = f(r["pump_frequency_ghz"])
    ip = f(r["pump_current_ua"])
    if x is None or y is None or ip is None:
        continue
    # Offset y slightly by current so the 3 currents are visible.
    xs.append(x)
    ys.append(y + 0.002 * (ip - 2.4))
    cs.append(status_rank.get(r["status"], -1))

plt.figure()
plt.scatter(xs, ys, c=cs)
plt.xlabel("N_cell")
plt.ylabel("Pump frequency GHz, slight offset by Ip")
plt.title("RF-JTL finite/non-finite status map")
plt.tight_layout()
plt.savefig(PLOTS / "finite_status_map.png", dpi=180)
plt.close()

# Plot 2: gain max by N, grouped by signal window.
plt.figure()
for window in sorted(set((r["signal_start_hz"], r["signal_stop_hz"]) for r in rows)):
    data = [
        r for r in rows
        if (r["signal_start_hz"], r["signal_stop_hz"]) == window and r["status"] == "PASS"
    ]
    if not data:
        continue
    data = sorted(data, key=lambda r: (f(r["n_cell"]), f(r["pump_frequency_ghz"]), f(r["pump_current_ua"])))
    plt.scatter(
        [f(r["n_cell"]) for r in data],
        [f(r["gain_db_max"]) for r in data],
        label=f"{f(window[0])/1e9:.2f}-{f(window[1])/1e9:.2f} GHz",
    )

plt.xlabel("N_cell")
plt.ylabel("Max gain over signal sweep (dB)")
plt.title("Max |S21| gain across boundary cases")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "gain_max_vs_ncell.png", dpi=180)
plt.close()

# Plot 3: gain vs signal frequency for selected largest PASS rows.
selected = [
    r for r in rows
    if r["status"] == "PASS"
    and r["n_cell"] == "2393"
    and r["signal_points"] in ("51", "31")
]

# Keep a few readable representative curves.
selected = selected[:8]

plt.figure()
for r in selected:
    freq = json.loads(r["freq_hz_json"])
    gain = json.loads(r["gain_db_json"])
    label = f"N={r['n_cell']}, fp={float(r['pump_frequency_ghz']):.3f}GHz, Ip={float(r['pump_current_ua']):.1f}uA"
    plt.plot([x / 1e9 for x in freq], gain, label=label)

plt.xlabel("Signal frequency (GHz)")
plt.ylabel("Gain / |S21| (dB)")
plt.title("RF-JTL gain vs signal frequency, colleague-like regime")
plt.legend(fontsize="x-small")
plt.tight_layout()
plt.savefig(PLOTS / "gain_vs_signal_frequency_selected.png", dpi=180)
plt.close()

# Markdown summary.
pass_rows = [r for r in rows if r["status"] == "PASS"]
bad_rows = [r for r in rows if r["status"] != "PASS"]

md = [
    "# RF-JTL colleague-regime boundary plots",
    "",
    f"- Total rows: {len(rows)}",
    f"- PASS rows: {len(pass_rows)}",
    f"- non-PASS rows: {len(bad_rows)}",
    "",
    "## Gain range among PASS rows",
    "",
]

if pass_rows:
    gmax = [f(r["gain_db_max"]) for r in pass_rows if f(r["gain_db_max"]) is not None]
    gmin = [f(r["gain_db_min"]) for r in pass_rows if f(r["gain_db_min"]) is not None]
    md += [
        f"- min(gain_db_min): {min(gmin)}",
        f"- max(gain_db_max): {max(gmax)}",
        "",
    ]

md += [
    "## Plots",
    "",
    "- `plots/finite_status_map.png`",
    "- `plots/gain_max_vs_ncell.png`",
    "- `plots/gain_vs_signal_frequency_selected.png`",
]

(ROOT / "rf_jtl_colleague_regime_plots.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", PLOTS)
print("WROTE", ROOT / "rf_jtl_colleague_regime_plots.md")
