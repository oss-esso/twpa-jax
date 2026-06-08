from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_rf_jtl_colleague_regime_boundary")
CSV = ROOT / "rf_jtl_colleague_regime_rows.csv"
PLOTS = ROOT / "physics_plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def f(x):
    if x is None or x == "":
        return None
    return float(str(x).replace(",", "."))

def ip_uA_to_dbm(ip_uA: float, z0: float = 50.0) -> float:
    i_peak = ip_uA * 1e-6
    p_w = 0.5 * i_peak * i_peak * z0
    return 10.0 * math.log10(p_w / 1e-3)

with CSV.open("r", newline="", encoding="utf-8") as file:
    rows = [r for r in csv.DictReader(file) if r["status"] == "PASS"]

# Summary table of best rows.
best = sorted(rows, key=lambda r: f(r["gain_db_max"]) or -1e99, reverse=True)

best_csv = ROOT / "rf_jtl_best_gain_rows.csv"
with best_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(best[0].keys()))
    writer.writeheader()
    writer.writerows(best)

# Heatmaps: one per N_cell and signal window.
for n_cell in sorted(set(r["n_cell"] for r in rows), key=lambda x: int(x)):
    for window in sorted(set((r["signal_start_hz"], r["signal_stop_hz"], r["signal_points"]) for r in rows)):
        subset = [
            r for r in rows
            if r["n_cell"] == n_cell
            and (r["signal_start_hz"], r["signal_stop_hz"], r["signal_points"]) == window
        ]
        if not subset:
            continue

        fps = sorted(set(f(r["pump_frequency_ghz"]) for r in subset))
        ips = sorted(set(f(r["pump_current_ua"]) for r in subset))

        # matrix rows = pump current, columns = pump frequency
        z = []
        for ip in ips:
            row = []
            for fp in fps:
                hit = [
                    r for r in subset
                    if abs(f(r["pump_frequency_ghz"]) - fp) < 1e-12
                    and abs(f(r["pump_current_ua"]) - ip) < 1e-12
                ]
                row.append(f(hit[0]["gain_db_max"]) if hit else float("nan"))
            z.append(row)

        plt.figure()
        plt.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=[min(fps), max(fps), min(ips), max(ips)],
        )
        plt.colorbar(label="Max gain over signal sweep (dB)")
        plt.xlabel("Pump frequency (GHz)")
        plt.ylabel("Pump current (µA)")
        fs0 = f(window[0]) / 1e9
        fs1 = f(window[1]) / 1e9
        plt.title(f"RF-JTL gain map, N={n_cell}, signal {fs0:.2f}-{fs1:.2f} GHz")
        plt.tight_layout()
        out = PLOTS / f"gain_map_N{n_cell}_signal_{fs0:.2f}_{fs1:.2f}GHz.png"
        plt.savefig(out, dpi=180)
        plt.close()

# Best gain-vs-signal curves.
top_curves = best[:10]

plt.figure()
for r in top_curves:
    freq = json.loads(r["freq_hz_json"])
    gain = json.loads(r["gain_db_json"])
    label = (
        f"N={r['n_cell']}, fp={float(r['pump_frequency_ghz']):.3f}GHz, "
        f"Ip={float(r['pump_current_ua']):.1f}µA, max={float(r['gain_db_max']):.2f}dB"
    )
    plt.plot([x / 1e9 for x in freq], gain, label=label)

plt.xlabel("Signal frequency (GHz)")
plt.ylabel("Gain / |S21| (dB)")
plt.title("Top RF-JTL gain curves in colleague-like pumped regime")
plt.legend(fontsize="x-small")
plt.tight_layout()
plt.savefig(PLOTS / "top_gain_vs_signal_frequency.png", dpi=180)
plt.close()

# Pump-current to dBm helper table.
ip_rows = []
for ip in sorted(set(f(r["pump_current_ua"]) for r in rows)):
    ip_rows.append({
        "pump_current_uA": ip,
        "equivalent_power_dBm_50ohm_peak_current": ip_uA_to_dbm(ip),
    })

ip_csv = ROOT / "pump_current_to_dbm_reference.csv"
with ip_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(ip_rows[0].keys()))
    writer.writeheader()
    writer.writerows(ip_rows)

md = [
    "# RF-JTL colleague-regime physics plots",
    "",
    "All plots are generated from the finite PASS boundary dataset.",
    "",
    f"- rows: {len(rows)}",
    f"- best gain_db_max: {float(best[0]['gain_db_max']):.6g} dB",
    f"- best case: {best[0]['case_id']}, N={best[0]['n_cell']}, fp={best[0]['pump_frequency_ghz']} GHz, Ip={best[0]['pump_current_ua']} µA",
    "",
    "## Generated files",
    "",
    "- `physics_plots/top_gain_vs_signal_frequency.png`",
    "- `physics_plots/gain_map_N*_signal_*.png`",
    "- `rf_jtl_best_gain_rows.csv`",
    "- `pump_current_to_dbm_reference.csv`",
]

(ROOT / "rf_jtl_physics_plots.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", PLOTS)
print("WROTE", ROOT / "rf_jtl_physics_plots.md")
print("WROTE", best_csv)
print("WROTE", ip_csv)
