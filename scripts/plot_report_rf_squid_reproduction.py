from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

SPECTRUM_ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_report_rf_squid_spectrum")
MAP_ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_report_rf_squid_small_map")

SPECTRUM_CSV = SPECTRUM_ROOT / "report_rf_squid_reproduction_rows.csv"
MAP_CSV = MAP_ROOT / "report_rf_squid_reproduction_rows.csv"

PLOTS = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_report_rf_squid_reproduction_plots")
PLOTS.mkdir(parents=True, exist_ok=True)

def f(x):
    if x is None or x == "":
        return None
    return float(str(x).replace(",", "."))

def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

spectrum_rows = read_csv(SPECTRUM_CSV)
map_rows = read_csv(MAP_CSV)

# 1. Figure-12-like gain spectrum.
plt.figure()
for r in spectrum_rows:
    if r["status"] != "PASS":
        continue
    freq = json.loads(r["freq_ghz_json"])
    gain = json.loads(r["gain_db_json"])
    label = f"fp={f(r['pump_frequency_ghz']):.3f} GHz, Ip={f(r['pump_current_ua']):.2f} µA, max={f(r['gain_db_max']):.1f} dB"
    plt.plot(freq, gain, label=label)

plt.xlabel("Signal frequency (GHz)")
plt.ylabel("Gain / |S21| (dB)")
plt.title("RF-SQUID TWPA gain spectrum reproduction")
plt.legend(fontsize="x-small")
plt.tight_layout()
plt.savefig(PLOTS / "figure12_like_rf_squid_gain_spectrum.png", dpi=200)
plt.close()

# 2. Figure-8-style pump-current/frequency map.
fps = sorted(set(f(r["pump_frequency_ghz"]) for r in map_rows if r["status"] == "PASS"))
ips = sorted(set(f(r["pump_current_ua"]) for r in map_rows if r["status"] == "PASS"))

z = []
for ip in ips:
    row = []
    for fp in fps:
        hit = [
            r for r in map_rows
            if r["status"] == "PASS"
            and abs(f(r["pump_frequency_ghz"]) - fp) < 1e-12
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
plt.colorbar(label="Gain at fs = fp/2 - 10 MHz (dB)")
plt.xlabel("Pump frequency (GHz)")
plt.ylabel("Pump current (µA)")
plt.title("RF-SQUID pump-frequency/current gain map")
plt.tight_layout()
plt.savefig(PLOTS / "figure8_style_rf_squid_pump_current_frequency_map.png", dpi=200)
plt.close()

# 3. Best rows summary.
best_spectrum = sorted(
    [r for r in spectrum_rows if r["status"] == "PASS"],
    key=lambda r: f(r["gain_db_max"]) or -1e99,
    reverse=True,
)

best_map = sorted(
    [r for r in map_rows if r["status"] == "PASS"],
    key=lambda r: f(r["gain_db_max"]) or -1e99,
    reverse=True,
)

md = [
    "# RF-SQUID report-plot reproduction summary",
    "",
    "All plotted rows are finite PASS rows. Some Julia runs emitted nonlinear-solver convergence warnings; treat these as finite-but-warning rows until convergence metadata is captured explicitly.",
    "",
    "## Figure-12-like gain spectrum",
    "",
    f"- rows: {len(spectrum_rows)}",
    f"- PASS rows: {sum(1 for r in spectrum_rows if r['status'] == 'PASS')}",
    f"- best gain: {f(best_spectrum[0]['gain_db_max']):.6g} dB",
    f"- best point: fp={best_spectrum[0]['pump_frequency_ghz']} GHz, Ip={best_spectrum[0]['pump_current_ua']} µA",
    "",
    "## Figure-8-style small pump map",
    "",
    f"- rows: {len(map_rows)}",
    f"- PASS rows: {sum(1 for r in map_rows if r['status'] == 'PASS')}",
    f"- best gain: {f(best_map[0]['gain_db_max']):.6g} dB",
    f"- best point: fp={best_map[0]['pump_frequency_ghz']} GHz, Ip={best_map[0]['pump_current_ua']} µA",
    "",
    "## Generated plots",
    "",
    "- `figure12_like_rf_squid_gain_spectrum.png`",
    "- `figure8_style_rf_squid_pump_current_frequency_map.png`",
]

(PLOTS / "rf_squid_report_plot_reproduction_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("WROTE", PLOTS)
print("REPORT", PLOTS / "rf_squid_report_plot_reproduction_summary.md")
