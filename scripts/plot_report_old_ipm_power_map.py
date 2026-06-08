
from __future__ import annotations



import csv

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np



ROOT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_report_old_ipm_power_map_35")

CSV = ROOT / "report_old_ipm_power_map_rows.csv"

PLOTS = ROOT / "plots"

PLOTS.mkdir(parents=True, exist_ok=True)



def f(x):

    if x is None or x == "":

        return np.nan

    return float(str(x).replace(",", "."))



with CSV.open("r", newline="", encoding="utf-8") as file:

    rows = list(csv.DictReader(file))



fps = sorted(set(f(r["pump_frequency_ghz"]) for r in rows))

powers = sorted(set(f(r["external_power_dbm"]) for r in rows))



gain = np.full((len(powers), len(fps)), np.nan)

status = np.empty((len(powers), len(fps)), dtype=object)



for r in rows:

    i = powers.index(f(r["external_power_dbm"]))

    j = fps.index(f(r["pump_frequency_ghz"]))

    g = f(r["gain_db_max"])

    gain[i, j] = g

    if r["status"] != "PASS":

        status[i, j] = r["status"]

    elif not np.isfinite(g):

        status[i, j] = "NONFINITE"

    elif g < -100 or g > 80:

        status[i, j] = "PASS_SUSPECT"

    else:

        status[i, j] = "PASS"



plot_gain = np.clip(gain, -20, 20)



plt.figure()

plt.imshow(

    plot_gain,

    origin="lower",

    aspect="auto",

    extent=[min(fps), max(fps), min(powers), max(powers)],

)

plt.colorbar(label="Gain, clipped to [-20, 20] dB")

plt.xlabel("Pump frequency (GHz)")

plt.ylabel("External pump power (dBm)")

plt.title("Old-IPM report-style power/frequency map smoke")

plt.tight_layout()

plt.savefig(PLOTS / "old_ipm_power_frequency_map_smoke_clipped.png", dpi=220)

plt.close()



finite = np.isfinite(gain)

best = np.nanargmax(np.where((gain > -100) & (gain < 80), gain, np.nan))

best_i, best_j = np.unravel_index(best, gain.shape)



unique, counts = np.unique(status, return_counts=True)



md = [

    "# Old-IPM report-style power/frequency map smoke",

    "",

    f"- rows: `{len(rows)}`",

    f"- finite gain cells: `{int(finite.sum())}/{gain.size}`",

    f"- raw gain min: `{float(np.nanmin(gain))}` dB",

    f"- raw gain max: `{float(np.nanmax(gain))}` dB",

    f"- best non-suspect gain: `{float(gain[best_i, best_j])}` dB",

    f"- best non-suspect point: fp=`{fps[best_j]}` GHz, Pext=`{powers[best_i]}` dBm",

    "",

    "## Status counts",

    "",

]



for u, c in zip(unique, counts):

    md.append(f"- `{u}`: `{int(c)}`")



md += [

    "",

    "## Note",

    "",

    "Rows with gain below -100 dB or above 80 dB are treated as numerically suspect for plotting. The plot clips gain to [-20, 20] dB.",

    "",

    "## Generated plots",

    "",

    "- `plots/old_ipm_power_frequency_map_smoke_clipped.png`",

]



(ROOT / "old_ipm_power_frequency_map_smoke_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")



print("WROTE", PLOTS / "old_ipm_power_frequency_map_smoke_clipped.png")

print("WROTE", ROOT / "old_ipm_power_frequency_map_smoke_summary.md")

