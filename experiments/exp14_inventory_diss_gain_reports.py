from pathlib import Path
import json
import csv
import math

ROOTS = [
    Path(r"D:\Projects\Thesis\twpa_jax\outputs"),
    Path(r"D:\Projects\Thesis\outputs"),
    Path(r"D:\Projects\outputs"),
]

rows = []

for root in ROOTS:
    if not root.exists():
        continue

    for p in root.rglob("gain_report.json"):
        s = str(p).lower()
        if "fqjtwpa_diss" not in s and "diss" not in s:
            continue

        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = obj.get("metadata", {})
        results = obj.get("results", [])

        gains = []
        freqs = []
        for r in results:
            try:
                gains.append(float(r.get("gain_db")))
                freqs.append(float(r.get("signal_ghz")))
            except Exception:
                pass

        if not gains:
            continue

        imax = max(range(len(gains)), key=lambda i: gains[i])

        rows.append({
            "path": str(p),
            "peak": gains[imax],
            "peak_freq": freqs[imax] if freqs else "",
            "n": len(gains),
            "sidebands": meta.get("sidebands", ""),
            "gamma_nt": meta.get("gamma_nt", ""),
            "pump_dir": meta.get("pump_dir", ""),
            "ipm_dir": meta.get("ipm_dir", ""),
            "pump_mode_policy": meta.get("pump_mode_policy", ""),
            "pump_basis": meta.get("pump_basis", ""),
            "loss_linearization_model": meta.get("loss_linearization_model", ""),
            "pump_current_a": meta.get("pump_current_a", ""),
        })

rows.sort(key=lambda r: (
    -("diss_a" in r["path"].lower() or "_diss_a" in r["path"].lower() or "lossyp_lossyg" in r["path"].lower()),
    -r["peak"],
    r["path"],
))

out = Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp14_diss_gain_report_inventory.csv")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "peak", "peak_freq", "n", "sidebands", "gamma_nt",
        "pump_mode_policy", "pump_basis", "loss_linearization_model",
        "pump_current_a", "path", "pump_dir", "ipm_dir",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})

print(f"WROTE {out}")
print()
print("Top candidates:")
for r in rows[:40]:
    print(
        f"peak={r['peak']:9.4f} "
        f"f={r['peak_freq']} "
        f"n={r['n']:3d} "
        f"sidebands={r['sidebands']} "
        f"gamma_nt={r['gamma_nt']} "
        f"policy={r['pump_mode_policy']} "
        f"loss={r['loss_linearization_model']} "
        f"path={r['path']}"
    )
