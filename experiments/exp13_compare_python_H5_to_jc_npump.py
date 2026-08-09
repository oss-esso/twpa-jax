from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

root = Path(r"D:\Projects\Thesis\twpa_jax")
outdir = root / "outputs" / "exp13_jtwpa_harmonic_math"

py_candidates = [
    root / "outputs" / "exp13_jtwpa_harmonic_ladder" / "H5_nt96" / "gain" / "gain_sweep.csv",
    root / "outputs" / "exp13_jtwpa_fast_scale2" / "gain_h5_21pt" / "gain_sweep.csv",
]

py_csv = None
for p in py_candidates:
    if p.exists():
        py_csv = p
        break

if py_csv is None:
    raise SystemExit("Could not find Python H5 gain_sweep.csv")

def load_curve(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fk = "signal_ghz" if "signal_ghz" in rows[0] else "frequency_ghz"
    x = np.array([float(r[fk]) for r in rows], dtype=float)
    y = np.array([float(r["gain_db"]) for r in rows], dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]

fp, gp = load_curve(py_csv)

print("Python H5 curve =", py_csv)
print("Python H5 max =", float(np.max(gp)), "peak =", float(fp[int(np.argmax(gp))]))
print()
print("compare_to py_max jc_max py_peak jc_peak max_abs_err mean_abs_err rms_err")

best = None

for npump in [6, 8, 10]:
    jc_csv = outdir / f"jc_jtwpa_npump{npump}_curve_21pt.csv"
    fj, gj = load_curve(jc_csv)

    gp_on_jc = gp if np.array_equal(fj, fp) else np.interp(fj, fp, gp)
    err = gp_on_jc - gj

    row = {
        "npump": npump,
        "py_max": float(np.max(gp)),
        "jc_max": float(np.max(gj)),
        "py_peak": float(fp[int(np.argmax(gp))]),
        "jc_peak": float(fj[int(np.argmax(gj))]),
        "max_abs_err": float(np.max(np.abs(err))),
        "mean_abs_err": float(np.mean(np.abs(err))),
        "rms_err": float(np.sqrt(np.mean(err**2))),
    }

    print(
        f"JC_N{npump:<2d} "
        f"{row['py_max']:.6f} "
        f"{row['jc_max']:.6f} "
        f"{row['py_peak']:.6f} "
        f"{row['jc_peak']:.6f} "
        f"{row['max_abs_err']:.6f} "
        f"{row['mean_abs_err']:.6f} "
        f"{row['rms_err']:.6f}"
    )

    if best is None or row["rms_err"] < best["rms_err"]:
        best = row
        best_jc = (fj, gj, npump)

fj, gj, npump = best_jc
gp_on_jc = gp if np.array_equal(fj, fp) else np.interp(fj, fp, gp)
err = gp_on_jc - gj

plt.figure(figsize=(10, 6))
plt.plot(fj, gj, label=f"JC Npump={npump}")
plt.plot(fp, gp, label="Python H5 scale2")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Gain (dB)")
plt.title(f"JTWPA: Python H5 vs JC Npump={npump}")
plt.grid(True)
plt.legend()
plt.tight_layout()
overlay = outdir / f"python_H5_vs_jc_N{npump}_overlay.png"
plt.savefig(overlay, dpi=160)
plt.close()

plt.figure(figsize=(10, 4.8))
plt.plot(fj, err)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Python - JC (dB)")
plt.title(f"JTWPA error: Python H5 vs JC Npump={npump}")
plt.grid(True)
plt.tight_layout()
error_png = outdir / f"python_H5_vs_jc_N{npump}_error.png"
plt.savefig(error_png, dpi=160)
plt.close()

print()
print("BEST =", best)
print("wrote_overlay =", overlay)
print("wrote_error =", error_png)
