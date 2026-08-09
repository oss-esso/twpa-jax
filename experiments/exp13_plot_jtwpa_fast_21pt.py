from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

outdir = Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp13_jtwpa_fast_scale2")
jc_csv = outdir / "jc_jtwpa_curve_21pt.csv"
py_csv = outdir / "gain_h5_21pt" / "gain_sweep.csv"

def load_curve(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fk = "signal_ghz" if "signal_ghz" in rows[0] else "frequency_ghz"
    gk = "gain_db"
    x = np.array([float(r[fk]) for r in rows])
    y = np.array([float(r[gk]) for r in rows])
    idx = np.argsort(x)
    return x[idx], y[idx]

fj, gj = load_curve(jc_csv)
fp, gp = load_curve(py_csv)

gp_on_jc = gp if np.array_equal(fj, fp) else np.interp(fj, fp, gp)
err = gp_on_jc - gj

print("JC max =", float(np.max(gj)), "peak =", float(fj[int(np.argmax(gj))]))
print("OURS max =", float(np.max(gp)), "peak =", float(fp[int(np.argmax(gp))]))
print("max_abs_error_db =", float(np.max(np.abs(err))))
print("mean_abs_error_db =", float(np.mean(np.abs(err))))
print("rms_error_db =", float(np.sqrt(np.mean(err**2))))

plt.figure(figsize=(10, 6))
plt.plot(fj, gj, label="JC")
plt.plot(fp, gp, label="OURS H5 sidebands3 scale2")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Gain (dB)")
plt.title("JTWPA fast 21-point gain comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "jtwpa_fast_21pt_overlay.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 4.8))
plt.plot(fj, err)
plt.xlabel("Frequency (GHz)")
plt.ylabel("OURS - JC (dB)")
plt.title("JTWPA fast 21-point gain error")
plt.grid(True)
plt.tight_layout()
plt.savefig(outdir / "jtwpa_fast_21pt_error.png", dpi=160)
plt.close()

print("wrote_overlay =", outdir / "jtwpa_fast_21pt_overlay.png")
print("wrote_error =", outdir / "jtwpa_fast_21pt_error.png")
