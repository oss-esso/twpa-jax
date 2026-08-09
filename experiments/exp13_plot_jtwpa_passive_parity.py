from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

outdir = Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp13_jtwpa_passive_parity")
jc_csv = outdir / "jc_jtwpa_passive_curve_21pt.csv"
py_csv = outdir / "python_passive_gain_21pt" / "gain_sweep.csv"

def load_curve(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fk = "signal_ghz" if "signal_ghz" in rows[0] else "frequency_ghz"
    x = np.array([float(r[fk]) for r in rows])
    y = np.array([float(r["gain_db"]) for r in rows])
    idx = np.argsort(x)
    return x[idx], y[idx]

fj, gj = load_curve(jc_csv)
fp, gp = load_curve(py_csv)

gp_on_jc = gp if np.array_equal(fj, fp) else np.interp(fj, fp, gp)
err = gp_on_jc - gj

print("JC passive max =", float(np.max(gj)), "peak =", float(fj[int(np.argmax(gj))]))
print("OURS passive max =", float(np.max(gp)), "peak =", float(fp[int(np.argmax(gp))]))
print("max_abs_error_db =", float(np.max(np.abs(err))))
print("mean_abs_error_db =", float(np.mean(np.abs(err))))
print("rms_error_db =", float(np.sqrt(np.mean(err**2))))

plt.figure(figsize=(10, 6))
plt.plot(fj, gj, label="JC passive")
plt.plot(fp, gp, label="OURS passive")
plt.xlabel("Frequency (GHz)")
plt.ylabel("S21 gain (dB)")
plt.title("JTWPA passive/zero-pump S21 comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "jtwpa_passive_overlay.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 4.8))
plt.plot(fj, err)
plt.xlabel("Frequency (GHz)")
plt.ylabel("OURS - JC (dB)")
plt.title("JTWPA passive/zero-pump S21 error")
plt.grid(True)
plt.tight_layout()
plt.savefig(outdir / "jtwpa_passive_error.png", dpi=160)
plt.close()

print("wrote_overlay =", outdir / "jtwpa_passive_overlay.png")
print("wrote_error =", outdir / "jtwpa_passive_error.png")
