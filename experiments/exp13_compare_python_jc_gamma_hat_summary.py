from pathlib import Path
import csv
import math

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
OUT = ROOT / "outputs" / "exp13_jtwpa_gamma_hat_compare"

py_csv = OUT / "python_gamma_hat_summary.csv"
jc_csv = OUT / "jc_gamma_hat_summary.csv"

def load(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(float(r["ell"])): r for r in rows}

py = load(py_csv)
jc = load(jc_csv)

print("ell py_rel_l2 jc_rel_l2 ratio_rel py_mean_abs jc_mean_abs ratio_mean_abs py_max jc_max ratio_max")
for ell in [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]:
    pr = py[ell]
    jr = jc[ell]

    py_rel = float(pr["l2_abs_over_zero_l2"])
    jc_rel = float(jr["l2_abs_over_zero_l2"])
    py_mean = float(pr["mean_abs"])
    jc_mean = float(jr["mean_abs"])
    py_max = float(pr["max_abs"])
    jc_max = float(jr["max_abs"])

    ratio_rel = py_rel / jc_rel if jc_rel != 0 else math.inf
    ratio_mean = py_mean / jc_mean if jc_mean != 0 else math.inf
    ratio_max = py_max / jc_max if jc_max != 0 else math.inf

    print(
        f"{ell:+4d} "
        f"{py_rel:.9e} {jc_rel:.9e} {ratio_rel:.6f} "
        f"{py_mean:.9e} {jc_mean:.9e} {ratio_mean:.6f} "
        f"{py_max:.9e} {jc_max:.9e} {ratio_max:.6f}"
    )
