from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
SEED = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "pump" / "pump_solution.npz"
SOLVED = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "pump_solved" / "pump_solution.npz"
OUT = ROOT / "outputs" / "exp14_fxjtwpa_seed_vs_solved_compare"
OUT.mkdir(parents=True, exist_ok=True)

def load(path: Path):
    z = np.load(path, allow_pickle=True)
    X = z["X_real"] + 1j * z["X_imag"]
    modes = z["pump_modes"].astype(int) if "pump_modes" in z else z["harmonics"].astype(int)
    return X, modes

def best_alpha(a, b):
    # alpha minimizing ||alpha*a - b||
    den = np.vdot(a, a)
    if abs(den) == 0:
        return 0.0 + 0.0j
    return np.vdot(a, b) / den

def top_entries(v, n=30):
    order = np.argsort(np.abs(v))[::-1][:n]
    return [
        {
            "index": int(i),
            "real": float(np.real(v[i])),
            "imag": float(np.imag(v[i])),
            "abs": float(abs(v[i])),
        }
        for i in order
    ]

X0, m0 = load(SEED)
X1, m1 = load(SOLVED)

if list(m0) != list(m1):
    raise SystemExit(f"mode mismatch seed={m0} solved={m1}")

rows = []
top_rows = []

alpha_global = best_alpha(X0.ravel(), X1.ravel())
X0g = alpha_global * X0
global_diff = X0g - X1

print("GLOBAL_ALPHA")
print(f"alpha = {alpha_global.real:+.16e} {alpha_global.imag:+.16e}j")
print(f"abs   = {abs(alpha_global):.16e}")
print(f"angle = {np.angle(alpha_global):.16e} rad")
print(f"global_rel_l2_after_alpha = {np.linalg.norm(global_diff.ravel()) / np.linalg.norm(X1.ravel()):.16e}")
print()

for r, mode in enumerate(m0):
    seed = X0[r]
    solved = X1[r]

    alpha = best_alpha(seed, solved)
    seed_scaled = alpha * seed
    diff_raw = seed - solved
    diff_scaled = seed_scaled - solved

    row = {
        "mode": int(mode),
        "seed_l2": float(np.linalg.norm(seed)),
        "solved_l2": float(np.linalg.norm(solved)),
        "raw_diff_l2": float(np.linalg.norm(diff_raw)),
        "raw_rel_l2": float(np.linalg.norm(diff_raw) / max(np.linalg.norm(solved), 1e-300)),
        "alpha_real": float(np.real(alpha)),
        "alpha_imag": float(np.imag(alpha)),
        "alpha_abs": float(abs(alpha)),
        "alpha_angle_rad": float(np.angle(alpha)),
        "scaled_diff_l2": float(np.linalg.norm(diff_scaled)),
        "scaled_rel_l2": float(np.linalg.norm(diff_scaled) / max(np.linalg.norm(solved), 1e-300)),
        "max_abs_diff_scaled": float(np.max(np.abs(diff_scaled))),
    }
    rows.append(row)

    print(
        f"mode={mode:2d} "
        f"raw_rel={row['raw_rel_l2']:.6e} "
        f"alpha_abs={row['alpha_abs']:.9g} "
        f"alpha_angle={row['alpha_angle_rad']:.6e} "
        f"scaled_rel={row['scaled_rel_l2']:.6e} "
        f"max_scaled={row['max_abs_diff_scaled']:.6e}"
    )

    for tr in top_entries(diff_scaled, 40):
        tr["mode"] = int(mode)
        top_rows.append(tr)

with (OUT / "mode_scalar_summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for row in rows:
        w.writerow(row)

with (OUT / "top_scaled_difference_entries.csv").open("w", newline="", encoding="utf-8") as f:
    keys = ["mode", "index", "real", "imag", "abs"]
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for row in top_rows:
        w.writerow(row)

print()
print(f"WROTE {OUT}")
print()
print("TOP mode-1 scaled differences:")
for row in [r for r in top_rows if r["mode"] == 1][:40]:
    print(row)
