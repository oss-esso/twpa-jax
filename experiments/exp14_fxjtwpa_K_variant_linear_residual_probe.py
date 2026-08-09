from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import numpy as np
import scipy.sparse as sp

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
VARIANT_ROOT = ROOT / "outputs" / "exp14_fxjtwpa_K_variants_v2"
SEED_ROOT = ROOT / "outputs" / "exp14_fxjtwpa_jcseed"
OUT = ROOT / "outputs" / "exp14_fxjtwpa_K_variant_linear_residuals"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "baseline_copy",
    "flip_huge_offdiag",
    "scale_huge_all_by_denom",
    "flip_offdiag_and_scale_huge_all",
]

PUMP_FILES = [
    SEED_ROOT / "pump" / "pump_solution.npz",
    SEED_ROOT / "pump_solved" / "pump_solution.npz",
]

# FXJTWPA doc pump is very likely 8 GHz. The K term dominates this diagnostic,
# so a tiny omega mismatch cannot fake a 45 -> small structural change.
FALLBACK_OMEGA = 2.0 * math.pi * 8.0e9


def load_omega_from_report(seed_pump_file: Path) -> float:
    report = seed_pump_file.parent / "pump_report.json"
    if not report.exists():
        return FALLBACK_OMEGA
    try:
        obj = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return FALLBACK_OMEGA

    for key in ["omega_p", "pump_omega", "pump_omega_rad_s", "omega_rad_s"]:
        if key in obj:
            return float(obj[key])

    for key in ["pump_freq_ghz", "freq_ghz", "pump_frequency_ghz"]:
        if key in obj:
            return 2.0 * math.pi * float(obj[key]) * 1e9

    for key in ["pump_freq_hz", "freq_hz", "pump_frequency_hz"]:
        if key in obj:
            return 2.0 * math.pi * float(obj[key])

    meta = obj.get("metadata", {}) if isinstance(obj, dict) else {}
    for key in ["omega_p", "pump_omega", "pump_omega_rad_s", "omega_rad_s"]:
        if key in meta:
            return float(meta[key])
    for key in ["pump_freq_ghz", "freq_ghz", "pump_frequency_ghz"]:
        if key in meta:
            return 2.0 * math.pi * float(meta[key]) * 1e9

    return FALLBACK_OMEGA


def load_X(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X_real"] + 1j * z["X_imag"]
    if "pump_modes" in z:
        modes = z["pump_modes"].astype(int)
    elif "harmonics" in z:
        modes = z["harmonics"].astype(int)
    else:
        modes = np.arange(1, X.shape[0] + 1, dtype=int)
    return X, modes


def top_entries(v: np.ndarray, n: int = 20) -> list[dict]:
    order = np.argsort(np.abs(v))[::-1][:n]
    rows = []
    for idx in order:
        rows.append({
            "index": int(idx),
            "real": float(np.real(v[idx])),
            "imag": float(np.imag(v[idx])),
            "abs": float(abs(v[idx])),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


summary_rows = []
top_rows = []

for variant in VARIANTS:
    d = VARIANT_ROOT / variant
    C = sp.load_npz(d / "C.npz").tocsr()
    G = sp.load_npz(d / "G.npz").tocsr()
    K = sp.load_npz(d / "K.npz").tocsr()

    for pump_file in PUMP_FILES:
        X, modes = load_X(pump_file)
        omega_p = load_omega_from_report(pump_file)

        all_res = []
        mode_reports = []

        for row_idx, mode in enumerate(modes):
            omega = float(mode) * omega_p
            D = K + (-omega * omega) * C + (1j * omega) * G
            r = D @ X[row_idx]
            all_res.append(r)
            mode_reports.append({
                "mode": int(mode),
                "inf": float(np.max(np.abs(r))),
                "l2": float(np.linalg.norm(r)),
                "mean_abs": float(np.mean(np.abs(r))),
            })

            if int(mode) == 1:
                for tr in top_entries(r, 40):
                    top_rows.append({
                        "variant": variant,
                        "pump_file": str(pump_file),
                        "mode": int(mode),
                        **tr,
                    })

        all_flat = np.concatenate([np.ravel(r) for r in all_res])
        mode1 = [m for m in mode_reports if m["mode"] == 1][0]

        row = {
            "variant": variant,
            "pump_file": str(pump_file),
            "omega_p": omega_p,
            "all_inf": float(np.max(np.abs(all_flat))),
            "all_l2": float(np.linalg.norm(all_flat)),
            "mode1_inf": mode1["inf"],
            "mode1_l2": mode1["l2"],
            "mode1_mean_abs": mode1["mean_abs"],
            "K_absmax": float(np.max(np.abs(K.data))),
        }
        summary_rows.append(row)

        print("\n" + "=" * 100)
        print("variant:", variant)
        print("pump:", pump_file)
        print("=" * 100)
        for k, v in row.items():
            if k not in ("variant", "pump_file"):
                print(f"{k}: {v}")
        print("mode reports:")
        for mr in mode_reports[:8]:
            print(mr)

write_csv(OUT / "summary.csv", summary_rows)
write_csv(OUT / "mode1_top_residual_entries.csv", top_rows)

print("\nSUMMARY SORTED BY mode1_l2")
for r in sorted(summary_rows, key=lambda x: x["mode1_l2"]):
    print(
        f"{r['variant']:32s} "
        f"pump={Path(r['pump_file']).parent.name:12s} "
        f"mode1_l2={r['mode1_l2']:.6e} "
        f"mode1_inf={r['mode1_inf']:.6e} "
        f"K_absmax={r['K_absmax']:.6e}"
    )

print(f"\nWROTE {OUT}")
