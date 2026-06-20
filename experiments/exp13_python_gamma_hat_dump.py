from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"

spec = importlib.util.spec_from_file_location("exp09", EXP09)
exp09 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = exp09
spec.loader.exec_module(exp09)

CASE_DIR = ROOT / "outputs" / "jc_doc_python_designs" / "jc_jtwpa"

# Prefer the H5/nt96 pump used in the latest close comparison.
PUMP_DIR = ROOT / "outputs" / "exp13_jtwpa_harmonic_ladder" / "H5_nt96" / "pump"
if not (PUMP_DIR / "pump_solution.npz").exists():
    PUMP_DIR = ROOT / "outputs" / "exp13_jtwpa_fast_scale_ladder" / "scale_2p000" / "pump"

OUT_ROOT = ROOT / "outputs" / "exp13_jtwpa_gamma_hat_compare"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_ELL = 20
GAMMA_NT = 96

def norm2(x):
    return float(np.linalg.norm(np.asarray(x).ravel()))

def main():
    ipm = exp09.load_ipm(CASE_DIR)
    pump = exp09.load_pump(PUMP_DIR, None)

    gamma_hat = exp09.compute_gamma_hat(
        ipm=ipm,
        pump=pump,
        max_ell=MAX_ELL,
        gamma_nt=GAMMA_NT,
        dc_branch_flux=None,
    )

    zero_gamma = ipm.Ic / ipm.phi0
    zero_norm = max(norm2(zero_gamma), 1e-300)

    summary_rows = []
    coeff_rows = []

    selected_ells = list(range(-MAX_ELL, MAX_ELL + 1))

    for ell in selected_ells:
        gh = np.asarray(gamma_hat[ell], dtype=np.complex128)
        gh_neg = np.asarray(gamma_hat[-ell], dtype=np.complex128)

        denom = max(norm2(gh), norm2(gh_neg), 1e-300)
        conj_err = norm2(gh_neg - np.conj(gh)) / denom

        row = {
            "ell": ell,
            "nbranches": int(gh.size),
            "l2_abs": norm2(gh),
            "l2_abs_over_zero_l2": norm2(gh) / zero_norm,
            "max_abs": float(np.max(np.abs(gh))),
            "mean_abs": float(np.mean(np.abs(gh))),
            "mean_real": float(np.mean(np.real(gh))),
            "mean_imag": float(np.mean(np.imag(gh))),
            "conj_symmetry_rel_err": float(conj_err),
        }
        summary_rows.append(row)

    # Store branch-level coefficients for the harmonics most likely to matter.
    branch_dump_ells = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    for ell in branch_dump_ells:
        gh = np.asarray(gamma_hat[ell], dtype=np.complex128)
        for j, z in enumerate(gh):
            coeff_rows.append({
                "branch_index": j,
                "ell": ell,
                "real": float(np.real(z)),
                "imag": float(np.imag(z)),
                "abs": float(abs(z)),
                "rel_to_zero_branch": float(abs(z) / max(abs(zero_gamma[j]), 1e-300)),
            })

    out_summary = OUT_ROOT / "python_gamma_hat_summary.csv"
    out_coeffs = OUT_ROOT / "python_gamma_hat_branch_coeffs_selected.csv"
    out_npz = OUT_ROOT / "python_gamma_hat_coeffs.npz"
    out_meta = OUT_ROOT / "python_gamma_hat_meta.json"

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with out_coeffs.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coeff_rows[0].keys()))
        w.writeheader()
        w.writerows(coeff_rows)

    np.savez_compressed(
        out_npz,
        **{f"ell_{ell:+d}": np.asarray(gamma_hat[ell], dtype=np.complex128) for ell in selected_ells},
        zero_gamma=zero_gamma,
    )

    meta = {
        "case_dir": str(CASE_DIR),
        "pump_dir": str(PUMP_DIR),
        "max_ell": MAX_ELL,
        "gamma_nt": GAMMA_NT,
        "nbranches": int(zero_gamma.size),
        "pump_omega_p": float(pump.omega_p),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("PYTHON_GAMMA_DUMP_OK")
    print("pump_dir =", PUMP_DIR)
    print("nbranches =", zero_gamma.size)
    print("wrote_summary =", out_summary)
    print("wrote_coeffs =", out_coeffs)
    print("wrote_npz =", out_npz)
    print()
    print("Most relevant summary rows:")
    for ell in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        r = summary_rows[ell + MAX_ELL]
        print(
            f"ell={ell:+3d} "
            f"rel_l2={r['l2_abs_over_zero_l2']:.9e} "
            f"max_abs={r['max_abs']:.9e} "
            f"mean_abs={r['mean_abs']:.9e} "
            f"conj_err={r['conj_symmetry_rel_err']:.3e}"
        )

if __name__ == "__main__":
    main()
