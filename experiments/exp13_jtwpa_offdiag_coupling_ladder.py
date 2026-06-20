from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"

spec = importlib.util.spec_from_file_location("exp09", EXP09)
exp09 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = exp09
spec.loader.exec_module(exp09)

CASE_DIR = ROOT / "outputs" / "jc_doc_python_designs" / "jc_jtwpa"
PUMP_DIR = ROOT / "outputs" / "exp13_jtwpa_fast_scale_ladder" / "scale_2p000" / "pump"
JC_CSV = ROOT / "outputs" / "exp13_jtwpa_fast_scale2" / "jc_jtwpa_curve_21pt.csv"
OUT_ROOT = ROOT / "outputs" / "exp13_jtwpa_offdiag_coupling_ladder"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SIDE_BANDS = 3
GAMMA_NT = 80
SOURCE_PORT = 1
OUT_PORT = 2

BETAS = [
    -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
     0.0,
     0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
]

ELL_RULES = ["m_minus_q", "q_minus_m"]

def load_curve(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fk = "signal_ghz" if "signal_ghz" in rows[0] else "frequency_ghz"
    x = np.array([float(r[fk]) for r in rows], dtype=float)
    y = np.array([float(r["gain_db"]) for r in rows], dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]

def assemble_variant(ipm, khat, omega_s, omega_p, ms, *, beta_offdiag: float, ell_rule: str):
    zero = sp.csr_matrix(ipm.C.shape, dtype=np.complex128)
    rows = []

    for m in ms:
        row = []
        Dm = exp09.dynamic_block(ipm, omega_s + m * omega_p)

        for q in ms:
            if ell_rule == "m_minus_q":
                ell = m - q
            elif ell_rule == "q_minus_m":
                ell = q - m
            else:
                raise ValueError(ell_rule)

            block = khat.get(ell, zero)

            # This is the key diagnostic:
            # keep gamma_0 exactly as-is, scale/sign only harmonic coupling blocks.
            if ell != 0:
                block = beta_offdiag * block

            if m == q:
                block = block + Dm

            row.append(block.tocsr())
        rows.append(row)

    return sp.bmat(rows, format="csc")

def s_from_v(v, source_current=1.0, z0=50.0):
    return 2.0 * v / (z0 * source_current)

def db_from_s(s):
    return 10.0 * math.log10(max(abs(s) ** 2, 1e-300))

def run_variant(beta_offdiag: float, ell_rule: str, jc_f, jc_g):
    ipm = exp09.load_ipm(CASE_DIR)
    pump = exp09.load_pump(PUMP_DIR, None)

    ms = exp09.sideband_list(SIDE_BANDS)
    max_ell = max(abs(m - q) for m in ms for q in ms)

    gamma_hat = exp09.compute_gamma_hat(
        ipm=ipm,
        pump=pump,
        max_ell=max_ell,
        gamma_nt=GAMMA_NT,
        dc_branch_flux=None,
    )
    khat = exp09.build_khat(ipm.Bphi, gamma_hat, drop_tol=0.0)

    n = ipm.C.shape[0]
    source_index = ipm.port_to_index[SOURCE_PORT]
    out_index = ipm.port_to_index[OUT_PORT]

    gains = []

    for fghz in jc_f:
        omega_s = 2.0 * math.pi * float(fghz) * 1e9

        A = assemble_variant(
            ipm,
            khat,
            omega_s,
            pump.omega_p,
            ms,
            beta_offdiag=beta_offdiag,
            ell_rule=ell_rule,
        )

        b = np.zeros(len(ms) * n, dtype=np.complex128)
        b[ms.index(0) * n + source_index] = 1.0

        y = spla.spsolve(A, b)
        phi = complex(y[ms.index(0) * n + out_index])
        v = exp09.voltage_from_flux(omega_s, phi)
        s = s_from_v(v)
        gains.append(db_from_s(s))

    gains = np.array(gains, dtype=float)
    err = gains - jc_g

    return {
        "ell_rule": ell_rule,
        "beta_offdiag": beta_offdiag,
        "py_max": float(np.max(gains)),
        "py_mean": float(np.mean(gains)),
        "py_min": float(np.min(gains)),
        "py_peak": float(jc_f[int(np.argmax(gains))]),
        "max_abs_err": float(np.max(np.abs(err))),
        "mean_abs_err": float(np.mean(np.abs(err))),
        "rms_err": float(np.sqrt(np.mean(err ** 2))),
    }, gains

def main():
    jc_f, jc_g = load_curve(JC_CSV)

    print("JC full max =", float(np.max(jc_g)), "peak =", float(jc_f[int(np.argmax(jc_g))]))
    print("ell_rule beta py_max py_peak max_abs_err mean_abs_err rms_err")

    rows = []
    best = None
    best_gains = None

    for ell_rule in ELL_RULES:
        for beta in BETAS:
            stats, gains = run_variant(beta, ell_rule, jc_f, jc_g)
            rows.append(stats)

            print(
                f"{ell_rule:10s} "
                f"{beta:8.3f} "
                f"{stats['py_max']:10.6f} "
                f"{stats['py_peak']:8.3f} "
                f"{stats['max_abs_err']:12.6f} "
                f"{stats['mean_abs_err']:12.6f} "
                f"{stats['rms_err']:12.6f}"
            )

            if best is None or stats["rms_err"] < best["rms_err"]:
                best = stats
                best_gains = gains.copy()

    out_json = OUT_ROOT / "offdiag_coupling_ladder_summary.json"
    out_csv = OUT_ROOT / "offdiag_coupling_ladder_summary.csv"
    best_csv = OUT_ROOT / "best_offdiag_curve.csv"

    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with best_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal_ghz", "jc_gain_db", "python_gain_db", "error_db"])
        for fghz, gjc, gpy in zip(jc_f, jc_g, best_gains):
            w.writerow([fghz, gjc, gpy, gpy - gjc])

    print()
    print("BEST_BY_RMS")
    print(json.dumps(best, indent=2))
    print("wrote_json =", out_json)
    print("wrote_csv =", out_csv)
    print("wrote_best_curve =", best_csv)

if __name__ == "__main__":
    main()
