from __future__ import annotations

import importlib.util
import itertools
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

IPM_DIR = ROOT / "outputs" / "jc_doc_python_designs" / "jc_jpa"
PUMP_DIR = ROOT / "outputs" / "exp13_jcstyle_jpa_pump_h16"

JC = {
    "gain_db_max": 13.300727259957004,
    "gain_db_mean": 0.9745846837647946,
    "gain_db_min": 0.0027159157639631156,
    "peak_frequency_ghz": 4.75,
}

def assemble_variant(ipm, khat, omega_s, omega_p, ms, *, ell_rule: str, khat_sign: float, mode_scale: float):
    zero = sp.csr_matrix(ipm.C.shape, dtype=np.complex128)
    rows = []

    for m in ms:
        row = []
        omega_m = omega_s + mode_scale * m * omega_p
        Dm = exp09.dynamic_block(ipm, omega_m)

        for q in ms:
            if ell_rule == "m_minus_q":
                ell = m - q
            elif ell_rule == "q_minus_m":
                ell = q - m
            elif ell_rule == "neg_m_minus_q":
                ell = -(m - q)
            else:
                raise ValueError(ell_rule)

            block = khat_sign * khat.get(ell, zero)
            if m == q:
                block = block + Dm
            row.append(block.tocsr())
        rows.append(row)

    return sp.bmat(rows, format="csc")

def sideband_node(y, n, ms, m, node_index):
    return complex(y[ms.index(m) * n + node_index])

def s_from_v(v, source_current, z0=50.0):
    return 2.0 * v / (z0 * source_current) - 1.0

def db_from_s(s):
    return 10.0 * math.log10(max(abs(s) ** 2, 1e-300))

def run_variant(*, ell_rule, khat_sign, source_sign, mode_scale, sidebands):
    ipm = exp09.load_ipm(IPM_DIR)
    pump = exp09.load_pump(PUMP_DIR, None)

    ms = exp09.sideband_list(sidebands)
    max_ell = max(abs(m - q) for m in ms for q in ms)

    gamma_hat = exp09.compute_gamma_hat(
        ipm=ipm,
        pump=pump,
        max_ell=max_ell,
        gamma_nt=256,
        dc_branch_flux=None,
    )
    khat = exp09.build_khat(ipm.Bphi, gamma_hat, drop_tol=0.0)

    source_index = ipm.port_to_index[1]
    out_index = ipm.port_to_index[1]
    signal_m = 0
    n = ipm.C.shape[0]
    source_current = source_sign * 1.0

    freqs = np.linspace(4.5, 5.0, 501)
    gains = []

    for fghz in freqs:
        omega_s = 2.0 * math.pi * float(fghz) * 1e9
        A = assemble_variant(
            ipm, khat, omega_s, pump.omega_p, ms,
            ell_rule=ell_rule,
            khat_sign=khat_sign,
            mode_scale=mode_scale,
        )

        b = np.zeros(len(ms) * n, dtype=np.complex128)
        b[ms.index(signal_m) * n + source_index] = source_current

        y = spla.spsolve(A, b)

        phi = sideband_node(y, n, ms, signal_m, out_index)
        v = exp09.voltage_from_flux(omega_s, phi)
        s = s_from_v(v, source_current, z0=50.0)
        gains.append(db_from_s(s))

    gains = np.array(gains, dtype=float)
    best_i = int(np.argmax(gains))

    return {
        "ell_rule": ell_rule,
        "khat_sign": khat_sign,
        "source_sign": source_sign,
        "mode_scale": mode_scale,
        "sidebands": sidebands,
        "max": float(np.max(gains)),
        "mean": float(np.mean(gains)),
        "min": float(np.min(gains)),
        "peak": float(freqs[best_i]),
        "err_max": float(np.max(gains) - JC["gain_db_max"]),
        "err_mean": float(np.mean(gains) - JC["gain_db_mean"]),
        "err_min": float(np.min(gains) - JC["gain_db_min"]),
        "err_peak": float(freqs[best_i] - JC["peak_frequency_ghz"]),
    }

def score(r):
    return (
        abs(r["err_max"])
        + 0.2 * abs(r["err_mean"])
        + 0.1 * abs(r["err_min"])
        + 20.0 * abs(r["err_peak"])
    )

def main():
    variants = []

    for ell_rule, khat_sign, source_sign, mode_scale in itertools.product(
        ["m_minus_q", "q_minus_m", "neg_m_minus_q"],
        [1.0, -1.0],
        [1.0, -1.0],
        [1.0, 2.0],
    ):
        try:
            r = run_variant(
                ell_rule=ell_rule,
                khat_sign=khat_sign,
                source_sign=source_sign,
                mode_scale=mode_scale,
                sidebands=8,
            )
            variants.append(r)
            print(
                f"{r['ell_rule']:13s} khat={r['khat_sign']:4.0f} src={r['source_sign']:4.0f} scale={r['mode_scale']:.0f} "
                f"max={r['max']:10.6f} mean={r['mean']:10.6f} min={r['min']:10.6f} peak={r['peak']:.6f} "
                f"err_max={r['err_max']:10.6f} err_mean={r['err_mean']:10.6f} err_peak={r['err_peak']: .6f}"
            )
        except Exception as e:
            print(f"FAILED {ell_rule} khat={khat_sign} src={source_sign} scale={mode_scale}: {type(e).__name__}: {e}")

    print("\n=== best by rough score ===")
    for r in sorted(variants, key=score)[:8]:
        print(
            f"{r['ell_rule']:13s} khat={r['khat_sign']:4.0f} src={r['source_sign']:4.0f} scale={r['mode_scale']:.0f} "
            f"max={r['max']:10.6f} mean={r['mean']:10.6f} min={r['min']:10.6f} peak={r['peak']:.6f} score={score(r):.6f}"
        )

if __name__ == "__main__":
    main()
