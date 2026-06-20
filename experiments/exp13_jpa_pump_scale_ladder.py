from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
EXP09 = ROOT / "experiments" / "exp09_full_ipm_gain_from_pump.py"
IPM_DIR = ROOT / "outputs" / "jc_doc_python_designs" / "jc_jpa"
OUT_ROOT = ROOT / "outputs" / "exp13_jpa_pump_scale_ladder"

spec = importlib.util.spec_from_file_location("exp09", EXP09)
exp09 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = exp09
spec.loader.exec_module(exp09)

BASE_RATIO = 0.017167722161028475
JC_MAX = 13.300727259957004
JC_PEAK = 4.75

with np.load(IPM_DIR / "ipm_arrays.npz", allow_pickle=True) as _z:
    PHI0_REDUCED = float(_z["phi0_reduced"][0])

def run_pump(scale: float) -> Path:
    outdir = OUT_ROOT / f"pump_scale_{scale:.6f}".replace(".", "p")
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EXP08),
        "--ipm-dir", str(IPM_DIR),
        "--pump-port", "1",
        "--pump-freq-ghz", "4.75001",
        "--pump-current-ratio-ic", repr(BASE_RATIO * scale),
        "--harmonics", "16",
        "--nt", "256",
        "--continuation-steps", "10",
        "--continuation-predictor", "secant",
        "--newton-tol", "1e-8",
        "--gmres-rtol", "1e-6",
        "--jvp-mode", "aft",
        "--quiet",
        "--skip-time-residual",
        "--outdir", str(outdir),
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()
    return outdir

def assemble(ipm, khat, omega_s, omega_p, ms):
    zero = sp.csr_matrix(ipm.C.shape, dtype=np.complex128)
    rows = []
    for m in ms:
        row = []
        Dm = exp09.dynamic_block(ipm, omega_s + m * omega_p)
        for q in ms:
            block = khat.get(m - q, zero)
            if m == q:
                block = block + Dm
            row.append(block.tocsr())
        rows.append(row)
    return sp.bmat(rows, format="csc")

def s_from_v(v, source_current=1.0, z0=50.0):
    return 2.0 * v / (z0 * source_current) - 1.0

def db_from_s(s):
    return 10.0 * math.log10(max(abs(s) ** 2, 1e-300))

def gain_for_pump(pump_dir: Path, points: int = 501):
    ipm = exp09.load_ipm(IPM_DIR)
    pump = exp09.load_pump(pump_dir, None)

    sidebands = 8
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
    n = ipm.C.shape[0]

    freqs = np.linspace(4.5, 5.0, points)
    gains = []

    for fghz in freqs:
        omega_s = 2.0 * math.pi * float(fghz) * 1e9
        A = assemble(ipm, khat, omega_s, pump.omega_p, ms)

        b = np.zeros(len(ms) * n, dtype=np.complex128)
        b[ms.index(0) * n + source_index] = 1.0

        y = spla.spsolve(A, b)
        phi = complex(y[ms.index(0) * n + out_index])
        v = exp09.voltage_from_flux(omega_s, phi)
        gains.append(db_from_s(s_from_v(v)))

    gains = np.array(gains)
    best_i = int(np.argmax(gains))
    return {
        "max": float(np.max(gains)),
        "mean": float(np.mean(gains)),
        "min": float(np.min(gains)),
        "peak": float(freqs[best_i]),
    }

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    scales = [1.0, 1.2, math.sqrt(2.0), 1.6, 1.8, 2.0, 2.4, 2.8]

    print("scale ratio pump_status psi_over_phi0 gain_max err_max peak err_peak")
    for scale in scales:
        pump_dir = run_pump(scale)
        report = json.loads((pump_dir / "pump_report.json").read_text())

        status = str(
            report.get(
                "status",
                report.get(
                    "final_status",
                    report.get("step_status", "UNKNOWN"),
                ),
            )
        )

        psi = float(report.get("branch_psi_max_abs", report.get("x_max_abs", float("nan"))))
        psi_over_phi0 = psi / PHI0_REDUCED

        g = gain_for_pump(pump_dir)

        print(
            f"{scale:7.4f} "
            f"{BASE_RATIO * scale:13.9e} "
            f"{status:16s} "
            f"{psi_over_phi0:13.6f} "
            f"{g['max']:12.6f} "
            f"{g['max'] - JC_MAX:12.6f} "
            f"{g['peak']:8.5f} "
            f"{g['peak'] - JC_PEAK:9.5f}"
        )

if __name__ == "__main__":
    main()
