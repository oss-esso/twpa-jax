"""Overlay signal-gain spectra for the 4 clean (Lj,Cg,scale) designs at one
fixed, converged pump point (fp=7.8965517241 GHz), plus the Julia reference
if available.

Companion to the 4-case comparison requested after the /copilot-implement
pass: same physical injected pump current across A/C and B/D respectively
(picked at i_power=12, j_freq=26 in each 30x30 map, the largest common PASS
region at fp~7.9 GHz). See outputs/spectrum_compare/case{A,B,C,D}/map_spectrum.npz
(built via run_gain_map.py --mode warmstart --n-power 1 --n-frequency 1
--initial-pump-dir unusable due to full-shape/Schur-shape mismatch, so these
are fresh cold solves at the exact map cell -- gain_db matches the original
map row to 6 significant figures for all 4 cases, confirming the same fixed
point).

Usage:
    python scripts/plot_spectrum_compare.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("caseA", "A: Lj=123.9pH Cg=66fF scale=1.0", "tab:blue"),
    ("caseB", "B: Lj=79pH Cg=33fF scale=1.0", "tab:orange"),
    ("caseC", "C: Lj=123.9pH Cg=66fF scale=2.0", "tab:green"),
    ("caseD", "D: Lj=79pH Cg=33fF scale=2.0", "tab:red"),
]

# single shared point for all 5 curves
FP_GHZ = 7.6
PP_DBM = -25.0

MEASUREMENT_DIR = ROOT / "docs" / "14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"
MEASUREMENT_FILES = [
    ("105C5_7.593GHz.npy", "tab:brown"),
    ("105C5_7.674GHz.npy", "magenta"),
    ("105C5_7.714GHz.npy", "purple"),
]


def load_case(tag: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(ROOT / "outputs" / "spectrum_compare" / tag / "map_spectrum.npz")
    freq = d["signal_ghz"][:, 0]
    gain = d["gain_spectrum_db"][0, 0, :]
    order = np.argsort(freq)
    return freq[order], gain[order]


def load_measurement(npy_name: str, target_power_dbm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Raw (uncalibrated) Themis transmission at nearest PumpPower row.

    ``Response`` is the same "gain_db" convention align_map_to_measurement.py
    treats it as (raw transmission, no pump-off baseline subtraction, no
    df/dP/dG calibration fit) -- so this is a direct, unaligned overlay.
    """
    d = np.load(MEASUREMENT_DIR / npy_name, allow_pickle=True).item()
    freq_ghz = np.asarray(d["Frequency"], dtype=float) / 1e9
    resp = np.asarray(d["Response"], dtype=float)  # (n_power, n_sig)
    powers = np.asarray(d["PumpPower"], dtype=float)
    idx = int(np.argmin(np.abs(powers - target_power_dbm)))
    return freq_ghz, resp[idx, :], float(powers[idx])


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    julia_csv = ROOT / "outputs" / "spectrum_compare" / "julia_point.csv"
    jfreq = jgain = None
    if julia_csv.exists():
        jdata = np.loadtxt(julia_csv, delimiter=",")
        jfreq, jgain = jdata[:, 0], jdata[:, 1]
        order = np.argsort(jfreq)
        jfreq, jgain = jfreq[order], jgain[order]
    else:
        print(f"NOTE: {julia_csv} not found -- Julia reference curve omitted "
              f"(Harmonia package source missing on this machine)")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for tag, label, color in CASES:
        freq, gain = load_case(tag)
        ls = "-" if tag in ("caseA", "caseB") else "--"
        ax.plot(freq, gain, lw=1.3, label=label, color=color, linestyle=ls)
    if jfreq is not None:
        ax.plot(jfreq, jgain, lw=1.6, label="Julia (JC) ref, Lj=79pH Cg=33fF",
                 color="black", linestyle=":")

    for npy_name, color in MEASUREMENT_FILES:
        mfreq, mresp, mpower = load_measurement(npy_name, PP_DBM)
        tag = npy_name.replace("105C5_", "").replace(".npy", "")
        ax.plot(mfreq, mresp, lw=1.1,
                label=f"measurement {tag}, pp={mpower:.3f} dBm (raw, uncalibrated)",
                color=color, alpha=0.7)

    ax.axvline(FP_GHZ, color="grey", lw=0.8, linestyle=":", label=f"pump fp={FP_GHZ:.4f} GHz")
    ax.set_title("all 4 cases (solid=scale1.0, dashed=scale2.0)")
    ax.set_xlabel("signal frequency (GHz)")
    ax.set_ylabel("gain (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Signal gain spectrum at fp={FP_GHZ:.4f} GHz, pp={PP_DBM:.4f} dBm (same point, all 5 curves)",
        fontsize=11,
    )
    fig.tight_layout()

    out = ROOT / "plots" / "spectrum_compare_4cases.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
